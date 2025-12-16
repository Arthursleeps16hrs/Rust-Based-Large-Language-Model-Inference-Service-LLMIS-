use crate::config::{AppConfig, LimitConfig, ModelConfig, SafetyConfig};
use crate::metrics::{InflightGuard, Metrics};
use crate::model::{GenerateParams, ModelError, ModelManager, ModelSummary};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse};
use axum::{Json, Router};
use axum::{routing::get, routing::post};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub models: Arc<ModelManager>,
    pub metrics: Arc<Metrics>,
    pub safety: SafetyConfig,
    pub version: String,
}

#[derive(Serialize)]
struct VersionResponse {
    version: String,
}

#[derive(Serialize)]
struct ModelListResponse {
    data: Vec<ModelSummary>,
}

#[derive(Deserialize)]
pub struct LoadModelRequest {
    pub name: String,
    pub path: Option<String>,
    pub device: Option<String>,
    pub quantization: Option<String>,
    pub max_concurrent: Option<usize>,
    pub backend: Option<String>,
    pub arch: Option<String>,
    pub context_length: Option<usize>,
    pub server_url: Option<String>,
}

#[derive(Deserialize)]
pub struct UnloadModelRequest {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    model: String,
    choices: Vec<ChatStreamDelta>,
}

#[derive(Serialize)]
struct ChatStreamDelta {
    index: usize,
    delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiErrorResponse {
    error: String,
}

pub fn routes(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/version", get(version))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/admin/models/load", post(load_model))
        .route("/admin/models/unload", post(unload_model))
        .route("/", get(index))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

async fn healthz() -> impl IntoResponse {
    StatusCode::OK
}

async fn version(State(state): State<AppState>) -> impl IntoResponse {
    Json(VersionResponse {
        version: state.version.clone(),
    })
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4"),
    );
    (headers, state.metrics.render_prometheus())
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let data = state.models.list_models();
    Json(ModelListResponse { data })
}

async fn load_model(
    State(state): State<AppState>,
    Json(body): Json<LoadModelRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let cfg = ModelConfig {
        name: body.name,
        path: body.path,
        device: body.device,
        quantization: body.quantization,
        max_concurrent: body.max_concurrent,
        backend: body.backend,
        arch: body.arch,
        context_length: body.context_length,
        server_url: body.server_url,
    };
    let summary = state.models.load_model(cfg).await?;
    Ok((StatusCode::CREATED, Json(summary)))
}

async fn unload_model(
    State(state): State<AppState>,
    Json(body): Json<UnloadModelRequest>,
) -> Result<impl IntoResponse, ApiError> {
    state.models.unload_model(&body.name).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    enforce_safety(&state.safety, &body.messages)?;

    let prompt = build_prompt(&body.messages);
    let params = build_params(&state.config.limits, prompt, &body.max_tokens, &body.temperature, &body.top_p, &body.stop, &body.seed);
    if body.stream {
        stream_chat(state, body.model, params).await
    } else {
        aggregate_chat(state, body.model, params).await
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(body): Json<CompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    enforce_prompt_safety(&state.safety, &body.prompt)?;

    let params = build_params(
        &state.config.limits,
        body.prompt,
        &body.max_tokens,
        &body.temperature,
        &body.top_p,
        &body.stop,
        &body.seed,
    );
    if body.stream {
        stream_chat(state, body.model, params).await
    } else {
        aggregate_chat(state, body.model, params).await
    }
}

async fn stream_chat(
    state: AppState,
    model: String,
    params: GenerateParams,
) -> Result<axum::response::Response, ApiError> {
    state.metrics.inc_request();
    let inflight = state.metrics.guard();

    let id = Uuid::new_v4().to_string();
    let mut stream = state.models.stream(&model, params).await?;
    let metrics = state.metrics.clone();

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);

    tokio::spawn(async move {
        let _guard: InflightGuard = inflight;
        let _ = tx
            .send(event(ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                model: model.clone(),
                choices: vec![ChatStreamDelta {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant".into()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            }))
            .await;

        let mut token_count = 0u64;
        while let Some(token) = stream.next().await {
            token_count += 1;
            let finish_reason = if token.finished {
                Some("stop".to_string())
            } else {
                None
            };
            let content = if token.token.is_empty() { None } else { Some(token.token.clone()) };
            let _ = tx
                .send(event(ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    model: model.clone(),
                    choices: vec![ChatStreamDelta {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content,
                        },
                        finish_reason: finish_reason.clone(),
                    }],
                }))
                .await;

            if token.finished {
                break;
            }
        }
        metrics.add_tokens(token_count);
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = Sse::new(ReceiverStream::new(rx)).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(10))
            .text("keep-alive-text"),
    );
    Ok(stream.into_response())
}

async fn aggregate_chat(
    state: AppState,
    model: String,
    params: GenerateParams,
) -> Result<axum::response::Response, ApiError> {
    state.metrics.inc_request();
    let _guard = state.metrics.guard();

    let id = Uuid::new_v4().to_string();
    let mut stream = state.models.stream(&model, params).await?;
    let mut content = String::new();
    let mut tokens = 0u64;

    while let Some(token) = stream.next().await {
        tokens += 1;
        if token.finished {
            break;
        }
        content.push_str(&token.token);
    }
    state.metrics.add_tokens(tokens);

    let response = ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
    };
    Ok(Json(response).into_response())
}

fn build_params(
    limits: &LimitConfig,
    prompt: String,
    max_tokens: &Option<usize>,
    temperature: &Option<f32>,
    top_p: &Option<f32>,
    stop: &Option<Vec<String>>,
    _seed: &Option<u64>,
) -> GenerateParams {
    let requested_tokens = max_tokens.unwrap_or(limits.max_tokens);
    let capped_tokens = requested_tokens.min(limits.max_tokens);
    GenerateParams {
        prompt,
        max_tokens: capped_tokens,
        temperature: temperature.unwrap_or(0.7),
        top_p: top_p.unwrap_or(0.95),
        stop: stop.clone(),
    }
}

fn build_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn enforce_safety(safety: &SafetyConfig, messages: &[ChatMessage]) -> Result<(), ApiError> {
    let prompt = messages
        .iter()
        .map(|m| m.content.clone())
        .collect::<Vec<_>>()
        .join("\n");
    enforce_prompt_safety(safety, &prompt)
}

fn enforce_prompt_safety(safety: &SafetyConfig, prompt: &str) -> Result<(), ApiError> {
    let lowered = prompt.to_lowercase();
    for term in &safety.denylist {
        if lowered.contains(&term.to_lowercase()) {
            return Err(ApiError::Safety(format!(
                "prompt rejected due to safety denylist: {}",
                term
            )));
        }
    }
    Ok(())
}

fn event(chunk: ChatCompletionChunk) -> Result<Event, Infallible> {
    Ok(Event::default().json_data(chunk).unwrap())
}

fn default_stream() -> bool {
    true
}

async fn index(State(state): State<AppState>) -> axum::response::Response {
    if !state.config.server.enable_ui {
        return StatusCode::NOT_FOUND.into_response();
    }
    Html(include_str!("../static/index.html")).into_response()
}

#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    Overloaded,
    Safety(String),
    Internal(String),
}

impl From<ModelError> for ApiError {
    fn from(err: ModelError) -> Self {
        match err {
            ModelError::NotFound(name) => ApiError::NotFound(name),
            ModelError::Overloaded => ApiError::Overloaded,
            ModelError::Backend(msg) => ApiError::Internal(msg),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Overloaded => (
                StatusCode::TOO_MANY_REQUESTS,
                "model is at capacity, retry later".to_string(),
            ),
            ApiError::Safety(msg) => (StatusCode::FORBIDDEN, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        let payload = Json(ApiErrorResponse { error: message });
        (status, payload).into_response()
    }
}
