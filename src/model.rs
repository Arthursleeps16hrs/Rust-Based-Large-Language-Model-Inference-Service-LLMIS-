use crate::config::{LimitConfig, ModelConfig};
use crate::metrics::Metrics;
use async_trait::async_trait;
use dashmap::DashMap;
use futures::stream::{BoxStream, Stream};
use serde::Serialize;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use futures::StreamExt;

#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub name: String,
    pub device: String,
    pub backend: String,
    pub quantization: Option<String>,
    pub max_concurrent: usize,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub device: String,
    pub backend: String,
    pub quantization: Option<String>,
    pub max_concurrent: usize,
}

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub token: String,
    pub finished: bool,
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("model not found: {0}")]
    NotFound(String),
    #[error("model overloaded")]
    Overloaded,
    #[error("backend error: {0}")]
    Backend(String),
}

pub type ModelStream = GuardedStream<BoxStream<'static, TokenEvent>>;

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn load(&self, cfg: &ModelConfig) -> Result<(), ModelError>;
    async fn unload(&self) -> Result<(), ModelError>;
    async fn generate_stream(
        &self,
        params: GenerateParams,
    ) -> Result<BoxStream<'static, TokenEvent>, ModelError>;
}

pub struct ModelHandle {
    pub info: ModelInfo,
    backend: Arc<dyn ModelBackend>,
    semaphore: Arc<Semaphore>,
}

impl ModelHandle {
    pub async fn stream(
        &self,
        params: GenerateParams,
    ) -> Result<ModelStream, ModelError> {
        let permit = self
            .semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| ModelError::Overloaded)?;

        let stream = self.backend.generate_stream(params).await?;
        Ok(GuardedStream::new(stream, permit))
    }
}

pub struct ModelManager {
    models: DashMap<String, Arc<ModelHandle>>,
    limits: LimitConfig,
    metrics: Arc<Metrics>,
}

impl ModelManager {
    pub fn new(limits: LimitConfig, metrics: Arc<Metrics>) -> Self {
        Self {
            models: DashMap::new(),
            limits,
            metrics,
        }
    }

    pub async fn load_model(&self, cfg: ModelConfig) -> Result<ModelSummary, ModelError> {
        let backend_choice = cfg
            .backend
            .clone()
            .unwrap_or_else(|| "llm".to_string());

        let backend: Arc<dyn ModelBackend> = match backend_choice.as_str() {
            "llm" | "llama-server" => Arc::new(
                LlamaServerBackend::new(cfg.clone())
                    .map_err(|e| ModelError::Backend(e.to_string()))?,
            ),
            other => {
                return Err(ModelError::Backend(format!(
                    "unsupported backend '{}', use 'llama-server'",
                    other
                )))
            }
        };
        backend.load(&cfg).await?;

        let max_concurrent = cfg.max_concurrent.unwrap_or(self.limits.max_concurrent);

        let info = ModelInfo {
            name: cfg.name.clone(),
            device: cfg.device.unwrap_or_else(|| "cpu".to_string()),
            backend: backend_choice,
            quantization: cfg.quantization.clone(),
            max_concurrent,
        };

        let handle = Arc::new(ModelHandle {
            info: info.clone(),
            backend,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        });

        self.models.insert(cfg.name.clone(), handle);
        self.metrics
            .set_models_loaded(self.models.len() as u64);

        Ok(ModelSummary {
            name: info.name,
            device: info.device,
            backend: info.backend,
            quantization: info.quantization,
            max_concurrent: info.max_concurrent,
        })
    }

    pub async fn unload_model(&self, name: &str) -> Result<(), ModelError> {
        if let Some((_, handle)) = self.models.remove(name) {
            handle.backend.unload().await?;
            self.metrics
                .set_models_loaded(self.models.len() as u64);
            Ok(())
        } else {
            Err(ModelError::NotFound(name.to_string()))
        }
    }

    pub fn list_models(&self) -> Vec<ModelSummary> {
        self.models
            .iter()
            .map(|entry| ModelSummary {
                name: entry.info.name.clone(),
                device: entry.info.device.clone(),
                backend: entry.info.backend.clone(),
                quantization: entry.info.quantization.clone(),
                max_concurrent: entry.info.max_concurrent,
            })
            .collect()
    }

    pub async fn stream(
        &self,
        model: &str,
        params: GenerateParams,
    ) -> Result<ModelStream, ModelError> {
        let handle = self
            .models
            .get(model)
            .ok_or_else(|| ModelError::NotFound(model.to_string()))?;
        handle.stream(params).await
    }
}

pub struct GuardedStream<S> {
    inner: S,
    _permit: OwnedSemaphorePermit,
}

impl<S> GuardedStream<S> {
    pub fn new(inner: S, permit: OwnedSemaphorePermit) -> Self {
        Self { inner, _permit: permit }
    }
}

impl<S: Stream + Unpin> Stream for GuardedStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.inner) };
        inner.poll_next(cx)
    }
}

#[derive(Clone)]
pub struct LlamaServerBackend {
    model_name: String,
    server_url: String,
    client: reqwest::Client,
    max_context: usize,
}

impl LlamaServerBackend {
    pub fn new(cfg: ModelConfig) -> anyhow::Result<Self> {
        let server_url = cfg
            .server_url
            .unwrap_or_else(|| "http://127.0.0.1:8081".to_string());
        Ok(Self {
            model_name: cfg.name,
            server_url,
            client: reqwest::Client::new(),
            max_context: cfg.context_length.unwrap_or(2048),
        })
    }
}

#[async_trait]
impl ModelBackend for LlamaServerBackend {
    async fn load(&self, _cfg: &ModelConfig) -> Result<(), ModelError> {
        // Best-effort check: ensure server is reachable.
        let url = format!("{}/health", self.server_url);
        let _ = self.client.get(url).send().await;
        Ok(())
    }

    async fn unload(&self) -> Result<(), ModelError> {
        Ok(())
    }

    async fn generate_stream(
        &self,
        params: GenerateParams,
    ) -> Result<BoxStream<'static, TokenEvent>, ModelError> {
        #[derive(serde::Serialize)]
        struct ChatMessage {
            role: String,
            content: String,
        }

        #[derive(serde::Serialize)]
        struct RequestBody {
            model: String,
            messages: Vec<ChatMessage>,
            temperature: f32,
            top_p: f32,
            max_tokens: usize,
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            stop: Option<Vec<String>>,
        }

        let GenerateParams {
            prompt,
            max_tokens,
            temperature,
            top_p,
            stop,
            ..
        } = params;

        let n_predict = max_tokens.min(self.max_context);
        let body = RequestBody {
            model: self.model_name.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            temperature,
            top_p,
            max_tokens: n_predict,
            stream: true,
            stop,
        };

        let url = format!("{}/v1/chat/completions", self.server_url);
        let client = self.client.clone();
        let (tx, rx) = mpsc::channel::<TokenEvent>(32);

        tokio::spawn(async move {
            let resp = match client.post(url).json(&body).send().await {
                Ok(r) => r,
                Err(err) => {
                    let _ = tx
                        .send(TokenEvent {
                            token: format!("error: request failed: {err}"),
                            finished: true,
                        })
                        .await;
                    return;
                }
            };

            let mut stream = resp.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = stream.next().await {
                let Ok(bytes) = chunk else { break };
                buf.push_str(&String::from_utf8_lossy(&bytes));

                loop {
                    if let Some(idx) = buf.find("\n\n") {
                        let mut part = buf[..idx].trim().to_string();
                        buf.drain(..idx + 2);

                        if part.is_empty() {
                            continue;
                        }
                        if let Some(stripped) = part.strip_prefix("data:") {
                            part = stripped.trim().to_string();
                        }

                        if part == "[DONE]" {
                            let _ = tx
                                .send(TokenEvent {
                                    token: String::new(),
                                    finished: true,
                                })
                                .await;
                            return;
                        }

                        if let Ok(v) = serde_json::from_str::<Value>(&part) {
                            let token_text = v
                                .get("token")
                                .and_then(|t| t.get("text"))
                                .and_then(|t| t.as_str())
                                .or_else(|| {
                                    v.get("content")
                                        .or_else(|| v.get("text"))
                                        .and_then(|t| t.as_str())
                                })
                                .or_else(|| {
                                    v.get("choices")
                                        .and_then(|c| c.get(0))
                                        .and_then(|c0| {
                                            c0.get("delta")
                                                .and_then(|d| d.get("content"))
                                                .and_then(|d| d.as_str())
                                                .or_else(|| {
                                                    c0.get("text").and_then(|d| d.as_str())
                                                })
                                        })
                                })
                                .unwrap_or_default()
                                .to_string();

                            let finish_reason = v
                                .get("choices")
                                .and_then(|c| c.get(0))
                                .and_then(|c0| c0.get("finish_reason"))
                                .and_then(|f| f.as_str())
                                .unwrap_or("");

                            let done_flag = v
                                .get("done")
                                .or_else(|| v.get("stop"))
                                .or_else(|| v.get("completed"))
                                .and_then(|d| d.as_bool())
                                .unwrap_or(false)
                                || finish_reason == "stop";

                            if !token_text.is_empty() || done_flag {
                                let _ = tx
                                    .send(TokenEvent {
                                        token: token_text,
                                        finished: done_flag,
                                    })
                                    .await;
                            }
                            if done_flag {
                                return;
                            }
                        } else {
                            // Fallback: emit raw line content if JSON parse fails
                            let _ = tx
                                .send(TokenEvent {
                                    token: part.clone(),
                                    finished: false,
                                })
                                .await;
                        }
                    } else {
                        break;
                    }
                }
            }

            let _ = tx
                .send(TokenEvent {
                    token: String::new(),
                    finished: true,
                })
                .await;
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
