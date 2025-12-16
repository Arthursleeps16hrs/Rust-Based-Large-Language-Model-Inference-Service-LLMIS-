mod config;
mod metrics;
mod model;
mod routes;

use crate::config::AppConfig;
use crate::metrics::Metrics;
use crate::model::ModelManager;
use crate::config::ModelConfig;
use crate::routes::AppState;
use clap::Parser;
use std::sync::Arc;
use tokio::signal;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to configuration file (TOML). Defaults to config/config.example.toml when present.
    #[arg(short, long)]
    config: Option<String>,
    /// Quick load a local GGUF model (requires --features llm-backend)
    #[arg(long)]
    gguf_path: Option<String>,
    /// Name to register the GGUF model under (defaults to "local-llm")
    #[arg(long, default_value = "local-llm")]
    gguf_name: String,
    /// Model architecture for GGUF (llama|mistral)
    #[arg(long, default_value = "llama")]
    gguf_arch: String,
    /// Context length override for GGUF
    #[arg(long)]
    gguf_context: Option<usize>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();

    let config_path = cli.config.or_else(|| {
        let default_path = "config/config.example.toml";
        if std::path::Path::new(default_path).exists() {
            Some(default_path.to_string())
        } else {
            None
        }
    });

    let mut cfg = AppConfig::load(config_path.as_deref())?;
    if let Some(path) = cli.gguf_path {
        cfg.models.push(ModelConfig {
            name: cli.gguf_name.clone(),
            path: Some(path),
            device: Some("cpu".to_string()),
            quantization: None,
            max_concurrent: Some(cfg.limits.max_concurrent),
            backend: Some("llama-server".to_string()),
            arch: Some(cli.gguf_arch.clone()),
            context_length: cli.gguf_context,
            server_url: None,
        });
        info!(
            target: "llmis",
            "queued GGUF model '{}' (arch: {}) for load via CLI flag (expects llama.cpp server)",
            cli.gguf_name,
            cli.gguf_arch
        );
    }
    let metrics = Arc::new(Metrics::default());
    let manager = Arc::new(ModelManager::new(cfg.limits.clone(), metrics.clone()));

    for model_cfg in cfg.models.clone() {
        match manager.load_model(model_cfg).await {
            Ok(summary) => info!(
                target: "llmis",
                "loaded model '{}' on {}",
                summary.name, summary.device
            ),
            Err(err) => warn!(target: "llmis", "failed to load model: {err:?}"),
        }
    }

    let state = AppState {
        config: cfg.clone(),
        models: manager,
        metrics: metrics.clone(),
        safety: cfg.safety.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let router = routes::routes(state).layer(TraceLayer::new_for_http());
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!(target: "llmis", "listening on http://{}", addr);

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|err| {
            error!(target: "llmis", "server error: {err}");
            err
        })?;
    Ok(())
}

fn init_tracing() {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,hyper=warn"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
