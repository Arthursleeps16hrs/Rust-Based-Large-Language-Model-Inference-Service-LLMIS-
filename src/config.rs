use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "ServerConfig::default_ui")]
    pub enable_ui: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            enable_ui: true,
        }
    }
}

impl ServerConfig {
    fn default_ui() -> bool {
        true
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitConfig {
    #[serde(default = "LimitConfig::default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "LimitConfig::default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "LimitConfig::default_queue_depth")]
    pub queue_depth: usize,
}

impl Default for LimitConfig {
    fn default() -> Self {
        Self {
            max_tokens: Self::default_max_tokens(),
            max_concurrent: Self::default_max_concurrent(),
            queue_depth: Self::default_queue_depth(),
        }
    }
}

impl LimitConfig {
    fn default_max_tokens() -> usize {
        512
    }

    fn default_max_concurrent() -> usize {
        2
    }

    fn default_queue_depth() -> usize {
        32
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub path: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub max_concurrent: Option<usize>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub arch: Option<String>,
    #[serde(default)]
    pub context_length: Option<usize>,
    #[serde(default)]
    pub server_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct SafetyConfig {
    #[serde(default)]
    pub denylist: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub limits: LimitConfig,
    #[serde(default)]
    pub safety: SafetyConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            models: Vec::new(),
            limits: LimitConfig::default(),
            safety: SafetyConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let mut builder = config::Config::builder()
            .set_default("server.host", Self::default().server.host.clone())?
            .set_default("server.port", Self::default().server.port as i64)?
            .set_default("server.enable_ui", Self::default().server.enable_ui)?
            .set_default("limits.max_tokens", Self::default().limits.max_tokens as i64)?
            .set_default("limits.max_concurrent", Self::default().limits.max_concurrent as i64)?
            .set_default("limits.queue_depth", Self::default().limits.queue_depth as i64)?;

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        builder = builder.add_source(config::Environment::with_prefix("LLMIS").separator("__"));

        let cfg = builder.build()?;
        let app: AppConfig = cfg.try_deserialize()?;
        Ok(app)
    }
}
