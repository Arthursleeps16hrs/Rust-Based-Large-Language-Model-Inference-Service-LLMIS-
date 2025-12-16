use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Default)]
pub struct Metrics {
    requests_total: AtomicU64,
    tokens_total: AtomicU64,
    active_requests: AtomicU64,
    models_loaded: AtomicU64,
}

pub struct InflightGuard {
    metrics: Arc<Metrics>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.metrics.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Metrics {
    pub fn guard(self: &Arc<Self>) -> InflightGuard {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        InflightGuard {
            metrics: Arc::clone(self),
        }
    }

    pub fn inc_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_tokens(&self, tokens: u64) {
        self.tokens_total.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn set_models_loaded(&self, count: u64) {
        self.models_loaded.store(count, Ordering::Relaxed);
    }

    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP llmis_requests_total Total HTTP requests handled\n");
        out.push_str("# TYPE llmis_requests_total counter\n");
        out.push_str(&format!(
            "llmis_requests_total {}\n",
            self.requests_total.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP llmis_tokens_total Tokens emitted by generators\n");
        out.push_str("# TYPE llmis_tokens_total counter\n");
        out.push_str(&format!(
            "llmis_tokens_total {}\n",
            self.tokens_total.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP llmis_active_requests Active requests in flight\n");
        out.push_str("# TYPE llmis_active_requests gauge\n");
        out.push_str(&format!(
            "llmis_active_requests {}\n",
            self.active_requests.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP llmis_models_loaded Models currently registered\n");
        out.push_str("# TYPE llmis_models_loaded gauge\n");
        out.push_str(&format!(
            "llmis_models_loaded {}\n",
            self.models_loaded.load(Ordering::Relaxed)
        ));
        out
    }
}
