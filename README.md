# Project Proposal: Rust-Based Large Language Model Inference Service (LLMIS)

## Motivation

Modern LLM systems are typically served from Python-first stacks (e.g., vLLM, TGI, OpenAI-like proxies). While these are mature, they come with trade-offs for teams that want **a compact, memory-safe, low-latency, single-binary** deployment story. In the Rust ecosystem, we have promising building blocks (e.g., **Candle**, **Burn**, **mistral.rs**) but there is no widely adopted, end-to-end **inference server** that combines:

* **Multi-model lifecycle management** (load/unload, quantization awareness, memory safety).
* **Streaming token output** with **back-pressure** and **structured cancellation**.
* **OpenAI-compatible APIs** for easy client adoption.
* **Observability and resource accounting** with minimal runtime footprint.
* A **basic built-in chat UI** for quick manual validation and demos.

This project aims to fill that gap with a Rust-native service designed around **tight control of CPU/GPU resources**, **predictable performance**, and **safe concurrency**. It will be satisfying to build, aligns well with Rust’s strengths (ownership + zero-cost abstractions), and provides a genuinely useful artifact for the community: a small-scale but complete, stream-first LLM server.

## Objective and Key Features

### Objective

Build a **Rust back-end inference service** that can **load and serve multiple LLMs** with **streaming responses** and an **OpenAI-style API**, plus a minimal **web chat interface** to interact with it. The design will emphasize **simplicity, robustness, and performance** over maximal feature breadth.

### Key Features

1. **Multi-Model Management**

   * Load, unload, and list models at runtime (`/v1/models`).
   * Named model handles (e.g., `llama-3-8b-instruct-q4`, `mistral-7b-instruct-fp16`).
   * Configurable device placement (CPU, CUDA if available), with guardrails to prevent over-allocation.
   * Optional low-bit quantization support when backend permits (e.g., q4/q5).

2. **Inference API (OpenAI-Compatible Subset)**

   * `/v1/chat/completions` (preferred) and `/v1/completions` with parameters:

     * `model`, `messages` (or `prompt`), `max_tokens`, `temperature`, `top_p`, `stop`, `stream`, `presence_penalty`, `frequency_penalty`, `seed`.
   * **Streaming** via **SSE** by default; optional WebSocket endpoint for clients that prefer WS.
   * Cancellation via request context drop; propagate to decoding loop.

3. **Streaming and Scheduling**

   * Token-by-token streaming with **Tokio** back-pressure and timeouts.
   * A simple **scheduler** that serializes or batches decode steps (depending on backend capability).
   * Per-request **token/event budgets**, **rate limiting**, and **concurrency limits**.

4. **Observability & Safety**

   * Structured logs (JSON), request and token counters, latency percentiles.
   * Prometheus metrics (`/metrics`) with key gauges: loaded models, VRAM/RAM use (if accessible), requests in flight, decode tokens/sec.
   * Basic safety filters (regex/keyword denylist) as an optional middleware to demonstrate server-side content controls.

5. **Basic Chat Interface**

   * Minimal self-contained **Axum** server route serving a small web app (vanilla HTML/JS with EventSource or HTMX) to:

     * Select a model.
     * Enter system/user messages.
     * See streamed output in real time.
   * No auth for MVP; add a simple token header if time permits.

6. **Packaging and Developer Experience**

   * Single binary (`llmis`) with TOML config for models and limits.
   * Dockerfile with slim runtime.
   * Example clients (curl snippets + tiny Rust CLI).

### Why this fills a gap

* There is no widely used Rust server providing **OpenAI-style streaming, model management, and observability** in one small, idiomatic package.
* By **targeting a narrow, production-shaped MVP**, the project becomes achievable within weeks yet remains genuinely useful for Rust users who want to host models without a Python stack.

## Tentative Plan

> I will own architecture, implementation, testing, docs, and packaging. The plan is organized by **workstreams**, not dates, to keep it concise and feasible.

### 1) Foundations (Service Skeleton & Config)

* **Deliverables**

  * `axum` server with health check (`/healthz`) and version (`/version`).
  * Config loader (TOML): default model list, device preferences, server limits.
  * Graceful shutdown and tracing (structured logs via `tracing` + `tracing-subscriber`).
* **Why now**: Enables fast iteration and stable logs/metrics from day one.

### 2) Backend Abstraction & First Model

* **Approach**

  * Define a `ModelBackend` trait:

    ```rust
    trait ModelBackend: Send + Sync {
        fn name(&self) -> &str;
        fn supports_batching(&self) -> bool;
        async fn load(&mut self, path: &str, device: DeviceConfig) -> Result<()>;
        async fn unload(&mut self) -> Result<()>;
        async fn tokenize(&self, text: &str) -> Result<Vec<Token>>;
        async fn generate_stream(&self, params: GenerateParams) -> Result<impl Stream<Item=TokenEvent>>;
    }
    ```
  * Implement **Candle-based** backend first (e.g., GGUF/transformer loader) to keep scope tight.
* **Deliverables**

  * Load a local instruct model (e.g., LLaMA/Mistral family) on CPU; optional CUDA if hardware available.
  * Deterministic decoding (greedy) + simple sampling (top-p, temperature).

### 3) Multi-Model Manager

* **Features**

  * Thread-safe registry (DashMap) of named `ModelHandle`s.
  * Endpoints:

    * `GET /v1/models`: list loaded models.
    * `POST /admin/models/load`: `{name, path, device, quant}`.
    * `POST /admin/models/unload`: `{name}`.
  * Guardrails: max N models loaded, reject if RAM/VRAM estimation exceeds thresholds.
* **Deliverables**

  * Runtime add/remove models; errors logged clearly.

### 4) OpenAI-Compatible Chat API + Streaming

* **Features**

  * `POST /v1/chat/completions`:

    * Accepts `messages: [{role, content}]`, `model`, decode params, and `stream`.
    * **SSE** streaming of `data:` chunks with incremental deltas, ending with `[DONE]`.
  * Cancellation when client disconnects; ensure decoder loop stops promptly.
* **Deliverables**

  * Conformance tests with curl examples.
  * Optional `/v1/completions` for prompt-style requests.

### 5) Scheduler, Limits, and Back-Pressure

* **Features**

  * Simple fair queue: cap concurrent decodes per model; queue additional requests.
  * Timeouts on idle decoding steps; maximum tokens per request.
  * Optional basic batching if backend supports shared KV cache per step; otherwise serialize.
* **Deliverables**

  * Clear metrics: queue length, tokens/sec, active decodes.

### 6) Observability & Safety

* **Features**

  * `/metrics` (Prometheus): requests_total, tokens_generated_total, p50/p95 latencies, memory usage probes.
  * Log correlation IDs per request.
  * Optional content safety middleware: configurable denylist that tags responses or truncates.
* **Deliverables**

  * Dashboard-ready metrics; easy to demo with `prometheus` + `grafana` (docs only).

### 7) Basic Chat UI

* **Scope**

  * Single HTML page served at `/` with:

    * Model dropdown (populated from `/v1/models`).
    * Textarea for prompt; stream display via `EventSource` on `stream=true`.
    * Minimal styling; no framework needed.
* **Deliverables**

  * Manual sanity checks without extra tooling; gif in README if time allows.

### 8) Packaging, Examples, and Docs

* **Deliverables**

  * README with quickstart: download a model, run server, curl examples, UI usage.
  * Dockerfile (debian slim + static build if feasible).
  * Config examples for CPU vs CUDA.

### Stretch Goals (Only if time permits)

* WebSocket streaming option.
* Basic **function calling** schema passthrough.
* **Speculative decoding** or **paged KV cache** if backend allows.
* **JWT auth** for APIs.
* **Batch scheduler** akin to micro-vLLM (experimental).

## Feasibility and Scope Control

* **Backend Choice:** Start with **Candle** for stable Rust inference. Burn or mistral.rs can be added later behind the same `ModelBackend` trait if time allows.
* **Model Size:** Target a **7–8B parameter instruct model** with quantization for CPU feasibility. CUDA path is optional.
* **Streaming First:** Token streaming is core and implemented early; it guides scheduler and back-pressure design.
* **Minimal UI:** Keep the web UI tiny to avoid front-end complexity while still demonstrating end-to-end value.

## Evaluation Plan

* **Functional tests**

  * Load/unload model; list models; generate completion (non-stream + stream).
  * Cancellation test: client disconnect mid-stream → server stops decoding quickly.
  * Concurrency test: N parallel requests under concurrency cap.
* **Performance smoke checks**

  * Tokens/sec on short prompts; latency to first token.
  * Memory footprint while N requests are queued vs active.
* **Reliability**

  * Fuzz prompt sizes; ensure guardrails (max tokens, max context) are respected.
  * Verify error surfaces are structured and actionable.

## Risks & Mitigations

* **Model loading complexity / CUDA differences**

  * Start with CPU path + small quantized models; document CUDA as optional.
* **Batching support may be limited by backend**

  * Implement a serialize-first scheduler; expose batching flags only if solid.
* **Time constraints**

  * Prioritize MVP: single backend, streaming chat API, one model type, basic UI, metrics.

## Deliverables Summary

* `llmis` binary with:

  * Axum server, `/v1/chat/completions` (streaming), `/v1/models`, `/metrics`, `/healthz`.
  * Candle-based backend implementation.
  * Multi-model manager with runtime load/unload.
  * Basic HTML chat UI served at `/`.
  * Configurable limits and logging.
* Documentation:

  * Quickstart, API examples, deployment notes, config reference.
* Example scripts:

  * `curl` snippets and a tiny Rust CLI sample.

## Responsibilities

* **Architecture & Implementation:** All Rust code (server, backend trait, Candle backend, scheduler, streaming, safety filters).
* **DevOps & Packaging:** Dockerfile, config templates, sample systemd unit (if time).
* **Testing & Benchmarks:** Functional tests, simple load tests, and metrics validation.
* **Docs & UI:** README with setup, API description, and minimal chat page.

## Proposed API Sketch (for clarity)

```http
GET  /healthz
GET  /version
GET  /v1/models
POST /admin/models/load        { "name": "mistral-7b-q4", "path": "/models/mistral-7b-q4.gguf", "device": "cpu" }
POST /admin/models/unload      { "name": "mistral-7b-q4" }

POST /v1/chat/completions
{
  "model": "mistral-7b-q4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Rust ownership in 2 sentences."}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": true
}
```

**SSE Stream Example (response):**

```
data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant","content":""}}]}
data: {"choices":[{"delta":{"content":"Rust"}}]}
data: {"choices":[{"delta":{"content":" enforces"}}]}
...
data: [DONE]
```
