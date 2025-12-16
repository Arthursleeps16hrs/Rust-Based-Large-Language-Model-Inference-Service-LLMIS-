LLMIS — Final Report

## Team Members
- Jiaxu Li (sole contributor)
- Student Number: 1006924866
- Preferred Email: ar.li@mail.utoronto.ca

## Motivation
- Rust teams rarely get a single-binary, memory-safe LLM server with OpenAI-style APIs, streaming, observability, and UI in one package. Most solutions are Python-first (vLLM/TGI) or heavy gateways; Rust-native options are fragmented (bindings, partial backends).
- We wanted a compact, low-latency control plane that leverages Rust’s strengths (ownership, async I/O) and slots into existing OpenAI-compatible tooling without extra glue code.
- GGUF + llama.cpp gives CPU-friendly local serving, but the ecosystem lacks a Rust-first API surface, model lifecycle controls, and minimal UI; filling that gap is both useful and satisfying to build.

## Objectives
- Deliver an OpenAI-like chat/completions API with SSE streaming and clean cancellation semantics.
- Provide runtime model lifecycle controls (list, load, unload) with simple guardrails and metrics.
- Ship a minimal, dependency-free chat UI for manual validation and demos.
- Keep deployment lean: single Rust binary plus a llama.cpp backend for GGUF models.
- Ensure reproducibility on macOS/Ubuntu with clear commands and config examples.

## Features
- API surface: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/admin/models/{load,unload}`, `/metrics`, `/healthz`, `/version`.
- Streaming: SSE token streaming with graceful end-of-stream handling; non-streamed responses supported.
- Model lifecycle: register models pointing to a llama.cpp server; per-model concurrency limit; list/unload endpoints.
- Observability: Prometheus-style counters (`llmis_requests_total`, `llmis_tokens_total`, `llmis_active_requests`, `llmis_models_loaded`).
- Safety: denylist filter on prompts/messages to block disallowed content.
- UI: standalone HTML/JS at `/` to pick a model, enter system/user text, stream output live, and cancel in-flight requests.
- Config/CLI: TOML config with env overrides and CLI flags to register a model at startup.

## User’s / Developer’s Guide
- Start llama.cpp with the demo model (macOS Homebrew path shown; adjust for Ubuntu build):
  ```bash
  /opt/homebrew/bin/llama-server \
    -m /absolute/path/to/llama-2-7b-chat.Q4_K_M.gguf \
    --port 8081 --ctx-size 2048
  ```
  (GGUF source used: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- Run the Rust server and auto-register the model:
  ```bash
  cargo run -- \
    --gguf-path /absolute/path/to/llama-2-7b-chat.Q4_K_M.gguf \
    --gguf-name local-llm \
    --gguf-arch llama
  ```
- Use the API:
  - List models: `curl localhost:8080/v1/models`
  - Streamed chat:
    ```bash
    curl -N -X POST localhost:8080/v1/chat/completions \
      -H "content-type: application/json" \
      -d '{
        "model": "local-llm",
        "stream": true,
        "messages": [
          {"role":"system","content":"You are a helpful assistant."},
          {"role":"user","content":"Give me two bullet points about Rust networking."}
        ],
        "max_tokens": 400,
        "temperature": 0.7
      }'
    ```
  - Non-stream: set `"stream": false` and read the JSON body.
- Use the UI: open `http://localhost:8080/`, pick `local-llm`, enter system/user text, and watch tokens stream; Stop cancels the request.

## Reproducibility Guide (Ubuntu/macOS)
1) Install Rust: `curl https://sh.rustup.rs -sSf | sh` (restart shell).
2) Install build tools (Ubuntu: `sudo apt-get update && sudo apt-get install -y build-essential cmake clang`).
3) Get llama.cpp server:
   - macOS: `brew install llama.cpp` (binary at `/opt/homebrew/bin/llama-server`).
   - Ubuntu:  
     ```bash
     git clone https://github.com/ggerganov/llama.cpp
     cd llama.cpp
     cmake -B build -DGGML_BLAS=OFF
     cmake --build build -j
     ```
4) Download the model (example): `wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf`
5) Start llama.cpp server (adjust path/port as needed):
   ```bash
   ./build/bin/server -m /absolute/path/to/llama-2-7b-chat.Q4_K_M.gguf --port 8081 --ctx-size 2048
   ```
6) In this repo, run the Rust server with CLI flags (or config):
   ```bash
   cargo run -- \
     --gguf-path /absolute/path/to/llama-2-7b-chat.Q4_K_M.gguf \
     --gguf-name local-llm \
     --gguf-arch llama
   ```
7) Validate: `curl localhost:8080/healthz`, `curl localhost:8080/v1/models`, then stream a chat request or open the UI at `http://localhost:8080/`.

## Contributions
- Sole contributor: architecture, Rust server, model manager, llama.cpp backend adapter, SSE streaming, metrics, safety checks, UI, docs, and testing.

## Lessons Learned & Conclusion
- Rust excels as a lean, resilient control plane for streaming LLMs: async I/O + ownership make back-pressure and cancellation predictable. The main friction is that upstream serving APIs (GGUF/chat formats) evolve quickly, so adapters must be robust and tolerant of schema drift.
- Separation of concerns pays off: keeping inference in llama.cpp and wrapping it with Axum gives a small, portable binary while retaining an OpenAI-like surface area.
- Guardrails matter: explicit `max_tokens`, concurrency caps, and clear defaults build user trust and avoid silent truncation.
- DX/UI value: even a minimal web console accelerates validation and debugging compared to curl-only workflows.
- Future work: richer stop/templating controls, optional auth, alternative backends (e.g., Candle) behind the same trait, and better prompt templates tuned for llama.cpp chat.

## Video Slide Presentation
- [Link to Video Slide Presentation](https://youtu.be/CA2HCZ5bvjY)

## Video Demo
- (Link to be added)
