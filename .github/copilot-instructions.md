# Copilot Instructions for Text Embeddings Inference (TEI)

## Project Overview
TEI is a high-performance Rust inference server for text embedding, sequence classification, and reranking models. It uses the Candle deep learning framework with Flash Attention optimizations for GPU acceleration.

## Architecture

### Crate Structure
- **`router/`** - Main binary, HTTP/gRPC server using Axum, CLI args parsing
- **`core/`** - Inference orchestration (`Infer`), request queuing, tokenization, prompt templates
- **`backends/`** - Backend abstraction layer selecting runtime (Candle, ONNX)
- **`backends/candle/`** - Candle-based model implementations with Flash Attention variants

### Data Flow
1. Request → `router/src/http/server.rs` (rerank, embed, predict endpoints)
2. Template formatting → `core/src/templates.rs` (for Qwen3 rerankers)
3. Tokenization → `core/src/tokenization.rs`
4. Batching → `core/src/queue.rs`
5. Inference → `backends/candle/src/lib.rs` → model files

### Model Pattern
Each supported model has two implementations in `backends/candle/src/models/`:
- Standard (CPU/Metal): `bert.rs`, `nomic.rs`, `qwen3.rs`, etc.
- Flash Attention (CUDA): `flash_bert.rs`, `flash_nomic.rs`, `flash_qwen3.rs`, etc.

Models implement the `Model` trait from `backends/candle/src/models/mod.rs`:
- `embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)>` - for embeddings
- `predict(&self, batch: Batch) -> Result<Tensor>` - for classification/reranking

### Model Types
Defined in `backends/core/src/lib.rs`:
- `ModelType::Embedding(Pool)` - embedding models with pooling strategy
- `ModelType::Classifier` - sequence classification models
- `ModelType::ListwiseReranker` - generative rerankers (e.g., Qwen3-Reranker)

## Build Commands

```bash
# CPU with ONNX backend (recommended for dev)
cargo build --release -p text-embeddings-router -F ort

# CPU with Intel MKL
cargo build --release -p text-embeddings-router -F mkl

# macOS Metal
cargo build --release -p text-embeddings-router -F metal

# CUDA Ampere+ (A100, A10, H100)
CUDA_COMPUTE_CAP=80 cargo build --release -p text-embeddings-router -F candle-cuda

# CUDA Turing (T4, RTX 2000)
CUDA_COMPUTE_CAP=75 cargo build --release -p text-embeddings-router -F candle-cuda-turing
```

## Testing

```bash
# CPU integration tests
make integration-tests

# CUDA integration tests (requires GPU)
make cuda-integration-tests

# Review snapshot changes
make integration-tests-review
make cuda-integration-tests-review
```

Tests use `insta` for snapshot testing. Model tests are in `backends/candle/tests/` with snapshots in `backends/candle/tests/snapshots/`. Tests download models from HuggingFace Hub automatically.

## Feature Flags
Key feature combinations in `router/Cargo.toml`:
- `http` / `grpc` - Server protocol
- `candle` - Candle backend
- `candle-cuda` - Candle + Flash Attention v2 (Ampere+)
- `candle-cuda-turing` - Candle + Flash Attention v1 (Turing)
- `ort` - ONNX Runtime backend
- `metal` / `mkl` / `accelerate` - Platform-specific acceleration

## Adding a New Model

1. Add config struct in `backends/candle/src/models/{model}.rs`
2. Implement `Model` trait with `embed()` and optionally `predict()` methods
3. Add CUDA variant in `backends/candle/src/models/flash_{model}.rs` if needed
4. Register in `backends/candle/src/models/mod.rs` exports
5. Add config parsing in `backends/candle/src/lib.rs` `Config` enum
6. Add model type detection in `router/src/lib.rs` `get_backend_model_type()`
7. Add tests in `backends/candle/tests/test_{model}.rs`

## Qwen3 Reranker Implementation

Qwen3 Rerankers (`Qwen/Qwen3-Reranker-*`) use a generative approach:
1. **Input**: Chat template with system prompt asking for "yes"/"no" judgment
2. **Output**: Extract logits at last position for "yes"/"no" tokens
3. **Score**: `P(yes) = softmax([logit_no, logit_yes])[1]`

Key files:
- `core/src/templates.rs` - `Qwen3RerankerTemplate` for prompt formatting
- `backends/candle/src/models/qwen3.rs` - CPU implementation with `predict()` for ListwiseReranker
- `backends/candle/src/models/flash_qwen3.rs` - CUDA implementation
- `router/src/lib.rs` - Model type detection via `is_reranker` config or model name

Template format:
```
<|im_start|>system
Judge whether the Document meets the requirements...<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
```

## Code Conventions

- Use `#[serial_test::serial]` for GPU tests (single GPU resource)
- Tests compare against snapshots using cosine similarity (threshold 0.999)
- Model configs use `#[serde(rename_all = "kebab-case")]` for HuggingFace compatibility
- CUDA code gated with `#[cfg(feature = "cuda")]`
- Weights path detection: check for `model.{key}` prefix for nested model structures

## Running the Server

```bash
# Embedding model
cargo run --release -p text-embeddings-router -F ort -- --model-id BAAI/bge-small-en-v1.5

# Qwen3 Reranker with GPU
CUDA_COMPUTE_CAP=80 cargo run --release -p text-embeddings-router -F candle-cuda -- --model-id Qwen/Qwen3-Reranker-0.6B
```

## Key Environment Variables
- `CUDA_COMPUTE_CAP` - GPU compute capability (75, 80, 86, 89, 90)
- `HF_TOKEN` - HuggingFace Hub token for gated models
- `MODEL_ID` - Model to load (default: `BAAI/bge-large-en-v1.5`)
- `USE_FLASH_ATTENTION` - Enable/disable flash attention (default: `True`)
