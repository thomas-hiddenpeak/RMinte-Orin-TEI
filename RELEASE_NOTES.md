# TEI v0.6.0-qwen3-reranker (Jetson Orin Build)

## 🎯 Overview

This release adds **native Qwen3-Reranker support** to Text Embeddings Inference (TEI), enabling high-performance reranking using Qwen3-Reranker models without any model conversion.

## ✨ Key Features

### Native ListwiseReranker Support
- Direct support for `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-Reranker-4B`, `Qwen/Qwen3-Reranker-8B`
- No model conversion required (but bfloat16 models need fp16 conversion for TEI)
- Full CUDA/Flash Attention acceleration

### Implementation Details
- **Generative Approach**: Extracts "yes"/"no" token logits from last position
- **Scoring**: `P(yes) = softmax([logit_no, logit_yes])[1]`
- **Token IDs**: yes=9693, no=2152
- **Chat Template**: Automatic prompt formatting with customizable instructions

## 📊 Validation Results

Comparison between TEI and Transformers reference implementation:

| Query | Document | Transformers | TEI | Diff |
|-------|----------|-------------|-----|------|
| "What is deep learning?" | Relevant explanation | 0.8616 | 0.8589 | -0.3% |
| "What is deep learning?" | Irrelevant (Paris) | 0.000067 | 0.000067 | ≈0% |
| "What is deep learning?" | Partial (ML models) | 0.000760 | 0.000774 | +1.8% |
| "How to cook pasta?" | Cooking steps | 0.2218 | 0.2173 | -2.0% |
| "How to cook pasta?" | Pasta ingredients | 0.001779 | 0.001744 | -2.0% |
| "How to cook pasta?" | Stock market | 0.000001 | 0.000001 | ≈0% |

**Result**: Accuracy within FP16 precision limits (< 2% difference)

## 🚀 Quick Start

### One-Line Installation (Jetson Orin)

```bash
# Download and run the installation script
curl -sSL https://github.com/thomas-hiddenpeak/RMinte-Orin-TEI/releases/download/v0.6.0-qwen3-reranker-orin/install-sm87.sh | bash
```

Or with custom install directory:
```bash
INSTALL_DIR=~/bin curl -sSL https://github.com/thomas-hiddenpeak/RMinte-Orin-TEI/releases/download/v0.6.0-qwen3-reranker-orin/install-sm87.sh | bash
```

The script will:
- ✅ Check architecture (aarch64)
- ✅ Verify CUDA and compute capability
- ✅ Check required libraries
- ✅ Download the binary from GitHub
- ✅ Install to `/usr/local/bin` (or custom path)
- ✅ Verify installation

### Manual Installation

1. Download the binary:
```bash
wget https://github.com/thomas-hiddenpeak/RMinte-Orin-TEI/releases/download/v0.6.0-qwen3-reranker-orin/text-embeddings-router
chmod +x text-embeddings-router
sudo mv text-embeddings-router /usr/local/bin/
```

### Convert Model (if bfloat16)
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.bfloat16)
model = model.to(torch.float16)
model.save_pretrained("/path/to/fp16-model", safe_serialization=True)
# Copy tokenizer files manually
```

### Start Server
```bash
./text-embeddings-router --model-id /path/to/fp16-model --port 8080
```

### 3. Send Rerank Request
```bash
curl http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "texts": [
      "Deep learning is a subset of machine learning...",
      "Paris is the capital of France."
    ]
  }'
```

## 🔧 Build Information

- **Platform**: NVIDIA Jetson Orin (ARMv8/aarch64)
- **CUDA**: 12.6
- **Compute Capability**: 8.7
- **Build Command**: `CUDA_COMPUTE_CAP=87 cargo build --release -p text-embeddings-router -F candle-cuda`

## 📁 Files Changed

### New Files
- `core/src/templates.rs` - Chat template system for Qwen3 rerankers
- `docs/QWEN3_RERANKER.md` - Comprehensive documentation

### Modified Files
- `backends/candle/src/models/qwen3.rs` - CPU implementation
- `backends/candle/src/models/flash_qwen3.rs` - CUDA/Flash Attention implementation
- `backends/core/src/lib.rs` - Added ListwiseReranker model type
- `core/src/infer.rs` - Skip sigmoid for ListwiseReranker
- `router/src/lib.rs` - Model type detection
- `router/src/http/server.rs` - Template application

## ⚠️ Notes

1. This binary is built for **NVIDIA Jetson Orin** (aarch64 + CUDA 12.6)
2. Models must be in **float16 format** (not bfloat16)
3. The `is_reranker: true` config is automatically detected from model name or can be set in config.json

## 📚 Documentation

See [docs/QWEN3_RERANKER.md](https://github.com/thomas-hiddenpeak/RMinte-Orin-TEI/blob/feature/qwen3-reranker/docs/QWEN3_RERANKER.md) for full implementation details.
