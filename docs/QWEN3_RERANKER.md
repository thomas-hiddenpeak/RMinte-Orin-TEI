# Qwen3-Reranker Support for Text Embeddings Inference

本文档详细说明了为 Text Embeddings Inference (TEI) 添加原生 Qwen3-Reranker 支持的所有修改。

## 概述

Qwen3-Reranker 是阿里巴巴通义千问团队发布的生成式重排序模型，基于 Qwen3 架构。与传统的交叉编码器 (Cross-Encoder) 不同，它使用生成式方法进行相关性判断：

- **输入**: Chat 模板格式的 prompt，包含系统指令、query 和 document
- **输出**: 模型生成 "yes" 或 "no" 来判断相关性
- **评分**: P(yes) = softmax([logit_no, logit_yes])[1]

## 支持的模型

- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen3-Reranker-4B`
- `Qwen/Qwen3-Reranker-8B`

## 主要修改

### 1. 新增模型类型: `ListwiseReranker`

**文件**: `backends/core/src/lib.rs`

```rust
pub enum ModelType {
    Embedding(Pool),
    Classifier,
    ListwiseReranker,  // 新增
}
```

### 2. 模板系统

**新文件**: `core/src/templates.rs`

为 Qwen3-Reranker 实现专用模板格式：

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
```

默认 instruction: `"Given a web search query, retrieve relevant passages that answer the query"`

### 3. Qwen3 模型实现

**文件**: `backends/candle/src/models/qwen3.rs`

- 实现 `Qwen3Config` 配置结构
- 实现 `Qwen3Model` 支持 embedding 和 reranking
- `predict()` 方法：提取最后一个 token 的 hidden state，计算 yes/no logits

**文件**: `backends/candle/src/models/flash_qwen3.rs`

- Flash Attention 优化的 CUDA 实现
- 支持 FP16 推理
- 使用 tied weights (embed_tokens.weight) 作为 LM head

### 4. 模型类型检测

**文件**: `router/src/lib.rs`

```rust
fn get_backend_model_type(...) -> Result<ModelType, ...> {
    // 检测 Qwen3-Reranker 模型
    if model_id.contains("Qwen3") && model_id.contains("Reranker") {
        return Ok(ModelType::ListwiseReranker);
    }
    // ...
}
```

### 5. 评分处理修复

**文件**: `core/src/infer.rs`

ListwiseReranker 返回的已经是 P(yes) 概率值，不需要再做 sigmoid 变换：

```rust
let is_listwise_reranker = matches!(
    self.backend.model_type,
    text_embeddings_backend::ModelType::ListwiseReranker
);

if !raw_scores && !is_listwise_reranker {
    // 只对普通 Classifier 应用 sigmoid
}
```

### 6. HTTP API 修改

**文件**: `router/src/http/server.rs`

- 自动检测 Qwen3-Reranker 模型并应用模板
- 支持 `use_template` 参数控制模板使用

## Token ID

Qwen3 tokenizer 中的关键 token:
- `yes` → 9693
- `no` → 2152

## 编译方法

### NVIDIA Jetson Orin (Compute Capability 8.7)

```bash
CUDA_COMPUTE_CAP=87 cargo build --release -p text-embeddings-router -F candle-cuda
```

### 其他 CUDA 设备

```bash
# Ampere (A100, A10, RTX 3090)
CUDA_COMPUTE_CAP=80 cargo build --release -p text-embeddings-router -F candle-cuda

# Turing (T4, RTX 2080)
CUDA_COMPUTE_CAP=75 cargo build --release -p text-embeddings-router -F candle-cuda-turing
```

## 运行方法

```bash
./text-embeddings-router \
    --model-id Qwen/Qwen3-Reranker-0.6B \
    --dtype float16 \
    --max-batch-tokens 65536 \
    --port 8080
```

**注意**: 如果模型是 bfloat16 格式，需要先转换为 float16：

```python
from safetensors.torch import load_file, save_file
import torch

tensors = load_file("model.safetensors")
converted = {k: v.to(torch.float16) if v.dtype == torch.bfloat16 else v 
             for k, v in tensors.items()}
save_file(converted, "model.safetensors")
```

## API 使用

### Rerank 端点

```bash
curl http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Deep Learning?",
    "texts": [
      "Deep Learning is a subset of machine learning...",
      "Paris is the capital of France."
    ]
  }'
```

**响应**:
```json
[
  {"index": 0, "score": 0.8589},
  {"index": 1, "score": 0.00007}
]
```

## 验证结果

与 Transformers 对照测试结果：

| Query | Document | Transformers | TEI | 差异 |
|-------|----------|-------------|-----|------|
| What is Deep Learning? | DL explanation | 0.8616 | 0.8589 | -0.3% |
| What is Deep Learning? | Paris is capital... | 0.000067 | 0.000067 | ≈0 |
| What is Deep Learning? | ML models learn... | 0.000760 | 0.000774 | +1.8% |
| How to cook pasta? | Cooking instructions | 0.2218 | 0.2173 | -2% |
| How to cook pasta? | Pasta ingredients | 0.001779 | 0.001744 | -2% |
| How to cook pasta? | Stock market crashed | 0.000001 | 0.000001 | ≈0 |

微小差异来自 FP16 精度和 Flash Attention 实现差异，属于正常范围。

## 文件修改清单

### 新增文件
- `core/src/templates.rs` - 模板系统
- `docs/QWEN3_RERANKER.md` - 本文档
- `.github/copilot-instructions.md` - Copilot 指令

### 修改文件
- `backends/core/src/lib.rs` - 添加 ListwiseReranker 类型
- `backends/candle/src/models/qwen3.rs` - Qwen3 CPU 实现
- `backends/candle/src/models/flash_qwen3.rs` - Qwen3 CUDA 实现
- `backends/candle/src/models/mod.rs` - 模块导出
- `backends/candle/src/lib.rs` - 模型加载逻辑
- `core/src/lib.rs` - 导出 templates 模块
- `core/src/infer.rs` - 修复 ListwiseReranker 评分处理
- `router/src/lib.rs` - 模型类型检测
- `router/src/http/server.rs` - 模板应用逻辑

## 许可证

本项目基于 Apache 2.0 许可证，与原始 TEI 项目保持一致。
