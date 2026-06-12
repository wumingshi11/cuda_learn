# LLM Deployment

这一部分记录大语言模型部署。LLM 部署不是普通神经网络推理的简单放大版，核心差异在于自回归生成、KV cache 和 attention 计算。

## 1. LLM 推理整体流程

核心问题：

```text
语言模型为什么不是一次 forward 就结束，而是逐 token 生成。
```

需要掌握：

- tokenizer 和 token id。
- prompt、input ids、attention mask。
- prefill 和 decode 的区别。
- logits、采样、temperature、top-k、top-p。

## 2. KV Cache

核心问题：

```text
为什么 LLM 推理需要缓存历史 key/value。
```

需要掌握：

- attention 中 Q、K、V 的作用。
- prefill 阶段如何生成 KV cache。
- decode 阶段如何复用 KV cache。
- KV cache 为什么会占用大量显存。

## 3. Batch 和 Continuous Batching

核心问题：

```text
多个请求长度不同，如何高效组织推理。
```

需要掌握：

- 静态 batch 和动态请求的矛盾。
- padding 的成本。
- continuous batching 的基本思想。
- paged attention 解决的问题。

## 4. LLM 量化

核心问题：

```text
如何在显存和速度压力下运行更大的模型。
```

需要掌握：

- FP16 / BF16。
- INT8 / INT4。
- AWQ、GPTQ 等权重量化路线。
- 量化对显存、速度、精度的影响。

## 5. 常见 LLM 推理框架

核心问题：

```text
不同框架解决的问题不同，不应只按“能不能跑”选择。
```

需要掌握：

- Transformers：最通用，适合验证。
- vLLM：服务化和高吞吐。
- TensorRT-LLM：NVIDIA GPU 上的高性能部署。
- llama.cpp：轻量和量化生态。

## 推荐顺序

```text
LLM 推理整体流程
-> KV Cache
-> Batch / Continuous Batching
-> LLM 量化
-> 常见 LLM 推理框架
```
