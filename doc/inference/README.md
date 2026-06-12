# 推理学习索引

这个目录用于记录模型推理和部署相关学习笔记。整体目标是先掌握通用推理链路，再学习 TensorRT 优化，最后扩展到 LLM、视觉模型和 BEVFusion 这类复杂模型。

## 总体路线

```text
训练好的模型
-> 跑通原始框架推理
-> 明确输入输出和前后处理
-> 导出或转换中间格式
-> 用推理后端验证正确性
-> 构建高性能部署版本
-> 做精度和性能回归
```

## 目录结构

- [General Inference](general/README.md)
- [TensorRT](tensorrt/README.md)
- [LLM Deployment](llm/README.md)
- [Vision Models](vision_models/README.md)
- [BEVFusion Case Study](bevfusion/README.md)

## 1. 通用推理基础

先学习任何模型推理都绕不开的共同问题：

- 训练模型和推理模型的区别。
- 模型结构、权重、输入输出、前处理、后处理之间的关系。
- Hugging Face 上模型的常见使用方式。
- PyTorch / Transformers 跑通原始模型。
- ONNX 的作用和局限。
- ONNX Runtime 验证模型正确性。

学习入口：

- [General Inference](general/README.md)

## 2. TensorRT 通用部署

这条线关注如何把模型部署到 TensorRT：

- ONNX 到 TensorRT engine。
- `trtexec` 和 C++ Runtime。
- input/output buffer 管理。
- CUDA stream 和异步推理。
- dynamic shape。
- FP16 / INT8。
- TensorRT plugin。
- 多输入多输出模型。

学习入口：

- [TensorRT](tensorrt/README.md)

## 3. 大语言模型部署

LLM 部署不是普通 TensorRT 模型的简单放大版，需要单独学习：

- tokenizer。
- prefill / decode。
- KV cache。
- attention 优化。
- batching / continuous batching。
- 量化：FP16、INT8、INT4、AWQ、GPTQ。
- TensorRT-LLM、vLLM、llama.cpp 等推理框架。

学习入口：

- [LLM Deployment](llm/README.md)

## 4. 视觉模型和通用开源模型

这条线覆盖 VGG、ResNet、YOLO、VGGT 等模型：

- 图像前处理。
- 分类、检测、分割、重建等不同输出形式。
- Hugging Face / GitHub 模型使用方式。
- ONNX 导出和验证。
- TensorRT 加速。
- 后处理搬到 CPU、CUDA kernel 或 TensorRT plugin 的取舍。

学习入口：

- [Vision Models](vision_models/README.md)

## 5. BEVFusion 综合实战

BEVFusion 作为最后的综合项目，用来串起前面的知识：

- 多输入：图像、点云、标定参数。
- 多分支：image branch、LiDAR branch、fusion、head。
- 前处理、体素化、特征提取、BEV 融合。
- ONNX / TensorRT 可转换部分。
- 需要自定义 CUDA kernel 或 TensorRT plugin 的部分。
- 从可运行版本逐步优化到高性能部署版本。

学习入口：

- [BEVFusion Case Study](bevfusion/README.md)

## 推荐学习顺序

```text
General Inference
-> TensorRT
-> Vision Models
-> BEVFusion Case Study
```

如果目标是大语言模型，则走：

```text
General Inference
-> LLM Deployment
```

两条线都需要 CUDA 基础中的内存、stream、同步知识作为底层背景。
