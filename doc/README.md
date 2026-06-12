# 学习笔记索引

这个目录用于组织 CUDA、推理和深度学习网络搭建相关的学习笔记。不同方向放在不同子目录中，避免后续内容混在同一层。

## CUDA 基础

CUDA 基础笔记已经整理到：

- [CUDA Basics](cuda_basics/README.md)

当前内容包括：

- 内存优化
- 同步机制
- 执行抽象

## 推理

推理笔记已经整理到：

- [Inference](inference/README.md)

当前计划先学习通用推理基础，再学习 TensorRT 部署，并把 LLM、视觉模型和 BEVFusion 拆成独立分支。内容包括：

- General Inference：Hugging Face、PyTorch / Transformers、ONNX、ONNX Runtime
- TensorRT：engine 构建、C++ runtime、dynamic shape、FP16、INT8、plugin
- LLM Deployment：tokenizer、prefill / decode、KV cache、batching、量化、推理框架
- Vision Models：VGG、ResNet、YOLO、VGGT 等视觉模型部署
- BEVFusion Case Study：多模态模型综合实战

## 深度学习网络搭建

后续用于记录网络结构和训练框架相关内容，例如：

- 常见网络层
- 前向传播和反向传播
- 参数、梯度和优化器
- PyTorch / CUDA 扩展结合方式
