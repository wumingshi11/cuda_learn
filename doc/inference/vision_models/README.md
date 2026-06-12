# Vision Models

这一部分记录视觉模型和通用开源模型部署，包括 VGG、ResNet、YOLO、VGGT 等。

## 1. 图像模型推理流程

核心问题：

```text
视觉模型的输入输出通常比纯 tensor forward 多一层前后处理语义。
```

需要掌握：

- image decode。
- resize、crop、normalize。
- NCHW / NHWC。
- BGR / RGB。
- 输出类别、box、mask、depth、point map 等不同结果形式。

## 2. 分类模型

核心问题：

```text
从 VGG / ResNet 这类简单模型学习完整部署链路。
```

需要掌握：

- 图像分类前处理。
- softmax 和 top-k。
- PyTorch / ONNX / TensorRT 输出一致性。
- 简单模型如何作为推理框架验证样例。

## 3. 检测和分割模型

核心问题：

```text
检测和分割模型的后处理往往和网络本体一样重要。
```

需要掌握：

- box decode。
- NMS。
- mask 后处理。
- 后处理放在 CPU、CUDA kernel 或 TensorRT plugin 的取舍。

## 4. VGGT 等复杂视觉模型

核心问题：

```text
复杂开源视觉模型通常有自己的 repo、依赖和自定义推理流程。
```

需要掌握：

- Hugging Face 权重和 GitHub 代码如何配合。
- 官方 demo 如何拆成前处理、模型执行、后处理。
- 哪些部分能导出 ONNX。
- 哪些部分适合保留 Python，哪些部分适合 TensorRT 化。

## 推荐顺序

```text
图像模型推理流程
-> 分类模型
-> 检测和分割模型
-> VGGT 等复杂视觉模型
```
