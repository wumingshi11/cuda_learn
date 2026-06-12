# BEVFusion Case Study

这一部分记录 BEVFusion 综合实战。它不是入门模型，而是用来把 CUDA、TensorRT、视觉模型、多输入和自定义算子串起来。

## 1. BEVFusion 推理链路

核心问题：

```text
BEVFusion 的推理流程由多个子模块组成，不是单个 engine 就能解释全部工程。
```

需要掌握：

- 图像输入。
- 点云输入。
- 标定参数。
- image branch。
- LiDAR branch。
- BEV fusion。
- detection head。

## 2. 输入和前处理

核心问题：

```text
多模态模型的输入组织往往决定部署复杂度。
```

需要掌握：

- 多相机图像如何组织。
- 点云如何变成模型可用表示。
- 标定矩阵如何进入模型。
- 前处理在 CPU、CUDA、TensorRT plugin 中实现的取舍。

## 3. ONNX / TensorRT 拆分

核心问题：

```text
复杂模型不一定整体导出，常常需要按模块拆分部署。
```

需要掌握：

- 哪些模块适合直接导出 ONNX。
- 哪些模块需要 plugin 或自定义 CUDA kernel。
- 多个 engine 之间如何传递中间 tensor。
- 如何保证拆分前后结果一致。

## 4. 性能优化路径

核心问题：

```text
先跑通，再定位瓶颈，最后逐步优化。
```

需要掌握：

- 可运行 baseline。
- 精度回归。
- profiler 定位瓶颈。
- FP16 / INT8。
- stream overlap。
- 自定义 kernel / plugin 优化。

## 推荐顺序

```text
BEVFusion 推理链路
-> 输入和前处理
-> ONNX / TensorRT 拆分
-> 性能优化路径
```
