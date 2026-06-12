# TensorRT

这一部分记录通用 TensorRT 模型部署。目标是理解 ONNX 如何变成 TensorRT engine，以及 C++ 程序如何加载 engine 并执行推理。

## 1. TensorRT Engine 构建

核心问题：

```text
TensorRT 如何把 ONNX 网络优化成可执行的 engine。
```

需要掌握：

- builder、network、parser、config、engine 的关系。
- `trtexec` 的基本用法。
- workspace、optimization profile、precision flag 的作用。
- engine 为什么和 GPU 架构、TensorRT 版本强相关。
- 为什么 engine 通常不作为跨平台模型格式。

## 2. TensorRT C++ Runtime

核心问题：

```text
如何在 C++ 中加载 TensorRT engine 并执行推理。
```

需要掌握：

- runtime、engine、execution context 的关系。
- binding / tensor name 如何对应输入输出。
- host buffer 和 device buffer 如何分配。
- `cudaMemcpyAsync`、CUDA stream 和 `enqueueV2` / `enqueueV3` 的基本关系。
- 推理结束后如何取回输出并做后处理。

## 3. Dynamic Shape

核心问题：

```text
输入尺寸不固定时，TensorRT 如何选择可执行的优化范围。
```

需要掌握：

- min / opt / max shape 的含义。
- optimization profile 的作用。
- runtime 设置实际输入 shape 的流程。
- 动态 shape 对性能和显存的影响。

## 4. FP16 和 INT8

核心问题：

```text
在可接受精度损失内，用低精度提高推理吞吐。
```

需要掌握：

- FP32、FP16、INT8 的性能和精度差异。
- FP16 为什么通常比 INT8 更容易落地。
- INT8 calibration 的作用。
- 哪些层可能对低精度敏感。
- 如何做精度回归测试。

## 5. TensorRT Plugin

核心问题：

```text
当 TensorRT 不支持某些算子时，如何用 plugin 补齐。
```

需要掌握：

- plugin 解决的是“不支持算子”或“自定义融合算子”的问题。
- plugin 和普通 CUDA kernel 的关系。
- plugin 的输入输出 shape、数据类型、序列化接口。
- 为什么复杂模型经常绕不开 plugin。

## 6. 多输入多输出模型

核心问题：

```text
真实模型通常不只是一个 input tensor 和一个 output tensor。
```

需要掌握：

- 多输入 binding / tensor name 管理。
- 不同输入的 host/device buffer 如何组织。
- 多输出后处理如何拆分。
- batch 维度和多 sensor 输入的区别。

## 推荐顺序

```text
Engine 构建
-> C++ Runtime
-> Dynamic Shape
-> FP16 / INT8
-> Plugin
-> 多输入多输出模型
```
