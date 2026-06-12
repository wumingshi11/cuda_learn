# General Inference

这一部分记录模型推理的通用基础。无论后面使用 ONNX Runtime、TensorRT、TensorRT-LLM、vLLM，第一步都是先把模型在原始框架里跑通，并明确输入输出语义。

## 1. 推理整体流程

核心问题：

```text
一个训练好的模型要真正跑起来，需要哪些工程环节。
```

需要掌握：

- 模型结构、权重、输入输出约定、前处理、后处理之间的关系。
- `.pth`、`.onnx`、TensorRT `.engine` / `.plan` 的区别。
- 模型文件通常只描述网络计算本体，前处理和后处理经常需要推理程序自己实现。
- ONNX 主要表达标准算子组成的计算图，自定义算子或框架私有逻辑需要额外处理。
- 网络计算图可以有分支、合并和多输入多输出，但部署格式更擅长表达静态计算图，不擅长表达任意 Python 控制逻辑。
- 训练代码和推理代码为什么不是一回事。
- 推理程序的基本数据流：读取模型、准备输入、前处理、执行、后处理、校验结果。
- 为什么要先验证正确性，再做性能优化。

### 1.1 模型文件格式

训练好模型后，常见文件格式有三类：

```text
.pth
= PyTorch 训练结果，通常主要保存权重参数，依赖原始 Python 网络结构代码。

.onnx
= 跨框架计算图，通常包含网络结构、标准算子、权重、输入输出 tensor 描述。

.engine / .plan
= TensorRT 针对具体 GPU、TensorRT 版本、shape profile、precision 策略优化后的推理执行计划。
```

更准确地说：

- `.pth` 不一定包含完整网络结构。常见 `state_dict` 只是“参数名到 tensor”的映射，加载时仍需要模型类定义。
- `.onnx` 更像“要算什么”的描述，适合做跨框架验证和 TensorRT 解析。
- TensorRT `.engine` / `.plan` 更像“在这张 GPU 上怎么高效算”的执行计划，不适合作为跨平台通用模型格式。

### 1.2 模型文件不等于完整推理程序

模型文件里通常只包含网络计算本体。真正可运行的推理系统还需要：

```text
前处理
-> 网络执行
-> 后处理
```

例如图像分类模型中，网络可能只知道输入是 `[1, 3, 224, 224]` 的 tensor，但它通常不知道：

- 原始图片如何 decode。
- 是否要 resize / crop。
- 使用 RGB 还是 BGR。
- 是否要减 mean、除 std。
- 输出类别 id 如何映射到类别名。

检测、分割、三维感知和大语言模型更明显。NMS、box decode、tokenizer、点云体素化、标定参数处理等，经常在模型文件之外，由推理代码、配置文件或自定义 plugin 完成。

### 1.3 ONNX 和计算图

ONNX 主要表达标准算子组成的计算图。它可以表达：

- 顺序网络。
- 有分支的网络。
- 多输入、多输出网络。
- 残差连接、concat、add、attention 等图结构。

所以网络不是只能单向一条链路。ResNet、Inception、Transformer、BEVFusion 这类模型都有分支、合并或多输入结构。

ONNX 的限制主要在于：

- 不支持或不完全支持某些框架私有算子。
- 任意 Python 逻辑不能直接放进 ONNX。
- 动态 shape、循环、条件分支虽然有表达方式，但部署后端未必都支持得好。
- 自定义算子需要 ONNX Runtime custom op 或 TensorRT plugin 等额外机制。

## 2. Hugging Face 模型使用方式

核心问题：

```text
Hugging Face 上的模型通常如何下载、加载和运行。
```

需要掌握：

- Model Card、Files and versions、Use this model 的作用。
- `pipeline()` 的快速使用方式。
- `AutoTokenizer`、`AutoProcessor`、`AutoModel`、`AutoModelForCausalLM` 的区别。
- `from_pretrained()` 如何加载模型和权重。
- `trust_remote_code=True` 的含义和风险。
- 标准 Transformers 模型和依赖自定义 repo 代码模型的区别。

## 3. PyTorch / Transformers 原始推理

核心问题：

```text
先用训练框架或官方代码跑通模型，作为后续部署的正确性基准。
```

需要掌握：

- CPU / GPU 上执行模型推理。
- `model.eval()` 和 `torch.no_grad()` 的作用。
- 输入 tensor 的 shape、dtype、device。
- 输出 tensor 如何解释。
- 如何保存一组输入输出作为回归测试样本。

## 4. ONNX 基础

核心问题：

```text
ONNX 是模型部署的中间表示，但不是所有模型都能无损表达。
```

需要掌握：

- `torch.onnx.export` 的基本使用方式。
- dummy input 的作用。
- opset version 的含义。
- 静态 shape 和动态 shape 导出差异。
- 导出后如何检查输入输出名称、shape 和算子。

## 5. ONNX Runtime 验证

核心问题：

```text
在进入高性能部署前，先确认 ONNX 模型语义正确。
```

需要掌握：

- ONNX Runtime 如何加载和执行 ONNX。
- 如何比较 PyTorch 输出和 ONNX Runtime 输出。
- 浮点误差的判断方式。
- ONNX 模型常见导出问题如何定位。

## 推荐顺序

```text
推理整体流程
-> Hugging Face 模型使用方式
-> PyTorch / Transformers 原始推理
-> ONNX 基础
-> ONNX Runtime 验证
```
