# cuda_learn

记录学习 CUDA 和推理部署的示例，方便后续按主题补充与复盘。

## CUDA 学习示例

### 1. 向量加法（入门）

```cpp
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

编译运行示例：

```bash
nvcc vec_add.cu -o vec_add
./vec_add
```

### 2. 共享内存（进阶）

- 学习点：`__shared__`、线程块内同步 `__syncthreads()`
- 场景：矩阵乘法优化、卷积计算优化

## 推理部署示例

### 1. ONNX Runtime（基础部署）

```bash
python infer_onnx.py --model model.onnx --input demo.jpg
```

### 2. TensorRT（GPU 加速部署）

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16
```

```bash
python infer_tensorrt.py --engine model.plan --input demo.jpg
```

## 后续补充方向

- CUDA：流（stream）、事件（event）、异步拷贝、性能分析（Nsight）
- 推理：动态 shape、INT8 量化、服务化部署（Triton）
