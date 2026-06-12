# Thread, Block and Grid

## 核心问题

CUDA kernel 的执行层级是：

```text
thread -> block -> grid
```

一个 kernel launch 会启动一个 grid。grid 由多个 block 组成，每个 block 由多个 thread 组成。

典型启动方式：

```cpp
kernel<<<grid, block>>>(...);
```

这里：

```text
grid  -> block 的数量和布局
block -> 每个 block 内 thread 的数量和布局
```

## 常用内置变量

CUDA kernel 内常用这些内置变量定位当前线程：

```cpp
threadIdx
blockIdx
blockDim
gridDim
```

含义：

```text
threadIdx -> 当前线程在 block 内的索引
blockIdx  -> 当前 block 在 grid 内的索引
blockDim  -> 每个 block 的线程维度
gridDim   -> grid 的 block 维度
```

例如一维数组：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

这个 `idx` 是当前线程负责处理的全局元素编号。

## 一维映射

向量加法常用一维映射：

```cpp
__global__ void add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

启动：

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;

add<<<blocks, threads>>>(a, b, c, n);
```

边界判断：

```cpp
if (idx < n)
```

用于处理最后一个 block 可能不满的情况。

## 二维映射

矩阵通常用二维 block/grid 更自然：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

例如矩阵加法：

```cpp
int idx = row * width + col;
```

二维映射让代码直接对应矩阵坐标：

```text
x -> column
y -> row
```

## 矩阵转置中的二维 block/grid

在 `src/cuda_basics/matrix_trans.cu` 中：

```cpp
dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
          (height + TILE_DIM - 1) / TILE_DIM);
dim3 block(TILE_DIM, BLOCK_ROWS);
```

这里：

```text
grid.x -> 水平方向 tile 数量
grid.y -> 垂直方向 tile 数量
block.x = TILE_DIM
block.y = BLOCK_ROWS
```

`block(32, 8)` 不是说只处理 `32 x 8` 个矩阵元素。它是用 256 个线程处理一个 `32 x 32` tile，每个线程通过循环处理多个元素：

```cpp
for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    ...
}
```

当 `TILE_DIM = 32`、`BLOCK_ROWS = 8`：

```text
i = 0, 8, 16, 24
每个线程处理 4 个元素
256 threads * 4 = 1024 elements = 32 * 32 tile
```

## block 是同步和 shared memory 的边界

block 很重要，因为：

```text
shared memory 是 block 级资源
__syncthreads() 只能同步同一个 block 内线程
不同 block 之间不能直接共享 shared memory
普通 kernel 内不同 block 之间没有通用同步
```

所以 CUDA kernel 设计时经常要考虑：

```text
每个 block 处理一块独立数据
block 内线程通过 shared memory 协作
block 之间通过 global memory 或多个 kernel 阶段交接
```

## 总结

```text
thread:
  最小执行单元，负责一个或多个数据元素

block:
  thread 的分组，共享 shared memory，可用 __syncthreads()

grid:
  block 的集合，对应一次 kernel launch
```

一句话：

```text
Thread / Block / Grid 是 CUDA 把数据任务映射到 GPU 执行资源上的基本抽象。
```
