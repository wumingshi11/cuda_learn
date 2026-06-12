# Block-Level Synchronization

## 核心问题

`__syncthreads()` 是 block 级同步。它同步的是同一个 block 内的所有线程。

典型写法：

```cpp
__syncthreads();
```

含义：

```text
同一个 block 内所有线程都必须到达这个同步点
所有线程到达后，整个 block 才继续执行后续代码
```

它还保证同步点之前的 shared memory 写入，对 block 内线程可见。

## shared memory 是 block 级资源

shared memory 的作用域是 block：

```text
每个 block 一份 shared memory
block 内所有线程可见
不同 block 之间不共享
```

例如：

```cpp
__shared__ float tile[32][33];
```

这表示：

```text
每个 block 都有一份 tile
block 内所有 warp 都能访问这份 tile
block 执行结束后，这份 tile 释放
```

因此，如果 shared memory 被跨 warp 使用，就需要 block 级同步。

## 为什么矩阵转置里需要 __syncthreads

在 `src/cuda_basics/matrix_trans.cu` 中：

```cpp
tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * width + x];

__syncthreads();

output[(y + i) * height + x] = tile[threadIdx.x][threadIdx.y + i];
```

这里必须同步，因为：

```text
前半段：block 内线程把 global memory 读入 shared memory
后半段：block 内线程从 shared memory 读出转置后的数据
```

写 tile 的线程和读 tile 的线程可能属于不同 warp。`__syncwarp()` 只能同步单个 warp，不能保证其他 warp 已经完成 shared memory 写入。

所以需要：

```cpp
__syncthreads();
```

来保证整个 block 的 shared memory tile 已经写完。

## 分支中的 __syncthreads

`__syncthreads()` 可以出现在分支中，但要求非常严格：

```text
同一个 block 内所有线程都必须能到达同一个 __syncthreads()
```

危险写法：

```cpp
if (threadIdx.x < 128) {
    __syncthreads();
}
```

如果 block 有 256 个线程，那么只有一半线程进入同步，另一半线程不进入。这可能导致死锁或未定义行为。

安全写法：

```cpp
if (threadIdx.x < 128) {
    // 做一些只由部分线程执行的工作
}

__syncthreads();

if (threadIdx.x < 128) {
    // 同步后继续处理
}
```

如果分支条件对整个 block 一致，例如：

```cpp
if (blockIdx.x == 0) {
    __syncthreads();
}
```

那么这个条件下进入该分支的 block 内所有线程都会进入同步点，语义上是可行的。但实际写代码时，仍建议把 `__syncthreads()` 放在所有 block 线程都会经过的位置，避免误用。

## 为什么 warp 同步不能替代 block 同步

`__syncwarp()` 只同步一个 warp 内 mask 指定的 lane：

```text
warp 0 内同步
```

它不保证：

```text
warp 1 到达同步点
warp 2 到达同步点
整个 block 的 shared memory 写入完成
```

而 `__syncthreads()` 保证的是：

```text
同一个 block 内所有线程都到达同步点
同步点之前的 shared memory 写入对 block 内线程可见
```

因此：

```text
warp 内通信 -> __syncwarp(mask)
跨 warp / block 内 shared memory 通信 -> __syncthreads()
```

## 成本

`__syncthreads()` 的成本通常高于 `__syncwarp()`，因为它是 block 级 barrier：

```text
block 内所有线程都必须到达
所有 warp 都要汇合
快的 warp 要等待慢的 warp
```

block 越大，潜在等待范围越大：

```text
256 threads -> 8 warps 要汇合
512 threads -> 16 warps 要汇合
1024 threads -> 32 warps 要汇合
```

但成本不只由线程数量决定，还取决于：

```text
不同 warp 到达同步点的时间差
同步前是否有长延迟 global memory 访问
是否有分支导致某些 warp 更慢
barrier 使用频率
```

如果所有 warp 几乎同时到达，成本相对可控。如果某些 warp 被长延迟访存拖住，其他 warp 就必须等待。

总结：

```text
__syncthreads():
  同步范围：同一个 block 内所有线程
  代价：通常高于 warp 同步
  适合：block 内 shared memory 阶段切换、跨 warp 通信
```
