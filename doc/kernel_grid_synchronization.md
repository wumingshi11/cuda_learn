# Kernel / Grid-Level Synchronization

## 核心问题

普通 CUDA kernel 内没有通用的 grid-wide barrier。

也就是说，没有类似下面这样的普通内置同步：

```cpp
grid_syncthreads(); // 普通 CUDA kernel 中没有这种通用同步
```

`__syncthreads()` 只能同步同一个 block 内的线程，不能同步不同 block。

因此，普通 CUDA 中最常用的全局同步方式是：

```text
拆成多个 kernel，用 kernel launch 边界作为全局同步点。
```

## Kernel launch 边界是全局同步点

两个 kernel 在同一个 stream 中启动：

```cpp
kernel1<<<grid1, block1>>>(...);
kernel2<<<grid2, block2>>>(...);
```

CUDA 保证：

```text
kernel1 完成后，kernel2 才开始执行。
```

所以：

```text
kernel1 结束
-> 所有 block 都完成
-> global memory 中的结果对 kernel2 可见
-> kernel2 开始
```

这里的同步来自 kernel launch 的顺序和 stream 语义，不是来自 `__global__` 关键字本身。

## __global__ / __device__ 不是同步手段

`__global__` 表示 kernel 入口：

```cpp
__global__ void kernel(...) {
}
```

它可以从 host 端 launch。

`__device__` 表示 device 端函数：

```cpp
__device__ float helper(...) {
}
```

它通常在 kernel 内被 device 线程调用。

它们本身都不是同步机制：

```text
__global__ -> 定义 kernel 入口
__device__ -> 定义 device 端辅助函数
kernel launch 边界 -> 常用全局同步点
```

## 典型应用：多阶段归约

普通 grid 内不能直接同步所有 block，所以全局归约通常拆成多个阶段。

第一阶段：

```text
每个 block 处理一段 input
block 内用 shared memory / warp reduction 得到 block_sum
每个 block 把 block_sum 写入 partial[blockIdx.x]
```

第二阶段：

```text
启动另一个 kernel
读取 partial[]
继续归约得到最终结果
```

结构：

```cpp
reduce_stage1<<<grid1, block>>>(input, partial);
reduce_stage2<<<grid2, block>>>(partial, output);
```

同步点是：

```text
reduce_stage1 kernel 结束。
```

不一定要把 `partial[]` 拷回 CPU。可以继续在 GPU 上做下一阶段归约。

## 拷回 CPU 累加

也可以：

```text
GPU kernel 生成 partial[]
cudaMemcpy partial[] 到 CPU
CPU 上累加
```

这种方式适合：

```text
partial 数量很小
后续本来就要 CPU 使用结果
实现简单优先
```

如果 partial 很多，或者后续仍要 GPU 使用结果，通常更适合继续在 GPU 上做多阶段归约。

## Global atomic 不是 grid 同步

可以用 global atomic 做跨 block 累计：

```cpp
atomicAdd(global_sum, block_sum);
```

推荐模式通常是：

```text
每个 block 先在 block 内归约出 block_sum
每个 block 只 atomicAdd 一次
```

这比每个线程都 atomicAdd 更好。

但 atomic 不是 grid synchronization：

```text
atomic 解决的是多个 block 并发更新同一个地址不会互相覆盖
atomic 不表示所有 block 到达某个同步点
atomic 不让所有 block 等待
```

它适合：

```text
block 数不太多
累计开销不是主瓶颈
单 kernel 简化实现
```

它的代价：

```text
竞争严重时会成为瓶颈
浮点 atomic 顺序不确定，结果可能有细微差异
不能表达“所有 block 都到齐”
```

## Fence + atomic 的完成计数模式

如果某个 block 写完数据后，再用 atomic 更新完成计数，通常需要考虑 fence：

```cpp
partial[blockIdx.x] = block_sum;

__threadfence();

int ticket = atomicAdd(counter, 1);
```

含义：

```text
先保证 partial 写入对 device 可见
再通过 counter 通知其他线程这个 block 已完成
```

这个模式仍然不是普通的 grid barrier。它只是用 atomic/fence 构造特定的跨 block 通知协议，复杂且容易写错。

多数情况下，多 kernel 分阶段更清晰。

## Cooperative Groups 的 grid sync

CUDA cooperative groups 可以在特定条件下做 grid 级同步：

```cpp
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void kernel(...) {
    cg::grid_group grid = cg::this_grid();

    // stage 1

    grid.sync();

    // stage 2
}
```

但它不是普通 kernel 默认能力。限制包括：

```text
需要 cooperative launch
硬件和 driver 支持
所有 block 必须能同时驻留在 GPU 上
可能限制 grid size
会影响调度和 occupancy
```

因此 cooperative groups grid sync 适合特殊场景，不是普通 CUDA 代码的默认全局同步方式。

## 选择建议

全局累计 / 归约通常按这个优先级考虑：

```text
高性能通用路径:
  多阶段 GPU reduction

简单单 kernel:
  block 内归约 + 每 block 一次 atomicAdd

partial 很小且 CPU 需要结果:
  partial 拷回 CPU 累加

特殊场景:
  cooperative groups grid.sync
```

总结：

```text
普通 CUDA kernel 内没有通用 grid 同步。
最常用的全局同步点是 kernel launch 边界。
atomic 可以做跨 block 更新，但不是同步屏障。
cooperative groups 可以 grid sync，但限制较多。
```
