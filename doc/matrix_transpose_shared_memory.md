# Matrix Transpose and Shared Memory

## 为什么不直接从 global memory 转置

矩阵转置的核心问题是：直接从 global memory 读写时，读和写很难同时保持连续访问。

直接转置可以写成：

```cpp
output[col * height + row] = input[row * width + col];
```

如果一个 warp 内线程让 `col` 连续变化：

```text
input[row * width + col]  连续读取，global read 容易合并
output[col * height + row] 跨行写入，global write 不连续
```

也就是：

```text
global read  coalesced
global write strided / uncoalesced
```

如果反过来组织线程，让写入连续，读取又会变成跨步访问：

```text
global write coalesced
global read  strided / uncoalesced
```

所以直接转置通常无法让 global memory 的读和写同时高效。

## shared memory 中转的作用

使用 shared memory 的目的不是为了重复读取同一份数据，而是为了改变 global memory 的访问形态。

典型流程：

```text
1. 按行从 global memory 连续读取到 shared memory
2. 在 shared memory 内完成 tile 级别的转置
3. 按行把转置后的数据连续写回 global memory
```

在 `src/matrix_trans.cu` 中，读取阶段：

```cpp
tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * width + x];
```

这里 `threadIdx.x` 连续，所以同一个 warp 内线程读取的 global memory 地址连续，容易 coalescing。

写回阶段：

```cpp
output[(y + i) * height + x] = tile[threadIdx.x][threadIdx.y + i];
```

写回前交换了 block 坐标，使 output 的写入也能保持连续。

最终效果是：

```text
global memory 读：连续、可合并
shared memory：完成 tile 内转置
global memory 写：连续、可合并
```

这比直接转置时让 global memory 读写中至少一边不连续更高效。

## 为什么 shared memory 只读一次也值得

收益不是来自 shared memory 被重复读取，而是来自避免低效的 global memory 访问。

可以理解为：

```text
多走一步很快的 shared memory
换掉一步很慢的 scattered global memory
```

global memory 的非连续访问会导致更多 memory transaction，吞吐明显下降。shared memory 虽然多了一次读写，但延迟低、带宽高，并且可以通过 padding 避免 bank conflict。

因此在大矩阵转置这种访存主导的 kernel 中，shared memory tile transpose 通常是值得的。

## global memory 和 shared memory 的速度差异

粗略量级：

```text
shared memory latency：约 20-40 cycles
global memory latency：约 400-800 cycles，甚至更高
```

单次访问延迟上，global memory 可能比 shared memory 慢 10x 到 30x 以上。

但实际性能不能只看单次 latency，因为 GPU 会用很多机制隐藏 global memory 延迟：

- 大量 warp 并发切换执行
- L1/L2 cache
- memory coalescing
- 高带宽 GDDR/HBM
- 内存访问流水线

对于矩阵转置，更关键的是 memory transaction 数量：

```text
连续、对齐的 global memory 访问 -> transaction 少，吞吐高
跨步、分散的 global memory 访问 -> transaction 多，吞吐低
```

所以 shared memory 中转的核心价值是减少低效 global memory transaction，而不是单纯追求 shared memory 的低延迟。

## 和 bank conflict 的关系

shared memory 中转也有自己的限制。如果 tile 定义为：

```cpp
__shared__ float tile[32][32];
```

转置读取 shared memory 时容易按列访问，导致同一个 warp 内多个线程访问同一个 bank 的不同地址，产生 bank conflict。

所以常见优化是：

```cpp
__shared__ float tile[32][33];
```

多出的 1 列是 padding，用来改变行跨度，让列访问分散到不同 bank。

总结：

```text
shared memory 中转解决 global memory 非连续访问问题
padding 解决 shared memory 内部 bank conflict 问题
两者配合，才能得到高效的矩阵转置
```
