# Occupancy

## 核心问题

occupancy 描述的是一个 SM 上实际驻留的 active warps 相对于硬件最大 warps 的比例。

可以粗略理解为：

```text
occupancy = 当前 SM 上 active warps 数 / SM 支持的最大 active warps 数
```

它影响 GPU 隐藏延迟的能力。

## 为什么需要多个 active warps

GPU 通过大量 warp 并发来隐藏延迟。

如果一个 warp 等 global memory：

```text
warp 0 等内存
```

SM 可以切换去执行其他 ready warp：

```text
warp 1
warp 2
warp 3
...
```

因此 active warps 越多，通常越容易隐藏长延迟访存。

但 occupancy 高不等于性能一定最高。

## 什么会限制 occupancy

每个 SM 的资源有限。一个 block 使用的资源越多，一个 SM 能同时驻留的 block/warp 可能越少。

常见限制：

```text
每线程 register 数量
每 block shared memory 使用量
block size
每 SM 最大 block 数
每 SM 最大 warp 数
```

## Register 对 occupancy 的影响

每个 SM 的寄存器总量有限。

假设：

```text
一个 SM 有 65536 个寄存器
block 有 256 个线程
每线程使用 32 个寄存器
```

每个 block 消耗：

```text
32 * 256 = 8192 个寄存器
```

从寄存器角度看，一个 SM 最多可驻留：

```text
65536 / 8192 = 8 个 block
```

如果每线程使用 64 个寄存器：

```text
64 * 256 = 16384 个寄存器 / block
65536 / 16384 = 4 个 block
```

active block/warp 变少，occupancy 可能下降。

但强行减少寄存器也可能导致 spill 到 local memory，反而更慢。

## Shared memory 对 occupancy 的影响

每个 block 使用 shared memory 越多，一个 SM 能同时驻留的 block 越少。

例如：

```text
SM 可用 shared memory = 64 KB
每 block 使用 16 KB -> 最多 4 blocks
每 block 使用 32 KB -> 最多 2 blocks
```

所以 shared memory tiling 虽然能提升访存效率，但也会消耗 SM 资源，影响 occupancy。

## Block size 对 occupancy 的影响

block size 决定每个 block 有多少线程，也决定每个 block 有多少 warp。

例如：

```text
128 threads -> 4 warps
256 threads -> 8 warps
512 threads -> 16 warps
```

block 太小：

```text
每个 block 可用线程少
可能难以充分利用 SM
```

block 太大：

```text
每个 block 资源消耗大
可同时驻留 block 数减少
同步等待范围也变大
```

常见初始选择是：

```text
128 / 256 / 512 threads per block
```

具体要结合 kernel 资源使用和 profiler 判断。

## occupancy 高不一定性能最好

occupancy 是延迟隐藏能力的指标，不是最终性能指标。

例如：

```text
更高 occupancy 可能需要限制寄存器
限制寄存器可能导致 local memory spill
spill 会显著增加访存成本
```

或者：

```text
使用更多 shared memory 降低 occupancy
但显著减少 global memory transaction
整体性能仍可能更高
```

所以优化目标不是盲目追求最高 occupancy，而是平衡：

```text
访存效率
寄存器使用
shared memory 使用
active warps
同步成本
```

## 如何观察

可以通过编译器和 profiler 查看资源使用。

编译时查看寄存器使用：

```sh
nvcc -Xptxas -v ...
```

可能看到：

```text
Used 32 registers
```

性能分析工具可以观察：

```text
achieved occupancy
active warps
register usage
shared memory usage
local memory spill
memory throughput
```

## 总结

```text
occupancy:
  一个 SM 上 active warps 的驻留程度

作用:
  帮助隐藏内存和执行延迟

限制因素:
  register、shared memory、block size、硬件最大 block/warp 数

注意:
  occupancy 高不一定性能最好
```

一句话：

```text
occupancy 是重要诊断指标，但不是唯一优化目标。
```
