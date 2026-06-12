# Warp and SIMT

## 核心问题

GPU 通常不是给每个线程一个独立调度器，而是以 warp 为重要调度单位。

在 NVIDIA GPU 中：

```text
一个 warp = 32 个线程
```

warp 内线程通常执行同一条指令，这种模型叫 SIMT：

```text
Single Instruction, Multiple Threads
```

## Warp 和 Block 的关系

一个 block 可以包含多个 warp。

例如：

```cpp
dim3 block(32, 8);
```

线程数：

```text
32 * 8 = 256 threads
```

warp 数：

```text
256 / 32 = 8 warps
```

所以：

```text
block 是程序员配置的线程分组
warp 是硬件执行和调度的重要单位
```

## SIMT 的含义

SIMT 可以理解为：

```text
一个 warp 内多个线程执行同一条指令
但每个线程有自己的寄存器、threadIdx 和数据
```

例如：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = input[idx];
```

warp 内线程执行同一条 load 指令，但因为 `threadIdx.x` 不同，每个线程访问不同元素。

## Warp 调度不是每线程独立调度

可以粗略理解为：

```text
调度器发射 warp 级指令
warp 内 lane 执行同一条指令
每个 lane 使用自己的寄存器和数据
```

这解释了为什么 CUDA 性能经常围绕 warp 展开：

```text
global memory coalescing 看 warp 内地址是否连续
shared memory bank conflict 看 warp 内 bank 访问是否冲突
warp divergence 看 warp 内线程是否走不同分支
```

## Volta 之后的 independent thread scheduling

现代 NVIDIA GPU 引入 independent thread scheduling。它不意味着 warp 不重要，而是不能过度依赖早期那种严格 lockstep 假设。

因此：

```text
warp 仍是重要执行单位
但 warp 内线程并不总能被当作每个时刻都严格同步
```

如果 warp 内线程通过 shared memory 通信，并且正确性依赖先写后读，应使用：

```cpp
__syncwarp(mask);
```

而不是隐式假设 warp 内一定同步。

## 和分支发散的关系

如果 warp 内线程走相同控制流，执行效率高：

```text
32 个 lane 都走 true
或 32 个 lane 都走 false
```

如果 warp 内一部分线程走 true，一部分走 false，就会发生 warp divergence。不同路径通常被 mask 后分批执行。

这个主题在 [Warp Divergence](warp_divergence.md) 中单独记录。

## 总结

```text
warp:
  通常 32 个线程，是 GPU 调度和执行的重要单位

SIMT:
  warp 内线程执行同一条指令，但每个线程有自己的数据

block:
  可以包含多个 warp，是 shared memory 和 __syncthreads() 的作用范围
```

一句话：

```text
理解 warp 和 SIMT，是理解 CUDA 访存、分支和同步性能的基础。
```
