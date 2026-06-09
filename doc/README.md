# CUDA 学习索引

这个目录用于记录 CUDA 学习笔记。索引按主题分成三个章节：

```text
1. 内存优化
2. 同步机制
3. 执行抽象
```

建议先学习执行抽象中的基本概念，再看内存优化和同步机制。结合当前代码 `src/matrix_trans.cu`，可以先按“当前重点路线”学习。

## 1. 内存优化

内存优化关注的是：

```text
数据从哪里来，访问是否连续，是否能复用，是否触发硬件访问冲突。
```

### 1.1 Global Memory Coalescing

核心问题：

```text
一个 warp 内线程访问 global memory 时，地址是否连续、对齐、可合并。
```

需要掌握：

- 连续访问和跨步访问的区别。
- memory transaction 为什么会影响吞吐。
- 矩阵转置中为什么直接写回 global memory 会变成跨步访问。

已有文档：

- [Matrix Transpose and Shared Memory](matrix_transpose_shared_memory.md)

### 1.2 Shared Memory Tiling

核心问题：

```text
用 shared memory 做 tile 中转，把低效的 global memory 访问变成高效访问。
```

需要掌握：

- 为什么矩阵转置要先写 shared memory。
- shared memory 只读一次为什么也可能有收益。
- tile 如何改变 global memory 的读写访问形态。

已有文档：

- [Matrix Transpose and Shared Memory](matrix_transpose_shared_memory.md)

### 1.3 Shared Memory Bank Conflict

核心问题：

```text
在一个 warp 执行同一条 shared memory load/store 指令时，一个 bank 通常一次只能服务一个不同地址。
```

需要掌握：

- bank conflict 的硬件根因。
- `tile[32][32]` 为什么按列访问会冲突。
- `tile[32][33]` 的 padding 为什么能避免冲突。
- broadcast 和 bank conflict 的区别。

已有文档：

- [Shared Memory Bank Conflict](shared_memory_bank_conflict.md)

### 1.4 Register 和 Local Memory

核心问题：

```text
寄存器最快，但寄存器使用过多会降低 occupancy；local memory 名字像本地内存，实际可能落在 global memory。
```

需要掌握：

- 每线程寄存器数量和 occupancy 的关系。
- 寄存器溢出为什么会访问 local memory。
- 大的线程局部数组为什么危险。
- 如何通过编译输出或 profiler 观察寄存器和 local memory 使用。

已有文档：

- [Register, Local Memory, L1/L2 and Shared Memory](register_local_memory.md)

### 1.5 Constant Memory 和 Read-Only Cache

核心问题：

```text
只读数据如果访问模式合适，可以利用缓存或 broadcast 提高效率。
```

需要掌握：

- constant memory 适合 warp 内线程读取同一个地址。
- warp 内线程读取不同地址时，constant memory 可能不理想。
- read-only cache / texture 适合哪些只读访问模式。

已有文档：

- [Constant Memory and Global Memory](constant_memory.md)

### 1.6 Host-Device Transfer

核心问题：

```text
理解 CPU 内存和 GPU 显存之间最基本的数据传输方式，以及为什么要减少传输次数。
```

需要掌握：

- 最简单的 `cudaMemcpy` 使用方式。
- `cudaMemcpyHostToDevice`、`cudaMemcpyDeviceToHost`、`cudaMemcpyDeviceToDevice` 的区别。
- 同步拷贝对 host 和 device 执行顺序的影响。
- pageable host memory 到 device memory 的基本传输路径。
- 减少 `cudaMemcpy` 次数。
- 批量传输比频繁小传输更高效。
- 数据尽量留在 GPU 上，避免中间结果频繁拷回 CPU。

已有文档：

- [Host-Device Transfer](host_device_transfer.md)

### 1.7 Pinned Memory

核心问题：

```text
CPU 侧 host memory 是否 page-locked，会影响 Host-Device 传输效率和异步拷贝能力。
```

需要掌握：

- pageable memory 和 pinned memory 的区别。
- 为什么 pinned memory 传输通常更快。
- 为什么 `cudaMemcpyAsync` 通常需要 pinned memory 才能真正异步。
- `cudaMallocHost()` 和 `cudaHostAlloc()` 的基本用法。
- pinned memory 的缺点：锁页内存不能滥用，分配/释放成本也更高。

已有文档：

- [Pinned Memory](pinned_memory.md)

### 1.8 不同平台下的 CPU/GPU 内存同步

核心问题：

```text
CPU 和 GPU 是否共享物理内存、是否有统一地址空间、是否由 CUDA runtime 自动迁移，会决定数据同步方式和性能模型。
```

需要掌握：

- 离散 GPU 中 CPU RAM 和 GPU VRAM 通常是物理分离的，需要显式拷贝。
- PCIe / NVLink 等互连带宽和延迟如何影响 Host-Device transfer。
- `cudaMallocManaged()` 的 Unified Memory 如何通过 page migration、page fault、prefetch 管理数据位置。
- `cudaMemPrefetchAsync()` 和 `cudaMemAdvise()` 在 managed memory 中的作用。
- mapped pinned memory 如何让 GPU 直接访问 host memory，以及为什么它不等于访问 GPU 显存。
- 集成 GPU / SoC 中 CPU 和 GPU 可能共享 DRAM，但仍要理解 cache coherence、同步点和带宽竞争。
- “统一内存”或“共享物理内存”不等于没有同步成本。

已有文档：

- [CPU/GPU Memory Model and Synchronization](cpu_gpu_memory_model.md)

## 2. 同步机制

同步机制关注的是：

```text
哪些执行单元需要等待，等待范围有多大，同步是否会降低并行度。
```

同步需要按层级学习，不同层级的同步成本和作用范围不同。

### 2.1 Warp 级同步

核心问题：

```text
同步同一个 warp 内的线程。
```

典型接口：

```cpp
__syncwarp();
```

需要掌握：

- `__syncwarp()` 的作用范围。
- warp 内通信和 shuffle 为什么需要理解 warp 级同步。
- warp 级同步不能替代 block 级同步。

已有文档：

- [Warp-Level Synchronization](warp_level_synchronization.md)

### 2.2 Block 级同步

核心问题：

```text
同步同一个 block 内的所有线程。
```

典型接口：

```cpp
__syncthreads();
```

需要掌握：

- shared memory 写后读为什么通常需要 `__syncthreads()`。
- 同一个 block 内所有线程都必须能到达同步点。
- 分支中错误使用 `__syncthreads()` 为什么可能导致死锁或未定义行为。

已有文档：

- [Block-Level Synchronization](block_level_synchronization.md)

### 2.3 Kernel / Grid 级同步

核心问题：

```text
普通 kernel 内通常不能直接同步整个 grid，kernel 边界常被用作全局同步点。
```

需要掌握：

- 同一个 stream 中 kernel launch 的顺序语义。
- 为什么一个 kernel 结束可以作为下一个 kernel 的同步边界。
- cooperative groups 的 grid 级同步适用条件。

已有文档：

- [Kernel / Grid-Level Synchronization](kernel_grid_synchronization.md)

### 2.4 Stream 级同步和异步执行

核心问题：

```text
stream 是 CUDA 的任务队列，用来表达 kernel、memcpy 和 event 之间的执行顺序与依赖关系。
```

典型接口：

```cpp
cudaStreamCreate(&stream);
cudaStreamSynchronize(stream);
cudaEventRecord(event, stream);
cudaStreamWaitEvent(stream, event);
```

需要掌握：

- default stream 和 non-default stream 的区别。
- stream 内任务为什么有顺序。
- 不同 stream 中的任务什么时候可能并发。
- event 如何建立不同 stream 之间的依赖。
- `cudaMemcpyAsync` 如何挂到指定 stream 上。

已有文档：

- [CUDA Stream Synchronization and Async Execution](cuda_stream.md)

### 2.5 Overlap Copy and Compute

核心问题：

```text
把数据传输和 kernel 计算流水线化，让 copy engine 和 compute engine 尽量同时工作。
```

需要掌握：

- overlap copy and compute 依赖 stream 机制实现。
- pinned memory、`cudaMemcpyAsync`、multiple streams 之间的关系。
- chunk 化处理数据，构建 H2D -> kernel -> D2H pipeline。
- 硬件需要支持 copy/compute overlap。
- 不必要同步会破坏 overlap。

典型结构：

```text
stream 0: H2D chunk 0 -> kernel chunk 0 -> D2H chunk 0
stream 1: H2D chunk 1 -> kernel chunk 1 -> D2H chunk 1
stream 2: H2D chunk 2 -> kernel chunk 2 -> D2H chunk 2
```

已有文档：

- [Overlap Copy and Compute](overlap_copy_compute.md)

### 2.6 Device / Host 级同步

核心问题：

```text
host 等待 device 上的任务完成，通常成本较高。
```

典型接口：

```cpp
cudaDeviceSynchronize();
```

需要掌握：

- `cudaDeviceSynchronize()` 为什么是较重的同步。
- 同步版 `cudaMemcpy()` 对 host/device 执行顺序的影响。
- 为什么不要在循环中频繁做 device 级同步。

已有文档：

- [Device / Host Synchronization](device_host_synchronization.md)

### 2.7 Atomic、Fence 和 Barrier 的区别

核心问题：

```text
atomic、fence、barrier 分别解决并发更新、内存可见性和线程等待，不能互相混用。
```

需要掌握：

- atomic 保护某个地址的并发读改写，但不是 barrier。
- fence 约束当前线程前后内存写入的顺序和可见性，但不等待其他线程。
- barrier 让一组线程互相等待，例如 `__syncwarp()` 和 `__syncthreads()`。
- fence 的典型用法是先写 data，再 fence，再写 flag / counter。
- atomic 不应被简单理解为 fence。

已有文档：

- [Atomic, Fence and Barrier](atomic_fence_barrier.md)

## 3. 执行抽象

执行抽象关注的是：

```text
CUDA 如何组织线程，硬件如何调度这些线程，控制流如何影响执行效率。
```

### 3.1 Thread / Block / Grid

核心问题：

```text
CUDA kernel 的线程层级如何映射到计算任务。
```

需要掌握：

- `threadIdx`、`blockIdx`、`blockDim`、`gridDim` 的含义。
- 一维、二维、三维 grid/block 如何映射到数组和矩阵。
- 为什么矩阵转置中使用二维 block/grid 更自然。

状态：

```text
待补充文档
```

### 3.2 Warp 和 SIMT

核心问题：

```text
GPU 通常以 warp 为调度单位，而不是每个线程一个独立调度器。
```

需要掌握：

- 一个 warp 通常是 32 个线程。
- 一个 block 可以包含多个 warp。
- SIMT 的含义：Single Instruction, Multiple Threads。
- warp 调度和线程私有数据之间的关系。

已有文档：

- [Warp Divergence](warp_divergence.md)

### 3.3 Warp Divergence

核心问题：

```text
同一个 warp 内线程走不同 if/switch 路径时，会被 mask 后分批执行。
```

需要掌握：

- `if` / `switch` 如何导致分支发散。
- mask 和 reconverge 的基本过程。
- 边界检查为什么通常只影响边缘 block。
- 如何尽量让同一个 warp 内线程走相同控制流。

已有文档：

- [Warp Divergence](warp_divergence.md)

### 3.4 Occupancy

核心问题：

```text
一个 SM 上能同时驻留多少 warp/block，会影响延迟隐藏能力。
```

需要掌握：

- occupancy 和延迟隐藏的关系。
- block size、shared memory 使用量、寄存器使用量如何限制 occupancy。
- occupancy 高不一定代表性能最好。

状态：

```text
待补充文档
```

## 当前重点路线

结合 `src/matrix_trans.cu`，建议先按这条路线学习：

```text
Thread / Block / Grid
-> Warp 和 SIMT
-> Global Memory Coalescing
-> Shared Memory Tiling
-> Shared Memory Bank Conflict
-> Block 级同步
-> Warp Divergence
-> Host-Device Transfer
-> Pinned Memory
-> Stream 级同步和异步执行
-> Overlap Copy and Compute
```

这条路线能解释当前矩阵转置代码中最重要的设计点。
