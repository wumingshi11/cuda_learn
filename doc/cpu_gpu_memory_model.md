# CPU/GPU Memory Model and Synchronization

## 核心问题

CPU 和 GPU 的内存同步方式取决于底层平台：

```text
CPU 和 GPU 是否共享物理内存
是否有统一虚拟地址
是否由 CUDA runtime 自动迁移页面
互连是 PCIe 还是 NVLink
cache coherence 能力如何
```

不要把这些概念混成一句“CPU/GPU 共享内存”。不同平台的性能模型差异很大。

最重要的结论：

```text
统一地址或共享物理内存，不等于没有同步成本。
```

## 离散 GPU：CPU RAM 和 GPU VRAM 分离

常见 PC / 服务器独显模型：

```text
CPU memory -> system RAM
GPU memory -> device VRAM / HBM / GDDR
连接       -> PCIe / NVLink
```

典型代码：

```cpp
float *h = (float *)malloc(bytes);
float *d = nullptr;
cudaMalloc(&d, bytes);

cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d);
cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
```

这个模型下，CPU RAM 和 GPU VRAM 通常是物理分离的。CPU 和 GPU 之间传数据需要显式拷贝，或者使用 managed memory / mapped memory 等机制让 runtime 或 driver 处理。

性能重点：

```text
减少 Host-Device 拷贝
批量传输
中间结果留在 GPU
必要时用 pinned memory + async copy + stream overlap
```

## PCIe / NVLink 如何影响传输

PCIe / NVLink 是 CPU 和 GPU 之间的路。Host-Device transfer 的上限由互连带宽和延迟决定。

传输时间可以粗略理解为：

```text
time ≈ 固定开销 + bytes / 有效带宽
```

带宽影响大块传输：

```text
PCIe 3.0 x16: 约 12-16 GB/s
PCIe 4.0 x16: 约 24-32 GB/s
PCIe 5.0 x16: 约 48-64 GB/s
NVLink: 视代际和链路数量，可能明显高于 PCIe
GPU 本地显存: 数百 GB/s 到数 TB/s
```

延迟和固定开销影响小拷贝：

```text
大量小拷贝 -> 反复支付固定开销和链路延迟
少量大拷贝 -> 更容易接近有效带宽
```

即使 NVLink 比 PCIe 快，也不意味着 CPU 内存等价于 GPU 本地显存。GPU 本地 HBM/GDDR 仍然通常更适合高吞吐计算。

## Managed Memory / Unified Memory

Managed memory 使用统一虚拟地址：

```cpp
float *p = nullptr;
cudaMallocManaged(&p, bytes);
```

CPU 和 GPU 都能使用同一个指针 `p`：

```cpp
p[i] = 1.0f;          // CPU 写
kernel<<<...>>>(p);   // GPU 用同一个 p
cudaDeviceSynchronize();
float x = p[i];       // CPU 读
```

底层可能发生：

```text
page fault
page migration
访问权限切换
cache / TLB 维护
```

所以 `cudaMallocManaged()` 不是零成本。用不好时，CPU/GPU 频繁交替访问同一批数据，可能触发很多小粒度页面迁移，性能很差。

更好的模式是：

```text
CPU 初始化
prefetch 到 GPU
GPU 连续处理
必要时 prefetch 回 CPU 或同步后读取
```

## cudaMemPrefetchAsync 如何决定方向

`cudaMemPrefetchAsync()` 不用 `cudaMemcpyHostToDevice` 这种方向枚举。它通过目标位置决定迁移方向：

```cpp
cudaMemPrefetchAsync(p, bytes, device_id, stream);
```

含义：

```text
请把 p 这段 managed memory 预取到 device_id 对应的 GPU 附近。
```

如果当前页面在 CPU RAM，逻辑方向就是：

```text
CPU -> GPU
```

如果当前页面在另一个 GPU，可能是：

```text
GPU A -> GPU B
```

如果要预取回 CPU：

```cpp
cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId, stream);
```

逻辑方向：

```text
GPU -> CPU
```

因此 prefetch 的方向由：

```text
当前页面驻留位置 -> dstDevice 指定的目标位置
```

共同决定。

## cudaMemAdvise 的作用

`cudaMemAdvise()` 不直接搬数据。它给 CUDA runtime / driver 提供访问模式提示：

```cpp
cudaMemAdvise(p, bytes, advice, device);
```

常见 advice：

```cpp
cudaMemAdviseSetReadMostly
cudaMemAdviseSetPreferredLocation
cudaMemAdviseSetAccessedBy
```

`cudaMemAdviseSetReadMostly` 表示主要是读：

```text
适合模型参数、查表数据、只读配置
可能帮助 runtime 减少不必要迁移或允许只读副本
```

`cudaMemAdviseSetPreferredLocation` 表示偏好驻留位置：

```text
这段数据长期更希望放在某个 device 或 CPU 附近
```

它不是立即迁移。要立即迁移，用 `cudaMemPrefetchAsync()`。

`cudaMemAdviseSetAccessedBy` 表示某个 device 会访问这段内存：

```text
runtime 可以提前建立映射，减少访问时 page fault 或映射开销
```

总结：

```text
cudaMemPrefetchAsync -> 主动迁移数据
cudaMemAdvise        -> 提供访问模式和位置偏好
```

## Managed Memory 和显式拷贝的关系

`cudaMemPrefetchAsync()` 和显式 `cudaMemcpyAsync()` 的性能意图类似：

```text
提前把数据放到接下来要访问的处理器附近。
```

但两者不是完全等价：

```text
cudaMemcpyAsync:
  host pointer 和 device pointer 通常是两份 allocation
  程序员明确控制源、目标、方向、大小

cudaMemPrefetchAsync:
  一个 managed pointer
  runtime 管理页面驻留位置
  程序员给出目标位置
```

Managed memory 代码更简洁，但控制权更少。高性能稳定路径通常更偏向显式拷贝；原型开发、访问模式复杂或希望减少手写拷贝时，可以考虑 managed memory，并配合 prefetch / advise。

## mapped pinned memory：GPU 直接访问 CPU 内存

mapped pinned memory 允许 GPU kernel 直接访问一段 host memory：

```cpp
float *h_ptr = nullptr;
cudaHostAlloc(&h_ptr, bytes, cudaHostAllocMapped);

float *d_ptr = nullptr;
cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);

kernel<<<grid, block>>>(d_ptr);
```

这里：

```text
h_ptr -> CPU 侧 host pointer
d_ptr -> GPU kernel 可使用的 device pointer
```

底层访问类似：

```text
GPU -> PCIe/NVLink -> CPU memory
```

优点：

```text
不需要显式 cudaMemcpy
适合少量、低频、流式访问
可以避免一次完整拷贝
```

缺点：

```text
延迟高
带宽通常低于 GPU 本地显存
访问模式差时性能很差
不适合大量随机访问或高复用计算
```

结论：

```text
GPU 可以直接访问 CPU 内存，但这不是等价扩展显存。
```

## mapped pinned memory 和 managed memory 的区别

它们看起来都像 “CPU/GPU 可以用一个指针访问数据”，但底层含义不同。

mapped pinned memory 的本质是：

```text
CPU host memory 被 page-locked，然后映射到 GPU 地址空间。
```

典型情况下有两个指针：

```text
h_ptr -> CPU 用
d_ptr -> GPU 用
```

在 Unified Virtual Addressing 下，有些情况下 host pointer 和 device pointer 的数值可能相同，或者 host pointer 可以直接作为 kernel 参数使用。但概念上仍然是：

```text
GPU 在访问 CPU host memory。
```

数据不会自动迁移到 GPU 显存。

managed memory 的本质是：

```text
CUDA runtime 管理的一段统一虚拟地址内存。
```

典型情况下只有一个指针：

```cpp
float *p = nullptr;
cudaMallocManaged(&p, bytes);

p[0] = 1.0f;        // CPU 使用 p
kernel<<<...>>>(p); // GPU 也使用 p
```

物理页面可以在 CPU RAM 和 GPU VRAM 之间迁移：

```text
CPU 访问 -> 页面可能在 CPU RAM
GPU 访问 -> 页面可能迁移到 GPU VRAM
```

对比：

| 对比 | mapped pinned memory | managed memory |
| --- | --- | --- |
| 分配方式 | `cudaHostAlloc(..., cudaHostAllocMapped)` | `cudaMallocManaged()` |
| 本质 | host pinned memory 映射给 GPU | runtime 管理的统一内存 |
| 物理位置 | CPU RAM | 可在 CPU RAM / GPU VRAM 间迁移 |
| GPU 访问 | 远程访问 CPU 内存 | 可能访问已迁移到 GPU 的页面 |
| 是否自动迁移 | 否 | 是 |
| 是否可能用 GPU 显存 | 通常不迁移到 GPU 显存 | 可以迁移到 GPU 显存 |
| 典型风险 | 远程访问慢 | page fault / migration 抖动 |
| 适合场景 | 少量、流式、低复用数据 | 编程简化、访问模式可控、配合 prefetch |

一句话：

```text
mapped pinned memory 是“GPU 直接访问 CPU 内存”；
managed memory 是“CPU/GPU 用同一指针，runtime 决定页面放 CPU 还是 GPU”。
```

## GPU 显存不够时能不能用 CPU 内存

技术上可以，但不一定高效。

可选方式：

```text
mapped pinned memory:
  GPU 直接访问 host memory
  适合少量、低频、低复用数据

Unified Memory oversubscription:
  managed allocation 可能超过 GPU 显存
  runtime 在 CPU/GPU 之间迁移页面
  可能频繁 page fault / migration，性能波动大

手动分块 streaming:
  数据分 chunk
  H2D -> kernel -> D2H
  配合 pinned memory、cudaMemcpyAsync、multiple streams
  高性能路径更可控
```

一般建议：

```text
先跑通 / 原型 -> managed memory oversubscription 可以尝试
少量低频访问 CPU 数据 -> mapped pinned memory 可以考虑
高性能生产路径 -> 手动分块 + pinned memory + async copy + streams
```

## 集成 GPU / SoC：共享 DRAM 不等于 CPU 多线程

在一些集成 GPU / SoC 中，CPU 和 GPU 可能共享同一块物理 DRAM：

```text
CPU cores -> system DRAM
GPU cores -> system DRAM
```

从物理内存角度看，它有点像多个处理单元访问同一块内存。但它不能简单等同于 CPU 多线程共享内存。

差异：

```text
CPU 和 GPU 是异构执行单元
GPU 有自己的执行模型、cache、队列和同步机制
cache coherence 能力取决于平台
共享物理内存不一定等于统一虚拟地址
CPU/GPU 会竞争同一套 DRAM 带宽
同步 API 和 CPU 多线程不同
```

CPU 多线程通常依赖：

```cpp
std::mutex
std::atomic
condition_variable
```

CUDA / GPU 场景更多依赖：

```text
cudaDeviceSynchronize
cudaStreamSynchronize
cudaEvent
kernel launch ordering
memory fence / atomic
```

所以可以类比为：

```text
像共享内存，但不是同一个编程模型。
```

## 总结

不同平台的数据同步模型可以这样分层：

```text
离散 GPU:
  CPU RAM 和 GPU VRAM 分离，显式拷贝最清楚

Pinned / mapped pinned:
  锁住 host 页，支持高效 DMA 或 GPU 直接访问 host memory

Managed memory:
  统一指针，runtime 管理页面迁移

集成 GPU / SoC:
  可能共享物理 DRAM，但仍要处理 cache coherence、同步和带宽竞争
```

最终性能判断：

```text
统一地址 / 直接访问 / 共享 DRAM 都不等于零成本。
```
