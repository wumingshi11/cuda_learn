# Pinned Memory

## 核心问题

pinned memory 也叫 page-locked memory，指被操作系统锁定的 host memory 页面。

它解决的问题是：

```text
Host <-> Device 拷贝时，CPU 内存页能不能稳定、高效地被 GPU DMA 访问。
```

最重要的结论：

```text
pinned memory 不会被 OS swap out，也不会被 OS 随意迁移物理页。
```

这让 GPU DMA / copy engine 可以稳定地访问这些 host physical pages。

底层硬件能力可以这样理解：

```text
GPU DMA / copy engine 面向的是稳定的物理页映射
pageable host memory 的物理页可能被 OS 换出或迁移
因此 pageable host memory 不能作为长期稳定的 DMA 目标
```

## pageable memory 和 pinned memory

普通 `malloc` / `new` 得到的是 pageable memory：

```cpp
float *h = (float *)malloc(bytes);
```

pageable memory 可以被操作系统换页、迁移。GPU DMA 不能长期依赖它当前的物理地址。

pinned memory 由 CUDA API 分配：

```cpp
float *h = nullptr;
cudaMallocHost(&h, bytes);
```

或者：

```cpp
float *h = nullptr;
cudaHostAlloc(&h, bytes, cudaHostAllocDefault);
```

也可以尝试把已有 host memory 注册成 pinned：

```cpp
cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
```

但 `cudaHostRegister()` 不保证任意内存都能成功注册。它依赖 OS、driver、硬件/IOMMU 支持，也可能受到锁页内存限制影响。

## pageable memory 也能 cudaMemcpy 吗

可以。

例如：

```cpp
float *h = (float *)malloc(bytes);
cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
```

这能工作。但从 pageable host memory 传输时，CUDA runtime/driver 通常需要内部 staging。关键原因不是 CUDA 不会拷贝 pageable memory，而是底层 DMA 需要稳定的锁页物理内存。

```text
HostToDevice:
pageable host memory
-> CPU 拷贝到 CUDA 内部临时 pinned buffer
-> GPU DMA / copy engine 拷到 device memory
```

DeviceToHost 类似：

```text
DeviceToHost:
GPU DMA / copy engine 拷到 CUDA 内部临时 pinned buffer
-> CPU 拷贝到 pageable host memory
```

也就是说：

```text
用户传入的 malloc 内存通常仍是 pageable
CUDA 内部可能使用临时 pinned buffer 做中转
真正给 GPU DMA 稳定访问的是 pinned buffer，不是原始 pageable memory
```

这也是 pageable memory 传输可能更慢的原因：

```text
多一次 CPU 侧拷贝
多 staging buffer 管理开销
异步能力受限
```

如果用户直接提供 pinned memory，传输路径更接近：

```text
pinned host memory
-> GPU DMA
-> device memory
```

## pinned memory 和 cudaMemcpyAsync

`cudaMemcpyAsync()` 要真正异步，通常需要 host 侧内存是 pinned memory：

```cpp
cudaMemcpyAsync(d, h_pinned, bytes, cudaMemcpyHostToDevice, stream);
```

如果 host memory 是 pageable，runtime 可能需要先做内部 staging，这会导致：

```text
可能阻塞 host
可能退化为同步行为
难以实现稳定的 copy/compute overlap
```

因此高性能传输通常使用：

```text
pinned memory
+ cudaMemcpyAsync
+ stream
```

## 拷贝时的数据一致性

pinned memory 只保证 host physical pages 稳定，适合 DMA。它不自动解决所有 CPU/GPU 并发一致性问题。

对于 HostToDevice：

```cpp
cudaMemcpyAsync(d, h_pinned, bytes, cudaMemcpyHostToDevice, stream);
```

需要保证：

```text
异步拷贝完成前，host 不要修改 h_pinned 中作为源的数据。
```

对于 DeviceToHost：

```cpp
cudaMemcpyAsync(h_pinned, d, bytes, cudaMemcpyDeviceToHost, stream);
```

需要等拷贝完成后，host 才能读取 `h_pinned`：

```cpp
cudaStreamSynchronize(stream);
```

或者使用 event 判断完成。

总结：

```text
cudaMemcpy / cudaMemcpyAsync 会按调用或 stream 顺序保证拷贝本身的数据可见性
pinned memory 不允许 CPU/GPU 对同一内存随意并发读写
并发访问仍然需要 stream、event、synchronize 建立顺序
```

## pinned memory 和 managed memory 的区别

pinned memory 和 managed memory 不是一回事。

```text
pinned memory 解决：Host <-> Device 拷贝时，CPU 内存页能不能稳定高效地被 DMA 访问
managed memory 解决：CPU 和 GPU 能不能用同一个指针访问数据，并由 runtime 管理数据位置
```

Pinned memory：

```cpp
float *h = nullptr;
cudaMallocHost(&h, bytes);
```

特点：

```text
主要位置：CPU RAM
属性：page-locked
用途：高效 cudaMemcpy / cudaMemcpyAsync
是否自动迁移：否
是否通常还需要 cudaMemcpy：是
```

Managed memory：

```cpp
float *p = nullptr;
cudaMallocManaged(&p, bytes);
```

特点：

```text
逻辑上：CPU/GPU 都能访问同一个指针
底层：可能发生 page fault、page migration、prefetch、访问权限切换
用途：简化 CPU/GPU 共享访问
是否等于 pinned：否
```

一句话：

```text
pinned memory 是“锁住 CPU 内存页，让拷贝更高效”；
managed memory 是“给 CPU/GPU 一个统一指针，让 runtime 管理数据在哪”。
```

## GPU 直接访问 CPU 内存：mapped pinned memory

有些 pinned host memory 可以映射给 GPU 直接访问，也常被叫做 zero-copy / mapped pinned memory。

典型用法：

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

GPU kernel 可以通过 `d_ptr` 直接访问这段 host memory。

但底层访问仍然类似：

```text
GPU -> PCIe/NVLink -> CPU memory
```

所以它的优点是：

```text
不需要显式 cudaMemcpy
适合少量、低频、流式访问
可以避免一次完整拷贝
```

缺点是：

```text
延迟高
带宽通常低于 GPU 本地显存
访问模式差时性能很差
不适合大量随机访问或高复用计算
```

关键结论：

```text
GPU 能直接访问 CPU 内存，不等于 CPU 内存变成了 GPU 本地高速显存。
```

## 是否所有内存都支持 pinned

不是。

pinned memory 需要 OS、CUDA driver、硬件/IOMMU 共同支持。还可能受到系统限制：

```text
锁页内存限制
ulimit -l / RLIMIT_MEMLOCK
容器限制
driver 策略
可用物理内存
```

因此 pinned memory 不能滥用。分配太多 pinned memory 会减少系统可分页内存，影响操作系统内存管理和其他进程。

适合做法：

```text
为长期复用的传输 buffer 分配 pinned memory
避免频繁分配/释放 pinned memory
不要把大量普通业务数据都锁成 pinned
```

## 和 1.8 的关系

这一节关注的是 pinned memory 本身：

```text
page-locked host memory
DMA
cudaMemcpyAsync
mapped pinned memory
```

1.8 会进一步讨论更大的平台差异：

```text
离散 GPU 的 CPU RAM / GPU VRAM 分离
Unified Memory 的 page migration
集成 GPU / SoC 共享 DRAM
PCIe / NVLink 对同步和传输模型的影响
```
