# Overlap Copy and Compute

## 核心问题

Overlap copy and compute 的目标是：

```text
让数据传输和 kernel 计算尽量同时发生。
```

它不是单靠某一个 API 实现的，而是多个机制配合：

```text
pinned memory
cudaMemcpyAsync
non-default streams
copy engine / compute engine 并发能力
event / stream dependency
```

`chunk` 不是底层机制，而是常见的工程组织方式。

## 底层机制

### 1. Pinned memory

Host-Device 异步传输通常需要 pinned host memory：

```cpp
cudaMallocHost(&h_buf, bytes);
```

原因是 GPU DMA / copy engine 需要稳定的 host physical pages。pageable memory 可能需要内部 staging，可能阻塞 host 或破坏异步效果。

### 2. cudaMemcpyAsync

同步拷贝：

```cpp
cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
```

会让 host 等拷贝完成。

异步拷贝：

```cpp
cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream);
```

会把 copy 任务提交到指定 stream，让 host 可以继续提交后续工作。

### 3. Stream

同一个 stream 内任务有顺序：

```text
H2D -> kernel -> D2H
```

不同 stream 之间没有显式依赖时，任务可能并发：

```text
stream 0 做 compute
stream 1 做 H2D copy
stream 2 做 D2H copy
```

是否真的并发取决于硬件资源和任务依赖。

### 4. Copy engine / compute engine

overlap 需要硬件支持拷贝和计算同时进行。

可以粗略理解为：

```text
copy engine   -> 负责 H2D / D2H 数据传输
compute engine -> 负责 kernel 计算
```

如果硬件支持 copy/compute overlap，并且任务之间没有不必要依赖，就可能实现：

```text
拷贝下一块数据的同时，计算当前数据。
```

## chunk 是工程方法

chunk 化处理数据指：

```text
把大数据切成多个片段，每个片段单独执行 H2D -> kernel -> D2H。
```

例如：

```text
total data = 10 GB
chunk size = 1 GB
chunk 0: 0 ~ 1 GB
chunk 1: 1 ~ 2 GB
...
```

最简单的串行流程：

```text
H2D chunk 0 -> kernel chunk 0 -> D2H chunk 0
H2D chunk 1 -> kernel chunk 1 -> D2H chunk 1
H2D chunk 2 -> kernel chunk 2 -> D2H chunk 2
```

使用多个 stream 后，可以组织成 pipeline：

```text
stream 0: H2D chunk 0 -> kernel chunk 0 -> D2H chunk 0
stream 1: H2D chunk 1 -> kernel chunk 1 -> D2H chunk 1
stream 2: H2D chunk 2 -> kernel chunk 2 -> D2H chunk 2
```

理想情况下：

```text
GPU 正在计算 chunk 0
同时 H2D 拷贝 chunk 1
同时 D2H 拷回 chunk -1
```

所以：

```text
chunk 是实现 pipeline 的常见组织方式，
不是 overlap 的底层原因。
```

## 需要手动管理吗

通常需要手动管理。

CUDA 不会自动帮你把大数组切成 chunk，也不会自动安排：

```text
H2D -> kernel -> D2H
```

程序员通常需要管理：

```text
每个 chunk 的 offset
每个 chunk 的大小
H2D 拷贝哪一段
kernel 处理哪一段
D2H 拷回哪一段
使用哪个 stream
device buffer 何时可以复用
chunk 之间是否需要 halo / overlap 区域
```

简单伪代码：

```cpp
for (int i = 0; i < num_chunks; ++i) {
    int s = i % num_streams;
    size_t offset = i * chunk_size;
    size_t n = min(chunk_size, total_size - offset);

    cudaMemcpyAsync(d_in[s], h_in + offset, n,
                    cudaMemcpyHostToDevice, streams[s]);

    kernel<<<grid, block, 0, streams[s]>>>(d_in[s], d_out[s], n);

    cudaMemcpyAsync(h_out + offset, d_out[s], n,
                    cudaMemcpyDeviceToHost, streams[s]);
}
```

如果复用 device buffer，需要确保前一个使用该 buffer 的 chunk 已经完成。常见方式是：

```text
double buffering
ring buffering
event 标记 buffer 可复用时机
```

## 什么时候有收益

适合：

```text
数据量大
单个 chunk 的 copy 和 compute 时间都足够长
H2D / compute / D2H 可以相互重叠
使用 pinned memory
使用 cudaMemcpyAsync
使用多个 non-default streams
硬件支持 copy/compute overlap
```

不适合或收益不明显：

```text
数据很小
kernel 太短
chunk 太小，调度开销占主导
host memory 是 pageable
使用同步 cudaMemcpy
频繁 cudaDeviceSynchronize
算法 chunk 之间依赖强
```

## 常见破坏 overlap 的因素

```text
使用同步 cudaMemcpy
host memory 不是 pinned
所有任务放在同一个 stream 且无法并行
legacy default stream 造成隐式同步
cudaDeviceSynchronize 放在循环中
chunk buffer 复用没有正确安排依赖
kernel 占满 GPU，导致其他 kernel 无法并发
硬件 copy engine 数量或能力不足
```

## 和 managed memory 的区别

`cudaMallocManaged()` 可能按页面自动迁移数据，看起来像自动分块，但它不是高性能 chunk pipeline。

Managed memory 是：

```text
runtime 按页面管理数据位置
```

手动 chunk pipeline 是：

```text
程序员按算法边界切分数据
明确安排 H2D / kernel / D2H
用 stream 控制并发和依赖
```

如果追求稳定高性能，手动 chunk + pinned memory + async copy + streams 通常更可控。

## 总结

底层机制：

```text
pinned memory
cudaMemcpyAsync
stream
copy engine / compute engine
event dependency
```

工程组织：

```text
chunk 化数据
double buffering / ring buffering
multi-stream pipeline
```

一句话：

```text
chunk 是实现 overlap 的常见方法，不是 overlap 的底层机制。
```
