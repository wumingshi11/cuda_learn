# Host-Device Transfer

## 核心原则

Host-Device Transfer 指 CPU 内存和 GPU 显存之间的数据搬运。

这一节的核心很简单：

```text
能不拷就不拷，必须拷就批量拷。
```

很多 CUDA 程序的瓶颈不在 kernel，而在 CPU 和 GPU 之间频繁传数据。GPU 内部计算和显存访问很快，但 Host-Device 传输通常要经过 PCIe / NVLink 等互连，成本比 GPU 内部访存高得多。

## 最简单的 cudaMemcpy

常见流程：

```cpp
cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

kernel<<<grid, block>>>(d_input, d_output);

cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
```

这里有两次跨 CPU/GPU 边界的数据传输：

```text
Host -> Device
Device -> Host
```

`cudaMemcpy` 的第四个参数表示传输方向：

```cpp
cudaMemcpyHostToDevice   // CPU 内存 -> GPU 显存
cudaMemcpyDeviceToHost   // GPU 显存 -> CPU 内存
cudaMemcpyDeviceToDevice // GPU 显存 -> GPU 显存
cudaMemcpyHostToHost     // CPU 内存 -> CPU 内存
```

## 同步拷贝的含义

普通 `cudaMemcpy` 是同步拷贝。可以先粗略理解为：

```text
host 调用 cudaMemcpy 后，需要等这次拷贝完成，才继续执行后面的 host 代码。
```

如果是：

```cpp
cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_input, d_output);
```

那么 kernel 启动前，`d_input` 中的数据已经准备好。

如果是：

```cpp
kernel<<<grid, block>>>(d_input, d_output);
cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
```

那么 `cudaMemcpyDeviceToHost` 会等待前面的相关 device 工作完成，再把结果拷回 host。

因此同步拷贝很直观，但容易让 CPU 等 GPU，或者让 GPU 等 CPU，频繁使用会破坏并行度。

## pageable host memory 的基本路径

普通 `malloc` / `new` 得到的 CPU 内存通常是 pageable memory：

```cpp
float *h_input = (float *)malloc(bytes);
```

pageable memory 可以被操作系统换页移动。GPU DMA / copy engine 做传输时需要稳定的物理页映射，因此从 pageable host memory 到 device memory 的传输中，runtime 通常需要额外的 staging：

```text
pageable host memory
-> CPU 拷贝到临时 pinned buffer
-> GPU DMA / copy engine 拷到 device memory
```

反向拷贝也是类似路径：

```text
GPU device memory
-> GPU DMA / copy engine 拷到临时 pinned buffer
-> CPU 拷贝到 pageable host memory
```

所以普通 host memory 虽然可以直接传给 `cudaMemcpy`，但底层真正适合 DMA 的仍然是锁页内存。

这个机制解释了为什么 pinned memory 可以进一步优化传输。但 pinned memory 属于下一节，这里只需要记住：

```text
普通 host memory 可以用 cudaMemcpy，但不是最高效的传输源/目标。
```

## 为什么频繁小拷贝很差

每次 Host-Device 传输都有固定开销。大量小拷贝会反复支付启动和调度成本：

```text
小拷贝 -> kernel -> 小拷贝 -> kernel -> 小拷贝
```

通常不如：

```text
一次批量拷贝 -> 多个 kernel 在 GPU 上连续处理 -> 最后一次拷回
```

原则：

```text
少量大拷贝 > 大量小拷贝
```

## 数据尽量留在 GPU 上

一个常见低效模式是每个 kernel 后都把中间结果拷回 CPU：

```text
H2D input
kernel A
D2H temp
H2D temp
kernel B
D2H result
```

如果 CPU 不需要读取中间结果，更好的方式是：

```text
H2D input
kernel A
kernel B
D2H result
```

也就是：

```text
中间结果留在 GPU 上
只在必要时跨 Host-Device 边界传输
```

## 1.6 的边界

这一节只讲最基础的传输原则：

```text
cudaMemcpy 基本用法
传输方向
同步拷贝
减少拷贝次数
批量传输
数据留在 GPU 上
```

更复杂的内容放到后续章节：

```text
Pinned Memory -> 为什么 host 内存类型影响传输
不同平台下的 CPU/GPU 内存同步 -> 离散 GPU、Unified Memory、mapped memory、集成 GPU
Stream 机制 -> 异步任务队列和依赖
Overlap Copy and Compute -> 用 stream 把拷贝和计算流水线化
```

总结：

```text
Host-Device Transfer 的基础优化就是减少跨 CPU/GPU 边界的数据移动。
```
