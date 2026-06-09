# CUDA Stream Synchronization and Async Execution

## 核心问题

CUDA stream 是 GPU 工作队列，用来表达 kernel、memcpy、event 之间的执行顺序和依赖关系。

可以先这样理解：

```text
同一个 stream 内：任务按提交顺序执行
不同 stream 之间：没有显式依赖时，任务可能并发执行
```

stream 不是线程，也不是 CPU 队列。它是 CUDA runtime/driver 向 GPU 提交工作的顺序通道。

## 创建和使用 stream

创建 stream：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
```

把 kernel 提交到指定 stream：

```cpp
kernel<<<grid, block, 0, stream>>>(...);
```

把异步拷贝提交到指定 stream：

```cpp
cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream);
```

销毁 stream：

```cpp
cudaStreamDestroy(stream);
```

## stream 内任务有顺序

同一个 stream 中提交：

```cpp
cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_input, d_output);
cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream);
```

执行顺序是：

```text
H2D copy
-> kernel
-> D2H copy
```

因此同一个 stream 内通常不需要额外 event 来保证前后任务顺序。

## 不同 stream 可能并发

不同 stream 中的任务没有天然顺序关系：

```cpp
kernelA<<<grid, block, 0, stream1>>>(...);
kernelB<<<grid, block, 0, stream2>>>(...);
```

如果硬件资源允许，`kernelA` 和 `kernelB` 可能并发执行。

是否真的并发取决于：

```text
GPU 是否支持 concurrent kernels
每个 kernel 使用的 SM / register / shared memory 资源
copy engine / compute engine 是否可并行
stream 之间是否存在 event 或隐式同步
```

所以 stream 表达的是：

```text
允许并发
```

不是保证一定并发。

## event 是 stream 中的标记点

CUDA event 可以理解为：

```text
在某个 stream 中插入一个标记点。
```

创建 event：

```cpp
cudaEvent_t event;
cudaEventCreate(&event);
```

记录 event：

```cpp
kernelA<<<grid, block, 0, stream1>>>(...);
cudaEventRecord(event, stream1);
```

含义：

```text
event 会在 stream1 中排队
只有 stream1 中 event 前面的任务完成后
event 才会变成 completed 状态
```

所以 event 可以表示：

```text
某个 stream 执行到了哪一步。
```

## 用 event 同步多个 stream

让另一个 stream 等待 event：

```cpp
kernelA<<<grid, block, 0, stream1>>>(...);

cudaEventRecord(event, stream1);

cudaStreamWaitEvent(stream2, event, 0);

kernelB<<<grid, block, 0, stream2>>>(...);
```

执行关系：

```text
stream1: kernelA -> event
stream2: wait event -> kernelB
```

也就是：

```text
kernelB 必须等 kernelA 完成后才能开始。
```

注意：这不是让 CPU 等待，而是让 `stream2` 的 GPU 工作队列等待 event。host 端可以继续提交其他工作。

## event 和 cudaStreamSynchronize 的区别

host 等 stream 完成：

```cpp
cudaStreamSynchronize(stream1);
```

含义：

```text
CPU 阻塞，等待 stream1 前面已提交的任务完成。
```

stream 等 event：

```cpp
cudaStreamWaitEvent(stream2, event, 0);
```

含义：

```text
stream2 等 event
CPU 不一定阻塞
```

因此：

```text
cudaStreamSynchronize -> host/device 同步，CPU 等待
cudaStreamWaitEvent   -> device 队列之间建立依赖，CPU 可继续提交任务
```

## event 也可以用于计时

event 常用于测 GPU 时间：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

这测的是 GPU stream 中 `start` 到 `stop` 之间的 elapsed time。

## default stream 注意点

CUDA 有 default stream，但它的语义要小心。

常见有两种模式：

```text
legacy default stream
per-thread default stream
```

legacy default stream 可能和其他 stream 产生隐式同步，导致看似用了多个 stream，但并发被破坏。

因此学习 stream 时要记住：

```text
默认 stream 语义会影响并发
不理解默认 stream 语义时，优先显式创建 non-default stream
```

## 和 cudaMemcpyAsync 的关系

`cudaMemcpyAsync()` 可以挂到指定 stream：

```cpp
cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream);
```

但要实现稳定的异步传输和 copy/compute overlap，host memory 通常需要是 pinned memory。

如果 host memory 是 pageable memory，runtime 可能需要内部 staging，异步能力可能受限。

所以常见高性能组合是：

```text
pinned memory
cudaMemcpyAsync
non-default streams
event dependency
```

## 总结

```text
stream:
  GPU 工作队列，表达任务顺序

event:
  stream 中的完成标记点

cudaStreamWaitEvent:
  让一个 stream 等另一个 stream 的标记点

cudaStreamSynchronize:
  让 CPU 等某个 stream 完成
```

一句话：

```text
stream 管任务排队，event 管队列之间的依赖点。
```
