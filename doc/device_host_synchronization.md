# Device / Host Synchronization

## 核心问题

Device / Host 级同步指 host CPU 等待 GPU device 上的工作完成。

典型接口：

```cpp
cudaDeviceSynchronize();
cudaStreamSynchronize(stream);
cudaEventSynchronize(event);
```

这类同步通常成本较高，因为它会让 CPU 停下来等待 GPU。

## cudaDeviceSynchronize

`cudaDeviceSynchronize()` 等待当前 host thread 已提交到 device 的相关工作完成。

典型用法：

```cpp
kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();
```

含义：

```text
host 等 GPU 上前面提交的工作完成
GPU 工作完成后，host 继续执行
```

它很直观，但比较重：

```text
CPU 被阻塞
可能等待多个 stream 的工作
破坏异步提交
破坏 copy/compute overlap
```

因此性能路径中不要在循环里频繁调用：

```cpp
for (...) {
    kernel<<<grid, block>>>(...);
    cudaDeviceSynchronize(); // 通常不推荐
}
```

更好的方式通常是：

```text
连续提交多个 kernel / memcpy
只在真正需要 host 读取结果或测量结束时同步
```

## cudaStreamSynchronize

`cudaStreamSynchronize(stream)` 只等待指定 stream 中已提交的工作完成：

```cpp
cudaStreamSynchronize(stream);
```

它比 `cudaDeviceSynchronize()` 范围更小：

```text
cudaDeviceSynchronize -> 等 device 上相关工作
cudaStreamSynchronize -> 只等某个 stream
```

如果只需要等待一个 stream，优先使用 stream 级同步，避免无意义地阻塞其他 stream 的并发。

## cudaEventSynchronize

`cudaEventSynchronize(event)` 等待某个 event 完成：

```cpp
cudaEventRecord(event, stream);
cudaEventSynchronize(event);
```

含义：

```text
host 等到 stream 执行到 event 标记点。
```

这比等待整个 device 更细粒度。

常见用途：

```text
等待某个阶段完成
GPU 时间测量
host 只等待必要依赖点
```

## 同步版 cudaMemcpy

普通 `cudaMemcpy()` 是同步拷贝。

例如 DeviceToHost：

```cpp
kernel<<<grid, block>>>(d_out);
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
```

这里 `cudaMemcpyDeviceToHost` 会等待前面的相关 device 工作完成，然后把数据拷回 host。

所以同步版 `cudaMemcpy()` 经常隐含了 host/device 同步：

```text
host 必须等拷贝完成
拷贝前相关 GPU 写入必须完成
```

这很方便，但如果在性能路径中频繁使用，会破坏异步 pipeline。

## 隐式同步风险

除了显式同步 API，一些操作也可能造成同步或阻塞：

```text
同步版 cudaMemcpy
某些 cudaMalloc / cudaFree 行为
pageable memory 参与 cudaMemcpyAsync 时的内部 staging
managed memory page fault / migration
legacy default stream 的隐式同步
```

因此调优 stream overlap 时，要检查是否有隐式同步点。

典型破坏 overlap 的写法：

```cpp
cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d);
cudaDeviceSynchronize(); // 这里会让 host 等 device，破坏后续流水线提交
```

## 调试同步

kernel launch 是异步的。很多错误不会立刻在 launch 行暴露。

调试时常见写法：

```cpp
kernel<<<grid, block>>>(...);

cudaError_t launch_err = cudaGetLastError();
if (launch_err != cudaSuccess) {
    // launch 配置或参数错误
}

cudaError_t exec_err = cudaDeviceSynchronize();
if (exec_err != cudaSuccess) {
    // kernel 执行期间的异步错误
}
```

区别：

```text
cudaGetLastError():
  检查 launch 相关错误

cudaDeviceSynchronize():
  等 kernel 执行完成，也能暴露异步执行错误
```

但 `cudaDeviceSynchronize()` 会同步 device，调试时很有用，性能路径中要谨慎使用。

## 什么时候需要 host 等 device

合理场景：

```text
host 需要读取 GPU 计算结果
程序结束前确保 GPU 工作完成
调试 kernel 错误
精确计时某段 GPU 工作
资源释放前需要确认使用完成
```

不合理或需要谨慎的场景：

```text
每个 kernel 后都 cudaDeviceSynchronize
每个 chunk 后都同步 device
用 device synchronize 代替 stream/event 依赖
为了“保险”随手加同步
```

## 总结

```text
cudaDeviceSynchronize:
  host 等整个 device 相关工作，范围大，成本高

cudaStreamSynchronize:
  host 等某个 stream，范围较小

cudaEventSynchronize:
  host 等某个 event，依赖点更细

cudaMemcpy:
  同步版拷贝，常隐含 host/device 同步
```

一句话：

```text
Device / Host 同步是让 CPU 等 GPU；需要时再用，能用 stream/event 缩小范围就不要直接同步整个 device。
```
