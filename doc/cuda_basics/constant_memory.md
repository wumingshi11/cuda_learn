# Constant Memory and Global Memory

## 核心问题

constant memory 是 device 端只读、host 端可设置的一块常量地址空间。它适合小型、只读、warp 内访问地址高度一致的数据。

但要先记住：

```text
如果 warp 内线程总是读同一个 global memory 地址，global memory 理论上也可以很快。
```

现代 GPU 通常不会把同一个 warp 内 32 个线程读取同一个 global 地址，简单处理成 32 次完全独立的 DRAM 访问。这类 uniform load 可能被 cache 或硬件路径优化。

所以 constant memory 不是因为 “global memory 同地址读取一定很慢” 才存在，而是因为它提供了一个更明确、更专门的只读 broadcast-friendly 路径。

## constant memory 和 global memory 的区别

可以这样理解：

```text
constant memory -> 专门优化小型只读数据的 warp-uniform 访问
global memory   -> 更通用，主要优化连续、对齐、可合并访问
```

global memory 的典型高效模式是：

```cpp
float v = data[base + threadIdx.x];
```

同一个 warp 内线程访问连续地址，硬件可以 coalescing，合并成较少的 memory transaction。

constant memory 的典型高效模式是：

```cpp
float k = coeff[0];
```

同一个 warp 内线程读取同一个 constant 地址，constant cache 可以 broadcast。

总结：

```text
warp 内读取连续地址 -> global memory 很合适
warp 内读取同一个小型只读地址 -> constant memory 可以考虑
```

## 使用 constant memory 的优势

### 1. 对同地址读取有专门 broadcast 机制

如果一个 warp 内多个线程读取同一个 constant 地址：

```text
thread 0  -> coeff[0]
thread 1  -> coeff[0]
...
thread 31 -> coeff[0]
```

constant cache 可以读取一次，然后把值广播给 warp 内线程。

### 2. 语义明确

`__constant__` 明确表示这块数据对 device kernel 来说是只读的：

```cpp
__constant__ float d_filter[9];
```

这适合表达：

```text
卷积核系数
小型参数表
配置常量
所有线程共享的只读标量
```

### 3. 可以减少普通 global/cache 路径压力

小型只读参数如果频繁被所有线程访问，放到 constant memory 可以让它走专门路径，减少对普通 global load 路径的竞争。

## 使用 constant memory 的劣势

### 1. 容量有限

constant memory 容量有限，常见是 64KB 级别。它不适合存放大型数组或大型模型权重。

例如 CNN 权重可能很快超过这个大小：

```text
64 * 64 * 3 * 3 = 36864 个 float ≈ 144 KB
```

这种规模已经不适合全部放入 constant memory。

### 2. warp 内读取不同地址时可能变慢

constant memory 最适合 warp 内线程读取同一个地址。如果 warp 内线程读取不同 constant 地址：

```cpp
float k = coeff[threadIdx.x];
```

就可能失去 broadcast 优势，甚至发生串行化，性能不一定好。

### 3. 需要显式声明和拷贝

constant memory 需要 device 端声明符号，并由 host 端拷贝：

```cpp
__constant__ float d_filter[9];
```

host 端设置：

```cpp
float h_filter[9] = { ... };
cudaMemcpyToSymbol(d_filter, h_filter, sizeof(h_filter));
```

kernel 中读取：

```cpp
__global__ void kernel(...) {
    float k = d_filter[0];
}
```

也可以在不同 kernel launch 前更新：

```cpp
cudaMemcpyToSymbol(d_filter, h_filter1, sizeof(h_filter1));
kernel1<<<...>>>();

cudaMemcpyToSymbol(d_filter, h_filter2, sizeof(h_filter2));
kernel2<<<...>>>();
```

更新时要注意 stream 顺序，不能在 kernel 正在读取时随意覆盖。异步版本可以和 stream 配合：

```cpp
cudaMemcpyToSymbolAsync(
    d_filter,
    h_filter,
    sizeof(h_filter),
    0,
    cudaMemcpyHostToDevice,
    stream
);
```

## 卷积会不会用 constant memory

会，但主要是小卷积核。

传统 2D 卷积核通常很小：

```text
3x3 -> 9 个系数
5x5 -> 25 个系数
7x7 -> 49 个系数
```

这些系数小、只读，而且所有线程会反复读取同一组系数：

```cpp
sum += input[...] * filter[ky][kx];
```

如果一个 warp 内线程在同一轮循环中都读取同一个 `filter[ky][kx]`，constant memory 很合适。

但大型 CNN 权重通常不适合全部放入 constant memory，因为参数量可能很大：

```text
out_channels * in_channels * kernel_h * kernel_w
```

而且如果 warp 内不同线程读取不同输出通道的不同 filter：

```text
thread 0 -> filter[out0][...]
thread 1 -> filter[out1][...]
thread 2 -> filter[out2][...]
```

访问地址分散，constant memory 可能失去优势。

## 选择建议

优先考虑 constant memory 的情况：

```text
数据小
只读
host 端可在 kernel 前设置
warp 内线程经常读取同一个地址
所有线程共享同一组参数
```

优先使用 global memory 的情况：

```text
数据较大
warp 内线程读取连续地址
访问模式更通用
需要普通数组式读写
不确定 constant memory 是否能带来收益
```

一句话：

```text
小而共享、warp-uniform 的只读数据 -> constant memory 可以考虑
大型、连续访问或分散访问的数据 -> global memory 更通用
```

最终是否值得使用 constant memory，应该结合 benchmark 或 profiler 判断。
