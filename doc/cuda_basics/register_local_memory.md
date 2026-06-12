# Register, Local Memory, L1/L2 and Shared Memory

## 片上资源和访问路径

register、shared memory、L1 cache、L2 cache 都属于 GPU 片上资源，但它们不是同一种东西。

可以先按这个模型理解：

```text
register file   -> 每个 SM 上专门的寄存器资源
shared memory   -> 每个 SM 上的片上 SRAM，block 内线程共享
L1 cache        -> SM 附近的片上缓存
L2 cache        -> GPU 全局共享的片上缓存
global memory   -> 片外 device DRAM / 显存
```

register 和 shared memory 都会限制 SM 上的并发：

```text
每线程 register 用得多 -> 每个 SM 能驻留的 active warps/blocks 可能变少
每 block shared memory 用得多 -> 每个 SM 能驻留的 active blocks 可能变少
```

但 register 和 shared memory 通常不是同一个容量池：

```text
register 多了，不表示 shared memory 容量一定变少
shared memory 多了，也不表示 register 容量一定变少
```

需要单独注意的是 shared memory 和 L1 cache。在不少 NVIDIA 架构上，shared memory 和 L1 cache 可能共享或划分同一块片上 SRAM，具体比例和组织方式取决于 GPU 架构。

这不等于 shared memory 访问会经过 L1 cache。shared memory 是显式管理的片上内存，典型访问路径可以理解为：

```text
thread -> shared memory
```

global/local memory 的访问路径更像：

```text
thread -> L1 -> L2 -> global memory
```

所以：

```text
shared memory 不是被 L1/L2 cache 缓存的 global memory
shared memory 访问通常不按 global memory 那样经过 L1/L2 cache
shared memory 和 L1 在部分架构上可能共享物理 SRAM，但访问语义和路径不同
```

## shared memory / L1 比例由谁决定

当 shared memory 和 L1 cache 在某些架构上共享或划分同一块片上 SRAM 时，程序可以给出配置偏好，但不能手动管理物理 SRAM 地址。

更准确地说：

```text
shared memory 用多少：由 __shared__ 声明和 kernel launch 时的动态 shared memory 大小决定
L1/shared 的比例偏好：可以通过 CUDA runtime API 设置
实际物理 SRAM 如何划分和管理：由 GPU 架构、driver/runtime 和硬件决定
```

静态 shared memory 由代码声明：

```cpp
__shared__ float tile[32][33];
```

动态 shared memory 由 kernel launch 的第三个参数指定：

```cpp
extern __shared__ float buf[];

kernel<<<grid, block, shared_bytes>>>(...);
```

可以给某个 kernel 设置 cache 偏好：

```cpp
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferEqual);
```

含义大致是：

```text
PreferShared -> 更偏向 shared memory
PreferL1     -> 更偏向 L1 cache
PreferEqual  -> 尽量均衡
```

在较新的架构上，也可以设置 shared memory carveout 偏好：

```cpp
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared
);
```

或者设置动态 shared memory 大小上限：

```cpp
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    size
);
```

这些设置表达的是：

```text
这个 kernel 更希望片上可配置资源偏向 shared memory 或 L1 cache
```

但它们不是精确的物理内存分配，也不能指定某个地址属于 shared memory 或 L1。

一句话：

```text
可以指定“我要多少 shared memory”和“更偏向 shared 还是 L1”，但不能手动管理 shared/L1 共享 SRAM 的物理分配。
```

## 先记住：register

CUDA kernel 中普通的线程局部标量变量，编译器会优先放进寄存器：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = input[idx];
float y = x * 2.0f;
```

这些变量通常会进入 register，而不是 shared memory，也不是 local memory。

寄存器是每个线程私有的片上资源，访问速度最快。写 CUDA kernel 时，应该优先理解：

```text
普通局部变量 -> 优先 register
register 放不下或不适合 -> 可能进入 local memory
显式 __shared__ -> shared memory
```

## local memory 是逻辑概念，不是独立物理内存

local memory 的名字容易误导。它的 "local" 指的是：

```text
local to thread，也就是每个线程私有
```

它不表示物理上离线程很近，也不表示有一块独立的片上 local memory。

更准确地说：

```text
local memory 是 CUDA 的线程私有地址空间概念
物理上通常位于 device DRAM，也就是 global memory 对应的显存
访问时可能经过 L1/L2 cache
```

所以：

```text
local memory = 逻辑上线程私有
local memory != 物理上高速本地内存
```

常见进入 local memory 的情况：

- 每线程局部数组太大。
- 数组索引无法在编译期确定，编译器无法把它完全放进 register。
- 每线程寄存器不够，发生 register spill。

例如：

```cpp
float tmp[1024]; // 每个线程一个大数组，通常很危险
```

## L1/L2、register 和 shared memory 是否在片上

可以按下面方式理解：

```text
register      -> 片上，每线程私有，最快
shared memory -> 片上，每 block 共享，显式声明
L1 cache      -> 片上，靠近 SM，缓存部分访存
L2 cache      -> 片上，通常是 GPU 全局共享缓存
global memory -> 片外 device DRAM / 显存
local memory  -> 逻辑线程私有，物理通常在 global memory，可能经过 L1/L2
```

注意：不同 GPU 架构上 L1/shared memory 的组织方式可能有差异，但对写 kernel 来说，下面这个模型足够重要：

```text
register 和 shared memory 是显式可感知的片上资源
L1/L2 是硬件缓存层
global/local memory 的物理数据通常在显存中
```

## 寄存器多为什么会降低并发

每个 SM 的寄存器数量是有限的。一个 kernel 中每个线程使用的寄存器越多，一个 SM 能同时驻留的 block/warp 可能越少。

这降低的是：

```text
每个 SM 上的 active blocks / active warps
```

不是降低 GPU 的 SM 数量。SM 数量是硬件固定的。

例子：

```text
假设一个 SM 有 65536 个寄存器
block 有 256 个线程
每线程使用 32 个寄存器
```

每个 block 消耗：

```text
32 * 256 = 8192 个寄存器
```

从寄存器角度看，一个 SM 最多可放：

```text
65536 / 8192 = 8 个 block
```

如果每线程使用 64 个寄存器：

```text
64 * 256 = 16384 个寄存器 / block
65536 / 16384 = 4 个 block
```

active block/warp 变少后，GPU 隐藏 global memory 延迟的能力可能下降。

因此：

```text
寄存器过多 -> occupancy 可能下降
occupancy 下降 -> 延迟隐藏能力可能下降
```

但这不表示寄存器越少越好。强行减少寄存器可能导致 spill 到 local memory，性能反而更差。

## 可以显式控制寄存器使用吗

可以，但要谨慎。

可以通过 `nvcc` 限制每个线程最多使用多少寄存器：

```sh
nvcc --maxrregcount=64 ...
```

CMake 中可以按 target 添加：

```cmake
target_compile_options(target PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=64>
)
```

也可以用 `__launch_bounds__` 给编译器提供资源约束信息：

```cpp
__global__ __launch_bounds__(256, 2)
void kernel(...) {
}
```

含义大致是：

```text
每个 block 最多/期望 256 个线程
每个 SM 至少希望驻留 2 个 block
```

编译器会据此调整资源使用，包括寄存器分配。

查看寄存器使用可以加：

```sh
nvcc -Xptxas -v ...
```

输出中会出现类似：

```text
Used 32 registers
```

如果看到 spill 或 local memory 使用，需要重点关注。

## 局部变量多了会自动放到 shared memory 吗

一般不会。

普通局部变量是线程私有的：

```cpp
float x;
int idx;
float tmp[16];
```

编译器会优先尝试放到 register。如果不适合放 register，可能进入 local memory。

它不会自动变成 shared memory。shared memory 必须显式声明：

```cpp
__shared__ float tile[32][33];
```

或者使用动态 shared memory：

```cpp
extern __shared__ float buf[];
```

原因是 shared memory 是 block 内共享的资源，语义和普通线程局部变量完全不同。编译器不能随便把线程私有变量变成 block 共享变量。

总结：

```text
普通局部变量 -> 优先 register
寄存器不够/数组复杂 -> local memory
显式 __shared__ -> shared memory
```

## 快速判断表

| 名称 | 逻辑可见性 | 常见物理位置 | 速度/特点 |
| --- | --- | --- | --- |
| register | 线程私有 | 片上 | 最快，数量有限 |
| local memory | 线程私有 | 通常在显存，经 L1/L2 | 名字像本地，实际可能很慢 |
| shared memory | block 内共享 | 片上 | 快，需要显式声明，注意 bank conflict |
| L1 cache | SM 附近缓存 | 片上 | 硬件管理 |
| L2 cache | GPU 全局缓存 | 片上 | 硬件管理，全局共享 |
| global memory | 全局可见 | 片外显存 | 容量大，延迟高，重视 coalescing |
