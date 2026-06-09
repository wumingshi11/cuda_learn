# Warp-Level Synchronization

## 核心问题

`__syncwarp()` 同步的是同一个 warp 内指定 mask 中的线程，也就是指定 lane 集合。

常见写法：

```cpp
__syncwarp();
```

完整形式：

```cpp
__syncwarp(mask);
```

含义：

```text
mask 中指定的 lane 必须都到达这个同步点
这些 lane 到达后，再继续执行后续指令
```

`__syncwarp()` 只作用于 warp 内线程，不会同步同一个 block 内的其他 warp。

## 它同步了什么

`__syncwarp(mask)` 主要提供两个保证：

```text
1. mask 中指定的 warp lane 在这里汇合
2. 同步点之前的内存操作，对这些参与 lane 可见
```

典型用途是 warp 内线程通过 shared memory 通信：

```cpp
shared[lane] = value;

__syncwarp(mask);

value2 = shared[lane ^ 1];
```

这里的同步意图是：

```text
同一个 warp 内参与 lane 先完成 shared memory 写入
然后这些 lane 再读取 shared memory
```

## SIMT 下是否总需要 __syncwarp

不总需要。

warp 内线程通常按 SIMT 执行：

```text
Single Instruction, Multiple Threads
```

也就是一个 warp 内线程通常执行同一条指令。因此简单、无分支、无跨 lane 通信的代码不需要显式 warp 同步：

```cpp
int x = threadIdx.x;
int y = x + 1;
```

如果使用 warp shuffle API，例如：

```cpp
int y = __shfl_xor_sync(mask, x, 1);
```

这类 API 自身带有同步 mask，通常也不需要额外 `__syncwarp()`。

但如果正确性依赖 warp 内线程先写后读，尤其涉及 shared memory 或复杂控制流，应该显式使用 `__syncwarp(mask)`。这能避免依赖隐式 warp-synchronous 行为。

## 为什么现代代码建议显式同步

早期 GPU 上，warp 内线程基本 lockstep 执行，很多代码默认：

```text
同一个 warp 内，上一条 shared 写完，下一条 shared 读就能看到。
```

Volta 之后引入 independent thread scheduling，warp 仍是重要执行单位，但硬件可以做更细粒度的线程调度。再加上分支、mask 和编译器优化，不能永远假设所有 lane 在每个点都严格同步。

因此：

```text
很多简单 warp 内 shared memory 通信不加 __syncwarp 也可能正确；
但这是依赖隐式 warp-synchronous 行为。
现代 CUDA 中，如果正确性依赖 warp 内先写后读，应显式使用 __syncwarp(mask)。
```

## 分支中的 __syncwarp

`__syncwarp(mask)` 可以出现在分支中，但 mask 必须准确描述实际会到达同步点的 lane。

危险写法：

```cpp
if (lane < 16) {
    __syncwarp(0xffffffff);
}
```

这里的问题是：

```text
mask 包含 32 个 lane
实际只有 16 个 lane 到达同步点
```

更合理的写法：

```cpp
unsigned mask = __ballot_sync(0xffffffff, lane < 16);

if (lane < 16) {
    __syncwarp(mask);
}
```

规则：

```text
mask 中指定的 lane 都必须执行同一个 __syncwarp(mask)
```

## 为什么不能替代 block 同步

`__syncwarp()` 只保证当前 warp 内 mask 指定 lane 到达同步点。

它不保证：

```text
warp 1 到没到
warp 2 到没到
整个 block 的 shared memory 写完没写完
```

如果数据依赖跨 warp，例如：

```text
warp 0 写 shared memory
warp 1 读 shared memory
```

`__syncwarp()` 不够，需要 block 级同步：

```cpp
__syncthreads();
```

## 成本

warp 级同步通常代价较低：

```text
同步范围小
参与线程最多 32 个 lane
不需要等待 block 内其他 warp
```

但如果 mask 中某些 lane 因分支或长延迟操作迟迟不到，同样会等待。

总结：

```text
__syncwarp(mask):
  同步范围：同一个 warp 内 mask 指定 lane
  代价：通常较低
  适合：warp 内通信、warp 内 shared memory 先写后读
```
