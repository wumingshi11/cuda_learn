# Warp Divergence

## 核心原因

GPU 不是给每个线程配一个独立调度器。NVIDIA GPU 的基本调度单位通常是 warp：

```text
一个 warp = 32 个线程
调度器调度 warp，而不是单个 thread
```

warp 内 32 个线程通常执行同一条指令，这种模型叫 SIMT：

```text
Single Instruction, Multiple Threads
```

每个线程有自己的寄存器、线程索引和数据，但指令发射通常以 warp 为单位。

## if / switch 如何执行

如果同一个 warp 内所有线程的分支判断结果一致，执行效率较高：

```text
32 个线程都 true  -> 执行 if 内代码
32 个线程都 false -> 跳过 if 内代码
```

如果同一个 warp 内一部分线程走 true，另一部分线程走 false，就会发生 warp divergence：

```text
thread 0-15  -> true 分支
thread 16-31 -> false 分支
```

GPU 通常会把不同路径分开执行：

```text
1. 执行 true 路径，false 线程被 mask 掉
2. 执行 false 路径，true 线程被 mask 掉
3. 分支结束后 reconverge，warp 重新汇合
```

所以不同分支不是在同一个 warp 内真正同时并行执行，而是可能被串行化执行。分支路径越多、分歧越严重，warp 的有效执行效率越低。

`switch` 也类似。如果同一个 warp 内线程进入多个不同 case，硬件可能需要分别执行这些 case，并对不属于当前 case 的线程做 mask。

## 在矩阵转置代码中的体现

例如 `src/matrix_trans.cu` 中的边界检查：

```cpp
if ((y + i) < height && x < width) {
    tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * width + x];
}
```

这个 `if` 用来防止越界访问。

如果矩阵尺寸刚好是 `TILE_DIM` 的整数倍，大多数 block 内线程的判断结果一致，基本不会造成明显分支发散。

如果矩阵尺寸不是 `TILE_DIM` 的整数倍，边界 block 里可能出现部分线程有效、部分线程越界：

```text
内部 block：线程判断通常一致，分支发散少
边界 block：部分线程 true、部分线程 false，可能发生分支发散
```

这种边界检查通常是可以接受的，因为只有边缘区域受影响。

## 编写 CUDA 分支代码时的原则

- 尽量让同一个 warp 内线程走相同控制流。
- 把不可避免的分支限制在少量边界 block 中。
- 对高频执行路径，避免让 warp 内线程进入大量不同分支。
- 分支条件如果只依赖 block 级别信息，通常比依赖 thread 级别信息更不容易发散。

总结：

```text
GPU 调度单位通常是 warp
warp 内线程通常执行同一条指令
同 warp 内线程走不同 if/switch 路径时，会发生 warp divergence
不同路径通常被 mask 后分批执行，性能会下降
```
