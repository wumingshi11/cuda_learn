# Shared Memory Bank Conflict

## 核心原因

shared memory 被硬件划分成多个 bank。bank conflict 的根本原因是硬件限制：

> 在一个 warp 执行同一条 shared memory load/store 指令时，一个 bank 通常一次只能服务一个不同地址。

如果同一个 warp 内多个线程同时访问同一个 bank 的不同地址，硬件需要把这次访问拆成多次处理，访问会被部分串行化，性能下降。

## Warp、Block 和 Bank

- 一个 warp 通常是 32 个线程。
- 一个 block 可以包含多个 warp。
- shared memory 通常被划分为多个 bank。
- bank conflict 发生在同一个 warp 的同一条 shared memory 访问指令中。

以 `dim3 block(32, 8)` 为例：

```text
block 线程数 = 32 * 8 = 256
warp 线程数  = 32
block 中 warp 数 = 256 / 32 = 8
```

## 为什么 `tile[32][33]` 可以避免冲突

如果 shared memory tile 定义为：

```cpp
__shared__ float tile[32][32];
```

访问同一列时，线性地址间隔是 32 个 `float`：

```text
tile[0][0]  -> index 0
tile[1][0]  -> index 32
tile[2][0]  -> index 64
...
tile[31][0] -> index 992
```

对于 `float`，可以粗略理解为：

```text
bank_id = element_index % 32
```

这些访问都会落到同一个 bank：

```text
0 % 32   = 0
32 % 32  = 0
64 % 32  = 0
...
992 % 32 = 0
```

这会导致同一个 warp 内多个线程访问同一个 bank 的不同地址，产生严重 bank conflict。

改成：

```cpp
__shared__ float tile[32][33];
```

每一行多出 1 个 padding 元素，行跨度从 32 变成 33。访问同一列时：

```text
tile[0][0]  -> index 0    -> bank 0
tile[1][0]  -> index 33   -> bank 1
tile[2][0]  -> index 66   -> bank 2
...
tile[31][0] -> index 1023 -> bank 31
```

这样一个 warp 的 32 个线程可以分散访问 32 个 bank，硬件可以并行服务，性能更高。

## Broadcast 例外

如果同一个 warp 的多个线程访问的是 shared memory 的同一个地址，硬件可以 broadcast：

```cpp
float v = tile[0][0];
```

如果 warp 内 32 个线程都读同一个 `tile[0][0]`，硬件只需要读取一次，然后把值分发给多个线程。这种情况不算普通 bank conflict。

总结：

```text
同 bank + 同地址   -> broadcast，通常高效
同 bank + 不同地址 -> bank conflict，需要拆分访问
不同 bank          -> 可以并行访问
```
