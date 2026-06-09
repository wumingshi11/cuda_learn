# Atomic, Fence and Barrier

## 核心区别

atomic、fence、barrier 是三个不同概念。

```text
atomic  -> 保护某个地址的并发读改写
fence   -> 约束当前线程前后内存操作的顺序和可见性
barrier -> 让一组线程互相等待
```

不要把它们混成一种同步。

## Atomic 解决什么

atomic 解决的是多个线程同时更新同一个地址时，不互相覆盖。

例如：

```cpp
atomicAdd(counter, 1);
```

它保证这个地址上的 read-modify-write 是原子的：

```text
读取 counter
加 1
写回 counter
```

不会被其他线程对同一地址的原子操作打断。

但 atomic 不是 barrier。它不表示：

```text
所有线程都到达这里
所有线程互相等待
所有 shared/global 写入都已经完成
```

如果用全局原子变量做累计：

```cpp
atomicAdd(global_sum, block_sum);
```

它保证的是 `global_sum` 这个地址的更新正确，不是做 grid 级同步。

## Fence 解决什么

fence 解决的是当前线程的内存写入顺序和可见性。

常见 CUDA fence：

```cpp
__threadfence_block();   // block 范围
__threadfence();         // device 范围
__threadfence_system();  // system 范围，host / peer device
```

更准确地说：

```text
fence 保证当前线程在 fence 之前的内存写入，
在指定作用域内，不会被 fence 之后的写入越过。
```

它不是简单的 “把数据刷新到其他线程缓存”。GPU cache 层级和实现细节更复杂。学习时应该理解为：

```text
fence 建立当前线程写入的顺序和可见性保证
```

## Fence 的典型发布模式

典型场景是 producer-consumer：

```text
先写真正的数据
fence
再写 flag / counter / doorbell 通知别人
```

例如：

```cpp
data[idx] = value;
__threadfence();
atomicExch(flag, 1);
```

消费者：

```cpp
while (atomicAdd(flag, 0) == 0) {
    // wait
}

int x = data[idx];
```

这里的语义是：

```text
如果消费者看到了 flag == 1，
那么生产者在 fence 前写的 data[idx] = value
也应该已经在 device 范围内可见。
```

没有 fence 的风险是：

```text
消费者可能先看到 flag 更新，
但 data 写入还没有按预期对消费者可见。
```

## 多 block partial sum 的例子

一个常见模式：

```cpp
partial[blockIdx.x] = block_sum;

__threadfence();

int ticket = atomicAdd(counter, 1);
```

含义：

```text
先写本 block 的 partial result
fence 保证 partial 写入对 device 可见
再通过 atomicAdd 告诉别人：这个 block 完成了
```

如果最后一个 block 通过 `counter` 判断所有 block 都完成，并开始读取 `partial[]` 汇总，那么 fence 能避免：

```text
counter 显示某个 block 完成
但这个 block 的 partial 写入还没对其他 block 可见
```

## Fence 不是等待

`__threadfence()` 不会让其他线程停下来。

它只约束执行该 fence 的当前线程：

```text
当前线程之前的写入
和当前线程之后的写入
在指定作用域内的顺序和可见性
```

它不能保证：

```text
其他线程也执行到了这里
其他线程的写入也完成了
所有线程互相等待
```

如果需要等待一组线程到达某个点，需要 barrier：

```cpp
__syncwarp(mask);   // warp 级 barrier
__syncthreads();    // block 级 barrier
```

## Barrier 解决什么

barrier 解决的是一组线程互相等待。

例如 block 级同步：

```cpp
__syncthreads();
```

含义：

```text
block 内所有线程都到达后，才继续执行
```

它常用于 shared memory 阶段切换：

```cpp
tile[...] = input[...];

__syncthreads();

output[...] = tile[...];
```

barrier 和 fence 的区别：

```text
barrier -> 等人
fence   -> 保证当前线程写入的顺序和可见性
```

## Atomic 是否属于 fence

不完全属于。

可以保守理解为：

```text
atomic 保证目标地址的原子读改写
fence 保证更一般的内存写入顺序和可见性
```

某些 atomic 在特定 CUDA 版本、架构和内存语义下会带有一定顺序约束，但不能简单把所有 atomic 当作 fence。

如果需要这种模式：

```text
先写 data
再更新 flag / counter
其他线程看到 flag / counter 后读取 data
```

就应该明确考虑 fence：

```cpp
data[idx] = value;
__threadfence();
atomicExch(flag, 1);
```

总结：

```text
atomic 不是 fence，也不是 barrier。
atomic 主要保证某个地址的并发读改写正确。
```

## 总结

```text
atomic:
  保护某个地址的并发读改写
  不等待其他线程

fence:
  约束当前线程前后内存操作的顺序和可见性
  不等待其他线程

barrier:
  让一组线程互相等待
  常常也提供对应范围内的内存可见性保证
```

记忆方式：

```text
atomic 管一个变量怎么改
fence 管我写的内容按什么顺序被看见
barrier 管大家是否都到齐
```
