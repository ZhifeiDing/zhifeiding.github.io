---
title : Concurrent Linked List Introduction and Implementation
categories : programming
tags : [data structure, c++, concurrent, lock-free]
---

# 背景介绍

在具体实现之前， 现解释一下在*Concurrent Programming* 中需要用到的一些概念。

* __RMW__ - [Read-Modify-Write](https://en.wikipedia.org/wiki/Read-modify-write)

> 
__RMW__是指在一个操作中读取*memory* 的值，然后写入新的值的原子操作(*Atomic*)。这些指令能防止在多线程编程中出现竞争。

* __CAS__ - [Compare-And-Swap](https://en.wikipedia.org/wiki/Compare-and-swap)

> 
__CAS__是上面的`RMW`指令中的一种，具体就是要修改一个*memory* 地址的值时，先用当前的值和之前写入的值进行比较，如果一致，则写入新值。否则写入失败。由于__CAS__是原子操作，所以在多线程中可以保证同时只被一个线程修改。用`C++`可以表示如下:

```cpp
bool cas(int *ptr, int oldVal, int newVal) {
    if( *ptr == oldVal ) {
        *ptr = newVal;
        return true;
    }
    return false;
}
```

* __Sequential Consistency__

> 
__Sequential Consistency__ 是指对*memory* 的读写操作的顺序在所有线程中都是和程序代码中一致的。具体的理解就是在所有线程都运行在单核,并且程序是在关掉编译器优化功能后编译的。这样对于`CPU`，所有的*memory* 读写访问操作都是确定的。

* __Memory Order__

> __Memory Order__是指执行指令时`Load`和`Store`的顺序。影响*memory*
的读写指令顺序因素有两个 :

1. __compiler reordering__ : 指*compiler* 在编译优化过程中不改变程序正确性前提下对`Load`和`Store`指令执行顺序进行的调整。
对于*x86*上的`gcc`可以用下面来保证代码后面`Load/Store`不会在代码之前执行:

```cpp
inline void MemoryBarrier() {
  // prevent compiler reordering
  __asm__ __volatile__("" : : : "memory");
}
```
2. __processor reordering__ : 指*CPU*本身的`Out-of-Order`功能在执行过程中对`Load/Store`指令的调整。
对于*x86*上的`gcc`可以用下面来保证代码后面`Load/Store`不会在代码之前执行:
```cpp
inline void MemoryBarrier() {
  // prevent compiler && cpu processor reordering
  __asm__ __volatile__("mfence" : : : "memory");
}
```
* __Memory Model__

> 
__Memory Model__是指能够执行`Out-of-Order`的`CPU`能够对`Load`和`Store`指令进行`Reorder`的方式。不同的`CPU`对应各自的`Memory Order`, 根据能够`Reorder`的总类， 一般可以分为`Strong Memory Order`和`Relaxed Memory Order`。 一般的， *X86*系列的是`Strong Memory Order`, 而`ARM`系列的则属于`Relaxed Memory Order`。 具体可见下表:

| Reordering Activity	| x86 and x64	| ARM |
| ------------------- |:-----------:|:----|
| Reads moving ahead of reads	| No | Yes |
| Writes moving ahead of writes	| No | Yes |
| Writes moving ahead of reads	| No | Yes |
| Reads moving ahead of writes	| Yes	| Yes |

* __[ABA Problem](https://en.wikipedia.org/wiki/ABA_problem)__

> 
上面提到的__CAS__中我们读取一个值并与其之前值进行比较， 当两者一致时，我们认为在这之间状态没有变化。但是，其他线程可能已经对数据进行一系列操作，致使最后退出时将该值修改为之前的值。 这种出现在__Lock-Free__结构里的现象，就被称为__ABA Problem__。 
具体可以看下面__linklist__ `head->NodeA->NodeB->NodeC`的例子:

1. __Thread 1__从*memory* 里读取 `NodeA`值， 然后__Thread 1__被调度出，__Thread 2__开始执行
2. __Thread 2__ 删除`NodeA`和`NodeB`, 然后插入`NodeA`, 变成`head->NodeA->NodeC`
3. __Thread 1__继续执行， 删除`NodeA`, 将`head->next = NodeB`， 这是就会出现访问未分配的地址

__ABA__问题可以对使用指针地址加上`tag bit`来避免。 对于`x86`来说， 指针是`4 byte`, 所以后`2 bit`可以用来作为`tag bit`。

* __Read Acquire__和__Write Release__



***

# Generic Linked List

# Linked List using Lock

# Lock-free Linked List

# Reference

* [Implementing Concurrent Data Structures on Modern Multicore Machines](https://people.eecs.berkeley.edu/~stephentu/presentations/workshop.pdf)
* [Generic Concurrent Lock-free Linked list](https://people.csail.mit.edu/bushl2/rpi/project_web/page5.html)
* [Lock Free Linked Lists and Skip Lists](http://www.cse.yorku.ca/~ruppert/papers/lfll.pdf)
* [Lockless Programming Considerations for Xbox 360 and Microsoft Windows](https://msdn.microsoft.com/en-us/library/windows/desktop/ee418650(v=vs.85).aspx)
* [An Introduction to Lock-Free Programming](http://preshing.com/20120612/an-introduction-to-lock-free-programming/)    
* [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)    
