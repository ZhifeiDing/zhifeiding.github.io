---
title: Implement malloc using C/C++
categories: programming
tags: [c++,algorithm]
---

# `malloc`是做什么的?

`malloc`是*C*程序中用来分配内存的一个库函数。*C++*里的*new*也是调用`malloc`实现的。为了清楚内存是怎么分配的，我们用*C*来实现一个简单的`malloc`函数.

## `malloc`,`brk`以及`sbrk`系统调用

我们知道，
![memory organization](assets/images/MemoryOrganization.png)
![Heap organization](assets/images/HeapOrganization.png)

# `malloc`实现

## 一个简单的`malloc`实现

## `malloc`

## `calloc`,`free`,`ralloc`实现

# 参考

* [Malloc Tutorial](http://www.inf.udec.cl/~leo/Malloc_tutorial.pdf)
* [A simple malloc implementation](http://danluu.com/malloc-tutorial/)
