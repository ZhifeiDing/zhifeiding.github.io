---
title: Implement malloc using C/C++
categories: programming
tags: [c++,algorithm,memory]
---

# `malloc`是做什么的?

`malloc`是*C*程序中用来分配内存的一个库函数。*C++*里的*new*也是调用`malloc`实现的。为了清楚内存是怎么分配的，我们用*C*来实现一个简单的`malloc`函数.

## `malloc`,`brk`以及`sbrk`系统调用

首先，我们知道计算机内存一般是分为下面几部份的：

* Text Segment : 程序本身
* Data Segment : 所有初始化的全局变量和静态变量
* BSS Segment : 所有未初始化的全局变量
* Heap Segment : 程序中动态分配的内存区
* Stack Segment : 程序中局部变量的内存区

计算机内存组织一般如下图所示:
![memory organization](/assets/images/MemoryOrganization.png)

而其中*Heap*就是用`malloc`分配的内存，一般分为*mapped region*, *unmapped region*, 如下图所示:
![Heap organization](/assets/images/HeapOrganization.png)

`malloc`函数可以在*Heap Segment*里*unmapped region*区域分配指定大小的内存并返回起始地址。
清楚了内存和`malloc`的关系，我们可以定义`malloc`的函数原型如下:

```cpp
void* malloc(size_t size);
```

`brk`以及`sbrk`的函数原型

```cpp
int brk(const void *addr);
void* sbrk(intptr_t incr);
```

# `malloc`实现

## 一个简单的`malloc`实现

```cpp
#include <sys/types.h>
#include <unistd.h>

void *malloc(size_t size) {
  void *p;
  p = sbrk (0);
  /* If sbrk fails , we return NULL */
  if (sbrk(size) == (void*)-1)
    return NULL;
  return p;
}
```

## `malloc`

## `calloc`,`free`,`ralloc`实现

# 参考

* [Malloc Tutorial](http://www.inf.udec.cl/~leo/Malloc_tutorial.pdf)
* [A simple malloc implementation](http://danluu.com/malloc-tutorial/)
