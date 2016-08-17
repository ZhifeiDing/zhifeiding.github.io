---
title: Implement malloc using C/C++
categories: programming
tags: [c++,algorithm,memory]
---

*Note:* 本文基本上是[Malloc Tutorial](http://www.inf.udec.cl/~leo/Malloc_tutorial.pdf)的整理和翻译。

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

从上面*Heap Segment*示意图中能够知道*Heap Segment*是由*starting address*, *break point*和*maximum limit*定义的一段连续内存空间。如果我们在*Heap*里分配内存，那我们必须知道*break point*的地址。我们可以借助`brk`和`sbrk`系统调用。

* `brk`将*break point*设置到指定的地址，如果成功返回0，否则返回1。
* `sbrk`则将*break point*向前移动指定的字节大小。失败则返回`(void*)-1`,执行成功后的返回值根据不同实现而不同。

> 当增加地址为零时(`sbrk(0)`)，返回值是实际的*break point*地址

`brk`以及`sbrk`的函数原型

```cpp
int brk(const void *addr);
void* sbrk(intptr_t incr);
```

# `malloc`实现

有了上面背景知识，我们可以开始一步一步实现`malloc`。

## 一个简单的`malloc`实现

首先，借助于上面的`sbrk`我们可以很容易想到一个实现`malloc`的方法，即首先用`sbrk(0)`得到*break point*地址，然后调用`sbrk`来调整大小，如果成功则返回之前得到的*break point*的地址。代码如下:

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
一个简单的`malloc`函数就这么实现了。我们可以用这个简易版`malloc`来动态分配内存。但是很快问题就出现了，分配的内存我们没有办法去释放，还需要继续改进。

## 更实际的`malloc`实现

* 首先，为了让我们`malloc`的memory能够`free`,我们需要定义`free`含义，如果一块memory能够再次被`malloc`分配使用，我们就说这块memory是`free`掉的。这样，我们很容易就想到使用一个结构体来保存当前分配memory的大小，释放等信息。

```cpp
typedef struct s_block *t_block;

struct s_block {
  size_t size;      // size of current block
  t_block next;     // pointer to next block
  int free;         // sign to indicate free or not
};

#define BLOCK_SIZE sizeof(struct s_block)

void *base=NULL;  // global pointer to the 
```
需要注意的是我们每次调用`sbrk`的时候需要加上`s_block`的大小。使用`s_block`之后*Heap Segment*是下面这样分布的:

![Heap List](/assets/images/HeapList.png)

* 有了上面的`s_block`之后，每次需要分配内存时我们简单的遍历`s_block`找到`free`并且大小满足的`s_block`就可以了。

```cpp
t_block find_block(t_block *last , size_t size) {
  t_block b=base;     // start from the heap starting point
  while (b && !(b->free && b->size >= size)) {
    *last = b;
    b = b->next;
  }
  return (b);
}
```

* 如果找不到合适的`s_block`,我们需要调用`sbrk()`来分配内存

```cpp
t_block extend_heap(t_block last , size_t s){
  t_block b;
  b = sbrk (0);
  if (sbrk(BLOCK_SIZE + s) == (void*)-1)
    /* sbrk fails , go to die */
    return (NULL);
  b->size = s;
  b->next = NULL;
  if (last)
    last ->next = b;
  b->free = 0;
  return (b);
}
```

## `malloc`实现

根据之前的思路，为了实现`malloc`需要下面这些操作:

* 如果是第一次调用`malloc`,则调用`extend_heap`来分配内存并且设置全局变量`base`
* 否则就调用`find_block`

```cpp
void *malloc(size_t size) {
  t_block b,last;
  if (base) {
    /* First find a block */
    last = base;
    b = find_block(&last ,s);
    if (b) {
      b->free=0;
    } else {
      /* No fitting block , extend the heap */
      b = extend_heap(last ,s);
      if (!b)
        return(NULL);
    }
  } else {
    /* first time */
    b = extend_heap(NULL ,s);
    if (!b)
      return NULL;
    base = b;
  }
  return b+1;
}
```

## `calloc`,`free`,`ralloc`实现

`malloc`类似函数还有:

* `calloc` ：除了实现`malloc`功能之外，还会初始化分配内存的值

```cpp
void *calloc(size_t nelem, size_t size) {
  size = nelem * size;
  void *ptr = malloc(size);
  memset(ptr, 0, size);
  return p;
}
```

* `free` : 释放`malloc`,`calloc`或`ralloc`分配的内存区域,简单的实现方式就是将`s_block`里的`free`标志位置为1

```cpp
void *get_block_ptr(void *ptr) {
  return (t_block)ptr - 1;
}

void free(void *ptr) {
  if( !ptr )
    return;
  
  t_block p = get_block_ptr(ptr);
  assert(p->free == 0);
  p->free = 1;
}
```

* `ralloc` : 调整之前分配的内存大小， 对于新的`size`小于当前`size`的，我们可以直接返回，否则重新分配并释放之前的内存

```cpp
void *ralloc(void *ptr, size_t size) {
  if( !ptr )
    return malloc(size);
  
  // if request size is smaller than current size, just return
  t_block p = get_block_ptr(ptr);
  if( p->size >= size )
    return ptr;
    
  // or request size is larger than current size, we need
  // malloc the new size, copy old data and free old space
  void *new_ptr = malloc(size);
  if( !new_ptr )
    return NULL;
  memcpy(new_ptr, ptr, p->size);
  free(ptr);
  return new_ptr;
}
```

# 总结

上面实现了一个基本功能的`malloc`, 不过还有一些功能还需要完善，比如memory 对齐，分配时memory split和释放时memory fragment处理等

# 参考

* [Malloc Tutorial](http://www.inf.udec.cl/~leo/Malloc_tutorial.pdf)
* [A simple malloc implementation](http://danluu.com/malloc-tutorial/)
