---
title : Concurrent Linked List Introduction and Implementation
categories : programming
tags : [data structure, c++, concurrent, lock-free]
---

# 背景介绍

## `RMW` - Read-Modify-Write

## `CAS` - Compare-And-Swap

## Sequential Consistency

## Memory Order

`Memory Order`是指执行指令时`Load`和`Store`的顺序。

## Memory Model

`Memory Model`是指能够执行`Out-of-Order`的`CPU`能够对`Load`和`Store`指令进行`Reorder`的方式。不同的`CPU`对应各自的`Memory Order`, 根据能够`Reorder`的总类， 一般可以分为`Strong Memory Order`和`Relaxed Memory Order`。 一般的， *X86*系列的是`Strong Memory Order`, 而`ARM`系列的则属于`Relaxed Memory Order`。 具体可见下表:

| Reordering Activity	| x86 and x64	| ARM |
| ------------------- |:-----------:|:----|
| Reads moving ahead of reads	| No | Yes |
| Writes moving ahead of writes	| No | Yes |
| Writes moving ahead of reads	| No | Yes |
| Reads moving ahead of writes	| Yes	| Yes |

***

# Generic Linked List

# Linked List using Lock

# Lock-free Linked List

# Reference

* [Techniques for Implementing Concurrent Data Structures on Modern Multicore Machines](https://people.eecs.berkeley.edu/~stephentu/presentations/workshop.pdf)           
* [Generic Concurrent Lock-free Linked list](https://people.csail.mit.edu/bushl2/rpi/project_web/page5.html)       
* [Lock Free Linked Lists and Skip Lists](http://www.cse.yorku.ca/~ruppert/papers/lfll.pdf)            
* [Lockless Programming Considerations for Xbox 360 and Microsoft Windows](https://msdn.microsoft.com/en-us/library/windows/desktop/ee418650(v=vs.85).aspx)         
* [An Introduction to Lock-Free Programming](http://preshing.com/20120612/an-introduction-to-lock-free-programming/)           
