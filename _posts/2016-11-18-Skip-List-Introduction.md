---
title: Skip List Introduction and Implementation
categories: programming
tags: [data structure,probability]
---

# 什么是*Skip List* ？

## *skip list*定义和特性

> __skip list__是一种可以对有序的元素序列进行快速搜索的数据结构。其结构如下图所示:
![skiplist](https://upload.wikimedia.org/wikipedia/commons/8/86/Skip_list.svg)

可以看到，*skip list* 是由多层*list* 组成的。 其最底层是普通的有序地链表。而上面每一层都是作为下面层的链表的快速查找通道。对于在第`i`层的元素，其出现在`i+1`层的概率是固定的`p`。

## *skip list* 查找操作
## *skip list* 插入操作

![skiplist insertion](https://upload.wikimedia.org/wikipedia/commons/2/2c/Skip_list_add_element-en.gif)

# *Skip List* 实现

```cpp
```

# 参考

以下是一些参考信息 

1.[wikipedia - skip list](https://en.wikipedia.org/wiki/Skip_list)   
2.[leveldb - skiplist](https://github.com/google/leveldb/blob/master/db/skiplist.h)  
