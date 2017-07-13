---
title : leveldb Instroduction
categories : database, key-value
tags : [database, file system]
---

# 什么是 __leveldb__ ?

> __leveldb__ 是由 *google* 的*jeff dean* 开发的*in-memory* 的*key-value* 数据库， 提供了基本的`Put`, `Get`, `Delete` 操作。

# __leveldb__ 基本结构

了解 __leveldb__ 基本操作之前， 下面先介绍其使用的数据结构

## __memtable__

__memtable__ 是用来在 *in-memory* 中存储*key-value* 的数据结构， 其实质是 *[skiplist](http://zhifeiding.github.io/programming/2016/11/18/Skip-List-Introduction/)* 。存储的数据的结构如下所示：

![memtable](/assets/images/leveldb/memtable.png)

由上图可知：

* `userkey` 和 `tag` 组成了`internal_key`
* 其中 *8 byte* 的`tag`是由 *7 byte* 的`sequence_number` 和 *1 byte* 的`type`组合而成的
* `sequence_number`是一个递增的无符号数据， 每次对 __leveldb__ 的`Put`,`Get`,`Delete`操作都会使其加一
* `type`可取的值分别是：`kTypeValue` 表示当前数据有效，`kTypeDeletion`表示当前数据已删除。 需要注意的是， 当使用`Delete`操作时, __memtable__ 实际只是插入一个`type` 为`kTypeDeletion`的值。 

*skiplist* 内部数据是有序的， 也就是我们存储在 __memtable__ 里的数据会根据`userkey` 和 `sequence_number`来排序， 基本原则是先比较`userkey`，之后再根据`sequence_number`大的在前。

## __write batch__ 格式

当调用`Put`和`Delete`
![write_batch](/assets/images/leveldb/write_batch.png)

![log](/assets/images/leveldb/log.png)

![sstable](/assets/images/leveldb/sstable.png)

![block](/assets/images/leveldb/block.png)

![footer](/assets/images/leveldb/footer.png)

![metaindex_block](/assets/images/leveldb/metaindex_block.png)

![index_block](/assets/images/leveldb/index_block.png)

![filter_block](/assets/images/leveldb/filter_block.png)

# 参考
