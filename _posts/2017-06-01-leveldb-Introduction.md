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
* `type`可取的值分别是：`kTypeValue` 表示当前数据有效，`kTypeDeletion`表示当前数据已删除。 需要注意的是， 当使用`Delete`操作时, __memtable__ 实际只是插入一个`type` 为`kTypeDeletion`的空值。 

*skiplist* 内部数据是有序的， 也就是我们存储在 __memtable__ 里的数据会根据`userkey` 和 `sequence_number`来排序， 基本原则是先比较`userkey`，之后再根据`sequence_number`大的在前。

## __write batch__ 格式

`WriteBatch`可以用来保证一系列操作的完整性，里面的操作要么都成功， 要么都失败。当调用`Put`或`Delete`时， __leveldb__ 上会将其操作放到 `WriteBatch`对象中。另外 __leveldb__ 在实际执行`WriteBatch`里操作之前，会尝试将几个`WriteBatch`合并一起，可以减少`write lock`的消耗。`WriteBatch`里的数据格式如下图所示:

![write_batch](/assets/images/leveldb/write_batch.png)

当合并几个`WriteBatch`时，取其中最大的`SequenceNumber`作为新的`SequenceNumber`。

## __log__ 格式

为了保证数据安全， 在每一个`WriteBatch`里操作在`memtable`里执行之前， 会将`WriteBatch`里的数据写入到`log`文件里保存。当`memtable`写入到`sstable`文件里时，`log`文件会打开一个新的文件。`log`文件保存数据格式如下图所示:

![log](/assets/images/leveldb/log.png)

由上图可知，`log`文件是以`kBlockSize`来存储数据， 其中一个`kBlockSize`多个`kHeader`和`data`, 只有一个`kBlockSize`剩下不到`kHeader`的空间才会填充`trailer(ox00)`。存储的`data`就是上面介绍的`WriteBatch`的数据。其中`type`可能值是:

* `kFullType` : 表示`data`全部存储在当前`kBlockSize`内
* `kFirstType` : 表示当前`kBlockSize`存储的是`data`的第一部分
* `kMiddleType`：表示当前`kBlockSize`存储的是`data`的中间部分
* `kLastType` ：表示当前`kBlockSize`存储的是`data`的最后一部分

## __sstable__ 格式

当`memtable`占用内存空间大小超过`options_.write_buffer_size( default = 4MB)`时， __leveldb__ 会将`memtable`写到`sstable`文件中。`sstable`文件存储的基本格式如下图所示:

![sstable](/assets/images/leveldb/sstable.png)

由上图可知， ``sstable`文件中是以`options_.block_size(default = 4KB)`来组织数据的， 首先存储数据库数据。 如果有使用`FilterPolicy`， 接下来就存储`FilterPolicy`的数据。然后接下来存储数据库的信息，接下来就是用来索引`data block`的`index block`。文件最后会存储`footer`。上述各自具体格式下面会详细介绍。

## __block__ 格式

上面不管是`data block`, `index block`, `metaindex block`还是`filter block`， 其基本结构都如下图所示：

![block](/assets/images/leveldb/block.png)



## __footer__ 格式

`footer`保存在``sstable`文件最后，其保存数据及其格式如下图所示:

![footer](/assets/images/leveldb/footer.png)

其中分别保存了`metaindex_handle`和`index_handle`对象， 可以通过这两个对象来访问`sstable`里`metaindex block`和`index block`。

## __metaindex block__ 格式

`metaindex block`当前只保存了`filter name`及`filter_block_handle`， 具体格式参见下图：

![metaindex_block](/assets/images/leveldb/metaindex_block.png)

通过`filter_block_handle`可以访问`sstable`里的`filter block`。 

## __index block__ 格式

![index_block](/assets/images/leveldb/index_block.png)

## __filter block__ 格式

![filter_block](/assets/images/leveldb/filter_block.png)

# 参考

* [leveldb-github](https://github.com/google/leveldb)  
