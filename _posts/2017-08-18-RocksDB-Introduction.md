---
title : RocksDB Introduction
categories : programming
tags : [c++,database]
---

# Introduction

*RocksDB* 是一个基于[LevelDB](https://github.com/facebook/rocksdb)开发的支持并发的键值数据库。主要在*LevelDB*上增加了多线程*Compaction*, 多线程*Memtable*插入， 并且从*HBase*里引入了*ColumnFamily*。

## RocksDB Architecture

与*LevelDB*在架构上基本是一致的， 如下图所示:
![rocksdb arch](/assets/images/rocksdb/rocksdb_arch.png)

## RocksDB Write Path

类似的， *RocksDB*首先写入*WAL log*里， 然后写入*MemTable*,
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_0.png)
当*MemTable*写满了之后， 就转换成*IMemtable*,
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_1.png)
并准备写到*L0 sstfile*里
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_2.png)
而*LSM* 结构的*sstfile*则会进行*Compaction*来对数据进行去重。
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_3.png)

## LSM Compaction

*Compaction* 是指周期的读入不同文件并合并删掉重复的信息，形成新的文件的过程。

*LSM*的*sstfiles*结构如下所示:
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_0.png)

首先从L0和L1里选出Key range重叠的*sstfile*， 经过删除重复冗余键值然后写出新文件到L1
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_1.png)

接下来从Compact L1和L2 *sstfiles*， 类似之前操作， 不过现在新文件写出到L2
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_2.png)

*LSM* 里每一层都进行类似操作
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_3.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_4.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_5.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_6.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_7.png)

## LSM Size分布

在*LSM*里， 最下面一层保存数据占据了总体的90%, 如下面分布
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_base_size.png)

## LSM Multithread Compaction

为了提高数据*Compaction*的效率( 可以影响数据库写效率), *RocksDB*使用了多线程Compaction操作， 
![rocksdb arch](/assets/images/rocksdb/rocksdb_multithread_compaction.png)

## Multithread Memtable Insert

和*LevelDB*主要不同是， *RocksDB*支持*MemTable* 的并行插入操作（只有*skiplist*的*Memtable*支持)
![rocksdb arch](/assets/images/rocksdb/rocksdb_memtable_concurrent_insert.png)

## ColumnFamily

每个*ColumnFamily*都有自己独立的*Memtable*和*sstfiles*
![rocksdb arch](/assets/images/rocksdb/rocksdb_column_family.png)

# Reference

* [wiki - RocksDB](https://en.wikipedia.org/wiki/RocksDB)
* [github - RocksDB](https://github.com/facebook/rocksdb)
* [RocksDB Introduction on Percona](https://www.percona.com/live/data-performance-conference-2016/sites/default/files/slides/RocksDB_Siying_Dong.pdf)
* [RocksDB Storage Engine](https://www.percona.com/live/europe-amsterdam-2015/sites/default/files/slides/canadi_percona_live_talk.pdf)
