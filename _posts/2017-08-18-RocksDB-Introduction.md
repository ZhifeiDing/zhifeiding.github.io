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

类似的， *RocksDB*首先写入
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_0.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_1.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_2.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_write_path_3.png)

![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_0.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_1.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_2.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_3.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_4.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_5.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_6.png)
![rocksdb arch](/assets/images/rocksdb/rocksdb_level_based_compaction_7.png)

![rocksdb arch](/assets/images/rocksdb/rocksdb_level_base_size.png)

![rocksdb arch](/assets/images/rocksdb/rocksdb_multithread_compaction.png)

![rocksdb arch](/assets/images/rocksdb/rocksdb_memtable_concurrent_insert.png)

![rocksdb arch](/assets/images/rocksdb/rocksdb_column_family.png)

# Reference

* [wiki - RocksDB](https://en.wikipedia.org/wiki/RocksDB)
* [github - RocksDB](https://github.com/facebook/rocksdb)
* [RocksDB Introduction on Percona](https://www.percona.com/live/data-performance-conference-2016/sites/default/files/slides/RocksDB_Siying_Dong.pdf)
* [RocksDB Storage Engine](https://www.percona.com/live/europe-amsterdam-2015/sites/default/files/slides/canadi_percona_live_talk.pdf)
