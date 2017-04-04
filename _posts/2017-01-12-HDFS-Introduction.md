---
title : HDFS Instroduction
categories : bit data
tags : [big data, file system]
---

# 什么是 __HDFS__ ?

> __HDFS__ 与之前介绍的[Btrfs](http://zhifeiding.github.io/programming/2016/09/07/Btrfs-Introduction/)不同，__HDFS__ 是针对 __[Hadoop](https://en.wikipedia.org/wiki/Apache_Hadoop)__ 廉价的计算机开发的有容错功能的分布式文件系统。主要用来存储大文件，能够提供很高的吞吐量，但是不适合對时延要求高的场景。另外，为了保持一致性， __HDFS__ 只支持 `write-once-read-many`, 文件一旦创建，操作结束之后， 只能对文件进行追加和截取操作。

# __HDFS__ 基本结构

__HDFS__ 在结构上是由一个 `NameNode` 和 多个 `DataNode` 组成， 其中`NameNode`
和 `DataNode`之间使用`TCP/IP Socket`进行通信， 而使用 `RCP`和客户端通信。
下图是基本的 __HDFS__ 结构 :
![hdfs arch](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/images/hdfsarchitecture.png){HDFS Architecture}

# 参考

* [Hadoop wiki - HDFS](https://wiki.apache.org/hadoop/HDFS/)
* [Hadoop wikipedia - HDFS](https://en.wikipedia.org/wiki/Apache_Hadoop#HDFS)
* [Hadoop HDFS Document](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)