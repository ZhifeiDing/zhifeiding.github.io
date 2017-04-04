---
title : HDFS Instroduction
categories : bit data
tags : [big data, file system]
---

# 什么是 __HDFS__ ?

> __HDFS__ 与之前介绍的[Btrfs](http://zhifeiding.github.io/programming/2016/09/07/Btrfs-Introduction/)不同，__HDFS__ 是针对 __[Hadoop](https://en.wikipedia.org/wiki/Apache_Hadoop)__ 框架适用于廉价计算机并具有容错功能的分布式文件系统。主要用来存储大文件，能够提供很高的吞吐量，但是不适合對时延要求高的场景。另外，为了保持一致性， __HDFS__ 只支持 `write-once-read-many`, 文件一旦创建，操作结束之后， 只能对文件进行追加和截取操作。

# __HDFS__ 基本结构

__HDFS__ 在结构上是由一个 `NameNode` 和 多个 `DataNode` 组成， 其中`NameNode`
和 `DataNode`之间使用`TCP/IP Socket`进行通信， 而使用 `RCP`和客户端通信。
下图是基本的 __HDFS__ 结构 :
![hdfs arch](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/images/hdfsarchitecture.png)

## __NameNode__ 功能

从结构上， __HDFS__ 是`Master/Slave`架构，
而其中`NameNode`就充当`Master`，来管理文件空间，以及管理客户端对文件系统的访问。`NameNode`提供基本的文件操作如文件创建，删除，复制等操作，并且提供文件访问和权限功能，但是不支持链接。所有对文件系统的操作都会在`NameNode`里记录, 在`NameNode`里，有一个`EditLog`，会记录对文件系统的所有的操作，不管是创建一个新文件，还是改变文件保存的复制数，__HDFS__ 将该文件保存在`NameNode`所在的主机的本地文件系统里。另外，关于 __HDFS__ 的一些基本信息，包括文件保存的`blocks`，文件系统配置会被保存在本地文件系统的`Fsimage`文件中。一般的，__HDFS__ 会在内存中存储一份`Fsimage`和`EditLog`，当 __NameNode__ 启动或者触发 __checkpoint__ 时，__HDFS__ 会读取磁盘中的 `Fsimage`和`EditLog`，并且应用`EditLog`中的操作，同时更新磁盘中的`Fsimage`和`EditLog`。

## __DataNode__ 功能


# 参考

* [Hadoop wiki - HDFS](https://wiki.apache.org/hadoop/HDFS/)
* [Hadoop wikipedia - HDFS](https://en.wikipedia.org/wiki/Apache_Hadoop#HDFS)
* [Hadoop HDFS Document](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)