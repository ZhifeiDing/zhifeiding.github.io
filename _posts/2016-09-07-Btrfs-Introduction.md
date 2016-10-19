---
title : Btrfs Introduction
categories : programming
tags : [data structure, file system, linux]
---

# 文件系统历史回顾

为了能够更好的理解*Btrfs*　的特性，下面首先回顾以一下*Linux* 上使用的各个文件系统历史及其特点：

* 1992.04 *ext[^1]* 是第一个专门为*Linux* 设计的文件系统， 有*Unix File System(UFS)[^3]* 类似的*metedata* , 并且是第一个使用*VFS[^4]* 。解决了之前*Minix* 里64M最大容量和14个字符文件名的限制(支持2G容量和255个字符的文件名)，但是不支持单独时间戳的文件访问以及数据修改

* 1993.01 *ext2[^5]* 基于*Berkely Fast File System* 同样的原则开发并取代*ext* 成为*Linux* 文件系统。
   *ext2* 文件系统结构图如下所示：
![ext2 structure](/assets/images/ext2fs.png) 

可以看到，*ext2* 文件系统主要由*Boot record* 和*Block Groups* 组成，而后者包含*Super block* , *Block Group Descriptor Table* , *Block Bitmap*, *Inode Bitmap*, *Inode table*, *Data blocks* , 其中：　

1. *Boot record* : 主要是系统启动引导纪录，每一个逻辑分区都有，并占据其开始的512B空间　
   
2. *Super Block* : 纪录当前*ext2* 文件系统的基本信息，包括*Block Size*, *Block Groups*个数，每个*Block Group*的*Blocks*数量和*Blocks* 已使用数量等。每个*Block Group* 前1KB空间用来存储*Super Block*　信息。(除了*Block Group 0*必须有*Super Block*，其他可以不存储。并且*Block Group 0*总是从逻辑分区的第二个1KB开始)

3. *Block Group Descriptor Table* : 紧随*Super Block* 之后, 纪录每个*Block Group*的信息，包括每个*Group Block* 的第一个*Block Bitmap*, *Inode Bitmap*以及*Inode Table*的地址。q其大小由文件系统中*Block Groups*数量决定, 并且与*Super Block* 一样，在每个*Block Group*里都有备份。

4. *Block Bitmap*和*Inode Bitmap*　: 两者都是用来*bit*值来指示其对应的*Block*或*Inode*是否使用。其中*Block Bitmap*大小始终是1个*Data Block*。可以参见下面示意图:
![ext2 bitmap](/assets/images/ext2fs_bitmap.png)

5. *Inode Table* : 类似于*Block Group Descriptor Table*, 只是用来管理*Block Group*内部的*Inode*

了解上面一些概念之后，w可以参考下面示意图来理解*ext2*　是怎么保存`/etc/vim/vimrc`:
![ext2 file](/assets/images/ext2-file.png)

* 2001.12 *ext3[^6]* 取代*ext2*成为*Linux*
   文件系统，主要是加入日志功能，提高了系统的可靠性

* 2008.10 *ext4[^9]* 加入*Linux*， 相对于*ext3*，加入了一些新的特性， 比如：Delayed allocation， Journal checksumming， Extents等

* 2009.03 *Btrfs[^8]* 加入*Linux*, 增强了*ooling, snapshots,
   checksums*。


ZFS[^2]

# Btrfs特性

# 参考

* [Btrfs homepage](https://btrfs.wiki.kernel.org/index.php/Main_Page)  
* [Oracle - Btrfs Introduction](https://oss.oracle.com/projects/btrfs/dist/documentation/btrfs-ukuug.pdf)  
* [IBM - BtrFS Research](http://domino.research.ibm.com/library/cyberdig.nsf/papers/6E1C5B6A1B6EDD9885257A38006B6130/$File/rj10501.pdf)

[^1]: [wikipedia - ext](https://en.wikipedia.org/wiki/Extended_file_system)  
[^2]: [wikipedia - ZFS](https://en.wikipedia.org/wiki/ZFS)
[^3]: [wikipedia - UFS](https://en.wikipedia.org/wiki/Unix_file_system)
[^4]: [wikipedia - VFS](https://en.wikipedia.org/wiki/Virtual_file_system)
[^5]: [wikipedia - Ext2](https://en.wikipedia.org/wiki/Ext2)
[^6]: [wikipedia - Ext3](https://en.wikipedia.org/wiki/Ext3)
[^7]: [wikipedia - Ext3](https://en.wikipedia.org/wiki/Ext3)
[^8]: [wikipedia - Btrfs](https://en.wikipedia.org/wiki/Btrfs)
[^9]: [wikipedia - Ext4](https://en.wikipedia.org/wiki/Ext4)

