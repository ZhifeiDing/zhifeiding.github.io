---
title : Btrfs Introduction
categories : programming
tags : [data structure, file system, linux]
---

# 文件系统历史回顾

为了能够更好的理解*Btrfs*　的特性，下面首先回顾以一下*Linux* 上使用的各个文件系统历史及其特点：

* __1992.04__ *ext[^1]* 是第一个专门为*Linux* 设计的文件系统， 有*Unix File System(UFS)[^3]* 类似的*metedata* , 并且是第一个使用*VFS[^4]* 。解决了之前*Minix* 里64M最大容量和14个字符文件名的限制(支持2G容量和255个字符的文件名)，但是不支持单独时间戳的文件访问以及数据修改

* __1993.01__ *ext2[^5]* 基于*Berkely Fast File System* 同样的原则开发并取代*ext* 成为*Linux* 文件系统。基本单位是*Block*,　以*Inode*管理并组成*Block Group*。
   *ext2* 文件系统结构图如下所示：
![ext2 structure](/assets/images/ext2fs.png)

可以看到，*ext2* 文件系统主要由*Boot record* 和*Block Groups* 组成，而后者包含*Super block* , *Block Group Descriptor Table* , *Block Bitmap*, *Inode Bitmap*, *Inode table*, *Data blocks* , 其中：　

1. __Boot record__ : 主要是系统启动引导纪录，每一个逻辑分区都有，并占据其开始的512B空间　
2. __Super Block__ : 纪录当前*ext2* 文件系统的基本信息，包括*Block Size*, *Block Groups*个数，每个*Block Group*的*Blocks*数量和*Blocks* 已使用数量等。每个*Block Group* 前1KB空间用来存储*Super Block*　信息。(除了*Block Group 0*必须有*Super Block*，其他可以不存储。并且*Block Group 0*总是从逻辑分区的第二个1KB开始)
3. __Block Group Descriptor Table__ : 紧随*Super Block* 之后, 纪录每个*Block Group*的信息，包括每个*Group Block* 的第一个*Block Bitmap*, *Inode Bitmap*以及*Inode Table*的地址。q其大小由文件系统中*Block Groups*数量决定, 并且与*Super Block* 一样，在每个*Block Group*里都有备份。
4. __Block Bitmap__和__Inode Bitmap__　: 两者都是用来*bit*值来指示其对应的*Block*或*Inode*是否使用。其中*Block Bitmap*大小始终是1个*Data Block*。可以参见下面示意图:
![ext2 bitmap](/assets/images/ext2fs_bitmap.png)

5. __Inode Table__和__Inode__ : 类似于*Block Group Descriptor Table*, 只是用来管理*Block Group*内部的*Inode*和文件目录的查找。实际文件和目录自身读写属性，大小，类型等都保存在*Inode*里。每个*Inode*包含１２个直接指向的*Data Block*, 分别１个*single indirect Data Block, doubly indirect Data Block*和*triply indirect Data Block*。具体结构可参考下图：

了解上面一些概念之后，可以参考下面示意图来理解*ext2*　是怎么保存`/etc/vim/vimrc`:
![ext2 file](/assets/images/ext2-file.png)

* __2001.12__ *ext3[^6]* 取代*ext2*成为*Linux*文件系统，兼容*ext2*。主要是加入日志功能，提高了系统的可靠性，能够支持文件系统在线扩容，对大目录使用*HTree*进行索引。*ext3*支持三种日志级别：

1. __Journal__ : *metadata*和数据先纪录在日志之后再写入文件系统。这种风险最低，但是性能会有下降。
2. __Ordered__ : 日志里*metadata*在数据写入文件系统之后会更新。对于写或追加文件过程中发生故障，这种级别可以保证文件系统不会被破坏。但是对于修改文件过程中发生的故障，文件系统会被破坏。
3. __Writeback__ : 日志里的*metadata*不能保证是在数据写入文件系统之后还是之前更新。对于写或追加文件过程中发生的故障，可能日志里已经更新导致数据s被破坏。

*ext3*在实际中使用不多，主要是由于和*ext2*兼容，导致性能和特性没有优势，而且由于删除文件时会删掉文件的*Inode*而不支持*undelete*，不支持快照和日志没有效验等。

* __2008.10__ *ext4[^9]* 加入*Linux*， 对于*ext3*上述的一些缺点，*ext4* 加入了一些新的特性， 比如：*Delayed allocation*， *Journal checksumming*， *Extents* 等, 但仍然与*ext2*, *ext3*兼容，　整体结构上也是一致的。对于*defragmentation* 和*performance*等相对于其他新的文件系统仍然有差距。

* __2009.03__ 受*ZFS[^2]* 启发， *Btrfs[^8]* 加入*Linux*, 增强了*Snapshots*, *Checksums*等功能。与之前的*ext* 系列*Inode*不同的是， *Btrfs* 采用了*Copy-On-Write*的*B+Tree*的结构来索引和存储文件。

以上简单的回顾了*Linux*上文件系统的历史和结构， 下面重点讲解*Btrfs* 的一些特性和相应实现结构。

# *Btrfs* 特性

## *Btrfs* 使用的*B+Tree* 结构

  在[Implement-B+-Tree-with-C++](http://zhifeiding.github.io/programming/2016/08/01/Implement-B+-Tree-with-C++/)一文中我详细介绍了标准*B+-Tree* 的结构及其实现， 可以知道*B+-Tree* 的*leaf node*是组成一个链表的， 这种结构不利于*COW*, 所以*Btrfs* 相对于标准*B+-Tree* 去掉了*leaf-chaining*的结构。*Btrfs* 使用*B+-Tree* 结构如下所示:
![Btrfs_B+_Tree](/assets/images/Btrfs_B+_Tree.png)

## *Btrfs* 的整体结构
  
* 从整体上看，*Btrfs* 是由一系列的*B+-Tree* 组成的*Forest* 构成。类似的，在固定位置有一块*Superblock* 区域， 指向一个*tree of tree root*, 然后这个*tree of tree root* 可以索引组成文件系统的*B+-Tree*。 *Btrfs* 整体架构如下:

![Btrfs_full](/assets/images/Btrfs_full.png)

* 由上图可知，*Btrfs* 整体上是由如下*B+-Tree* 结构组成：

1. __Sub-volumes Tree__ : 是主要用来存储文件和目录的*B+-Tree*，每一个*sub-volume*都是一个单独的*Sub-Volume Tree*, 并且能够快照和克隆，使用*ref-counting*来记录对同一个*Extent*的引用。 所有*Sub-Volumes Tree*的根节点都可以被*tree of tree root* 索引。
2. __Extent Allocation Tree__ : 记录所有的分配的extents和剩余空间。可以用来移动*extents*或者从损坏德磁盘区域恢复文件系统。
3. __Checksum Tree__ : 对于每一个分配的*extent* 记录一个对应的数据和元数据的*Checksum*。
4. __Chunk and Device Tree__ : 物理设备操作层， 支持*RAID*。
5. __Reloc Tree__ : 移动*Extents*时特殊操作使用的。 当*defragmentation*时， *Btrfs*会克隆*Sub-Volume Tree*创建*Reloc Tree*， 然后移动*Extent*到新的连续的*Extent*, 最后将*Reloc Tree*合并到*Sub-Volume Tree*上。

## *Btrfs* 的读写改操作

* *Sub-Volume Tree*的*inner node*存储了`[key, block-pointer]`对， 而*leaf node*则存储了`[item,data]`对。 由于`data`的`size`是变化的，所以每一个*leaf node*在一个`block`的前段存储`item`,然后从`block`后端开始存储`data`。每个*leaf node*存储结构如下:

![leaf node](/assets/images/leaf_node.png)

* *Btrfs*读操作很简单就是对*Sub-Volume Tree*作简单的*B+-Tree*查找操作。
* 当修改文件或目录时会导致*extent*的更新， 然后由于使用__Copy-On-Write__, 更新会一直传导到*Sub-Volume Tree*的根节点。 同时也会导致*Extent Allocation Tree需要更新。而数据和元数据变化也会导致*Checksum Tree*需要同样更新操作。所有这些更新最后在*root*层就是产生一个新的*tree of tree root*。 *Btrfs*一个更新操作可以参考下图：
![Btrfs](/assets/images/Btrfs.png)

# 参考

以下是一些参考信息:
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

