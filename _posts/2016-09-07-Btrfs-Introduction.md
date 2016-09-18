---
title : Btrfs Introduction
categories : programming
tags : [data structure]
---

# 文件系统历史回顾

下面主要回顾*Linux* 上使用的文件系统：

1. 1992.04 *ext[^1]* 是第一个专门为*Linux* 设计的文件系统， 有*Unix File System
   (UFS)[^3]* 类似的*metedata* , 并且是第一个使用*VFS[^4]* 。解决了之前*Minix*
   里64M
   最大容量和14个字符文件名的限制(支持2G容量和255个字符的文件名)，但是不支持单独时间戳的文件访问以及数据修改

2. 1993.01 *ext2[^5]* 基于*Berkely Fast File System* 同样的原则开发并取代*ext*
   成为*Linux* 文件系统。

3. 2001.12 *ext3[^6]* 取代*ext2*成为*Linux*
   文件系统，主要是加入日志功能，提高了系统的可靠性

4. 2008.10 *ext4[^9]* 加入*Linux*， 相对于*ext3*，加入了一些新的特性， 比如：Delayed allocation， Journal checksumming， Extents等

5. 2009.03 *Btrfs[^8]* 加入*Linux*, 增强了*ooling, snapshots,
   checksums*。


ZFS[^2]

# Btrfs特性

# 参考

* [Btrfs homepage](https://btrfs.wiki.kernel.org/index.php/Main_Page)  
* [IBM - 新一代 Linux 文件系统 btrfs 简介](http://www.ibm.com/developerworks/cn/linux/l-cn-btrfs/index.html)  

[^1]: [wikipedia - ext](https://en.wikipedia.org/wiki/Extended_file_system)  
[^2]: [wikipedia - ZFS](https://en.wikipedia.org/wiki/ZFS)  
[^3]: [wikipedia - UFS](https://en.wikipedia.org/wiki/Unix_file_system)  
[^4]: [wikipedia - VFS](https://en.wikipedia.org/wiki/Virtual_file_system)  
[^5]: [wikipedia - Ext2](https://en.wikipedia.org/wiki/Ext2)  
[^6]: [wikipedia - Ext3](https://en.wikipedia.org/wiki/Ext3)  
[^7]: [wikipedia - Ext3](https://en.wikipedia.org/wiki/Ext3)  
[^8]: [wikipedia - Btrfs](https://en.wikipedia.org/wiki/Btrfs)  
[^9]: [wikipedia - Ext4](https://en.wikipedia.org/wiki/Ext4)  

