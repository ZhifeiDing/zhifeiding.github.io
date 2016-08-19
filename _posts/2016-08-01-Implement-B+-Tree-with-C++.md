---
title : Implement B+ Tree with C++
categories : programming
tags : [data structure,c++，tree]
---

# 什么是*B+ Tree*？

说道*B+ Tree*不能不提*B Tree*， 所以下面先介绍*B Tree*结构和性质，再重点描述*B+ Tree*。

## *B Tree* 的结构和性质

*B Tree*和*BST*类似,也是一种有序的搜索树，不同点是*B Tree*是*M-ary*的:
* 每一个内部节点最多有*M-1*个*key*, *[M/2]*到*M*个*Children*
* 叶子节点能够存储*[(M-1)/2]*到*M-1*个*Key*，并且所有叶子节点都是同样的深度
* 根节点可以有2到*M*个*Children*或者没有*Children*作为叶子节点
* 每个节点内*Key*从左到右递增，且大于左子树所有节点的*Key*，小于右子树所有节点的*Key*

可以发现当*M = 2*时*B Tree*就退化成一个平衡的*BST*了。当*M=3*时内部节点只能有2或者3个*Children*,也被称为*[2-3-tree](https://en.wikipedia.org/wiki/2%E2%80%933_tree)*。下图是一个典型的*M=5*的*B Tree*结构图：
![B Tree](/assets/images/BTree.png)

虽然相对于*BST*来说，每次*IO*读写能够操作的数据更多了，但是由于内部节点也存储了数据导致数据很大时不能将整个*B Tree*放进主存或缓存里，对*IO*操作仍然不友好。 所以就出现了下面的*B+ Tree*。

## *B+ Tree* 的结构和性质

*B+ Tree*除了内部节点不存储数据外，和*B Tree*很类似：
* 每一个内部节点最多有*M-1*个*key*, *[M/2]*到*M*个*Children*，而且不存储对应的数据
* 叶子节点能够存储*[L/2]*到*L*个*Key*，同时还有对应的数据，并且所有叶子节点都是同样的深度
* 根节点可以有2到*M*个*Children*或者没有*Children*作为叶子节点
* 每个节点内*Key*从左到右递增，且大于左子树所有节点的*Key*，小于或等于右子树所有节点

相比于*B Tree*，可以发现如果*L >> M*，大部份*Key*都在叶子节点，*B+ Tree*可以将整个*B+ Tree*放进主存或缓存里。下图是一个典型的当*M = 5 L = 4*的*B+ Tree*结构图：
![B+ Tree](/assets/images/B+Tree.png)

# *B+ Tree* 实现

## *B+ Tree*叶子节点和内部节点结构

```cpp
```

## 构造*B+ Tree*类

```cpp
```

## *B+ Tree* 查找算法实现

```cpp
```

## *B+ Tree* 插入算法实现

以上面的*M = 5 L = 4*的*B+ Tree*为例：
* 当插入*16*时，查找到*LeafNode*有3个*key*， 插入之后是4个*key*，满足*LeafNode*要求，所以直接插入就可
![B+ Tree InsertLeaf](/assets/images/B+Tree_InsertLeaf.png)
* 当我们继续插入*19*之后，*LeafNode*的*Key = 5 > L = 4*，所以我们插入之后需要将这个*LeafNode*进行*Split*
![B+ Tree InsertLeaf](/assets/images/B+Tree_InsertSplit0.png)
*Split*方法是将*LeafNode*从*[L/2]*处分成左右两个*LeafNode*， 并将*[L/2]*处*Key*传递到父节点，父节点重复上面的插入动作
![B+ Tree InsertLeaf](/assets/images/B+Tree_InsertSplit1.png)
* 当我们向下面*B+ Tree*插入*9*时，*LeafNode*需要*Split*
![B+ Tree InsertLeaf](/assets/images/B+Tree_InsertSplit2.png)
同时父节点插入传递上来的*5*之后也需要*Split*，同时传递*10*到根节点
![B+ Tree InsertLeaf](/assets/images/B+Tree_InsertSplit3.png)

```cpp
```

## *B+ Tree* 删除算法实现

同样以上面插入之后的*M = 5 L = 4*的*B+ Tree*为例：
* 当删除*18*时，*18*所在*LeafNode*有3个*Key*，删掉之后*Key = 2 >= L/2 = 2*,所以直接删掉*18*即可
![B+ Tree InsertLeaf](/assets/images/B+Tree_DeleteLeaf.png)
* 当继续删除*16*时，*LeafNode*只有1个*Key*了，这时我们首先需要尝试从*Neighbor*吸收元素。但是左右*Neighbor*都只有*L/2 = 2*个*Key*，我们只能将当前*LeafNode*与*Neighbor*合并并更新父节点
![B+ Tree InsertLeaf](/assets/images/B+Tree_DeleteMerge0.png)
更新父节点之后发现父节点*Key = 1 < [L/2]*,需要重复上面的删除动作(这儿也只能和*Neighbor*合并)
![B+ Tree InsertLeaf](/assets/images/B+Tree_DeleteMerge1.png)
合并之后需要将父节点的*10*传递下来，最终树结构如下图：
![B+ Tree InsertLeaf](/assets/images/B+Tree_DeleteMerge2.png)
* 当继续删除*17*时，需要从*Neighbor*吸收元素15，同时更新父节点
![B+ Tree InsertLeaf](/assets/images/B+Tree_DeleteMerge3.png)

```cpp
```

# 测试程序

```cpp
```

# 参考

* [wikipedia - B+ Tree](https://en.wikipedia.org/wiki/B%2B_tree)
* [B+Tree index structures in InnoDB](https://blog.jcole.us/2013/01/10/btree-index-structures-in-innodb/)
* [The physical structure of InnoDB index pages](https://blog.jcole.us/2013/01/07/the-physical-structure-of-innodb-index-pages/)
* [B+ Tree Course of Washington University](http://courses.cs.washington.edu/courses/cse326/08sp/lectures/11-b-trees.pdf)
* [B+ Tree Visualization](https://www.cs.usfca.edu/~galles/visualization/BPlusTree.html)
* [B+ Tree Insertion and Deletion](http://www.cburch.com/cs/340/reading/btree/index.html)
