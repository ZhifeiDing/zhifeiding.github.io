---
title : Binary Search Tree and RedBlack Tree
category : programming
tags : [c++,data structure]
---

# Binary Search Tree

## 什么是BST ?

Binary Search Tree是符合下面定义的二叉树:

> 任意节点值比其左子树所有节点值大，比右子树所有节点小。

![BST](https://upload.wikimedia.org/wikipedia/commons/d/da/Binary_search_tree.svg)

对于BST的叶子节点，定义其为`nil`。根据BST的性质，
我们可知`in-order-traversal`可以得到BST的一个有序数据集合。并且n个节点组成平衡BST高度为`O(logn)`，则对其的操作时间复杂度是`O(logn)`。

### BST的缺点

上面说了对于平衡BST，其操作时间复杂度为`O(logn)`，可是如果BST不是平衡的呢，比如一个只包含右子树的BST,我们知道其时间复杂度是`O(n)`。因此，BST没法保证`O(logn)`的时间复杂度。

## BST的实现

首先我们需要知道BST上要实现的操作：

* 插入, 有插入数据我们才能进行其他操作
* 查找, 由于BST的性质， 我们知道在BST上可以实现二分查找
* 删除, 删除不需要的值
* 遍历, 对于BST来讲，就是`in-order-traversal`了

## 代码

* 首先， 我们声明一个BST类,
  包含上面所描述的四种操作。我们使用`TreeNode`作为BST的节点

```cpp
```

* 实现`insert(T v)`来插入一个新的数据

```cpp
```

* `find(T v)`查找数据是否存在

```cpp
```

# __Red-Black Tree__

## 什么是*Red-Black Tree* ?

上面说过， 在最坏情况下，__BST__会退化成`link-list`，其时间复杂度是`O(n)`。因此为了保证`O(logn)`的时间复杂度，需要设计一种能够使Tree保持平衡的机制。这儿的*Red-Black Tree*就是一种*自平衡的二叉查找树*。
Red-Black Tree是符合下面定义的二叉树:

> 1. 任意节点值比其左子树所有节点值大，比右子树所有节点小。(BST的定义)   
  2. 任意节点要么是黑色要么是红色  
  3. 根节点必须是黑色  
  4. 所有的叶子节点(nil)都是黑色  
  5. 如果一个节点是红色，则其左右子节点必须是黑色  
  6. 从任意节点到叶子节点(nil)所有path上的黑色节点个数相同  
  
![RBT](https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Red-black_tree_example.svg/320px-Red-black_tree_example.svg.png)
  
上面说满足这些约束的就是*Red-Black Tree*，是能够保证`O(logn)`最坏时间复杂度的自平衡的二叉查找树。真的是这样吗？假设从一个节点到叶子节点的path上黑色节点个数(不算叶子节点)为`B`,则最长path上能够有的最多的节点个数为`2*B`(黑红节点交叉)， 而最短的path上的节点最少也要有`B`个(全是黑色节点)。因此对于一个*Red-Black Tree*, 没有path会比其他path长两倍。这就保证了*Red-Black Tree*的高度大致保持平衡，而*Red-Black Tree*上各种操作和高度是

### *Red-Black Tree*的缺点

上面说了对于平衡BST，其操作时间复杂度为`O(logn)`，可是如果BST不是平衡的呢，比如一个只包含右子树的BST,我们知道其时间复杂度是`O(n)`。因此，BST没法保证`O(logn)`的时间复杂度。

## *Red-Black Tree*的实现

既然*Red-Black Tree*也是二叉查找树， 自然能够在*BST*上实现的操作必然能够在*Red-Black Tree*上实现:

* 插入, 有插入数据我们才能进行其他操作
* 删除, 删除不需要的值
* 查找, 由于BST的性质， 我们知道在BST上可以实现二分查找
* 遍历, 对于BST来讲，就是`in-order-traversal`了

查找和遍历由于没有写操作，和*BST*上相关操作一致即可。而插入和删除由于改变了树可能会违反*Red-Black Tree*约束，需要一些额外操作来使树重新满足约束。

## 代码

* 首先， 我们声明一个BST类,
  包含上面所描述的四种操作。我们使用`TreeNode`作为BST的节点

```cpp
```

* 实现`insert(T v)`来插入一个新的数据

```cpp
```

* `find(T v)`查找数据是否存在

```cpp
```
