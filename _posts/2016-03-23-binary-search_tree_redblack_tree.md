---
title : Binary Search Tree and RedBlack Tree
category : programming
tags : [c++,data structure]
---

# Binary Search Tree

## 什么是BST ?

Binary Search Tree是符合下面定义的二叉树:

> 任意节点值比其左子树所有节点值大，比右子树所有节点小。

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
