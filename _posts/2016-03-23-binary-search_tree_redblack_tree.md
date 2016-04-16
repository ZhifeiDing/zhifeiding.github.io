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
// TreeNode<T> definition
template<typename T>
class TreeNode {
public:
    T val;
    TreeNode<T> *left;
    TreeNode<T> *right;
    TreeNode<T> *parent;
    // constructor
    TreeNode() : left(nullptr), right(nullptr), parent(nullptr){};
    TreeNode(T v) : val(v), left(nullptr), right(nullptr), parent(nullptr){};
    TreeNode(T v, TreeNode<T> *l, TreeNode *r, TreeNode *p = nullptr) : val(v), left(l), right(r),parent(p){};
};


// Binary search Tree declaration
template<typename T>
class BST {
public:
    // default constructor
    explicit BST();
    // copy constructor
    BST(const BST &other);
    // assignment constructor
    const BST<T>& operator=(const BST &other);
    // destructor
    ~BST();

    // STL-style iterator
    class iterator {
    public:
        friend class BST;
        explicit iterator();
        const iterator& operator=(const iterator &other); // assignment constructor
        iterator& operator++(); // prefix increment
        iterator operator++(int); // postfix increment
        T& operator*() const;
        bool operator!=(const BST<T>::iterator &other) const;
        bool operator==(const BST<T>::iterator &other) const;
        // for iterator_traits to refer
        typedef std::output_iterator_tag iterator_category;
        typedef T value_type;
        typedef std::ptrdiff_t difference_type;
        typedef T* pointer;
        typedef T& reference;
    private:
        iterator(TreeNode<T>* n);
        TreeNode<T>* node;
        TreeNode<T>* lastNode;
    };

    // iterator begin() and end()
    iterator begin() const;
    iterator end() const;

    // insert an element into BST
    void insert(T v);

    // find an element
    iterator find(const T& v) const;
    //const_iterator find(const T& v) const;

    // remove one element
    void erase(iterator itr);

    // return the size of BST
    const std::size_t size() const;

    // check whether the BST is nullptr
    const bool empty() const;

private:
    std::size_t cnt;
    TreeNode<T> *root;

    // private insert function
    void insert(T v, TreeNode<T> *&r, TreeNode<T> * const &p = nullptr);
    // private : replace node in parent
    void replace_node_in_parent(TreeNode<T> *node, TreeNode<T> *newNode = nullptr);
    // private : find the max value node
    TreeNode<T> *findMax(TreeNode<T> *node);
};
```


为了模拟STL容器类，在`BST<T>`里我们实现了inner class
`iterator`来提供iterator类的支持。`iterator`类的构造函数和成员函数实现如下:

```cpp
// BST<T>::iterator
// default constructor
template<typename T>
BST<T>::iterator::iterator() {
    node = nullptr;
    lastNode = nullptr;
}

// one argument constructor
template<typename T>
BST<T>::iterator::iterator(TreeNode<T>* n) {
    node = n;
    lastNode = nullptr;
}

// assignment constructor
template<typename T>
const typename BST<T>::iterator& BST<T>::iterator::operator=(const iterator &other) {
    this->node = other.node;
    return *this;
}

template<typename T>
T& BST<T>::iterator::operator*() const {
    return this->node->val;
}

// overload operator ==
template<typename T>
bool BST<T>::iterator::operator==(const BST<T>::iterator &other) const {
    return this->node == other.node;
}

// overload operator !=
template<typename T>
bool BST<T>::iterator::operator!=(const BST<T>::iterator &other ) const {
    return this->node != other.node;
}

// overload prefix ++
template<typename T>
typename BST<T>::iterator& BST<T>::iterator::operator++() {
    // if current node is root , we could get the lastNode
    if( node->parent == nullptr ) {
        lastNode = node;
        while( lastNode->right )
            lastNode = lastNode->right;
    }
    if( node->right != nullptr ) {
        node = node->right;
        while( node->left )
            node = node->left;
    } else if( node == lastNode ){
        node = nullptr;
    } else if( node == node->parent->left ) {
        node = node->parent;
    } else if( node == node->parent->right ) {
        while( node->parent->val <= node->val )
            node = node->parent;
        node = node->parent;
    }
    return *this;
}

// overload postfix ++
template<typename T>
typename BST<T>::iterator BST<T>::iterator::operator++(int) {
    // if current node is root , we could get the lastNode
    if( node->parent == nullptr ) {
        lastNode = node;
        while( lastNode->right )
            lastNode = lastNode->right;
    }
    TreeNode<T> *p = this->node;
    if( node->right != nullptr ) {
        node = node->right;
        while( node->left )
            node = node->left;
    } else if( node == lastNode ){
        node = nullptr;
    } else if( node == node->parent->left ) {
        node = node->parent;
    } else if( node == node->parent->right ) {
        while( node->parent->val <= node->val )
            node = node->parent;
        node = node->parent;
    }
    return iterator(p);
}
```

* `BST`类构造函数

```cpp
// default constructor -- only initialize the private variable
template<typename T>
BST<T>::BST() {
    cnt = 0;
    root = nullptr;
}

// copy constructor -- deep copy every element of the other BST
template<typename T>
BST<T>::BST(const BST &other) {
    cnt = other.cnt;
    if( cnt == 0 )
        return;
    std::queue<std::pair<TreeNode<T>*, TreeNode<T>*> > q;
    q.push(std::make_pair(root(other.root->val),other.root));
    while( !q.empty() ) {
        TreeNode<T> *node1 = q.front().first;
        TreeNode<T> *node2 = q.front().second;
        q.pop();
        if( node2->left ) {
            node1->left = new TreeNode<T>(node2->left->val);
            node1->left->parent = node1;
            q.push(make_pair(node1->left, node2->left));
        }
        if( node2->right ) {
            node1->right = new TreeNode<T>(node2->right->val);
            node1->irght->parent = node1;
            q.push(make_pair(node1->right, node2->right));
        }
    }
}

// assignment constructor
template<typename T>
const BST<T>& BST<T>::operator=(const BST &other) {
    cnt = other.cnt;
    if( cnt == 0 )
        return *this;
    std::queue<std::pair<TreeNode<T>*, TreeNode<T>*> > q;
    q.push(std::make_pair(root(other.root->val),other.root));
    while( !q.empty() ) {
        TreeNode<T> *node1 = q.front().first;
        TreeNode<T> *node2 = q.front().second;
        q.pop();
        if( node2->left ) {
            node1->left = new TreeNode<T>(node2->left->val);
            node1->left->parent = node1;
            q.push(make_pair(node1->left, node2->left));
        }
        if( node2->right ) {
            node1->right = new TreeNode<T>(node2->right->val);
            node1->right->parent = node1;
            q.push(make_pair(node1->right, node2->right));
        }
    }
    return *this;
}

// destructor
template<typename T>
BST<T>::~BST() {
    if( cnt == 0 )
        return ;
    cnt = 0;
    std::queue<TreeNode<T>*> q;
    q.push(root);
    while( !q.empty() ) {
        root = q.front();
        q.pop();
        if( root->left ) {
            q.push(root->left);
        }
        if( root->right ) {
            q.push(root->right);
        }
        delete root;
    }
}
```

* `begin()`及`end()`函数

```cpp
// iterator begin() and end()
template<typename T>
typename BST<T>::iterator BST<T>::begin() const {
    TreeNode<T>* node = root;
    while( node && node->left )
        node = node->left;
    return iterator(node);
}

template<typename T>
typename BST<T>::iterator BST<T>::end() const {
    return iterator();
}

```

* `size()`及`empty()`函数

```cpp
// return the size of BST
template<typename T>
const std::size_t BST<T>::size() const {
    return cnt;
}

// check whether the BST is nullptr
template<typename T>
const bool BST<T>::empty() const {
    return cnt == 0;
}

```

* 实现`insert(T v)`来插入一个新的数据

```cpp
// insert an element into BST
template<typename T>
void BST<T>::insert(T v) {
    insert(v,root);
    // increase cnt
    ++cnt;
}

// private : insert an element into BST
template<typename T>
void BST<T>::insert(T v, TreeNode<T> *&r, TreeNode<T> * const &p) {
    if( r == nullptr )
        r = new TreeNode<T>(v, nullptr, nullptr, p);
    else if( v < r->val )
        insert(v, r->left, r);
    else
        insert(v, r->right,r);
}

```

* `find(T v)`查找数据是否存在

```cpp
// find an element
template<typename T>
typename RBT<T>::iterator RBT<T>::find(const T& v) const {
    TreeNode<T> *p = root;
    while( p ) {
        if( p->val == v )
            return iterator(p);
        if( p->val < v )
            p = p->right;
        else
            p = p->left;
    }
    return end();
}
```

* `erase(iterator itr)`删除一个数据，可以分为三种情况：

> * 要删除数据节点没有左右子树，这种情况我们可以直接删掉该数据
  * 要删除的数据只有左子树或右子树，这种情况下我们可以用左子树或右子树来替换要删掉的节点
  * 要删掉的数据节点存在左右子树，
  ![erase](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Binary_search_tree_delete.svg/320px-Binary_search_tree_delete.svg.png)

```cpp
// remove one element
template<typename T>
void BST<T>::erase(iterator itr) {
    --cnt;
    if( cnt == 0 ) {
        root = nullptr;
        return;
    }
    // if left & right child both exist
    if( (itr.node)->left && (itr.node)->right ) {
        TreeNode<T> *predecessor = findMax((itr.node)->left);
        (itr.node)->val = predecessor->val;
        erase(iterator(predecessor));
    // if only left child exist
    } else if( (itr.node)->left ) {
        replace_node_in_parent(itr.node, (itr.node)->left);
    // if only right child exist
    } else if( (itr.node)->right ) {
        replace_node_in_parent(itr.node, (itr.node)->right);
    // if no child
    } else {
        replace_node_in_parent(itr.node);
    }
}

template<typename T>
void BST<T>::replace_node_in_parent(TreeNode<T> *node, TreeNode<T> *newNode) {
    if( node->parent ) {
        if( node->parent->left == node )
            node->parent->left = newNode;
        else
            node->parent->right = newNode;
    } else {
        root = newNode;
    }
    if( newNode )
        newNode->parent = node->parent;
}

template<typename T>
TreeNode<T> *BST<T>::findMax(TreeNode<T> *node) {
    while( node && node->right )
        node = node->right;
    return node;
}

```

# __Red-Black Tree__

## 什么是*Red-Black Tree* ?

上面说过， 在最坏情况下，`BST`会退化成`link-list`，其时间复杂度是`O(n)`。因此为了保证`O(logn)`的时间复杂度，需要设计一种能够使Tree保持平衡的机制。这儿的*Red-Black Tree*就是一种*自平衡的二叉查找树*。
Red-Black Tree是符合下面定义的二叉树:

> 1. 任意节点值比其左子树所有节点值大，比右子树所有节点小。(BST的定义)
  2. 任意节点要么是黑色要么是红色
  3. 根节点必须是黑色
  4. 所有的叶子节点(nil)都是黑色
  5. 如果一个节点是红色，则其左右子节点必须是黑色
  6. 从任意节点到叶子节点(nil)所有path上的黑色节点个数相同

![RBT](https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Red-black_tree_example.svg/320px-Red-black_tree_example.svg.png)

上面说满足这些约束的就是*Red-Black Tree*，是能够保证`O(logn)`最坏时间复杂度的自平衡的二叉查找树。真的是这样吗？假设从一个节点到叶子节点的path上黑色节点个数(不算叶子节点)为`B`,则最长path上能够有的最多的节点个数为`2*B`(黑红节点交叉)， 而最短的path上的节点最少也要有`B`个(全是黑色节点)。因此对于一个*Red-Black Tree*, 没有path会比其他path长两倍。这就保证了*Red-Black Tree*的高度大致保持平衡，而*Red-Black Tree*上各种操作和高度是

## *Red-Black Tree*的实现

既然*Red-Black Tree*也是二叉查找树， 自然能够在*BST*上实现的操作必然能够在*Red-Black Tree*上实现:

* 插入, 有插入数据我们才能进行其他操作
* 删除, 删除不需要的值
* 查找, 由于BST的性质， 我们知道在BST上可以实现二分查找
* 遍历, 对于BST来讲，就是`in-order-traversal`了

查找和遍历由于没有写操作，和*BST*上相关操作一致即可。而插入和删除由于改变了树可能会违反*Red-Black Tree*约束，需要一些额外操作来使树重新满足约束。

## 代码

* 首先， 我们声明一个RBT类,
  包含上面所描述的四种操作。我们使用`TreeNode`作为RBT的节点,和上面的`BST<T>`类似。

```cpp
```

* 实现`insert(T v)`来插入一个新的数据

```cpp
```

![insert case3](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Red-black_tree_insert_case_3.svg/320px-Red-black_tree_insert_case_3.svg.png)

![insert case4](https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Red-black_tree_insert_case_4.svg/320px-Red-black_tree_insert_case_4.svg.png)

![insert case5](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Red-black_tree_insert_case_5.svg/320px-Red-black_tree_insert_case_5.svg.png)

* `find(T v)`查找数据是否存在

```cpp
```

* `erase(iterator itr)`删除数据

```cpp
```

![delete case2](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Red-black_tree_delete_case_2_as_svg.svg/320px-Red-black_tree_delete_case_2_as_svg.svg.png)

![delete case3](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Red-black_tree_delete_case_3_as_svg.svg/320px-Red-black_tree_delete_case_3_as_svg.svg.png)

![delete case4](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Red-black_tree_delete_case_4_as_svg.svg/320px-Red-black_tree_delete_case_4_as_svg.svg.png)

![delete case5](https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Red-black_tree_delete_case_5_as_svg.svg/320px-Red-black_tree_delete_case_5_as_svg.svg.png)

![delete case6](https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Red-black_tree_delete_case_6_as_svg.svg/320px-Red-black_tree_delete_case_6_as_svg.svg.png)

# 测试程序

```cpp
```

# 参考

1.[wikipedia - binary search tree](https://en.wikipedia.org/wiki/Binary_search_tree)
2.[wikipedia - red black tree](https://en.wikipedia.org/wiki/Red%E2%80%93black_tree)
