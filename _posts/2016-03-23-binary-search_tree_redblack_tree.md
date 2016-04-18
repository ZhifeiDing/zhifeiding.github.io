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
    int cnt;
    TreeNode<T> *root;

    // private insert function
    void insert(T v, TreeNode<T> *&r, TreeNode<T> * const &p = nullptr);
    // private : replace node in parent
    void replace_node_in_parent(TreeNode<T> *node, TreeNode<T> *newNode = nullptr);
    // private : find the max value node
    TreeNode<T> *findMax(TreeNode<T> *node);
};
```


* 为了模拟STL容器类，在`BST<T>`里我们实现了inner class
`iterator`来提供iterator类的支持。`iterator`类的构造函数和成员函数实现如下:

> `iterator`无参数构造函数， 将内部指针设为`nullptr`即可

```cpp
// BST<T>::iterator
// default constructor
template<typename T>
BST<T>::iterator::iterator() {
    node = nullptr;
    lastNode = nullptr;
}
```

> `iterator`一个参数构造函数，将内部指针指向该参数指向节点即可

```cpp
// one argument constructor
template<typename T>
BST<T>::iterator::iterator(TreeNode<T>* n) {
    node = n;
    lastNode = nullptr;
}
```

> `iterator`赋值构造函数

```cpp
// assignment constructor
template<typename T>
const typename BST<T>::iterator& BST<T>::iterator::operator=(const iterator &other) {
    this->node = other.node;
    return *this;
}
```

> `iterator`类`operator*`重载，返回内部指针指向node的值

```cpp
template<typename T>
T& BST<T>::iterator::operator*() const {
    return this->node->val;
}
```

> `iterator`类`operator==`重载, 比较内部node是否相同

```cpp
// overload operator ==
template<typename T>
bool BST<T>::iterator::operator==(const BST<T>::iterator &other) const {
    return this->node == other.node;
}
```

> `iterator`类`operator!=`重载, 比较内部node是否相同

```cpp
// overload operator !=
template<typename T>
bool BST<T>::iterator::operator!=(const BST<T>::iterator &other ) const {
    return this->node != other.node;
}
```

> `iterator`类`operator++`重载, 返回类型是引用，因此是前缀`++`。对于`BST<T>`，实际就是实现`in order traversal`。借助于`parent`指针可以不借用`stack`。

```cpp
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
```

> `iterator`类`operator++`重载, 返回类型是值，因此是后缀`++`。与前缀`++`类似，实际就是实现`in order traversal`。

```cpp
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

> `BST<T>`无参数构造函书

```cpp
// default constructor -- only initialize the private variable
template<typename T>
BST<T>::BST() {
    cnt = 0;
    root = nullptr;
}
```

> `BST<T>`复制构造函数，需要实现*deep copy*。借助`queue`,使用`bfs`来复制`BST<T>`中每一个node

```cpp
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
```

> `BST<T>`赋值构造函数，与复制构造函数一样，只是需要返回当前对象的引用

```cpp
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
```

> `BST<T>`析构函数，当对象不需要时，释放节点空间。与上面复制构造函数类似，也是借助`queue`来实现`bfs`。

```cpp
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

> `BST<T>`类`begin()`函数，返回`BST<T>`最左节点的`iterator`实例

```cpp
// iterator begin() and end()
template<typename T>
typename BST<T>::iterator BST<T>::begin() const {
    TreeNode<T>* node = root;
    while( node && node->left )
        node = node->left;
    return iterator(node);
}
```

> `BST<T>`类`end()`函数，返回`null iterator`实例

```cpp
template<typename T>
typename BST<T>::iterator BST<T>::end() const {
    return iterator();
}
```

* `size()`及`empty()`函数

> `size()`函数返回当前`BST<T>`中元素个数

```cpp
// return the size of BST
template<typename T>
const std::size_t BST<T>::size() const {
    return cnt;
}
```

> `empty()`函数返回`true`当`BST<T>`中不存在元素时 

```cpp
// check whether the BST is nullptr
template<typename T>
const bool BST<T>::empty() const {
    return cnt == 0;
}
```

* 实现`insert(T v)`来插入一个新的数据

> 作为`BST<T>`的接口函数，插入新元素计数变量加一，然后调用`private insert(T v, TreeNode<T> *r)`函数

```cpp
// insert an element into BST
template<typename T>
void BST<T>::insert(T v) {
    insert(v,root);
    // increase cnt
    ++cnt;
}
```

> 对于`BST<T>`来说，我们需要插入一个新的元素最简单的方法就是通过二分查找，找到一个`leaf node`，然后把当前值作为左节点或右节点。

```cpp
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

>
* 要删除数据节点没有左右子树，这种情况我们可以直接删掉该数据  
* 要删除的数据只有左子树或右子树，这种情况下我们可以用左子树或右子树来替换要删掉的节点  
* 要删掉的数据节点存在左右子树，我们可以用`in-order traversal`的`predecessor`或`postdecessor`来替换要删掉的node,然后删除`predecessor`或`postdecessor`，这样就回到上面的情况了  
  ![erase](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Binary_search_tree_delete.svg/320px-Binary_search_tree_delete.svg.png)

```cpp
// remove one element
template<typename T>
void BST<T>::erase(iterator itr) {
    --cnt;
    if( cnt == 0 ) {
        delete root;
        root = nullptr;
        return;
    }
    // if left & right child both exist
    if( (itr.node)->left && (itr.node)->right ) {
        TreeNode<T> *predecessor = findMax((itr.node)->left);
        (itr.node)->val = predecessor->val;
        erase(iterator(predecessor));
    // if only left child exist
    } else {
        TreeNode<T> *child = (itr.node)->left ? (itr.node)->left : (itr.node)->right;
        replace_node_in_parent(itr.node, child);
        delete itr.node;
    }
}
```

> 当要删除节点有左子树或右子树时，我们用其子树替换掉该节点。当该节点是`root`节点时，要更新`root`节点到新的节点

```cpp
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
```

> 当要删除节点同时存在有左子树和右子树时，我们找到`in order traversal`的`predecessor`来替换该节点

```cpp
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
// Binary search Tree declaration
template<typename T>
class RBT {
public:
    // default constructor
    explicit RBT();
    // copy constructor
    RBT(const RBT &other);
    // assignment constructor
    const RBT<T>& operator=(const RBT &other);
    // destructor
    ~RBT();

    enum Color { RED, BLACK };
    // TreeNode definition
    class TreeNode {
    public:
        friend class RBT;
        T val;
        TreeNode *left;
        TreeNode *right;
        TreeNode *parent;
        // indicate this node's color
        Color color;
        // constructor
        TreeNode() : left(nullptr), right(nullptr), parent(nullptr), color(BLACK){};
        TreeNode(T v) : val(v), left(nullptr), right(nullptr), parent(nullptr), color(RED){};
        TreeNode(T v, TreeNode *l, TreeNode *r, TreeNode *p = nullptr, Color c = RED) : val(v), left(l), right(r),parent(p), color(c){};
    };


    // STL-style iterator
    class iterator {
    public:
        friend class RBT;
        explicit iterator();
        const iterator& operator=(const iterator &other); // assignment constructor
        iterator& operator++(); // prefix increment
        iterator operator++(int); // postfix increment
        T& operator*() const; // derefence the pointer
        bool operator!=(const RBT<T>::iterator &other) const;
        bool operator==(const RBT<T>::iterator &other) const;
        // for iterator_traits to refer
        typedef std::output_iterator_tag iterator_category;
        typedef T value_type;
        typedef std::ptrdiff_t difference_type;
        typedef T* pointer;
        typedef T& reference;
    private:
        iterator(TreeNode* n);
        TreeNode* node;
        TreeNode* lastNode;
    };

    // iterator begin() and end()
    iterator begin() const;
    iterator end() const;

    // insert an element into RBT
    void insert(T v);

    // find an element
    iterator find(const T& v) const;
    //const_iterator find(const T& v) const;

    // remove one element
    void erase(iterator itr);

    // return the size of RBT
    const std::size_t size() const;

    // check whether the RBT is nullptr
    const bool empty() const;

private:

    int cnt;
    TreeNode *root;

    // private insert helper function
    void insert(T v, TreeNode *&r, TreeNode * const &p = nullptr);
    void insert_case1(TreeNode *r);
    void insert_case2(TreeNode *r);
    void insert_case3(TreeNode *r);
    void insert_case4(TreeNode *r);
    void insert_case5(TreeNode *r);
    TreeNode* grandParent(TreeNode * const &r) const;
    TreeNode* uncle( TreeNode * const &r) const;
    TreeNode* sibling( TreeNode * const &r) const;
    // private : replace node in parent
    void replace_node_in_parent(TreeNode *node, TreeNode *newNode = nullptr);

    // delete helper functions
    void delete_case1(TreeNode *node);
    void delete_case2(TreeNode *node);
    void delete_case3(TreeNode *node);
    void delete_case4(TreeNode *node);
    void delete_case5(TreeNode *node);
    void delete_case6(TreeNode *node);

    // rotate the node
    void rotate_right(TreeNode *r);
    void rotate_left(TreeNode *r);
    // private : find the max value node
    TreeNode *findMax(TreeNode *node);
};
```

* `RBT<T>::iterator`实现, 这一部分与`BST`一样

```cpp
// RBT<T>::iterator
// default constructor
template<typename T>
RBT<T>::iterator::iterator() {
    node = nullptr;
    lastNode = nullptr;
}

// one argument constructor
template<typename T>
RBT<T>::iterator::iterator(TreeNode* n) {
    node = n;
    lastNode = nullptr;
}

// assignment constructor
template<typename T>
const typename RBT<T>::iterator& RBT<T>::iterator::operator=(const iterator &other) {
    this->node = other.node;
    return *this;
}

template<typename T>
T& RBT<T>::iterator::operator*() const {
    return this->node->val;
}

// overload operator ==
template<typename T>
bool RBT<T>::iterator::operator==(const RBT<T>::iterator &other) const {
    return this->node == other.node;
}

// overload operator !=
template<typename T>
bool RBT<T>::iterator::operator!=(const RBT<T>::iterator &other ) const {
    return this->node != other.node;
}

// overload prefix ++
template<typename T>
typename RBT<T>::iterator& RBT<T>::iterator::operator++() {
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
typename RBT<T>::iterator RBT<T>::iterator::operator++(int) {
    // if current node is root , we could get the lastNode
    if( node->parent == nullptr ) {
        lastNode = node;
        while( lastNode->right )
            lastNode = lastNode->right;
    }
    TreeNode *p = this->node;
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

* `RBT<T>`无参数构造函数

```cpp
// default constructor -- only initialize the private variable
template<typename T>
RBT<T>::RBT() {
    cnt = 0;
    root = nullptr;
}
```

* `RBT<T>`复制构造函数，需要实现*deep copy*

```cpp
// copy constructor -- deep copy every element of the other RBT
template<typename T>
RBT<T>::RBT(const RBT &other) {
    cnt = other.cnt;
    if( cnt == 0 )
        return;
    std::queue<std::pair<TreeNode*, TreeNode*> > q;
    q.push(std::make_pair(root(other.root->val),other.root));
    while( !q.empty() ) {
        TreeNode *node1 = q.front().first;
        TreeNode *node2 = q.front().second;
        q.pop();
        if( node2->left ) {
            node1->left = new TreeNode(node2->left->val);
            node1->left->parent = node1;
            q.push(make_pair(node1->left, node2->left));
        }
        if( node2->right ) {
            node1->right = new TreeNode(node2->right->val);
            node1->irght->parent = node1;
            q.push(make_pair(node1->right, node2->right));
        }
    }
}
```

* `RBT<T>`赋值构造函数， 与复制构造函数一样

```cpp
// assignment constructor
template<typename T>
const RBT<T>& RBT<T>::operator=(const RBT &other) {
    cnt = other.cnt;
    if( cnt == 0 )
        return *this;
    std::queue<std::pair<TreeNode*, TreeNode*> > q;
    q.push(std::make_pair(root(other.root->val),other.root));
    while( !q.empty() ) {
        TreeNode *node1 = q.front().first;
        TreeNode *node2 = q.front().second;
        q.pop();
        if( node2->left ) {
            node1->left = new TreeNode(node2->left->val);
            node1->left->parent = node1;
            q.push(make_pair(node1->left, node2->left));
        }
        if( node2->right ) {
            node1->right = new TreeNode(node2->right->val);
            node1->right->parent = node1;
            q.push(make_pair(node1->right, node2->right));
        }
    }
    return *this;
}
```

* `RBT<T> destructor`函数，当对象不需要时释放内存

```cpp
// destructor
template<typename T>
RBT<T>::~RBT() {
    if( cnt == 0 )
        return ;
    cnt = 0;
    std::queue<TreeNode*> q;
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

* `RBT<T>`的`size()`和`empty()`函数

> `size()`函数返回当前`BST<T>`中的元素个数

```cpp
// return the size of RBT
template<typename T>
const std::size_t RBT<T>::size() const {
    return cnt;
}
```

> `empty()`函数， 当`BST<T>`没有数据时返回`true`

```cpp
// check whether the RBT is nullptr
template<typename T>
const bool RBT<T>::empty() const {
    return cnt == 0;
}
```

* `RBT<T>`的`begi()`和`end()`函数，分别返回对应的`iterator`

> `begin()`函数需要首先找到`BST<T>`类的最左节点，然后返回其`iterator`实例

```cpp
// iterator begin() and end()
template<typename T>
typename RBT<T>::iterator RBT<T>::begin() const {
    TreeNode* node = root;
    while( node && node->left )
        node = node->left;
    return iterator(node);
}
```

> `end()`直接返回`null iterator`即可

```cpp
template<typename T>
typename RBT<T>::iterator RBT<T>::end() const {
    return iterator();
}

```

* 实现`insert(T v)`来插入一个新的数据

> 对于`insert(T v)`来说，我们可以和`BST<T>`一样，使用二分法将数据作为`leaf node`插入`RBT<T>`。和`BST<T>`不同的是，插入元素默认是`RED node`之后还要满足`RBT`的规则要求

```cpp
// insert an element into RBT
template<typename T>
void RBT<T>::insert(T v) {
    insert(v,root);
    // increase cnt
    ++cnt;
}

// private : insert an element into RBT
template<typename T>
void RBT<T>::insert(T v, TreeNode *&r, TreeNode * const &p) {
    if( r == nullptr ) {
        r = new TreeNode(v, nullptr, nullptr, p);
        // here we need to remove the violation of RBT if exists
        insert_case1(r);
    } else if( v < r->val )
        insert(v, r->left, r);
    else
        insert(v, r->right,r);
}
```

> 当插入的元素作为`root`节点，我们只需要将节点改成`BLACK`

```cpp
// this function handle case that the current node is the root
template<typename T>
void RBT<T>::insert_case1(TreeNode *r) {
    // in this case we just paint the node to black
    if( r->parent == nullptr )
        r->color = BLACK;
    else
        insert_case2(r);
}
```

> 当插入节点的父节点是`BLACK node`,新插入节点没有违反`RBT`规则，不需要操作

```cpp
// this handles when the node's parent is black
template<typename T>
void RBT<T>::insert_case2(TreeNode *r) {
    // in this case we just return
    if( r->parent->color == BLACK )
        return;
    else
        insert_case3(r);
}
```

> 如下图所示，当插入节点`parent node`和`uncle node`都是`RED`时，我们可以将`parent node`和`uncle node`改成`BLACK`,`grandparent node`改成`RED`,这样只有`grandparent node`违反`RBT`规则。所以需要从`insert_case1()`开始检查`grandparent node`。
![insert case3](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Red-black_tree_insert_case_3.svg/320px-Red-black_tree_insert_case_3.svg.png)

```cpp
// this handles when both the parent and the uncle are red
template<typename T>
void RBT<T>::insert_case3(TreeNode *r) {
    // when both the parent and the uncle are red, we can paint the parent and
    // uncle to black and the grandParent to red
    // this will remain all RBT rules but the grandParent may violate rule
    TreeNode* u = uncle(r);
    if( u && u->color == RED ) {
        r->parent->color = BLACK;
        u->color = BLACK;
        u->parent->color = RED;
        insert_case1(u->parent);
    } else
        insert_case4(r);
}
```

> 如下图所示，当插入节点`parent node`是`RED`，而`uncle node`都是`BLACK`时，如果当前节点是左子树而其父节点是右子树，我们可以在父节点上进行`rotate_right()`操作。如果相反，则进行`rotate_left()`操作。
![insert case4](https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Red-black_tree_insert_case_4.svg/320px-Red-black_tree_insert_case_4.svg.png)

```cpp
// this handles when current node's parent is red but the uncle is black
template<typename T>
void RBT<T>::insert_case4(TreeNode *r) {
    TreeNode *g = grandParent(r);
    // when current node is the right child and parent is the left child of
    // grandParent , we rotate left
    if( r == r->parent->right && r->parent == g->left ) {
        rotate_left(r->parent);
        r = r->left;
    // in this case we rotate right
    } else if( r == r->parent->left && r->parent == g->right ) {
        rotate_right(r->parent);
        r = r->right;
    }
    insert_case5(r);
}
```

> `rotate_left(TreeNode *r)`操作，就是将当前节点的右子树作为父节点而当前节点作为其左子树。

```cpp
// this rotate the current node to the left
template<typename T>
void RBT<T>::rotate_left(TreeNode *r) {
    TreeNode *saved_right_left = r->right->left, *child = r->right;
    if( r->parent ) {
        if( r == r->parent->left )
            r->parent->left = child;
        else
            r->parent->right = child;
    // if current node is the root we need update the root node also
    } else {
        root = child;
    }
    child->left = r;
    child->parent = r->parent;
    r->parent = child;
    r->right = saved_right_left;
    if( saved_right_left )
        saved_right_left->parent = r;
}
```

> `rotate_right(TreeNode *r)`操作，就是将当前节点的左子树作为父节点而当前节点作为其右子树。

```cpp
// thi rotate the current node to the right
template<typename T>
void RBT<T>::rotate_right(TreeNode *r) {
    TreeNode *saved_left_right = r->left->right, *child = r->left;
    if( r->parent ) {
        if( r == r->parent->left )
            r->parent->left = child;
        else
            r->parent->right = child;
    // if current node is the root we need update the root node also
    } else {
        root = child;
    }
    child->right = r;
    child->parent = r->parent;
    r->parent = child;
    r->left = saved_left_right;
    if( saved_left_right )
        saved_left_right->parent = r;
}
```

> 如下图所示，当插入节点`parent node`是`RED`，而`uncle node`都是`BLACK`时，我们可以将`grandparent node`变成`RED`,而父节点变成`BLACK`。如果当前节点和父节点都是左子树，对`grandparent node`进行`rotate_right()`操作，这样整个`RBT<T>`就满足所有平衡规则了。如果当前节点和父节点都是右子树则相反。
![insert case5](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Red-black_tree_insert_case_5.svg/320px-Red-black_tree_insert_case_5.svg.png)

```cpp
// this handles that the parent is red but uncle is black
template<typename T>
void RBT<T>::insert_case5(TreeNode *r) {
    TreeNode *g = grandParent(r);
    // in this case we just paint the parent to BLACK and grandParent to RED
    // then rotate left or right
    g->color = RED;
    r->parent->color = BLACK;
    if( r == r->parent->left )
        rotate_right(g);
    else
        rotate_left(g);
}

```

> `grandParent(TreeNode * const &r)`函数得到当前节点的`grandparent node`

```cpp
// get current node's grandparent
template<typename T>
typename RBT<T>::TreeNode* RBT<T>::grandParent(TreeNode * const &r) const {
    if( r && r->parent )
        return r->parent->parent;
    else
        return nullptr;
}
```

> `uncle(TreeNode * const &r)`函数得到父节点的兄弟节点

```cpp
// get current node's uncle
template<typename T>
typename RBT<T>::TreeNode* RBT<T>::uncle(TreeNode * const &r) const {
    TreeNode *g = grandParent(r);
    // if no grandParent then no uncle
    if( g == nullptr )
        return nullptr;
    if( r->parent == g->left )
        return g->right;
    else
        return g->left;
}
```

* `find(T v)`查找数据是否存在, 与`BST`一致，只需要二分查找即可

```cpp
// find an element
template<typename T>
typename RBT<T>::iterator RBT<T>::find(const T& v) const {
    TreeNode *p = root;
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

* `erase(iterator itr)`删除数据

```cpp
// remove one element
template<typename T>
void RBT<T>::erase(iterator itr) {
    --cnt;
    if( cnt == 0 ) {
        delete root;
        root = nullptr;
        return;
    }
    // if left & right child both exist
    if( (itr.node)->left && (itr.node)->right ) {
        TreeNode *predecessor = findMax((itr.node)->left);
        (itr.node)->val = predecessor->val;
        erase(iterator(predecessor));
    } else {
        TreeNode *child = (itr.node)->left ? (itr.node)->left : (itr.node)->right;
        // here difference with bst, we need rebalance rbt to satisfy rbt rules
        if( (itr.node)->color == BLACK ) {
            // when current node is black and its child is RED, we can
            // simply repaint child to black
            if( child && child->color == RED )
                child->color = BLACK;
            // or the deleted node is the leaf cell, we should rebalance it
            else
                delete_case1(itr.node);
        }
        replace_node_in_parent(itr.node, child);
        delete itr.node;
    }
}
```


```cpp
template<typename T>
void RBT<T>::delete_case1(TreeNode *node) {
    // when current node is the new root , it's okay
    if( node->parent != nullptr )
        delete_case2(node);
}
```


![delete case2](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Red-black_tree_delete_case_2_as_svg.svg/320px-Red-black_tree_delete_case_2_as_svg.svg.png)

```cpp
template<typename T>
void RBT<T>::delete_case2(TreeNode *node) {
    // in this case when current node's sibling is RED, we should replace its
    // parent's color with its sibling, and rotate the parent
    TreeNode *s = sibling(node);
    if( s->color == RED ) {
        node->parent->color = RED;
        s->color = BLACK;
        if( node == node->parent->left )
            rotate_left(node->parent);
        else
            rotate_right(node->parent);
    }
    delete_case3(node);
}
```

![delete case3](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Red-black_tree_delete_case_3_as_svg.svg/320px-Red-black_tree_delete_case_3_as_svg.svg.png)

```cpp
template<typename T>
void RBT<T>::delete_case3(TreeNode *node) {
    // in this case current node and its parent, its sibling and its sibling's
    // two children are both black. we just repaint its sibling to red and
    // rebalance the tree for its parent
    TreeNode *s = sibling(node);
    if( node->parent->color == BLACK && s->color == BLACK &&
        (s->left == nullptr || s->left->color == BLACK) && (s->right == nullptr || s->right->color == BLACK) ) {
        s->color = RED;
        delete_case1(node->parent);
    } else {
        delete_case4(node);
    }
}
```

![delete case4](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Red-black_tree_delete_case_4_as_svg.svg/320px-Red-black_tree_delete_case_4_as_svg.svg.png)

```cpp
template<typename T>
void RBT<T>::delete_case4(TreeNode *node) {
    // in this case current node and its sibling and its sibling's two children
    // are all black but the parent is red, we can just exchange the parent and
    // its sibling's color

    TreeNode *s = sibling(node);
    if( node->parent->color == RED && s->color == BLACK &&
        (s->left == nullptr || s->left->color == BLACK) && (s->right == nullptr || s->right->color == BLACK) ) {
        s->color = RED;
        node->parent->color = BLACK;
    } else {
        delete_case5(node);
    }
}
```

![delete case5](https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Red-black_tree_delete_case_5_as_svg.svg/320px-Red-black_tree_delete_case_5_as_svg.svg.png)

```cpp
template<typename T>
void RBT<T>::delete_case5(TreeNode *node) {
    // in this case current node's sibling has a left red child and right black
    // child, we can perform a right rotation on the sibling and exchange the
    // color with its red child
    TreeNode *s = sibling(node);
    if( s->color == BLACK ) {
        if( node == node->parent->left && (s->left && s->left->color == RED) &&
            (s->right == nullptr || s->right->color == BLACK) ) {
            s->color = RED;
            s->left->color = BLACK;
            rotate_right(s);
        } else if( node == node->parent->right && (s->right && s->right->color == RED) &&
                    (s->left == nullptr || s->left->color == BLACK) ) {
            s->color = RED;
            s->right->color = BLACK;
            rotate_left(s);
        }
    }
    delete_case6(node);
}
```

![delete case6](https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Red-black_tree_delete_case_6_as_svg.svg/320px-Red-black_tree_delete_case_6_as_svg.svg.png)

```cpp
template<typename T>
void RBT<T>::delete_case6(TreeNode *node) {
    // in this case current node's sibling has a right red child
    // we can perform rotate on current node's parent and exchange the color
    // with sibling, and repaint the sibling's right red child to black
    TreeNode *s = sibling(node);
    s->color = node->parent->color;
    node->parent->color = BLACK;
    if( node == node->parent->left ) {
        s->right->color = BLACK;
        rotate_left(node->parent);
    } else {
        s->left->color = BLACK;
        rotate_right(node->parent);
    }
}
```


```cpp
template<typename T>
void RBT<T>::replace_node_in_parent(TreeNode *node, TreeNode *newNode) {
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
```


```cpp
template<typename T>
typename RBT<T>::TreeNode *RBT<T>::findMax(TreeNode *node) {
    while( node && node->right )
        node = node->right;
    return node;
}
```

> `sibling(TreeNode * const &r)`函数得到当前节点的兄弟节点

```cpp
// get current node's silbing
template<typename T>
typename RBT<T>::TreeNode* RBT<T>::sibling(TreeNode * const &r) const {
    if( r == nullptr || r->parent == nullptr )
        return nullptr;
    if( r->parent->left == r )
        return r->parent->right;
    else
        return r->parent->left;
}
```

# 测试程序

```cpp
```

# 参考

1.[wikipedia - binary search tree](https://en.wikipedia.org/wiki/Binary_search_tree)  
2.[wikipedia - red black tree](https://en.wikipedia.org/wiki/Red%E2%80%93black_tree)  
