---
title : Binary Search Tree
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

# 测试程序

```cpp
template<typename T>
void testTree() {
    cout << endl;
    int n = rand()%5;
    int i = 0;
    while( i++ < n ) {
        T bst;
        cout << "###########" << endl;
        cout << "Test step : " << i << endl;
        int m = rand()%10;
        cout << "insert : ";
        int v;
        while( m-- ) {
            v = rand()%100;
            cout << v << "\t";
            bst.insert(v);
        }
        cout << endl << "bst = ";
        typename T::iterator itr = bst.begin();
        while( itr != bst.end() ) {
            cout << *itr << "\t";
            itr++;
        }
        itr = bst.begin();
        if( itr != bst.end() ) {
            cout << endl << "after erase " << *itr << endl << "bst =";
            bst.erase(itr);
        }
        itr = bst.begin();
        while( itr != bst.end() ) {
            cout << *itr << "\t";
            itr++;
        }
        int x = v;
        itr = bst.find(x);
        if( itr != bst.end() ) {
            cout << "\nFound " << x << ", now delete it" << endl;
            bst.erase(itr);
        } else {
            cout << "\nCouldn't find " << x << endl;
        }
        cout << "bst = ";
        itr = bst.begin();
        while( itr != bst.end() ) {
            cout << *itr << "\t";
            itr++;
        }
        cout << "\nbst.size() = " << bst.size();
        cout << endl << "###########" << endl;
        cout << endl;
    }
}

int main() {
    srand((unsigned int)time(NULL));
    cout << "Test BST<int>\n";
    testTree<BST<int> >();
    cout << "Test RBT<int>\n";
    testTree<RBT<int> >();
    return 0;
}
```

测试程序输出:

```cpp
Test BST<int>

###########
Test step : 1
insert : 86	58	84	93	48	41
bst = 41	48	58	84	86	93
after erase 41
bst =48	58	84	86	93
Couldn't find 41
bst = 48	58	84	86	93
bst.size() = 5
###########

###########
Test step : 2
insert : 69	1	65	51	43	13	69
bst = 1	13	43	51	65	69	69
after erase 1
bst =13	43	51	65	69	69
Found 69, now delete it
bst = 13	43	51	65	69
bst.size() = 5
###########

###########
Test step : 3
insert : 86
bst = 86
after erase 86
bst =
Couldn't find 86
bst =
bst.size() = 0
###########

###########
Test step : 4
insert : 40	45	93	72	70	13	36
bst = 13	36	40	45	70	72	93
after erase 13
bst =36	40	45	70	72	93
Found 36, now delete it
bst = 40	45	70	72	93
bst.size() = 5
###########

Test RBT<int>

###########
Test step : 1
insert : 86	93	58	97
bst = 58	86	93	97
after erase 58
bst =86	93	97
Found 97, now delete it
bst = 86	93
bst.size() = 2
###########
```

# 参考

1.[wikipedia - binary search tree](https://en.wikipedia.org/wiki/Binary_search_tree)
