---
title: Skip List Introduction and Implementation
categories: programming
tags: [data structure,probability]
---

# 什么是*Skip List* ？

## *Skip List*定义和特性

>
__Skip List__ 是一种可以对有序的元素序列进行快速搜索的数据结构。其结构如下图所示:
![skiplist](https://upload.wikimedia.org/wikipedia/commons/8/86/Skip_list.svg)

可以看到，*Skip List* 是由多层*list* 组成的。 其最底层是普通的有序地链表。而上面每一层都是作为下面层的链表的快速查找通道。对于在第`i`层的元素，其出现在`i+1`层的概率是`p`。

## *Skip List* 查找操作

分析上面*Skip List* 的结构，可以很容易就得到，查找一个元素时，从`head node`顶层开始，沿着`next node`一直查找，直到`next node`不小于查找的`key`。如果相等，则查找结束。 如果`next node`是`null`或者比`key`大， 则从下一层开始继续查找。在每一层中查找需要的次数期望值时`1/p`。

## *Skip List* 插入操作

对于*Skip List*的插入操作，可以借鉴上面的查找操作，不同点是我们需要在每一次向下层查找是纪录当前`Node`。 这样最后得到一个比要插入`key`小的`Node`集合，我们只需要将其指向要插入`key`的`Node`即可。 可以参考下图：
![skiplist insertion](https://upload.wikimedia.org/wikipedia/commons/2/2c/Skip_list_add_element-en.gif)

***

# *Skip List* 实现

根据上面对*skiplist* 结构及其操作的介绍， 我们用`C++11`来实现一个简单的`skiplist`类。

## `skiplist`类声明

首先， 定义`skiplist`类的`public` 成员:

* `bool contains(const Key& key)`来查找`key`是否在`skiplist`
* `void insert(const Key& key)`来插入`key`
* `void erase(const Key& key)`来删除`key`
* 作为基本单元的`Node`

以及要实现上述函数的一些辅助的`private`成员函数和变量

```cpp
template<typename Key, typename Comparator>
class skiplist {
public:
    // skiplist node class
    class Node;
    // constructor
    explicit skiplist(Comparator cmp);
    // skiplist search method
    bool contains(const Key& key);
    // erase method
    // TODO:
    void erase(const Key& key);
    // insert an item into skiplist
    // don't allowed duplicate items
    void insert(const Key& key);
private:
    Comparator cmp_;
    Node* head_;
    unsigned maxHeight_;
    // the max height limitation
    const unsigned int MAX_HEIGHT = 12;
    // the probability of each item in level i appeared in level i+1
    // usually 2 or 4
    const unsigned int BRANCH = 4;
    // construct new node
    Node* newNode(const Key& key, const unsigned int height);
    // check if the key is after the current node
    bool isAfterNode(const Key& key, Node *node);
    // find the first node which is greater or equal to the key
    Node* findGreater(const Key& key, Node** prev);
    // get the random height between 0 and MAX_HEIGHT
    unsigned int getHeight();
};
```

## `skiplist::Node`类实现

根据*Skip List* 的结构， 作为基本单元的`Node`需要满足:

* 能够存储`Key`值
* 至少有一个指向下一个`Node`的指针，并且能够根据需要增加指向下一个`Node`的指针
* 得到当前层指向的下一个`Node`, `Node* next(const unsigned int level)`
* 设置当前层指向下一个`Node`, `void setNext(const unsigned int level, Node* node)`

```cpp
// class Node implementation
template<typename Key, typename Comparator>
class skiplist<Key, Comparator>::Node {
public:
    Key key;
    // constructor
    explicit Node(const Key& key) : key(key) {}
    // get next node
    Node* next(const unsigned int level) {
        assert( level >= 0 );
        return next_[level];
    }
    // set next node pointer
    void setNext(const unsigned int level, Node *node) {
        assert( level >= 0 );
        next_[level] = node;
    }
private:
    Node* next_[1];
};
```

## `skiplist`构造函数实现

* 辅助函数`skiplist::newNode`实现

该函数每次可以分配高度为`height`的`Node`所需的空间

```cpp
// construct new node
template<typename Key, typename Comparator>
typename skiplist<Key, Comparator>::Node* skiplist<Key, Comparator>::newNode(const Key& key, const unsigned int height) {
    char* mem = (char*)malloc(sizeof(Node) + sizeof(Node*) * (height - 1));
    // 使用placement new
    return new (mem) Node(key);
}
```

* `skiplist`构造函数

`skiplist`的`head_`根据结构， 需要是一个设计的最高的`Node`, 构造时每层都指向`nullptr`。同时需要初始化随机数生成器

```cpp
// skiplist constructor
template<typename Key, typename Comparator>
skiplist<Key, Comparator>::skiplist(Comparator cmp) : cmp_(cmp), maxHeight_(1) {
    head_ = newNode(-1, MAX_HEIGHT);
    for(int i = 0; i < MAX_HEIGHT; ++i)
        head_->setNext(i, nullptr);
    // set the seed of rand
    srand((unsigned int)time(NULL));
}
```

## `bool isAfterNode(const Key& key, Node* node)`成员函数

该函数用来判断当前`Node`是否比`key`小

```cpp
// check if the key is after the current node
template<typename Key, typename Comparator>
bool skiplist<Key, Comparator>::isAfterNode(const Key& key, Node* node) {
    return ( node != nullptr ) && cmp_(node->key, key) < 0;
}
```

## `Node findGreater(const Key& key, Node** prev)`成员函数

上面对`Skip List`查找和插入操作都需要找到第一个不比`key`小的`Node`。同时为了插入操作， 该函数同时记录每一层的前一个`Node`

```cpp
// find the first node which is greater or equal to the key
template<typename Key, typename Comparator>
typename skiplist<Key, Comparator>::Node* skiplist<Key, Comparator>::findGreater(const Key& key, Node** prev) {
    int level = maxHeight_ - 1;
    Node* p = head_;
    while( true ) {
        Node* n = p->next(level);
        if( isAfterNode(key, n) )
            p = n;
        else {
            if( prev != nullptr )
                prev[level] = p;
            if( level == 0 )
                return n;
            else
                --level;
        }
    }
}
```


## `void insert(const Key& key)`成员函数

* `unsigned int getHeight()`辅助函数

我们知道，插入一个`Node`到`skiplist`，我们需要随机的决定该`Node`的高度。这里我们取`Node i`出现在`Node i+1`的概率为`1/4`

```cpp
// get the random height between 0 and MAX_HEIGHT
template<typename Key, typename Comparator>
unsigned int skiplist<Key, Comparator>::getHeight() {
    unsigned int h = 1;
    while( h < MAX_HEIGHT && ( rand()%BRANCH == 0 ) ) {
        ++h;
    }
    assert( h > 0 );
    assert( h <= MAX_HEIGHT );
    return h;
}
```

* 插入一个元素时，我们需要先找到第一个不比`key`小的`Node`
* 如果该`Node`值和`key`一致，则该元素已经存在，直接返回
* 否则， 随机得到一个高度，然后将该`Node`每一层插入`skiplist`

```cpp
// insert an item into skiplist
// don't allowed duplicate items
template<typename Key, typename Comparator>
void skiplist<Key, Comparator>::insert(const Key& key) {

    // first get the node which is greater or equal to the key
    Node* prev[MAX_HEIGHT];
    Node* t = findGreater(key, prev);

    // if the key has already exist, just return
    if( t != nullptr && cmp_(t->key, key) == 0 )
        return;

    // randomly get the new node's height
    int h = getHeight();
    // if the new Node's height is higher than current skiplist's maxHeight_
    // we should set the express lane fron the head_
    // and update the maxHeight_
    if( h > maxHeight_ ) {
        for(int i = maxHeight_; i < h; ++i)
            prev[i] = head_;
        maxHeight_ = h;
    }

    // construct the node with the key and height
    t = newNode(key, h);
    // then we can simply wire the prev[] to the new node
    for(int i = 0; i < h; ++i) {
        t->setNext(i, prev[i]->next(i));
        prev[i]->setNext(i, t);
    }
}
```

## `bool contains(const Key& key)`成员函数

根据上面对`Skip List`查找操作的描述，我们只需要找到第一个不比`key`小的`Node`，然后将其值和`key`进行比较即可

```cpp
// skiplist search method
template<typename Key, typename Comparator>
bool skiplist<Key, Comparator>::contains(const Key& key) {
    Node *p = findGreater(key, nullptr);
    return ( p != nullptr ) && cmp_(key, p->key) == 0;
}
```

## 测试程序

下面简单的测试了上面实现的*skiplist* *class* 的构造及其成员函数:
```cpp
#include "skiplist.hpp"
#include <iostream>
#include <vector>

using namespace std;

template<typename Key>
struct Comparator {
  int operator()(const Key& a, const Key& b) const {
    if (a < b) {
      return -1;
    } else if (a > b) {
      return +1;
    } else {
      return 0;
    }
  }
};

void testSkiplist() {
    Comparator<int> cmp;
    vector<int> nums = {10, 5, 7, 2, 9, 0, 1};
    skiplist<int, Comparator<int> > sl(cmp);
    assert( sl.contains(nums[0]) == false );
    for(int i = 0; i < nums.size(); ++i) {
        sl.insert(nums[i]);
        assert( sl.contains(nums[i]) == true );
        if( i < nums.size() - 1 )
            assert( sl.contains(nums[i+1]) == false );
    }
}

int main() {
    testSkiplist();
    return 0;
}
```

# 参考

以下是一些参考信息   
1.[wikipedia - skip list](https://en.wikipedia.org/wiki/Skip_list)      
2.[leveldb - skiplist](https://github.com/google/leveldb/blob/master/db/skiplist.h)     
