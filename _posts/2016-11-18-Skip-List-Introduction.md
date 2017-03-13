---
title: Skip List Introduction and Implementation
categories: programming
tags: [data structure,probability]
---

# 什么是*Skip List* ？

## *skip list*定义和特性

> __skip list__是一种可以对有序的元素序列进行快速搜索的数据结构。其结构如下图所示:
![skiplist](https://upload.wikimedia.org/wikipedia/commons/8/86/Skip_list.svg)

可以看到，*skip list* 是由多层*list* 组成的。 其最底层是普通的有序地链表。而上面每一层都是作为下面层的链表的快速查找通道。对于在第`i`层的元素，其出现在`i+1`层的概率是固定的`p`。

## *skip list* 查找操作
## *skip list* 插入操作

![skiplist insertion](https://upload.wikimedia.org/wikipedia/commons/2/2c/Skip_list_add_element-en.gif)

# *Skip List* 实现

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

```cpp
// construct new node
template<typename Key, typename Comparator>
typename skiplist<Key, Comparator>::Node* skiplist<Key, Comparator>::newNode(const Key& key, const unsigned int height) {
    char* mem = (char*)malloc(sizeof(Node) + sizeof(Node*) * (height - 1));
    return new (mem) Node(key);
}
```

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

```cpp
// check if the key is after the current node
template<typename Key, typename Comparator>
bool skiplist<Key, Comparator>::isAfterNode(const Key& key, Node* node) {
    return ( node != nullptr ) && cmp_(node->key, key) < 0;
}

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

// skiplist search method
template<typename Key, typename Comparator>
bool skiplist<Key, Comparator>::contains(const Key& key) {
    Node *p = findGreater(key, nullptr);
    return ( p != nullptr ) && cmp_(key, p->key) == 0;
}

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

## test

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
