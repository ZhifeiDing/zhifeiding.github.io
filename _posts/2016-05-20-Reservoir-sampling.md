---
title : Reservoir Sampling
categories : programming
tags : [algorithm,c++]
---

# 问题描述

首先，*Leetcode*上有一道题:

> Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen. 

> Follow up:
    What if the linked list is extremely large and its length is unknown to you? Could you solve this efficiently without using extra space?

这道题很好的引出了一个问题，那就是如何从一个很大的集合或者大小不确定的集合中随机的选择*k*个样本。

# Reservoir Sampling

下面将首先从*n*个样本集中随机选择一个元素到从 *n* 个样本集合中随机选择 *k* 个元素开始，然后推广到如何根据权值来选择元素

## 随机选择一个元素

上面*leetcode* 的问题就是典型的从一个集合中随机选择一个元素的问题。如果我们知道集合大小或者集合能够全部读入内存，我们可以简单的随机生成一个小于 *n* 的数然后得到对应*index*的元素。 然而，问题困难就在于集合大小没法确定或者太大而不能全部读入内存。我们可以设计这样一种算法：

* 选择第一个元素
* 对于第 *i(i>1)* 个元素，我们可以使用 *1/i* 的概率来选择该元素
* 否则就是 *1-1/i* 的概率保留之前选择的元素

对于上面算法的正确性， 我们可以简单分析如下：

* 当*n==1*时选择概率为1
* 当*n==2*时每个元素被选择概率都是*1/2*
* 当*n==3*时第三个元素被选择概率是*1/3*, 而前面两个被选择的概率则为*2/3 * 1/2 = 1/3*
* 当*n>3*时，第*i-1*轮时元素被选择概率为*1/(i-1)*,第*i*轮时元素不被选择的概率是 *1/i* ,所以继续被选择的概率是 *1/(i-1) * (1 - 1/i) = 1/i* , 所以第 *n* 轮时每个元素被选择的概率是 *1/n*

使用上面算法解决上面的*leetcode*问题代码如下:

```cpp
class Solution {
public:
    /** @param head The linked list's head. Note that the head is guanranteed to be not null, 
     * so it contains at least one node.
     */
    Solution(ListNode* head) {
        this->head = head;
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        int cnt = 1;
        ListNode *h = head;
        int res = h->val;
        while( h ) {
            if( rand()%cnt == 0 )
                res = h->val;
            h = h->next;
            ++cnt;
        }
        return res;
    }
private:
    ListNode *head;
};
```

## 随机选择 *k* 个元素

对于上面的算法，我们可以很容易的推广到从集合中随机选择 *k* 个元素的问题：

* 选择前面的 *k* 个元素
* 对于第 *i(i>k)* 个元素，我们可以使用 *k/i* 的概率来选择该元素
* 否则就是 *1-k/i* 的概率保留之前选择的元素

第 *i-1* 轮时元素被选择概率为 *k/(i-1)* ,第 *i* 轮时元素不被选择的概率是 *1/i* ,所以继续被选择的概率是 *k/(i-1) * (1 - 1/i) = k/i* , 所以第 *n* 轮时每个元素被选择的概率是 *k/n* 。 该算法也被称作*Algorithm R* 。

假设上面的 *leetcode* 的问题变成随机选择 *k* 个元素则代码：

```cpp
class Solution {
public:
    /** @param head The linked list's head. Note that the head is guanranteed to be not null, 
     * so it contains at least one node.
     */
    Solution(ListNode* head) {
        this->head = head;
    }
    
    /** Returns a random node's value. */
    vector<int> getRandom(int k) {
        int cnt = 1;
        ListNode *h = head;
        vector<int> res;
        while( h ) {
           res.push_back(h->val);
           h = h->next;
           ++cnt;
        }
        while( h ) {
            int idx = rand()%cnt;
            if( idx < k )
                res[idx] = h->val;
            h = h->next;
            ++cnt;
        }
        return res;
    }
private:
    ListNode *head;
};
```

## 根据权重来随机选择元素

假设集合中元素带有权重 *w* ，我们希望选择元素的概率和该元素的权重 *w* 正相关，我们该怎么做呢？
根据选择概率和权重 *w* 的不同，分为下面两种情况：

* 所有元素权重是 *W* , 则某一个权重为*w*的元素被选择的概率 *p = w/W*

对于这种情况 *Efraimidis* 和 *Spirakis* 给出了 *Algorithm A-Res* 算法，使用一个 *priority queue* 来保存 *k* 个元素，

* 假设元素 *i* 权重是 *wi*, 元素 *j* 的权重是 *wj*, 元素 *i* 被选择的概率是 *pi*, 我们定义元素 *j* 的概率是 *pj = pi * max(1, wj/wi)*


# 参考

* [wikipedia - Reservoir_sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)  
* [Leetcode - Linked List Random Node](https://leetcode.com/problems/linked-list-random-node/)  
