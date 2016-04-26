---
title : Segment Tree
categories : programming
tags : [c++,tree,data structure]
---

# 什么是*Segment Tree*?

> `segment tree`(线段树)每个leaf node都是输入数据，而internal node则是子节点特点的总结。

# *Segment Tree*实现
这儿以实现一个*sum range*的*segment tree*为例。

* 首先，需要一个`SegmentTreeNode`来表示每个节点
    1. `left`和`right`节点指向左右子节点
    2. `start`和`end`表示当前节点包含`nums[start:end]`
    3. `sum`表示`nums[start:end]`之和

```cpp
// SegmentTreeNode definition
struct SegmentTreeNode {
    int start, end, sum;
    SegmentTreeNode* left;
    SegmentTreeNode* right;
    // constructor
    SegmentTreeNode(int a, int b):start(a),end(b),sum(0),left(nullptr),right(nullptr){}
};
```

* 使用上面的`SegmentTreeNode`来定义`SegmentTree`:
    * `update(int i, int val)`来更新指定位置的值
    * `sumRange(int i, int j)`得到nums从`i`到`j`的和

```cpp
// SegmentTree definition
class SegmentTree {
public:
    // constructor
    SegmentTree(vector<int> &nums) {
        int n = nums.size();
        root = buildTree(nums,0,n-1);
    }
    
    // public member function to update specified value
    void update(int i, int val) {
        modifyTree(i,val,root);
    }
    
    // public member function to query the sum of range
    int sumRange(int i, int j) {
        return queryTree(i, j, root);
    }

private:
    SegmentTreeNode* root;
        
    // private helper function to build the segment tree
    SegmentTreeNode* buildTree(vector<int> &nums, int start, int end) {
        if(start > end) return nullptr;
        SegmentTreeNode* root = new SegmentTreeNode(start,end);
        if(start == end) {
            root->sum = nums[start];
            return root;
        }
        int mid = start + (end - start) / 2;
        root->left = buildTree(nums,start,mid);
        root->right = buildTree(nums,mid+1,end);
        root->sum = root->left->sum + root->right->sum;
        return root;
    }
    
    // private function to get sum of range
    int queryTree(int i, int j, SegmentTreeNode* root) {
        if(root == nullptr) return 0;
        if(root->start == i && root->end == j) return root->sum;
        int mid = (root->start + root->end) / 2;
        if(i > mid) return queryTree(i,j,root->right);
        if(j <= mid) return queryTree(i,j,root->left);
        return queryTree(i,mid,root->left) + queryTree(mid+1,j,root->right);
    }
    
    // private function to update value
    int modifyTree(int i, int val, SegmentTreeNode* root) {
        if(root == nullptr) return 0;
        int diff;
        if(root->start == i && root->end == i) {
            diff = val - root->sum;
            root->sum = val;
            return diff;
        }
        int mid = (root->start + root->end) / 2;
        if(i > mid) {
            diff = modifyTree(i,val,root->right);
        } else {
            diff = modifyTree(i,val,root->left);
        }
        root->sum = root->sum + diff;
        return diff;
    }
};
```

# 参考

1.[wikipeida - Segment Tree](https://en.wikipedia.org/wiki/Segment_tree)
