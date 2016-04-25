---
title : Segment Tree
categories : programming
tags : [c++,tree,data structure]
---

# 什么是*Segment Tree*?

# *Segment Tree*实现

```cpp
// SegmentTreeNode definition
struct SegmentTreeNode {
    int start, end, sum;
    SegmentTreeNode* left;
    SegmentTreeNode* right;
    // constructor
    SegmentTreeNode(int a, int b):start(a),end(b),sum(0),left(nullptr),right(nullptr){}
};

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
