---
title : Binary Indexed Tree
categories : programming
tags : [c++,tree,data structure]
---

# 什么是**Binary Indexed Tree**

> 任何一个正数都可以由一系列2的次方的和组成。基于这个，**Binary Indexed Tree**每个node都存储2的次方个数据的和。

> **Binary Indexed Tree** 每个parent node = idx - ( idx & -idx )

# **Binary Indexed Tree**应用

**Binary Indexed Tree**时间复杂度是`O(logn)`，因此可以用来很快速的得到数据的range sum.

# **Binary Indexed Tree**实现

基于**Binary Indexed Tree**的定义，我们可以很容易实现下面的`BIT`:

```cpp
// Binary Index Tree definition
class BIT {

public:
    // constructor
    BIT(vector<int> &nums) {
        nSize = nums.size();
        num = BITsum = vector<int>(nSize+1, 0);
        for(int i=0;i<nSize; ++i)
            update(i, nums[i]);
    }
    
    // interface to update value
    void update(int i, int val) {
        int delta = val - num[++i];
        for(num[i] = val;i<=nSize;i += i & -i)
            BITsum[i] +=delta;
    }
    
    // interface to get the sum of range
    int sumRange(int i, int j) {
        return (i>j || i<0 || j>=nSize)? 0: i==j?num[i+1]:( getSum(j+1) - getSum(i) );
    }
    
private:   
    int nSize;
    vector<int> BITsum; // BIT sum array
    vector<int> num; // original nums array

    int getSum(int i)
    {
        if(i<=0 || i>nSize) return 0;
        int res=BITsum[i];
        while( (i -= i & -i) > 0 )
            res +=BITsum[i];
        return res;
    }
};
```

# 参考

1.[wikipedia-Fenwick_tree](https://en.wikipedia.org/wiki/Fenwick_tree)  
