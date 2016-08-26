---
title: Fisher–Yates Shuffle Algorithm
categories : programming
tags : [algorithm,c++]
---

# 洗牌问题

*Leetcode*上最近出了一道新题，是这样的：

> Shuffle a set of numbers without duplicates.
>
> Example:
>
> // Init an array with set 1, 2, and 3.
> int[] nums = {1,2,3};
> Solution solution = new Solution(nums);
>
> // Shuffle the array [1,2,3] and return its result. Any permutation of [1,2,3] must equally likely to be returned.
> solution.shuffle();
>
> // Resets the array back to its original configuration [1,2,3].
> solution.reset();
>
> // Returns the random shuffling of array [1,2,3].
> solution.shuffle();

这是一道关于洗牌的问题，看到这个问题我就想到了之前的[Reservoir Sampling](http://zhifeiding.github.io/programming/2016/05/20/Reservoir-sampling/)，也是关于随机选择数据的问题。那么这两者是否有关系呢？首先要说明的是，*Reservoir Sampling*中的*Algorithm R*可以看作下面要介绍的*Fisher-Yates*洗牌算法的一种特殊情况。

# *Fisher-Yates*洗牌算法

*Fisher-Yates*洗牌算法，故名思意，就是用来将有限的序列来生成随机排列的序列。*Fisher-Yates*洗牌算法是由*Ronald Fisher*和*Frank Yates*首先提出的，算法描述如下(使用纸和笔，并用标有数字的纸堆产生随机数)：

1. 列出数字*1*到*N*
2. 产生一个随机数*k*，其范围在*1*到剩下未划掉的数字个数之间
3. 从未段开始选择没有被划掉的第*k*个数字并写到另一个集合的末段，同时划掉该数字
4. 重复步骤*2-3*只到所有数字都被划掉
5. 步骤*3*得到的集合就是原始集合的一个随机序列

## *Fisher-Yates*洗牌算法实现

```cpp
class Solution {
public:
    Solution(vector<int> nums) {
        val = nums;
        data = nums;
        srand((int)(time(NULL)));
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        data = val;
        return data;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        for(int i = data.size()-1; i > 0; --i) {
            int j = rand()%(i+1);
            swap(data[i], data[j]);
        }
        return data;
    }
private:
    vector<int> data, val;
};
```

# 参考 

* [wikipedia - Fisher-Yates Shuff](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)
* [Leetcode - Shuff an Array](https://leetcode.com/problems/shuffle-an-array/)
