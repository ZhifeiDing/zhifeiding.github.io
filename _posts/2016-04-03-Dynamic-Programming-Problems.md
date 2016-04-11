---
title: Dynamic Programming Problems
categories: programming
tags: [c++,algorithm]
---

# 什么是动态规划？

> 动态规划本质上是用空间换取时间的方法。为了解决一个复杂的问题， 我们通过解决小规模的子问题并记录其结果，然后得到复杂问题的结果。

上面定义我们可以得到动态规划问题两个条件: 可以分解的子问题，并且子问题有最优解。

* Overlapping Subproblems

> 动态规划分解的子问题应该是有重叠的，否则就是分治法。

* Optimal Substructure

> 分解的子问题应该有最优解,而且又子问题的最优解能够得到问题的最优解

# 一些动态规划解决的问题

## Longest Common Subsequence

```cpp
int longestCommonSubsequence() {
}
```

## Longest Increasing Subsequence

`Longest Increasing Subsequence` 是指从给定的序列中找出最长的递增子序列，子序列可以是不连续的，也可以由重复的数据。如果使用`brute force`来解决， 我们需要从第一个元素开始查找以该元素开始的子序列的递增子序列, 这样得到的时间复杂度是`O(n^2)`。那么有没有更好的算法呢?

```cpp
int longestIncreasingSubsequence(vector<int> nums) {
		if( nums.size() == 0 )
			return 0;
		vector<int> ends(nums[0]);
		for(int i = 1; i < nums.size(); ++i) {
			if( nums[i] <= ends[0] )
				ends[0] = nums[i];
			else if( ends.back() <= nums[i] )
				ends.push_back(nums[i]);
			else {
				int l = 0, r = ends.size()-1;
				int m = l + ( r - l )/2;
				while( l < r ) {
					if( ends[m] < nums[i] )
						l = m + 1;
					else
						r = m;
				}
				ends[r] = nums[i];
			}
		}
		return ends.size();
}
```

## Edit Distance

```cpp
```

# 参考

1. [wikipedia-Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming)   
2. [introduction-to-dynamic-programming](http://20bits.com/article/introduction-to-dynamic-programming)  
3. [wikipedia-Longest Increasing Subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence)
 
