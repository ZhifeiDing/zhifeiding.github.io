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

`Longest Common Subsequence`是寻找多个序列的最长共同子序列，子序列可以是非连续的。对于这个问题，我们发现如果在知道了子序列的最长共同子序列后，可以根据当前子序列是否相等来得到最长共痛子序列。所以满足上面提到的*Overlapping Subproblems*和*Optimal Substructure*。

只考虑两个序列的最长共同子序列的情况，用数组`lcs`表示共同子串长度,则:

> lcs(i+1,j+1) = s1(i) == s2(j) ? lcs(i,j)+1 : max(lcs(i,j+1), lcs(i+1,j))

```cpp
int longestCommonSubsequence(string &s1, string &s2) {
	if( s1.empty() || s2.empty() )
		return 0;
		
	vector<vector<int> > lcs(s1.size()+1, vector<int>(s2.size()+1,0));
	
	for(int i = 0; i < s1.size(); ++i)
		for(int j = 0; j < s2.size(); ++j)
			lcs[i+1][j+1] = s1[i] == s2[j] ? lcs[i][j] + 1 : max(lcs[i][j+1], lcs[i+1][j]);
	return lcs[s1.size()][s2.size()];
}
```

## Longest Increasing Subsequence

`Longest Increasing Subsequence` 是指从给定的序列中找出最长的递增子序列，子序列可以是不连续的，也可以由重复的数据。如果使用`brute force`来解决， 我们需要从第一个元素开始查找以该元素开始的子序列的递增子序列, 这样得到的时间复杂度是`O(n^2)`。那么有没有更好的算法呢? 

我们知道，如果我们知道子序列的最长递增子序列，那么增加一个数据之后递增子序列也是可是得到的。因此这个问题是符合上面的*Overlapping Subproblems*和*Optimal Substructure*的。

假设有一个数组包含到当前数据的递增子序列，那么如果如果下一个数据比递增子序列最后一个大，我们可以把数据放到递增序列最后，否则我们用此数据替换递增子序列里第一个不小于此数据。最后返回该字赠子序列大小即为最长递增子序列长度。

我们用数组`F`的大小表示当前子序列`nums[0...i]`的最长递增子序列的长度，则

> F[k] = nums[i+1] where
F[k-1] < nums[i+1] and F[k] >= nums[i+1]

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

`Edit Distance`（即_Levenshtein distance_)是用来表示字符串之间的编辑距离： 

> 通过插入，删除，替换将一个字符串变成另一个字符串的最小操作数

同样的假如我们知道子串的编辑距离，我们可以得到整个字符串的距离。因此这个问题是符合上面的*Overlapping Subproblems*和*Optimal Substructure*的。

假设`dist[m+1][n+1]`用来记录两个字符串的距离则:

> dist(i+1,j+1) = min(dist[i+1][j] , // delete s2[j]  
		      dist[i][j+1] , // insert s1[i]  
		      dist[i][j]+1 , // substitution  
		      dist[i][j] // no change  
		      )  

```cpp
int LevenshteinDistance(string &s1, string &s2) {
	if( s1.empty() )
		return s2.size();
		
	if( s2.empty() )
		return s1.size();
		
	vector<vector<int> > dist(s1.size()+1, vector<int>(s2.size()));
	
	for(int i = 1; i <= s1.size();++i)
		dist[i][0] = i;
	for(int i = 1; i <= s2.size();++i)
		dist[0][i] = i;
		
	for(int i = 0; i < s1.size(); ++i)
		for(int j = 0; j < s2.size(); ++j) {
			int cost = 0;
			if( s1[i] != s2[j] )
				cose = 1;
			dist[i+1][j+1] = min(
							min(dist[i+1][j] +1, //delete
								dist[i][j+1]+1), // insertion
								dist[i][j]+cost); // substitution
		}
	return dist[s1.size()][s2.size()];
}
```

# 参考

1. [wikipedia-Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming)   
2. [introduction-to-dynamic-programming](http://20bits.com/article/introduction-to-dynamic-programming)  
3. [wikipedia-Longest Increasing Subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence)
 
