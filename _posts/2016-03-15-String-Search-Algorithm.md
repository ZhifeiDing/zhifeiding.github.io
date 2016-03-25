---
title : String Searching Algorithm
categories : programming
tags : [algorithm, c++, string search]
---

  字符串查找算法(String Searching algorithm) 就是在给定字符串(Text)里查找给定字符串(Pattern)。关于字符串查找也有很多不同算法，最近学习了一下比较出名的**KMP算法**和**Boyer-Moore算法**， 自己实现了一下。

# Native String Searching 

  何谓**Native** ?就是自己本能想到的。 那么看到字符串查找， 我们想到什么呢? 我就想到了， 这还不简单啊， 把Text和Pattern从第一个开始一个一个比较， 如果都相同，就查找到了， 否则就把从Text第二个开始直到找到或者Text剩下字符串不足Pattern长度。
  既然这么简单，那我们就实现出来看看吧。

```cpp
int nativeStringSearch(string &text, string &p) {
	for(int i = 0; i < text.size() - p.size(); ++i)
		for(int j = 0; j < p.size() && text[i+j] == p[j]; ++j);
		if( j == p.size() - 1 )
			return i;
	return -1;  // not found		
}
```

  看， 代码也够简单的。那为什么还有其他算法呢？我们看看上面的时间复杂度居然是O(n*m)，很明显这个实现不够efficient。 那有没有办法查找得更快的算法呢？所以就有了下面的KMP算法。

# KMP算法

  大名鼎鼎的***KMP算法***其实是**Knuth–Morris–Pratt**的缩写， 而里面每一个人都是一个人名。上面的native string searching是从Text一个一个移动，为了提高比较速度， 我们想能不能在Text和Pattern不等时不要像上面一样一个一个移动呢？如果可以那又该怎么办呢？  
  为了提高比较速度， 当Text与Pattern不匹配时，***KMP算法***利用pattern自身的特点来跳过一些native string searching算法中比较过程。原理就是当遇到不匹配字符时， 前面已经匹配到的子串中如果存在前缀子串则可以从匹配到前缀子串处开始比较。具体例子可以参考[wikipedia](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)

## 部分匹配表
  部分匹配表其实就是当Pattern和Text不匹配时，在已经匹配的子串中前缀子串和后缀子串匹配的长度来决定Text移动长度。
  
  部分匹配表生成代码:
  
```cpp   
void KMPTable(string &p, vector<int> &kmp_table) {
	kmp_table[0] = -1;
	if( p.size() <= 2 )
		return;
	int pos = 2; // the current position
	int cnd = 0; // the next character of current candidate
	while( pos < p.size() ) {
		if(p[pos-1] == p[cnd]) {
			kmp_table[pos] = cnd;
			++pos;
			++cnd;
		} else if( cnd > 0 ) {
			cnd = kmp_table[cnd];
		} else {
			kmp_table[pos] = 0;
			++pos;
		}
	}
}
```

## KMP算法实现
  
  在部分匹配表已经生成情况下***KMP算法***实现比较简单的，使用上面生成的部分匹配表的***KMP算法***代码:
  
```cpp   
int KMPSearch(string &text, string &p) {
	vector<int> kmp_table(p.size(),0);
	KMPTable(p, kmp_table);
	
	int m = 0; // the index of text
	int i = 0; // the index of p
	while( m + p.size() < text.size() ) {
		if( text[m+i] == p[i] ) {
			if( i == p.size() - 1 )
				return m;
			++i;
		} else if( kmp_table[i] >= 0 ) {
			m = m + i - kmp_table[i];
			i = kmp_table[i];
		} else {
			m = m + 1;
			i = 0;
		}
	}
	return -1;
}
```

在部分匹配表已经存在情况下， 上面代码的时间复杂度是`O(n)`(n是Text长度), 而计算部分匹配表的时间复杂度是`O(k)`(k是Pattern的长度). 所以KMP总的时间复杂度是`O(k+n)`。 

# Boyer-Moore 算法

了解了上面的***KMP算法***之后， 你可能会疑惑， 怎么还有***Boyer-Moore算法***? 难道还能比***KMP算法***更efficient？   
在回答上面问题之前， 我们先了解一下***Boyer-Moore算法***原理。

与`KMP`算法类似， `Boyer-Moore`算法也是利用Pattern本身特点来跳过一些比较。不同的是，`Boyer-Moore`算法使用了两种规则来shift pattern，并且从右往左比较。.

* 坏字符规则

当比较的字符不等时，如果我们可以将该字符和在pattern中出现的该字符对齐比较。如果不存在我们可以直接将pattern移到该字符之后开始比较。

* 好后缀规则

这儿好后缀指的是当pattern和Text出现mismatch时候，如果



```cpp
```
