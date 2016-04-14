---
title : Trie And Suffix Tree
categories : programming
tags : [c++,data structure，tree]
---

# Trie

## 什么是Trie?

> *Trie*(读try)是一种*prefix tree*, 即可以通过前缀来查找。和二叉树不同，*Trie*并不存储值本身，而是根据字符位置来索引。

*Trie*结构如下图所示：
![Trie](https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Trie_example.svg/256px-Trie_example.svg.png)

## Trie实现

* Trie类声明

首先， 我们需要定义一个`TrieNode`来表示`Trie`中节点:

```cpp
class TrieNode {
public:
	// constructor
	explicit TrieNode() {
		node.resize(26);
		isLeaf = false;
	}
	const TrieNode*& operator [](int idx) {
		return node[idx];
	}
	void setLeaf() {
		isLeaf = true;
	}
	bool isLeaf() const {
		return isLeaf;
	}
private:
	// here only consider lowercase alphabetic 
	vector<TrieNode*> node;
	// indicate whether the node is leaf
	bool isLeaf;
};
```

然后我们以上面`TrieNode`类为节点声明`Trie`类:

```cpp
class Trie {
public:
	// constructor
	explicit Trie();
	// insert an element
	void insert(const string &s);
	// check if an element exists
	bool exists(const string &s);
private:
	TrieNode *root;
};
```

* Trie类构造函数: 创建root节点， 并设置其为leaf节点

```cpp
Trie::Trie() {
	root = new TrieNode();
	root->setLeaf() = true;
}
```

* Trie类`insert(const string &s)`成员函数

```cpp
void Trie::insert(const string &s) {
	TrieNode* node = root;
	for(auto ch : s) {
		if( node[ch - 'a'] == nullptr )
			node[ch-'a'] = new TrieNode();
		node = node[ch-'a'];
	}
	node->setLeaf();
}
```

* Trie类`exists(const string &s)`成员函数

```cpp
bool Trie::exists(const string &s) {
	TrieNode* node = root;
	for(auto ch : s) {
		if( node[ch - 'a'] == nullptr )
			return false;
		node = node[ch-'a'];
	}
	return node->isLeaf();
}
```

# Suffix Tree

## 什么是Suffix Tree?

![Suffix Tree](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Suffix_tree_BANANA.svg/226px-Suffix_tree_BANANA.svg.png)

## Suffix Tree实现

* Suffix Tree类声明

```cpp
```

* Suffix Tree类构造函数

```cpp
```

* Suffix Tree类`insert()`成员函数

```cpp
```

# 参考

[wikipedia - Trie](https://en.wikipedia.org/wiki/Trie)  
[wikipedia - Suffix Tree](https://en.wikipedia.org/wiki/Suffix_tree)  

