---
title: [Reading Notes][Effective STL] Note 9 : 选择删除元素的方法
categories: programming
tags: [c++, algorithm]
---

# 结论

书里对于要删除元素情况分成下面三类：

* 对于删除特定值的所有对象

> 对于`vector`,`string`或`deque`使用`erase-remove`

> 对于`list`，则使用`list::remove`

> 对于关联容器`set`/`map`,则使用`set::erase`或`map::erase`

* 只删除满足特定条件的对象

> 对于`vector`,`string`或`deque`使用`erase-remove_if`

> 对于`list`，则使用`list::remove_if`

> 对于关联容器`set`/`map`,则使用`remove_copy_if`和`swap`或循环遍历时对于传给`erase`的迭代器要进行后缀递增


* 删除对象同时还需要其它操作

> 对于`vector`,`string`，`deque`或`list`标准序列容器，需要循环遍历容器中元素，需要注意的是每次调用`erase`时，需要用返回值更新迭代器

> 对于关联容器`set`/`map`,循环遍历时对于传给`erase`的迭代器要进行后缀递增

# 参考

* [remove/remove_if](http://en.cppreference.com/w/cpp/algorithm/remove)  
* [vector::erase](http://en.cppreference.com/w/cpp/container/vector/erase)  
* [remove_copy/remove_copy_if](http://en.cppreference.com/w/cpp/algorithm/remove_copy)  
