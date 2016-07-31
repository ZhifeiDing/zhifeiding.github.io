---
title: [Reading Notes][Effective STL] Note 9 : 选择删除元素的方法
categories: programming
tags: [c++, algorithm]
---

# 结论

书里对于要删除容器内元素的情况分成了下面三类：

* 删除特定值的所有对象
* 删除满足特定条件的对象
* 删除对象同时还需要其它操作

对于每一类都有不同的方法来处理， 下面分别介绍。

## 删除特定值的所有对象

* 对于`vector`,`string`或`deque`使用`erase-remove`

这一条其实对于连续内存的容器都适用，一般使用下面方法:

```cpp
c.erase(remove(c.begin(), c.end(), removeVal), c.end);
```

其中`remove`是`algorithm`库中提供的方法，功能是使用*move assignment*来将不是`removeValue`的元素移到前面，最后返回前面不是`removeValue`的元素组成的新范围的`end`迭代器。然后可以使用容器提供的`erase`函数来删掉`remove`返回迭代器后面所有值。


* 对于`list`，则使用`list::remove`

* 对于关联容器`set`/`map`,则使用`set::erase`或`map::erase`

## 只删除满足特定条件的对象

* 对于`vector`,`string`或`deque`使用`erase-remove_if`

* 对于`list`，则使用`list::remove_if`

* 对于关联容器`set`/`map`,则使用`remove_copy_if`和`swap`或循环遍历时对于传给`erase`的迭代器要进行后缀递增


## 删除对象同时还需要其它操作

* 对于`vector`,`string`，`deque`或`list`标准序列容器，需要循环遍历容器中元素，需要注意的是每次调用`erase`时，需要用返回值更新迭代器

* 对于关联容器`set`/`map`,循环遍历时对于传给`erase`的迭代器要进行后缀递增

# 参考

* [remove/remove_if](http://en.cppreference.com/w/cpp/algorithm/remove)  
* [vector::erase](http://en.cppreference.com/w/cpp/container/vector/erase)  
* [remove_copy/remove_copy_if](http://en.cppreference.com/w/cpp/algorithm/remove_copy)  
