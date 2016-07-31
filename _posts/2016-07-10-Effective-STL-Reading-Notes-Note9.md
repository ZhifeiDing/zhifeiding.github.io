---
title:  Reading Notes - Effective STL Note 9 - 选择删除元素的方法
categories : programming
tags : [c++, algorithm]
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

其中[`remove`][1]是`algorithm`库中提供的方法，功能是使用*move assignment*来将不是`removeVal`的元素移到前面，最后返回前面不是`removeVal`的元素组成的新范围的`end`迭代器。然后可以使用容器提供的`erase`函数来删掉`remove`返回迭代器后面所有值。

* 对于`list`，则使用`list::remove`

`list`使用上面的方法也是适用的，但是使用`list`成员函数`remove`更加有效:

```cpp
c.remove(removeVal);
```

* 对于关联容器`set`/`map`,则使用`set::erase`或`map::erase`

对于关联容器`set`/`map`(以及`multiset`/`multimap`)既没有`remove`成员函数，也不能使用`remove`函数。正确的方法是调用成员函数`erase`:

```cpp
c.erase(removeVal);
```

## 只删除满足特定条件的对象

这类问题描述是将上面直接给定要删除的值，现在则是删除满足下面条件的对象:

```cpp
bool badValue(int );
```

* 对于`vector`,`string`或`deque`使用`erase-remove_if`, 对于`list`，则使用`list::remove_if`

比较简单， 我们可以直接将上面的`remove`算法换成[`remove_if`][^1]:

```cpp
c.erase(remove_if(c.begin(), c.end(), removeVal), c.end);
c.remove_if(badValue);
```

* 对于关联容器`set`/`map` 

我们不能简单地使用`erase`, 可以使用:

  * `remove_copy_if`和`swap`
  
  这种方法直接简单但是效率比较低，因为要移动元素
  
  ```cpp
  AssocContainer<int> c;
  AssocContainer<int> goodValues;
  remove_copy_if(c.begin(), c.end(), 
                inserter(goodValues, goodValues.end()), 
                badValue);
  c.swap(goodValues);
  ```
  
  其中`remove_copy_if`功能是将`c`中满足`badValue`的元素复制到`goodValues`中。
  
  * 循环遍历时对于传给`erase`的迭代器要进行后缀递增
  
  ```cpp
  for(AssocContainer<int>::iterator i = c.begin(); i != c.end(); ) {
    if( badValue(*i) )
      c.erase(i++);
    else
      ++i;
  }
  ```
  其中关键点就在于传递给`erase`的迭代器要使用后缀递增， 而不能放在`for`循环里， 因为删除元素之后，指向该元素的所有迭代器都变得无效。

## 删除对象同时还需要其它操作

这种情况其实是在上面又加了一条约束， 对于关联容器，我们只需要添加相应操作即可。 而对于顺序容器，则必须采用类似关联容器的方法。

* 对于`vector`,`string`，`deque`或`list`标准序列容器，需要循环遍历容器中元素，需要注意的是每次调用`erase`时，需要用返回值更新迭代器

```cpp
for(SeqContainer<int>::iterator i = c.begin(); i != c.end(); ) {
  if( badValue(*i) )
    // other operation
    i = c.erase(i);
  else
    ++i;
}
```

关键点在于删除时需要使用`erase`返回的迭代器来更新循环迭代器 

* 对于关联容器`set`/`map`,循环遍历时对于传给`erase`的迭代器要进行后缀递增

```cpp
for(AssocContainer<int>::iterator i = c.begin(); i != c.end(); ) {
  if( badValue(*i) )
    // other operation
    c.erase(i++);
  else
    ++i;
}
```

# 参考

[1]: http://en.cppreference.com/w/cpp/algorithm/remove "remove"
* [remove/remove_if](http://en.cppreference.com/w/cpp/algorithm/remove)  
* [vector::erase](http://en.cppreference.com/w/cpp/container/vector/erase)  
* [remove_copy/remove_copy_if](http://en.cppreference.com/w/cpp/algorithm/remove_copy)  
