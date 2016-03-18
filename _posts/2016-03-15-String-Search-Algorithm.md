---
title : String Searching Algorithm
categories : programming
tags : [algorithm, c++, string search]
---

  字符串查找算法(String Searching Algorithm) 就是在给定字符串(Text)里查找给定字符串(Pattern)。关于字符串查找也有很多不同算法，最近学习了一下比较出名的**KMP算法**和**Boyer-Moore算法**， 自己实现了一下。

# Native String Searching 

  何谓**Native** ?就是自己本能想到的。 那么看到字符串查找， 我们想到什么呢? 我就想到了， 这还不简单啊， 把Text和Pattern从第一个开始一个一个比较， 如果都相同，就查找到了， 否则就把从Text第二个开始直到找到或者Text剩下字符串不足Pattern长度。
  既然这么简单，那我们就实现出来看看吧。

```cpp
```

  看， 代码也够简单的。那为什么还有其他算法呢？我们看看上面的时间复杂度居然是O(n*m)，很明显这个实现不够efficient。 那有没有办法查找得更快的算法呢？所以就有了下面的KMP算法。

# KMP算法

  大名鼎鼎的KMP算法其实是Knuth–Morris–Pratt的缩写， 而里面每一个人都是一个人名。上面的native string searching是从Text一个一个移动，为了提高比较速度， 我们想能不能在Text和Pattern不等时不要像上面一样一个一个移动呢？如果可以那又该怎么办呢？ 当然方法是存在的。我们以__thisisstringsearchingexample__里查找__amp__为例说明。
  
  * 首先，我们从第一个字符串比较:
    this is searching example
    amp
    
    如果是native string searching我们下面只能移到Text第二个字符开始比较。 可以通过观察我们发现

  

```cpp
```

# Boyer-Moore 算法

```cpp
```
