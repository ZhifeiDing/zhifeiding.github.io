---
title : Bloom Filter
categories : programming
tags : [c++, data structure]
---

# 什么是*Bloom Filter* ?

> *Bloom Filter*是一种_space efficient probabilistic data structure_，其使用`m bit`来记录`n`个记录。每个记录会用`k`个hash函数来影射到`m bit`中的`k`位上。

*Bloom Filter*是_space efficient probabilistic data structure_, 可以用来查询记录是否存在。

![bloom filter](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Bloom_filter.svg/640px-Bloom_filter.svg.png)

# *Bloom FIlter* 有什么应用 ?

# 怎么实现*Bloom Filter* ?

```cpp
```

# 参考

1.[wikipedia - Bloom Filter](https://en.wikipedia.org/wiki/Bloom_filter)  
2.[how to write a bloom filter](http://blog.michaelschmatz.com/2016/04/11/how-to-write-a-bloom-filter-cpp/)  
3.[MurmurHash3](https://github.com/aappleby/smhasher)  
4.[Network Applicatio of Bloom Filter](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=6CA79DD1A90B3EFD3D62ACE5523B99E7?doi=10.1.1.127.9672&rep=rep1&type=pdf)
