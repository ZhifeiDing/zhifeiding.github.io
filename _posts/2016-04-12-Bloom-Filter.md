---
title : Bloom Filter
categories : programming
tags : [c++, data structure]
---

# 什么是*Bloom Filter* ?

> *Bloom Filter*是一种_space efficient probabilistic data structure_，其使用`m bit`来记录`n`个记录。每个记录会用`k`个hash函数来影射到`m bit`中的`k`位上。

> 通常`k`是常值，并且比m小很多，而`m`的值和要插入元素`n`有关。一般`m`增大则false positive rate下降，而`n`增大，false positive rate上升。

由其结构可知，*Bloom Filter*可以用来查询记录是否存在。并且不会存在false negative, 但是会有false positive。因此一般*Bloom Filter*没法删除。
如下图所示*Bloom Filter*, `m = 18, k = 3`:
![bloom filter](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Bloom_filter.svg/640px-Bloom_filter.svg.png)

# *Bloom FIlter* 有什么应用 ?

* Cache filter ：CDN网络使用*Bloom Filter*来避免缓存一次点击的数据
* Google Chrome浏览器使用*Bloom Filter*来过滤有害的URLs. 所有URL首先在一个local Bloom Filter里查询，如果返回positive则进行全面检查。

# 怎么实现*Bloom Filter* ?

* 通常*Bloom
Filter*用来查询记录是否存在，因此需要实现操作就包括插入记录和查询记录。我们可以声明如下的*BloomFilter*类：

```cpp
#include<vector>
#include "MurmurHash3.h"
#include<array>

template<typename T>
class BloomFilter {
public:
   // constructor
   BloomFilter(int size, int num);
   // insert element
   void insert(const T *v, size_t len);
   // check whether the bloom filter contains the element
   bool exists(const T *v, size_t len) const;
private:
   std::vector<bool> bloomFilter;
   int numHashes;
   // generate the nth hash values
   uint64_t nthHash(int n, std::array<uint64_t,2> &hashVals) const;
   // generate 128bit hash value using MurmurHash3
   std::array<uint64_t,2> hash(const T *data, size_t len) const;
};
```

* *BloomFilter*类构造函数， 初始化内部变量：

```cpp
// constructor
template<typename T>
BloomFilter<T>::BloomFilter(int size, int num) {
   numHashes = num;
   bloomFilter.resize(size);
}
```

* *BloomFilter*类成员函数`insert(T v)`

```cpp
// insert element
template<typename T>
void BloomFilter<T>::insert(const T *v,size_t len) {
   std::array<uint64_t,2> hashValues = hash(v,len);
   for(int i = 0; i < numHashes; ++i)
      bloomFilter[nthHash(i,hashValues)] = true;
}

```

* *BloomFilter*类成员函数`exists(T v)`

```cpp
// check if element exists in BloomFilter
template<typename T>
bool BloomFilter<T>::exists(const T *v,size_t len) const {
   std::array<uint64_t,2> hashValues = hash(v,len);
   for(int i = 0; i < numHashes; ++i) {
      if( bloomFilter[static_cast<int>(nthHash(i,hashValues))] == false )
         return false;
   }
   return true;
}
```

* *BloomFilter*类私有函数`nthHash()`

*BloomFilter*最重要的是`hash`函数之间相互独立，实际中我们采用的是*double
hash*来获得多个相互独立的`hash`值

```cpp
// private function : generate nth hash value
template<typename T>
uint64_t BloomFilter<T>::nthHash(int n, std::array<uint64_t,2> &hashValues) const {
   return ( hashValues[0] + n * hashValues[1]) % bloomFilter.size();
}
```

`hash`函数使用了128bit的*MurmurHash3*函数：

```cpp
template<typename T>
std::array<uint64_t,2> BloomFilter<T>::hash(const T *data, size_t len) const {
    std::array<uint64_t,2> hashValues;
    MurmurHash3_x64_128(data, len, 0, hashValues.data());
    return hashValues;
}
```

# 测试程序

简单的测试了一下`BloomFilter`类的实现:

```cpp
#include <iostream>
#include "BloomFilter.hpp"

using namespace std;

int main() {
    vector<string> s = {"hello", "world", "good", "morning"};
    int size = 25, num = 3;
    BloomFilter<char> bf(size,num);
    cout << "Insert : ";
    for(auto itr : s) {
        cout << itr << "\t";
        bf.insert(itr.c_str(),itr.size());
    }
    cout << endl;
    vector<string> test = {"world", "morning", "China", "Red"};
    for(auto itr : test) {
        cout << itr << " : ";
        if( bf.exists(itr.c_str(), itr.size()) )
            cout << "maybe exist in BloomFilter";
        else
            cout << "don't in BloomFilter";
        cout << endl;
    }
    return 0;
}
```

测试输出:

```cpp
Insert : hello	world	good	morning
world : maybe exist in BloomFilter
morning : maybe exist in BloomFilter
China : don't in BloomFilter
Red : don't in BloomFilter
```

# 参考

1.[wikipedia - Bloom Filter](https://en.wikipedia.org/wiki/Bloom_filter)
2.[how to write a bloom filter](http://blog.michaelschmatz.com/2016/04/11/how-to-write-a-bloom-filter-cpp/)
3.[MurmurHash3](https://github.com/aappleby/smhasher)
4.[Network Applicatio of Bloom Filter](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=6CA79DD1A90B3EFD3D62ACE5523B99E7?doi=10.1.1.127.9672&rep=rep1&type=pdf)
