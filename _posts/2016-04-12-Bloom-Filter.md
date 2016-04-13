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

```cpp
template<typename T, typename Hash = std::hash<T> >
class BloomFilter {
public:
	// constructor
	BloomFilter(int size, int numHashes);
	// insert element
	void inser(T v);
	// check whether the bloom filter contains the element
	bool exists(T v);
private:
	std::vector<bool> bloomFilter;
	int numHashes;
	// generate the nth hash values
	vector<int> nthHash(vector<int> &hashA, vector<int> &hashB);
};
template<typename T, typename Hash = std::hash<T> >
BloomFilter::BloomFilter(int size, int numHashes) : numHashes(numHashes) {
	bloomFilter.resize(size);
}

template<typename T, typename Hash = std::hash<T> >
void BloomFilter::insert(T v) {
	auto hashValues = Hash(v);
	for(int i = 0; i < numHashes; ++i)
		bloomFilter[nthHash(i,hashValues)];
}

template<typename T, typename Hash = std::hash<T> >
bool BloomFilter::exists(T v) {
	auto hashValues = Hash(v);
	for(int i = 0; i < numHashes; ++i) {
		if( bloomFilter[nthHash(i,hashValues)] == false )
			return false;
	}
	return true;
}

template<typename T, typename Hash = std::hash<T> >
vector<int> BloomFilter::nthHash(int n, vector<int> &hashValues) {
	return ( hashValues[0] + n * hashValues[1]) % bloomFilter.size();
}
```


# 参考

1.[wikipedia - Bloom Filter](https://en.wikipedia.org/wiki/Bloom_filter)  
2.[how to write a bloom filter](http://blog.michaelschmatz.com/2016/04/11/how-to-write-a-bloom-filter-cpp/)  
3.[MurmurHash3](https://github.com/aappleby/smhasher)  
4.[Network Applicatio of Bloom Filter](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=6CA79DD1A90B3EFD3D62ACE5523B99E7?doi=10.1.1.127.9672&rep=rep1&type=pdf)
