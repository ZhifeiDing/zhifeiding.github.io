---
title : DEFLATE - A Lossless Data Compression Algorithm
categories : [programming]
tags : [compression, algorithm]
---

# What's DEFLATE ?

*DEFLATE* 是结合[LZ77](https://en.wikipedia.org/wiki/LZ77_and_LZ78)和[Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding)的一种无损压缩算法。广泛用于*gzip*, *png*和*zip*压缩文件中。

# Algorithm Description

在介绍*DEFLATE*之间， 我们需要先理解上面提到的*DEFLATE*的两个重要组成部分 : *LZ77* 和 *Huffman Coding*。

## Huffman Coding

*Huffman*是一种变长前缀编码格式， 对于出现频率高的字符， 使用短的编码， 而频率低的字符， 使用长编码。
例如， 对于如下权重的字符,

```cpp
    A    16
    D     8
    E     8
```
使用*Huffman*编码得到的*Huffman Tree*如下:

```cpp
       ( )
    0 /   \ 1
    ( )    A
 0 /   \ 1
  D     E
```


## LZ77 Compression

*LZ77*算法使用 *sliding windows* 来记录前面的数据流，当后面的数据流和*sliding windows*里数据重复时，可以使用距离之前重复数据出现的距离和重复的长度来代替实际数据。
以下面数据为例:
```cpp
Blah blah blah blah blah!
```
可以看到， 当读入`Blah b`之后， 接下来的五个字符和之前的是一样的， 所以用*LZ77*算法表示为:

```cpp
Blah b[D=5,L=5]
```


## DEFLATE Algorithm

*DEFLATE*算法结合上面介绍的*LZ77 Compression*算法和*Huffman Code*，来对数据进行压缩处理。其中*LZ77*用来对数据中重复出现的部分进行处理， 而*Huffman Coding*则对*LZ77*处理过的数据进行编码， 将出现频率高的用更短的编码。

具体的对于*DEFLATE*压缩后的数据是由一系列的*block*组成的， 每个*block*格式如下:

* 第一个 bit: 是否为连续的数据中最后一个*block*

```cpp
1: 是最后一个block
0: 后面还有连续的block
```

* 第二和三 bits: 当前*block*的编码方式:

```cpp
00: 原始数据, 0到65,535字节长度
01: 使用静态Huffman 编码
10: 使用动态Huffman 编码
11: 保留
```


# Reference

* [wikipedia - DEFLATE](https://en.wikipedia.org/wiki/DEFLATE)
* [An Explanation of the Deflate Algorithm](http://www.zlib.net/feldspar.html)
