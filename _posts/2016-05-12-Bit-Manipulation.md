---
title : Bit Manipulation
categories : programming
tags : [c++,skills]
---

# *Bit Manipulation*定义

> 直接操作*bit*或少于一个字节的操作就是*Bit Manipulation*。通常使用的*Bit Manipulation*包括:_AND(与)_, _OR(或)_, _XOR(异或)_, _NOT(非)_, 和 _bit shifts(位移)_。


> 一些情况下，*Bit Manipulation*代码比较难实现和维护，但是使用*Bit Manipulation*能够避免使用循环并且提高运行速度。

# *Bit Manipulation*详解

## *Bit Manipulation*基本概念

*Bit Manipulation*的核心是为运算符：`& (and)`, `| (or)`, `~ (not)` ，`^ (xor)` 和 `移位运算 a << b和a >> b`.

> 一些*Bit Manipulation*的意义：

* 并集： `A | B`
* 交集： `A & B`
* 集合减法： `A & ~B`
* 位取反： `^ A or ~A`
* `bit`置`1`: `A |= 1 << bit`
* `bit`清`0`: `A &= ~(1 << bit)`
* 测试`bit`: `(A & 1 << bit) != 0`
* 提取最后一个`1-bit`: `A&-A or A&~(A-1) or x^(x&(x-1))`
* 清零最后一个`1-bit`: `A&(A-1)`

## *Bit Manipulation*示例

* 计算给定数的二进制表示中1的个数

```cpp
int count_one(int n)
{
    while(n)
    {
        n = n&(n-1);
        count++;
    }
    return count;
}
```

* 是否是4的幂次方

```cpp
bool isPowerOfFour(int n) 
{
    return !(n&(n-1)) && (n&0x55555555);
    //check the 1-bit location;
}
```

### `^` 技巧

* 使用`^`和`&`计算整数加法

```cpp
int getSum(int a, int b) 
{
    return b==0? a:getSum(a^b, (a&b)<<1); //be careful about the terminating condition;
}
```

* 给定一个从0, 1, 2, ..., n的包含n的不同数字的数组，找到缺少的一个数字。例如，给定数组[0, 1, 3] 返回2。

```cpp
int missingNumber(vector<int>& nums) 
{
    int ret = 0;
    for(int i = 0; i < nums.size(); ++i)
    {
        ret ^= i;
        ret ^= nums[i];
    }
    return ret^=nums.size();
}
```

### `|` 技巧

> `|`可以保留尽可能多的`1-bits`

* 找到给定N下最大的2的幂的数

```cpp
long largest_power(long N)
{
    //changing all right side bits to 1.
    N = N | (N>>1);
    N = N | (N>>2);
    N = N | (N>>4);
    N = N | (N>>8);
    N = N | (N>>16);
    return (N+1)>>1;
}
```

### `&`技巧

> 可以用来选择想要的_bit_

* 翻转整数

```cpp
x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
x = ((x & 0xffff0000) >> 16) | ((x & 0x0000ffff) << 16);
```

* 给定一个范围[m, n] 其中0 <= m <= n <= 2147483647, 返回范围内所有数的位与值。 例如给定范围 [5, 7], 返回4.

```cpp
int rangeBitwiseAnd(int m, int n) 
{
    int a = 0;
    while(m != n)
    {
        m >>= 1;
        n >>= 1;
        a++;
    }
    return m<<a; 
}
```

> 所有DNA序列都是由*A, C, G* 和 *T* 组成, 例如: *"ACGAATTCCG"*. 识别*DNA*内重复序列是一项重要研究. 找到*DNA*序列里重复且长度为10的子串.

> 例如,

> `s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"`

> 返回`["AAAAACCCCC", "CCCCCAAAAA"]`

```cpp
class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s) 
    {
        int sLen = s.length();
        vector<string> v;
        if(sLen < 11) return v;
        char keyMap[1<<21]{0};
        int hashKey = 0;
        for(int i = 0; i < 9; ++i) hashKey = (hashKey<<2) | (s[i]-'A'+1)%5;
        for(int i = 9; i < sLen; ++i)
        {
            if(keyMap[hashKey = ((hashKey<<2)|(s[i]-'A'+1)%5)&0xfffff]++ == 1)
                v.push_back(s.substr(i-9, 10));
        }
        return v;
    }
};
```

> 给定一个大小为n的数组, 找到*majority element*. *majority element*是其中出现超过⌊ n/2 ⌋次的元素.

```cpp
int majorityElement(vector<int>& nums) 
{
    int len = sizeof(int)*8, size = nums.size();
    int count = 0, mask = 1, ret = 0;
    for(int i = 0; i < len; ++i)
    {
        count = 0;
        for(int j = 0; j < size; ++j)
            if(mask & nums[j]) count++;
        if(count > size/2) ret |= mask;
        mask <<= 1;
    }
    return ret;
}
```


> 给定一个整数数组, 除了一个元素其它都出现3次. 找到只出现一次的元素. 

```cpp
//inspired by logical circuit design and boolean algebra;
//counter - unit of 3;
//current   incoming  next
//a b            c    a b
//0 0            0    0 0
//0 1            0    0 1
//1 0            0    1 0
//0 0            1    0 1
//0 1            1    1 0
//1 0            1    0 0
//a = a&~b&~c + ~a&b&c;
//b = ~a&b&~c + ~a&~b&c;
//return a|b since the single number can appear once or twice;
int singleNumber(vector<int>& nums) 
{
    int t = 0, a = 0, b = 0;
    for(int i = 0; i < nums.size(); ++i)
    {
        t = (a&~b&~nums[i]) | (~a&b&nums[i]);
        b = (~a&b&~nums[i]) | (~a&~b&nums[i]);
        a = t;
    }
    return a | b;
}
```

> 注意 : 左移或右移过多位的结果是undefined。负数右移行为也是undefined。位与和位或运算优先级比比较低


### BITSET

> `bitset`存储为值(元素只有两个可能的值: 0 或 1, true 或 false, ...).

```cpp
// bitset::count
#include <iostream>       // std::cout
#include <string>         // std::string
#include <bitset>         // std::bitset

int main ()
{
  std::bitset<8> foo (std::string("10110011"));
  std::cout << foo << " has ";
  std::cout << foo.count() << " ones and ";
  std::cout << (foo.size()-foo.count()) << " zeros.\n";
  return 0;
}
```

# 参考

* 1. [a-summary-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently](https://discuss.leetcode.com/topic/50315/a-summary-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently)  
