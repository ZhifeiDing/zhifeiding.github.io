---
title : String Searching Algorithm
categories : programming
tags : [algorithm, c++, string search]
---

  字符串查找算法(String Searching algorithm) 就是在给定字符串(Text)里查找给定字符串(Pattern)。关于字符串查找也有很多不同算法，最近学习了一下比较出名的**KMP算法**和**Boyer-Moore算法**， 自己实现了一下。

# Native String Searching

  何谓**Native** ?就是自己本能想到的。 那么看到字符串查找， 我们想到什么呢? 我就想到了， 这还不简单啊， 把Text和Pattern从第一个开始一个一个比较， 如果都相同，就查找到了， 否则就把从Text第二个开始直到找到或者Text剩下字符串不足Pattern长度。
  既然这么简单，那我们就实现出来看看吧。

```cpp
int nativeStringSearch(string &text, string &p) {
    for(int i = 0; i < text.size() - p.size(); ++i) {
        int j;
        for(j = 0; j < p.size() && text[i+j] == p[j]; ++j);
        if( j == p.size() )
            return i;
    }
    return -1;  // not found
}
```

  看， 代码也够简单的。那为什么还有其他算法呢？我们看看上面的时间复杂度居然是O(n*m)，很明显这个实现不够efficient。 那有没有办法查找得更快的算法呢？所以就有了下面的KMP算法。

# KMP算法

  大名鼎鼎的***KMP算法***其实是**Knuth–Morris–Pratt**的缩写， 而里面每一个人都是一个人名。上面的native string searching是从Text一个一个移动，为了提高比较速度， 我们想能不能在Text和Pattern不等时不要像上面一样一个一个移动呢？如果可以那又该怎么办呢？
  为了提高比较速度， 当Text与Pattern不匹配时，***KMP算法***利用pattern自身的特点来跳过一些native string searching算法中比较过程。原理就是当遇到不匹配字符时， 前面已经匹配到的子串中如果存在前缀子串则可以从匹配到前缀子串处开始比较。具体例子可以参考[wikipedia](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)

## 部分匹配表
  部分匹配表其实就是当Pattern和Text不匹配时，在已经匹配的子串中前缀子串和后缀子串匹配的长度来决定Text移动长度。

  部分匹配表生成代码:

```cpp
void KMPTable(string &p, vector<int> &kmp_table) {
    kmp_table[0] = -1;
    if( p.size() <= 2 )
        return;
    int pos = 2; // the current position
    int cnd = 0; // the next character of current candidate
    while( pos < p.size() ) {
        if(p[pos-1] == p[cnd]) {
            kmp_table[pos] = cnd;
            ++pos;
            ++cnd;
        } else if( cnd > 0 ) {
            cnd = kmp_table[cnd];
        } else {
            kmp_table[pos] = 0;
            ++pos;
        }
    }
}
```

## KMP算法实现

  在部分匹配表已经生成情况下***KMP算法***实现比较简单的，使用上面生成的部分匹配表的***KMP算法***代码:

```cpp
int KMPSearch(string &text, string &p) {
    vector<int> kmp_table(p.size(),0);
    KMPTable(p, kmp_table);

    int m = 0; // the index of text
    int i = 0; // the index of p
    while( m + p.size() <= text.size() ) {
        if( text[m+i] == p[i] ) {
            if( i == p.size() - 1 )
                return m;
            ++i;
        } else if( kmp_table[i] >= 0 ) {
            m = m + i - kmp_table[i];
            i = kmp_table[i];
        } else {
            m = m + 1;
            i = 0;
        }
    }
    return -1;
}
```

在部分匹配表已经存在情况下， 上面代码的时间复杂度是`O(n)`(n是Text长度), 而计算部分匹配表的时间复杂度是`O(k)`(k是Pattern的长度). 所以KMP总的时间复杂度是`O(k+n)`。

# Boyer-Moore 算法

了解了上面的***KMP算法***之后， 你可能会疑惑， 怎么还有***Boyer-Moore算法***? 难道还能比***KMP算法***更efficient？
在回答上面问题之前， 我们先了解一下***Boyer-Moore算法***原理。

与`KMP`算法类似， `Boyer-Moore`算法也是利用Pattern本身特点来跳过一些比较。不同的是，`Boyer-Moore`算法使用了两种规则来shift pattern，并且从右往左比较。.

* 坏字符规则

当比较的字符不等时，如果我们可以将该字符和在pattern中出现的该字符对齐比较。如果不存在我们可以直接将pattern移到该字符之后开始比较。

```cpp
// build the bad character shift rule table
void BMBadChar(string &p, vector<int> &badChar) {
    // we aasume badChar has been initialized and
    // have default value of pattern's length
    for(int i = 0; i < p.size(); ++i)
        // if the mismatched char is found in pattern
        // we can shift the pattern to match this char
        badChar[p[i]] = p.size() -1 - i;
}
```

* 好后缀规则

这儿好后缀指的是当pattern和Text出现mismatch时候，如果已经match的子串在pattern里存在则我们可以将pattern移动来使子串与其对齐来比较。如果不存在但是pattern前缀和和子串后缀有匹配，则可以将其对其来比较。如果都不存在则将pattern移到mismatch字符之后开始比较。

```cpp
// check whether the substring p[p..p.size()-1] is a prefix of itself
bool isPrefix(string &p, int idx) {
    for(int i = 0, j = idx; j < p.size(); ++i, ++j)
        if( p[i] != p[j] )
            return false;
    return true;
}

// record the matched suffix's length
int suffix(string &p, int idx) {
    int i, j;
    for(i = idx, j = p.size()-1; i >= 0 && p[i] == p[j]; --i,--j);
    return idx - i;
}

// build the good suffix shift rule table
void BMGoodSuffix(string &p, vector<int> &goodSuffix) {
    // first check that the matched suffix has a matched prefix
    int lastMatchedPrefix = p.size();  // record the last matched prefix's index
    for(int i = p.size() - 1; i >= 0; --i) {
        if( isPrefix(p,i+1) )
            lastMatchedPrefix = i+1;
        // when there's matched prefix we should shift the pattern to the
        // matched the suffix
        goodSuffix[p.size() - 1 - i] = lastMatchedPrefix - i + p.size() - 1;
    }

    // second if there exists substring matched with the matched suffix
    // we should shift pattern to match the substring instead of previous prefix
    for(int i = 0; i < p.size(); ++i) {
        int slen = suffix(p,i);
        goodSuffix[slen] = slen + p.size() - 1 - i;
    }
}
```

## *Boyer-Moore*算法

有了上面坏字符和好后缀规则，我们可以在每次比较fail的时候来移动两者间大值来跳过很多必然fail的比较。

```cpp
int BMSearch(string &text, string &p) {
    // initialize bad character table to pattern's length
    vector<int> badChar(256, p.size());
    BMBadChar(p,badChar);

    // declare of the good suffix shift table
    vector<int> goodSuffix(p.size());
    BMGoodSuffix(p, goodSuffix);

    int i = p.size() - 1; // index of text
    while( i <= text.size() - p.size() ) {
        int j = p.size() - 1;
        for(; j >= 0 && p[j] == text[i]; --j,--i);
        if( j < 0 )
            return i+1;
        // the difference with other string searching algorithm
        // choose the larger between good suffix table and bad character table
        i += max(goodSuffix[p.size() - 1 - j], badChar[text[i]]);
    }
    return -1;  // not found
}
```

那么上面代码的时间复杂度是多少呢？我们可以很清楚知道最好情况是每次比较pattern最后一个字符时就fail，这样我们可以移动整个pattern长度，这样时间复杂度就是`O(n/m)`(m是pattern长度，n是Text长度)。那么最worst情况呢？那就是每个字符串都比较了`O(n)`。

# Boyer–Moore–Horspool 算法

从算法命名我们就能知道肯定和上面的*Boyer-Moore*算法有关系。*Boyer-Moore*算法同时使用了坏字符和好后缀两个规则来跳过一些不必要的比较。其中，好后缀规则比较相对于怀字符规则比较难实现，而且实际中发现好后缀规则使用并不高。因此，就有了只使用怀字符规则的string searching algorithm. *Boyer–Moore–Horspool*就是其中一种。

具体来说，和*Boyer–Moore*算法也是采用了*Boyer-Moore*怀字符规则，只不过不同于*Boyer-Moore*使用mismatched character来做为下次对齐不同，*Boyer-Moore-Horspool*则使用已经matched的子串中第一个字符来对齐的。

## Boyer-Moore-Horspool 坏字符规则实现

```cpp
// build Boyer-Moore-Horspool bad character table
void BMHBadChar(string &p, vector<int> &badChar) {
    for(int i = 0; i < p.size() - 1; ++i)
        badChar[p[i]] = p.size() -1 - i;
}
```

## Boyer–Moore–Horspool 算法实现

相比于上面*Boyer-Moore*算法，*Boyer-Moore-Horspool*算法实现比较简单。

```cpp
int BMHSearch(string &text, string &p) {
    // initialize bad character table to pattern's length
    vector<int> badChar(256, p.size());
    BMHBadChar(p,badChar);

    int i = p.size() - 1; // index of text
    while( i < text.size() ) {
        int j = p.size()-1; // index of pattern
        for(; j >= 0 && text[i-p.size()+1+j] == p[j]; --j);
        if( j < 0 )
            return i-p.size()+1;
        // the difference with BOyer-Moore slgorithm
        // we always choose the first matched text char
        i += badChar[text[i]];
    }
    return -1; // not found
}
```

# 测试程序

写了一个简单的程序来测试上面几种string searching algorithm。

```cpp
typedef int(*fptr)(string&,string&);
void test_strSearch(fptr func, string &text, string &p, string funcName) {
    int idx = func(text,p);
    cout << funcName << "(" << text << "," << p << ") return " << idx << endl;
}
void test_strSearch() {
    unordered_map<string, fptr> funcMap = {
        {"nativeStringSearch",nativeStringSearch},
        {"KMPSearch", KMPSearch},
        {"BMSearch", BMSearch},
        {"BMHSearch", BMHSearch}
    };
    unordered_map<string, string> TextPattern = {
        {"this is a simple example", "example"},
        {"this should have no match", "gave"},
        {"match at the begin", "match"},
        {"match in the middle", "th"}
    };
    srand((unsigned int)time(NULL));
    int i = 0;
    auto itr = TextPattern.begin();
    while( itr != TextPattern.end() ) {
        ++i;
        cout << endl << "#################" << endl;
        cout << "Test Step : " << i << endl;
        string text = (*itr).first;
        string p = (*itr).second;
        ++itr;
        for(auto itr : funcMap)
            test_strSearch(static_cast<fptr>(itr.second), text, p, static_cast<string>(itr.first));
        cout << "#################" << endl;
    }
}
int main() {
    test_strSearch();
    return 0;
}
```

测试输出是这样的 :

```cpp
#################
Test Step : 1
BMHSearch(match in the middle,th) return 9
BMSearch(match in the middle,th) return 9
KMPSearch(match in the middle,th) return 9
nativeStringSearch(match in the middle,th) return 9
#################

#################
Test Step : 2
BMHSearch(match at the begin,match) return 0
BMSearch(match at the begin,match) return 0
KMPSearch(match at the begin,match) return 0
nativeStringSearch(match at the begin,match) return 0
#################

#################
Test Step : 3
BMHSearch(this should have no match,gave) return -1
BMSearch(this should have no match,gave) return -1
KMPSearch(this should have no match,gave) return -1
nativeStringSearch(this should have no match,gave) return -1
#################

#################
Test Step : 4
BMHSearch(this is a simple example,example) return 17
BMSearch(this is a simple example,example) return 17
KMPSearch(this is a simple example,example) return 17
nativeStringSearch(this is a simple example,example) return 17
#################
```

# Reference

[1.Johns Hopkins - Boyer-Moore](http://www.cs.jhu.edu/~langmea/resources/lecture_notes/boyer_moore.pdf)  
[2.Wikipedia - Boyer-Moore](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string_search_algorithm)  
[3.Wikipedia - Knuth-Morris-Pratt](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)  

