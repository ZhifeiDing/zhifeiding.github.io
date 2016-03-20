---
title : sorting algorithm
categories : programming
tags : [sort, c++, algorithm]
---

    无论是算法学习还是实际应用中，排序算法可谓是不可避过的一环。由于排序的重要性，因此各种排序算法也是层出不穷。最近在整理一些算法，所以也把排序算法整理一下，记录一下。

# 排序算法的分类

    排序算法根据不同比较方法可以下面这些分类方法:
    * 根据排序方式是否是比较可以分为 comparison based , 比如插入排序和桶排序
    * 根据空间复杂度可以分为是否 ```in-place``` ， 比如插入排序和合并排序
    * 根据时间复杂度可以分为```O(n^2)```, ```O(nlogn)```和```O(log^2n)```
    * 根据相同值是否保持相对位置可以分为```stable```

# 具体算法分析和实现

## 冒泡排序

    冒泡排序时间复杂度是```O(n^2)```, 空间复杂度是```O(1)```, 所以是  ```in-place comparison based stable``` 排序算法。 因为时间复杂度冒泡排序很少会被使用， 不过由于原理和实现简单， 所以也是出现很频繁。 冒泡排序，就如名字所示， 就像冒泡一样， 大的往下沉，这样最后整个数据就是有序的。因此也被称为```sinking sort```

[!bubbleSort](https://upload.wikimedia.org/wikipedia/commons/c/c8/Bubble-sort-example-300px.gif)

```cpp
// bubble sort
template<typename T>
void bubbleSort(std::vector<T> &nums) {
    for(int i = nums.size()-1; i > 0; --i)
        for(int j = 0; j < i; ++j)
            if( nums[j+1] < nums[j] )
                std::swap(nums[j+1], nums[j]);
}
```

## 插入排序

    插入排序时间复杂度也是```O(n^2)```, 空间复杂度是```O(1)```, 所以是  ```in-place comparison based stable``` 排序算法。 不过在数据已经有序情况下复杂度可以下降到```O(n)```。
    插入排序可以理解成一组数据分成拍好序前半部分和无序的后半部分。我们每次取无序部分第一个数据和有序部分从后往前比较，一直到找到前面一个数据不大于这个数位置，然后将这个数据插入。接下来重复只到无序部分没有数据。

[!insertionSort](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)

```cpp
// insertion sort
template<typename T>
void insertionSort(std::vector<T> &nums) {
    for(int i = 1; i < nums.size(); ++i) {
        int j = i - 1, val = nums[i];
        while( j >= 0 && val < nums[j] ) {
            nums[j+1] = nums[j];
            --j;
        }
        nums[j+1] = val;
    }
}
```

## 选择排序

    选择排序时间复杂度也是```O(n^2)```, 空间复杂度是```O(1)```, 所以也是```in-place comparison based``` 排序算法, 但不是```stable```算法 。 而且在数据已经有序情况下复杂度不会像插入排序一样变化.
    选择排序和插入排序类似， 也可以理解成一组数据分成拍好序前半部分和无序的后半部分。只是我们接下来从无序部分找到最小值插入到有序部分后面。接下来重复只到无序部分没有数据。

[!selectionSort](https://upload.wikimedia.org/wikipedia/commons/9/94/Selection-Sort-Animation.gif)

```cpp
template<typename T>
void selectionSort(std::vector<T> &nums) {
    for(int i = 0; i < nums.size() - 1; ++i) {
        int j = i + 1;
        int minVal = i;
        while( j < nums.size() ) {
            if( nums[j] < nums[minVal] )
                minVal = j;
            ++j;
        }
        std::swap(nums[i], nums[minVal]);
    }
}
```

## 合并排序

## 堆排序

## 快排

# 各种排序算法的复杂度

