---
title : Sorting Algorithm
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

![bubbleSort](https://upload.wikimedia.org/wikipedia/commons/c/c8/Bubble-sort-example-300px.gif)

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

![insertionSort](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)

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

![selectionSort](https://upload.wikimedia.org/wikipedia/commons/9/94/Selection-Sort-Animation.gif)

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

  合并排序应用分治法，将数据分成两部分， 分别排好序之后将其合并成同一个序列。
  其时间复杂度为`O(nlogn)`,而空间复杂度为`O(n)`。

![mergeSort](https://upload.wikimedia.org/wikipedia/commons/c/cc/Merge-sort-example-300px.gif)

```cpp
// merge sort
template<typename T>
void merge(std::vector<T> &nums, int left, int mid, int right) {
    std::vector<T> r;
    int i = left, j = mid+1;
    while( i <= mid && j <= right ) {
        if( nums[i] <= nums[j] ) {
            r.push_back(nums[i]);
            ++i;
        } else {
            r.push_back(nums[j]);
            ++j;
        }
    }
    if( j <= right ) {
        i = j;
        mid = right;
    }
    while( i <= mid ) {
        r.push_back(nums[i]);
        ++i;
    }
    copy(r.begin(), r.end(), nums.begin()+left);
}

template<typename T>
void mergeSort(std::vector<T> &nums, int left, int right) {
    if( right <= left )
        return;
    int mid = left + ( right - left ) / 2;
    mergeSort(nums, left, mid);
    mergeSort(nums, mid+1, right);
    merge(nums, left, mid, right);
}

template<typename T>
void mergeSort(std::vector<T> &nums) {
    mergeSort(nums, 0, nums.size()-1);
}
```

## 堆排序

  堆排序是选择排序一种，
  但是由于使用堆这种数据结构，使得选择最大（最小）值的过程复杂度变成`O(logn)`,因此堆排序比选择排序更加efficient。堆排序的时间复杂度是
  `O(nlogn)`。

```cpp
// heap sort
template<typename T>
void siftDown(std::vector<T> &nums, int start, int end) {
    while( 2 * start + 1 <= end ) {
        int child = 2 * start + 1;
        int swap = start;
        if( nums[start] < nums[child] )
            swap = child;
        if( child + 1 <= end && nums[swap] < nums[child+1] )
            swap = child + 1;
        if( swap == start )
            return;
        else {
            std::swap(nums[swap], nums[start]);
            start = swap;
        }
    }
}
template<typename T>
void heapify(std::vector<T> &nums) {
    for(int i = (nums.size()-2)/2; i >= 0; --i)
        siftDown(nums, i, nums.size()-1);
}
template<typename T>
void heapSort(std::vector<T> &nums) {
    // build the heap
    heapify(nums);
    // exchange the largest value to the end of the array
    // then call siftDown() to rebalance the heap[0..end]
    int end = nums.size()-1;
    while( end ) {
        std::swap(nums[0], nums[end]);
        --end;
        siftDown(nums, 0, end);
    }
}
```

## 快排

  快排也是分治法的一种，平均时间复杂度是`O(nlogn)`，在数据有序情况下最坏时间复杂度是`O(n^2)`。快排主要思想是把数据分成两部分，和选定一个值比较，大的数据在一边，小的在另一边，重复这一步只到两边数据为空或只有一个。这样整个数据就拍好序了。因此，选定值的好坏关系到快排的时间复杂度。

```cpp
// quick sort
template<typename T>
int pivot(std::vector<T> &nums, int start, int end) {
    int mid = start + ( end - start )/2;
    int p = mid;
    if( nums[start] < nums[end] ) {
        if( nums[mid] <= nums[start] )
            p = start;
        else if( nums[end] < nums[mid] )
            p = end;
    } else {
        if( nums[mid] <= nums[end] )
            p = end;
        else if( nums[start] < nums[mid] )
            p = start;
    }
    return nums[p];
}
template<typename T>
int partition(std::vector<T> &nums, int start, int end) {
    int p = pivot(nums, start, end);
    int l = start - 1, r = end + 1;
    while( l < r ) {
        while( nums[++l] < p );
        while( p < nums[--r] );
        if( l >= r )
            return r;
        std::swap(nums[l], nums[r]);
    }
    return r;
}
template<typename T>
void quickSort(std::vector<T> &nums, int start, int end) {
    if( end <= start )
        return;
    int p = partition(nums, start, end);
    quickSort(nums, start, p);
    quickSort(nums, p + 1, end);
}
template<typename T>
void quickSort(std:vector<T> &nums) {
    quickSort(nums, 0, nums.size()-1);
}
```

# 各种排序算法的复杂度
  上面列举了六种排序算法，
  其中冒泡排序，选择排序和插入排序时间复杂度都是`O(n^2)`,
  而堆排序，合并排序和快排平均时间复杂度都是`O(nlogn)`。

# 参考
