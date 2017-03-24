---
title : Concurrent Linked List Introduction and Implementation
categories : programming
tags : [data structure, c++, concurrent, lock-free]
---

# 背景介绍

在具体实现之前， 现解释一下在*Concurrent Programming* 中需要用到的一些概念。

* __RMW__ - [Read-Modify-Write](https://en.wikipedia.org/wiki/Read-modify-write)

>
__RMW__ 是指在一个操作中读取*memory* 的值，然后写入新的值的原子操作(*Atomic*)。这些指令能防止在多线程编程中出现竞争。

* __CAS__ - [Compare-And-Swap](https://en.wikipedia.org/wiki/Compare-and-swap)

>
__CAS__ 是上面的`RMW`指令中的一种，具体就是要修改一个*memory* 地址的值时，先用当前的值和之前写入的值进行比较，如果一致，则写入新值。否则写入失败。由于 __CAS__ 是原子操作，所以在多线程中可以保证同时只被一个线程修改。用`C++`可以表示如下:

```cpp
bool cas(int *ptr, int oldVal, int newVal) {
    if( *ptr == oldVal ) {
        *ptr = newVal;
        return true;
    }
    return false;
}
```

* __Sequential Consistency__

>
__Sequential Consistency__ 是指对*memory* 的读写操作的顺序在所有线程中都是和程序代码中一致的。具体的理解就是在所有线程都运行在单核,并且程序是在关掉编译器优化功能后编译的。这样对于`CPU`，所有的*memory* 读写访问操作都是确定的。

* __Memory Order__

>
__Memory Order__ 是指执行指令时`Load`和`Store`的顺序。影响*memory*
的读写指令顺序因素有两个 :

1. __compiler reordering__ : 指*compiler* 在编译优化过程中不改变程序正确性前提下对`Load`和`Store`指令执行顺序进行的调整。

对于*x86*上的`gcc`可以用下面来保证代码后面`Load/Store`不会在代码之前执行:

```cpp
inline void MemoryBarrier() {
  // prevent compiler reordering
  __asm__ __volatile__("" : : : "memory");
}
```
2. __processor reordering__ : 指*CPU*本身的`Out-of-Order`功能在执行过程中对`Load/Store`指令的调整。

对于*x86*上的`gcc`可以用下面来保证代码后面`Load/Store`不会在代码之前执行:

```cpp
inline void MemoryBarrier() {
  // prevent compiler && cpu processor reordering
  __asm__ __volatile__("mfence" : : : "memory");
}
```
* __Memory Model__

>
__Memory Model__ 是指能够执行`Out-of-Order`的`CPU`能够对`Load`和`Store`指令进行`Reorder`的方式。不同的`CPU`对应各自的`Memory Order`, 根据能够`Reorder`的总类， 一般可以分为`Strong Memory Order`和`Relaxed Memory Order`。 一般的， *X86*系列的是`Strong Memory Order`, 而`ARM`系列的则属于`Relaxed Memory Order`。 具体可见下表:

| Reordering Activity	| x86 and x64	| ARM |
| ------------------- |:-----------:|:----|
| Reads moving ahead of reads	| No | Yes |
| Writes moving ahead of writes	| No | Yes |
| Writes moving ahead of reads	| No | Yes |
| Reads moving ahead of writes	| Yes	| Yes |

* __[ABA Problem](https://en.wikipedia.org/wiki/ABA_problem)__

>
上面提到的 __CAS__ 中我们读取一个值并与其之前值进行比较， 当两者一致时，我们认为在这之间状态没有变化。但是，其他线程可能已经对数据进行一系列操作，致使最后退出时将该值修改为之前的值。 这种出现在__Lock-Free__ 结构里的现象，就被称为__ABA Problem__ 。
具体可以看下面__linklist__ `head->NodeA->NodeB->NodeC`的例子:

1. __Thread 1__ 从*memory* 里读取 `NodeA`值， 然后__Thread 1__ 被调度出，__Thread 2__ 开始执行
2. __Thread 2__ 删除`NodeA`和`NodeB`, 然后插入`NodeA`, 变成`head->NodeA->NodeC`
3. __Thread 1__ 继续执行， 删除`NodeA`, 将`head->next = NodeB`， 这是就会出现访问未分配的地址

__ABA__ 问题可以对使用指针地址加上`tag bit`来避免。 对于`x86`来说， 指针是`4 byte`, 所以后`2 bit`可以用来作为`tag bit`。

* __Read Acquire__ 和 __Write Release__ Barrier

__Read Acquire__ 和 __Write Release__ Barrier是 *Lock-Free* 里经常提到的概念， 具体的:

* __Read Acquire Barrier__ : 指该*Barrier* 之后的同一个线程的`Load/Store` 指令都在之后执行

不管是在*lock-based* 还是*lock-free* 代码中， 当我们需要对*memory* 进行操作时， 一般的都会去测试一个`lock`或者一个`pointer`来判断能否进行后面的操作。所以后面操作能否进行依赖于这个测试操作，即必须保证测试操作在之前完成， 而不能被*reorder*。所以称之为 __Read Acquire__ 。

* __Write Release Barrier__ : 指该*Barrier* 之前的同一个线程的`Load/Store` 指令都在之前执行

同样的， 当我们对*memory* 执行操作结束之后，通常都要对*memory* 进行写操作来通知其他线程当前*memory* 可用。 因此只能在当前成功完成对*memory* 操作之后才能执行该写操作。所以称之为 __Write Release__ 。

上面__Memory Order__ 提到的`MemoryBarrier()`可以用来作为阻止`Load/Store` *Reordering* 的指令。

***

# Generic Linked List

```cpp
// generic linklist without multi-thread support
template<typename T>
class linklist {
public:
    // empty-argument constructor
    explicit linklist() : head_(nullptr), tail_(nullptr) { }

    // return whether the list is empty
    bool empty() const {
        return size() == 0;
    }
    // return the size of list
    std::size_t size() const {
        std::size_t size = 0;
        node* p = head_;
        while( p ) {
            ++size;
            p = p->next_;
        }
        return size;
    }

    // return the front element
    T& front() {
        assert( head_ != nullptr );
        return head_->val_;
    }

    T& back() {
        assert( tail_ != nullptr );
        assert( tail_->next_ == nullptr );
        return tail_->val_;
    }

    // pop the front element
    void pop_front() {
        assert( head_ );
        head_ = head_->next_;
        if( head_ == nullptr )
            tail_ = head_;
    }

    // push back element
    void push_back(const T& val) {
        node* n = new node(val, nullptr);
        if( !tail_ ) {
            assert( !head_ );
            head_ = tail_ = n;
        } else {
            tail_->next_ = n;
            tail_ = tail_->next_;
        }
    }

    void remove(const T& val) {
        node* prev = nullptr;
        node* p = head_;
        node** pp = &head_;
        while( p ) {
            if( p->val_ == val ) {
                *pp = p->next_;
                p = *pp;
                if( !*pp )
                    tail_ = prev;
            } else {
                prev = p;
                pp = &p->next_;
                p = p->next_;
            }
        }
    }
private:
    // node class definition
    struct node;
    struct node {
        T val_;
        node *next_;
        // constructor
        node() : val_(), next_(nullptr) {}
        node(const T& value, node* next)
            : val_(value), next_(next) {}
        // non-copyable
        node(const node&) = delete;
        node(node &&) = delete;
        node &operator=(const node&) = delete;
    };
    node* head_;
    node* tail_;
};

```

# Linked List using Lock

## global locked linked list

```cpp
// global lock linklist implementation
// for this linklist, before every operation
// should obtain a unique_lock
// with c++11 unique_lock, we needn't manage the lock
// by ourselfves
template<typename T>
class linklist_glock {
public:
    // empty-argument constructor
    explicit linklist_glock() : head_(nullptr), tail_(nullptr), mutex_() { }

    // return whether the list is empty
    bool empty() const {
        return size() == 0;
    }
    // return the size of list
    std::size_t size() const {
        // first obtain the unique_lock
        unique_lock lock(mutex_);
        std::size_t size = 0;
        node* p = head_;
        while( p ) {
            ++size;
            p = p->next_;
        }
        return size;
    }

    // return the front element
    T& front() {
        // first obtain the unique_lock
        unique_lock lock(mutex_);
        assert( head_ != nullptr );
        return head_->val_;
    }

    T& back() {
        // first obtain the unique_lock
        unique_lock lock(mutex_);
        assert( tail_ != nullptr );
        assert( tail_->next_ == nullptr );
        return tail_->val_;
    }

    // pop the front element
    void pop_front() {
        // first obtain the unique_lock
        unique_lock lock(mutex_);
        assert( head_ );
        head_ = head_->next_;
        if( head_ == nullptr )
            tail_ = head_;
    }

    // pop the front element and return the pair
    std::pair<bool,T> try_pop_front() {
        // first obtain the unique_lock
        unique_lock lock(mutex_);
        //assert( head_ );
        if( head_ == nullptr )
            return std::make_pair(false,T());
        T v = head_->val_;
        head_ = head_->next_;
        if( head_ == nullptr )
            tail_ = head_;
        return std::make_pair(true, v);
    }

    // push back element
    void push_back(const T& val) {
        // first obtain the unique_lock
        unique_lock lock(mutex_);
        node* n = new node(val, nullptr);
        if( !tail_ ) {
            assert( !head_ );
            head_ = tail_ = n;
        } else {
            tail_->next_ = n;
            tail_ = tail_->next_;
        }
    }

    void remove(const T& val) {
        // first obtain the unique_lock
        unique_lock lock(mutex_);
        node* prev = nullptr;
        node* p = head_;
        node** pp = &head_;
        while( p ) {
            if( p->val_ == val ) {
                *pp = p->next_;
                p = *pp;
                if( !*pp )
                    tail_ = prev;
            } else {
                prev = p;
                pp = &p->next_;
                p = p->next_;
            }
        }
    }
private:
    // typedef unique_lock
    typedef std::unique_lock<std::mutex> unique_lock;
    // mutex definition
    mutable std::mutex mutex_;
    // node class definition
    struct node;
    struct node {
        T val_;
        node *next_;
        // constructor
        node() : val_(), next_(nullptr) {}
        node(const T& value, node* next)
            : val_(value), next_(next) {}
        // non-copyable
        node(const node&) = delete;
        node(node &&) = delete;
        node &operator=(const node&) = delete;
    };
    node* head_;
    node* tail_;
};
```

## node locked linked list

```cpp
// per-node lock linklist implementation
// this implementation add lock for the node which is different
// from the glock lock
template<typename T>
class linklist_nodelock {
public:
    // empty-argument constructor
    explicit linklist_nodelock() : head_(new node()), tail_(nullptr) { }

    // return whether the list is empty
    bool empty() const {
        return size() == 0;
    }
    // return the size of list
    std::size_t size() const {
        std::size_t size = 0;
        // first obtain the unique_lock of head_
        head_->mutex_.lock();
        node* prev = head_;
        node* p = head_->next_;
        while( p ) {

            // if we don't lock current node
            // it may be deleted after we unlock prev
            // node
            p->mutex_.lock();
            prev->mutex_.unlock();
            ++size;

            prev = p;
            p = p->next_;
        }
        prev->mutex_.unlock();
        return size;
    }

    // return the front element
    T& front() {
        // first obtain the unique_lock of head_
        unique_lock lock(head_->mutex_);
        node* p = head_->next_;
        assert( p != nullptr );
        return p->val_;
    }

    T& back() {
        // first obtain the unique_lock of tail_
        unique_lock lock(tail_->mutex_);
        assert( tail_ != nullptr );
        assert( tail_->next_ == nullptr );
        return tail_->val_;
    }

    // pop the front element
    void pop_front() {
        // first obtain the unique_lock of head_
        unique_lock lock(head_->mutex_);
        node* p = head_->next_;
        assert( p );

        // lock current node
        unique_lock lock0(p->mutex_);
        bool isTail = !p->next_;

        if( isTail ) {

            assert( head_->next_ == p );
            if( p->next_ ) {
                // no longer tail, retry
                return pop_front();
            }
            assert( tail_ == p );
        }
        head_->next_ = p->next_;
        if( isTail ) {
            tail_ = head_->next_;
        }
    }

    // pop the front element and return the pair
    std::pair<bool,T> try_pop_front() {
        // first obtain the unique_lock
        unique_lock lock(head_->mutex_);
        node* p = head_->next_;
        //assert( head_ );
        if( p == nullptr )
            return std::make_pair(false,T());
        unique_lock lock0(p->mutex_);

        T v = p->val_;
        bool isTail = !p->next_;
        if( isTail ) {
            assert(head_->next_ == p);
            if( p->next_ ) {
                // no longer tail, retry
                return try_pop_front();
            }
            assert( tail_ == p );
        }

        head_->next_ = p->next_;
        if( isTail )
            tail_ = head_;
        return std::make_pair(true, v);
    }

    // push back element
    void push_back(const T& val) {
        // first obtain the unique_lock
        unique_lock lock(head_->mutex_);
        node* n = new node(val, nullptr);
        if( !tail_ ) {
            assert( !head_->next_ );
            head_->next_ = tail_ = n;
        } else {
            unique_lock lock(tail_->mutex_);
            tail_->next_ = n;
            tail_ = tail_->next_;
        }
    }

    void remove(const T& val) {
        // first obtain the unique_lock
        head_->mutex_.lock();

        node* prev = head_;
        node* p = head_->next_;

        while( p ) {
            p->mutex_.lock();
            if( p->val_ == val ) {
                bool isTail = !p->next_;
                if( isTail ) {
                    assert(tail_ == p);
                }
                prev->next_ = p->next_;
                if( isTail ) {
                    tail_ = prev;
                }
                p->mutex_.unlock();
                p = prev->next_;
            } else {
                prev->mutex_.unlock();
                prev = p;
                p = p->next_;
            }
        }
        prev->mutex_.unlock();
    }
private:
    // typedef unique_lock
    typedef std::unique_lock<std::mutex> unique_lock;
    // mutex definition
    mutable std::mutex mutex_;
    // node class definition
    struct node;
    struct node {
        // mutex definition
        mutable std::mutex mutex_;
        T val_;
        node *next_;
        // constructor
        node() : val_(), next_(nullptr) {}
        node(const T& value, node* next)
            : val_(value), next_(next) {}
        // non-copyable
        node(const node&) = delete;
        node(node &&) = delete;
        node &operator=(const node&) = delete;
    };
    node* head_; // head is a sentinel node
    node* tail_;
};

```

# Lock-free Linked List

```cpp
#define CAS(addr, oldVal, newVal) __sync_bool_compare_and_swap(addr, oldVal, newVal)

template<typename T>
class linklist_lockfree {
public:
    // empty-argument constructor
    explicit linklist_lockfree() : head_(new node()), tail_(nullptr) { }

    // return whether the list is empty
    bool empty() const {
        return size() == 0;
    }
    // return the size of list
    std::size_t size() const {
        std::size_t size = 0;
        node* p = head_->next();
        while( p ) {
            if( !p->isMarked() )
                ++size;
            else{
            }
            p = p->next();
        }
        return size;
    }

    // return the front element
    T& front() {
        assert( !head_->isMarked() );
        node* p = head_->next();
        assert( p != nullptr );
        if( p->isMarked() )
            return front();
        T& val = p->val_;
        if( p->isMarked() )
            return front();

        if( !p->next() && tail_ != p )
            tail_ = p;
        return val;
    }

    T& back() {
        assert( !head_->isMarked() );
        assert( tail_ != nullptr );
        if( tail_->next() )
            return back();
        if( tail_->isMarked() )
            return back();
        return tail_->val_;
    }

    // pop the front element
    void pop_front() {

        assert( !head_->isMarked() );
        node* p = head_->next();
        //assert( p );

        if( p == nullptr )
            return;
        if( p->isMarked() )
            return pop_front();
        // mark current node
        p->mark();
        if( !p->isMarked() )
            return pop_front();

        //node* pp = head_->next();
        if( !CAS(&head_->next_ , p, p->next()) )
            return pop_front();

        CAS(&tail_, p, head_->next());
    }

    // pop the front element and return the pair
    std::pair<bool,T> try_pop_front() {

        assert( !head_->isMarked() );
        node* p = head_->next();
        //assert( p );
        if( p == nullptr )
            return std::make_pair(false,T());
        if( p->isMarked() )
            return try_pop_front();

        p->mark();
        if( !p->isMarked() )
            return try_pop_front();
        //node* pp = head_->next();
        if( !CAS(&head_->next_, p, p->next()) )
            return try_pop_front();
        CAS(&tail_, p, head_->next());

        return std::make_pair(true, p->val_);
    }

    // push back element
    void push_back(const T& val) {

        //assert( !head_->isMarked() );

        node* n = new node(val, nullptr);
        if( !tail_ ) {
            if( !head_->next() ) {
                if( !CAS(&head_->next_, tail_, n) )
                    return push_back(val);
                tail_ = head_->next_;
            } else
                return push_back(val);
        } else {
            if( tail_->next() || tail_->isMarked() )
                return push_back(val);
            tail_->setNext(n);
            tail_ = tail_->next();
        }
    }

    void remove(const T& val) {

        node* prev = head_;
        node* p = head_->next();

        while( p ) {
            if( p->val_ == val ) {
                p->mark();
                if( !CAS(&prev->next_, p, p->next()) )
                    return remove(val);
                CAS(&tail_, p, head_->next());
                p = prev->next();
            } else {
                prev = p;
                p = p->next();
            }
        }
    }
private:
    // node class definition
    struct node;
    struct node {
        T val_;
        node* next_;
        // constructor
        node() : val_() , next_(nullptr) {}
        node(const T& value,node* p) : val_(value), next_(p) {}
        // return if current node is marked
        inline bool isMarked() {
            return intptr_t(next_) & 0x1;
        }
        inline void mark() {
           next_ = (node*)(intptr_t(next_) | 0x1);
        }
        inline node* next() {
            return (node*)(intptr_t(next_) & ~0x1);
        }
        inline void setNext(node* ptr) {
            next_ = ptr;
        }
        // non-copyable
        node(const node&) = delete;
        node(node &&) = delete;
        node &operator=(const node&) = delete;
    };
    node* head_; // head is a sentinel node
    node* tail_;
};
```

# 测试程序

```cpp
#include "linklist.hpp"
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <functional>
#include <algorithm>

using namespace std;

template<typename list>
void single_threaded_test() {
    list l;
    assert(l.empty());

    l.push_back(1);
    assert( l.front() == 1 );
    assert( l.back() == 1 );
    assert( l.size() == 1 );

    l.push_back(2);
    assert( l.front() == 1 );
    assert( l.back() == 2 );
    assert( l.size() == 2 );

    l.pop_front();
    assert( l.front() == 2 );
    assert( l.back() == 2 );
    assert( l.size() == 1 );

    l.pop_front();
    assert( l.empty() );

    l.push_back(10);
    l.push_back(10);
    l.push_back(20);
    l.push_back(30);
    l.push_back(10);
    assert( l.front() == 10 );
    assert( l.back() == 10 );
    assert( l.size() == 5 );

    l.remove(10);
    while( !l.empty() ) {
        assert( l.front() != 10 );
        l.pop_front();
    }
}

template<typename list, typename T>
void getElemsOfList(list &l, vector<T> &res) {
    while( !l.empty() ) {
        res.push_back(l.front());
        l.pop_front();
    }
}

template<typename T>
void rangeCompare(vector<T>& res, int start, int end) {
    assert( res.size() == end - start );
    for(int i = start; i < end; ++i)
        assert(res[i-start] == i);
}

inline void nop_pause() {
    __asm__ volatile ("pause" ::);
}

template<typename list, typename T>
void list_insert(list &l, atomic<bool> &f, T start, T end) {
    while( !f.load() )
        nop_pause();
    for(T i = start; i < end; ++i)
        l.push_back(i);
}

template<typename list, typename T>
void list_pop_front(list &l, atomic<bool> &f, atomic<bool> &stop, vector<T> &res) {
    while( !f.load() )
        nop_pause();
    while( true ) {
        auto val = l.try_pop_front();
        if( val.first == false && stop.load() )
            break;
        if( val.first )
            res.push_back(val.second);
    }
}

template<typename list, typename T>
void list_remove(list &l, atomic<bool> &f, T start, T end) {
    while( !f.load() )
        nop_pause();
    for(T i = start; i < end; ++i)
        l.remove(i);
}


template<typename list>
void multiple_threaded_test() {
    // try a bunch of concurrent inserts
    // make sure no value lost
    {
        list l;
        const int NUM_ELEMENTS_PER_THREAD = 2000;
        const int NUM_THREADS = 8;
        vector<thread> threads;
        atomic<bool> start_flag(false);
        for(int i = 0; i < NUM_THREADS; ++i) {
            thread t(list_insert<list, int>, ref(l), ref(start_flag),
                i * NUM_ELEMENTS_PER_THREAD, (i+1) * NUM_ELEMENTS_PER_THREAD);
            threads.push_back(move(t));
        }
        start_flag.store(true);

        for(auto &t : threads)
            t.join();
        vector<int> list_elems;
        getElemsOfList(l, list_elems);
        sort(list_elems.begin(), list_elems.end());
        rangeCompare(list_elems, 0, NUM_ELEMENTS_PER_THREAD * NUM_THREADS);
    }

    // try a bunch of concurrent pop_front
    {
        list l;
        const int NUM_ELEMENTS_PER_THREAD = 2000;
        const int NUM_THREADS = 8;
        for(int i = 0; i < NUM_ELEMENTS_PER_THREAD; ++i)
            l.push_back(i);
        vector<thread> threads;
        vector<vector<int> > res;
        res.resize(NUM_THREADS);
        atomic<bool> start_flag(false);
        atomic<bool> stop(true);
        for(int i = 0; i < NUM_THREADS; ++i) {
            thread t(list_pop_front<list, int>, ref(l), ref(stop), ref(start_flag),
                ref(res[i]));
            threads.push_back(move(t));
        }
        start_flag.store(true);

        for(auto &t : threads)
            t.join();
        assert( l.empty() );
        vector<int> list_elems;
        for(auto &r : res)
            list_elems.insert(list_elems.end(), r.begin(), r.end());
        sort(list_elems.begin(), list_elems.end());
        rangeCompare(list_elems, 0, NUM_ELEMENTS_PER_THREAD);
    }

    // try a bunch of concurrent remove
    {
        list l;
        const int NUM_ELEMENTS_PER_THREAD = 2000;
        const int NUM_THREADS = 8;
        for(int i = 0; i < NUM_THREADS * NUM_ELEMENTS_PER_THREAD; ++i)
            l.push_back(i);
        assert( l.size() == NUM_ELEMENTS_PER_THREAD * NUM_THREADS );

        vector<thread> threads;
        atomic<bool> start_flag(false);

        for(int i = 0; i < NUM_THREADS; ++i) {
            thread t(list_remove<list, int>, ref(l), ref(start_flag),
                i * NUM_ELEMENTS_PER_THREAD, (i + 1) * NUM_ELEMENTS_PER_THREAD);
            threads.push_back(move(t));
        }
        start_flag.store(true);

        for(auto &t : threads)
            t.join();
        assert( l.empty() );
    }

    // try remove with push_back
    {
        list l;
        const int NUM_ELEMENTS_PER_THREAD = 2000;
        const int NUM_THREADS = 8;

        for(int i = 0; i < NUM_THREADS * NUM_ELEMENTS_PER_THREAD; ++i)
            l.push_back(i);
        assert( l.size() == NUM_ELEMENTS_PER_THREAD * NUM_THREADS );

        vector<thread> threads;
        atomic<bool> start_flag(false);
        // remove first
        for(int i = 0; i < NUM_THREADS; ++i) {
            thread t(list_remove<list, int>, ref(l), ref(start_flag),
                i * NUM_ELEMENTS_PER_THREAD, (i+1) * NUM_ELEMENTS_PER_THREAD);
            threads.push_back(move(t));
        }

        // then push_back
        for(int i = 0; i < NUM_THREADS; ++i) {
            thread t(list_insert<list, int>, ref(l), ref(start_flag),
                (i+NUM_THREADS) * NUM_ELEMENTS_PER_THREAD, (i+1+NUM_THREADS) * NUM_ELEMENTS_PER_THREAD);
            threads.push_back(move(t));
        }
        start_flag.store(true);

        for(auto &t : threads)
            t.join();
        vector<int> list_elems;
        getElemsOfList(l, list_elems);
        sort(list_elems.begin(), list_elems.end());
        rangeCompare(list_elems, NUM_ELEMENTS_PER_THREAD * NUM_THREADS,NUM_ELEMENTS_PER_THREAD * NUM_THREADS * 2);
    }

    // try a producer/consumer queue
    {
        list l;

        atomic<bool> start_flag(false);
        atomic<bool> stop(false);

        thread producer(list_insert<list, int>, ref(l), ref(start_flag),0,10000);

        vector<int> res;
        thread consumer(list_pop_front<list, int>, ref(l), ref(start_flag),ref(stop), ref(res));
        start_flag.store(true);

        producer.join();
        stop.store(true);

        consumer.join();

        rangeCompare(res, 0, 10000);
        assert( l.empty() );
    }

}

template<typename Function>
void Test(Function &&f, const string &name) {
    f();
    cout << "Test -- " << name << " passed" << endl;
}

int main(int argc, char **argv) {
    // generic linklist which doesn't support multithread
    Test(single_threaded_test<linklist<int> >, "single-thread-generic-linklist");
    //Test(multiple_threaded_test<linklist<int> >, "multiple-thread-generic-linklist");

    // glock lock linklist
    Test(single_threaded_test<linklist_glock<int> >, "single-thread-linklist-glock");
    Test(multiple_threaded_test<linklist_glock<int> >, "multiple-thread-linklist-glock");
    //
    // per-node lock
    Test(single_threaded_test<linklist_nodelock<int> >, "single-thread-linklist-nodelock");
    Test(multiple_threaded_test<linklist_nodelock<int> >, "multiple-thread-linklist-nodelock");

    // lock free linklist
    Test(single_threaded_test<linklist_lockfree<int> >, "single-thread-linklist-lockfree");
    Test(multiple_threaded_test<linklist_lockfree<int> >, "multiple-thread-linklist-lockfree");
    return 0;
}
```

# Reference

* [Implementing Concurrent Data Structures on Modern Multicore Machines](https://people.eecs.berkeley.edu/~stephentu/presentations/workshop.pdf)
* [Generic Concurrent Lock-free Linked list](https://people.csail.mit.edu/bushl2/rpi/project_web/page5.html)
* [Lock Free Linked Lists and Skip Lists](http://www.cse.yorku.ca/~ruppert/papers/lfll.pdf)
* [Lockless Programming Considerations for Xbox 360 and Microsoft Windows](https://msdn.microsoft.com/en-us/library/windows/desktop/ee418650(v=vs.85).aspx)
* [An Introduction to Lock-Free Programming](http://preshing.com/20120612/an-introduction-to-lock-free-programming/)
* [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
