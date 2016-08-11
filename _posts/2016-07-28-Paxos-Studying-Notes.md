---
title : Paxos Study Notes
categories : programming
tags : [consensus, reading notes]
---

# *Paxos* 简介

当多个节点同时参与完成一件事情时候，需要对过程中的事件达成一致，而*Paxos*就是在不可靠的计算机网络中用来保证一致性（可靠性）的算法。
除了 *Paxos* 之外，还有保持多个副本来保证可靠性的方法，比如主从异步复制。而 *Paxos* 则属于多数派写的算法， 即仿照选举的过程中少数服从多数的方法，只要一个系统中一半以上节点正常即可保证整个系统的一致性和可靠性。

## *Paxos* 基本假设和概念

* 

## *Basic Paxos*

## *MultiPaxos*

# 参考

* [Paxos Introduction](http://drmingdrmer.github.io/pdf/paxos-slide/paxos.pdf)
* [Wechat - phxpaxos](https://github.com/tencent-wechat/phxpaxos)
* [C++ - libpaxos](http://www.leonmergen.com/libpaxos-cpp/)
* [wikipedia - paxos](https://en.wikipedia.org/wiki/Paxos_(computer_science))
