---
title : Paxos Study Notes
categories : programming
tags : [consensus, reading notes]
---

# *Paxos* 简介

当多个节点同时参与完成一件事情时候，需要对过程中的事件达成一致，而*Paxos*就是在不可靠的计算机网络中用来保证一致性（可靠性）的算法。
除了 *Paxos* 之外，还有保持多个副本来保证可靠性的方法，比如主从异步复制。而 *Paxos* 则属于多数派写的算法， 即仿照选举的过程中少数服从多数的方法，只要一个系统中一半以上节点正常即可保证整个系统的一致性和可靠性。

## *Paxos* 基本假设和概念

*Paxos* 算法工作先决条件是数据存储没有错误和丢失(消息可以丢失和乱序)， 在此基础上， *Paxos* 有几个概念：

* *Proposer* : 发起请求并请求*Acceptor*接受
* *Acceptor* ：作为*Paxos*存储节点，接受和存储数据，一半以上节点*Acceptor*组成*Quorums*。而且消息必须发给一个*Quorums*或者被一个*Quorums*发出
* *Quorums* : 至少一半*(n/2 + 1)* *Acceptors*, 保证了消息的可靠性
* *Proposal Number* 和 *Agreed Value* : *Agreed Value*是每次*Proposer*要写的数据，*Proposal Number*是每次*Proposer*的一个全局唯一的值， 并且单调递增。

## *Typical Deployment*


## *Basic Paxos*



## *MultiPaxos*

# 参考

* [Paxos Introduction](http://drmingdrmer.github.io/pdf/paxos-slide/paxos.pdf)
* [Wechat - phxpaxos](https://github.com/tencent-wechat/phxpaxos)
* [C++ - libpaxos](http://www.leonmergen.com/libpaxos-cpp/)
* [wikipedia - paxos](https://en.wikipedia.org/wiki/Paxos_(computer_science))
