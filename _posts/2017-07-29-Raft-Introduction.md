---
title: Raft Introduction
tags: [concensus,programming]
categories: [programming]
---

# What's Raft ?

> 与之前介绍过的[Paxos](http://zhifeiding.github.io/programming/2016/07/28/Paxos-Study-Notes/)一样， *Raft*也是用来解决分布系统中*concensus*问题的协议。 我们知道， 在*Paxos*中，任何节点都同时是*Proposer*和*Acceptor*, 而在*Raft*中，同一时间，最多只能有一个*Leader*, 其他节点只能是*Follower*， 和*Multi-Paxos*有一点类似。

# Role in Raft

在*Raft*中， 只存在下面三种节点:

* *Leader* : 负责和*Client*通信，并管理*Follower*， 同一时间最多只能有一个
* *Follower* : 除了*Leader*之外其他节点
* *Candidate* : 中间状态，在*Follower*选举*Leader*期间存在

三者之间的转换关系如下图所示：

![role-transfer](/assets/images/raft/role_transfer.png)

其中每个节点都有一个递减的*HeartBeat*值，和一个递增的*currentTerm*。 当一个*Follower*节点的*HeartBeat*为零时，其转化为*Candidate*,同时增加*currentTerm*，向其他节点发送*RequestVote*(当前节点选自己)，当*Candidate*收到大多数节点，即[n/2]+1选举时转化为*Leader*。因此正常工作时*leader*需要在一定时间之内给*Follower*重置*HeartBeat*。

# Leader Election

和*Paxos*利用*Poposal Number*来选举一个*Propoer*不一样， 在*Raft*中， *Leader* 选举过程如下:

1. 初始状态，所有节点都是*Follower*
2. *heartbeat* 

# Log Replication

# Reference

* [Raft github](https://raft.github.io/)
* [The sceret lives of data](http://thesecretlivesofdata.com/raft/)
* [Raft Paper](https://ramcloud.stanford.edu/wiki/download/attachments/11370504/raft.pdf)
* [Design for Understandability -- the Raft Concensus Algorithm](https://raft.github.io/slides/uiuc2016.pdf)
* [The Raft Protocol : A better Paxos](http://engineering.cerner.com/2014/01/the-raft-protocol-a-better-paxos/)
* [Paxos Simple](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/paxos-simple-Copy.pdf)
