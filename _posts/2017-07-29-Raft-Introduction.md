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
* *Follower* : 除了*Leader*之外其他节点, 当*hearbeat*为零时转化为*Candidate*
* *Candidate* : 中间状态，在*Follower*选举*Leader*期间存在

三者之间的转换关系如下图所示：

![role-transfer](/assets/images/raft/role_transfer.png)

其中每个节点都有一个递减的*HeartBeat*值，和一个递增的*currentTerm*。 当一个*Follower*节点的*HeartBeat*为零时，其转化为*Candidate*,同时增加*currentTerm*，向其他节点发送*RequestVote*(当前节点选自己)，当*Candidate*收到大多数节点，即[n/2]+1节点的选举时转化为*Leader*。因此正常工作时*leader*需要在一定时间之内给*Follower*重置*HeartBeat*。

上述说明中的同一时间具体的说是同一个`term`, 更具体的可以用下图表示：
![term](/assets/images/raft/terms.png)

# Normal Operation

了解了*Raft*中的角色之后， 可以理解下面*Raft*里正常的运行流程了:

1. *Client*发送命令`command`到*Leader*
2. *Leader*将命令`command`记录到自己的`log`里
3. *Leader*发送`AppendEntries` *RPC*给*Follower*, 要求记录`command`
4. *Leader*执行`command`并返回结果给*Client*, 同时发送`AppendEntries`要求*Follower*执行`command`
5. 如果上述过程中有*Follower* 非正常工作， *Leader*持续发送`AppendEntries`直到成功

# Leader Election

和*Paxos*利用*Poposal Number*来选举一个*Propoer*不一样， 在*Raft*中， *Leader* 选举过程如下:

![leader election](/assets/images/raft/leader_election.png)

1. 初始状态，所有节点都是*Follower*
2. *Follower*的*heartbeat* 归零，转化为*Candidate*
3. 当前节点`currentTerm++`，同时发出`RequestVote` *RPC*给其他节点
4. 其他节点比较收到的`RequestVote`的`term`和自己的`currentTerm`,如果`term`比自己`currentTerm`小则不同意当前*Candidate*发出的`RequestVote`
5. 否则如果发出`RequestVote`的*Candidate*的`log`不比自己`log`旧则同意*candidate*的选举
6. 得到大多数节点选举的*Candidate*转化为*Leader*, 同时发送空的`AppendEntries`来重置其他节点，使其成为*Follower*

# Log Replication

上面提到*Leader*会发送`AppendEntries`来要求*Follower*同步操作，另外， *Leader*在选举成功之后，也会记录每个节点的`log`的:

* `nextIndex`  : 下一个记录在`log`里的`index`, *Leader*会将其初始化为其当前`log index + 1`
* `matchIndex` : 记录由当前*Leader*复制的记录的最高的`index`, 初始化为零

`term`, `log index`和*Leader*, *Follower*的关系参见下图所示:

![log replication](/assets/images/raft/log_structure.png)



# Reference

* [Raft github](https://raft.github.io/)
* [The sceret lives of data](http://thesecretlivesofdata.com/raft/)
* [Raft Paper](https://ramcloud.stanford.edu/wiki/download/attachments/11370504/raft.pdf)
* [Design for Understandability -- the Raft Concensus Algorithm](https://raft.github.io/slides/uiuc2016.pdf)
* [The Raft Protocol : A better Paxos](http://engineering.cerner.com/2014/01/the-raft-protocol-a-better-paxos/)
* [Paxos Simple](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/paxos-simple-Copy.pdf)
