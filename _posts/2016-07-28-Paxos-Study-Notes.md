---
title : Paxos Study Notes
categories : programming
tags : [consensus, reading notes]
---

# *Paxos* 简介

当多个节点同时参与完成一件事情时候，需要对过程中的事件达成一致，而*Paxos*就是在不可靠的计算机网络中用来保证一致性（可靠性）的算法。
除了 *Paxos* 之外，还有保持多个副本来保证可靠性的方法，比如主从异步复制。而 *Paxos* 则属于多数派写的算法， 即仿照选举的过程中少数服从多数的方法，只要一个系统中一半以上节点正常即可保证整个系统的一致性和可靠性。可以将*Paxos*看成一种特殊的主从复制机制， 类似红黑树和二叉搜索树的关系。

## *Paxos* 基本假设和概念

*Paxos* 算法工作先决条件是数据存储没有错误和丢失(消息可以丢失和乱序)， 在此基础上， *Paxos* 有几个概念：

* *Proposer* : 发起请求并请求*Acceptor*接受
* *Acceptor* ：作为*Paxos*存储节点，接受和存储数据，一半以上节点*Acceptor*组成*Quorums*。而且消息必须发给一个*Quorums*或者被一个*Quorums*发出
* *Quorums* : 至少一半*(n/2 + 1)* *Acceptors*, 保证了消息的可靠性
* *Proposal Number* 和 *Agreed Value* : *Agreed Value*是每次*Proposer*要写的数据，*Proposal Number*是每次*Proposer*的一个全局唯一的值， 并且单调递增。

## *Typical Deployment*

在*Paxos*中, 每个参与的节点都会充当*Proposer*和*Acceptor* 。*Proposer*给*Quorums*发送*Proposal Number* 和 *Agreed Value* , 然后*Acceptors*根据*Proposal Number*来决定是否接受*Agreed Value*。如果*Proposer*不能和至少一个*Quorum*通信则不能发起*Paxos*过程。

## *Basic Paxos*

这是最基本的*Paxos*协议，完成一次*Paxos*需要两个*Round*，每个*Round*分为两个*Phase*：

* *Phase 1a : Prepare* : *Proposer*给至少一个*Quorum*发送*Proposal Number*， *Proposer Number*要比之前该*Proposer*用的都大。
* *Phase 1b : Promise* : 如果*Proposal Number*比*Acceptor*之前收到的任何*Proposal Number*都大，则返回之前收到的*Proposal Number*和*Agreed Value*，并承诺之后不接受任何其他*Proposal*。否则，*Acceptors*应该ignore， 可以不应答也可以返回*Nack*告诉*Proposer*放弃该*Proposal*。
* *Phase 2a : Accept Request* ：如果收到至少一个*Quorum*的应答，则可以发送*Agreed Value*。如果返回的*Agreed Value*都是空则可以任意选择自己要写入的*Agreed Value*，否则选择*Proposal Number*最大的*Agreed Value*。
* *Phase 2b : Accepted* : *Acceptors*接受之前Promised的*Proposal Number*的*Agreed Value*， 拒绝其他Proposal。

上述*Paxos*过程可以用下面的图表示：

```cpp
            Proposal x                Acceptor 1,2,3
Prepare     proposalNum  ---------->  1, 2,
Promise                  <----------  1 lastProposalNum, lastAgreeVal
                                      2 lastProposalNum, lastAgreeVal
AcceptReq   agreeVal     ---------->  1, 2,
Accepted                              1, 2, agreeVal
```

## *Multi Paxos*

上面的*Basic Paxos*每次都需要2个round， 4个delay来完成一次*Paxos*。而*Multi Paxos*则省掉了上面的*Phase 1*过程， 也就是相当于只有一个*Proposer*, 上面已经提到过， *Paxos*可以看成加了约束来保证可靠性的主从复制机制， 而*Multi Paxos*更加说明这一点。

## *Fast Paxos*

上面的*Multi Paxos*一次*Paxos*只需要2个delay，但是却只能有一个*Proposer*。*Fast Paxos*则结合上面两个协议， 在没有冲突时类似*Multi Paxos*协议，*Proposer*只进行*Phase 2*， 当有冲突发生时则回退到*Basic Paxos*。有区别的是为了保证可靠性*Fast Paxos*中的*Quorum*不再是*n/2+1*, 而变成了*n * 3/4*。

# 参考

* [Paxos Introduction](http://drmingdrmer.github.io/pdf/paxos-slide/paxos.pdf)
* [Wechat - phxpaxos](https://github.com/tencent-wechat/phxpaxos)
* [C++ - libpaxos](http://www.leonmergen.com/libpaxos-cpp/)
* [wikipedia - paxos](https://en.wikipedia.org/wiki/Paxos_(computer_science))
