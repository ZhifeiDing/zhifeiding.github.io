---
title : Paper Reading - A Cloud-Scale Acceleration Architecture
categories : [fpga]
tags : [fpga, datacenter]
---

# Purpose of the Paper

在[上一篇论文](http://zhifeiding.github.io/programming/fpga/2018/03/10/Paper-Reading-A-Reconfigurable-Fabric-for-Accelerating-Large-Scale-Datacenter-Services/)中设计了一种基于FPGA加速的结构， 但是由于架构上的限制( 6x8 FPGA连接比较复杂，能够直接通信的只有一个机架内的6x8个FPGA等 ), 这篇论文新提出了一种架构 -- `Configurable Cloud`, 可以加速网络流， 存储过程，安全操作以及分布式应用。和之前架构主要不同点是FPGA和服务器的NIC接口直接相连，然后通过PCIe和CPU通信，这样，所有网络通信都可以先到FPGA里进行处理。 同时FPGA之间使用LTL( Lightweight Transport Layer )来低延时通信。

下图显示了`Configurable Clound`的基本结构:
![fpga server](/assets/images/09_fpga_server_sch.png)

使用这种架构， 
* 从CPU+FPGA角度看， FPGA可以当成CPU的本地加速器
* 从整个数据中心看， FPGA可以看作是一个大规模加速资源池，可以作为远程FPGA资源

# Method Introduced in the Paper

## Hardware Design

基本的硬件包括 Altera Stratix V D5 FPGA, 4 GB DDR3, PCIe接口以及两个40G QSFP 端口，分别和服务器NIC和TOR网络通信。逻辑框图如下所示:

![fpga server](/assets/images/10_fpga_sch.png)

上面的实际电路板如下图所示:
![fpga server](/assets/images/11_fpga_board.png)

## FPGA Design

和上一篇论文一样， FPGA设计也是分成`Shell`和`Role`部分， 其中`Shell`部分设计如下图所示:
![fpga server](/assets/images/12_fpga_impl.png)

除了常规的带DMA的PCIe控制器， DDR3控制器和LTL Engine, 40G MAC控制器外，主要的是两个40G MAC之间的bypass mux, 可以将FPGA当成`dump-in-the-wire`， 网络数据直接从FPGA和NIC之间通过， 不需要CPU干预，FPGA可以直接检查修改网络包，同时NIC和FPGA都有独立的PCIe和CPU通信， 可以独立工作。


## Application on Design

根据上面的架构， FPGA可以在数据中心被当作本地加速器， 网络加速器以及远程加速器

![fpga server](/assets/images/13_fpga_result.png)

* 本地加速器时， 实现了和前一篇论文一样的Bing Search Ranking逻辑， 相比软件实现， 可以达到2.25倍的吞吐量提升。
下图是分别在FPGA和软件实现情况下连续五天的监控结果， 可以看到FPGA上实现了更小的时延
![fpga server](/assets/images/14_fpga_result.png)
下图是连续五天的负载和时延结果， 可以看到在相同负载下， FPGA能实现更低的时延
![fpga server](/assets/images/15_fpga_result.png)

* 网络加速器时， 利用上面的`bump-in-the-wire`结构可以将网络数据的加密，解密功能在FPGA实现，不需要CPU的干预。

* 远程加速器时， 可以利用上面的`Lightweight Transport Layer`和40G QPFS接口将FPGA组成一个资源池。 关键就是LTL的实现

`LTL`在FPGA中实现的框图如下所示:

# Results of the Paper

论文最后总结， 使用论文提出的架构， 可以实现`Hardware as a Service`模型， 将FPGA当成一种资源来调度，类似于`YARN`来管理任务。

# Reference

* [A Cloud-Scale Acceleration Architecture](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/Cloud-Scale-Acceleration-Architecture.pdf)
