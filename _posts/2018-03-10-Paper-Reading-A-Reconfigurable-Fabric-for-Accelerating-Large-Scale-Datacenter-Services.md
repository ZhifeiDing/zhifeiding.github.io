---
title : Paper Reading - A Reconfigurable Fabric for Accelerating Large-Scale Datacenter Services
categories : [fpga]
tags : [fpga, datacenter]
---

# Puspose of the Paper

由于现在数据中心负载越来越大，
同时CPU随着摩尔定律和散热问题导致性能无法继续提升。为了解决这个问题，
这篇论文提出研究了使用FPGA来对服务进行加速的方案。
论文中介绍的方案是使用6x8个高端Stratix FPGA组成的二维网络放置于48个机器的机架中。每一个FPGA和对应的服务器通过PCIe来通信，并可以通过10Gb SAS线直接和其他FPGA通信。使用Bing Web Search来对系统进行测试，发现在高负载情况下，该系统在固定的延时分布下可以提高服务器95%的吞吐，或者在保持相等吞吐情况下减少29%的延时。

整个系统架构如下图所示:

![fpga overview](/assets/images/00_fpga_overview.png)


# Method used in the Paper

FPGA Implementation Architecture
![fpga arch](/assets/images/03_fpga_arch.png)

Web Search Implementation Flow
![fpga fe](/assets/images/05_fpga_Web_Search.png)

Feature Extraction Implementation
![fpga fe](/assets/images/02_fpga_fe.png)

Free Form Expression Implementation
![fpga fe](/assets/images/04_fpga_fe.png)

Bing Web Search Flow System View
![fpga flow](/assets/images/01_fpga_flow.png)

# Results of the Paper

# Reference

* [A Reconfigurable Fabric for Accelerating Large-Scale Datacenter Services](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Catapult_ISCA_2014.pdf)
