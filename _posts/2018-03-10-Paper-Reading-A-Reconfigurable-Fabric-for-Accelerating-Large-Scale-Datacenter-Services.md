---
title : Paper Reading - A Reconfigurable Fabric for Accelerating Large-Scale Datacenter Services
categories : [fpga]
tags : [fpga, datacenter]
---

# Puspose of the Paper

由于现在数据中心负载越来越大，
同时CPU随着摩尔定律和散热问题导致性能无法继续提升。为了解决这个问题，
这篇论文提出研究了使用FPGA来对服务进行加速的方案。论文中介绍的方案是使用6x8
个高端Stratix
FPGA组成的二维网络放置于48个机器的机架中。每一个FPGA和对应的服务器通过PCIe来通信，并可以通过10Gb SAS线直接和其他FPGA通信。

![fpga overview](/assets/images/00_fpga_overview.png)
![fpga flow](/assets/images/01_fpga_flow.png)
![fpga fe](/assets/images/02_fpga_fe.png)

# Method used in the Paper

# Results of the Paper

# Reference

*[A Reconfigurable Fabric for Accelerating Large-Scale Datacenter Services](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Catapult_ISCA_2014.pdf)
