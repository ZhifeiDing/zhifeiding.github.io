---
title: NVLink-NVSwitch-DGX | The interconnection of AI Era
categories:
  - history
tags:
  - chip
  - architecture
  - history
  - AI
  - Interconnect
---
# 前言
随着AI发展，其对内存容量和计算性能的飞速增长，计算的主角从CPU转移到AI加速器，同时AI加速器系统的扩展性也越来越重要，而其中的关键就是片间互联以及网络拓扑的选择。NVIDIA的芯片互联设计始于2016年随Pascal一起推出的NVLink互联接口；2018年在DGX-2中使用的NVSwitch芯片则显示了NVIDIA对系统扩展能力的进一步加强；而DGX系统则是对NVLink和NVSwitch进行扩展的大规模集群；而收购Mellanox，则揭示了传统数据中心网络和高性能计算网络(HPC)之间的融合趋势。NVIDIA对片间互联以及计算网络这些技术上的演进伴随着AI计算对片间互联，系统网络的需求；同时，这些技术曾经也是CPU组建大型SMP和高性能计算网络使用过的技术。通过分析NVLink，NVSwitch和DGX的历史和现状，可以看出计算和网络发展趋势，提供AI加速器系统未来可能发展方向。

# 概述
NVLink是NVIDIA在2016年推出的Tesla P100和Pascal GP100 GPU上使用的高速互联技术，称为NVLink1；2017年的Tesla V100则使用了NVLink2；2020年的A100搭配NVLink3，提高了单个lane的速率，在保持同样带宽下减少了lane数量；2022年的H100推出了NVLink4，继续提供单个lane的速率，同时减少lane数量。NVLink整体发展情况如下所示：
![0.png](/assets/images/nvlink/0.png)
NVSwitch 是一种 GPU 桥接设备，可提供DGX系统所需的 NVLink 交叉网络。NVSwitch目前发展到第三代，如下所示：
* 2018年V100 DGX-2出现的NVSwitch 1.0
* 2020年A100 DGX的NVSwitch 2.0
* 2022年H100 DGX的NVSwitch 3.0
![1.png](/assets/images/nvlink/1.png)
DGX系统则是采用NVLink和NVSwitch进行互连和扩展的计算系统，下图展示了DGX系统的内存容量变化：
![2.png](/assets/images/nvlink/2.png)

本文组织形式如下:
* 第一节介绍NVLink，从NVLink 1到NVLink 4，以及NVLink-C2C
* 第二节介绍NVSwitch，从NVSwitch 1.0到NVSwitch 3.0
* 第三节介绍DGX系统
* 文章最后列出了主要参考文献

# NVLINK

## NVLINK1
NVLink 是 NVIDIA 用于 GPU 加速计算的新型高速互连技术。它显著提高了 GPU 到 GPU 通信以及 GPU 对系统内存的访问的性能。高性能计算集群的节点中通常使用多个GPU，比较常见的配置是每个节点通常最多有 8 个 GPU，因此，在多处理系统中，强大的互连非常重要。NVLink 旨在为 GPU 创建一个片间互连，该互连将提供比 PCI Express Gen 3 （PCIe） 高得多的带宽，并与 GPU ISA 兼容以支持共享内存多线程任务。使用带 NVLink 的 GPU，应用程序既可以在本地内存上运行，也可以在相连的另一个 GPU 的内存上执行，并且保持内存操作的正确（例如，为 Pascal 的原子操作提供完全支持）。

### NVLINK拓扑
NVLink支持多种拓扑结构，不同的拓扑可以针对不同的应用进行优化。下面主要讨论以下 NVLink 配置：
* GPU-to-GPU NVLink连接
* CPU-to-GPU NVLink连接

#### GPU-to-GPU NVLink连接
下图显示了8个 GPU 组成的混合立方体网格(Hybrid Cube Mesh)，其中包括两个使用NVLink完全连接的四边形 GPU网络，四边形之间由NVLink 进行连接，每个四边形中的 GPU 直接通过 PCIe 连接到各自的 CPU。通过使用单独的 NVLink 来连接两个四边形，减轻了到每个 CPU 的 PCIe 上行链路的带宽压力，同样避免了通过系统内存和 CPU 间链路的数据传输。
![3.png](/assets/images/nvlink/3.png)

#### GPU-to-GPU NVLink连接
虽然 NVLink 主要用于将多个 NVIDIA Tesla P100 加速器连接在一起，但它也可以用作 CPU 到 GPU 的互连。例如，Tesla P100 加速器可以通过 NVIDIA NVLink 技术连接到 IBM 的 POWER8。POWER8 支持四个 NVLink。
* 下图显示了单个 GPU连接到支持 NVLink 的 CPU 。在这种情况下，GPU 可以以高达 160 GB/秒的双向带宽访问系统内存，比 PCIe 的可用带宽高 5 倍。
![4.png](/assets/images/nvlink/4.png)
* 下图显示了CPU使用NVLink 连接两个GPU 。每个 GPU 上的其余两个链路用于点对点通信。
![5.png](/assets/images/nvlink/5.png)

### NVLINK接口
NVLink接口由三层组成：
* 物理层Physical Layer (PL)
* 数据链路层Data Link Layer (DL)
* 传输层Transaction Layer (TL)

协议使用可变长度的数据包，其数据包大小范围从 1（例如简单的读请求命令）到 18（带有地址扩展的 256B 数据的写请求）不等。下图显示了 NVLink 物理层 （PHY）、数据链路层 （DL）、传输层 （TL）和层和链路：
![6.png](/assets/images/nvlink/6.png)

#### NVHS
NVLink 使用 NVIDIA 的高速信号技术 （NVHS）
* 每个信号对以 20 Gbit/s 的速度差分发送数据
* 每个方向的八个差分对组合成一个链路
	* 单个链路的原始双向带宽为 40 GB/s。信号采用 NRZ（Non-Return to-Zero）
	* 该链路为直流耦合，差分阻抗为 85 欧姆
	* 链路可以容忍极性反转和通道反转，以支持有效的 PCB 布线
* 在芯片上，数据使用 128 位 Flit（流量控制单位）以 1.25GHz 数据速率从 PHY（physical level circuit）发送到 NVLink 控制器
* NVHS使用嵌入式时钟。在接收器上，恢复的时钟用于接收传入数据

#### NVLINK物理层
PL 与 PHY 连接，并将接收到的数据传送到数据链路层，主要负责：
* **deskew** 八个通道
* **framing** 计算每个数据包的开头
* **scrambling/descrambling** 确保足够的位转换密度以支持时钟恢复
* **极性反转**
* **通道反转**

#### NVLINK数据链路层
数据链路层主要负责数据包在链路上的可靠传输，并将数据发送到事务层 （TL）：
* 传输的数据包使用 25 位 循环冗余校验CRC（Cyclic Redundancy Check）进行保护
* 传输的数据包存储在重放缓冲区中，直到链路另一端的接收方确认 （ACK） 为止
	* 如果 DL 在传入数据包上检测到 CRC 错误，则不会发送 ACK，并准备接收重新传输的数据
	* 同时，在没有 ACK 的情况下，发送器超时并从重放缓冲区启动数据重传
	* 仅当数据包被确认时，数据包才会从重放缓冲区中移除
	* 25 位 CRC 允许在任何通道上检测多达 5 个随机位错误或多达 25 位突发错误。CRC 是根据当前包头和上一个有效负载（如果有）计算的
* DL 还负责链路启动和维护

#### NVLINK传输层
传输层主要处理：
* 同步
* 链路流量控制
* 虚拟通道
* 将多个链路聚合在一起，以在处理器之间提供非常高的通信带宽

NVLink 数据包的长度从单个 128 位 flit 到最多 18 个 128 位 flit 不等，以支持 256 字节传输。NVLink传输至少包括一个请求和一个响应（post操作没有响应数据包）以及可选的地址扩展 （AE） flit、字节使能（BE） flit和 0 到 16 个数据有效负载flit。

请求头 flit 包括
* 25 位 CRC
* 83 位传输层字段，包含请求类型、地址、流量控制信用和标签标识符
* 20 位数据链路 （DL） 层字段，包括确认标识符、数据包长度信息和应用程序编号标记。

地址扩展 AE flit 包含请求之间相对静态的信息（sticky bits）、特定于命令的信息或更改命令类型默认值的信息。静态信息在更改时传输，并存储在接收端，以便用于非 AE 数据包。

字节使能BE flit 用于写命令或原子命令，128 个使能位表示要写入的数据字节，最多 128 个字节。BE 不能用于 256 字节传输。

下图展示了有请求头 flit、地址扩展AE flit、字节使能BE flit 和 数据flit 的写传输的数据包。
![7.png](/assets/images/nvlink/7.png)

设计的 CRC 允许在最大传输数据包中检测 5 个随机位的错误，或者在单个差分对上检测多达 25 个顺序位错误。数据包存储在重放缓冲区中。每个数据包都有一个序列 ID。当数据包没有CRC错误时，会返回确认信息。如果发送端指定时间内未收到确认信息，则启动重放序列，并重新传输错误的数据包和所有后续数据包。

数据包长度是可变的，长度信息作为 DL 字段的一部分进行传递。由于包头包含数据包长度信息，而协议不包含帧符号，因此在解析数据包的其余部分之前，必须检查覆盖包头的CRC。不需要针对包头和数据的单独的CRC字段，而是根据包头和前一个数据进行计算。如下图所示，序列 ID 为 1：4 的 flit 与前一个包头相关联，即 flit 0 中的 64 字节读响应。flit 5 中读请求的 CRC 是按 flits 1：5 计算的，而 64 字节写请求 （flit 6） 的 CRC 是仅根据 flit 6 计算的，因为之前没有有效数据。flit 7：10 将被flit 11 中的 CRC 覆盖（未显示）。如果在 flit 11 中没有准备好的请求或响应，则会传达与空传输关联的 CRC。
![8.png](/assets/images/nvlink/8.png)

### GPGPU里NVLINK
在 GPU 架构里，NVLink 控制器通过另一个称为高速集线器 （HSHUB） 的模块与 GPU 内部进行通信。HSHUB 可以直接访问 GPU 内部crossbar和其他系统模块，例如以 NVLink 峰值速率将数据移入和移出 GPU的高速复制引擎 （HSCE）。下图显示了 NVLink 与 HSHUB 和 GP100 GPU 中一些模块的关系。
![9.png](/assets/images/nvlink/9.png)

## NVLINK 2
与 Pascal 上的 NVLink 相比，V100 上的 NVLink 将信号速率从 20 Gb/s提高到 25 Gb/s。每个链路在每个方向提供 25 GB/秒的带宽。并且链路数从 4 个增加到 6 个，将支持的 GPU NVLink 总带宽提升到 300 GB/s。NVLink 2 允许CPU 直接对GPU 的 HBM2 内存进行加载/存储/原子访问。新增加功能包括：
* 结合新的 CPU 主控功能，NVLink 支持缓存一致性操作，允许从图形内存读取的数据存储在 CPU 的缓存中
* NVLink 2 增加了对由 GPU 或 CPU 发起的原子访问的支持。虽然 P100 支持对等的 GPU 原子访问，但不支持通过 NVLink 发送 GPU 原子访问并在目标 CPU 上完成
* 通过支持地址转换服务 （ATS），GPU 可以直接访问 CPU 的页表
* 增加链路层低功耗模式，可在链路未大量使用时节省功耗

## NVLINK 3.0
在 A100 GPU 中实现的NVLink 3和NVSwitch 显著增强了多 GPU 的可扩展性、性能和可靠性。由于每个 GPU 和交换机的支持的链路更多，NVLink 3 提供了更高的 GPU-GPU 通信带宽，并改进了错误检测和恢复功能。NVLink 3在每个方向上使用四个差分对（通道）来创建单个链路，在每个方向上提供 25 GB/s的有效带宽，NVLink 3的数据速率为每对 50 Gbit/s，几乎是 V100 中 25.78 Gbits/s 速率的两倍。单个 A100 NVLink 在每个方向上提供 25 GB/秒的带宽，与 V100 类似，但与 V100 相比，每个链路仅使用一半的信号对数。A100 中的链路总数增加到 12 个，而 V100 中为 6 个，总带宽为 600GB/s，而 V100 为 300 GB/s。

## NVLINK 4.0
NVLink 是 NVIDIA 高带宽、高能效、低延迟、无损的 GPU 到 GPU 互连，包括弹性功能，例如链路级错误检测和数据包重传机制，以确保数据的成功传输。NVLink 4 在 H100 GPU 中实现，与 NVIDIA A100 中使用的之前的NVLink 3相比，可提供 1.5 倍的通信带宽。H100 GPU中一共18个NVLink 4链路，总带宽900 GB/s，用于多 GPU I/O 和共享内存访问。NVLink 4在每个方向上仅使用两个高速差分对来形成单个链路，在每个方向上提供 25 GB/秒的有效带宽。

## NVLink-C2C
NVIDIA NVLink-C2C 是 NVIDIA 的用于Grace Hopper 超级芯片的内存一致性、高带宽和低延迟的互连，提供高达 900GB/s的总带宽。

NVLink-C2C 内存一致性可提高开发人员的工作效率、性能和 GPU 可访问的内存容量。CPU 和 GPU 线程现在可以并发和透明地访问 CPU 和 GPU的内存，让开发人员能够集中精力在算法上，而不是显式内存管理。内存一致性允许开发人员只传输他们需要的数据，而不是将整个页面进行迁移。通过支持CPU 和 GPU 原子操作可以实现CPU和GPU 的线程间的轻量级原语。

具有地址转换服务 （ATS） 的 NVLink-C2C 可以允许NVIDIA Hopper DMA 引擎加速跨主机和设备批量分页内存的传输。 NVLink-C2C 允许应用程序能够超额订阅 GPU 的内存，并以高带宽直接利用 NVIDIA Grace CPU 的内存。每个 Grace Hopper 超级芯片具有高达 480GB 的 LPDDR5X CPU 内存，GPU 可以直接访问。结合 NVIDIA NVLink Switch，在 最多 256 个 NVLink 连接的 GPU组成的DGX GH200 上运行的所有 GPU 线程可以访问高达 144TB 的内存。

# NVSwitch

## NVSwitch 1.0
NVSwitch 是一款 NVLink 交换机芯片，采用TSMC 12FFN工艺，一共20亿晶体管，面积106 mm^2；每个交换机有 18 个 NVLink 端口。在内部是一个全连接的 18 x 18 交叉网络。任何端口都能以50 GB/s 与任何其他端口通信，总聚合带宽为 900 GB/s。提供PCIe进行管理，配置缓冲区分配，使用基于软件的清理的钩子来处理表损坏和错误情况，以及部分复位和队列排空等。每个端口每个方向带宽 25 GB/s。交叉网络是非阻塞的，允许所有端口以完整的 NVLink 带宽与所有其他端口通信。下图展示了NVSwitch的物理版图：
![10.png](/assets/images/nvlink/10.png)
在高带宽下，数据完整性至关重要。NVLink 本身使用循环冗余编码 （CRC） 进行保护，以检测错误并重传。NVSwitch 的数据路径、路由和状态结构使用纠错码ECC （error-correcting code） 进行保护。NVSwitch 还支持最终hop地址保真度检查以及缓冲区溢出和下溢检查。为了安全起见，NVSwitch 的路由表由NVIDIA fabric manager索引和控制，通过限制应用程序访问范围来提供保护。NVLink 数据包里携带物理地址。下图展示了NVSwitch的架构框图：
![11.png](/assets/images/nvlink/11.png)

下图展示了16 个 GPU 都以相同的方式连接到 NVSwitch 芯片的拓扑示意图：
![12.png](/assets/images/nvlink/12.png)
一个基板上的所有 8 个 GPU 都通过单个 NVLink 连接到所有 6 个 NVSwitche。每个 NVSwitch 上的八个端口用于与另一个基板通信。同一个基板上的 8 个 GPU 中的每一个都能以 300 GB/s 的全带宽与同一基板上的任何其他 GPU 进行通信，而只经过一个NVSwitch的延迟。每个 GPU 还可以以全带宽与另一个基板上的任何 GPU 进行通信，而只需要经过两个NVSwitch延迟。基板之间的双向带宽为 2.4 TB/s（48 个链路，每个方向 25 GB/s）。但是NVIDIA DGX-2 平台仅使用每个交换机的 16 个可用端口。

## NVSwitch 2.0
NVSwitch 2.0比之前的版本快两倍，应用在NVIDIA DGX-2 系统。六个 NVSwitch 和NVLink 3 的组合使单个 GPU 到 GPU 的通信达到 600 GB/s 的峰值，能提供双向4.8 TB/s的总带宽。
![13.png](/assets/images/nvlink/13.png)

## NVSwitch 3.0
 NVSwitch 3.0采用TSMC 4N工艺，251亿晶体管，面积294mm2，提供 64 个NVLink 4链路端口，支持3.2TB/s 全双工带宽。总交换机吞吐量从上一代的 7.2 Tbits/s 增加到 13.6 Tbits/s。下图展示了NVSwitch 3.0的物理规划版图：
![14.png](/assets/images/nvlink/14.png)
NVSwitch 3.0还提供多播和 NVIDIA SHARP in-network reduction硬件集群通信加速，包括`all_gather`、`reduce_scatter`和`broadcast atomics`。与在 A100 上使用 NVIDIA Collective Communications Library （NCCL） 相比，多播和in-network reduction可提供高达 2 倍的吞吐量增益，同时显著降低小数据集群通信的延迟。NVSwitch 集群通信加速显著降低了 SM 上用于集群通信的负担。

NVSwitch 3.0的物理 电气接口（PHY） 与 400 Gbps Ethernet和 InfiniBand 兼容。管理控制器支持连接OSFP(Octal Small Formfactor Pluggable)模块，每个笼子有四个 NVLink。使用定制固件，可以支持有源线缆。还添加了额外的前向纠错FEC(forward error correction)模式，以增强 NVLink 网络的性能和可靠性。

NVSwitch 3.0使用 50 Gbaud PAM4 信令，每个差分对的带宽为 100 Gbps，可在 64 个 NVLink 端口（每个 NVLink x2）上提供 3.2 TB/s 的全双工带宽。

NVSwitch 3.0包括许多用于 SHARP 加速的新硬件模块：
- SHARP控制器，可以并行管理多达 128 个 SHARP 组
- 复用 NVIDIA Hopper 架构中的 SHARP 算术逻辑单元(ALU)，支持逻辑、最小/最大、加法等多种运算和格式（有符号、无符号整数、FP16、FP32、FP64、BF16）， 提供高达 400 GFLOPS 的 FP32 吞吐量
- 使用嵌入式SRAM支持SHARP计算

NVSwitch 3.0增加的NVLink 相关模块包括：
▪ 安全处理器，保护数据和芯片配置免受攻击
▪ 分区功能，将端口子集隔离到单独的 NVLink 网络中
▪ 管理控制器现在还可以处理连接的 OSFP 电缆
▪ 扩展遥测以支持 InfiniBand 式监视

下图展示了NVSwitch 3.0的逻辑框图:
![15.png](/assets/images/nvlink/15.png)

# DGX
## DGX-1 P100
DGX-1 有 8 个 Tesla P100 GPU 加速器，通过 NVLink组成hybrid cube-mesh网络拓扑。DGX-1 与双路 Intel Xeon CPU 和四个 100 Gb InfiniBand 网络接口卡共同组成深度学习训练系统。

### DGX-1系统架构
DGX-1 是专为高吞吐量和高互连带宽而设计的深度学习系统，可最大限度地提高神经网络训练性能。系统的核心是：
* 由 8 个 Tesla P100 GPU 通过NVLink 组成的hybrid cube-mesh网络拓扑
* 两个 CPU，用于启动、存储管理和协调深度学习框架

DGX-1 内置于3U（three-rack-unit） 机箱中，可提供电源、冷却、网络、多系统互连和 SSD 文件系统缓存。下图显示了 DGX-1 系统组件：
![16.png](/assets/images/nvlink/16.png)
Tesla P100 的页面迁移引擎(Page Migration Engine)允许在 GPU 和主机大容量内存之间高带宽、低开销的共享数据。为了扩展到多节点高性能集群，DGX-1 通过 InfiniBand （IB） 网络提供系统到系统高带宽连接。

### DGX-1网络拓扑
DGX-1 的 NVLink 网络拓扑设计旨在优化许多因素，包括各种点对点和集合通信可实现的带宽，拓扑结构的灵活性，以及GPU 的性能。hybrid cube-mesh网络拓扑可以认为是：
* 一个立方体，立方体的角是 GPU，所有 12 条边都通过 NVLink 连接，六个面中的两个也通过对角线连接
* 也可以被认为是单个 NVLink 连接的两个交织环
![17.png](/assets/images/nvlink/17.png)

### DGX-1网络
最新使用多系统扩展的计算工作负载，尤其是深度学习，为了发挥 GPU 性能，需要提供系统内部和系统之间的 GPU 的通信。除了用于 GPU 内部高速通信的 NVLink 外，DGX-1 还使用 Mellanox ConnectX-4 EDR InfiniBand 端口提供系统间通信。 在 DGX-1 中配置的InfiniBand 标准 EDR IB 提供：
- 对于每个端口，8 个数据通道提供 25 Gb/s 或 200 Gb/s 的带宽（4 个输入通道 （100 Gb/s） 和 4 个输出通道 （100 Gb/s））
- 低延迟通信和内置集合通信原语，可加速跨多个系统的大型计算
- 支持高性能网络拓扑，可同时在多个系统之间以最小的冲突传输数据
- NVIDIA GPUDirect RDMA 使用InfiniBand，在多个系统的 GPU 之间直接传输数据

DGX-1 配置了四个 EDR IB 端口，可提供 800 Gb/s带宽（400 Gb/s输入和 400 Gb/s 输出），可用于构建 DGX-1 系统的高速集群。四个 EDR IB 端口可平衡节点内和节点间带宽，在某些情况下，这些带宽可以被节点间通信完全占用。在大型多系统集群中与以太网等经典网络技术相比，InfiniBand 能提供 20 倍的带宽和 低4 倍的延迟。DGX-1 多系统集群使用基于fat tree的网络拓扑，提供系统间的可预测、无争用的路由通信：
* fat tree是一种树状结构的网络拓扑，其叶子上的系统通过多个交换机连接到中央顶级交换机 
* fat tree中的每一级有相同数量的链路，提供相等的带宽
* fat tree拓扑结构确保了计算和深度学习应用中常见的all-to-all或all-gather集合通信的最高bisection bandwidth和最低延迟
![18.png](/assets/images/nvlink/18.png)

### DGX-1性能
下图显示了使用 Microsoft Cognitive Toolkit、TensorFlow 和 Torch 的 ResNet-50 和 ResNet-152 深度神经网络在不同硬件系统上的速度比较：
![19.png](/assets/images/nvlink/19.png)
这张图说明：
1. 在深度学习训练方面，DGX-1 比上一代 NVIDIA Tesla M40 GPU 吞吐量高得多
2. DGX-1 的性能明显高于使用 PCIe 互连的 8 个 Tesla P100 GPU 的系统

## DGX-1 V100
NVIDIA DGX-1 是专为深度学习设计的高吞吐量和高互连带宽的系统，可最大限度地提高神经网络训练性能。该系统的核心是由 8 个 Tesla V100 GPU 通过NVLink组成的hybrid cube-mesh网络拓扑。除了 8 个 GPU 之外，DGX-1 还包括两个 CPU，用于启动、存储管理和深度学习框架协调。DGX-1 内置于3U机箱中，可提供电源、冷却、网络、多系统互连和 SSD 文件系统缓存。

### DGX-1 系统架构
DGX-1 配置了以太网和 InfiniBand （IB） 网络接口。两个 10 Gb 以太网接口提供系统的远程访问能力。为了连接多个 DGX-1 系统，每个系统都有四个高带宽、低延迟的 EDR IB（Extended Data Rate InfiniBand）端口，提供 800 Gb/s 的双向通信。DGX-1 系统里每两个 GPU 共享一个 IB 端口。此外，DGX-1 EDR IB 与 NVIDIA GPUDirect RDMA兼容，能够将数据直接从一个系统中的 GPU 内存传输到另一个系统中的 GPU 内存，而无需涉及 CPU 或系统内存。
![20.png](/assets/images/nvlink/20.png)

### DGX-1网络拓扑
8 个GPU 组成的hybrid cube-mesh网络可以看作是使用NVLink 连接的三个交织的双向环。以这种方式处理拓扑可确保除all-to-all之外的集合通信的性能在很大程度上是等效的。
![21.png](/assets/images/nvlink/21.png)
![22.png](/assets/images/nvlink/22.png)
深度学习训练时 GPU 间传输使用这三个不同的双向环。每个 Volta GPU有6个NVLink，每个环连接所有八个 GPU。通过这种方法，reduction和broadcast可以以超过 130 GB/s 的速度执行，而在前几代硬件上使用 PCIe 的速度为 10 GB/s。这种性能对于实现深度学习训练的高扩展性至关重要。

### DGX-1网络
除了用于 GPU 内部高速通信的 NVLink 外，DGX-1 还使用 Mellanox ConnectX-4 EDR 100Gb InfiniBand 端口提供系统间极低延迟和高带宽的通信，以减少瓶颈。在 DGX-1 中配置的最新 InfiniBand 标准 EDR IB 提供：
* 四个 EDR IB 端口，为每个 DGX-1 系统提供 400 Gb/s 输入和 400 Gb/s 输出带宽
* 低延迟通信和内置集合通信原语，可加速跨多个系统的大型计算
* 支持高性能网络拓扑，可同时在多个系统之间以最小的冲突传输数据
- NVIDIA GPUDirect RDMA 使用InfiniBand，在多个系统的 GPU 之间直接传输数据

DGX-1 配置了四个 EDR IB 端口，提供 800 Gb/s 的双向总带宽，可用于构建 DGX-1 系统的高速集群。四个 EDR IB 端口可平衡节点内和节点间带宽，在某些情况下，节点间通信可以完全利用这些带宽。与以太网等经典网络技术相比，即使在大型多系统集群中，InfiniBand 也能提供 20 倍的带宽和 4 倍的更低延迟。
![23.png](/assets/images/nvlink/23.png)
从应用程序的角度来看，GPUDirect RDMA，独特的 NVLink 和 IB 网络设计使任何 GPU 核能够以最小的开销、延迟和争用直接访问网络中任何其他 GPU 的内存。

下图展示了 124 个DGX-1组成的系统，每个一级交换机支持 4 个 DGX-1 系统，最多可配置 32 个一级交换机和 16 个二级交换机，最多支持 128 个系统：
![24.png](/assets/images/nvlink/24.png)
使用的fat-tree拓扑结构可扩展到具有更多交换机的更大配置，同时仍保持 InfiniBand 的高性能。为了扩大规模，将添加第三级交换机，或者将控制器级 IB 交换机用于 2 级交换机。

### DGX-1性能
与 Tesla P100 GPU 相比，Tesla V100 GPU 提供了更高的性能。
* V100 的 FP32 CUDA 核增加了 40%
* 它还增加了新的 Tensor Core，可以为混合精度矩阵乘法和累加提供高达 8 倍的吞吐量，这是深度神经网络中的核心计算
* Tesla V100 还具有更高的峰值内存带宽和更快的第二代 NVLink 互连

下图展示了在不同框架训练 ResNet-50 卷积神经网络性能上，与使用上一代 NVIDIA Tesla P100 GPU 的 DGX-1 相比，使用 V100 GPU 的 DGX-1 实现了更高的吞吐量。
![25.png](/assets/images/nvlink/25.png)
下表展示了DGX-1 P100和DGX-1 V00两者对比：
![26.png](/assets/images/nvlink/26.png)

### NCCL
NVIDIA Collective Communication Library（NCCL，发音为“Nickel”）是一个拓扑感知的多 GPU 集合通信库，可以轻松集成到应用程序中。NCCL最初是作为一个开源研究项目开发的，它被设计为轻量级的，并且仅依赖于常见的C++和CUDA库。NCCL 可以部署在单进程或多进程应用程序中，透明地处理所需的进程间通信。任何具有使用 MPI 集合如broadcast, reduce, gather, scatter, all-gather, all-reduce, 或all-to-all经验的人都会熟悉NCCL API 。
有许多方法可以有效地实现集合通信。但是，高效的实现必须考虑处理器之间互连的拓扑结构。为了说明这一点，请考虑从 GPU0 向下图中 PCIe 树拓扑中的所有其他 GPU 广播数据。
![27.png](/assets/images/nvlink/27.png)
为了优化Broadcast带宽，更好的方法是将 PCIe 树形拓扑视为一个环，如下图所示。然后，将数据分成小块通过环从 GPU0 中继到 GPU3 来执行Broadcast。有趣的是，只要选择了正确的环形顺序，环形算法也能为几乎所有标准集合操作提供近乎最佳的带宽，即使应用于树状PCIe拓扑也是如此。为了提供最大带宽，NCCL 实现了环式集合通信算法，并在后台隐式地将 GPU 索引到最佳环形顺序中。这为应用程序提供了出色的性能，同时使开发人员不必担心特定的硬件配置。
![28.png](/assets/images/nvlink/28.png)

## DGX-2 V100
DGX-2 单个系统包含 16 个 NVIDIA Tesla V100 32 GB GPU ，两个 24 核 Xeon CPU，1.5 TB DDR4 DRAM 内存和 30 TB NVMe 存储，可提供 2 petaFLOPS 的性能，所有GPU都使用NVLink通过 NVSwitch连接。NVSwitch 有18 个NVLink端口，每个基板上都有六个 NVSwitch 芯片，可以与另一个基板通信。如下图所示：
![29.png](/assets/images/nvlink/29.png)
与两个 DGX -1（通过 InfiniBand 连接）相比，NVSwitch连接16 个 GPU组成的 DGX-2 可以提供更高的带宽和更低的延迟。
![30.png](/assets/images/nvlink/30.png)

## DGX A100
DGX A100 具有 5 petaFLOPS 的 AI 性能，在所有 AI 工作负载上都表现出色。具体配置如下所示：
![31.png](/assets/images/nvlink/31.png)

### DGX系统架构
下图显示了 NVIDIA DGX A100 系统中主要组件的分解图：
![32.png](/assets/images/nvlink/32.png)

### DGX网络拓扑
下图展示了DGX A100 系统网络拓扑：
![33.png](/assets/images/nvlink/33.png)
DGX A100 系统包含6个NVSwitch 2.0，每个 A100 GPU 使用 12 个 NVLink 与 6 个 NVSwitch 进行互联通信，因此每个 GPU 到每个交换机都有两条链路。

### DGX网络
除了用于 GPU 内部高速通信的 NVLink 外，DGX A100 还有 8 个单端口 Mellanox ConnectX-6 200Gb/s HDR InfiniBand 端口（也可配置为 200Gb/s 以太网端口），提供 3.2 Tb/s 的峰值带宽，可用于构建 基于DGX A100 系统（如 NVIDIA DGX SuperPOD）的高速集群。可以使用 Mellanox ConnectX-6 ，通过RDMA来进行GPU间数据传输。DGX A100 在 IO 卡和 GPU 之间建立了一对一的连接，每个 GPU 都可以直接与外部通信，而不会阻止其他 GPU 的网络访问。Mellanox ConnectX-6 I/O 卡可配置为 HDR InfiniBand 或 200Gb/s 以太网，因此 NVIDIA DGX A100 可以使用低延迟、高带宽 InfiniBand 或RDMA over Converged Ethernet（RoCE）与其他节点集群一起运行 HPC 和 AI 任务。DGX A100 包括一个额外的双端口 ConnectX-6 卡，可用于高速连接到外部存储器。

DGX A100 多系统集群使用基于fat tree拓扑的网络，利用先进的 Mellanox 自适应路由和Sharp集合，提供从系统间良好路由、可预测、无争用的通信。

### DGX性能
DGX A100 在深度学习训练和推理方面相比DGX-1 V100或CPU提供更高性能，如下图所示：
![34.png](/assets/images/nvlink/34.png)

## NVIDIA DGX SuperPOD
NVIDIA DGX SuperPOD是由DGX A100 组成的集群系统，包括：
- 140 个 DGX A100 系统
- 1,120 个 NVIDIA A100 GPU
- 170 个 Mellanox Quantum 200G InfiniBand 交换机
- 15Km光缆
- 4PB 高性能存储

具体硬件参数如下：
![35.png](/assets/images/nvlink/35.png)

### DGX SuperPod架构
DGX SuperPOD 的基本单元是 SU，单个 SU由 20 个 DGX A100 系统组成，算力 48 PFLOPS。DGX A100 系统具有 8 个用于计算流量的 HDR （200 Gbps） InfiniBand 主机通道适配器HCA（host channel adapters）。每对 GPU 都有一对关联的 HCA。为了实现最高效的网络，有八个网络平面，每个平面一个叶交换机和 一个HCA，一共使用 8 个叶交换机进行连接。这些平面通过主干交换机在网络的第二层互连。每个SU都具有完整的bisection bandwidth，以确保最大的应用灵活性。每个 SU 都有一个专用的管理机架，叶交换机集中放置在管理机架中。

DGX SuperPOD 的其他设备（例如二级主干交换机或管理服务器）可能位于 SU 管理机架的空白空间或单独的机架中，具体取决于数据中心布局。

### DGX SuperPod网络架构
DGX SuperPOD 有四个网络：
* **计算网络** 每个 DGX A100 系统通过单独的网络平面连接8 个 NVIDIA HDR 200 Gb/s ConnectX-6 HCA
* **存储网络** 通过 CPU 连接的两个双端口 ConnectX-6 HCA，每个提供一个端口
* **带内管理** 使用 DGX A100 系统上的两个 100 Gbps 端口连接到专用以太网交换机
* **带外管理** 每个 DGX A100 系统上的基板管理控制器（BMC）端口连接到其他以太网交换机

下图展示了DGX A100系统的各类网络接口：
![36.png](/assets/images/nvlink/36.png)

#### DGX SuperPod计算网络
计算网络设计最大限度地提高 AI 任务的典型通信流量的性能，并在发生硬件故障时提供冗余，最大限度地降低成本。

DGX SuperPod计算网络里使用到的InfiniBand 交换机分为以下几类：
* **叶交换机(leaf switch)** 每个 SU 有 8 个叶交换机。SU 中的 DGX A100 系统连接到每个叶交换机，每个系统中所有相同的 HCA 都连接到同一个叶交换机
* **脊柱组SG（Spine group）** 10 个 QM8790 交换机组成 SG，用于优化网络。由于每个 DGX A100 系统有 8 个 InfiniBand 模块，因此需要 8 个 SG
* **核心组CG （Core group）** 14 个 QM8790 交换机组成 CG，用于连接 SG。140 个节点需要两个 CG

每个 SU 的第一个叶交换机连接到 SG1 中的每个交换机，每个 SU 的第二个枝叶交换机连接到 SG2 中的每个交换机，依此类推。需要第三层交换才能完成fat-tree拓扑：
* 每个 SG 里奇数交换机连接到CG1
* 每个 SG 里偶数交换机连接到CG2

下图展示了140 节点 DGX SuperPOD 的计算拓扑结构：
![37.png](/assets/images/nvlink/37.png)
下图展示了80 节点 DGX SuperPOD 的计算拓扑结构：
![38.png](/assets/images/nvlink/38.png)

#### DGX SuperPod存储网络
存储网络采用 InfiniBand 网络以提供高带宽支持。因为DGX SuperPOD 的每个节点的 I/O 必须超过 40 GBps，而InfiniBand为存储网络提供了高带宽和高级网络管理功能（如拥塞控制和自适应路由）。
![39.png](/assets/images/nvlink/39.png)

#### 带内管理网络InBand Management Network
带内以太网网络具有以下几个重要功能：
* 连接管理集群的所有服务
* 允许访问主文件系统和存储池
* 为集群内服务（如 Slurm）以及与集群外部的其他服务（如 NGC 注册表、代码仓库和数据源）提供连接
![40.png](/assets/images/nvlink/40.png)

#### 带外管理网络Out-of-Band Management Network
带外以太网网络用于通过 BMC 进行系统管理，并提供连接以管理所有网络设备。带外管理对于群集的运行至关重要，它提供了低使用率路径，可确保管理流量不会与其他群集服务冲突。

带外以太网交换机的上行链路可以连接到带内叶交换机，也可以连接到客户的带外网络。所有以太网交换机都通过串行连接连接到数据中心的现有控制台服务器。发生网络故障时这些连接提供了与交换机连接的最后手段。

## DGX H100
NVIDIA DGX H100 是基于最新 NVIDIA H100 Tensor Core GPU 的 DGX 系统，包含：

- 8 个 NVIDIA H100 Tensor Core GPU，具有 640GB 总 GPU 内存
- 4 个NVIDIA NVSwitch 3.0
- 18 个 NVLink 网络 OSFP
- 由 72 个 NVLink 提供 3.6 TB/s 的全双工 NVLink 网络带宽
- 8 个 NVIDIA ConnectX-7 以太网/InfiniBand 端口
- 2 个双端口 BlueField-3 DPU
- 2 个Sapphire Rapids CPU
- 支持 PCIe Gen 5

在 DGX H100 中，每个 H100 Tensor Core GPU 连接到所有NVSwitch 3.0，流量通过四个不同的交换机平面发送，从而实现链路的聚合，以实现系统中 GPU 之间的all-to-all带宽。
![41.png](/assets/images/nvlink/41.png)

### DGX H100 SuperPod
DGX H100 SuperPOD由DGX H100作为基本单元组成：
- 有8个计算机架，每个机架有四台 DGX H100
- 共有 32 个 DGX H100 节点，包含 256 个 NVIDIA H100 Tensor Core GPU
- 提供高达 1 exaflop 的峰值性能

NVLink 网络提供了256 个 GPU间 57.6 TB/s 的bisection带宽 。此外，32 个 DGX里的 ConnectX-7 和关联的 InfiniBand 交换机提供了 25.6 TB/s 的全双工带宽，可在 Pod 内使用或横向扩展多个 SuperPOD。

结合新的 NVLINK 网络拓扑和NVSwitch 3.0，NVIDIA 能够构建高带宽的大规模 NVLink 交换网络。节点通过 NVLink 交换机模块中的第二级 NVSwitch 连接在一起，这些模块位于计算节点外部，并将多个节点连接在一起。NVLink 交换机系统最多支持 256 个 GPU。连接的节点可以提供 57.6 TB 的all-to-all 带宽，并可以提供1 exaFLOP FP8 稀疏性能。

下图显示了基于 A100 和 H100 的 32 个节点、256 个 GPU DGX SuperPOD 的比较。基于 H100 的 SuperPOD 内DGX 节点使用新的 NVLink 交换机来互连。
![42.png](/assets/images/nvlink/42.png)
NVLink 网络互连采用 2：1 锥形fat tree拓扑结构，可将bisection带宽惊人地增加 9 倍（例如，用于all-to-all交换），并且与上一代 InfiniBand 系统相比，all-reduce吞吐量增加了 4.5 倍。DGX H100 SuperPOD的 NVLINK 交换机系统作为可选项。

## Grace Hopper Superchip
NVIDIA Grace Hopper Superchip架构通过内存一致性的NVLink-C2C将 NVIDIA Hopper GPU与 NVIDIA Grace CPU 的结合在一起，组合成单个超级芯片，并支持新的 NVIDIA NVLink 交换机系统。下图展示了NVIDIA Grace Hopper 超级芯片逻辑框图：
![43.png](/assets/images/nvlink/43.png)
NVLink-C2C 是NVIDIA 内存一致性、高带宽和低延迟的超级芯片互连。它是 Grace Hopper 超级芯片的核心，可提供高达 900 GB/s 的总带宽。NVLink-C2C 内存一致性可提高开发人员的工作效率和性能，并使 GPU 能够访问大量内存。CPU 和 GPU 线程现在可以同时透明地访问 CPU 和 GPU 驻留的内存，开发人员能够专注于算法，而不是显式内存管理。

内存一致性可以仅传输所需的数据，而不是将整个页面迁移到 GPU 或从 GPU 迁移整个页面；还可以通过 CPU 和 GPU 的原子操作，提供跨 GPU 和 CPU 线程轻量级同步原语。NVLink-C2C支持ATS（Address Translation Services）NVIDIA Hopper DMA引擎可以利用ATS来加速主机和设备之间可分页内存的批量传输。

NVLink-C2C 使应用程序能够超额订阅 GPU 的内存，并以高带宽直接访问 NVIDIA Grace CPU 的内存。每个 Grace Hopper 超级芯片的CPU有512 GB的 LPDDR5X 内存 。结合 NVIDIA NVLink 交换机系统，在多达 256 个通过NVLink 连接的 GPU 上运行的所有 GPU 线程现在都可以在高带宽下访问高达 150 TB 的内存。NVLink 4 支持使用直接加载、存储和原子操作访问对端内存，使加速应用程序能够更轻松地解决更大的问题。

NVIDIA Grace Hopper 超级芯片的主要特性如下：
- NVIDIA Grace CPU：
    - 72 个 Arm Neoverse V2 内核，每个内核支持Armv9.0-A ISA 和 4×128 位 SIMD 单元
    - 117 MB 的 L3 缓存
    - 512 GB 的LPDDR5X内存，提供高达 546 GB/s 的内存带宽
    - 64 个 PCIe Gen5 通道
    - NVIDIA SCF（Scalable Coherency Fabric） 片上互联和分布式缓存，带宽 3.2 TB/s
    - 单个 CPU NUMA 节点，提高开发人员的工作效率
- NVIDIA Hopper GPU：
    - 144 个 SM，配备第四代 Tensor Core、Transformer Engine、DPX
    - 96 GB 的 HBM3 内存，提供 3000 GB/s 的速度
    - 60 MB 二级缓存
    - NVLink 4 和 PCIe 5
- NVIDIA NVLink-C2C：
    - 提供Grace CPU 和 Hopper GPU 之间的硬件一致性互连
    - 900 GB/s 的总带宽，450 GB/s/dir
    - 扩展 GPU 内存功能，Hopper GPU 能够将所有 CPU 内存作为 GPU 内存进行寻址。每个 Hopper GPU 可以在超级芯片可访问 608 GB 的内存
- NVIDIA NVLink 交换机系统：
    - 使用 NVLink 4 连接 256 个 NVIDIA Grace Hopper 超级芯片
    - 每个NVLink 连接的 Hopper GPU 都可以寻址网络中所有超级芯片的HBM3 和 LPDDR5X 内存，一共150 TB 的 GPU 可寻址内存

NVIDIA GH200 Grace Hopper 超级芯片配备96GB HBM3 内存，提供 4TB/s 的内存带宽。而新一代 NVIDIA GH200 Grace 有 144GB 的 HBM3e，提供4.9TB/s带宽。NVIDIA Grace Hopper 中的 HBM 与 CPU 通过 NVLink-C2C相结合，  可以为 GPU 提供高达 624GB 的快速存取内存。

NVIDIA Grace CPU有72 Arm Neoverse V2 CPU，可提供领先的每线程性能，同时提供更高的能效比； 480GB 的LPDDR5X内存支持500GB/s带宽；片内一致性互联可提供3.2TB/s 的带宽，可支持CPU核、内存、IO 和 NVLink-C2C带宽需求。

NVIDIA Hopper 是第九代 NVIDIA 数据中心 GPU，线程块集群(Thread Block Clusters)和线程块重配置(Thread Block Reconfiguration)可改善数据在空间和时间上的局部性，并结合新的异步执行引擎，使应用程序高效使用计算单元。

NVIDIA 开发了两个平台来满足不同的客户需求：
* **NVIDIA MGX GH200** 单个节点624GB 的内存，可以运行各种工作负载；当与 NVIDIA 网络解决方案（Connect-X7、Spectrum-X 和 BlueField-3 DPU）结合使用时，使用传统的 HPC/AI 集群网络架构，易于管理和部署
* **NVIDIA DGX GH200** 通过NVLink 连接的256个 GPU的所有线程能够以900GB/s 的总带宽、450GB/s 的all-reduce带宽和 115.2TB/s 的bisection带宽访问144TB 的内存

NVIDIA MGX 采用 InfiniBand 网络，每个节点包含一个 Grace Hopper 超级芯片和一个或多个 PCIe 设备，如 NVMe 固态硬盘和 BlueField-3 DPU、NVIDIA ConnectX-7 NIC 或 OEM 定义的 I/O。NDR400 InfiniBand NIC 利用16 个 PCIe Gen 5 通道，可提供100GB/s 的总带宽。
![44.png](/assets/images/nvlink/44.png)
NVIDIA DGX GH200则采用NVSwitch交换网络， 每个 Hopper GPU 能够以 900GB/s 的总带宽与 NVLink网络中的任何其他 GPU 进行通信。NVLink TLB 使单个 GPU 能够寻址所有 NVLink 连接的内存，即NVLink 连接的 256 个节点系统的144TB内存。一个 Pod 中最多可使用 NVLink 连接 256 个超级芯片，InfiniBand NIC 和交换机将多个超级芯片 Pod 连接在一起。NVLink-C2C 和 NVLink 交换机系统在 NVLink 网络内的所有超级芯片之间提供硬件一致性。
![45.png](/assets/images/nvlink/45.png)

### 编程模型
NVIDIA 为 CUDA 平台提供下列语言支持：
• ISO C++
• ISO Fortran
• Python

以及基于指示的编程模型，例如：
• OpenACC
• OpenMP
• CUDA C++
• CUDA Fortran
![46.png](/assets/images/nvlink/46.png)

## DGX GH200
NVIDIA Grace Hopper 超级芯片和 NVLink 交换机系统是 NVIDIA DGX GH200 架构的构建块。NVIDIA Grace Hopper 超级芯片使用 NVIDIA NVLink-C2C 结合了 Grace 和 Hopper 架构，以提供 CPU + GPU 一致性内存模型。NVLink 交换机系统采用 NVLink 4技术，将 NVLink 连接扩展到超级芯片之间，以创建无缝、高带宽、多 GPU 系统。

NVIDIA DGX GH200 中的每个 NVIDIA Grace Hopper 超级芯片有 480 GB LPDDR5 CPU 内存，每 GB 功耗只有DDR5的八分之一；96 GB 的HBM3。NVIDIA Grace CPU 和 Hopper GPU通过NVLink-C2C 互连，提供比 PCIe Gen5 高 7 倍的带宽，功耗仅为 PCIe Gen5 的五分之一。

NVLink 交换机系统形成一个两级、无阻塞、fat-tree拓扑的NVLink网络，对系统中256 个 Grace Hopper 超级芯片提供全连接。DGX GH200 中的每个 GPU 都可以以 900 GBps 的速度访问其他 GPU 的内存和所有 NVIDIA Grace CPU 的内存。托管 Grace Hopper 超级芯片的计算基板使用用于 NVLink 结构第一层的定制电缆线束连接到 NVLink 交换机系统。LinkX 电缆扩展了 NVLink 结构第二层的连接。下图展示了使用NVSwitch对256 个 GPU 进行互联组成的 NVIDIA DGX GH200的NVLink网络拓扑
![47.png](/assets/images/nvlink/47.png)
在 DGX GH200 系统中，GPU 线程可以通过NVLink page table来访问等NVLink 网络中其他 Grace Hopper 超级芯片的HBM3和LPDDR5X。NVIDIA Magnum IO加速库则针对256 个 GPU的系统优化了 GPU 通信。DGX GH200 中的每个 Grace Hopper 超级芯片适配一个 NVIDIA ConnectX-7网络适配器和一个 NVIDIA BlueField-3 NIC；DGX GH200提供128 TBps 的bisection带宽和 230.4 TFLOPS 的 NVIDIA SHARP 网内计算性能，可加速 AI 中常用的集合操作；通过减少集合操作的通信开销，使 NVLink 网络系统的有效带宽翻倍。

可以通过ConnectX-7 互连多个 DGX GH200 系统扩展到 256 个以上GPU。而BlueField-3 DPU 则可支持虚拟私有云，使组织能够在安全的多租户环境中运行应用程序。

# 参考文献
1. NVIDIA DGX-1: The Fastest Deep Learning System [WWW Document], 2017. . NVIDIA Technical Blog. URL [https://developer.nvidia.com/blog/dgx-1-fastest-deep-learning-system/](https://developer.nvidia.com/blog/dgx-1-fastest-deep-learning-system/)
2. Announcing NVIDIA DGX GH200: The First 100 Terabyte GPU Memory System [WWW Document], 2023. . NVIDIA Technical Blog. URL [https://developer.nvidia.com/blog/announcing-nvidia-dgx-gh200-first-100-terabyte-gpu-memory-system/](https://developer.nvidia.com/blog/announcing-nvidia-dgx-gh200-first-100-terabyte-gpu-memory-system/)
3. NVIDIA Grace Hopper Superchip Architecture In-Depth | NVIDIA Technical Blog [WWW Document], n.d. URL [https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/)
4. NVIDIA Ampere Architecture In-Depth [WWW Document], 2020. . NVIDIA Technical Blog. URL [https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
5. Defining AI Innovation with NVIDIA DGX A100 [WWW Document], 2020. . NVIDIA Technical Blog. URL [https://developer.nvidia.com/blog/defining-ai-innovation-with-dgx-a100/](https://developer.nvidia.com/blog/defining-ai-innovation-with-dgx-a100/)
6. NVSwitch Accelerates NVIDIA DGX-2 | NVIDIA Technical Blog [WWW Document], n.d. URL [https://developer.nvidia.com/blog/nvswitch-accelerates-nvidia-dgx2/](https://developer.nvidia.com/blog/nvswitch-accelerates-nvidia-dgx2/)
7. dgx1-v100-system-architecture-whitepaper
8. NVIDIA GH200 Grace Hopper Superchip Architecture
9. A. Ishii and R. Wells, "The Nvlink-Network Switch: Nvidia’s Switch Chip for High Communication-Bandwidth Superpods," 2022 IEEE Hot Chips 34 Symposium (HCS), Cupertino, CA, USA, 2022, pp. 1-23, doi: 10.1109/HCS55958.2022.9895480.
10. Y. Wei et al., "9.3 NVLink-C2C: A Coherent Off Package Chip-to-Chip Interconnect with 40Gbps/pin Single-ended Signaling," 2023 IEEE International Solid-State Circuits Conference (ISSCC), San Francisco, CA, USA, 2023, pp. 160-162, doi: 10.1109/ISSCC42615.2023.10067395.
11. D. Foley and J. Danskin, "Ultra-Performance Pascal GPU and NVLink Interconnect," in IEEE Micro, vol. 37, no. 2, pp. 7-17, Mar.-Apr. 2017, doi: 10.1109/MM.2017.37.
