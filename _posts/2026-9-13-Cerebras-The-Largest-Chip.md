
> "海到无边天作岸，山登绝顶我为峰。"

# 前言

别人把晶圆切成几百颗小芯片再想办法连起来，Cerebras反着来：一整片晶圆就是一颗芯片。山不必登，它自己就是峰。

Cerebras使用整个晶圆制造一个巨大的芯片WSE - Wafer Scale Engine，可以从下图中直观的感受到WSE和其他芯片在晶体管数量上的差异:

![0.png](/assets/images/cerebras/0.png)

Cerebras的WSE通过将wafer 作为一个芯片，获得了巨大的片上访存带宽，但同时也导致系统供电，芯片良率等复杂问题，本文针对这些问题，从Cerebras第一代WSE1的系统结构，WSE2的计算架构和系统设计，WSE3系统设计到第四代系统CS-4来讲解分析。具体章节安排如下：
* 第一章简单介绍采用Wafer Scale的一些背景
* 第二章介绍WSE1的系统设计和编译流程
* 第三章通过WSE2来介绍芯片内部计算架构和CS2
* 第四章介绍WSE3芯片及系统CS3
* 第五章介绍CS4
* 第六章结合公开的CSL示例代码，讲Cerebras的编程模型

# 概述

一般而言，每一代芯片的性能提升会被归类于架构上的优化，但实际上，工艺进步，时钟频率，芯片面积等都会对芯片性能产生影响，过去五十年来对芯片性能提升的相对贡献如图所示：
![1.png](/assets/images/cerebras/1.png)
可以看到工艺进步对性能影响其实是远远超过其他因素。芯片面积增加，为所需功能的晶体管数量提供了更多的硅面积。在当前技术下，存在一个最大尺寸限制。下图显示了一些比较有影响力的芯片面积增加的趋势曲线。
![2.png](/assets/images/cerebras/2.png)
芯片尺寸增长放缓的原因包括:
* 随着工艺进步，增加光罩尺寸（即晶圆的最大面积）变得更加困难，因为减少激光波长的同时维护激光聚焦在更大的区域越来越困难。这是芯片尺寸增长放缓的主要原因。
* 随着芯片尺寸的增加，每个芯片的良率会降低。虽然有几种不同的缺陷分布模型——例如，墨菲模型、泊松模型、二项式模型、摩尔模型和 Seeds 模型 - 所有模型的共同点是，随着芯片尺寸的增加，良率呈指数级下降。虽然将芯片尺寸扩展到整个晶圆可以最大限度地提高计算和带宽，但是良率模型基本上是零良率，因为无缺陷晶圆是极其罕见的事件。晶圆级设计是一个非常具有挑战的任务。

业界在此之前有过多次尝试：
* 1980 年，Trilogy Systems 尝试使用晶圆级集成(WSI)构建与 IBM 兼容的大型机，生产 2.5 英寸 （in） 的单芯片。这家公司由大型机时代的传奇工程师Gene Amdahl创立，拿到当时创纪录的2.3亿美元投资，但Trilogy Systems遇到了几个障碍，最终导致了这个早期晶圆级集成的努力失败，五年后公司解散。
* 2015 年，Cerebras Systems 开始设计晶圆级芯片，WSE于2019年首次推出，成功克服了光罩尺寸限制。

# WSE-1

在2019年，Cerebras推出了其第一代晶圆级芯片Wafer Scale Engine，采用台积电16nm工艺，包含1.2T晶体管，面积46225平方毫米。与此同时，当时最大的GPU则是NVIDIA的V100，包含21.1B晶体管，面积815平方毫米，具体对比如下所示：
![3.png](/assets/images/cerebras/3.png)
WSE-1一共有400,000 个AI 核心，芯片有 18GB 片上内存、9PB/s 内存带宽和超过 100Pb/s 的互连带宽。

这40万个核组织成84个tile（7×12排布），每个tile约4,800个核。核的官方名称是SLA core（Sparse Linear Algebra core）：为深度学习的稀疏线性代数设计，但仍是可编程的通用处理器。按Cerebras在白皮书中给出的口径，深度学习训练数据里50%到98%是零值，SLA核的数据流机制让零值不触发计算，直接在发送端被过滤掉。

## Cerebras CS-1 Wafer-Scale AI System
CS-1是由WSE-1组成的AI系统，前视图如下图所示，左上角是 I/O 模块，下方是 12 个电源。右边是泵模块，下面是风扇模块。
![4.png](/assets/images/cerebras/4.png)
在俯视图中，可以看到 12 个 100GbE 链路朝向 15U 机箱的前部。在左侧有两个泵模块，帮助芯片保持冷却。
![5.png](/assets/images/cerebras/5.png)
以下是泵模块的机械结构：
![6.png](/assets/images/cerebras/6.png)

每个 Cerebras CS-1 的设计功率高达 20kW，其中约15kW给芯片本身，4kW给冷却子系统，1kW是电源转换损耗。在 15U 机箱中，每 U 仅为 1.3kW/，大约和 2U 4 节点服务器或 GPU 计算服务器的功率相当，整机只占标准机架的三分之一。该系统利用 12 个 100GbE 连接提供 1.2Tbps 的外部带宽。

Cerebras 的软件堆栈支持从 Tensorflow 和 PyTorch 等高级框架到将模型编译到晶圆级芯片上并运行。
![7.png](/assets/images/cerebras/7.png)

# WSE-2

第二代 WSE-2 于 2021 年发布，采用台积电7nm工艺，面积 46,225 mm^2（215mm × 215mm），包含 2.6 万亿晶体管。一个晶圆级片上网络可连接 850,000 个内核，片上存储40GB，内存带宽20PB/s，片上结构（fabric）带宽220Pb/s，比当时最大的GPU大56倍。峰值算力FP16 dense为7.5 PFLOPS，FP16 sparse为75 PFLOPS（按10倍稀疏加速折算）。85万核 × 4个FP16 FMAC × 每周期2次运算 × 1.1GHz，正好是7.5 PFLOPS。集群架构利用 CS-2扩展能力 ，以提供简单的数据并行扩展到更大的模型和更大的集群。
![8.png](/assets/images/cerebras/8.png)

WSE-2的构建方式很有意思：先设计一个传统意义上可切割的die，每个die含 10,156 个核（66 × 154 排布），尺寸 17mm × 30mm，即 510 平方毫米；之后不进行切割，而是在直径300 毫米圆形晶圆内切割出最大的正方形，最终以 12 × 7 的排布容纳 84 个芯粒，合计 850,000 个核。
![9.png](/assets/images/cerebras/9.png)
不同于之前的 Trilogy 使用的冗余门设计，CereBras的冗余单元是PE；WSE-2中约1%的PE作为冗余备份，以便“修复”有缺陷的 PE。
![10.png](/assets/images/cerebras/10.png)

Cerebras 核是一个小核心设计, 核心物理尺寸为228μm × 170μm（约38,760μm^2），如图所示。在TSMC 7nm工艺下，一半的面积是 48 kB 的SRAM，另一半是逻辑。整个内核以 1.1GHz 时钟频率运行，峰值功耗为 30mW。
![11.png](/assets/images/cerebras/11.png)
每个小核都有48 kB的本地SRAM，分成8个单端口bank，每个bank都是32位宽，每个周期可以支持 256 位访问；小核每个周期支持两个 64 位读和一个64 位写， 一共 192 位访问。所有的内存都是核内单独寻址，没有传统意义上的共享内存。所有核之间的共享是通过互联网络完成。除了本地SRAM，还有一个256B 软件管理的缓存，用于经常访问的数据结构。
![12.png](/assets/images/cerebras/12.png)
张量操作在 64 位数据路径上执行，由四个 16 位融合乘法累加组成（FMAC），支持FP16 和BF16，可选支持FP32。为了优化性能和灵活性，指令集架构将张量作为一等的操作数，类似通用寄存器或内存。下图显示了一个示例在 3D 和 2D 张量上操作的 FMAC 指令操作数。
![13.png](/assets/images/cerebras/13.png)
小核使用数据结构寄存器 （DSR） 作为指令的操作数，有 44 个DSR，每个 DSR 都包含一个描述符，其中包含指向张量和其他关键信息的指针，如张量的长度、形状和大小。利用DSR，硬件架构可以原生支持最多4D 张量。

小核同时支持八个张量操作，称为微线程，硬件上可以逐个周期切换。调度器持续监控正在处理的张量的输入和输出的可用性，选择一个符合条件且资源可用的微线程。每个微线程也分配了优先级，因此，如果多个微线程有资格运行，调度器将选择优先级更高的微线程。每个微线程都可以完全访问所有资源，包括核中的寄存器和内存，没有任何静态分区。

互联使用 2-D 网络拓扑，如下图所示。其中每个核有一个Fabric Router。Fabric Router采用的五端口设计，每个都是32 位双向接口；使用低缓冲的无损流量控制，实现核间的单时钟周期延迟。基本数据包是单个 16 位数据元素（FP16 或 BF16）。除了这些 16 位数据外，还有 16 位数据的控制信息，构成 32 位细粒度包。为了进一步优化结构设计，它使用完全静态的布线，效率高且低开销。每个内核有 24 个本地静态可以配置的路由，称为Color。所有Color之间是非阻塞的，通过时间多路复用到相同的物理链路上。在整个网络范围内，这些本地独立Color被组合在一起支持整个核心阵列大量路由。此外，还使用每种Color对于各种不同的数据和控制通信，因此固定的 24 种Color使更多不同消息或任务总数。每个交换矩阵路由器中设计有本机广播和组播功能
![14.png](/assets/images/cerebras/14.png)

不同于传统的封装之后安装到PCB，向晶圆提供小于 1 V且接近 20,000 A 的峰值电流，同时保持良好的稳压是艰巨的挑战。Cerebras采用分布在晶圆表面的 300 多个单独的电压调节模块 （VRM），每个WSE-2使用多个 VRM 确保电源分配冗余，并可单独控制每个电源域。
![15.png](/assets/images/cerebras/15.png)



![16.png](/assets/images/cerebras/16.png)
Cerebras 开发了 MemoryX 技术来解决 CS-2 中超出 WSE-2 的内存容量问题。Cerebras CS-2 集群支持 1.5TB 和 12TB MemoryX 单元：
* 支持4TB-2.4PB容量
* 支持2000 亿 - 120 万亿个权重，带优化器状态
* 支持DRAM 和闪存混合存储
MemryX用于权重更新/优化器的内部计算，处理智能流水线以掩盖延迟，可扩展到极端模型大小，独立于计算的容量扩展
![17.png](/assets/images/cerebras/17.png)

模型权重存储在MemoryX内部的DRAM 和SSD中，权重流式传输到CS-2 系统，一次一层，并以1.2 Tb/s的带宽流式传输到 CS-2 中；CS-2 依赖数据流机制进行计算，权重永远不会存储在 CS-2 系统上。
![18.png](/assets/images/cerebras/18.png)
使用数据流调度，每个单独的权重将触发对片上存储的激活的AXPY 操作。训练时一旦每个权重计算完成，就会被丢弃，而不会存储在芯片上，并移动到下一个权重。为了降低从MemoryX流式传输每个权重的成本，每个权重在 CS-2 上复用多次。在向后传递时，梯度以相反的方向流出并返回 MemoryX 单元，并更新权重。
![19.png](/assets/images/cerebras/19.png)

GEMM在晶圆上的映射方式以Transformer层为例，激活张量的三个维度（Batch、Sequence、Hidden）中，Hidden沿fabric的x方向切分，Batch和Sequence沿y方向切分。权重按行广播到各列核上，每个非零权重触发本地一列激活的FMAC；部分和的归约走PSUM/FSUM两条命令：所有列收到PSUM命令后在行方向上做环形归约，其中一列收到FSUM命令负责存出最终结果。归约通过微线程与下一行权重的FMAC计算重叠，不占额外时间。这套映射下单芯片可以不切分地跑完100,000 × 100,000的MatMul——对当时任何模型都够用，这也是"不需要模型并行"的底气所在。
如此之多流式传输和计算资源，延迟可能是一个挑战，尤其是在更新期间。Cerebras 解决延迟问题
* 通过消除对延迟敏感的通信实现高效的性能扩展
* 粗粒度流水线
	* 前向/增量/梯度是完全流水线的
	* 迭代中的所有流都没有层间依赖关系
* 细粒度流水线
	* 权重更新和前向传递的重叠涵盖了迭代间依赖关系
![20.png](/assets/images/cerebras/20.png)


SwarmX Interconnect 是可横向扩展到多个系统的高速结构。 SwarmX 上的光学器件几乎不可插拔并且有 12 个端口。
![21.png](/assets/images/cerebras/21.png)
通过专门设计的称为 SwarmX的互连来实现数据并行横向扩展，如下图所示。SwarmX 位于存储权重的MemoryX 单元和用于计算的 CS-2 系统之间；SwarmX 将权重广播到所有 CS-2 系统，并对所有 CS-2 的梯度进行reduction。在内部，SwarmX 使用树形拓扑来支持模块化和低成本扩展。按2021年发布时的口径，SwarmX最多可把192台CS-2连成一台逻辑设备，合计1.63亿个核，训练同一个网络。
![22.png](/assets/images/cerebras/22.png)

CS-2整机的部署规格是15U机箱，445mm宽、1005mm深，300kg，最大功耗23kW，供电为6+6冗余的4kW电源模块，12个100GbE数据口合计1.2Tb/s I/O。冷却为内部闭环、直达芯片的液冷，泵模块1+1冗余热插拔；机房没有水路也可以选液冷转风冷的部署方式。这些数字与CS-1一脉相承，差别主要在芯片从16nm换成7nm后，同样功耗下性能翻了一倍还多。

WSE-2的数据流协议天然带稀疏加速。发送方的核会过滤掉零值数据，接收方因此直接跳过这部分计算，而不是照做一遍乘加。相比GPU照算不误，ReLU、Max Pool这类天然产生稀疏的算子在WSE-2上可以白捡一部分性能和功耗。CS-2集群的规模上限是192个系统。

# WSE-3

WSE-3 于2024年3月发布，采用台积电5nm工艺，比最大的GPU 大 57 倍，包含 4 万亿个晶体管、900,000 个 AI 优化计算核； 44 GB片上SRAM，可以提供21PB/s内存带宽，片上结构带宽214Pb/s（折合26.7PB/s），峰值AI算力125 PFLOPS。晶圆仍由84个die组成，每个die约10,700个核。WSE-3在与WSE-2相同的功耗和价格下做到了两倍性能。

对于计算核，16位数据（FP/BF16）的SIMD从4-wide翻倍到8-wide，8位定点（INT8）到16-wide，另加了一组快速非线性函数指令；调度与稀疏机制不变，仍是细粒度数据流调度加原生非结构化稀疏加速。存储侧，每核48kB SRAM不变，但本地缓存从256B扩到512B，DSR从44个增加到48个。
![23.png](/assets/images/cerebras/23.png)

跨die互连也给出了具体数字。每个die边界上是480条24Gb/s的并行链路，合计2,880GB/s；整片晶圆跨越划片线的链路共40,320条，带宽242TB/s，而所有这些互连的总功耗只有97W（单个die边界1.1W，折合0.05pJ/bit）。对照DGX-H100那种靠SerDes、线缆和交换芯片堆出来的互联（5.0pJ/bit），同带宽下功耗差出近百倍；WSE-3的die间I/O带宽是DGX-H100的33倍，功耗却只有六分之一。


## Cerebras CS-3 Wafer Scale System
Cerebras CS-3 是第三代晶圆扩展系统。顶部有 MTP/MPO 光纤连接，以及用于冷却的电源、风扇和冗余泵。
![24.png](/assets/images/cerebras/24.png)
Cerebras 需要为巨型芯片提供电力、数据和冷却，同时还需要管理相对较大的区域的热膨胀等。在内部，芯片是液冷的，热量可以通过风扇或水带走。
![25.png](/assets/images/cerebras/25.png)
![26.png](/assets/images/cerebras/26.png)
与具有八个 NVIDIA H100 GPU 和内部 NVSwitch 和 PCIe 交换机的 NVIDIA DGX H100 系统相比，CS-3是一个更大的构建块。
![27.png](/assets/images/cerebras/27.png)

![28.png](/assets/images/cerebras/28.png)

这一代的一大特点是更大的集群，多达 2048 个 CS-3，用于多达 256 exaFLOP 的 AI 计算。CS-3的系统I/O仍是12个100GbE端口（1.2Tb/s），与CS-2一致；对外部署的最大案例是与G42合作的Condor Galaxy网络：CG-1（圣克拉拉）与CG-2（斯托克顿）各4 exaFLOPS，CG-3到CG-5（达拉斯）合计20 exaFLOPS，规划中的CG-6到CG-9（明尼阿波利斯）再贡献32 exaFLOPS。
![29.png](/assets/images/cerebras/29.png)
这个 12PB 的内存数字是一个高端超大规模 SKU，专为快速训练 GPT-5 大小的模型而设计。Cerebras 还可以缩小到类似单个 CS-2 ，并支持服务器和网络。借助 CS-3，大幅增加了 MemoryX 选项，包括面向企业客户的 24TB 和 36TB SKU，以及面向超大规模客户的 120TB 和 1,200 TB 选项。
![30.png](/assets/images/cerebras/30.png)
部分内存不仅是片上内存（44GB），还包括支持服务器中的内存。


## 推理：权重从"流过"变成"住下"

2024年8月，Cerebras发布推理服务。推理的负载特征与训练完全不同：生成1,000个token就要把模型完整读1,000遍，瓶颈在内存带宽而不在计算。

Cerebras的应对是把执行模型整个反过来。训练时权重从MemoryX逐层流过晶圆、用完即弃；推理时权重一次性装进片上SRAM，住下不走。Llama3.1-8B是8B参数、FP16占16GB，44GB的片上SRAM直接装下整模型；Llama3.1-70B是140GB，摊到4片晶圆共176GB上，晶圆之间只传激活，走CS-3的低延迟RDMA over Ethernet。

计算组织也随之改变，模型的各层被映射到晶圆上的不同区域，区域大小按该层的存储和计算需求划分，权重与KV cache都存在本区域的SRAM里。每个区域一个token，局部batch为1；相邻区域物理上紧挨着，流水线级间几乎没有延迟；最后一个区域的输出绕回第一个区域，生成下一个token。GPU是多张卡合力跑一层，Cerebras是晶圆的一角跑一层，方向正好相反。单用户只占用一小部分内存带宽，富余的带宽用来并行多个用户；prompt已知全部内容，还可以把多个prompt token同时塞进流水线的不同级。官方的性能数据是Llama3.1-8B单用户1,800 tokens/s，70B单用户450 tokens/s

# CS-4

2026年8月18日，Cerebras发布第四代系统CS-4，首批出货在同年第三季度。按CS-1到CS-3的命名惯例，CS-4本该对应一块新的晶圆，也就是WSE-4。但并没有。CS-4装的是三颗WSE-3 Turbo，配套的是一套重新设计的机架，叫Nexus Platform Architecture。Cerebras第一次在没有换代硅片的情况下推出了换代系统。

![31.png](/assets/images/cerebras/31.png)

## WSE-3 Turbo

WSE-3 Turbo和已经在出货的WSE-3是同一块硅片：4万亿晶体管、46,225mm²、台积电5nm、900,000个AI核、44GB片上SRAM，全部没变。变的是时钟，从约1.4GHz提到约2.8GHz。所谓Turbo就是超频。主频翻倍，单晶圆的指标也跟着翻倍：AI算力125 PFLOPS提到250 PFLOPS，内存带宽21.6PB/s提到43.2PB/s，片上结构带宽26.7PB/s提到53.5PB/s，片外I/O带宽1.2Tbit/s提到2.4Tbit/s。乘以机架内的三颗晶圆，就是CS-4的系统级数字。

频率能提上去，靠的不是架构，是供电。Nexus把AC/DC转换放到距晶圆约0.5mm的位置，常规GPU板上这个距离大约50mm，近了约100倍；同时用铜母线（busbar）取代PCB走线供电。板级功率损耗被压到很小，送到WSE-3T的电力翻倍，频率才上得去。
![32.png](/assets/images/cerebras/32.png)

## 规格

官方datasheet给出的CS-3与CS-4对比如下。

| 规格 | CS-3（单晶圆） | CS-4（三晶圆） |
| --- | --- | --- |
| 架构 | 单晶圆机箱 | 有源供电机架，三个模块化计算背包 |
| 晶圆 | 1 × WSE-3 | 3 × WSE-3 Turbo |
| AI算力（sparse FP16） | 125 PFLOPS | 750 PFLOPS |
| 内存带宽 | 21.6 PB/s | 129.6 PB/s |
| 片上结构带宽 | 26.7 PB/s | 160.5 PB/s |
| 系统I/O带宽 | 1.2 Tbit/s | 7.2 Tbit/s |
| I/O延迟 | 5 微秒 | 2 微秒 |
| 片上SRAM | 44 GB | 132 GB |
| 支持模型规模 | 最高24万亿参数（配MemoryX） | 超过50万亿参数（厂商口径） |

单颗WSE-3 Turbo的规格，4万亿晶体管、46,225mm²、TSMC 5nm、900,000核、44GB SRAM、250 PFLOPS（sparse FP16）、43.2PB/s内存带宽、53.5PB/s片上结构带宽、2.4Tbit/s I/O带宽。

## Nexus机架平台

Nexus不是芯片，是可复用的机架框架，电源从前面进，计算从后面插。Cerebras说同一个机架会依次承载CS-4、CS-5、CS-6，换晶圆不必换机架。也就是说，CS-4是平台的第一代，不是硅片的第一代。Nexus围绕计算、供电、I/O三件事重新组织，各自做成自包含模块。

Wafer-Scale Backpack是计算模块。它后置安装，垂直挂接到电源阵列上，把电源转换、直接液冷、高速I/O和控制电子折叠进一个围绕晶圆的紧凑三维封装里。计算与供电解耦之后，制造简化，部署时间从数天缩到数小时；相比上一代，组件减少50%，自动化制造比例提高60%。

模块化I/O方面，CS-4引入了新的可编程I/O子系统，带宽翻倍、延迟减半。新的Wafer I/O Module从晶圆边缘引出模块化的可编程fabric，支持两种连接模式：标准RoCE v2 RDMA over Ethernet，用于接入现有基础设施；Direct Wafer Links，无交换机直连，支持机架内和跨机架，晶圆间延迟低至2微秒。

![33.png](/assets/images/cerebras/33.png)

![34.png](/assets/images/cerebras/34.png)

## 存储结构

CS-4的存储思路没有变，全部是片上SRAM，没有任何一级要出片取权重。单晶圆44GB、43.2PB/s，机架132GB、129.6PB/s。这个带宽GPU方案靠HBM达不到，也是Cerebras强调"每用户token速度"的原因，解码时的瓶颈通常不在浮点，而在权重从内存搬到计算单元的速度。

但容量是另一回事。44GB每晶圆、132GB每机架，对前沿模型来说仍然很小，大模型必须摊到很多晶圆上。晶圆面积也已经没有余量，晶圆面积已被逻辑和SRAM 100%用满，多加SRAM就要少放核。按路线图，CS-6（大约再两代）会把DRAM以3D方式堆叠到计算与SRAM晶圆之上，用高带宽的垂直互连连接，容量增长而权重不必走远路。CS-5定在2027年，会用新的WSE硅片。

## 编程模型与部署范式

CS-4这一代软件侧最大的变化不在编译器，而在部署范式，原生支持分离式推理（disaggregated inference）。推理被拆成两段，prefill处理输入prompt并准备好模型状态，decode逐token生成。CS-4几乎只为decode而生，2微秒的晶圆间延迟就是为这个场景准备的。至于prefill，Cerebras直接把客户指向AMD Helios和AWS Trainium。

![35.png](/assets/images/cerebras/35.png)

这在Cerebras的叙事里是一次后退，也是一次收缩。早年的说法是一块晶圆可以替代整个GPU机架，包办训练和推理；现在收缩到"只做低延迟token生成"这一段。在这段上巨大晶圆有物理优势，关键路径上没有芯片间跳变；而在计算密集的prefill上，不再硬碰NVIDIA的装机量和定价。这个定位比"通吃"更站得住。

执行模型沿用WSE-3那代确立的推理形态，权重一次性装进片上SRAM住下来，模型各层映射到晶圆区域，逐token流水线生成。这与训练时的权重流式（weight streaming）是两套模式——训练权重从MemoryX流过、用完即弃，推理权重驻留、激活流动。CS-4机架132GB的SRAM容量就是按"装下decode阶段的权重加KV cache"来定的一档规格，gpt-oss-120b这类模型FP8量化后正好放得下。

## 系统扩展

机架内有三颗晶圆，各带一个backpack；跨机架扩展靠Direct Wafer Links，无交换机，机架内与跨机架同一语义。


# 编程模型


整套软件栈分三层。CSL（Cerebras Software Language）是设备侧语言，语法基于Zig但并非100%兼容，编译器也不共用Zig的代码。选Zig是为了它的编译期计算能力，让PE程序既能写成可维护的模板，又能编译出接近手写的代码。SdkRuntime是主机侧Python运行时，负责加载程序、启动内核、搬运张量。再往上是cstorch，把PyTorch模型编译到晶圆上。

CSL没有线程调度器，没有缓存一致性，没有动态路由。每个PE跑一份自己的程序，占用哪块区域、走哪条通道、数据从哪个方向来，全部在编译期定死。程序员拿到的是整块晶圆的静态地图，不是一个可以"申请资源"的运行时。

## 一个PE，一个完整程序

最小的完整程序由两个文件构成。`layout.csl`是编译期求值的顶层文件，它决定程序占几个PE、每个PE绑哪段代码：

```csl
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });

layout {
  @set_rectangle(1, 1);
  @set_tile_code(0, 0, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(0) });
  @export_name("y", [*]f32, false);            // 主机只读
  @export_name("init_and_compute", fn()void);  // 主机可调用
}
```

PE上的程序就是普通函数加显式内存：`var A: [M*N]f32`这48kB是这颗核的全部本地内存，`gemv()`照常写双重循环，最后导出符号：

```csl
fn init_and_compute() void {
  initialize();
  gemv();
  sys_mod.unblock_cmd_stream();   // 每个主机可调用函数必须以它结尾
}
comptime {
  @export_symbol(y_ptr, "y");
  @export_symbol(init_and_compute);
}
```

`unblock_cmd_stream()`这个约定背后是一套简单的规则：memcpy基础设施一次只推进一条命令，设备函数结束必须显式解锁命令流，主机侧的`memcpy_d2h`才能继续。主机与设备的串行化就这样由程序员写在代码末尾，而不是藏在驱动里。

`--fabric-dims`这个编译参数还给出了晶圆的实际规模：WSE-2编译到硬件要写`757,996`，WSE-3写`762,1172`，后者约89万个可编址PE。

## Color、路由与Wavelet：数据到达即执行

CSL里最基础的概念是wavelet。它是晶圆上的基本传输单位，一个wavelet携带32位数据；数据加上控制信息构成32位细粒度包，在二维网格上以单周期延迟逐跳转发。程序的基本执行单位是数据触发任务（Wavelet-Triggered Task）：wavelet到达路由器，路由器把它转给计算单元，绑定的任务随即被激活，wavelet的payload直接作为任务参数传入：

```csl
task main_task(wavelet_data: i16) void {   // 数据到达即被激活
  global = wavelet_data;
  @mov16(out_dsd, global);                 // DSD描述"往哪儿搬"
}
comptime {
  @bind_data_task(main_task, main_task_id);
}
```

color是编译期命名的逻辑通道，每个PE有24个，互不阻塞、时分复用到同一物理链路；路由由`@set_color_config`静态指定，`rx`说明从哪个方向收，`tx`说明往哪些方向转发。也就是说，通信拓扑是程序的一部分，不是运行时协商出来的。

任务分两类：data task由数据到达触发，local task由`@activate(task_id)`显式唤醒，跨PE归约的收尾就走后者。还有一个细节：WSE-2上数据任务ID由color派生，WSE-3上改由输入队列派生，同一份示例代码用`@is_arch("wse2")`/`@is_arch("wse3")`分支同时兼容两代硬件。微线程与队列解耦也是WSE-3那一代核内改动的一部分。

## 跨PE的流式GEMV

把这些组合起来，就是一个完整的数据流内核。`gemv-checkerboard-pattern`做`y = Ax + b`，4×4的PE阵列，M=32、N=16：A按行切四份、按列切四份，每个PE拿8×4的一块常驻内存；x从北侧流入，沿列向南广播；b从西侧进入；部分积沿行向东做链式归约，最终从最右列流出。

![36.png](/assets/images/cerebras/36.png)

PE上的代码只有两个任务。收到一个x元素，就对A的一列做一次乘加；收满整列元素，唤醒归约：

```csl
task fmac_task(wlet_data: f16, idx: u16) void {
  @fmach(dsd_mul_temp, dsd_mul_temp, dsd_A, wlet_data, .{ .index = idx });
  num_recv_x += 1;
  if (num_recv_x >= X_local_dim_x) { @activate(REDUCE); }        // 收满即唤醒归约
}

task reduce_task() void {
  const c = if (last_column) @get_color(@bitcast(u16, sys_mod.MEMCPYD2H_1)) else send;
  const dsd_out = sys_mod.get_streaming_fabout_dsd(c, B_local_dim_x, send_oq);
  @faddh(dsd_out, dsd_in, dsd_mul_temp, .{ .async = true });     // y = A*x + y_west
}
```

这里没有任何全局屏障，也没有`__syncthreads`式的同步原语。整个矩阵向量乘的推进完全靠数据到达的先后关系：x到了就算，算完就往下传。DSD（Data Structure Descriptor）承担搬运描述的角色：`mem1d_dsd`描述本地内存的访问模式，`fabin_dsd`/`fabout_dsd`描述fabric收发；`.wavelet_index_offset = true`让wavelet自带的索引直接选中A的哪一列，FMA不必自己算地址。更流式的版本（`gemv-09-streaming`）用`@increment_dsd_offset`在每收到一个x之后把描述符推进一列，效果相同。

路由配置用的是棋盘式双色：

```csl
const RX_R_TX_RS = .{ .rx = .{RAMP},  .tx = .{RAMP, SOUTH} };  // 收下、转南
const RX_N_TX_RS = .{ .rx = .{NORTH}, .tx = .{RAMP, SOUTH} };  // 收北、转南
const RX_R_TX_E  = .{ .rx = .{RAMP},  .tx = .{EAST} };         // 收下、转东
```

偶数列用颜色1发、颜色2收，奇数列反过来。因为fabric的双向链路在归约时会被东西两侧同时占用，不做棋盘分配就会自撞。这类冲突在GPU上由硬件路由解决，在CSL里要自己安排。

## copy与streaming：权重流式在API里的样子

主机侧的接口只有三个函数：`memcpy_h2d()`、`memcpy_d2h()`、`launch()`。`memcpy_h2d`有一个布尔参数`streaming`，把Cerebras的两套数据供给语义分开了：

```python
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, 4, 4, M_per_PE*N_per_PE,
                  streaming=False, ...)                      # copy模式：整块驻留
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, 4, 1, N_per_PE,
                  streaming=True, ...)                       # streaming模式：逐wavelet注入
```

`streaming=False`是copy模式，张量整块搬进PE内存，A这样的常量走这条路；`streaming=True`是streaming模式，张量被拆成一串wavelet持续注入，由数据任务边收边算，激活值走这条路。前面WSE-2那一章讲的权重流，落到代码上就是这个布尔参数。权重不驻留，走streaming；每一层的激活也不需要整块落地，同样走streaming。"权重用完即弃"这个说法在API层面是直接可见的。

因为一个wavelet只有32位，FP16张量在主机侧要先做16位到32位的零扩展，回传时再把高16位剥掉。这是细粒度包设计的必然结果。

## PyTorch这一层

绝大多数用户不会写CSL。PyTorch侧只需要改几行：

```python
model = cstorch.compile(Model(), "CSX")        # 类似torch.compile，只做准备
optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01)
sparsity = cstorch.sparse.RigLSparsityOptimizer(model.named_parameters(),
                                                sparsity=0.9, schedule=1000)

@cstorch.trace                                  # 圈出"跑在晶圆上"的部分
def training_step(inputs, targets):
    optimizer.zero_grad(); sparsity.apply_sparsity()
    loss = loss_fn(model(inputs), targets)
    loss.backward(); optimizer.step(); sparsity.step()
    return loss

@cstorch.step_closure                           # 圈出"留在主机"的回调
def print_loss(loss): print(f"Loss: {loss.item()}")
```

三个装饰器和接口划分了主机与设备的边界：`compile`准备计算图，`trace`圈出上晶圆的部分，`step_closure`圈出留在主机的回调。`RigLSparsityOptimizer`对应"原生硬件加速动态稀疏"：训练过程中动态剪枝与重连，在GPU上要靠掩码和稀疏库绕一大圈，在WSE上由数据流硬件直接处理。
# 参考文献

1. S. Lie, "Inside the Cerebras Wafer-Scale Cluster: Cerebras Systems," 2023 IEEE Hot Chips 35 Symposium (HCS), Palo Alto, CA, USA, 2023, pp. 1-41, doi: 10.1109/HCS59251.2023.10254700. 
2. G. Lauterbach, "The Path to Successful Wafer-Scale Integration: The Cerebras Story," in IEEE Micro, vol. 41, no. 6, pp. 52-57, 1 Nov.-Dec. 2021, doi: 10.1109/MM.2021.3112025.
3. S. Lie, "Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning," in IEEE Micro, vol. 43, no. 3, pp. 18-30, May-June 2023, doi: 10.1109/MM.2023.3256384.
4. Wang, J., 2024. Cerebras CS-3: the world’s fastest and most scalable AI accelerator. Cerebras. URL [https://www.cerebras.net/blog/cerebras-cs3/](https://www.cerebras.net/blog/cerebras-cs3/)
5. "Wafer-Scale Deep Learning," 2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, 2019, pp. 1-31, doi: 10.1109/HOTCHIPS.2019.8875628. 
6. "Software Co-design for the First Wafer-Scale Processor (and Beyond)," 2020 IEEE Hot Chips 32 Symposium (HCS), Palo Alto, CA, USA, 2020, pp. 1-26, doi: 10.1109/HCS49909.2020.9220504. 
7. A. Yeung, E. Gardner, "Introducing Cerebras CS-4: The Fastest AI Just Got Faster," Cerebras Blog, Aug. 18, 2026. URL [https://www.cerebras.ai/blog/introducing-cerebras-cs-4](https://www.cerebras.ai/blog/introducing-cerebras-cs-4)
8. "Cerebras Unveils CS-4: Up to 30 Times Faster than GPU-based Solutions," Cerebras Press Release, Aug. 2026. URL [https://www.cerebras.ai/press-release/cerebras-unveils-cs-4-up-to-30-times-faster-than-gpu-based-solutions-3347356](https://www.cerebras.ai/press-release/cerebras-unveils-cs-4-up-to-30-times-faster-than-gpu-based-solutions-3347356)
9. "CS-4 Datasheet," Cerebras Systems, 2026. URL [https://www.cerebras.ai/cs4-datasheet](https://www.cerebras.ai/cs4-datasheet)
10. "Wafer-Scale Engine: The Largest Chip Ever Built (WSE-2 Datasheet)," Cerebras Systems, 2021.
11. "An AI Chip With Unprecedented Performance To Do the Unimaginable," Cerebras Blog, 2021. URL [https://cerebras.ai/blog/an-ai-chip-with-unprecedented-performance-to-do-the-unimaginable/](https://cerebras.ai/blog/an-ai-chip-with-unprecedented-performance-to-do-the-unimaginable/)
12. J. P. Fricker, S. Lie, Cerebras CS-4 system architecture session, 2026 IEEE Hot Chips 38 Symposium (HCS), 2026.
13. Cerebras Systems, "Cerebras SDK Documentation." URL [https://sdk.cerebras.ai](https://sdk.cerebras.ai)
14. Cerebras Systems, "SDK Examples" (Apache-2.0), GitHub. URL [https://github.com/Cerebras/sdk-examples](https://github.com/Cerebras/sdk-examples)
15. Cerebras Systems, "Host Runtime and Tensor Streaming," Cerebras SDK Documentation. URL [https://sdk.cerebras.ai/tensor-streaming](https://sdk.cerebras.ai/tensor-streaming)
16. Cerebras Systems, "PyTorch on Cerebras Wafer-Scale Cluster," PyTorch Conference 2023. URL [https://static.sched.com/hosted_files/pytorch2023/bb/PyTorchOnCerebrasWaferScaleCluster-CerebrasSystems.pdf](https://static.sched.com/hosted_files/pytorch2023/bb/PyTorchOnCerebrasWaferScaleCluster-CerebrasSystems.pdf)
17. S. Lie, "Wafer-Scale AI: GPU Impossible Performance," 2024 IEEE Hot Chips 36 Symposium (HCS), 2024. URL [http://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf](http://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf)
18. "Cerebras CS-2 Data Sheet," Cerebras Systems, 2021.
19. "Wafer-Scale Engine 3: The Largest Chip Ever Built (WSE-3 Datasheet)," Cerebras Systems, 2024.
20. "Cerebras Systems Announces World's First Brain-Scale Artificial Intelligence Solution," Cerebras Press Release, Aug. 24, 2021.
21. "Cerebras Systems Unveils World's Fastest AI Chip with Whopping 4 Trillion Transistors," Cerebras Press Release, Mar. 13, 2024.
22. S. K. Moore, "4 Things to Know About the Biggest Chip Ever Built," IEEE Spectrum, Aug. 2019.
