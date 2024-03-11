---
title: AI Accelerator | All processors aspire to be general purpose
categories:
  - history
tags:
  - chip
  - architecture
  - history
  - AI
---
# 前言

> 一花独放不是春，百花齐放春满园

相比通用计算的百花凋零，AI加速计算可谓千帆竞发，百舸争流。既有Groq TSP超大核空间计算，Graphcore IPU众核存内计算，SambaNova RDU可重构数据流计算，也有百度昆仑和华为昇腾采用通用计算结合矩阵加速方案，还有Alibaba的CNN专用加速器含光800；当然，也少不了Intel，从用于HPC的Xeon Phi，独立GPU，到Nervana NNP和Habana Gaudi，无不在彰显财大气粗同时透漏出内心的迷茫。

各种架构层出不穷，令人眼花缭乱，目不暇接，固然是行业繁荣的标志；但是作为从业者，能够从众多加速器设计中获取核心思想，提升自身水平，无疑是最重要的，须知

> 乱花渐欲迷人眼，浅草才能没马蹄

# 概述

本文主要关注数据中心AI加速器，至于端侧的AI加速器，由于应用场景限制，在设计思路上有所不同。
![0.png](/assets/images/ai/0.png)

* 含光800是Alibaba在2020年推出的专门针对CNN推理的加速器，在ResNet-50上实现了每秒78563次Inference，成为一时榜首。不过随着AI模型的迅速发展，这种专用的加速器在数据中心的场景下，越发显示出其缺点，即对CNN优化的有多好，对其他模型性能就有多差。
* 百度昆仑系列采用常见的通用计算核加矩阵加速器，这种方案在实现上比较简单，也比较容易随着工艺而扩展来提升性能；因而迭代也比较快，目前已经从Kunlun 1发展到Kunlun 2了，Kunlun 3有传闻2024年初会量产。
* 华为的昇腾系列采用达芬奇架构，也是通用计算核加矩阵加速器，只不过矩阵加速器与通用计算核紧耦合，有单独的指令集。这种方案利用了华为自身通用计算处理器的设计能力，在软件使用上也相对方便一些。
* Groq的TSP最近由于大模型的火热，也改名成LPU，不过名字变了，东西还是那个东西。上面可以贴的标签很多，从流水线上可以看成是一个巨大的单核处理器；从计算方式上看，是SIMD阵列；从编程模型上，属于数据流空间计算；从存储结构看，又是近存计算。但是，标签再多，TSP的软件编程上必定是非常不友好的。当然，Groq好像并不直接卖硬件，而是提供AI计算服务，这样复杂性都是自己消化了，好像商业上也没什么问题。
* SambaNova的RDU系列在架构上是采用SIMD近存计算的数据流加速器，芯片内分成几个小的Tile，每个Tile和Groq的TSP其实比较相似。编译器承担比较多的调度工作。
* Graphcore的IPU系列采用近存众核数据流计算方式，芯片本身也只有片上存储，不过后面开发了一款交换芯片，可以为计算芯片提供DDR内存。
* Intel在AI加速方面则是多点开花，主打一个不知所措
	* 首先是Xeon Phi，采用x86众核方案对向量计算加速。这个最开始是和NVIDIA的GPGPU在HPC领域竞争的，在天河超算中也有使用。后来由于美国禁令，这条产品线也就没有了。结合最近几年美国在半导体产业上的禁令，不得不说，美国这些禁令才是国内半导体发展第一推动力。
	* 之后是收购的Nervana的NNP加速器，专门用于矩阵加速，不过其用于推理和训练的产品居然使用不同架构，也不怪Intel后期也放弃了Nervana的产品。
	* Habana的Gaudi系列采用向量计算和矩阵计算的加速器方案，在纵向和横向扩展能力上比较好，也是目前Intel主推加速方案。不过从Intel的路线图上看，后期也会被合并到Intel Xe GPU架构里，前景也不佳。

总而言之，国内AI加速器架构设计上还是相对保守和传统的，这与国内当前的体系结构和设计能力还是匹配的。而国外的架构设计思路上则更加天马行空，不再拘泥于传统成熟的体系结构；在创新性上远超国内，这些和国外在通用计算和体系结构上深厚积累是息息相关的。

本文组织形式如下：
* 第一节介绍Alibaba的hanguang 800，了解其专用架构
* 第二节介绍Baidu的Kunlun系列，主要是Kunlun 1和Kunlun 2
* 第三节介绍Huawei的Ascend系列和Davinci架构
* 第四节介绍Groq的TSP架构，编程模型及系统扩展
* 第五节介绍SambaNova的RDU系列，主要是SN10
* 第六节介绍Graphcore的IPU系列，包括架构，编程模型和系统扩展
* 第七节则介绍Intel的AI加速解决方案，包括Xeon Phi，Nervana的NNP和Habana的Gaudi
* 文章最后列出了相应参考文献

# Alibaba 
Alibaba在2020年推出了专门针对CNN推理的加速器含光800，其ResNet-50推理性能远超当时各类AI加速器，如下图所示：
![1.png](/assets/images/ai/1.png)
## HanGuang800
含光800采用台积电12nm工艺，面积709mm^2，TDP功耗280W，工作频率700MHz。下图展示了含光800的物理版图：
![2.png](/assets/images/ai/2.png)
含光800包含4个用环形总线连接的核，一个命令处理器CP(Command Processor)，PCIE gen4 X16；每个核由张量引擎TE(Tensor Engine)，池化引擎PE(Pooling Engine)，存储引擎ME(Memory Engine)，指令缓冲IB和SEQ，常值缓冲CB等组成；一共192MB Local Memory (LM)，每个核48MB的LM。下图展示了含光800架构框图：
![3.png](/assets/images/ai/3.png)
**张量引擎tensor engine(TE)**主要负责对来自LM的张量数据进行卷积，矩阵乘加，量化，归一化等操作，并将结果写回LM。下图展示了卷积映射到TE的示意图：
![4.png](/assets/images/ai/4.png)
TE支持通过稀疏化，减枝和量化来对模型进行压缩，减少存储和计算需求；矩阵计算支持INT8/INT16，向量计算支持FP24(1sign.8exp.15man)。
![5.png](/assets/images/ai/5.png)

**池化引擎Pooling Engine(PE)**对数据进行池化，插值等操作，并将结果写到LM；主要由下列功能单元组成：
* **POOL单元** 负责POOLs, ROIs操作，以及ROI_align_2
* **INTP单元** 负责插值，以及ROI_align_1
* **Scale/Bias单元** 负责缩放/偏置操作，以及fp19和fp16/bf16/int8/int16数据格式的转换
* **输出缓冲Output Buffer** 负责输出结果的缓冲
![6.png](/assets/images/ai/6.png)
存储引擎ME(Memory Engine)负责张量复制，reshape，矩阵转置等数据操作，以及对LM的管理；分别使用2048位输入输出接口和环形总线连接。ME的架构示意图如下所示：
![7.png](/assets/images/ai/7.png)
片上一共192MB的SRAM，每个核48MB，分成两个Block，每个Block由16块384KB的SRAM组成。下图展示了LM的分布示意图：
![8.png](/assets/images/ai/8.png)
### 编程模型
主机将模型的缩放/偏置等常量，权重，和模型操作等数据和指令分别通过PCIE 4.0发送到对应的CB, LM和IB存储空间；之后含光800里的CP对IB里的指令进行译码并执行，同时主机通过PCIE 4.0将新的数据传输到LM，将计算出的结果写回到主机，完成流水操作，直到计算结束。
![9.png](/assets/images/ai/9.png)
含光800是专用的指令集，类似 CISC，融合不同的操作，操作数是张量级别的粗粒度数据；3 个计算引擎之间的同步发采用指令中的嵌入式位，硬件进行依赖性检查。
![10.png](/assets/images/ai/10.png)
# Baidu
百度的AI加速器始于2010年基于FPGA的SDA(Software Define Accelerator)项目，为了适应多种业务，2017年发布XPU(X procesor unit)，最后2019年正式从FPGA转向ASIC，开始Kunlun项目。具体发展历史如下图所示：
![11.png](/assets/images/ai/11.png)

## Kunlun 1
Kunlun 1采用三星14nm工艺，面积504mm^2，工作频率0.9GHz-1.1GHz, TDP功耗150W；2个HBM，一共16GB，提供512GB/s带宽；和主机接口采用PCIE 4.0 x8；提供512TOPS的INT8计算性能。下图展示了Kunlun 1的物理版图：
![12.png](/assets/images/ai/12.png)
Kunlun 1有两个计算单元，每个单元有专用的8GB HBM，提供256GB/s的带宽，和16MB片上共享存储SRAM，4个XPU-SDNN(Software defined neural network)，4个XPU-Cluster。两个单元使用NoC连接，带宽256GB/s，相当于一个HBM的带宽。XPU-SDNN主要负责矩阵计算，而XPU-Cluster则提供标量和向量计算能力。下图展示了Kunlun架构框图：
![13.png](/assets/images/ai/13.png)
XPU-SDNN针对大尺寸张量的运算进行了高度优化，如矩阵乘法、卷积、去卷积、激活和逐元素运算，主要由MAC阵列组成；从Buffer_A和Buffer_B读取操作数，经过MAC阵列和激活之后结果会被写到Buffer-C。XPU-SDNN架构框图如下所示：
![14.png](/assets/images/ai/14.png)

XPU-cluster由16个计算单元和本地存储(Local Memory)组成，计算单元分为标量单元和向量单元。标量单元可以支持常见的标量指令，主要负责标量及向量的访存地址计算，以及特殊功能单元（SFU）指令，如log、exp、sqrt、div和pow。向量单元支持256位的数据宽度。下图展示了XPU-cluster的架构框图：
![15.png](/assets/images/ai/15.png)
### 编程模型
Kunlun 1存储模型可以抽象成由计算核心和本地存储LM组成的Cluster，每个Cluster有一个共享存储SM，Cluster还可以直接访问HBM，作为GM。
![16.png](/assets/images/ai/16.png)
百度昆仑的软件栈分为两个主要组件：
* **XTDK** 包含一个C/C++编译器，XPU C/C++编译器支持数据并行编程模型，使用前缀关键字来声明函数或变量在硬件中的位置。XPU C/C++允许指针操作和内联汇编直接控制硬件。XDNN是一个优化的运算符库，具有高级数学计算，如BLAS。用户可以直接调用XDNN中定义的API来执行任务。
* **XTCL**是一个AOT/JIT张量编译器，**能够在XTDK中驱动XPU C++编译器**。所有支持的框架都可以提供子图或将预训练的模型文件导出到XTCL中。
![17.png](/assets/images/ai/17.png)

## Kunlun 2
Kunlun 2信息比较少，应该是2021年发布，采用台积电7nm工艺，相比Kunlun 1，在INT8上都是256TOPS，主要是提升FP16，从之前的64FLOPS升级到128FLOPS；从HBM替换为GDDR6，存储容量最大可达32GB；增加片间互联接口K-Link，一共4个，可提供200GB/s的总带宽；另外就是增加了Video Codec，提供了视频编解码能力；使用ARM 8核来做系统管理。Kunlun 2整体架构框图如下:
![18.png](/assets/images/ai/18.png)
Kunlun 2主要功能模块如下：
* **XPU-SDNN(Software Defined Neura Network engine)软件定义神经网络引擎**，是用来处理张量运算，通过软件编程，能灵活实现卷积，矩阵计算，激活等操作
* **XPU-Cluster**是通用计算核，用来处理标量和向量计算，支持SIMD指令。每个XPU-Cluster 有64个XPU Core，每个XPU Core有8KB的本地存储Local Memory。另外，每个Cluster还有256KB的共享存储Shared Memory
* **Video Codec**提供视频编解码、图像预处理功能，提供最高108路的解码和27路的编码能力
* **K-Link**片间互联，一共4个，提供200GB/s 总带宽

下图展示了XPU-Cluster的整体架构框图:
![19.png](/assets/images/ai/19.png)
### 系统扩展
8个Kunlun 2芯片利用4个K-Link接口进行互联，组成的系统节点之间带宽是不平衡的；每个Kunlun 2芯片通过PCIe和CPU连接。如下图所示：
![20.png](/assets/images/ai/20.png)

### 编程模型
Kunlun 2的存储模型和Kunlun 1类似，分别由Register, Local Memory, Shared Memory, Global Memory等组成，具体如下所示：
![21.png](/assets/images/ai/21.png)
Kunlun 2支持多核并行计算，类似CUDA，使用前缀来标识代码执行的硬件；kernel描述XPU上的计算程序；host端通过设置kernel函数的执行参数并发起执行。调用过程如下图所示：
![22.png](/assets/images/ai/22.png)
下面展示了执行ReLu的代码：
```cpp
int main()(
	///...
	float* data_xpu = nullptr; 
	// 在设备上分配空间
	xpu_malloc(&data_xpu, len * sizeof(float));
	// 输入拷贝到设备端
	xpu_memcpy(data_xpu, data_cpu， len * sizeof(float)， XPU_HOST_TO_DEVICE);
	// 在设备上运行relu，启动8个 cluster，每个 cluster 64个核
	relu_xpu<<<8，64>>>(data_xpu，len);
	// 获取输出,释放设备空间
	xpu_memcpy(data_cpu， data_xpu, len * sizeof(float)，XPU_DEVICE_TO_HOST);
	xpu_free(data_xpu);
}
```
```cpp

/// relu.xpu
__global__ void relu_xpu(float* data, int n) {
	int tid = core_id() * cluster_num() + cluster_id();
	int nthreads = cluster_num() * core_num();
	const int bufsize = 1024;
	_local__ float buffer[bufsize];
	// 根据tid进行数据划分
	for (int i = tid * bufsize; i < n; i= nthreads * bufsize)
		// 显式搬运数据到Local Memory 上
		GM2LM(data + i，buffer，bufsize * sizeof(float));
		// 计算
		for (int j=0;j< bufsize; j++) {
			buffer[j] = buffer[j] > 0 ? buffer[j] :0;
		}
		// 显式写回到 Global Memory
		LM2GM(buffer,data + i, bufsize * sizeof(float))
	}
}
```

# Huawei 

昇腾910(Ascend 910) 是华为2019年发布产品，计算部分采用达芬奇架构。
## Ascend 910
昇腾910采用chiplet方案，一共8个die，4个HBM，2个dummy die，1个soc die，一个NIMBUS die；其中两个dummy die用来保持芯片整体机械应力平衡; 四个HBM总带宽为 1.2TB/s；昇腾910整体布局如下图所示：
![23.png](/assets/images/ai/23.png)
不同die的面积如下所示：
![24.png](/assets/images/ai/24.png)
其中soc die主要用来计算，采用台积电7nm工艺，面积456mm^2，可以提供512TOPS的INT8性能；soc die的物理规划如下所示：
![25.png](/assets/images/ai/25.png)

SoC die包含 32 个 Ascend-Max 内核、16 个 Arm V8 TaiShan CPU 内核和 CPU LLC、视频编解码器(Digital Vision Pre-Processor)，支持128路的全高清视频解码，以及一个连接上述组件的片上网络 （NoC）。NoC采用4x6的mesh拓扑，以提供统一且可扩展的通信网络；两个相邻节点之间的链路工作频率为 2GHz，数据位宽为 1024 位，可以提供256GB/S 的带宽。NoC不使用缓冲，减少 NoC 的面积开销。SoC die和Nimbus die架构框图如下所示：
![26.png](/assets/images/ai/26.png)
### DaVinci
DaVinci核由三种计算单元、多级片上存储器和相应的加载/存储单元、指令管理单元等组成；DaVinci核是异构架构，结合了标量、向量和张量计算单元；总线接口单元（BIU）在昇腾内核和外部组件之间传输数据/指令；DaVinci架构框图如下所示：
![27.png](/assets/images/ai/27.png)
下表列出了三种计算单元支持的典型操作：
![28.png](/assets/images/ai/28.png)
* 标量计算单元主要负责地址等标量计算
* 向量计算单元可以进行归一化，激活等计算；向量单元还负责数据精度转换，例如 int32、fp16 和 int8 之间的量化和解量化操作；向量单元还可以实现 fp32 操作
* 张量计算单元主要是矩阵计算，包括卷积，全连接，矩阵乘等；张量计算中矩阵的典型尺寸为 16 x 16 x 16。因此，张量计算单元配备了 4096 个乘法器和 4096 个累加器。矩阵计算中的每个操作数被重复使用 16 次。因此，与向量单元相比，将操作数加载到张量单元的能耗降低到 1/16

DaVinci核包括多个缓冲区，分成不同层次。L0 缓冲区专用于张量计算单元，分成三个单独的 L0 缓冲区，分别是缓冲区 A L0、缓冲区 B L0 和缓冲区 C L0。它们分别用于保存输入特征数据、权重和输出特征数据。缓冲区 A L0 和缓冲区 B L0 中的数据从 L1 缓冲区加载。L0 缓冲区和 L1 缓冲区之间的通信由内存传输引擎MTE(Memory Transfer Engine) 管理。MTE 中有几个功能模块：
* decomp 模块借助零值压缩的算法来解压缩稀疏网络的数据
* img2col 模块用于将卷积转换成矩阵乘法
* trans 模块用于矩阵转置

缓冲区 C L0 中的输出结果可以由向量单元处理（例如归一化或激活）。向量单元的输出结果被分配到统一缓冲区Unified Buffer中，该缓冲区与标量单元共享。数据存储在 L1 缓冲区中，指令存储在指令缓存中。指令执行流程如下：
* 指令首先由PSQ （Program Sequence Queue） 排序
* 根据指令类型，分别分发到三个队列，即多维数据集队列(cube queue)、向量队列和 MTE 队列
* 指令分别由相应的计算单元处理 
 
由于三个计算单元和 MTE 并行工作，因此需要显式同步来确保不同执行单元之间的数据依赖关系。下图展示了相应流程，PSQ 不断向不同的单元发送指令，这些指令可以并行处理，直到遇到显式同步信号（屏障）；屏障由编译器或程序员生成。
![29.png](/assets/images/ai/29.png)
### 系统扩展
每台昇腾910服务器包含8个昇腾910芯片，并分为两组；组内连接基于高速缓存一致性网络HCCS （high-speed cache coherence network），提供30GB/S带宽 。两个组使用 PCI-E 总线相互通信，提供32GB/S带宽。整体形成hyper cube mesh网络拓扑。多台昇腾910服务器可以通过fat-tree的网络拓扑组织成一个集群。下图展示了一个 2048 节点的集群，可以提供512 Peta FLOPS的 fp16 总计算能力，包含 256 台服务器，服务器之间的链路带宽为 100Gbps。
![30.png](/assets/images/ai/30.png)
### 编程模型
PyTorch、TensorFlow、MindSpore等DNN模型开发框架位于顶端，输出“Graph”，表示算法中的粗粒度关系。然后，在图引擎的帮助下，“Graph”被转换为“Stream”，由几个按顺序排列的“Task”组成。“Streams”/“Tasks”可以直接从Operator Lib调用，也可以由程序员借助Operator Engine用不同级别的语言描述。TBE（Tensor Boost Engine）DSL（Domain Specific Language）是用Level-3编程模型开发的，称为数学编程级别，针对不了解硬件知识的用户。借助编译器，可以从 TBE DSL 描述中自动生成实例“Tasks”。程序员还可以在并行/内核级别（2 级）编程模型中开发实例“Task”，类似于 GPU 的 CUDA 或 OpenCL，并引入了张量迭代器核 TIK（Tensor Iterator Kernel）接口，可以使用 Python 进行并行编程。专用的编译器技术“Auto Tiling”，用于将大任务切割以适应 Ascend 架构。在强化学习算法的帮助下，该技术通过智能搜索合法的映射空间，为程序提供最佳的tiling和调度。编程模型的最低级别（级别 1）是 C 编程，也称为 CCE-C（Cube-based Compute En­gine）。在此级别中，每个体系结构的所有设计细节都暴露给程序员。程序员可以嵌入类似汇编的代码。整体结构如下所示：
![31.png](/assets/images/ai/31.png)

# Groq 
2016 年，Google TPU 架构师 Jonathan Ross 和 TPU 团队的其他成员创立了 Groq，Groq 采用了一种全新的架构来加速神经网络，称为软件定义的横向扩展张量流式多处理器(Software-Defined Scale-out Tensor Streaming Multi-Processor)。传统GPGPU使用一个轻量的可编程核并复制数十次或数百次，而Groq设计的TSP（Tensor Streaming Processor）是一个有数百个功能单元的单一的巨大的处理器，这种方法大大降低了指令解码开销。TSP（Tensor Streaming Processor）硬件是确定性的，但实际的吞吐量取决于编译器能否完成最佳调度。尽管TSP架构在某些方面简化了编译器调度任务的难度，但软件仍然必须协调 144 宽的VLIW执行单元，而每个VLIW有 320B的 SIMD 单元。因此，充分利用TSP庞大的MAC阵列来计算各种大小的张量是非常有挑战的。下图展示了TSP和many-core架构的区别：
![32.png](/assets/images/ai/32.png)
## TSP
TSP芯片包括一个PCI Express Gen4  x16 接口，用于连接主机处理器。Groq 提供在 x86 主机处理器上运行的软件，用于将神经网络程序和数据下载到 TSP;然后，加速器可以自主执行模型并将结果返回给主机。TSP片上一共220MB 的 SRAM，没有DRAM控制器和接口。TSP 芯片采用 14nm工艺，面积725mm2，包含 268 亿个晶体管。物理版图如下图所示：
![33.png](/assets/images/ai/33.png)
TSP 芯片一共有 20 个superlane，存储和计算单元在芯片上大致均匀分布（不包括 I/O）；指令控制只需要 3% 的芯片面积。芯片包含一个额外未使用的superlane，占4% 的芯片面积，可以替换存在制造缺陷的superlane; 这种冗余功能提高了芯片的良率。每个周期可以完成400,000 次整数乘法累加 （MAC） 操作； 同时还可以处理浮点数据，因此可以兼顾推理和训练。TSP一共有16个C2C模块，提供发送和接收原语，支持芯片间320 B向量传输。每个C2C链路带宽为30 Gbps，总片外带宽为 16 ×4 ×30Gb/s，×2 个方向 = 3.84 Tb/s 的片外互联带宽。

Groq的指令在处理器中流动，不同的时间在不同的单元中执行。如下图所示，指令首先流入Superlane 0 的功能单元并执行。在下一个周期中，指令进入Superlane 1 并执行，而 Superlane 0 执行第二组指令。该架构简化了设计和布线，消除了同步需求，并且易于扩展。在superlane中，每个时钟周期，数据水平流动，符合神经网络计算数据流，并且简化了路由。存储器嵌入在功能单元中，可以提供高带宽数据源，从而无需外部存储器。整体架构类似异构功能单元的脉动阵列，但数据只能水平移动，而指令垂直移动。TSP没有缓存、分支预测等逻辑，执行是完全确定性的，但给编译器带来了沉重的负担，编译器必须理解指令流和数据流来优化功能单元利用率。编译器必须安排所有数据移动，管理内存和功能单元，甚至手动取指。Groq宣称其编译器已经实现了类似于 Nvidia GPU 的利用率，并且将进一步改进。
![34.png](/assets/images/ai/34.png)
### Superlane
整个 TSP 执行单个指令流，因此可以把它当作是一个巨大的处理器内核。但它实际上是一个 144 宽的 VLIW 架构，每个周期发出一条指令来控制superlane。如下图所示:
![35.png](/assets/images/ai/35.png)
TSP的superlane实际上是两组镜像功能单元，Groq称为east hemisphere和west hemisphere。每个功能单元都包含多个接受指令的子单元。superlane左右对称，分成 16 个通道，每个通道的宽度为 8 位。数据从东向西和从西向东流动。
![36.png](/assets/images/ai/36.png)
一个superlane主要由 **VXM(Vector eXecute Module)**, **MXM(Matrix Multiply Engines)**, **SXM(Switch eXecution Module)** ,**MEM** 组成：
* **VXM(Vector eXecute Module)** 中间的向量单元每条通道包含 16 个 ALU。每个 ALU 都可以使用数据流里对齐的四个字节组作为操作数来执行 32 位计算。ALU除了通常的算术和逻辑运算外，还可以执行整数和浮点格式转换。VXM还执行常见的归一化函数 ReLU 和 tanh 以及幂和平方根，允许构建自己的归一化函数
![37.png](/assets/images/ai/37.png)
* **MXM(Matrix Multiply Engines)** 矩阵单元包含 320 个 MAC，分成 20 个超级单元supercell。每个 MAC 单元有两个 8 位权重寄存器和两个 32 位累加器。在每个周期，存储的权重值和数据流中的一对激活值相乘。每个 16x16 的超级单元可以在一个周期内计算一个整数部分和，并在 20 个周期内计算一个完整的 320 个元素点积。MAC 单元可以执行单个 FP16，但需要两个周期，相对于 INT8 操作，吞吐量降低了 75%。每个hemisphere有 320x320 个 MAC 单元，每个周期产生 409,600 个 INT8 操作或 102,400 个 FP16 操作。TSP 在每个方向上使用所有 32 个数据流，可以在不到 40 个周期的时间内加载所有 409,600 个权重寄存器。下图MXM框图显示了在阵列侧面加载的激活和权重，以及 INT32 或 FP32 结果从内边缘流回
![38.png](/assets/images/ai/38.png)
* **SXM(Switch eXecution Module)** 开关单元可以对张量数据进行reshape，以更好地适应计算单元。例如，它可以在通道之间对数据流旋转或转置。该单元可以复制字节以填充向量或将任何向量元素置零。开关单元是唯一可以在superlane之间通信的单元。每个单元都有两组通道可以将数据向上或向下（北/南）移动到相邻的superlane
![39.png](/assets/images/ai/39.png)
* **MEM** 每个存储单元包含 5.5MB 的 SRAM，分为 44 个切片（组），每个切片 128KB。存储单元每个周期可以执行两个 16 字节读和两个 16 字节写(访问不同的bank)，允许在东西两个方向上跨superlane的所有通道提供和接收数据。将 20 个superlane组合在一起，每个hemisphere有 110MB 的 SRAM。SRAM使用ECC进行保护，提高稳定性
![40.png](/assets/images/ai/40.png)

### 存储结构
整个TSP系统的内存层次结构可以 Rank-5 张量来寻址，\[Device, Hemisphere, Slice, Bank, Address Offset]，形状为 \[N， 2， 44， 2， 4096]。
![41.png](/assets/images/ai/41.png)
存储单元还需要存储 VLIW 指令，其宽度为 2,304 （144x16） 字节。取指占用不到的10%的SRAM总带宽。指令被解码并加载到队列中，并且允许程序预取。为了减小代码大小，可以使用`REPEAT N` 指令将前一条指令重复 N 次。由于 `NOP` 是最常见的指令，因此程序可以指定它持续 N 个周期。

流寄存器位于功能单元之间，通过编号来标识它们在superlane内的位置，如下图所示：
![42.png](/assets/images/ai/42.png)
流寄存器文件 （SRF） 用来保存操作数和结果，45 个 SRF 横跨 superlane，用于片上数据传输
![43.png](/assets/images/ai/43.png)
## 编程模型
从程序员的角度来看，数据被组织成流，这些流在物理上由每个通道一个字节（320 字节）组成。该架构支持 32 条东向数据流和 32 条西向数据流。每个数据流在每个周期中自动沿其指定方向前进，每个通道移动 32 个字节。指令对来自不同流的数据进行操作。例如，`ADD S1、S2、S3` 将数据流S1 中的每个值与数据流S2 中的相应值相加，并将结果存储在数据流S3 中。因此，每个功能单元没有固定的 32 个寄存器，而是对 32 个移动的值进行操作。每个superlane包括 16 条通道。每条指令同时在所有 16 条通道上执行，下一个周期然后在下一个superlane上执行，依此类推。在 20 个周期内，每条指令可以在 20 个superlane的所有 320 个通道上执行完成，因此它实际上变成了一个具有 20 个周期流水线的 320B的 SIMD 操作。由于架构缺少寄存器文件，因此编译器必须确保数据流在指定时间可供功能单元使用。通道结构针对 INT8 数据进行了优化，但可以通过组合数据流提供对其他操作数（INT16、INT32、FP16 或 FP32）的支持。这种方法使编译器能够对所有数据类型的 320 个元素的向量进行操作。为了简化硬件，更宽的数据类型必须沿自然对齐的边界分配给相邻数据流（例如，S0、S1、S2、S3）。为了实现高可靠性，superlane内部16 条通道上应用了 9 位纠错码 （ECC），可以纠正大部分错误，同时会记录并报告给主机软件。下图展示了superlane内的指令交错执行和数据流：
![44.png](/assets/images/ai/44.png)
由于确定性网络设计，硬件不允许反压，因为会破坏网络的确定性操作。取而代之的是，软件需要考虑每个物理链路通道上的信道带宽和延迟，来显式调度向量以确保不会使发送端溢出或接收端下溢。具体来说，当张量在网络中逐跳流动时，使用每个 TSP 上的本地 SRAM 作为中间缓冲；向量作为流控制单元 （flit）。由于网络路径是完全确定的并且接收 TSP 可以立即将向量发送到下一跳，因此可以使用虚拟直通 (virtual cut through) 流控来调度由一个或多个向量flit组成的张量。每个 TSP 都使用生产者-消费者流编程模型(producer-consumer stream programming model)进行编程，该模型允许将功能单元链接在一起。TSP 的功能单元被组织为 320 个元素的 SIMD指令执行，这些指令执行由20 个tile组成，每个tile对数据执行 16 路 SIMD 计算。各功能单元支持指令如下所示：
![45.png](/assets/images/ai/45.png)
虽然单个 TSP指令执行是确定性的，但 TSP 之间的 C2C 可能会引入非确定性，包括：
* 链路延迟变化 - 需要准确估计链路延迟
* TSP 之间没有全局同步时钟 - 需要全局时钟
* 时钟漂移 - 补偿时钟漂移

因此，在多TSP组成的系统中，需要硬件和软件支持来确保多个 TSP 之间的同步通信。将单芯片 TSP 扩展到多芯片分布式系统时，可以有效地共享全局 SRAM，TSP不使用互斥锁来保证对全局存储的原子访问，而是在特定时间使用明确的发送或接收指令进行通信，以保证程序的正确性。为了实现这种可预测的行为，TSP 软硬件接口公开了所有架构上可见的状态（所有 SRAM 和流寄存器），以便静态计算图可以表示为一系列依赖关系，这些依赖关系对所有通信的张量到达时间制定了截止时间。这些依赖关系表示为有向无环图 DAG（directed acyclic graph），以便显式地调度通信流量。下表显示了支持跨多个节点实现确定性的指令。
![46.png](/assets/images/ai/46.png)
多 TSP 系统依赖于三种机制来建立和保持同步：
* 每个 TSP的硬件对齐计数器，这些计数器每 256 个周期进行交换以保持全局共识时间
* 初始对齐程序，利用链接确保每个 TSP 同时开始执行指令
* 运行时重同步，用于补偿长时间计算的单个 TSP 时钟漂移
## 系统扩展
横向扩展多 TSP 系统的网络拓扑选择低网径、直连、软件调度的Dragonfly。当从单个 GroqChip 扩展到 GroqNode 和 GroqRack 时，RealScale 支持的性能和延迟如下所示：
![47.png](/assets/images/ai/47.png)
 GroqNode 由8个TSP芯片全连接组成，如下所示。每个 TSP 的互联带宽分为 7 个本地链路和 4 个全局链路。7 个本地链路用于同一 SMP 一致性域中的 8 个 TSP 间的全连接，并且可以与节点中的其他 TSP 同步通信。
![48.png](/assets/images/ai/48.png)
GroqNode节点内每个 TSP 的所有全局链路都可以组合在一起，以创建一个GroqRack。同时，将GroqNode节点作为具有32个端口的虚拟路由器，可以作为横向扩展的基本单位。
![49.png](/assets/images/ai/49.png)
在 GroqRack 架构中，每个GroqNode节点都连接到架构中的其他每个节点，单个 GropRack 是dragonfly拓扑，任何节点都可以当作备用节点。每个GroqNode节点都是全连接的。GroqRack如下图所示：
![50.png](/assets/images/ai/50.png)
GropRack 由9个节点组成，每个节点有8个 TSP，通过每个 TSP 的4个全局链路相互连接，总共 32 × 9 = 288 个全局链路端口：
![51.png](/assets/images/ai/51.png)
将GroqNode节点作为具有 32 个端口的虚拟路由器，TSP 系统所有节点间全连接，实现3-hop网络，可以扩展到多达 33 个节点，总共 33×8 = 264 个 TSP，并实现最小路由。
GropRack 由9个节点组成，每个节点有8个 TSP，通过每个 TSP 的4个全局链路相互连接，总共 32 × 9 = 288 个全局链路端口。为了扩展到更大的系统，我们可以将机架用作本地分组，并分出 288 个端口的一半，以使用144 个端口将每个机架中的9个节点进行双重连接。其余 144 个端口用于连接到系统中的其他机架。因此，最大配置的系统中最多提供 145 个机架，145 个机架×72 个（每个机架的 TSP），总共提供 10,440 个 TSP，使用最小路由且最多5-hop（源机架中两个，一个全局跃点， 两个在目标机架中）。

少于 16 个 TSP 的小型系统可以利用节点内丰富的链路提供高带宽通信，而多达数百个 TSP 的大型系统全局（bisection）带宽约 50 GB/s；随着系统规模增长到 264 TSP 以上，每个 TSP 端点的可用全局带宽约为 14 GB/秒。如下图所示，
![52.png](/assets/images/ai/52.png)
# SambaNova 

SambaNova Systems 成立于 2017 年，产品主要特点是可重构数据流单元RDU（Reconfigurable Dataflow Unit）。主要产品包括：
* 2019年发布的SN10，使用台积电7nm工艺制造，一共400亿晶体管
* 2022年推出的SN30，采用台积电7nm工艺制造，一共860亿晶体管
* 2023年推出的SN40L，采用台积电5nm工艺制造，一共1020亿晶体管，使用chiplet，包含两个计算芯片

## SN10
SN10 RDU使用TSMC 7nm工艺制造，一共400亿晶体管，包含4个tile，是由可编程互连的大量计算和内存单元组成，软件可以配置成独立工作或者组合成更大的tile；每个tile由160个PCU(Pattern Compute Unit)和160个PMU(Pattern Memory Unit)组成；由可编程的SWN(switch network)进行互联；片上一共300MB的SRAM，6个3200MHz的DDR4通道，提供最多1.5TB的存储。下图展示了SN10的物理版图：
![53.png](/assets/images/ai/53.png)
Tile可以以组合模式或独立模式工作：
* **组合模式** 相邻tile组合以形成一个更大tile来服务一个应用程序
* **独立模式** 每个tile独立控制，允许在不同的tile上同时运行不同的应用程序

下图展示了SN10和tile的整体架构：
![54.png](/assets/images/ai/54.png)
**PCU(Pattern Compute Unit)** 是RDU上可编程的计算引擎，可以支持FP32, BF16, INT32, INT16；主要功能模块包括：
* 可配置的多级流水SIMD运算单元
* 可编程的计数器链，负责处理循环
* Header，为SIMD提供操作数
* Tail，提供特殊函数(exponent, sigmoid)的加速
![55.png](/assets/images/ai/55.png)
**PMU(Pattern Memory Unit)** 是RDU上的分布式存储，主要组成部分包括：
* 分成不同的bank的SRAM阵列，提供并发读写能力
* 一个标量运算单元，负责地址计算
* 两个数据对齐单元(Data Align)，可以加速矩阵转置
![56.png](/assets/images/ai/56.png)
**SWN**是片上可编程的互联网络，分为数据和控制互连网络：
* 数据互联网络的路由可以由硬件和软件控制，软件可以覆盖硬件的路由规则，可编程路由表允许软件在并发的点对点和多播时提高带宽利用率。标量和向量数据网络也是分别独立的
* 控制互联网络可以由软件在不同功能单元之间路由单独的控制位来管理程序
![57.png](/assets/images/ai/57.png)
**AGCU(Address Generation and Coalescing Unit)** 是IO子系统接口，包括DDR以及PCIe。包含两个功能单元：
* **AGU(Address Generation Unit)** 负责存储加载操作，包含一个标量运算单元
* **CU(Coalescing Unit)** 负责虚拟内存管理以及对DDR通道的交织访问，同时支持RDU之间通信
![58.png](/assets/images/ai/58.png)

### 系统扩展
2个RDU组成一个SN10-2模块，4个SN10-2和一个host组成一个SN10-8系统。因为RDU本身缺少片间互联，所以都是使用PCIe，基于Ethernet进行互联。4个SN10-8可以组成一个Rack，即32个RDU SN10芯片。

### 编程模型
SN10 RDU使用SambaFlow软件栈来进行编程，SambaFlow使用编译器将使用类似Pytorch描述的模型转化成计算图，并映射到硬件的数据流。下图展示了SambaFlow的组成：
![59.png](/assets/images/ai/59.png)
下面展示了`Softmax()`的计算数据流以及其在SN10的PMU和PCU上的映射以及SWN的路由控制：
![60.png](/assets/images/ai/60.png)
下面展示了`LayerNorm`的计算数据流及其在SN10上的流水线：
![61.png](/assets/images/ai/61.png)
下面展示了`LayerNorm`的计算数据流及其在SN10上融合一些操作的流水线，可以减少存储和计算资源：
![62.png](/assets/images/ai/62.png)
下面展示了`LayerNorm`的计算数据流及其在SN10上融合一些操作同时分时复用计算和存储的流水线，可以进一步减少存储和计算资源：
![63.png](/assets/images/ai/63.png)

## SN30 & SN40L

SN30和SN40L的相关信息没有公开，不过可以推测架构本身修改不太大。目前看到信息包括：
* SN30还是台积电7nm工艺，2022年推出，860亿晶体管，SRAM增加到640MB，1TB的DDR，688TFLOPS的BF16性能；芯片采用chiplet，包含2个计算芯片
* SN40L采用更先进工艺，台积电5nm，计算单元增加到1040个，SRAM增加到520MB，支持64GB的HBM，1.5TB的DDR5

# Graphcore IPU
Graphcore是一家成立于2016年的英国公司，AI芯片为IPU(Intelligent Processing Unit)。IPU基于高度并行的架构，在混合精度浮点数据上提供了非常高的浮点性能，支持单精度浮点或半精度浮点。目前产品分别是
* 2018年推出的Colossus Mk1，采用台积电16nm工艺，216亿晶体管，是一款测试芯片，主要用来验证架构可行性
* 2020年推出的GC200 "Colossus Mk2"，采用台积电7nm工艺，593亿晶体管
* 2022年推出的Bow IPU，采用台积电7nm工艺，并使用Wafer-to-Wafer来给计算芯片供电

## Colossus Mk1
Colossus Mk1 IPU采用台积电16nm工艺，216亿晶体管；包含1,216个tile，304M的片上SRAM；tile由一个计算核IPU-Core和 256 KB 的本地存储组成；每个IPU-Core支持6个线程，IPU一共支持6x1216=7296个线程；每个IPU还包含10个IPU-Link接口，用于IPU处理器之间实现低延迟、高吞吐量的通信。另外，IPU包含两个PCIe Gen4 x16，用于主机通信。整体架构如下所示：
![64.png](/assets/images/ai/64.png)

## Colossus Mk2
Graphcore Colossus Mk2 GC200 IPU采用台积电 7nm 工艺，面积 823mm^2，一共 594 亿晶体管。IPU包含1,472个IPU-Tile，并能够执行8,832个独立的并行计算线程。片上一共897MB的SRAM。有250 TFLOPS的在FP16计算能力。IPU物理规划版图如下所示：
![65.png](/assets/images/ai/65.png)
IPU能够将所有算术运算保持在16位格式，从而降低内存需求，降低读写功耗并减少算术逻辑功耗。每个处理器核都可以生成指定噪声的随机数种子，支持概率或进化策略模型。整体架构和MK1比较类似，每个IPU除了1,472个IPU-Tile，还包含10个IPU-Link接口，用于IPU处理器之间实现低延迟、高吞吐量的通信，提供320GB/s的带宽。另外，IPU包含两个PCIe Gen4 x16，用于主机通信。整体架构如下所示：
![66.png](/assets/images/ai/66.png)
每个tile由一个多线程处理器和624 KB 的 SRAM本地存储组成，多线程处理器支持32位指令，最多双发射，2个执行单元：
* **MAIN** 负责控制流，整型和地址计算以及存储加载操作
* **AUX** 负责浮点算术运算，向量和矩阵运算，超越函数(ln, log2 , logistic, tanh)，随机数生成等

多线程处理器的结构如下所示：
![67.png](/assets/images/ai/67.png)
每个IPU tile维护6个常驻执行上下文，并将它们多路复用到共享资源上，从而隐藏指令延迟（依赖关系、内存访问和分支延迟），减少相应的流水线停顿，并增加聚合吞吐量。每个tile线程通过静态round-robin方式轮换。因此，整个IPU支持6×1,472 = 8,832个线程。为了获得最大占用率，程序需要实例化尽可能多的线程。

每个 tile 有 624 KB 的 SRAM，1,472个tile总共897 MB的SRAM。每个tile都使用一个连续的无符号 21 位地址空间，从地址 0x0 开始。实际上，真实可用内存从地址 0x4C000 开始，到 0xE7FFF 结束。下图展示了地址空间：
![68.png](/assets/images/ai/68.png)
片上内存分为两个区域，每个区域由几个 64 位宽的bank组成。不同的bank支持并发访问，而对同一bank的多次访问必须是串行的。所有加载和存储地址必须自然对齐，且只能访问本tile的片上内存：
* 区域 1 中的bank是交织的，地址的第 3 位从交替的奇数和偶数bank中选择 64 位。交织允许同时访问两个 64 位对齐的地址。因此，通过读取来自两个不同bank的连续字，一条指令可以执行 128 位加载
* 区域 0是非交织，指令只能从区域 0 获取

IPU中的内存由tile本地存储和Streaming Memory组成，通过**Exchange-Memory**访问Streaming Memory。Streaming Memory由多个 DDR 存储器芯片组成。tile无法直接执行加载和存储指令访问Streaming Memory。下图展示了IPU的存储结构：
![69.png](/assets/images/ai/69.png)
显式数据移动指令通过使用IPU交换互联网络，可以将数据在同一IPU的不同tile上的存储之间以及Streaming Memory 和tile本地存储之间移动。在执行数据移动指令之前，所有tile必须同步，保证所有tile都已完成执行其计算任务，并已准备好进行数据交换。

## 系统扩展
每个IPU-Machine M2000包含四个 Colossus Mk2 GC200 IPU处理器，提供 1 petaFLOP 的 AI 计算能力，有 3.6GB 处理器内存和256GB 的Streaming Memory。机架级IPU-POD64包括16个内置于标准19英寸机架中的IPU-Machine M2000。通过IPU-Fabric，IPU-Machine可以构建横向扩展的IPU-POD数据中心解决方案，最多扩展到64,000个IPU，提供16 ExaFlops的AI 计算能力。下图展示了IPU-Machine M2000连接拓扑：
![70.png](/assets/images/ai/70.png)
下图展示了IPU-Machine M2000的前视图：
![71.png](/assets/images/ai/71.png)

IPU-Machine M2000系统包括一个Graphcore GC4000 IPU网关芯片，提供2.8 Tbps带宽。为了和数据中心网络保持兼容，IPU-Fabric使用标准铜缆或光纤OSFP连接器，将IPU连接在机架上下。在较大的配置中，IPU-POD之间的通信使用以太网隧道技术来保持吞吐量，同时允许使用标准QSFP互连和100Gb以太网交换机。在整个系统中，IPU-Fabric使用3D环形拓扑结构。下图展示了IPU-Fabric的3D 环形拓扑结构：
![72.png](/assets/images/ai/72.png)

## 编程模型
主机将代码加载到一个或多个IPU上，然后IPU执行代码。IPU有两种模式：supervisor 和worker。
* 最多有一个线程可以在 supervisor 模式下运行，而在 supervisor 模式下运行的线程可以生成在worker下的线程。supervisor 线程负责运行控制tile的程序，是IPU上运行的应用的主程序。
* 在worker模式运行的任务完成后将退出。worker线程执行时，supervisor线程将挂起；当worker线程任务完成并释放线程时，supervisor线程将恢复。

处理器固定顺序逐条执行线程指令，每个线程以round-robin方式轮流。大多数指令需要一个计算周期才能执行完成，处理器指令集是专为机器学习和人工智能设计，包含：
- **控制流指令**, 包括跳转、条件等。每个处理器上的控制流独立于其他处理器上的控制流
- **内存访问指令** 
- **整数和浮点算术运算指令** 浮点指令包括单精度（32 位）和半精度（16 位）浮点运算。浮点运算可以支持大小为 2、4 和 8 的小向量。此外，每个tile上都有一个累积矩阵乘积单元（AMP 单元），每个周期最多可以执行 64 次乘法累加运算
- **超越函数指令** 例如指数函数
- **随机数生成指令** 随机数生成器还连接到浮点单元，以便在执行浮点运算时在硬件中启用随机舍入

当程序在IPU上执行时，IPU内的tile在交换数据和对其本地数据执行计算之间交替。IPU使用bulk-synchronous parallel（BSP）执行模型，其中任务的执行被拆分为不同的step，如下所示，每个step包括以下阶段：
- 本地tile计算
- 全局跨tile同步
- 数据交换
![73.png](/assets/images/ai/73.png)
硬件全局同步，片上需要大概150 个周期，芯片之间 15ns/hop。在计算阶段，所有tile都并行执行，对其本地数据进行操作。每个tile完成执行后，会启动一个同步过程，以便与其他tile同步。当所有tile都达到这一点时，所有tile都将同步完成，IPU进入交换阶段，在tile之间复制数据。具体过程如下所示：
![74.png](/assets/images/ai/74.png)
在交换阶段之后，重复该过程：tile进入新的计算阶段，使用其本地数据和交换期间收到的新数据执行计算。程序继续执行一系列此类步骤，在交换和计算阶段之间交替进行。从时间轴上看，我们可以看到每个tile重复执行同步、交换和计算的顺序，如图所示：
![75.png](/assets/images/ai/75.png)
上述编程模型是在Poplar graph library（`libpoplar`） 中实现，提供了用于构建和运行IPU程序的API，并执行必要的编译以在IPU设备上运行程序。程序在一组用户可配置的IPU上运行，并在编译程序之前选择，并且在程序执行过程中不能改变。tile上执行的线程称为 _vertices_，所有线程称为 _compute sets_ 。相关概念如下所示：
![76.png](/assets/images/ai/76.png)

## Bow IPU
**BOW IPU**第一个使用台积电Wafer-on-Wafer 3D 技术的芯片，Wafer-on-Wafer可以在芯片之间提供更高的带宽，并被用于优化电源效率，并在晶圆级别改善 Colossus 架构的供电效率。BOW IPU中的Wafer-on-Wafer，将两个晶圆键合在一起以产生一个新的3D芯片：
* 一个用于AI处理的晶圆，在架构上与GC200 IPU处理器兼容，有1,472个独立的IPU-tile，能够运行超过8,800个线程，具有900MB的处理器内存
* 一个用于供电的晶圆，通过在供电芯片中增加深沟槽电容器，并紧挨着处理内核和存储器，能够更有效地供电，使性能提高 40%，每个处理器可提供 350 TFLOPS 的 AI 计算能力

Bow-2000 IPU类似IPU-M2000，有四个Bow IPU和260 GB的内存，可提供1.4 PetaFLOPS的AI计算性能；并可以作为Bow Pod系统的基本单元，从四个 Bow-2000 和一个主机组成的 Bow Pod16，到8 个 Bow-2000 和一个主机组成的 Bow Pod32，再到 Bow Pod64，Bow Pod256 和 Bow Pod1024。Bow-2000与现有的IPU-POD系统完全向后兼容，其高速、低延迟的IPU-Fabric和灵活的1U外形尺寸都保持不变；IPU-Fabric由用于直接连接IPU的IPU-Link和用于通过IPU网关连接IPU机器机架的GW-Link组成。Bow-2000 IPU和Bow Pod如下所示：
![77.png](/assets/images/ai/77.png)
![78.png](/assets/images/ai/78.png)
# Intel 
Intel在并行计算和AI领域大致发展历史如下：
- 从2007年开始，Intel就计划开发独立显卡项目Larrabe, 试图打破软硬件的隔阂，使用同样架构来做图形和并行计算。但是随后项目就被取消。后续的Xeon Phi系列加速器（天河-2超算中心使用）是在Larrabe上长出来的，但是随着美国的禁令，Intel在2017年就停止了相关产品的开发
- 2016年，英特尔宣布以 3.5 亿美元收购收购Nervana，并推出了NNP-T(Nervana Neural Network Processor for Training)和NNP-I(Nervana Neural Network Processor for Inference)加速器芯片，但是在2020年宣布停止相关产品的开发，采用Habana加速方案
- 2019年，英特尔宣布以 20 亿美元收购总部位于以色列的 AI 创业公司 Habana Labs，并推出了Gaudi 2和Gaudi 3加速器芯片，但是2023年，Intel路线图显示后续Gaudi系列加速器会和Intel Max系列GPU合并
- 2017年，Intel重启独立显卡计划，推出Xe系列GPU

## Xeon Phi
英特尔于 2007 年正式宣布推出 Larrabee；但是Larrabee 无法与NVIDIA和AMD提供的高端硬件相匹配，英特尔最终搁置了其 GPU 野心。随后将 Larrabee 改名为Xeon Phi，用于HPC，主要产品如下：
* **Knights Ferry（KNF）**，2010年推出并交付给HPC开发人员，采用Intel 45nm工艺，面积684mm^2，23亿晶体管；Knights Ferry 是Larrabee GPGPU实现产品，并没有被设计为商业产品
* **Knights Corner**，2012年发布，采用Intel 22nm工艺，面积720mm^2，50亿晶体管；并重新命名为Xeon Phi
* **Knights Landing**，2014年发布，采用Intel 14nm工艺，面积683mm^2，80亿晶体管
* **Knights Mill**，2017年发布，采用Intel 14nm工艺，面积826mm^2，542亿晶体管；由于美国禁令，并未实际部署

2013 年 1 月，德克萨斯州奥斯汀的德克萨斯高级计算中心宣布了 Stampede 超级计算机，这是Xeon Phi的首次大规模部署，在 6400 个计算节点中使用了6880 个Xeon Phi，性能达到近 10PFLOPS。2013 年 6 月，中国超级计算机“天河二号”成为世界上最快的超级计算机，并一直保持到 2015 年底。 它由 32,000 个 Intel Xeon E5-2692 2.2GHz 12C Ivy Bridge 处理器和 **48,000 个 Xeon Phi** 协处理器提供支持，性能超过 33PFLOP。

### Larrabee
Larrabee 架构使用顺序执行CPU，基于x86 扩展指令集，包括 64 位指令，宽向量处理操作和一些专用标量指令。GPU 上一些固定硬件功能例如光栅化和后着色器混合等，在 Larrabee 中都是软件执行；但是纹理过滤与 GPU 一样，还是固定硬件逻辑实现；，但是CPU更灵活，支持子程序和页面错误。Larrabee 整体架构框图如下：
![79.png](/assets/images/ai/79.png)

Larrabee CPU核源自奔腾处理器，顺序执行，支持四个线程。Larrabee CPU核架构框图如下所示：
* L1 指令缓存大小为 32KB，L1 数据缓存大小为 32KB。环形网络将L2缓存和CPU核互联，L2缓存负责缓存一致性。
* Larrabee 的全局L2缓存被分割成单独的本地子集，每个 CPU 核一个，256KB。每个 CPU 都可以直接访问自己的本地 L2 缓存子集，CPU 核读取和写入的数据存储在其 L2 缓存子集中，并在必要时从其他子集刷新。
* Larrabee 支持四个线程，每个线程有单独的寄存器。线程切换涵盖了编译器无法在不停顿的情况下调度代码的情况，以及无法提前将数据预提取到 L1 缓存时从 L2 缓存加载到 L1 缓存的部分延迟。当在同一核心上运行的多个线程使用相同的数据集(例如将三角形渲染到同一图块)时，缓存使用更高效。

![80.png](/assets/images/ai/80.png)

Larrabee 主要计算单元是 16-wide向量处理单元VPU，负责执行整数、单精度浮点和双精度浮点指令。VPU 及其寄存器的面积约为 CPU 核面积的三分之一。VPU 支持三操作数指令，并支持旋转寄存器输入以及存储器输入上的数字转换和复制。下图展示了VPU的架构框图：
![81.png](/assets/images/ai/81.png)

VPU选择了 16-wide来平衡计算密度和向量单元利用率。分析表明，如果 16 个通道一次处理 16 个单独的像素，典型像素着色器的利用率为 88%。Nvidia GeForce 8 以类似的方式运行，其标量 SIMD 处理器组织成 32 个一组，执行相同的指令。主要区别在于，在 Larrabee 中，循环控制、缓存管理等操作是与 VPU 并行运行的代码，而不是作为固定功能逻辑实现。

Larrabee VPU 指令允许最多三个源操作数，其中一个可以直接来自 L1 缓存。如果数据已预取到缓存中，则 L1 缓存实际上是一个扩展寄存器文件。从缓存中读取的8 位 uint、16 位 sint 或16 位浮点数据可以转换为 32 位浮点或 32 位整数，而不会损失性能，这种方式增加了可以存储在缓存中的数据量，并且还减少了对单独数据转换指令的需求。之后将寄存器和存储器中的数据与 VPU 中的处理通道对齐。寄存器数据可以通过多种方式进行旋转，例如支持矩阵乘法。内存中的数据可以跨 VPU 通道复制。这是图形和非图形并行数据处理中的常见操作，可显著提高缓存效率。VPU 支持处理整数和浮点数据的各种指令。指令集提供标准算术运算（包括融合乘加）和标准逻辑运算（包括从像素中提取非字节对齐字段的指令）。这些指令从寄存器或存储器中读取操作数并将结果写入向量寄存器。加载和存储指令支持浮点值与大多数 GPU 上不太常见或更复杂的数据格式之间的转换，使用单独的指令处理这些数据格式转换可以节省大量面积和功耗，同时降低性能成本。

VPU 指令集还支持gather和scatter，即从非连续地址加载和存储。不是从单个地址加载 16 个宽的向量，而是从另一个向量寄存器中指定的最多 16 个不同地址加载或存储 16 个元素；这种方式可以并行运行 16 个着色器实例。gather和scatter的速度受缓存的限制，缓存通常每个周期只访问一行缓存。但是，许多任务具有高度一致的访问模式，因此执行所需的时间远少于 16 个周期。Larrabee VPU指令可以由掩码寄存器预测，每个向量通道有一个位。掩码寄存器控制向量寄存器或存储器的哪些部分被写入，哪些部分保持不变。例如，通过设置掩码寄存器，然后执行if 和 else 子句，根据掩码寄存器来控制是否写入结果，可以将标量 if-then-else 控制结构映射到 VPU。如果掩码寄存器全是零或全是 1，则可以完全跳过子句。这减少了分支错误预测惩罚，并为编译器的指令调度器提供了自由。VPU 还使用这些掩码来打包加载和存储指令，从而顺序访问内存中的数据。

Larrabee 使用双向环形网络对CPU核、L2 缓存和其他逻辑进行互联。当扩展到超过 16 个核时，使用多个短环。每个环的数据宽度为 512 位。在消息注入网络之前路由决策已经确定，例如，每个代理都可以在偶数时钟上接受来自一个方向的消息，在奇数时钟上接受来自另一个方向的消息。这种方式简化了路由逻辑，并且一旦消息进入网络就不需要存储，以非常低的成本实现高带宽和最小的争用。每个核可以并行访问自己的 L2 缓存子集，而无需与其他核通信。但是，在 L2 缓存中分配新行之前，环形网络会检查数据共享，以保持数据一致性。处理器间网络还为 L2 缓存提供访问内存的路径。

#### 编程模型
Larrabee天然支持x86上各种编程模型，包括POSIX线程和OpenMP等。

### Knights Ferry
**Knights Ferry**通过标PCI Express 作为协处理器和主机连接。Knights Ferry由 32 个 Pentium （P54C）通用核组成，运行频率为1.2GHz，可以执行 64 位标量指令和 512 位向量指令（每个向量指令 16 个单精度或 8 个双精度浮点值）。每个核可以执行四个硬件线程，使用round-robin进行调度，以便在每个周期中选择下一个指令流。Knights Ferry 使用每核32 KB的L1缓存(指令和数据各32KB)和256 KB的L2缓存的典型缓存结构。共享的 L2 缓存总共为 8 MB（32 个核），使用高带宽环形总线实现片上通信。内存支持1,800 MHz的GDDR5，提供125 GB /s带宽。下图展示了Knights Ferry的物理版图：
![82.png](/assets/images/ai/82.png)
### Knights Corner
**Knights Corner**采用Intel 22nm工艺，集成了 61 个 Pentium P54CS， 主频 1.2GHz。下图展示了 Knights Corner物理版图：
![83.png](/assets/images/ai/83.png)
CPU核，GDDR和PCIe通过环形总线互联，如下所示：
![84.png](/assets/images/ai/84.png)
CPU核是双发射顺序执行，支持四个硬件线程，增加了对 64 位指令的支持，和512 位宽 SIMD向量处理单元VPU；每个核 64K的L1 缓存（32K 指令和32K 数据）， 512K的L2 缓存；每个核有 4 个执行单元（VPU、FXU 和 2 个整数单元）。CPU核架构框图如下所示：
![85.png](/assets/images/ai/85.png)

### Knights Landing
**Knights Landing**是一个标准的独立处理器，可以启动现成的操作系统。KNL提供三种产品类型：KNL自启动处理器、带集成结构的KNL自启动处理器，以及作为KNC后续产品的KNL PCIe协处理器卡。KNL CPU 包含 38 个物理tile，最多使用 36 个，剩下的两个用于提升良率。下图展示了KNL的物理版图：
![86.png](/assets/images/ai/86.png)
每个tile包含两个核、每个核两个向量处理单元 （VPU） 以及两个核共享的 1 MB 的L2缓存。CPU核是源自 Intel Atom （基于 Silvermont 微架构）的全新双发射乱序核，每个核 4 个线程；KNL 推出了新的高级向量扩展指令集 AVX-512，提供 512 位宽的向量指令和更多的向量寄存器。KNL 引入了一种新的 2D 缓存一致性mesh互联，用于连接tile、内存控制器、I/O 控制器和其他逻辑；支持MESIF（modified, exclusive, shared, invalid, forward）缓存一致性协议； 采用分布式标签目录来维护 L2 缓存一致性。每个tile都包含一个home agent，保存分布式标记目录的一部分，并用作tile和互联网络之间的连接点。下图展示了KNL整体架构框图：
![87.png](/assets/images/ai/87.png)
KNL 有两种类型的存储器：多通道 DRAM （MCDRAM） 和DDR存储器。
* MCDRAM 是 16 GB 高带宽存储器，包括 8 个器件（每个器件 2 GB），集成在封装上，并通过专有的I/O 连接到 KNL 芯片。8 个 MCDRAM 提供450 GB/s 带宽。
* KNL 有两个内存控制器，一共 6 个 DDR4 通道，运行频率2,400 MHz，提供90 GBps 的总带宽。每个通道最多可以支持一个内存 DIMM，一共384 GB的总 DDR 内存容量。
 
两种类型的内存以三种内存模式呈现给用户：
* **缓存模式** 其中 MCDRAM 作为DDR 的缓存
* **扁平模式** 其中 MCDRAM 被视为与 DDR 相同的地址空间中的标准内存
* **混合模式** 其中 MCDRAM 的一部分是缓存，其余部分作为标准内存

KNL 中的三种内存模式在启动时通过 BIOS 进行选择。MCDRAM 内存提供满足大型计算能力所需的高带宽，而 DDR 提供在 KNL 上运行整个应用程序及所有必要的支持软件所需的大容量内存。

KNL 一共有 36 个 PCIe Gen3 通道，分为两个 x16 通道和一个 x4 通道；还有四通道DMI(Direct Media Interface)，可连接到南桥芯片，支持自启动系统所需的功能。

片上mesh互连网络基于环形架构，一共有四个并行网络，每个都提供不同类型的数据包（例如，命令、数据和响应），并针对 KNL 流量和协议进行了高度优化。mesh可以提供超过 700 GBps 的总聚合带宽。使用 YX 路由规则，消息在注入点和转弯时与网格上的现有流量进行仲裁，mesh上的现有流量具有更高的优先级。静态 YX 路由有助于减少死锁情况，从而简化协议。在 Y 方向上每一个hop需要一个时钟，在 X 方向上需要两个时钟。mesh支持三种集群操作模式，在启动时从BIOS中选择，可提供不同级别的地址亲和力，以提高整体性能。这些集群模式通过降低协议流在芯片上遍历的距离来降低延迟并提高带宽。
* **All-to-all 模式** 此模式在tile、目录和内存之间没有任何关联，是最通用的模式，对软件的内存配置没有特定要求，但它的性能通常低于其他集群模式。
* **象限模式** 此模式将 KNL 芯片划分为四个虚拟象限，每个虚拟象限提供目录和内存之间的亲和力。tile与目录或内存之间没有关联性，也就是说，来自任何tile的请求都可以访问任何目录。但是，该目录将仅访问其自身象限中的内存。此集群模式需要对称内存（即两个 DDR 内存控制器上的总容量相同）。它提供比全对全模式更好的延迟，并且对软件支持是透明的。
* **Sub-NUMA clustering（SNC）** 此模式通过将tile与目录和内存关联来进一步扩展象限模式。在此模式下，KNL 芯片被划分并作为两个或四个非一致性内存访问 （NUMA） 域公开给操作系统。对于 NUMA 优化的软件，tile、目录和内存之间将存在关联性，也就是说，来自tile的请求将访问其集群中的目录，而该目录又将访问该集群中的内存控制器。此集群模式在所有模式中具有最低的延迟，尤其是在加载操作下，因为大多数流量将包含在本地集群中。要使软件利用此模式的性能，它必须经过 NUMA 优化，也就是说，需要在运行它的同一 NUMA 群集中分配内存。

CPU核大致分为五个单元：前端单元FEU（ front-end unit）、分配单元、整数执行单元IEU（integer execution unit）、内存执行单元MEU（memory execution unit）和VPU。下图为CPU核的框图：
![88.png](/assets/images/ai/88.png)
* ***前端单元** CPU核的 FEU 包括一个 32 KB 指令缓存 （IL1） 和一个 48 项的指令 TLB。如果命中，指令缓存每个周期最多可以提供 16 个字节。然后，被发送到一个两宽的解码器。大多数指令被解码为单个微操作，但一些产生更多微操作的复杂指令由双宽微序列器引擎(micro sequencer engine)处理。取指方向由 gskew 分支预测器提供。解码后的微操作被放入 32 项的指令队列中。
* **分配单元** 分配单元每个周期从指令队列中读取两个微操作，分配微操作所需的必要流水线资源，例如重排序缓冲 （ROB）（72）、重命名缓冲 （72）、存储数据缓冲 （16）、gather-scatter表 （4） 和保留站。还负责重命名寄存器。重命名缓冲区存储正在进行的微操作的结果，直到它们退休，此时结果将传输到架构寄存器文件。在分配单元之后，微操作根据类型将发送到三个执行单元之一（IEU、MEU 或 VPU）。一些微操作可能会被发送到多个执行单元。例如，内存地址作为操作数的 Add 指令将发送到 MEU 以读取内存，然后发送到 IEU 以执行 Add 操作。
* **整数执行单元IEU** IEU 执行整数运算，使用通用寄存器 R0 到 R15 。CPU核中有两个 IEU。每个 IEU 包含一个 12 项的保留站，每个周期发出一个微操作。整数保留站的调度完全乱序。大多数操作单周期完成，并且两个 IEU 都支持。但少数有三个或五个周期的延迟（例如，“乘法”），并且仅有一个 IEU 支持。
* **内存执行单元MEU** MEU 执行内存操作，并在指令缓存未命中和指令 TLB 未命中时发出读请求。每个周期MEU 可以执行两个存储或加载内存操作。内存操作从 12 项的内存保留站按顺序发出，无序执行和完成。未成功完成的操作将分配到回收缓冲区(recycle buffer)中，并在冲突条件解决后重新发射到 MEU流水线。已完成的加载指令将保留在内存排序队列(memory ordering queue)中，直到退休。存储指令在地址转换后保留在存储缓冲区(store buffer)中，可以将数据转发到依赖的加载指令。存储指令按程序顺序每个周期一个提交到内存中。L1 micro TLB采用8 路组相联，一共64项；L2 数据TLB也是8路组相联，256项。数据 TLB 还包含一个用于 2 MB 页面的 8 路， 128 项的表和一个用于 1 GB 页面的全相联的 16 项的表。L1 数据缓存8 路组关联，32 KB，写回、非阻塞，支持两个512 位读和一个 512 位写，整数的加载到使用延迟为 4 个周期，浮点的加载使用延迟为 5 个周期。L1 硬件预取器监控内存地址模式，并向 L2 缓存发出数据预取请求，提前读入缓存行。MEU 包含专门的逻辑，可以有效地处理gather-scatter 指令。单个 gather-scatter 指令可以访问多个内存位置。这些多重访问是在非常接近 L1 缓存流水线的地方完成的。这允许最大限度地利用两个内存流水线，同时在核心的其余部分（例如 FEU、分配单元、保留站和重排序缓冲区）消耗最少的资源中。
* **向量处理单元 VPU** VPU 是 KNL 的向量和浮点执行单元，支持 x87、MMX、SSE（Streaming SIMD Extension）、AVX 和 AVX512 指令以及整数除法。每个核两个 VPU，这些 VPU 大多是对称的，每个 VPU 每个周期可以执行一条 AVX-512 指令，两个 VPU 每个周期提供 64 个单精度或 32 个双精度浮点运算的峰值性能。其中一个 VPU 经过扩展，可支持过时浮点指令，例如 x87、MMX 以及字节和字 SSE 指令子集。每个 VPU 有一个 20 项的浮点保留站，每个周期无序发出一个指令。


浮点保留站与 IEU 和 MEU 保留站的不同之处在于，为了减小大小，不保存源数据; 浮点微运算从浮点保留站发出后，从浮点重命名缓冲区和浮点寄存器文件中读取其源数据，与整数和内存指令相比，在保留站和执行之间额外花费一个周期周期。大多数浮点算术运算的延迟为 6 个周期，而其余运算的延迟为 2 或 3 个周期。VPU 还支持 KNL 中引入的超越函数和倒数指令，向量冲突检测指令。

在 KNL tile上，两个核共享一个  1 MB，16 路关联的统一 L2 缓存。总线接口单元 （BIU） 维护tile内一致性，还充当本地共享 L2 缓存管理单元。L2 缓存中的行保持在 MESIF 状态之一。每个内核使用专用请求接口向 BIU 发出请求。可缓存请求查找 L2 标签以进行命中、未命中和缓存状态评估，而其他请求则直接绕过缓存，由目标缓存/home代理提供服务。KNL 实现了独特的缓存拓扑，以最大程度地减少缓存一致性维护流量。首先，L2 缓存包括 L1数据缓存，但不包括 L1指令缓存。读入 L1指令缓存的行会填充 L2 缓存，但当这些行被逐出时，相应的 L1指令缓存行不会失效。这避免了由于热 IL1/冷 L2 场景而导致的失效，在这种情况下，由于不活跃而从 L2 缓存中逐出相应的行而导致L1指令缓存中处于活跃状态的行失效。其次，L2 缓存每行存储“存在”位，以跟踪哪些在 L1数据缓存中被活跃使用。此信息用来过滤对包含的L1数据缓存的缓存侦查 。它还考虑了 L2 受害者选举算法，以最大程度地减少对正在使用的缓存行的逐出。BIU 还包含一个 L2 硬件预取器，根据来自核的请求进行训练，支持 48 个独立的预取流。一旦检测到向前或向后稳定的请求，就会向该流中的按步进连续缓存行发出预取请求。
![89.png](/assets/images/ai/89.png)
KNL 核支持四个硬件上下文或线程。CPU核内资源可以动态分区、共享或复制，并且流水线由线程选择器控制。目的是最大限度地提高活动线程的资源利用率。一般而言，线程在执行 `halt` 或 `monitor wait （mwait）` 指令后变为非活月状态;否则，它们被视为活月状态。根据活跃线程的数量定义了三种线程模式：
* 单线程模式 只有一个线程处于活跃状态（任意一个），可以使用全部的 ROB
* 双线程模式 任意两个线程处于活动状态，每个线程使用32项的ROB
* 四线程模式 任意三个或所有四个线程处于活动状态，每个线程使用18项的ROB

除了 ROB 之外，包括重命名缓冲区、保留站、存储数据缓冲区、指令队列和gather-scatter 表都会动态分配。包括缓存、TLB、大多数 MEU 结构、分支预测器和硬件预取器表等共享资源不强制分区，尽管某些结构具有少量每个线程的保留条目以避免死锁。线程以先到先得的方式获取共享资源。除了缓存(缓存行不区分线程)之外，共享结构中的特定条目仅由单个线程拥有。
复制的结构被限制在最低限度。在复制结构中，每个线程都有其专用结构，当线程变为非活跃状态时，该结构不会被移除。这些结构包括重命名表、体系结构寄存器和其他控制寄存器。CPU核流水线在多个点上有线程选择器，以最大限度地提高流水线的利用率和吞吐量，并保持线程之间的公平性。它们主要存在于流水线的有序部分。线程选择器会考虑流水线中资源的可用性，并尝试做出明智的选择。重要的线程选择器位于 FEU、分配单元、退休流水线和 MEU 保留站中。无序部件与线程无关，并根据准备情况执行指令。

## Nervana
NNP-T是2019年推出的一款AI训练加速器，而NNP-I则是一款AI推理加速器；两者采用不同架构，导致软件不兼容。应该也是Intel放弃相关技术的主要原因。
### NNP-T
**NNP-T**采用台积电16nm FF+工艺，有270亿个晶体管，680平方毫米，频率1.1GHz，TDP为150~250W；下图展示了NNP-T的物理版图：
![90.png](/assets/images/ai/90.png)
NNP-T包含24个张量处理器TPC(Tensor Processing Core)，每个TPC有2.5MB的scratchpad memory，一共60MB片上存储SRAM；4个8GB的HBM2-2400内存，提供1.22TBps带宽；4个用于片间互联的ICL接口；一个PCIe 4 x16接口；每个张量处理单元(TPC)都有一个微控制器uController，用于控制协处理器的运算，允许自定义指令触发微控制器中的子程序以执行特定操作。下图展示了NNP-T的架构框图：
![91.png](/assets/images/ai/91.png)

每个 TPC 有2 个32x32矩阵乘法核Matrix Multiply Core，支持BF16的矩阵乘法，其他操作支持FP32 和 BF16，包括非 GEMM 的向量运算。张量处理核TPC有可以同时读取和写入的双端口SRAM，以及一个可以从内存中读取数据并使用卷积滤波器进行转换以进行矩阵乘法的卷积引擎(Convolution Engine)。数学运算发生在矩阵乘法核Matrix Multiply Core中，流水线支持乘法前的预操作，然后对最终结果进行多次操作。矩阵乘法核同时输出前操作和后操作的结果，可以最大程度地减少通过计算流水线进行连续数据移动的需要。
![92.png](/assets/images/ai/92.png)

TPC 连接到片上网络，该网络由双向 2D mesh组成，每个方向为 1.3TBps带宽，TPC 之间有单独的总线来移动数据，可以在不访问 HBM2 存储器子系统的情况下将数据移出芯片。这缓解了神经网络密集的读操作常见拥塞点，每次读操作需要多次访问 HBM会产生内存瓶颈，阻止内核得到充分利用。每个 TPC 有四条高速总线，其中两条专用于 HBM2 存储器，另外两条负责与其他 TPC 的通信。
![93.png](/assets/images/ai/93.png)

#### 系统扩展
NNP-T一共有16 个 112Gbps Serdes， 分成4个接口，总双向带宽为 3.58Tbps；4个NNP-T全连接组成一个基本节点，并提供4个对外互联接口；节点之间组成环形拓扑，可以扩展到支持1024个NNP-T。如下所示：
![94.png](/assets/images/ai/94.png)

### NNP-I
**NNP-I**采用Intel 10nm工艺，主要面向推理工作，可以安装在标准 M.2 设备上，然后将其插入主板上的标准 M.2 端口，可以卸载推理密集型，将CPU用于一般计算任务。NNP-I基于10nm Ice Lake 芯片，删除了两个CPU核和图形引擎，增加了 12 个推理计算引擎ICE （Inference Compute Engine）。ICE 加速器之间基于硬件同步，与 Sunny Cove 微架构的两个 IA 核共享一致性总线和 24MB 的 L3 缓存。IA 核是标准的 Ice Lake 核，支持 AVX-512 和 VNNI 指令，可加速卷积神经网络。有两个LPDDR4X内存控制器，提供 4.2 GT/s （68 GB/s） 的带宽，并支持带内 ECC。支持PCIe 4 x4/x8。
![95.png](/assets/images/ai/95.png)每个ICE单元有4MB的SRAM，以帮助减少芯片内数据移动。深度学习计算网格(Deep Learning Compute Grid)是一个张量引擎，每周期可完成4K MAC运算，支持 FP16 和 INT8，也可以支持 INT4、2 和 1，但是不支持 bfloat16；和SRAM以及VP6 DSP通过数据和控制总线互联。DSP用于向量计算，支持 INT8、16、32 和 FP16 的宽 VLIW；Ice Lake 核可以使用 VNNI 运行其他代码。下图展示了ICE单元架构框图：
![96.png](/assets/images/ai/96.png)

L3 缓存被分解成 8 个 3MB 的切片，在 IA 核和 ICE 单元之间共享。对于ICE，存储结构一共分成4层。下图左侧量化了每一层移动数据的延迟影响，以 DRAM 到 DL Compute Grid 的数据传输设置为基线，从 L3 缓存访问的速度比 DRAM 快 10 倍，而 DL Compute Grid中的数据则快了 1000 倍。
![97.png](/assets/images/ai/97.png)
Xeon CPU和NNP-I结合起来，可以根据不同计算类型进行分层，Xeon处理器运行高精度的通用任务，而神经网络计算则卸载到NNP-I中，并在NNP-I里进一步细分，如下所示：
![98.png](/assets/images/ai/98.png)
Nervana NNP-I采用 M.2 外形或 PCI Express 卡，分别提供不同的功耗和性能，如下所示：
![99.png](/assets/images/ai/99.png)

## Habana

Habana目前一共5款芯片，其中Gaudi, Gaudi2, Gaudi3用于数据中心的深度学习训练，Goya和Greco则用于推理。Gaudi3是2024年推出产品，采用台积电5nm工艺。

### Gaudi
**Gaudi** 基于张量处理核 TPC(Tensor Processing Core) 的可扩展架构，有八个 TPC 2.0。TPC 1.0是在 Goya 推理处理器中引入的。下图显示了 Gaudi 架构框图：
![100.png](/assets/images/ai/100.png)
TPC 2.0 是 VLIW4 SIMD 处理器，支持 2048 位 SIMD 操作，每个周期TPC可以执行 64 个浮点数/INT32 操作、128 个 INT16 操作或 256 个 INT8 操作其指令集和硬件是为深度学习训练而定制的；支持GEMM 操作，张量寻址，随机数生成以及特殊函数。 TPC 支持FP32、BF16、INT32、INT16、INT8、UINT32、UINT16 和 UINT8数据类型。Gaudi 存储架构包括片上 SRAM 和每个 TPC 中的本地存储，以及四个 HBM2 器件，提供 32 GB 的容量。PCIe接口提供主机接口，支持3.0代和4.0代模式。有20 对 56Gbps PAM4 SerDes，可配置为 10 个 100Gb Ethernet、20 个 50Gb/25Gb Ethernet或两者之间的任意组合，提供了纵向扩展和横向扩展的能力；2 Tb/s 的双向带宽，支持 RDMA over Converged Ethernet （RoCE v2）。

#### 编程模型
TPC 使用称为 TPC-C 的 C 语言进行编程，TPC-C 是 C99 的扩展，增加了向量数据类型，可利用 SIMD 功能；有许多专用功能来加速 DNN 操作，例如：
- 基于张量的内存访问
- 特殊功能函数
- 随机数生成
- 类似于MME的多种数据类型

TPC 程序由两部分组成：
- TPC代码 TPC 处理器执行的 ISA，实现kernel函数
- 主机代码 主机上执行，负责程序在 TPC之间切割


TPC 处理器有四个执行槽：
- 加载槽 - 从内存加载、移动和设置值。
- SPU 槽 - 执行标量运算。
- VPU 槽 - 执行向量运算。
- 存储槽 - 到内存的存储、移动和设置值。

TPC 的流水线是通用体系结构。每条指令都有一个预定义的延迟，大部分需要四个周期的延迟。TPC 核中的所有指令都可以预测。每个 VLIW 槽都以不同的方式预测：
- SPU 和存储槽仅支持标量预测。
- VPU 和 加载槽可以由单个标量值或位数组来预测，从而屏蔽特定向量元素的。预测通过内部函数向 TPC-C 程序员公开。

TPC 处理器有四个内存空间，：
- **标量本地存储** 大小为 1 KB，允许在对齐的 4 字节块中读取/写入
- **向量本地内存** 大小为 80 KB，如果程序使用 tanh、sin 或 cos 等特殊功能，则只有 16 KB 可用。允许以对齐的 128/256 字节块读取/写入此内存
- **全局内存** 全局内存使用名为 _tensors_ 的专用访问器进行访问
- **配置空间** TPC 配置空间包含成功执行程序所需的一组定义，例如张量描述符、程序二进制位置等

本地内存与程序执行一致，每个 TPC 处理器都有自己的本地内存实例。每个 TPC 只能访问自己的本地副本。也就是说，TPC A 无法访问 TPC B 本地内存。本地存储器可以在每个周期中读取或写入，没有带宽限制。本地内存在编译时通过定义带有 `___local___` 地址空间限定符的全局变量进行静态分配。

全局内存与程序执行不一致。这意味着程序在执行先写后读操作时必须发出原子信号量操作，以保证在读回之前读取操作结果是可见的。平均每四个周期可以从全局内存加载或写入 2,048 位向量。 `__global__`地址空间限定符将指针追加到全局内存。

#### 系统扩展
Gaudi 利用卓越的开放标准网络技术进行横向扩展。每个 Gaudi 芯片有 10 个标准 100Gbit Ethernet端口（或 20 个 50 GbE/25 GbE 端口）。将网络直接集成到 AI 处理器芯片中，可以创建一个没有带宽瓶颈的灵活系统。通过将多个 Gaudi 芯片与以太网交换相结合，可以在 8、16、32、64、128、1K、2K、8K 和更多 Gaudi 芯片上分配训练。由于 Gaudi 使用现成的以太网，因此可以使用许多不同的系统和网络配置。

HLS-1 包含 8 个 HL-205 OCP 加速模块 （OAM） 夹层卡和双 PCIe 交换机。Gaudi 芯片在主板上全连接，每个 Gaudi 使用7个 100GbE 端口。
![101.png](/assets/images/ai/101.png)
HLS-1H 包含四个 HL-205 OCP 加速器模块 （OAM） 夹层卡，专为大规模横向扩展而构建。大规模横向扩展支持使用现成的外部标准以太网交换机在任何规模的 Gaudi 处理器集群中训练大型模型。下图显示了包含四个 Gaudi HL-205 卡及其接口的 HLS-1H 系统。接口是 2x16 PCIe Gen4 ，可连接到外部主机服务器，以及 40x 100Gb 以太网链路（使用 10 个 QSFP-DD 连接器）。外部以太网链路可以连接到任何交换层次结构。
![102.png](/assets/images/ai/102.png)
下图是 Gaudi 系统的另一个示例，每个 Gaudi 的 10 个 100G 端口连接到 128x100G 以太网交换机，这种情况下，以太网交换机将连接到具有 10x100G 端口的聚合结构。
![103.png](/assets/images/ai/103.png)
下图显示一种机架配置，其中六个 Gaudi HLS-1，一共 48 个 Gaudi 。每个 Gaudi HLS-1通过 8 个 2x100GbE连接到单个以太网交换机，该交换机可以进一步连接到其他机架进行横向扩展。
![104.png](/assets/images/ai/104.png)

下图是基于 HLS-1H 的 POD 示例图，由 32 个 HLS-1H 系统组成，每个HLS-1H系统 4 个卡，一共 128 个 HL-205 Gaudi卡，用于在数据中心进行大规模横向扩展。每张卡有 10 个 100Gb/s 链路，每个系统有 10 个 400Gb/s 的 QSFP-DD 端口。10 个标准以太网交换机 （32 个 QSFP-DD） 可连接32 个 HLS-1H 系统， 128 个 Gaudi 处理器组成全连接非阻塞的 CLOS 网络拓扑。
![105.png](/assets/images/ai/105.png)

下图展示了高端 2K Gaudi 系统连接拓扑，8 Gaudi组成的服务器和64 端口以太网交换机连接。每个此类交换机都连接到由 8 个 256x100GbE 交换机构建的聚合网络，形成 Clos 网络拓扑。Gaudi 芯片之间只有三个网络跃点。
![106.png](/assets/images/ai/106.png)

### Gaudi 2
**Gaudi2** 包括两个计算引擎——矩阵乘法引擎MME （Matrix Multiplication Engine） 和完全可编程的张量处理器核TPC （Tensor Processor Core） 。MME 负责执行矩阵乘法，包括全连接层、卷积等运算，而 TPC则是为深度学习运算量身定制的 VLIW SIMD 处理器，用于加速其他运算。除了 MME 和 TPC，Gaudi2 还有与转置引擎相结合的DMA ，用于高效、动态的张量形状转换，以及从 Gaudi2 内存子系统读取和写入非连续多维张量。Gaudi2 有 24 x 100 Gbps RoCE V2 RDMA NIC，提供 2.4 TB 的网络带宽，可直接路由或通过标准以太网交换实现Gaudi2之间的通信。Gaudi2 内存子系统包括 96 GB 的 HBM2E 内存，可提供 2.45 TB/秒的带宽，此外还有 48 MB 的本地 SRAM。Gaudi2针对视觉应用，集成了多媒体解码器，可以支持HEVC、H.264、VP9 和 JPEG格式。下图展示了Gaudi2的架构框图：
![107.png](/assets/images/ai/107.png)
Gaudi2 支持FP32、TF32、BF16、FP16 和 FP8（E4M3 和 E5M2）等深度学习常用数据类型；在 MME 中，累加器支持 FP32。Gaudi2 集成了 Habana 的第四代张量处理器核TPC。TPC 是一种通用 VLIW 处理器，宽度为 256B SIMD，支持 FP32、BF16、FP16，FP8（E4M3 和 E5M2），以及 INT32、INT16 和 INT8 数据类型。

#### 系统扩展

HLBA-225 包括 8 个英特尔 Gaudi2 夹层卡，这些夹层卡的 21 个 NIC在 PCB 上被动互连，并组成非阻塞的全连接网络拓扑；3 个 NIC 连接到6 个板载 QSFP-DD 连接器， 用于横向扩展；网络拓扑如下所示：
![108.png](/assets/images/ai/108.png)

### Goya
Goya是用于推理的加速器，采用异构计算架构，主要包括TPC、GEMM 和 DMA等功能模块，使用50MB共享SRAM并发工作；张量处理器核TPC是第一代，采用VLIW SIMD 向量架构，支持FP32、INT32、INT16、INT8、UINT32、UINT16、UINT8混合精度；支持DDR4和PCIe 4；Goya架构框图如下所示：
![109.png](/assets/images/ai/109.png)

### Greco
Greco是第二代推理加速器，为了提高推理速度和效率，Greco 在片上集成了多媒体解码、编码和后处理功能，支持 HEVC、H.264、JPEG 和 P-JPEG 等媒体格式。Greco 支持Bfloat16、FP16 和 INT4等数据格式。Greco 支持 16GB LPDDR5 内存，片上 SRAM 从 50 MB 增加到 128 MB。Greco 从 Goya 双插槽 PCIe 卡减少到单插槽、半高、半长 （HHHL） PCIe Gen 4 x8 接口，使客户能够将服务器中的卡数量增加一倍。

# 参考文献
1. Y. Jiao, L. Han and X. Long, "Hanguang 800 NPU – The Ultimate AI Inference Solution for Data Centers," 2020 IEEE Hot Chips 32 Symposium (HCS), Palo Alto, CA, USA, 2020, pp. 1-29, doi: 10.1109/HCS49909.2020.9220619.
2. J. Ouyang et al., "Baidu Kunlun An AI processor for diversified workloads," 2020 IEEE Hot Chips 32 Symposium (HCS), Palo Alto, CA, USA, 2020, pp. 1-18, doi: 10.1109/HCS49909.2020.9220641.
3. J. Ouyang, X. Du, Y. Ma and J. Liu, "3.3 Kunlun: A 14nm High-Performance AI Processor for Diversified Workloads," 2021 IEEE International Solid-State Circuits Conference (ISSCC), San Francisco, CA, USA, 2021, pp. 50-51, doi: 10.1109/ISSCC42613.2021.9366056.
4. Zeng, J., Kou, M., Yao, H., 2022. KunlunTVM: A Compilation Framework for Kunlun Chip Supporting Both Training and Inference, in: Proceedings of the Great Lakes Symposium on VLSI 2022. Presented at the GLSVLSI ’22: Great Lakes Symposium on VLSI 2022, ACM, Irvine CA USA, pp. 299–304. [https://doi.org/10.1145/3526241.3530316](https://doi.org/10.1145/3526241.3530316)
5. 百度技术沙龙, n.d. 昆仑芯硬件架构：新一代自研架构XPU-R_哔哩哔哩_bilibili [WWW Document]. URL [https://www.bilibili.com/video/BV1CZ4y1q7yJ/](https://www.bilibili.com/video/BV1CZ4y1q7yJ/) .
6. H. Liao, J. Tu, J. Xia and X. Zhou, "DaVinci: A Scalable Architecture for Neural Network Computing," 2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, 2019, pp. 1-44, doi: 10.1109/HOTCHIPS.2019.8875654.
7. Liao, H., Tu, J., Xia, J., Liu, H., Zhou, X., Yuan, H., Hu, Y., 2021. Ascend: a Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing : Industry Track Paper, in: 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA). Presented at the 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), IEEE, Seoul, Korea (South), pp. 789–801. [https://doi.org/10.1109/HPCA51647.2021.00071](https://doi.org/10.1109/HPCA51647.2021.00071)
8. Gwennap, L., 2020. GROQ ROCKS NEURAL NETWORKS.
9. Abts, D., Ross, J., Sparling, J., Wong-VanHaren, M., Baker, M., Hawkins, T., Bell, A., Thompson, J., Kahsai, T., Kimmell, G., Hwang, J., Leslie-Hurd, R., Bye, M., Creswick, E.R., Boyd, M., Venigalla, M., Laforge, E., Purdy, J., Kamath, P., Maheshwari, D., Beidler, M., Rosseel, G., Ahmad, O., Gagarin, G., Czekalski, R., Rane, A., Parmar, S., Werner, J., Sproch, J., Macias, A., Kurtz, B., 2020. Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads, in: 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA). Presented at the 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), IEEE, Valencia, Spain, pp. 145–158. [https://doi.org/10.1109/ISCA45697.2020.00023](https://doi.org/10.1109/ISCA45697.2020.00023)
10. Abts, D., Kimmell, G., Ling, A., Kim, J., Boyd, M., Bitar, A., Parmar, S., Ahmed, I., DiCecco, R., Han, D., Thompson, J., Bye, M., Hwang, J., Fowers, J., Lillian, P., Murthy, A., Mehtabuddin, E., Tekur, C., Sohmers, T., Kang, K., Maresh, S., Ross, J., 2022. A software-defined tensor streaming multiprocessor for large-scale machine learning, in: Proceedings of the 49th Annual International Symposium on Computer Architecture. Presented at the ISCA ’22: The 49th Annual International Symposium on Computer Architecture, ACM, New York New York, pp. 567–580. [https://doi.org/10.1145/3470496.3527405](https://doi.org/10.1145/3470496.3527405)
11. D. Abts et al., "The Groq Software-defined Scale-out Tensor Streaming Multiprocessor : From chips-to-systems architectural overview," 2022 IEEE Hot Chips 34 Symposium (HCS), Cupertino, CA, USA, 2022, pp. 1-69, doi: 10.1109/HCS55958.2022.9895630.
12. R. Prabhakar and S. Jairath, "SambaNova SN10 RDU:Accelerating Software 2.0 with Dataflow," 2021 IEEE Hot Chips 33 Symposium (HCS), Palo Alto, CA, USA, 2021, pp. 1-37, doi: 10.1109/HCS52781.2021.9567250.
13. Seiler, L., Carmean, D., Sprangle, E., Forsyth, T., Abrash, M., Dubey, P., Junkins, S., Lake, A., Sugerman, J., Cavin, R., Espasa, R., Grochowski, E., Juan, T., Hanrahan, P., 2008. Larrabee: a many-core x86 architecture for visual computing. ACM Trans. Graph. 27, 1–15. [https://doi.org/10.1145/1360612.1360617](https://doi.org/10.1145/1360612.1360617)
14. Sodani, A., Gramunt, R., Corbal, J., Kim, H.-S., Vinod, K., Chinthamani, S., Hutsell, S., Agarwal, R., Liu, Y.-C., 2016. Knights Landing: Second-Generation Intel Xeon Phi Product. IEEE Micro 36, 34–46. [https://doi.org/10.1109/MM.2016.25](https://doi.org/10.1109/MM.2016.25)
15. A. Sodani, "Knights landing (KNL): 2nd Generation Intel® Xeon Phi processor," 2015 IEEE Hot Chips 27 Symposium (HCS), Cupertino, CA, USA, 2015, pp. 1-24, doi: 10.1109/HOTCHIPS.2015.7477467.
16. Reinders, J., n.d. Knights Landing – An Overview for Developers.
17. Reinders, J., n.d. Your Path to Knights Landing.
18. A. Yang, "Deep Learning Training At Scale Spring Crest Deep Learning Accelerator (Intel® Nervana™ NNP-T)," 2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, 2019, pp. 1-20, doi: 10.1109/HOTCHIPS.2019.8875643.
19. O. Wechsler, M. Behar and B. Daga, "Spring Hill (NNP-I 1000) Intel’s Data Center Inference Chip," 2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, 2019, pp. 1-12, doi: 10.1109/HOTCHIPS.2019.8875671.
20. Intel Habana WhitePaper
21. 13. E. Medina, "[Habana Labs presentation]," 2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, 2019, pp. 1-29, doi: 10.1109/HOTCHIPS.2019.8875670.
22. M. Emani et al., "Accelerating Scientific Applications With SambaNova Reconfigurable Dataflow Architecture," in Computing in Science & Engineering, vol. 23, no. 2, pp. 114-119, 1 March-April 2021, doi: 10.1109/MCSE.2021.3057203.
23. R. Prabhakar, S. Jairath and J. L. Shin, "SambaNova SN10 RDU: A 7nm Dataflow Architecture to Accelerate Software 2.0," 2022 IEEE International Solid-State Circuits Conference (ISSCC), San Francisco, CA, USA, 2022, pp. 350-352, doi: 10.1109/ISSCC42614.2022.9731612.
24. S. Knowles, "Graphcore," 2021 IEEE Hot Chips 33 Symposium (HCS), Palo Alto, CA, USA, 2021, pp. 1-25, doi: 10.1109/HCS52781.2021.9567075.
25. J. Moe, K. Pogorelov, D. T. Schroeder and J. Langguth, "Implementing Spatio-Temporal Graph Convolutional Networks on Graphcore IPUs," 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), Lyon, France, 2022, pp. 45-54, doi: 10.1109/IPDPSW55747.2022.00016.
26. Jia, Z., Tillman, B., Maggioni, M., Scarpazza, D.P., 2019. Dissecting the Graphcore IPU Architecture via Microbenchmarking.
