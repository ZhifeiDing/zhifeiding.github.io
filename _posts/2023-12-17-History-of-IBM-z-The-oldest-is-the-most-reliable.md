---
title : History of IBM z | The oldest is the most reliable
categories : [history]
tags : [chip, cpu, architecture, history]
---
# 前言
最近几年系统性的学习并梳理了近30多年的计算技术发展历史，稍有感悟。遂决定将之整理成文，目的有二，一来作为知识沉淀，串联起不同技术，挖掘不同架构之间的渊源，二来通过整理再次审视历史，期望窥见未来发展方向。我将本系列命名为**鉴往知远**, 主要关注**计算与互联**。 本文为第三篇，主要回顾IBM z系列。
# 0. 概述
什么是**IBM z**？
> IBM Z是一系列现代化z/Architecture 硬件，可运行 z/OS、Linux、z/VSE、z/TPF、z/VM 和 IBM Z 系统软件。从System/360到System/370，System/370-XA，Enterprise Systems Architecture/370 (ESA/370), 和Enterprise Systems Architecture/390 (ESA/390)，最后演进为z/Architecture。

上面是IBM官网对IBM z的描述。下图展示了z/Architecture的60多年的演进历史：
![Pasted image 20230718165037.png](/assets/images/z/Pasted image 20230718165037.png)
![Pasted image 20230731171816.png](/assets/images/z/Pasted image 20230731171816.png)
本文通过系统性回顾IBM z系列处理器及大型机，试图通过IBM z系列处理器发展的历史脉络，来展现近30年计算架构的变迁，技术的演进，进而窥见计算技术发展的未来。IBM z系列处理器除了支持的指令集和**POWER**不同，两者在处理器微架构上基本上是互相借鉴的，因此本文不关注微架构方面内容，主要侧重z系列大型机的系统架构设计。

为何叫大型机呢？该名字来源于Mainframe, 大型主机上的处理器、通信设备、内存等子系统全部融入在一个frame（机柜）中并形成一个完整的计算机系统。下面简要介绍IBM z大型机的历史发展：
* **System/360** 1964年4月7日，IBM推出了划时代的**System/360**大型电脑，**IBM System/360**协助美国太空总署建立阿波罗11号资料库，完成太空人登陆月球计划；建立银行跨行交易系统，以及航空业界最大的在线票务系统等。**IBM System/360**的开发极为复杂，被誉为IBM 360系统之父的Frederick P.Brooks在30年前著有《The Mythical Man-Month（人月神话）》一书，这本书至今仍然是软件领域的必读经典
* **System/370** 1970年6月，IBM发布**System/370**的两个型号。**System/370**大型机与**System/360**兼容。在**System/370**体系结构中引入了虚拟存储器的概念。虚拟存储器需要一个地址转换机制去增加应用程序可用的存储空间，它使得计算机系统具有比实际配置的内存大得多的存储空间
* **System/370 XA** 1981年，IBM公布了扩展的**System/370 体系结构（370-XA)**。它将地址线位数增加到31位，增强了**System/370**的寻址能力，同时保留24位兼容方式, 原来24位地址写的应用可以无缝在**System/370-XA**上运行。**System/370-XA**还增加了扩展存储器，这种存储器与主存分离，用来保存计算机中最为常用的信息，从而显著提升性能
* **System/370 ESA** 在1988年，**System/370**体系结构得到进一步改进。IBM发布了**ESA（Enterprise System Architecture）/370**。**ESA/370**增加了访问寄存器，改进虚拟存储性能，从而可以实现应用访问虚拟空间
* **ESA/390** 在1990年9月，IBM正式推出了更高级别的大型机系列**ESA/390**, 它也是IBM最后一个31位寻址/32位数字大型机。它在IBM大型机**System/370**和64位**z/Architecture**中扮演承上启下的作用。该系列大型机都统一以**System/390(S/390)** 售卖，它也是首个采用CMOS电路的高端大型机架构
	* 1997年，S/390 G4处理器，达到了之前使用二极管设计的G3的性能
	* 1998年，推出了S/390 G5处理器
* **IBM System z** 2000年，IBM对当时现有的**System/390**进行重新命名为eServer zSeries，并以IBM独有的红色标识e来加以区分，在zSeries使用中是以z900开始。它将IBM最新设计的**z/Architecture**融入到了64位的数字世界中。**z/Architecture**是代替此前31位寻址和32位数据**ESA/390**架构的64位架构体系，它保留了对原有24位寻址和32位数据**System/360**架构的兼容
* **System z9** IBM于2005年9月16日，推出了全新以数字结尾命名的**System z9**系列大型机，z9可以从1路扩展到54路（z990为32路），最多可支持512GB的内存
* **System z10** 2008年推出，**System z10**是首个采用**z10** 4.4GHz四核处理器的大型机，支持1.5TB的内存、高速网络Infiniband的数据速率也达到6GBps
* **zEnterprise 196** 采用**z196**芯片，主频为5.2GHz，单个处理器拥有四核心，基于**z/Architecture**。z196满配24个处理器，96个处理器核
* **zEnterprise EC12**采用了**zEC12**芯片，六核5.5GHz主频，也是基于**z/Architecture**。zEC12满配36个处理器、最高120个内核，其中101个内核可直接用于运行操作系统和各种应用负载
* **z13** 8核处理器，主频5GHz，支持SMT，提供向量运算的支持(SIMD)
* **z14** 10核处理器，主频5.2GHz，14nm工艺
* **z15** 12核处理器，主频5.2GHz，支持片上压缩加速器
* **z16** 8核处理器，主频5.2GHz，支持片上AI加速器

本文的故事从**S/390 G5** 开始，一直到最新的**z16** 。下表总结了各代处理器的一些参数：
![Pasted image 20231217225358.png](/assets/images/z/Pasted image 20231217225358.png)

本文组织形式如下:
* 第一章介绍**IBM S/390 G5**处理器及其系统组成
* 第二章介绍**z900**处理器及其系统组成
* 第三章介绍**z990**处理器及其系统组成
* 第四章介绍**z10**处理器及其系统组成
* 第五章介绍**z196**处理器及其系统组成
* 第六章介绍**zEC12**处理器及其系统组成
* 第七章介绍**z13**处理器及其系统组成
* 第八章介绍**z14**处理器及其系统组成
* 第九章介绍**z15**处理器及其系统组成
* 第十章介绍**z16**处理器及其系统组成

最后列出了主要参考文献。

# 1. IBM S/390 G5
## 1.1 G5处理器
G5实现了ESA/390指令集架构，并和S/360保持兼容，G5运行在500MHz，由于ESA/390指令集架构有大量需要运行几十甚至成百上千个周期的指令，G5在实现上是单发射处理器，并使用微码(millicode)来处理复杂的指令。下图展示了G5处理器的物理规划图：
![Pasted image 20231104155049.png](/assets/images/z/Pasted image 20231104155049.png)
G5处理器主要由下面几部分组成：
* L1缓存 包括缓存数据阵列，缓存目录，TLB。L1缓存一共256KB，缓存行256B，4路组相联并且指令数据共用；可以同时处理2个请求，并且支持和L2缓存4GB/s的带宽；使用绝对地址历史表AAHT(absolute address history table)来预测缓存访问的绝对地址。TLB一共1024项，4路组相联，访问寄存器翻译ART(access register translation)包含一个8项全相联的ALB(ART lookaside buffer)。有一个32KB可写的微码阵列，实现64个常见的复杂指令。
* I-unit 主要负责取指，译码，地址生成和指令发射队列。包含一个2048项，2路组相联的BTB。
* E-unit 主要是各种执行单元以及相应的寄存器。包括一个定点和一个浮点单元，ESA/390架构有很多十进制运算的指令，执行单元可以进行二进制和十进制的运算。
* R-unit 主要保存处理器架构状态的检查点
![Pasted image 20231104162315.png](/assets/images/z/Pasted image 20231104162315.png)
为了提高可靠性，G5的I-unit和E-unit同时存在两份，每个时钟周期，两个相同单元的信号会被送到R-unit进行比较，如果出现不匹配，会触发硬件错误恢复。
1. R-unit冻结检查点状态
2. L1缓存写数据到L2缓存
3. I-unit, E-unit和L1缓存被复位
4. R-unit里检查点的值读出并且更新到对应寄存器
5. R-unit里值会被第二次读出确保没有错误，否则系统进入check-stopped状态，即时钟会被停掉
6. E-unit重新开始执行
7. millicode里会记录以便后期分析
## 1.2 SMP互联
G5可以通过包含L2缓存的系统控制器组成12路SMP系统，整个系统分成两个相同的节点，并组装在一个MCM里。系统里每个处理器可以以相同延迟访问内存。下图展示了整体系统结构：
![Pasted image 20231104163043.png](/assets/images/z/Pasted image 20231104163043.png)
上图实线是128位的数据总线，运行在250MHz，系统主要由下列部件组成：
* 一共12个处理器CP(central processor)，最多可用10个CP，另外用作IO处理器来管理IO系统以及冗余备份。
* 系统控制器由包含L2目录和配置的L2 CNTLR和最多8个，每个1MB的L2缓存芯片组成。
	* 每个系统控制器还包含两个IO接口芯片，IO子系统通过STI(self timed interface)和IO接口芯片连接；每个IO接口芯片有6组STI，每组可提供333MB/s的带宽。
* G5系统一共有4个内存卡，每个分成4个bank，一共可以提供16GB/s带宽，最大内存24GB。
* 两个加解密协处理器CE，可以支持RSA和DES算法

G5里的L1缓存是写入，并且和L2是包含关系，下图展示了G5系统的缓存结构:
![Pasted image 20231114222612.png](/assets/images/z/Pasted image 20231114222612.png)
L2缓存管理采用的是修改的MESI算法，有两个不同的共享状态，一个是本地共享(locally shared)，即数据只在本节点共享；另一个是全局共享(globally shared)，既数据被所有节点共享。
L2缓存芯片面积17.36mmx17.43mm，一共59M晶体管，包含1MB的缓存整列，下图展示了G5系统里的L2缓存芯片的物理版图：
![Pasted image 20231114223707.png](/assets/images/z/Pasted image 20231114223707.png)
系统控制芯片面积是16.45mmx16.52mm，一共8.6B晶体管，下图展示了G5系统里系统控制芯片的物理版图：
![Pasted image 20231114223722.png](/assets/images/z/Pasted image 20231114223722.png)

# 2. z900
## 2.1 z900处理器
z900是第一个实现64位z/Architecture的处理器，指令集方面主要是增加了34个ESA/390指令，139个z/Architecture指令；微架构方面主要是实现了分离的指令和数据缓存。下图展示了z900的物理规划图：
![Pasted image 20231111213402.png](/assets/images/z/Pasted image 20231111213402.png)
z900实现了7级流水，下图展示了z900的流水线：
![Pasted image 20231111213950.png](/assets/images/z/Pasted image 20231111213950.png)
* 取指 从指令缓存里获取指令，需要几个周期
* 译码 对指令寄存器里的一条指令进行译码，并读取GPR
* 操作数地址生成 通过对GPR的值进行运算生成操作数存储地址
* 数据缓存访问 数据缓存，目录以及TLB会同时访问
* 操作数返回 将数据发送到执行单元，同时和操作数缓冲里的缓存行合并
* 执行 定点指令一般一个周期，浮点一般3个周期
* 写回 将结果写回到GPR

下图展示了z900整体的数据流：
![Pasted image 20231111215538.png](/assets/images/z/Pasted image 20231111215538.png)
* 指令单元(I-unit) 指令单元在左上角，如果BTB命中，会从指令缓存进行取指，并将指令存放到指令缓冲中，随后发送到指令寄存器。随后指令会被译码并送到指令队列。
* 缓冲控制单元(Buffer Control Element) 缓冲控制单元主要是缓存及缓存控制器，在右上方。逻辑地址首先会在真实地址历史表(absolute real address history table)里进行哈希查找，并对缓存使用真实地址访问，同时会访问目录和TLB。如果命中，返回数据；每周期数据缓存可以读取2个双字；指令缓存为指令单元和压缩转换单元提供指令，每周期可提供一个4字。
* 执行单元(E-unit) 由一个定点和一个浮点执行单元组成，位于上图左下角。从数据缓存过来的操作数先保存在操作数缓冲里，并合并对齐成双字。定点指令一般1个周期，浮点一般3个周期
* 转换和压缩单元(COP) 位于上图下方中间的是转换和压缩单元，作为协处理器(coprocessor)进行数据压缩和字符串转换
* 恢复单元(R-unit) 位于上图右下角，保存处理器的状态，并使用ECC保护，当系统发生错误时，使用保存的状态来恢复正常。

# 3. z990
## 3.1 z990处理器
z990是一个双核处理器，每个处理器核里有重复的I-unit, FXU和FPU，这些重复单元的输出结果会和对应单元结果比较以确保结果正确性。当任意一个处理器核发生错误时，两个处理器核都会进入错误恢复流程。同时每个处理器核有一个协处理器，可以进行压缩和加密运算。
![Pasted image 20231112183404.png](/assets/images/z/Pasted image 20231112183404.png)
下图展示了z990处理器核的流水线：
![Pasted image 20231112183711.png](/assets/images/z/Pasted image 20231112183711.png)
* 指令首先从指令缓存进入I-unit里的指令缓冲，并且由BTB进行分支预测，然后在D1阶段有两个译码器进行译码；
* 接下来在I-unit里进入AA1，计算操作数的地址，对指令进行分组，同时对应数据缓存的C0阶段，会访问ALB(access lookaside buffer)和AAHT(absolute address history table)；
* C1阶段会访问数据缓存，TLB，PAAHT(page absolute address history table)，如果命中则返回数据；同时E-unit进入E-1阶段，指令发送到执行单元
* C2阶段数据缓存数据会发送到E-unit，如果TLB缺失，则发送请求到X-unit做虚拟地址到真实地址转换；同时E-unit进入E0阶段，定点单元进行算术和逻辑运算，可以同时最多执行三条指令；同时会判断分支预测方向是否正确，如果预测方向错误，I-unit两个周期之后开始对正确代码进行译码
* PA阶段是将结果写到两个64位的结果总线，同时更新通用寄存器；同时存储数据写入到数据缓存；结果同时送到R-unit做比较和检查点
* 后续流水线主要是完成写操作以及R-unit的检查，执行单元的结果会和冗余执行单元的结果进行比较。

下图展示了z990的主要功能单元的微架构：
![Pasted image 20231112183556.png](/assets/images/z/Pasted image 20231112183556.png)
当遇到加密指令时，会执行相应的微码，对协处理器的命令寄存器进行写来激活加密协处理器，同时将数据写到输入FIFO，同时会轮询输出FIFO以便确定数据是否准备好。下图展示了加密协处理器的功能图：
![Pasted image 20231121222706.png](/assets/images/z/Pasted image 20231121222706.png)
## 3.2 SMP系统
z990可以组成最多48路SMP系统，分成4个book，每个book最多64GB的内存，一共256GB的内存(最多512G)；每个book共享32MB的L2缓存。下图展示了z990系统的存储结构：
![Pasted image 20231121230457.png](/assets/images/z/Pasted image 20231121230457.png)
一个book是一个MCM(multichip ceramic module)封装，每个book包含
* 6个处理器芯片，即12个处理器(最多16个)
* 3个MBA(IO memory bus adapter)
* 一个SCE(system control element)，每个SCE里32MB的L2缓存作为book内的缓存一致点；一个SCE由SCC和SCD两个芯片组成；
* 2个MSC(main storage controller)芯片
* 一个时钟芯片
![Pasted image 20231123204244.png](/assets/images/z/Pasted image 20231123204244.png)
不同于之前z900/G5系统的双节点拓扑，z900采用ring拓扑，不仅可以连接更过的处理器，缓存，内存和IO，更重要的是允许book可以进行热插拔。每个book有2个ring端口，可以和最多两个book形成双ring拓扑连接，顺时针是ring 0，逆时针是ring 1。少于4个book的拓扑，对应位置book需要替换成jumper card提供连接。下图展示了不同book数量的拓扑连接：
![Pasted image 20231123204554.png](/assets/images/z/Pasted image 20231123204554.png)
一个book内处理器发出的请求会同时从两个ring广播出去，所以从请求到数据返回平均是2.67个book到book距离。
# 4. z10
## 4.1 z10处理器
z10有4个处理器核，每个处理器核有3MB的L1.5缓存，一共两个加解密和数据压缩协处理器，每两个处理器核共享一个协处理器。片上还有一个内存控制器和一个IO控制器。z10和POWER6是一起开发的，共享了定点单元，二进制和十进制浮点单元，以及内存控制器和IO控制器。z10增加了50多条z/Architecture指令。
![Pasted image 20231112203106.png](/assets/images/z/Pasted image 20231112203106.png)
### 4.1.1 z10处理器核
z10处理器核由八部分组成：
* 取指单元(IFU) 包括64KB的指令缓存，分支预测逻辑，指令缓冲和相关控制逻辑。指令缓冲提供指令给指令译码单元(IDU)
* 指令译码单元(IDU) 负责解析和译码z/Architecture 里定义的900多条指令，识别指令间依赖关系，形成指令对以便超标量执行，并指令发射到操作数访问和执行单元
* 存储加载单元(LSU) 包含128KB的数据缓存，每周期支持2个4字读取
* 翻译单元(XU) 包含一个2级TLB和硬件地址翻译逻辑
* 顶点执行单元(FXU) 负责定点算术，逻辑和分支指令的执行，大部分一个周期完成，并支持数据前馈以便依赖的指令可以背靠背执行
* 二进制浮点单元(BFU) 负责IEEE-754浮点和S/360定义的十六进制-十进制浮点操作
* 十进制浮点单元(DFU) 执行IEEE-754R兼容的十进制浮点和定点操作
* 恢复单元(RU) 保存处理器核内状态，并使用ECC保护
![Pasted image 20231112204416.png](/assets/images/z/Pasted image 20231112204416.png)
z10的处理器核的流水线主要分成五部分:
* 取指流水线 
* 译码流水线  D1，D2和D3对指令解析并译码，识别指令间依赖关系，并发送到指令队列IQ和地址生成队列AQ；G1，G2和G3对指令分组以便可以超标量执行。当指令从IQ或AQ里发射到执行单元执行，发生缓存或TLB缺失时，不会停止流水线，而是回收指令从G1开始重新执行
* 定点执行流水线 A0-A3负责提供指令操作数给定点单元执行；对于需要从数据缓存里读取多个操作数或保存数据到数据缓存的指令，A0-A2生成操作数地址，访问TLB，缓存，A3从LSU操作数接收操作数，进行对齐，格式化；P1生成条件码，P2, P3解决条件分支，并将结果写回到寄存器
* 浮点执行流水线
* 检查点恢复流水线
![Pasted image 20231112203152.png](/assets/images/z/Pasted image 20231112203152.png)
## 4.2 SMP系统
5个z10处理器和2个系统控制器组成一个book，4个book通过全连接组成一个80个处理器核的SMP系统。每个CP里的处理器核分别有64K的指令缓存，128K的数据缓存，以及3M的L1.5缓存；系统控制器芯片有24MB的L2缓存，两片一共48M的L2缓存。L1.5和L1是包含关系，L2和L1.5也是包含关系。下图展示了z9 和z10处理器缓存结构以及SMP系统互联差异：
![Pasted image 20231114221746.png](/assets/images/z/Pasted image 20231114221746.png)
一个book一共有48个DIMM，4个book组成SMP系统一共192个DIMM，当使用8GB的DIMM时，一共1.5TB的内存。每个CP有一个内存控制器，有4个通道，每个通道可以连接3个级联的DIMM，在一个book内，只有4个CP有连接DIMM。下图展示了一个book的整体连接，包括IO和内存：
![Pasted image 20231115233432.png](/assets/images/z/Pasted image 20231115233432.png)
# 5. z196

## 5.1 z196处理器
z196处理器芯片有4个5.2GHz的处理器核，一共14亿晶体管。每个处理器核有64KB的指令缓存，128KB的数据缓存，以及1.5MB的L2缓存。4个处理器核共享24MB的写回式L3缓存，由192个512Kb的eDRAM组成，分成两个slice，每个slice可以为每个处理器核提供160GB/s的带宽；L3目录过滤来自L4缓存的请求；并且L3缓存作为4个处理器核，GX IO控制器，和内存控制器的互联。z196还有两个用于数据压缩和加密的协处理器，分别由两个处理器核共享。 下图展示了z196的物理版图：
![Pasted image 20230722172630.png](/assets/images/z/Pasted image 20230722172630.png)
### 5.1.1 z196处理器核
每个z196处理器核都是超标量乱序处理器，有六个执行单元：
* 两个定点单元
* 两个存储加载单元
* 一个二进制浮点单元
* 一个十进制浮点单元

每周期可以取指，译码和分发3条指令，并且执行5条指令。下图展示了一个z196处理器核的物理布局：
![Pasted image 20230926093221.png](/assets/images/z/Pasted image 20230926093221.png)
主要有下面这些功能单元：
* **Instruction sequence unit (ISU)**: ISU实现了乱序功能，包括记录寄存器名字和指令间依赖关系，并且分发指令
* **Instruction fetch and branch (IFB)** 和 **Instruction cache & merge (ICM)**: IFB和ICM负责取指以及进行分支预测
* **译码单元Instruction decode unit (IDU)** : 译码单元从IFU缓冲里获取指令并进行译码
* **存储加载单元Load-store unit (LSU)** : LSU处理所有的操作数访问，包含数据缓存
* **转换单元Translation unit (XU)** : XU 处理逻辑地址到物理地址的转换，包含*translation look-aside buffer (TLB)* 和 *Dynamic Address Translation (DAT)* 
* **定点单元Fixed-point unit (FXU)** :  FXU处理所有定点算术运算
* **二进制浮点单元Binary floating-point unit (BFU)** : BFU处理所有二进制和十六进制的浮点和定点乘除运算
* **十进制单元Decimal unit (DU)** : DU执行十进制浮点和定点操作
* **恢复单元Recovery unit (RU)** : RU记录处理器核里所有寄存器状态，收集硬件出错信号并且管理硬件恢复

下图展示了z196处理器核的指令流：
![Pasted image 20230722173214.png](/assets/images/z/Pasted image 20230722173214.png)
下图展示了z196处理器核的流水线：
![Pasted image 20230722173937.png](/assets/images/z/Pasted image 20230722173937.png)
### 5.1.2 协处理器
z196还有两个用于数据压缩和加密的协处理器，分别由两个处理器核共享。具体而论，每个协处理器里2个压缩加速器是独立的，分别服务对应的处理器核；而加密加速器则由两个处理器核共享。每个压缩加速器有一个64KB的静态字典，分成8K项，并且有16KB的字典缓存。加密加速器用于*CP assist for cryptographic function (CPACF)*。
![Pasted image 20230926094650.png](/assets/images/z/Pasted image 20230926094650.png)

## 5.2 存储控制芯片(SC)
存储控制芯片(SC)使用同样45nm SOI工艺，面积24.4 x 19.6 mm, 有15亿晶体管。每个MCM有两个SC芯片，每个SC芯片有96MB的L4缓存。下图展示了SC芯片的物理规划图：
![Pasted image 20230926093311.png](/assets/images/z/Pasted image 20230926093311.png)
芯片主要由L4控制器和L4缓存组成，L4缓存分成4个部分，每部分有24MB的eDRAM, 分成16个bank, 24路组相联。处理器上L3缓存和SC芯片上的L4缓存之间使用6组双向数据总线通信；SC芯片使用3组双向数据总线和另外3个Book连接，并充当交换节点。L4缓存的目录对远程Book的侦听请求进行过滤。
## 5.3 缓存层次结构
z196 CPC(central processor complex)实现了4层的缓存结构，如下图所示：
![Pasted image 20230926093617.png](/assets/images/z/Pasted image 20230926093617.png)
* 每个处理器核有192 KB L1缓存, 分成128 KB的数据缓存和64KB的指令缓存，L1缓存是写入，即修改的数据会立即写回到下一级。
* L2缓存也是每个处理器核独享，一共1.5 MB，也是写入方式。
* L3缓存位于处理器芯片(PU)上，一共24MB，由4个处理器核共享，并且是写回机制
* L4缓存位于两个存储控制芯片(SC)上，每个存储控制芯片96MB，由MCM里所有处理器共享，也是写回机制
## 5.4 内存结构
z196内存子系统使用高速差分接口，下图展示了一个Book的内存拓扑结构：
![Pasted image 20230926093751.png](/assets/images/z/Pasted image 20230926093751.png)
每个Book有10到30个DIMMs(dual in-line memory modules)，DIMMs分别和MCM里的PU0, PU1和PU2上的内存控制器(MCU)相连。每个MCU有5个通道，其中一个用于RAIM(Redundant array of independent memory)；每个通道有一到二个链式连接的DIMMs, 因此，一个MCU有5到10个DIMMs；每个DIMM有4, 16或32 GB, 每个Book不能混和使用不同大小的DIMM。

z196使用了内存冗余阵列RAIM(redundant array of independent memory，RAIM可以检测DRAM，socket，内存通道或者DIMM的错误并恢复。下图展示了RAIM的示意图：
![Pasted image 20230926093932.png](/assets/images/z/Pasted image 20230926093932.png)
4个数据通道的奇偶保存在第5个通道的DIMM里，内存系统里的任何错误都可以被发现并动态修正。RAIM使z196的RAS设计到了另一个层次，做到了全容错的N+1。
## 5.5 SMP系统

6个z196处理器芯片和两个系统控制芯片一起封装成一个MCM(multichip module)。下图展示了MCM的结构：
![Pasted image 20230926092635.png](/assets/images/z/Pasted image 20230926092635.png)
### 5.5.1 Book
一个Book包含一个MCM, 内存, 以及与I/O cages和其他CPCs的连接器。Books位于frame A的processor cage。z196 CPC可以安装1到4个Book。下图展示了Book的基本设计：
![Pasted image 20230926092139.png](/assets/images/z/Pasted image 20230926092139.png)
每个Book包括：
* 一个MCM，有6个4核z196处理器芯片，以及2个存储控制芯片
* 内存DIMMs，一共30个插槽，提供60 GB到960 GB内存
* 最多8个HCA-Optical, HCA2-Copper或PCIe扇出卡，HCA2-Copper主要用于连接CPC里I/O cages或I/O drawers; PCIe用于连接PCIe I/O; HCA-Optical (HCA2-O (12xIFB), HCA2-O LR (1xIFB), HCA3-O (12xIFB)和HCA3-O LR (1xIFB))用于CPCs之间连接
* 三个DCAs(distributed converter assemblies)提供Book的供电，其中一个作为冗余
* 两个FSP(flexible service processor)卡，用于系统控制

下图展示了Book内的各部件连接关系：
![Pasted image 20230926092348.png](/assets/images/z/Pasted image 20230926092348.png)
其中，GX0到GX7是到HCAs的接口, 10 GB/s；PSIs(Processor support interfaces)用于连接FSP卡；FBC(Fabric book connectivity) 用于Book间互联。

Books之间采用全连接拓扑, 允许Book之间直接通信，如下图所示：
![Pasted image 20230926094424.png](/assets/images/z/Pasted image 20230926094424.png)
### 5.5.2 Frames
System z frames是符合Electronic Industry Association (EIA) 标准的机柜，z196 CPC有两个42U EIA frames, 分别是frames, A和Z, 包括一个processor cage和I/O cages, I/O drawers, 或PCIe I/O drawers；所有的books,包括Book内的DCAs(distributed converter assemblies)以及冷却部件，都位于Frame A的上半部分；下图展示了使用空气冷却的Frame A和Frame Z前视图：
![Pasted image 20230926091747.png](/assets/images/z/Pasted image 20230926091747.png)

Frame A 主要有下列部件：
* 两个可选的Internal Battery Features (IBFs), 提供不间断的电源。IBF进一步加强了电源的稳定性，提高了电源抗干扰能力。当4路AC电源都断电时IFB可以提供电池供电来保存处理器数据，最多可支持10分钟供电，电池成对安装，可以安装2到6个。
* 两个modular refrigeration units (MRUs), 用于制冷或者两个Water Conditioning Units (WCUs)
* Processor cage, 最多包含4个books
* 一或两个I/O drawers，每个最多8个IO卡
* I/O cage, 有28个I/O卡槽，可以安装ESCON channels, FICON Express8 channels, OSA-Express2, OSA- Express3, 和 Crypto Express3 features。最多支持2个I/O
* Air-moving devices (AMD), 为扇出卡, 内存, 和DCAs提供N+1的制冷

Frame Z主要有下列部件：
* 两个可选的Internal Battery Features (IBFs)
* Bulk Power Assemblies (BPAs)
* I/O cage 2或者一到两个I/O drawers
* Support Element (SE)托盘，在I/O cage 2前面, 包含两个SEs

### 5.5.3 I/O Drawer
每个I/O drawer支持两个I/O域(A和B)，一共8个I/O卡槽。I/O drawer内的每个I/O域使用一个IFB-MP卡和一根铜线连接到Book内的Host Channel Adapter (HCA)上。HCA和IFB-MP之间连接支持6 GBps。下图展示了CPC上的I/O的连接示意图：
![Pasted image 20230926094946.png](/assets/images/z/Pasted image 20230926094946.png)
PCIe I/O Drawer最多支持32个 I/O卡，分成4个域，每个域使用一个PCIe switch，两个PCIe switch之间通过PCIe I/O Drawer背板连接互相备份；当一个PCIe扇出卡出错, 两个域的所有16 I/O卡可以通过另一个PCIe switch。
## 5.6 系统控制
CPC系统的控制使用flexible service processors (FSPs)，FSP基于IBM Power PC处理器；FSP连接到内部的 Ethernet LAN来和support elements (SEs)通信。下图展示了FSB和SE之间连接：
![Pasted image 20230926092527.png](/assets/images/z/Pasted image 20230926092527.png)

# 6. zEC12

## 6.1 CP (Central Processor)芯片
每个处理器芯片有6个处理器核，每个处理器核有64 KB指令缓存和96 KB数据缓存，以及1 MB的L2指令缓存和1 MB的L2数据缓存。6个处理器核共享48MB的L3缓存，L3目录过滤来自L4缓存的请求；并且L3缓存作为6个处理器核，GX IO控制器，和内存控制器的互联。每个处理器核还有一个用于数据压缩和加密的协处理器，。 下图展示了zEC12的物理版图：
![Pasted image 20230718165345.png](/assets/images/z/Pasted image 20230718165345.png)

### 6.1.1 zEC12处理器核
每个zEC12处理器核都是超标量乱序处理器，有六个执行单元：
* 两个定点单元(FXU)
* 两个存储加载单元(LSU)
* 一个二进制浮点单元(BFU)
* 一个十进制浮点单元(DFU)

每周期可以取指，译码和分发3条指令，并且执行7条指令。下图展示了一个zEC12处理器核的物理布局：
![Pasted image 20230925175655.png](/assets/images/z/Pasted image 20230925175655.png)
每个处理器核主要有下列功能单元组成：
* **Instruction sequence unit (ISU)** : ISU实现了乱序功能，包括记录寄存器名字和指令间依赖关系，并且分发指令
* **Instruction fetching unit (IFU)** : 负责取指以及进行分支预测, 包含指令缓存
* **Instruction decode unit (IDU)** : 译码单元从IFU缓冲里获取指令并进行译码
* **Load-store unit (LSU)** : LSU处理所有的操作数访问，包含数据缓存
* **Translation unit (XU)** : XU 处理逻辑地址到物理地址的转换，包含*translation look-aside buffer (TLB)* 和 *Dynamic Address Translation (DAT)* 
* **定点单元Fixed-point unit (FXU)** :  FXU处理所有定点算术运算
* **二进制浮点单元Binary floating-point unit (BFU)** : BFU处理所有二进制和十六进制的浮点和定点乘除运算
* **十进制单元Decimal unit (DU)** : DU执行十进制浮点和定点操作
* **恢复单元Recovery unit (RU)** : RU记录处理器核里所有寄存器状态，收集硬件出错信号并且管理硬件恢复
* **协处理器Co-Processor (COP)** : 负责数据压缩和加密

下图展示了一个zEC12处理器核的指令流：
![Pasted image 20230926084137.png](/assets/images/z/Pasted image 20230926084137.png)
zEC12处理器使用分支历史表(BHT)，模式历史表(PHT)以及分支目标缓冲(BTB)来进行分支方向及目标的预测。
#### 6.1.1.1 协处理器
每个处理器核内有一个用于数据压缩和加密的协处理器 (CoP)， 每个压缩加速器有一个64KB的静态字典，分成8K项，并且有16KB的字典缓存。加密加速器用于*CP assist for cryptographic function (CPACF)*。下图展示了CoP的物理位置及其微架构：
![Pasted image 20230926084255.png](/assets/images/z/Pasted image 20230926084255.png)
## 6.2 SC (System Controller)芯片
系统控制(SC)芯片面积28.4 x 23.9 mm, 一共33亿晶体管，其中21亿是eDRAM。每个MCM有两个SC芯片，每个SC芯片上有192 MB的L4缓存, 因此每个Book一共384MB的L4缓存，下图展示了SC芯片的物理规划图：
![Pasted image 20230718171247.png](/assets/images/z/Pasted image 20230718171247.png)
芯片主要由L4控制器和L4缓存组成，L4缓存分成4个部分，每部分有48 MB，由256个1.5 MB eDRAM组成。 L4缓存组织成16个bank，24路组相联结构。L4缓存控制器由多个独立控制器组成，可同时处理125个传输。处理器上L3缓存和SC芯片上的L4缓存之间使用单向数据总线通信，SC芯片使用3组双向数据总线和另外3个Book连接，并充当交换节点。L4缓存的目录对远程Book的侦听请求进行过滤。

## 6.3 缓存层次结构
zEC12实现了4层的缓存结构，如下图所示：
![Pasted image 20230926082914.png](/assets/images/z/Pasted image 20230926082914.png)
* 每个处理器核有160KB的L1缓存, 分成96 KB的数据缓存和64KB的指令缓存，L1缓存是写入，即修改的数据会立即写回到下一级。
* L2缓存也是每个处理器核独享，一共2 MB，分别是1MB的数据缓存和1MB的指令缓存，也是写入方式。
* L3缓存位于处理器芯片(PU)上，一共48MB，由6个处理器核共享，12路组相联；L3分成两个逻辑slice，每个24MB，由2个12MB的bank组成，缓存行256B；并且是写回机制
* L4缓存位于两个存储控制芯片(SC)上，每个存储控制芯片192MB，由MCM里所有处理器共享，也是写回机制
## 6.4 内存结构
zEC12内存子系统使用高速差分接口，下图展示了一个Book的内存拓扑结构：
![Pasted image 20230926083213.png](/assets/images/z/Pasted image 20230926083213.png)
每个Book最多支持960 GB内存, 一部分内存用于实现RAIM(redundant array of independent memory)，因此，每个Book实际可用内存最多768 GB。每个Book有10到30个DIMMs(dual in-line memory modules)，DIMMs分别和MCM里的PU0, PU1和PU2上的内存控制器(MCU)相连。每个MCU有5个通道，其中一个用于RAIM(Redundant array of independent memory)；每个通道有一到二个链式连接的DIMMs, 因此，一个MCU有5到10个DIMMs；每个DIMM有4, 16或32 GB, 每个Book不能混和使用不同大小的DIMM。

对于一个完全容错的N+1设计, zEC12使用了内存冗余阵列RAIM(redundant array of independent memory，RAIM可以检测DRAM，socket，内存通道或者DIMM的错误并恢复。下图展示了RAIM的示意图：
![Pasted image 20230926083353.png](/assets/images/z/Pasted image 20230926083353.png)
4个数据通道的奇偶保存在第5个通道的DIMM里，内存系统里的任何错误都可以被发现并动态修正。RAIM使z196的RAS设计到了另一个层次，做到了全容错的N+1。

## 6.5 SMP系统
### 6.5.1 Book
一个Book包含一个MCM, 内存,以及与其他CPCs的连接器。Books位于frame A的processor cage。zEC12 CPC可以安装1到4个Book。下图展示了Book的基本设计：
![Pasted image 20230925170312.png](/assets/images/z/Pasted image 20230925170312.png)
每个Book包括：
* 一个MCM，有6个6核z196处理器芯片，以及2个存储控制芯片
* 内存DIMMs，一共30个插槽，提供60 GB到960 GB内存
* 最多8个 (host channel adapter (HCA或PCIe)扇出卡. HCA2-Copper主要用于连接CPC里I/O cages或I/O drawers，带宽 6 GBps; PCIe用于连接PCIe I/O drawer，带宽8 GBps, HCA-Optical用于CPCs之间连接
* 三个DCAs(distributed converter assemblies)提供Book的供电，其中一个作为冗余
* 两个FSP(flexible service processor)卡，用于系统控制

下图展示了Book内的各部件连接关系：
![Pasted image 20230925170459.png](/assets/images/z/Pasted image 20230925170459.png)
其中，GX0到GX7是到HCAs的接口, 最高10 GB/s，可以支持 InfiniBand和PCIe.；PSIs(Processor support interfaces)用于连接FSP卡；FBC(Fabric book connectivity) 用于Book间互联。

Books之间采用全连接拓扑, 允许Book之间直接通信，如下图所示：
![Pasted image 20230926083927.png](/assets/images/z/Pasted image 20230926083927.png)
### 6.5.2 Frames
System z frames是符合Electronic Industry Association (EIA) 标准的机柜，zEC12 CPC有两个42U EIA frames, 分别是frames, A和Z, 包括一个processor cage和I/O cages, I/O drawers, 或PCIe I/O drawers；所有的books,包括Book内的DCAs(distributed converter assemblies)以及冷却部件，都位于Frame A的上半部分；下图展示了使用空气冷却的Frame A和Frame Z前视图：
![Pasted image 20230925165529.png](/assets/images/z/Pasted image 20230925165529.png)
Frame A 主要有下列部件：
* 两个可选的Internal Battery Features (IBFs), 提供不间断的电源。IBF进一步加强了电源的稳定性，提高了电源抗干扰能力。当4路AC电源都断电时IFB可以提供电池供电来保存处理器数据，最多可支持10分钟供电，电池成对安装，可以安装2到6个。
* 两个全备份的radiator units (RUs), 用于制冷或者两个Water Conditioning Units (WCUs)
* Processor cage, 最多包含4个books
* 根据配置，可以组合最多两个I/O drawer或一个I/O cage
	* 最多两个PCIe I/O drawers
	* 一个I/O drawers，每个最多8个IO卡
* I/O cage, 有28个I/O卡槽，可以安装ESCON channels, FICON Express8 channels, OSA-Express2, OSA- Express3, 和 Crypto Express3 features。最多支持2个I/O
* Air-moving devices (AMD), 为扇出卡, 内存, 和DCAs提供N+1的制冷

Frame Z主要有下列部件：
* 两个可选的Internal Battery Features (IBFs)
* Bulk Power Assemblies (BPAs)， 数量根据配置确定
* 最多4个drawer，可以最多两个I/O drawers和最多4个PCIe I/O drawers:
	* 不支持I/O cage
* Support Element (SE)托盘，在I/O cage 2前面, 包含两个SEs
* 当使用WCUs制冷时，会在frame背面额外安装一个散热器installed in the rear of the frame.

### 6.5.3 I/O drawers
每个I/O drawer支持两个I/O域(A和B)，一共8个I/O卡槽。I/O drawer内的每个I/O域使用一个IFB-MP卡和一根铜线连接到Book内的Host Channel Adapter (HCA)上。HCA和IFB-MP之间连接支持6 GBps。下图展示了CPC上的I/O的连接示意图：
![Pasted image 20230926090519.png](/assets/images/z/Pasted image 20230926090519.png)
PCIe switch卡将Book内的 x16 PCIe扩展到8个独立的卡槽，PCIe switch卡连接到Book内PCIe扇出卡上一个x16 PCIe Gen 2上，PCIe扇出卡将Book内部信号转换成PCIe。PCIe I/O drawer前面和后面都有一个PCIe switch卡，两个通过背板连接，互相备份。
## 6.6 系统控制
系统的控制使用flexible service processors (FSPs)，FSP基于IBM Power PC处理器；每个FSP有两个端口，连接到内部的 Ethernet LAN来和support elements (SEs)通信，并提供了subsystem interface (SSI)接口来控制系统。下图展示了FSB和SE之间连接：
![Pasted image 20230925175019.png](/assets/images/z/Pasted image 20230925175019.png)
zEC12上的PUs物理上都是一样的，但是初始化时会选择一个PU作为IFP (integrated firmware processor)，其他PU会被定性为 CP, IFL, ICF, zAAP, zIIP, 或SAP。
* **CP(Central processors)** CP可以支持所有z/Architecture指令集，并运行基于z/Architecture的操作系统(z/OS, z/VM, TPF, z/TPF, z/VSE, and Linux), Coupling Facility Control Code (CFCC), 和IBM zAware
* **IFL(Integrated Facility for Linux)** IFL用于运行Linux, z/VM上的Linux和IBM zAware
* **ICF(Internal Coupling Facility)** ICF用于在Parallel Sysplex 环境里运行CFCC
* **zAAP(System z Application Assist Processors)** zAAP用于运行IBM指定的z/OS负载, 例如Java或z/OS XML System Services
* **zIIP(System z Integrated Information Processor)** zIIP用于z/OS负载使用SRB (service request block) 
* **SAP(System assist processors)** SAP用于运行channel subsystem LIC(Licensed Internal Code)，来控制IO操作，SAP为所有的LPAR的IO操作服务
* **IFP(Integrated firmware processor)** 在系统初始化时，会从所有可用PUs里分配一个作为IFP。IFP专门用来支持 Peripheral Component Interconnect Express (PCIe) (10GbE Remote Direct Memory Access(RDMA) over Converged Ethernet (RoCE) Express和zEnterprise Data Compression (zEDC) Express) 。IFP支持Resource Group (RG) Licensed Internal Code (LIC)，提供本地PCIe I/O管理和虚拟化功能
## 6.7 RAS

在zEC12系统中, 会保留2个PUs作为冗余，可以用来替换任意两个PU，包括CP, IFL, ICF, zAAP, zIIP, SAP, 或IFP。当系统没有可用的冗余PU时候需要替换硬件，如果一个冗余PU坏了需要替换整个MCM。

PU的错误恢复机制称为静默恢复，即当发现错误时，指令单元会重试指令并尝试恢复错误；如果第二次尝试也失败了，会将当前处理器核上状态迁移到其他处理器核上。下图展示了处理器的错误检测核恢复流程：
![Pasted image 20230926084424.png](/assets/images/z/Pasted image 20230926084424.png)

# 7. z13
## 7.1 z13处理器芯片
z13处理器有8个处理器核，一共39.9亿晶体管,运行在5GHz；每个处理器核有96KB的L1指令缓存和128KB的L1数据缓存；以及2MB的L2数据缓存和2MB的L2的指令缓存；8个处理器核共享64MB的L3缓存，分成2个slice，L3目录过滤来自L4缓存的请求并且L3缓存作为8个处理器核，GX IO控制器，和内存控制器的互联。z13每个处理器核还有一个用于数据压缩和加密的协处理器。 下图展示了z13的物理版图：
![Pasted image 20230925143126.png](/assets/images/z/Pasted image 20230925143126.png)
### 7.1.1 z13处理器核
每个z13处理器核都是超标量乱序处理器，有十个执行单元：
* 四个定点单元(FXU)
* 两个存储加载单元(LSU)
* 两个二进制浮点单元(BFU)
* 两个十进制浮点单元(DFU)
* 两个向量浮点单元(VXU)

每周期可以取指，译码和分发6条指令，并且执行10条指令。下图展示了一个z13处理器核的物理布局：
![Pasted image 20230925161046.png](/assets/images/z/Pasted image 20230925161046.png)
每个处理器核主要有下列功能单元组成：
* **Instruction sequence unit (ISU)** : ISU实现了乱序功能，包括记录寄存器名字和指令间依赖关系，并且分发指令
* **Instruction fetch and branch (IFB)** 和 **instruction cache and merge (ICM)** : 负责取指以及进行分支预测, 包含指令缓存
* **Instruction decode unit (IDU)** : 译码单元从IFU缓冲里获取指令并进行译码
* **Load-store unit (LSU)** : LSU处理所有的操作数访问，包含数据缓存
* **Translation unit (XU)** : XU 处理逻辑地址到物理地址的转换，包含*translation look-aside buffer (TLB)* 和 *Dynamic Address Translation (DAT)* 
* **向量和浮点单元**
	* **定点执行单元(FXU)** ：FXU处理所有定点算术运算
	* **二进制浮点单元Binary floating-point unit (BFU)** : BFU处理所有二进制和十六进制的浮点和定点乘除运算
	* **十进制单元Decimal unit (DU)** : DU执行十进制浮点和定点操作
* **Core pervasive unit (PC)** : 用于指令和错误的收集
* **恢复单元Recovery unit (RU)** : RU记录处理器核里所有寄存器状态，收集硬件出错信号并且管理硬件恢复
* **协处理器Co-Processor (COP)** : 负责数据压缩和加密

下图展示了一个z13处理器核的指令流：
![Pasted image 20230925164444.png](/assets/images/z/Pasted image 20230925164444.png)
z13处理器使用分支历史表(BHT)，模式历史表(PHT)以及分支目标缓冲(BTB)来进行分支方向及目标的预测。

#### 7.1.1.1 协处理器单元
每个处理器核内有一个用于数据压缩和加密的协处理器 (CoP)。其中压缩加速器器使用静态字典，并利用L1指令缓存；加密加速器用于*CP assist for cryptographic function (CPACF)*。下图展示了CoP的物理位置及其微架构：
![Pasted image 20230925164714.png](/assets/images/z/Pasted image 20230925164714.png)

## 7.2 存储控制芯片(SC)
系统控制(SC)芯片面积28.4 x 23.9 mm, 一共71亿晶体管，其中21亿是eDRAM。每个CPC节点有一个SC芯片，每个SC芯片的L4缓存分成480MB的非包含缓存和224MB的Non-data Inclusive Coherent (NIC)目录, 因此每个CPC共享960MB的L4缓存，以及448 MB NIC目录。下图展示了SC芯片的物理规划图：
![Pasted image 20230925161619.png](/assets/images/z/Pasted image 20230925161619.png)
芯片主要由L4控制器和L4缓存组成，L4缓存分成4个部分，每部分有120MB，由256个1.5 MB eDRAM组成。 L4缓存组织成16个bank，30路组相联结构。L4缓存控制器由多个独立控制器组成，可同时处理125个传输。处理器上L3缓存和SC芯片上的L4缓存之间使用单向数据总线通信，SC芯片使用3组双向数据总线和另外3个CPC连接，并充当交换节点。L4缓存的目录对远程CPC的侦听请求进行过滤。
## 7.3 缓存层次结构
z13实现了4层的缓存结构，如下图所示：
![Pasted image 20230925162031.png](/assets/images/z/Pasted image 20230925162031.png)
* 每个处理器核有224KB的L1缓存, 分成128KB的数据缓存和96KB的指令缓存，L1缓存是写入，即修改的数据会立即写回到下一级。
* L2缓存也是每个处理器核独享，一共4MB，分别是2MB的数据缓存和2MB的指令缓存，也是写入方式。
* L3缓存位于处理器芯片(PU)上，一共64MB，由8个处理器核共享，16路组相联；L3分成两个逻辑slice，每个32MB，由2个16MB的bank组成，缓存行256B；并且是写回机制
* L4缓存位于两个存储控制芯片(SC)上，每个存储控制芯片480MB，由节点里所有处理器共享，也是写回机制；两个SC芯片之间通过S-bus通信；NIC目录里记录L3缓存里非包含的缓存行，可以通过X-bus访问L3缓存里独占的缓存行
## 7.4 内存结构
z13内存子系统使用高速差分接口，下图展示了一个CPC drawer的内存拓扑结构：  
![Pasted image 20230925162336.png](/assets/images/z/Pasted image 20230925162336.png)
每个CPC drawer有20到25个DIMMs(dual in-line memory modules)，DIMMs分别和PU上的内存控制器(MCU)相连。每个MCU有5个通道，其中一个用于RAIM(Redundant array of independent memory)；每个DIMM有16，32 GB, 64 GB, 或128 GB。 CPC drawer可以混用不同大小DIMM，但是同一个MCU不能混和使用不同大小的DIMM。每个CPC drawer使用4或5个MCUs；

z13使用了内存冗余阵列RAIM(redundant array of independent memory，RAIM可以检测DRAM，socket，内存通道或者DIMM的错误并恢复。下图展示了RAIM的示意图：
![Pasted image 20230925162738.png](/assets/images/z/Pasted image 20230925162738.png)
4个数据通道的奇偶保存在第5个通道的DIMM里，内存系统里的任何错误都可以被发现并动态修正。
## 7.5 SMP系统

### 7.5.1 CPC drawer
z13 CPC不同于之前的Book，而是用Drawer来组装处理器。一个13 CPC drawer有8个single chip modules (SCMs), 内存, SMP连接, 以及支持PCIe I/O drawers, I/O drawers的连接卡槽。CPC drawers放置在A frame。z13服务器最多可以安装4个CPC。下图展示了CPC Drawer的基本设计：
![Pasted image 20230925141535.png](/assets/images/z/Pasted image 20230925141535.png)
CPC drawer分成两个节点，每个节点包括：
* 三个8核处理器芯片(PU) SCMs
* 一个存储控制芯片SC SCM, 有480 MB L4缓存
* 一共10或15个DDR3 dual inline memory module (DIMM)卡槽

因此，一个CPC drawer包含：
* 六个8核处理器芯片PU SCMs
* 两个存储控制芯片SC SCM, 一共960 MB L4缓存
* 20或25个DIMM槽, 可以提供320 - 3,200 GB物理内存和256 - 2,560 GB可访问内存
* 十个PCIe Generation 3 (PCIe Gen3)插槽，用于PCIe I/O drawer或PCIe coupling links
* 四个GX++插槽，用于IFB或InfiniBand coupling
* 两个flexible service processor (FSP)，用于系统控制
* 两个DC converter assemblies (DCAs)，为CPC drawer提供电源，互为备份
* Water-cooling manifold，用于处理器芯片PU散热

下图展示了CPC Drawer内PU和SC之间的连接：
![Pasted image 20230925141731.png](/assets/images/z/Pasted image 20230925141731.png)

不同的总线作用如下:
* GX++ I/O总线，提供和host channel adapters (HCAs)的连接；每方向带宽6 GBps；GXX++ I/O提供对InfiniBand和非PCIe I/O(FICON Express 8)的支持
* PCIe I/O总线，带宽16 GBps
* X-bus用于同一个节点内PU与PU，PU和SC之间的连接
* S-bus用于同一个CPC Drawer内两个SC芯片之间的互联 
* A-bus用于不同Drawer的SC芯片之间的互联
* Processor support interfaces (PSIs)用于和FSP连接
### 7.5.2 Frames
System z frames是符合Electronic Industry Association (EIA) 标准的机柜，z13有两个42U EIA frames, 分别是frames, A和Z, 包括最多4个CPC drawer, 以及I/O drawers, 或PCIe I/O drawers；下图展示了使用空气冷却的Frame A和Frame Z前视图：
![Pasted image 20230925141104.png](/assets/images/z/Pasted image 20230925141104.png)
4个CPC Drawer通过两个Node分别形成全连接，如下图所示：
![Pasted image 20230925142441.png](/assets/images/z/Pasted image 20230925142441.png)

Frame A 从顶到底主要有下列部件：
* 两个Support Element (SE)服器，安装在Frame A的顶部，在之前的system z服务器，SE在Frame Z里，并且是笔记本电脑；而在z13服务器里，SE是一个1U的服务器
* 两个可选的Internal Battery Features (IBFs), 提供不间断的电源。IBF进一步加强了电源的稳定性，提高了电源抗干扰能力
* 一个PCIe I/O drawer
* 两个System Control Hubs (SCHs)，SCHs用于替换之间的Bulk Power Hubs
* 最多4个CPC drawers
* 散热单元
	* 对于风冷，三台泵和三台鼓风机(N+2冗余设计)
	* 对于水冷，两个Water Conditioning Units (WCUs)

Frame Z从顶到底主要有下列部件：
* 两个或四个可选的Internal Battery Features (IBFs)
* Bulk power regulators (BPRs)
* 键盘和显示器托盘，和SE连接
* 最多四个PCIe I/O drawers

###  7.5.3 I/O drawer
下图展示了CPC上的I/O的连接示意图：
![Pasted image 20230925165152.png](/assets/images/z/Pasted image 20230925165152.png)
PCIe I/O Drawer最多支持32个 I/O卡，分成4个域，每个域使用一个PCIe switch，两个PCIe switch之间通过PCIe I/O Drawer背板连接互相备份；当一个PCIe扇出卡出错, 两个域的所有16 I/O卡可以通过另一个PCIe switch。

## 7.6 系统控制
系统的控制使用flexible service processors (FSPs)，FSP基于IBM Power PC处理器；每个FSP有两个端口，连接到内部的Ethernet LAN，和support elements (SEs)通信，并提供了subsystem interface (SSI)接口来控制系统。下图展示了FSB和SE之间连接：
![Pasted image 20230925142813.png](/assets/images/z/Pasted image 20230925142813.png)
z13上的PUs物理上都是一样的，但是初始化时会选择一个PU作为IFP (integrated firmware processor)，其他PU会被定性为 CP, IFL, ICF, zAAP, zIIP, 或SAP。Licensed Internal Code (LIC)会根据客户订单来定义不同PU的功能，未被定性的PU最为冗余，LIC在系统上电之后加载。
* **CP(Central processors)** CP可以支持所有z/Architecture指令集，并运行基于z/Architecture的操作系统(z/OS, z/VM, TPF, z/TPF, z/VSE, and Linux), Coupling Facility Control Code (CFCC), 和IBM zAware
* **IFL(Integrated Facility for Linux)** IFL用于运行Linux, z/VM上的Linux和IBM zAware
* **ICF(Internal Coupling Facility)** ICF用于在Parallel Sysplex 环境里运行CFCC
* **zAAP(System z Application Assist Processors)** zAAP用于运行IBM指定的z/OS负载, 例如Java或z/OS XML System Services
* **zIIP(System z Integrated Information Processor)** zIIP用于z/OS负载使用SRB (service request block) 
* **SAP(System assist processors)** SAP用于运行channel subsystem LIC(Licensed Internal Code)，来控制IO操作，SAP为所有的LPAR的IO操作服务
* **IFP(Integrated firmware processor)** 在系统初始化时，会从所有可用PUs里分配一个作为IFP。IFP专门用来支持 Peripheral Component Interconnect Express (PCIe) (10GbE Remote Direct Memory Access(RDMA) over Converged Ethernet (RoCE) Express和zEnterprise Data Compression (zEDC) Express) 。IFP支持Resource Group (RG) Licensed Internal Code (LIC)，提供本地PCIe I/O管理和虚拟化功能
## 7.7 RAS
PU的错误恢复机制称为静默恢复，即当发现错误时，指令单元会重试指令并尝试恢复错误；如果第二次尝试也失败了，会将当前处理器核上状态迁移到其他处理器核上。下图展示了处理器的错误检测核恢复流程：
![Pasted image 20230925164839.png](/assets/images/z/Pasted image 20230925164839.png)
# 8. z14
## 8.1 z14处理器芯片
z14处理器有10个处理器核，一共61亿晶体管，运行在5.2 GHz； 每个处理器核有128KB的L1指令缓存和128KB的L1数据缓存；以及4MB的L2数据缓存和2MB的L2的指令缓存；10个处理器核共享128MB的L3缓存，分成2个slice，L3目录过滤来自L4缓存的请求并且L3缓存作为10个处理器核，GX IO控制器，和内存控制器的互联。z14每个处理器核还有一个用于数据压缩和加密的协处理器。 下图展示了z14的物理版图：
![Pasted image 20230925112707.png](/assets/images/z/Pasted image 20230925112707.png)

### 8.1.1 z14处理器核
每个z14处理器是超标量乱序处理器，有十个执行单元：
* 四个定点单元(FXU)
* 两个存储加载单元(LSU)
* 两个二进制浮点单元(BFU)
* 两个十进制浮点单元(DFU)
* 两个向量浮点单元(VXU)
![Pasted image 20230925112944.png](/assets/images/z/Pasted image 20230925112944.png)
每个处理器核主要有下列功能单元组成：
* **Instruction sequence unit (ISU)** : ISU实现了乱序功能，包括记录寄存器名字和指令间依赖关系，并且分发指令
* **Instruction fetch and branch (IFB)** 和 **instruction cache and merge (ICM)** : 负责取指以及进行分支预测, 包含指令缓存
* **Instruction decode unit (IDU)** : 译码单元从IFU缓冲里获取指令并进行译码
* **Load-store unit (LSU)** : LSU处理所有的操作数访问，包含数据缓存
* **Translation unit (XU)** : XU 处理逻辑地址到物理地址的转换，包含*translation look-aside buffer (TLB)* 和 *Dynamic Address Translation (DAT)* 
* **向量和浮点单元**
	* **定点执行单元(FXU)** ：FXU处理所有定点算术运算
	* **二进制浮点单元Binary floating-point unit (BFU)** : BFU处理所有二进制和十六进制的浮点和定点乘除运算
	* **十进制单元Decimal unit (DU)** : DU执行十进制浮点和定点操作
* **Core pervasive unit (PC)** : 用于指令和错误的收集
* **恢复单元Recovery unit (RU)** : RU记录处理器核里所有寄存器状态，收集硬件出错信号并且管理硬件恢复
* **协处理器Co-Processor (COP)** : 负责数据压缩和加密

下图展示了一个z14处理器核的指令流：
![Pasted image 20230925135714.png](/assets/images/z/Pasted image 20230925135714.png)
z14处理器使用分支历史表(BHT)，模式历史表(PHT)以及分支目标缓冲(BTB)来进行分支方向及目标的预测。

下图展示了z14处理器的流水线：
![Pasted image 20230920165853.png](/assets/images/z/Pasted image 20230920165853.png)

#### 8.1.1.1 协处理器单元
每个处理器核内有一个用于数据压缩和加密的协处理器 (CoP)。其中压缩加速器器使用静态字典，并利用L1指令缓存；加密加速器用于*CP assist for cryptographic function (CPACF)*。下图展示了CoP的物理位置及其微架构：
![Pasted image 20230925140019.png](/assets/images/z/Pasted image 20230925140019.png)

## 8.2 存储控制芯片(SC)
每个SC芯片的L4缓存分成672MB的缓存, 因此每个CPC共享672MB的L4缓存；用于CP和SC互联的X-Bus以及Drawer之间互联的A-bus。下图展示了SC芯片的物理规划图：
![Pasted image 20230925113213.png](/assets/images/z/Pasted image 20230925113213.png)
## 8.3 缓存层次结构
z14实现了4层的缓存结构，如下图所示：
![Pasted image 20230925134940.png](/assets/images/z/Pasted image 20230925134940.png)
* 每个处理器核有256KB的L1缓存, 分成128KB的数据缓存和128KB的指令缓存，L1缓存是写入，即修改的数据会立即写回到下一级。
* L2缓存也是每个处理器核独享，一共6MB，分别是4MB的数据缓存和2MB的指令缓存，也是写入方式。
* L3缓存位于处理器芯片(PU)上，一共128MB，由10个处理器核共享，32路组相联；L3分成两个逻辑slice，每个64MB，由2个32MB的bank组成，缓存行256B；并且是写回机制
* L4缓存位于存储控制芯片(SC)上，672MB，42路组相联；由CPC Drawer里所有处理器共享，也是写回机制；NIC目录里记录L3缓存里非包含的缓存行，可以通过X-bus访问L3缓存里独占的缓存行

L4是CPC Drawer的一致性点，即所有内存访问在发送到处理器之间都需要送到L4缓存。

## 8.4 内存结构
z14内存子系统使用高速差分接口，下图展示了一个CPC drawer的内存拓扑结构：  
![Pasted image 20230925113717.png](/assets/images/z/Pasted image 20230925113717.png)
每一个CPC Drawer支持3，4或5个MCUs，DIMMs分别和PU上的内存控制器(MCU)相连。每个处理器芯片PU SCM有一个MCU, 支持5个DIMM通道，其中一个用于RAIM(Redundant array of independent memory)；每个DIMM有32 GB, 64 GB, 128 GB，256, 或512 GB；同一个MCU不能混和使用不同大小的DIMM。

z14使用了内存冗余阵列RAIM(redundant array of independent memory，RAIM可以检测DRAM，socket，内存通道或者DIMM的错误并恢复。下图展示了RAIM的示意图：
![Pasted image 20230925114025.png](/assets/images/z/Pasted image 20230925114025.png)
4个数据通道的奇偶保存在第5个通道的DIMM里，内存系统里的任何错误都可以被发现并动态修正。
## 8.5 SMP系统
### 8.5.1 CPC drawer
z14服务器每个CPC Drawer支持1 - 4个 processor drawers，每一个processor drawer包含5或6个Central Processor (CP)和一个Storage Controller (SC)；另外，CPC drawers还包括DIMMs, I/O, Flexible Service Processors (FSPs), 和水冷管。下图展示了一个CPC drawer：
![Pasted image 20230925111432.png](/assets/images/z/Pasted image 20230925111432.png)
每个CPC Drawer最多支持8TB的内存，所以z14服务器支持256 GB - 32 TB内存。每个CPC drawer由两个逻辑CP clusters 和一个SC SCM组成，下图展示了CPC Drawer内部连接：
![Pasted image 20230925111625.png](/assets/images/z/Pasted image 20230925111625.png)
CP Cluster内CP之间使用X-Bus组成全连接拓扑，同时每个CP使用X-Bus和SC芯片连接。
### 8.5.2 Frames
z14有两个42U EIA frames, 分别是frames, A和Z, 包括最多4个CPC drawer, 以及I/O drawers, 或PCIe I/O drawers；下图展示了使用空气冷却的Frame A和Frame Z前视图：
![Pasted image 20230925111015.png](/assets/images/z/Pasted image 20230925111015.png)
4个CPC Drawer通过各自SC芯片形成全连接，下图展示了4个CPC Drawer之间的连接：
![Pasted image 20230925111914.png](/assets/images/z/Pasted image 20230925111914.png)

Frame A 从顶到底主要有下列部件：
* 两个Support Element (SE)服器，安装在Frame A的顶部，在之前的system z服务器，SE在Frame Z里，并且是笔记本电脑；而在z13服务器里，SE是一个1U的服务器
* 两个可选的Internal Battery Features (IBFs), 提供不间断的电源。IBF进一步加强了电源的稳定性，提高了电源抗干扰能力
* 一个PCIe I/O drawer
* 两个System Control Hubs (SCHs)，SCHs用于替换之间的Bulk Power Hubs
* 最多4个CPC drawers
* 散热单元
	* 对于风冷，三台泵和三台鼓风机(N+2冗余设计)
	* 对于水冷，两个Water Conditioning Units (WCUs)

Frame Z从顶到底主要有下列部件：
* 两个或四个可选的Internal Battery Features (IBFs)
* Bulk power regulators (BPRs)
* 键盘和显示器托盘，和SE连接
* 最多四个PCIe I/O drawers

### 8.5.3 I/O drawer
下图展示了CPC上的I/O的连接示意图：
![Pasted image 20230925140638.png](/assets/images/z/Pasted image 20230925140638.png)
PCIe I/O Drawer最多支持32个 I/O卡，分成4个域，每个域使用一个PCIe switch，两个PCIe switch之间通过PCIe I/O Drawer背板连接互相备份；当一个PCIe扇出卡出错, 两个域的所有16 I/O卡可以通过另一个PCIe switch。
## 8.6 系统控制
系统的控制使用flexible service processors (FSPs)，FSP基于IBM Power PC处理器；每个FSP有两个端口，连接到内部的Ethernet LAN，和support elements (SEs)通信，并提供了subsystem interface (SSI)接口来控制系统。下图展示了FSB和SE之间连接：
![Pasted image 20230925112101.png](/assets/images/z/Pasted image 20230925112101.png)
## 8.7 RAS
PU的错误恢复机制称为静默恢复，即当发现错误时，指令单元会重试指令并尝试恢复错误；如果第二次尝试也失败了，会将当前处理器核上状态迁移到其他处理器核上。下图展示了处理器的错误检测核恢复流程：
![Pasted image 20230925140234.png](/assets/images/z/Pasted image 20230925140234.png)
# 9. z15
## 9.1 z15处理器芯片
z15处理器有12个处理器核， 一共92亿晶体管，运行在5.2GHz；每个处理器核有128KB的L1指令缓存和128KB的L1数据缓存；以及4MB的L2数据缓存和4MB的L2的指令缓存；10个处理器核共享256MB的L3缓存；有一个片上压缩加速器NXU。下图展示了z15的物理版图：
![Pasted image 20230923181714.png](/assets/images/z/Pasted image 20230923181714.png)
### 9.1.1 z15处理器核芯片
z15处理器是超标量乱序处理器，有十二个执行单元：
* 四个定点单元(FXU)
* 两个存储加载单元(LSU)
* 两个二进制浮点单元(BFU)
* 两个十进制浮点单元(DFU)
* 两个向量浮点单元(VXU)

每周期可以译码6条指令，发射10条指令，最多执行12条指令。下图展示了z15处理器核的物理版图：
![Pasted image 20230923181917.png](/assets/images/z/Pasted image 20230923181917.png)
每个处理器核主要有下列功能单元组成：
* **Instruction sequence unit (ISU)** : ISU实现了乱序功能，包括记录寄存器名字和指令间依赖关系，并且分发指令
* **Instruction fetch and branch (IFB)** 和 **instruction cache and merge (ICM)** : 负责取指以及进行分支预测, 包含指令缓存
* **Instruction decode unit (IDU)** : 译码单元从IFU缓冲里获取指令并进行译码
* **Load-store unit (LSU)** : LSU处理所有的操作数访问，包含数据缓存
* **Translation unit (XU)** : XU 处理逻辑地址到物理地址的转换，包含*translation look-aside buffer (TLB)* 和 *Dynamic Address Translation (DAT)* 
* **Core pervasive unit (PC)** : 用于指令和错误的收集
* **恢复单元Recovery unit (RU)** : RU记录处理器核里所有寄存器状态，收集硬件出错信号并且管理硬件恢复
* **协处理器Co-Processor (COP)** : 负责数据压缩和加密
* Modulo arithmetic (MA) unit: 支持椭圆加密算法
* **向量和浮点单元**
	* **定点执行单元(FXU)** ：FXU处理所有定点算术运算
	* **二进制浮点单元Binary floating-point unit (BFU)** : BFU处理所有二进制和十六进制的浮点和定点乘除运算
	* **十进制单元Decimal unit (DU)** : DU执行十进制浮点和定点操作
	* 向量单元Vector Unit : 负责向量运算

下图展示了一个z15处理器核的指令流：
![Pasted image 20230925094835.png](/assets/images/z/Pasted image 20230925094835.png)
z15处理器使用分支历史表(BHT)，模式历史表(PHT)以及分支目标缓冲(BTB)来进行分支方向及目标的预测。

下图展示了z15处理器的流水线：
![Pasted image 20231212223725.png](/assets/images/z/Pasted image 20231212223725.png)
#### 9.1.1.1 协处理器单元
每个处理器核内有一个用于数据压缩，加密和排序的协处理器 (CoP)。其中压缩加速器器使用静态字典，并利用L1指令缓存；加密加速器用于*CP assist for cryptographic function (CPACF)*。下图展示了CoP的物理位置及其微架构：
![Pasted image 20230925101023.png](/assets/images/z/Pasted image 20230925101023.png)


### 9.1.2 片上压缩加速器
z15有一个片上压缩加速器 (Nest Accelerator Unit - NXU)，由处理器芯片上12个处理器核共享，支持DEFLATE兼容的压缩和解压缩；并提供同步和异步两种执行模式：
* 同步执行  用户应用程序在problem state执行
* 异步执行  为z/OS下授权的应用而优化

下图展示了片上压缩加速器位置：
![Pasted image 20231212223821.png](/assets/images/z/Pasted image 20231212223821.png)
## 9.2 系统控制芯片(SC)
系统控制芯片(SC)采用14nm SOI工艺，面积25.3 x 27.5 mm, 一共有122亿晶体管，每个CPC drawer有一个SC芯片。每个SC芯片包含960MB的L4缓存，以及用于Drawer之间连接的A-Bus。下图展示了SC芯片的物理规划图：
![Pasted image 20230923182508.png](/assets/images/z/Pasted image 20230923182508.png)
## 9.3 缓存层次结构
z15实现了4层的缓存结构，如下图所示：
![Pasted image 20230925084313.png](/assets/images/z/Pasted image 20230925084313.png)
* 每个处理器核有256KB的L1缓存, 分成128KB的数据缓存和128KB的指令缓存，L1缓存是写入，即修改的数据会立即写回到下一级。
* L2缓存也是每个处理器核独享，一共8MB，分别是4MB的数据缓存和4MB的指令缓存，也是写入方式。
* L3缓存位于处理器芯片(PU)上，一共256MB，由12个处理器核共享，32路组相联；L3分成两个逻辑slice，每个128MB，由2个64MB的bank组成，缓存行256B；并且是写回机制
* L4缓存位于存储控制芯片(SC)上，960MB，60路组相联；由CPC Drawer里所有处理器共享，也是写回机制

L4是CPC Drawer的一致性点，即所有内存访问在发送到处理器之间都需要送到L4缓存。
## 9.4 内存结构
z15内存子系统使用高速差分接口，下图展示了一个CPC drawer的内存拓扑结构： 
![Pasted image 20230923183029.png](/assets/images/z/Pasted image 20230923183029.png)
每一个CPC Drawer支持3或4个MCUs，DIMMs分别和PU上的内存控制器(MCU)相连。每个处理器芯片PU SCM有一个MCU, 支持5个DIMM通道，其中一个用于RAIM(Redundant array of independent memory)；每个DIMM有32 GB, 64 GB, 128 GB，256, 或512 GB；同一个MCU不能混和使用不同大小的DIMM。一个CPC Drawer最多支持8TB内存，5个CPC Drawer组成的服务器最多支持40TB的内存。

z15使用了内存冗余阵列RAIM(redundant array of independent memory，RAIM可以检测DRAM，socket，内存通道或者DIMM的错误并恢复。下图展示了RAIM的示意图：
![Pasted image 20231216170429.png](/assets/images/z/Pasted image 20231216170429.png)
4个数据通道的奇偶保存在第5个通道的DIMM里，内存系统里的任何错误都可以被发现并动态修正。
## 9.5 SMP系统
### 9.5.1 CPC drawer
z15 CPC drawer主要包括：
* 一个系统控制芯片(SC)
* 四个处理器芯片(CP)
* 最多20条内存DIMMs
* PCIe I/O Drawer连接插槽

下图展示了一个CPC drawer：
![Pasted image 20230923175853.png](/assets/images/z/Pasted image 20230923175853.png)

每个CPC drawer由两个逻辑CP clusters 和一个SC SCM组成，下图展示了CPC Drawer内部连接：
![Pasted image 20230923180454.png](/assets/images/z/Pasted image 20230923180454.png)
CP Cluster内CP之间使用X-Bus连接，同时每个CP使用X-Bus和SC芯片连接。
### 9.5.2 Frames
z15服务器主要组成部件包括：
* 1-4个19-inch 42u frame
* 1-5个CPC (Processor) drawers 
* 最多12个PCIe+ Gen3 I/O drawers 
* CPC drawer Cooling Units : Radiator cooling assembly (RCA) 或Water Cooling Unit (WCU)
* 每个Frame 2-4个Intelligent Power Distribution Units (iPDU)或1-6对Bulk Power Regulators

下图展示了一个全配置的z15服务器前视图：
![Pasted image 20230923174844.png](/assets/images/z/Pasted image 20230923174844.png)
5个CPC Drawer通过各自SC芯片形成全连接，下图展示了5个CPC Drawer之间的连接：
![Pasted image 20230925093551.png](/assets/images/z/Pasted image 20230925093551.png)

### 9.5.3 I/O drawer
z15支持4个GbE交换机, 5个CPC drawers, 和最多12个PCIe I/O drawers。下图展示了CPC上的I/O的连接示意图：
![Pasted image 20230923183738.png](/assets/images/z/Pasted image 20230923183738.png)
## 9.6 系统控制
系统的控制使用flexible service processors (FSPs)，FSP基于IBM Power PC处理器；每个FSP有两个端口，连接到内部的Ethernet LAN，和support elements (SEs)通信，并提供了subsystem interface (SSI)接口来控制系统。下图展示了FSB和SE之间连接：
![Pasted image 20230923181459.png](/assets/images/z/Pasted image 20230923181459.png)
## 9.7 RAS
PU的错误恢复机制称为静默恢复，即当发现错误时，指令单元会重试指令并尝试恢复错误；如果第二次尝试也失败了，会将当前处理器核上状态迁移到其他处理器核上。下图展示了处理器的错误检测核恢复流程：
![Pasted image 20230925101311.png](/assets/images/z/Pasted image 20230925101311.png)

## 9.8 散热冷却单元
下图展示了z15上的水冷散热单元：
![Pasted image 20230923184247.png](/assets/images/z/Pasted image 20230923184247.png)
经过CPC Drawer之后的热水由冷却泵抽到冷却区冷却之后，在重循环到CPC Drawer。
# 10. z16
## 10.1 z16处理器芯片
zz16处理器有8个处理器核，一共225亿晶体管，运行在5.2GHz；每个处理器核有128KB的L1指令缓存和128KB的L1数据缓存；以及32MB的L2缓存。有一个片上压缩加速器NXU以及一个人工智能加速器Artificial Intelligence Unit (AIU)。下图展示了z16的物理版图：
![Pasted image 20230921172719.png](/assets/images/z/Pasted image 20230921172719.png)
8个处理器核，压缩加速器NXU和人工智能加速器(AIU)使用一个双向环总线通信，人工智能加速器使用Neural Networks Processing Assist (NNPA)指令进行编程。下图展示了z16整体架构：
![Pasted image 20231216225654.png](/assets/images/z/Pasted image 20231216225654.png)
### 10.1.1 z16处理器核芯片
z16处理器是超标量处理器，每周期可以译码6条指令，并执行10条指令。下图展示了z16处理器核的物理版图：
![Pasted image 20230921173038.png](/assets/images/z/Pasted image 20230921173038.png)
每个处理器核主要有下列功能单元组成：
* **Instruction sequence unit (ISU)** : ISU实现了乱序功能，包括记录寄存器名字和指令间依赖关系，并且分发指令
* **Instruction fetch and branch (IFB)** 和 **instruction cache and merge (ICM)** : 负责取指以及进行分支预测, 包含指令缓存
* **Instruction decode unit (IDU)** : 译码单元从IFU缓冲里获取指令并进行译码
* **Load-store unit (LSU)** : LSU处理所有的操作数访问，包含数据缓存
* **Translation unit (XU)** : XU 处理逻辑地址到物理地址的转换，包含*translation look-aside buffer (TLB)* 和 *Dynamic Address Translation (DAT)* 
* **Core pervasive unit (PC)** : 用于指令和错误的收集
* **恢复单元Recovery unit (RU)** : RU记录处理器核里所有寄存器状态，收集硬件出错信号并且管理硬件恢复
* **协处理器Co-Processor (COP)** : 负责数据压缩和加密
* **Modulo arithmetic (MA) unit**: 支持椭圆加密算法
* **向量和浮点单元**
	* **定点执行单元(FXU)** ：FXU处理所有定点算术运算
	* **二进制浮点单元Binary floating-point unit (BFU)** : BFU处理所有二进制和十六进制的浮点和定点乘除运算
	* **十进制单元Decimal unit (DU)** : DU执行十进制浮点和定点操作
	* 向量单元Vector Unit : 负责向量运算

下图展示了一个z16处理器核的指令流：
![Pasted image 20230922153810.png](/assets/images/z/Pasted image 20230922153810.png)
### 10.1.2 AI加速器
片上AI加速器主要包括数据搬运单元(data mover)，以及Processor Tiles(PT)，Processor Elements(PE)和Special Function Processors(SFP)组成的计算整列。下图展示了AI捷速器的架构框图：
![Pasted image 20230923172318.png](/assets/images/z/Pasted image 20230923172318.png)
* **数据搬运单元** 和环形总线相连，提供200GB/s的带宽; 和计算整列接口可提供600GB/s的带宽
* **计算整列** 由128个可以计算8路FP16的FMA的Processor Tile以及32个可以计算8路FP16/FP32 SIMD组成

## 10.2 缓存层次结构
z16实现了4层的缓存结构，如下图所示：
![Pasted image 20230921100943.png](/assets/images/z/Pasted image 20230921100943.png)
* 每个处理器核有256KB的L1缓存, 分成128KB的数据缓存和128KB的指令缓存，L1缓存是写入，即修改的数据会立即写回到下一级。
* L2缓存也是每个处理器核独享，一共32MB，分别是16MB的数据缓存和16MB的指令缓存，也是写入方式。
* L3缓存位于处理器芯片(PU)上，一共256MB，由8个处理器核共享；L3缓存是shared-victim virtual缓存，由处理器芯片上8个处理器核的L2缓存组成
* L4缓存由一个CPC Drawer内处理器共享，一共2GB；L4缓存是shared-victim virtual缓存，由一个CPC Drawer内所有的L2缓存组成

## 10.3 内存结构
z16内存子系统使用高速差分接口，下图展示了一个CPC drawer的内存拓扑结构： 
![Pasted image 20230921171056.png](/assets/images/z/Pasted image 20230921171056.png)
## 10.4 SMP系统
### 10.4.1 CPC drawer
一个z16 CPC drawer主要有下列部分组成：
* 4个Dual chip modules (DCMs) 
* 最多48个内存DIMMs
* PCIe I/O Drawer连接插槽

下图展示了一个CPC drawer：
![Pasted image 20231216230346.png](/assets/images/z/Pasted image 20231216230346.png)

每个CPC Drawer有4个DCM，每个DCM有两个z16处理器芯片，下图展示了一个CPC Drawer内部连接：
![Pasted image 20230921134737.png](/assets/images/z/Pasted image 20230921134737.png)
具体链接如下：
* M-bus，用于同一个DCM里两个PU之间连接，
* X-bus, 用于同一个Drawer内部PU之间的连接
* A-bus，用于不同Drawer之间的连接
### 10.4.2 Frame
z16服务器主要组成部件包括：
* 1-4个19-inch 42u frame
* 1-4个CPC (Processor) drawers 
* 最多12个PCIe+ Gen3 I/O drawers 
* CPC drawer Cooling Units : Radiator cooling assembly (RCA) 或Water Cooling Unit (WCU)
* 每个Frame 2-4个Intelligent Power Distribution Units (iPDU)或1-6对Bulk Power Regulators

下图展示了一个全配置的z16服务器前视图：
![Pasted image 20231216230217.png](/assets/images/z/Pasted image 20231216230217.png)
4个CPC Drawer通过A-Bus形成全连接，下图展示了4个CPC Drawer之间的连接：
![Pasted image 20230921174248.png](/assets/images/z/Pasted image 20230921174248.png)
CPC drawer之间的通信由L4虚拟缓存管理，保证Drawer之间的一致性
### 10.4.3 I/O drawer
z16每个CPC drawer最多支持12个I/O卡，支持PCIe+ I/O drawer。下图展示了CPC上的I/O的连接示意图：
![Pasted image 20230921110738.png](/assets/images/z/Pasted image 20230921110738.png)

## 10.5 虚拟化

逻辑分区(Logical partitioning)是PR/SM实现的功能，可以运行在LPAR模式或则DPM(Dynamic Partition Manager模式。DPM提供了一个动态管理 I/O资源的图形界面。当一个LPAR激活时，PR/SM会为之分配处理器和内存；

虚拟化是IBM Z系列的一个主要优势，由硬件，固件和操作系统一起组成。所有的计算资源(CPU，内存和I/O)都被虚拟化，可以被不同操作系统使用。LPAR是IBM Z上提供虚拟化支持的技术，可以提供对客户系统最高程度的隔离。

IBM Z上提供虚拟化管理的硬件hypervisor(type 1)是PR/SM，集成在固件中，PR/SM管理硬件资源并建立LPAR来运行操作系统，中间件和应用软件。

同时，IBM Z上也支持软件hypervisor(type 2)：
* **z/VM**  支持在一个LPAR里运行多个虚拟客户系统以及嵌套虚拟化
* **KVM** 开源的虚拟化hypervisor

z/VM和KVM都运行在PR/SM之上，并可以同时存在，下图展示了IBM Z平台上支持的虚拟化方案： 
![Pasted image 20230921163414.png](/assets/images/z/Pasted image 20230921163414.png)

## 10.6 系统控制
系统的控制使用BMC，BMC替换了之前系统使用的FSP；每个Drawer有两个BMC，每个BMC有一个Ethernet端口，连接到内部的Ethernet LAN，和support elements (SEs)通信，并提供了subsystem interface (SSI)接口来控制系统。下图展示了BMC和SE之间连接：
![Pasted image 20230921171624.png](/assets/images/z/Pasted image 20230921171624.png)
所有处理器核在上电之后都需要定性，否则不可用；可以被定性为 CP, IFL, ICF, zIIP, IFP或SAP。
* **CP(Central Processor)** 用于通用计算任务
* **IFL(Integrated Facility for Linux)** 专门用于运行Linux
	* **Unassigned Integrated Facilities for Linux (UIFL)** 作为IFL购买但还未激活
* **zIIP(System z Integrated Information Processor)** zIIP用于卸载z/OS负载，以及使用SRB (System Recovery Boost)
	* **Unassigned zIIP** 作为zIIP购买但是还未激活，不可使用
* **Integrated Coupling Facility (ICF)** ICF用于在Parallel Sysplex 环境里运行CFCC
	* **Unassigned Coupling Facility** 作为ICF购买但是还未激活
* **SAP(System assist processors)** SAP用于运行channel subsystem LIC(Licensed Internal Code)，来控制IO操作，SAP为所有的LPAR的IO操作服务
* **IFP(Integrated firmware processor)** 在系统初始化时，会从所有可用PUs里分配一个作为IFP，作为系统管理使用，标准单元，不受客户定制
## 10.7 RAS
PU的错误恢复机制称为静默恢复，即当发现错误时，指令单元会重试指令并尝试恢复错误；如果第二次尝试也失败了，会将当前处理器核上状态迁移到其他处理器核上。下图展示了处理器的错误检测核恢复流程：
![Pasted image 20230922172019.png](/assets/images/z/Pasted image 20230922172019.png)

## 10.8 Cooling options
z16服务器CPC Drawer里的DCMs使用内部水冷系统制冷，而水冷散热器，PCIe+ I/O drawers则使用风冷散热。散热器，风扇和传感器都使用N+1的冗余设计。下图展示了z16的散热系统：
![Pasted image 20230921173918.png](/assets/images/z/Pasted image 20230921173918.png)
# 参考文献
1. Cedric Lichtenau, Alper Buyuktosunoglu, Ramon Bertran, Peter Figuli, Christian Jacobi, Nikolaos Papandreou, Haris Pozidis, Anthony Saporito, Andrew Sica, and Elpida Tzortzatos. 2022. AI accelerator on IBM Telum processor: industrial product. In Proceedings of the 49th Annual International Symposium on Computer Architecture (ISCA '22). Association for Computing Machinery, New York, NY, USA, 1012–1028. https://doi.org/10.1145/3470496.3533042
2. D. Berger, C. Jacobi, C. R. Walters, R. J. Sonnelitter, M. Cadigan and M. Klein, "Enterprise-Class Multilevel Cache Design: Low Latency, Huge Capacity, and High Reliability," in IEEE Micro, vol. 43, no. 1, pp. 58-66, 1 Jan.-Feb. 2023, doi: 10.1109/MM.2022.3193642.
3. O. Geva et al., "IBM Telum: a 16-Core 5+ GHz DCM," 2022 IEEE International Solid-State Circuits Conference (ISSCC), San Francisco, CA, USA, 2022, pp. 46-48, doi: 10.1109/ISSCC42614.2022.9731541.
4. C. Jacobi, "Real-time AI for Enterprise Workloads: the IBM Telum Processor," 2021 IEEE Hot Chips 33 Symposium (HCS), Palo Alto, CA, USA, 2021, pp. 1-22, doi: 10.1109/HCS52781.2021.9567422.
5. A. Saporito et al., "Design of the IBM z15 microprocessor," in IBM Journal of Research and Development, vol. 64, no. 5/6, pp. 7:1-7:18, Sept.-Nov. 2020, doi: 10.1147/JRD.2020.3008119.
6. A. Saporito, "The IBM z15 processor chip set," 2020 IEEE Hot Chips 32 Symposium (HCS), Palo Alto, CA, USA, 2020, pp. 1-17, doi: 10.1109/HCS49909.2020.9220508.
7. W. P. Kostenko, J. G. Torok and D. W. Demetriou, "IBM z15: Improved data center density and energy efficiency, new system packaging, and modeling," in IBM Journal of Research and Development, vol. 64, no. 5/6, pp. 16:1-16:10, Sept.-Nov. 2020, doi: 10.1147/JRD.2020.3008100.
8. C. Berry et al., "2.7 IBM z15: A 12-Core 5.2GHz Microprocessor," 2020 IEEE International Solid-State Circuits Conference - (ISSCC), San Francisco, CA, USA, 2020, pp. 54-56, doi: 10.1109/ISSCC19947.2020.9063030.
9. N. Adiga, J. Bonanno, A. Collura, M. Heizmann, B. R. Prasky and A. Saporito, "The IBM z15 High Frequency Mainframe Branch Predictor Industrial Product," 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), Valencia, Spain, 2020, pp. 27-39, doi: 10.1109/ISCA45697.2020.00014.
10. B. Abali et al., "Data Compression Accelerator on IBM POWER9 and z15 Processors : Industrial Product," 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), Valencia, Spain, 2020, pp. 1-14, doi: 10.1109/ISCA45697.2020.00012.
11. C. Jacobi et al., "Design of the IBM z14 microprocessor," in IBM Journal of Research and Development, vol. 62, no. 2/3, pp. 8:1-8:11, March-May 2018, doi: 10.1147/JRD.2018.2798718.
12. W. P. Kostenko, D. W. Demetriou and J. G. Torok, "IBM z14: Improved datacenter characteristics, energy efficiency, and packaging innovation," in IBM Journal of Research and Development, vol. 62, no. 2/3, pp. 16:1-16:11, 1 March-May 2018, doi: 10.1147/JRD.2018.2803438.
13. C. Berry et al., "IBM z14™: 14nm microprocessor for the next-generation mainframe," 2018 IEEE International Solid-State Circuits Conference - (ISSCC), San Francisco, CA, USA, 2018, pp. 36-38, doi: 10.1109/ISSCC.2018.8310171.
14. D. Wolpert et al., "IBM z14: Enabling physical design in 14-nm technology for high-performance, high-reliability microprocessors," in IBM Journal of Research and Development, vol. 62, no. 2/3, pp. 10:1-10:14, March-May 2018, doi: 10.1147/JRD.2018.2800499.
15. C. Berry et al., "IBM z14: Processor Characterization and Power Management for High-Reliability Mainframe Systems," in IEEE Journal of Solid-State Circuits, vol. 54, no. 1, pp. 121-132, Jan. 2019, doi: 10.1109/JSSC.2018.2873582.
16. B. W. Curran et al., "The IBM z13 multithreaded microprocessor," in IBM Journal of Research and Development, vol. 59, no. 4/5, pp. 1:1-1:13, July-Sept. 2015, doi: 10.1147/JRD.2015.2418591.
17. C. R. Walters, P. Mak, D. P. D. Berger, M. A. Blake, T. C. Bronson, K. D. Klapproth, A. J. O'Neill, R. J. Sonnelitter, and V. K. Papazova. 2015. The IBM z13 processor cache subsystem. IBM J. Res. Dev. 59, 4–5 (July/September 2015), 3:1–3:14. https://doi.org/10.1147/JRD.2015.2428591
18. C. Axnix, G. Bayer, H. Böhm, J. Von Buttlar, M. S. Farrell, L. Cranton Heller, J. P. Kubala, S. E. Lederer, R. Mansell, A. Nuñez Mencias, and S. Usenbinz. 2015. IBM z13 firmware innovations for simultaneous multithreading and I/O virtualization. IBM J. Res. Dev. 59, 4–5 (July/September 2015), 11:1–11:11. https://doi.org/10.1147/JRD.2015.2435494
19. E. W. Chencinski et al., "Advances in the IBM z13 I/O function and capability," in IBM Journal of Research and Development, vol. 59, no. 4/5, pp. 5:1-5:10, July-Sept. 2015, doi: 10.1147/JRD.2015.2429032.
20. P. J. Meaney et al., "The IBM z13 memory subsystem for big data," in IBM Journal of Research and Development, vol. 59, no. 4/5, pp. 4:1-4:11, July-Sept. 2015, doi: 10.1147/JRD.2015.2429031.
21. C. K. Shum, F. Busaba and C. Jacobi, "IBM zEC12: The Third-Generation High-Frequency Mainframe Microprocessor," in IEEE Micro, vol. 33, no. 2, pp. 38-47, March-April 2013, doi: 10.1109/MM.2013.9.
22. F. Busaba et al., "IBM zEnterprise 196 microprocessor and cache subsystem," in IBM Journal of Research and Development, vol. 56, no. 1.2, pp. 1:1-1:12, Jan.-Feb. 2012, doi: 10.1147/JRD.2011.2173962.
23. B. W. Curran et al., "The zEnterprise 196 System and Microprocessor," in IEEE Micro, vol. 31, no. 2, pp. 26-40, March-April 2011, doi: 10.1109/MM.2011.34.
24. P. Mak, C. R. Walters and G. E. Strait, "IBM System z10 processor cache subsystem microarchitecture," in IBM Journal of Research and Development, vol. 53, no. 1, pp. 2:1-2:12, Jan. 2009, doi: 10.1147/JRD.2009.5388579.
25. C. . -L. K. Shum et al., "Design and microarchitecture of the IBM System z10 microprocessor," in IBM Journal of Research and Development, vol. 53, no. 1, pp. 1:1-1:12, Jan. 2009, doi: 10.1147/JRD.2009.5388586.
26. C. F. Webb, "IBM z10: The Next-Generation Mainframe Microprocessor," in IEEE Micro, vol. 28, no. 2, pp. 19-29, March-April 2008, doi: 10.1109/MM.2008.26.
27. T. J. Slegel, E. Pfeffer and J. A. Magee, "The IBM eServer z990 microprocessor," in IBM Journal of Research and Development, vol. 48, no. 3.4, pp. 295-309, May 2004, doi: 10.1147/rd.483.0295.
28. B. W. Curran et al., "IBM eServer z900 high-frequency microprocessor technology, circuits, and design methodology," in IBM Journal of Research and Development, vol. 46, no. 4.5, pp. 631-644, July 2002, doi: 10.1147/rd.464.0631.
29. E. M. Schwarz et al., "The microarchitecture of the IBM eServer z900 processor," in IBM Journal of Research and Development, vol. 46, no. 4.5, pp. 381-395, July 2002, doi: 10.1147/rd.464.0381.
30. Curran, B., Camporese, P., Carey, S., Chan, Y., Chan, Y.-H., Crea, R., Hoffman, D., Koprowski, T., Mayo, M., Northrop, G., Sigal, L., Smith, H., Tanzi, F., 2001. 15.5 A 1.1GHz First 64b Generation Z900 Microprocessor.
31. T. McPherson et al., "760 MHz G6 S/390 microprocessor exploiting multiple Vt and copper interconnects," 2000 IEEE International Solid-State Circuits Conference. Digest of Technical Papers (Cat. No.00CH37056), San Francisco, CA, USA, 2000, pp. 96-97, doi: 10.1109/ISSCC.2000.839707.
32. T. J. Slegel et al., "IBM's S/390 G5 microprocessor design," in IEEE Micro, vol. 19, no. 2, pp. 12-23, March-April 1999, doi: 10.1109/40.755464.
33. C. F. Webb and J. S. Liptay, "A high-frequency custom CMOS S/390 microprocessor," Proceedings International Conference on Computer Design VLSI in Computers and Processors, Austin, TX, USA, 1997, pp. 241-246, doi: 10.1109/ICCD.1997.628874.
34. C. F. Webb et al., "A 400 MHz S/390 microprocessor," 1997 IEEE International Solids-State Circuits Conference. Digest of Technical Papers, San Francisco, CA, USA, 1997, pp. 168-169, doi: 10.1109/ISSCC.1997.585319.
35. Slegel, T.J., Averill, R.M., Check, M.A., Giamei, B.C., Krumm, B.W., Krygowski, C.A., Li, W.H., Liptay, J.S., MacDougall, J.D., McPherson, T.J., Navarro, J.A., Schwarz, E.M., Shum, K., Webb, C.F., 1999. IBM’s S/390 G5 microprocessor design. IEEE Micro 19, 12–23. [https://doi.org/10.1109/40.755464](https://doi.org/10.1109/40.755464)
36. Turgeon, P. R.; Mak, P.; Blake, M. A.; Fee, M. F.; Ford, C. B.; Meaney, P. J.; Seigler, R.; Shen, W. W. (1999). _The S/390 G5/G6 binodal cache. IBM Journal of Research and Development, 43(5.6), 661–670._ doi:10.1147/rd.435.0661
37. G. S. Rao, T. A. Gregg, C. A. Price, C. L. Rao and S. J. Repka, "IBM S/390 Parallel Enterprise Servers G3 and G4," in IBM Journal of Research and Development, vol. 41, no. 4.5, pp. 397-403, July 1997, doi: 10.1147/rd.414.0397.
38. P. Mak et al., "Processor subsystem interconnect architecture for a large symmetric multiprocessing system," in IBM Journal of Research and Development, vol. 48, no. 3.4, pp. 323-337, May 2004, doi: 10.1147/rd.483.0323.
39. Heller, L.C., Farrell, M.S., 2004. Millicode in an IBM zSeries processor. IBM J. Res. & Dev. 48, 425–434. [https://doi.org/10.1147/rd.483.0425](https://doi.org/10.1147/rd.483.0425)