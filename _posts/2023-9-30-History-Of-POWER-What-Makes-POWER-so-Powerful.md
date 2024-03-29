---
title : History of POWER | What makes POWER so powerful
categories : [history]
tags : [chip, cpu, architecture, history]
---

# 前言
最近几年系统性的学习并梳理了近30多年的计算技术发展历史，稍有感悟。遂决定将之整理成文，目的有二，一来作为知识沉淀，串联起不同技术，挖掘不同架构之间的渊源，二来通过整理再次审视历史，期望窥见未来发展方向。我将本系列命名为**鉴往知远**, 主要关注**计算与互联**。 本文为第一篇，主要回顾*IBM POWER*系列。
# 0. 概述
*POWER (Performance Optimization With Enhanced RISC)*架构起源于1990年IBM的RISC System/6000产品。1991年，Apple, IBM, 和Motorola一起合作开发了*PowerPC*架构。1997, Motorola和IBM合作将*PowerPC*用于嵌入式系统。2006, Freescale和IBM合作制定了POWER ISA 2.03。 2019年8月21, 开源了POWER Instruction Set Architecture (ISA)。PowerISA由必须的基础架构，4组可选特性，一组过时特性组成。OpenPOWER也允许自定义扩展。

本文通过系统性回顾整个*POWER*系列处理器，试图通过*POWER*系列处理器发展的历史脉络，来展现近30年计算架构的变迁，技术的演进，进而窥见计算技术发展的未来。

本文组织形式如下:
* 第一章简单介绍**POWER**指令集架构3.1B版本， 也是最新**POWER10**处理器使用的指令集。通过本章介绍，可以了解程序在**POWER**架构上运行的过程及预期结果，掌握异常处理，特权模型，以及**POWER**调试手段
* 第二章简单回顾整个**POWER**系列处理器，总结各代处理器的面积，功耗，缓存，IO等基本内容
* 第三章，第四章，第五章分别描述**POWER 1**和**POWER 2**的整体架构，简单介绍了**POWER 3**的微架构，主要是了解这些古老系统结构
* 第六章详细描述**POWER 4**微架构以及从单核到双核的进化
* 第七章详细描述**POWER 5**微架构，从单线程到双线程的演进，以及集成的片上内存控制器
* 第八章介绍**POWER 6**处理器微架构，了解从之前乱序执行变为顺序执行的取舍
* 第九章介绍**POWER 7**处理器微架构，了解在支持更多线程情况下如何减少面积和功耗，以及内部缓存协议状态
* 第十章介绍**POWER 8**处理器微架构，了解如何从上一代SMT4到SMT8
* 第十一章主要介绍**POWER 9**处理器微架构，了解基本单元之间解耦，SMP互联
* 第十二章完整介绍**POWER 10**处理器微架构，SMP互联，片上加速器，中断
* 文章最后列出了主要的参考文献，主要是**POWER**指令集架构，各代**POWER**的架构和微架构论文，以及各代**POWER**服务器的技术文档

# 1. POWER指令集架构
## 1.1 寄存器
* __Condition Register (CR)__ 是32寄存器，记录指令执行结果，供测试和条件分支指令使用
* __Link Register (LR)__ 是64位寄存器，保存 `Branch Conditional to Link Register` 指令跳转地址, 并且可以保存当 __LK=1__ 时分支指令和 `System Call Vectored` 指令后的返回地址。
* __Count Register (CTR)__ 是64位寄存器。当执行的分支指令的 __BO__ 编码时候可以作为`for`循环计数寄存器。__Count Register__ 也可以保存 `Branch Conditional to Count Register` 指令的跳转目标地址。
* __VR Save Register (VRSAVE)__ 是32位寄存器，软件作为SPR使用。
* __Fixed-Point Exception Register (XER)__ 是64位寄存器
	* 0:31 Reserved
	* 32 __Summary Overflow (SO)__ 当Overflow置位时Summary Overflow置位
	* 33 __Overflow (OV)__ 指令执行溢出时置位
	* 34 __Carry (CA)__ 
	* 35:43 Reserved
	* 44 __Overflow32 (OV32)__ OV32 32位运行模式时溢出位
	* 45 __Carry32 (CA32)__ CA32 32位运行模式时溢出位
	* 46:56 Reserved
	* 57:63 指定 `Load String Indexed`  和 `Store String Indexed`  指令传输的字节数
* __Floating Point Status and Control Register(FPSCR)__ 控制浮点异常处理和浮点指令执行结果状态。32:55位是状态位, 56:63是控制位

![Pasted image 20230904172658.png](/assets/images/power/Pasted image 20230904172658.png)

* __Logical Partitioning Control Register (LPCR)__  __LPCR__ 控制资源的逻辑分区，__Logical Partitioning (LPAR)__ 功能将线程和存储空间分配到一个逻辑分区，用来隔离不同分区之间的线程；通过一个hypervisor，这种隔离可以作用于problem和privileged non-hypervisor态。
* __Logical Partition Identification Register (LPIDR)__  __LPIDR__ 设置逻辑分区ID
* __Machine State Register (MSR)__ 是64位寄存器，控制和定义了线程的状态：
	* 0 __Sixty-Four-Bit Mode (SF)__ 当线程处于ultravisor时软件要保证 __SF=1__ 
		* 0 线程运行在32位模式
		* 1 线程运行在64位模式
	* 1:2 Reserved 
	* 3 __Hypervisor State (HV)__ 具体意义参考特权模型章节
	* 4 Reserved 
	* 5 常0
	* 6:37 Reserved 
	* 38 __Vector Available (VEC)__ 
		* 0 线程不能执行任何向量指令 
		* 1 线程可以执行向量指令
	* 39 Reserved 
	* 40 __VSX Available (VSX)__ 
		* 0 向量不能执行 __Vector Scalar Extension(VSX)__ 指令
		* 1 向量可以执行 __VSX__ 指令
	* 41 __Secure (S)__ 
		* 0 线程处于非安全态，不能访问安全域线程，且不在ultravisor态 
		* 1 线程处于安全态
	* 42:47 Reserved 
	* 48 __External Interrupt Enable (EE)__ 
		* 0 __External__ , __Decrementer__ , __Performance Monitor__ , 和 __Privileged Doorbell__ 中断被禁止 
		* 1 __External__ , __Decrementer__ , __Performance Monitor__ , 和 __Privileged Doorbell__使能
	* 49 __Problem State (PR)__ 具体意义参考特权模型章节
	* 50 __Floating-Point Available (FP)__ 
		* 0 线程不能执行浮点指令 
		* 1 线程可以执行浮点指令
	* 51 __Machine Check Interrupt Enable (ME)__ 
		* 0 __Machine Check__ 中断禁止 
		* 1 __Machine Check__ 中断使能
	* 52 __Floating-Point Exception Mode 0 (FE0)__ 
	* 53:54 __Trace Enable (TE)__
		* 00 Trace Disabled: 线程正常执行指令
		* 01 Branch Trace: 当线程执行完分支指令之后产生 __Branch type Trace__ 中断
		* 10 Single Step Trace: 当线程成功完成下一个指令时产生一个 __Single-Step type Trace__ 中断，`urfid` , `hrfid` , `rfid` , `rfscv` , 或 `Power-Saving Mode` 指令除外
		* 11 Reserved
	* 55 __Floating-Point Exception Mode 1 (FE1)__
	* 56:57 Reserved 
	* 58 __Instruction Relocate (IR)__ 0 禁止指令地址转换； 1 使能指令地址转换
	* 59 __Data Relocate (DR)__ 
		* 0 禁止数据地址转换，并且不发生 __Effective Address Overflow (EAO)__
		* 1 使能数据地址转换，__EAO__ 产生 __Data Storage__ 中断
	* 60 Reserved 
	* 61 __Performance Monitor Mark (PMM)__ 软件控制Performance Monitor

## 1.2 计算模式
处理器提供两种执行模式, 64位模式和32位模式。 两种模式下，设置64位寄存器指令仍然影响所有64位。计算模式控制有效地址的解释, __Condition Register__ 和 __XER__ 的设置, 当 __LK=1__ 时 __Link Register__ 被分支指令的设置 , 以及 __Count Register__  被条件分支指令的使用。几乎所有指令都可以在两种模式下运行。在两种模式下，有效地址的计算都使用所有相关寄存器的64位( __General Purpose Registers__ , __Link Register__ , __Count Register__ 等) 并且产生64位结果。

## 1.3 指令格式
下面列出了大部分的指令格式，还有一些请参考架构手册。
![Pasted image 20230904175410.png](/assets/images/power/Pasted image 20230904175410.png)

有一类特殊指令格式，称为指令前缀格式，前缀指令由4字节前缀和4字节后缀组成。所有前缀的0:5都是*0b000001*.

### 1.3.1 分支指令
分支指令按照下面5种方式计算有效地址(EA):
1. 将分支指令地址加上位移 (当分支或条件分支的 __AA=0__ 时).
2. 使用绝对地址 (当分支或条件分支的 __AA=1__ 时).
3. 使用 __Link Register__ 里的地址( __Branch Conditional to Link Register__ ).
4. 使用 __Count Register__ 里的地址 ( __Branch Conditional to Count Register__ ).
5. 使用 __Target Address Register__ 里的地址 ( __Branch Conditional to Target Address Register__ ).
![Pasted image 20230905092520.png](/assets/images/power/Pasted image 20230905092520.png)
![Pasted image 20230905092500.png](/assets/images/power/Pasted image 20230905092500.png)

### 1.3.2 条件寄存器指令
这些是操作条件寄存器 __CR__ 的指令
![Pasted image 20230905093726.png](/assets/images/power/Pasted image 20230905093726.png)
![Pasted image 20230905093750.png](/assets/images/power/Pasted image 20230905093750.png)
![Pasted image 20230905094031.png](/assets/images/power/Pasted image 20230905094031.png)
![Pasted image 20230905094050.png](/assets/images/power/Pasted image 20230905094050.png)
![Pasted image 20230905094419.png](/assets/images/power/Pasted image 20230905094419.png)

### 1.3.3 系统调用指令
系统调用指令主要用于切换特权模式
* 当 __LEV=1__ 时，唤起hypervisor
* 当 __LEV=2__ 和 __SMFCTRL.E = 1__ 时, 唤起ultravisor
* 当 __LEV=2__ 和 __SMFCTRL.E = 0__ 时, 唤起hypervisor。但是，这种方式是编程错误
![Pasted image 20230905095119.png](/assets/images/power/Pasted image 20230905095119.png)

### 1.3.4 定点加载存储指令
* 有效地址(EA)索引的字节，半字，字，双字被加载到 __RT__ 寄存器
* __RS__ 寄存器里字节，半字，字，双字被存储到有效地址(EA)索引空间
![Pasted image 20230905101900.png](/assets/images/power/Pasted image 20230905101900.png)
![Pasted image 20230905102304.png](/assets/images/power/Pasted image 20230905102304.png)
![Pasted image 20230905102532.png](/assets/images/power/Pasted image 20230905102532.png)
![Pasted image 20230905102612.png](/assets/images/power/Pasted image 20230905102612.png)
![Pasted image 20230905102630.png](/assets/images/power/Pasted image 20230905102630.png)
![Pasted image 20230905102726.png](/assets/images/power/Pasted image 20230905102726.png)
![Pasted image 20230905102804.png](/assets/images/power/Pasted image 20230905102804.png)

### 1.3.5 定点算术指令
* `addic` , `addic` , `subfic` , `addc` , `subfc` , `adde` , `subfe` , `addme` , `subfme` , `addze` 和 `subfze` 指令设置 __CR.CA__ , 在64位模式下反映位0的进位，在32位模式反映位32的进位
* 对于XO形式的`Multiply Low`和`Divide`指令, __CR.SO/OV/OV32__ 设置依赖计算模式, 反映 `mulld` , `divd` , `divde` , `divdu` 和 `divdeu` 的64位溢出, `mullw` , `divw` , `divwe` , `divwu` 和 `divweu` 低32位的溢出.

### 1.3.6 定点比较指令
定点比较指令将寄存器 __RA__ 和如下值比较
a. 符号扩展SI
b. 无符号扩展UI
c. __RB__ 寄存器的值

`cmpi` 和 `cmp` 是有符号比较, `cmpli` 和 `cmpl` 是无符号比较.
![Pasted image 20230905103727.png](/assets/images/power/Pasted image 20230905103727.png)

### 1.3.7 定点逻辑指令
定点逻辑指令对64位进行按位操作
![Pasted image 20230905104113.png](/assets/images/power/Pasted image 20230905104113.png)

### 1.3.8 定点旋转和移位指令
定点旋转和移位指令对通用寄存器值进行旋转和移位操作，从位0开始
![Pasted image 20230905104311.png](/assets/images/power/Pasted image 20230905104311.png)

### 1.3.9 Binary Coded Decimal (BCD) 辅助指令
Binary Coded Decimal辅助指令操作BCD( __cbcdtd__ 和 __addg6s__ )和十进制浮点操作数
![Pasted image 20230905104845.png](/assets/images/power/Pasted image 20230905104845.png)

## 1.4 特权模型
__Machine State Register (MSR)__ 是64位寄存器，定义了线程的特权状态。
线程的特权状态由 __MSR.S__ , __MSR.HV__ 和 __MSR.PR__ 组成，意义如下:
![Pasted image 20230905144055.png](/assets/images/power/Pasted image 20230905144055.png)
__MSR.SF__ 控制线程32/64位计算模式.

## 1.5 存储模型
存储属性是以页表为单位设置，每一个读写访问都需要检查对应的存储属性。主要有下列存储属性：
* __Write Through Required__ 写操作不会使数据缓存里的数据转变为修改
* __Caching Inhibited__ 对Caching Inhibited访问会直接在主存进行 
* __Memory Coherence Required__ 对Memory Coherence Required访问需要保持一致性
* __Guarded__  只有在下列情况才会对Guarded存储地址进行访问：
	* 需要顺序执行模型的指令的访问
	* 读访问且存储地址已经在缓存中

只有进行有效地址的转换的访问才受这些属性的影响。存储访问的内存模型是 __weakly consistent__ ，相对于 __stronger consistency__ 的模型提高了性能。

## 1.6 内存管理及虚拟化
地址转换模式是由 __Partition Table Entry__ 里的 __Host Radix__ 位控制。__Host Radix__  位控制当前分区是使用 __HPT(Hashed Page Table)__   还是 __Radix Tree__  进行转换。__MSR.HV/PR/IR/DR__  决定了地址转换入口和行为。
* __Partition Table Control Register (PTCR)__ 是64位寄存器，保存了分区表的host的真实基地址。
	* 4:51 __PATB__ 分区表基地址
	* 59:63 __PATS__ 分区表大小=2^(12+PATS)且 PATS ≤24
![ptcr.png](/assets/images/power/ptcr.PNG)

### 1.6.1 Ultravisor Real, Hypervisor Real, 和Virtual Real Addressing Modes
当指令地址转换或数据地址转换功能被禁止时，根据当前处于的特权模式，内存访问被称为 __ultravisor real addressing mode__ , __hypervisor real addressing mode__ 和 __virtual real addressing mode__ 。这三种模式内存访问根据 __MSRS.HV__ , __PATE.HR__ , __PATE.PS__ , __URMOR__ , __HRMOR__ , 有效地址第0位 (EA0), 和 __RMCF__ 值分为不同方式。 有效地址1:3位不使用。

#### 1.6.1.1 Ultravisor/Hypervisor Offset Real Mode Address
当 __MSR.HV = 1__  且 __EA.0 = 0__ 时, 存储访问由 __Ultravisor Real Mode Offset Register__  或 __Hypervisor Real Mode Offset Register__ 控制, 根据 __MSR.S__ 值：
* 当 __MSR.S=1__ 时, 有效地址4:63位和 __URMOR__ 里的60位偏移值按位或，得到的60位结果就是访问的真实地址
* 当 __MSR.S=0__ 时, 有效地址4:63位和 __HRMOR__ 里的60位偏移值按位或，得到的60位结果就是访问的真实地址

#### 1.6.1.2 Virtual Real Mode Addressing Mechanism
当 __MSR.HV = 0__  且 __MSR.DR=0__ 或 __MSR.IR=0__ 时，分区使用 __Paravirtualized HPT(PATE.HR=0)__ 进行地址转换，能够访问的内存空间由 __Virtualized Real Mode Area (VRMA)__ 控制。地址转换和存储保护仍然像地址转换使能一样处理，只是有效地址到虚拟地址的转换使用 __VRMA.SLBE__ ，而不是通过SLB。有效地址的0:23位不使用。下图展示了 SLBE 的内容：
![vrma.slbe.png](/assets/images/power/vrma.slbe.PNG)

### 1.6.2 Segment Translation
下图展示了64位有效地址通过 __Segment Lookaside Buffer (SLB)__ 转换为78位虚拟地址的过程：
![Pasted image 20230905155844.png](/assets/images/power/Pasted image 20230905155844.png)
如果 __SLB__ 未命中， __LPCR.UPRT=1__ , 且 __MSR.HV=0__  或 __LPID=0__ 时会搜索 __Segment Table__ 。__Segment Lookaside Buffer (SLB)__ 指定了 __Effective Segment IDs (ESIDs)__ 和 __Virtual Segment IDs (VSIDs)__ 的映射。

在 __Paravirtualized HPT__ 模式，使用Page Table来将78位虚拟地址转换为真实地址。__Hashed Page Table (HTAB)__ 指定了虚拟页表和真实页表的映射，__Hashed Page Table (HTAB)__ 由 __Page Table Entry Groups (PTEGs)__ 组成。一个 __PTEG__ 包含8个 __Page Table Entries (PTEs)__ ，每个16。
下图展示了使用 __HPT__ 将78位虚拟地址转换位60位真实地址过程
![Pasted image 20230906084212.png](/assets/images/power/Pasted image 20230906084212.png)

### 1.6.3 Radix Tree Translation
__Radix Tree root descriptor (RTRD)__  指定了转换的地址大小，根表的大小和位置。下图展示了4级Radix Tree在 __PDE.NLS=13__  和 __NLS=9__ 时将52位EA转换成56位RA的过程：
![Pasted image 20230905170905.png](/assets/images/power/Pasted image 20230905170905.png)
Radix Tree Page Directory Entry内容如下：
![Pasted image 20230905171134.png](/assets/images/power/Pasted image 20230905171134.png)
![Pasted image 20230905171203.png](/assets/images/power/Pasted image 20230905171203.png)
Radix Tree Page Table Entry
![radix_tree_pte.png](/assets/images/power/radix_tree_pte.png)
* 0 V 有效位
* 1 L 叶节点
* 2 sw SW bit 0 
* 7:51 RPN Real Page Number 
* 52:54 sw SW bits 1:3 
* 55 R Reference 
* 56 C 修改位
* 58:59 ATT 属性
	* 0b00- normal memory (0010) 
	* 0b01- reserved 
	* 0b10- non-idempotent I/O (0111) 
	* 0b11- tolerant I/O (0110) 
* 60:63 EAA 访问权限
	* 0 Privilege (只应用于process-scoped translation) 
		* 0 - problem可访问; privileged访问由 __[I]AMR__ 位0控制
		* 1 - 只可privileged访问
	* 1 Read 
		* 0 不可读 
		* 1 可读
	* 2 Read/Write 
		* 0 不可读写
		* 1 可读写
	* 3 Execute 
		* 0 不可执行 
		* 1 可执行

#### 1.6.3.1 Nested Translation
当 __MSR.HV=0__  且地址转换使能时，对于guest real address必须经过分区的hypervisor的Radix Tree转换。下图展示了Radix on Radix Page Table对52位EA转换过程，一共需要25次内存访问：
![Pasted image 20230905172603.png](/assets/images/power/Pasted image 20230905172603.png)
### 1.6.4 Secure Memory Protection
当 __SMFCTRL.E=1__ 时使能Secure Memory Protection。每个内存地址有Secure Memory属性 __mem.SM__ 。当 __mem.SM=1__ 时表示是安全内存区域；__mem.SM=0__  表示是普通内存区域。通常只有安全分区和ultravisor会访问安全内存区域。

## 1.7 异常和中断
Power指令集架构提供了中断机制，允许线程能够处理外部信号，错误或指令执行异常。系统复位和机器检查中断是不可覆盖的，其他中断可覆盖且处理器状态可保留。当中断发生时， __SRR0__ , __HSRR0__ 或 __USRR0__ 指向正在执行且未完成的指令。
中断可分为是否是执行指令引起或其他系统异常。系统异常包括：
* System Reset
* Machine Check
* External
* Decrementer
* Directed Privileged Doorbell
* Hypervisor Decrementer
* Hypervisor Maintenance
* Hypervisor Virtualization
* Directed Hypervisor Doorbell
* Directed Ultravisor Doorbell
* Performance Monitor

其他都是指令中断

### 1.7.1 中断寄存器
根据处理器所在特权状态，可以分为:
* __Machine Status Save/Restore Registers__  中断发生时，处理器状态被保存在 __Machine Status Save/Restore registers__ ( __SRR0__ 和 __SRR1__ )。
* __Hypervisor Machine Status Save/Restore Registers__  中断发生时，处理器状态被保存在 __Hypervisor Machine Status Save/Restore registers__ ( __HSRR0__  and __HSRR1__ )。
* __Ultravisor Machine Status Save/Restore Registers__  中断发生时，处理器状态被保存在 __Ultravisor Machine Status Save/Restore registers__ ( __HSRR0__  and __HSRR1__ )。

### 1.7.2 中断处理
中断处理包括保存一部分线程状态，识别中断原因，并从相应中断向量地址执行：
1. 根据中断类型将指令地址加载到 __SRR0__ , __HSRR0__ , 或 __USRR0__  
2. 根据中断类型将相关信息保存到 __SRR1__ , __HSRR1__ , 或 __USRR1__ 的33:36位和42:47位
3. 将 __MSR__ 保存到 __SRR1__ , __HSRR1__ , 或 __USRR1__  的0:32, 37:41, 和 48:63位
4. 根据 __LPCR.AIL__  或 __LPCR.HAIL__ 设置 __MSR.IR/DR__  位， 并且设置 __MSR.SF = 1__ 。第一条中断指令执行时生效
5. 在新的 __MSR__  设置下，从对应中断向量处取指执行代码。__LPCR.AIL__  或 __LPCR.HAIL__  决定地址是否需要偏移
下表列出了所有类型的中断向量及其有效地址：
![Pasted image 20230906141329.png](/assets/images/power/Pasted image 20230906141329.png)

## 1.8 调试
调试功能允许硬件和软件通过追踪指令流，比较数据地址，单步执行等进行调试：
* __Come From Address Register__  __Come From Address Register (CFAR)__ 是64位寄存器， 当执行 `rfebb` , `rfid` , 或 `rfscv` 执行时，寄存器值设置为当前执行的有效地址。
* __Completed Instruction Address Breakpoint__  __Completed Instruction Address Breakpoint__ 提供了发现完成执行特定地址指令的机制。地址比较是基于有效地址(EA)。__Completed Instruction Address Breakpoint__ 机制是由 __Completed Instruction Address Breakpoint Register (CIABR)__ 控制。
* __Data Address Watchpoint__  __Data Address Watchpoint__ 提供了发现多个双字有效地址(EA)加载存储访问的机制。至少两个独立地址范围可以指定。每个 __Data Address Watchpoint__ 是由一对SPRs控制：__Data Address Watchpoint Register(DAWRn)__ 和 __Data Address Watchpoint Register Extension (DAWRXn)__ 

# 2. POWER处理器概述
* 1975年，IBM Thomas J. Watson Research Center发明了第一个RISC机器，801。801原始设计目标是1 IPC，研究重点是定义一个能够每周期执行多个指令，即超标量的架构。研究的结果是第二代RISC架构，称为"AMERICA architecture"
* 1986年，IBM 位于Austin, Texas的开发RT System的实验室, 基于AMERICA architecture开始开发产品。这个研究最终实现了*IBM RISC System/6000(RS/6000)* ，即IBM's **POWER**架构
* 1990年，IBM发布RISC System/6000, 包括9种当时工业界最快最强大的工作站。RISC System/6000使用IBM研发的精简指令集，一个IBM实现的新版本UNIX操作系统*Advanced Interactive Executive (AIX)*。这就是**POWER 1**
* 1991年，Apple，IBM和Motorola宣布研发一种用于个人电脑和低端工作站的精简指令集处理器，可以运行IBM AIX和Macintosh。这就是**PowerPC**
* 1993年，IBM发布了自1990年以来RISC System/6000产品线最大的更新，包括四款使用PowerPC 601处理器的工作站，和三款使用IBM多芯片微处理器**POWER2**的高端工作站
* 1999年，IBM发布了基于**POWER3**微处理器的RS/6000 SP超级计算机。**POWER3**能每秒执行20亿次运算，比使用**POWER2** 的Deep Blue计算机快2倍。Deep Blue在1997年打败象棋世界冠军Garry Kasparov
* 2001年,  **POWER4**实现单芯片双核心，共享第二级缓存，以及第三级缓存控制器
* 2004年，**POWER5**实现双线程，并集成内存控制器
* 2007年，**POWER6**实现4.5G的高频。为了实现高频从之前的乱序执行简化成了顺序执行
* 2010年，**POWER7**实现单片8核四线程，片上集成32M 3级缓存
* 2016年，**POWER8**实现单片12核八线程，片外128M 4级缓存，具备NVLink接口
* 2017年，**POWER9**增加了更多IO和带宽
* 2021年，**POWER10**实现单片15核八线程，增加互联能力

下表总结了各代POWER处理器的各方面数据，可以从中看到POWER系列的发展路径。

|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||POWER1|POWER2|POWER3|POWER4|POWER5|POWER6|POWER7|   |POWER8|   |POWER9|   |   |POWER10|
|Date|1990|1993|1997|2001|2004|2007|2010|2012|2014|2016|2017|2018|2020|2021|
|Technology|1.0um|0.35um|0.22um|180nm SOI|130nm SOI, 90nm|65nm SOI|45nm SOI|32nm SOI|22nm|22nm|GF 14nm HP|GF 14nm HP|GF 14nm HP|Samsung 7nm|
|Transistors|8.9M||15M|174M|276M|790M|1.2B|2.1B|1.2B||8B|8B|8B|18B|
|Area(mm^2)|12.7x12.7(ICU,FXU,FPU)  <br>11.3x11.3(DCU,SCU)|12.7x12.7 (ICU,FXU,FPU)  <br>11.7x9.55 (DCU, SCU)||267|389|341|567|567|650|650|728(25.3x28.8)|728(25.3x28.8)|728(25.3x28.8)|602|
|TDP(W)|NA|NA|NA||||||120-190|120-190|190|190|190||
|Frequency(GHz)|0.02-0.03|0.055-0.0715|0.2-0.45|1.1-1.3|1.5|4-4.5|||2.0-3.3|2.0-3.3|2.75-3.8|2.75-3.8|2.75-3.8|3.5-4|
|No. Inst Decode|3|8|4|8|8|8|6|6|8|8|||||
|No. Inst Issue|2|6|4|5|5|7|8|8|10|10|||||
|COREs|1|1|1|2|2|2(in order)|8|8|12|12|12|12|12|15|
|Threads|ST|ST|ST|ST|SMT2|SMT2|SMT4|SMT4|SMT8|SMT8|SMT8|SMT8|SMT8|SMT8|
|L1 Cache|I$: 8K 2 way  <br>D$: 64K 4 way|I$: 32K  <br>D$: 4x32K|I$: 32K 128 way 128B cacheline  <br>D$: 64K 128 way 4 banks|I$: 64K direct map  <br>D$: 32K 2 way|I$: 64K 2 way  <br>D$: 32K 4 way|I$: 64K 4 way  <br>D$: 64K 8 way|I$: 32K 4 way  <br>D$: 32K 8 way|I$: 32K 4 way  <br>D$: 32K 8 way|I$: 32K  <br>D$: 64K|I$:32KB 8-way  <br>D$:32KB 8-way (SMT4)|I$:32KB 8-way  <br>D$:32KB 8-way (SMT4)|||I$:48KB 6-way  <br>D$:32KB 8-way(SMT4)|
|L2 Cache|NA|512K-3M direct map|1M-16M offchip|3x480K 8 way|3x640K 10 way|2x4M 8 way|256K/Core 8 way|256K/Core 8 way|512K/Core 8 way|1MB 8-way inclusive|1MB 8-way inclusive|||2MB 8-way|
|L3 Cache|NA|NA|NA|8 way cache directory  <br>32 M off chip  <br>512B cache line|3x12M 12 way  <br>256B cache line|32M 16 way  <br>128B off chip|4M/Core  <br>32MB eDRAM|10M/Core  <br>80MB eDRAM|8M/Core 96M eDRAM 8 way  <br>128M offchip L4|10MB/SMT8 Core 20-way  <br>120MB  <br>eDRAM|10MB/SMT8 Core 20-way  <br>120MB  <br>eDRAM|120MB||8MB/SMT8 Core 16-way  <br>120MB|
|IO|||||PCIE G2 3GB/s|10GB/s|PCIE G2@20GB/s  <br>SMP 6x60GB/s|PCIE G2@20GB/s  <br>SMP 6x60GB/s|PCIE Gen3 x32  <br>CAPI 1.0  <br>SMP 6.4G x3|PCIe Gen3  <br>CAPI 1.0  <br>NVLINK 25GT/s 160GB/s|PCIe Gen4 x48  <br>CAPI 2.0  <br>OpenCAPI3.0  <br>NVLINK2.0  <br>25GT/s x48  <br>300GB/s|PCIe Gen4 x48  <br>CAPI 2.0  <br>OpenCAPI3.0  <br>NVLINK2.0  <br>25GT/s x48  <br>300GB/s|PCIe Gen4 x48  <br>CAPI 2.0  <br>OpenCAPI4.0  <br>NVLINK2.0  <br>25GT/s x48  <br>300GB/s|PCIe Gen5 x64  <br>PowerAXON 16 x8@32GT/s  <br>SMP Interconnect 14 x9@32GT/s|
|Memory|8-256M|128bit 64M-2G||0-16G|15GB/s|30GB/s|100GB/s|100GB/s|230GB/s 9.6G DMI|230GB/s|150GB/s|8 DMI DDR4 ports@230GB/s|16 x8@25GT/s OMI  <br>650GB/s|16 x8@32GT/s  <br>1TB/s|
|SMP(sockets)|NA||2|16|32|32|32|32|32|32|||||
|Comments|RISC architecture|SMP|64 bit|hypervisor mode|integrated memory ctrl|private L2  <br>decimal floating point  <br>vector multimedia ext|eDRAM L3 cache|on die acceleration|big data optimized|high bandwidth GPU attach|scale out  <br>direct-attach DDR4 memory|scale up  <br>memory buffers|memory attached via OMI|single chip module , 16 sockets  <br>dual chip module, 4 sockets|


# 3. POWER 1
RS/6000架构在当时一个主要特性就是集成了浮点算法单元，同时将不同功能单元独立，主要包括：
* 定点单元
* 浮点单元
* 分支单元

下图展示了RISC system/6000架构逻辑结构
![Pasted image 20230912135927.png](/assets/images/power/Pasted image 20230912135927.png)
RISC System/6000架构定义了分离的指令和数据缓存。这些缓存都是写入。指令缓存主要和分支单元耦合，而数据缓存由定点和浮点单元共享。
* **分支单元**  分支单元主要负责取指，地址翻译和检查， 中断处理。除非对应定点或浮点单元上指令队列已满或对定点或浮点单元上数据存在依赖性，分支单元能不断对下一个指令进行取指，解码，并执行相应指令，或者将对应定点或浮点指令分发到对应定点或浮点单元。分支单元每周期可以至少获取3条指令，分别对应定点单元，浮点单元，及分支单元。并且每周期可以分发一个定点和浮点指令。**POWER 1**不支持分支延迟槽。
* **定点单元** 定点单元除了处理所有定点算术指令外，还需要为浮点单元及自身计算数据地址。因此，定点单元需要负责调度浮点单元和数据缓存之间数据交换。浮点单元寄存器只负责接收或提供数据，因此，浮点单元加载和存储操作是消耗定点单元周期数。
* **浮点单元** 浮点单元支持ANSI/IEEE Standard 754-1985。 RISC System/6000浮点运算是双精度计算。因此，单精度浮点运算也会被转换为双精度进行运算。

下图展示了**POWER 1**的有效地址(EA)转换过程，32位有效地址首先经过段寄存器转换为52位虚拟地址(VA), 然后52位虚拟地址(VA)经过TLB翻译成32位真实地址(RA)：
![Pasted image 20230912141125.png](/assets/images/power/Pasted image 20230912141125.png)


# 4. POWER 2
**POWER2** 包括高性能的Multi-Chip Module (MCM)和和低端的Single Chip Module(SCM)，区别主要是DCU和内存接口，MCM逻辑框图如下所示：

![Pasted image 20230816174929.png](/assets/images/power/Pasted image 20230816174929.png)

包括4个Data Cache Unit芯片，每个128 Kbytes数据缓存，32 Kbyte指令缓存，512 Kbyte - 2 Mbyte L2缓存，4 Word宽内存接口，支持 64 Mbyte - 2048 Mbyte内存

SCM逻辑框图如下所示：

![Pasted image 20230816175250.png](/assets/images/power/Pasted image 20230816175250.png)

包括两个DCU，每个64KB数据缓存，512 Kbyte - 1 Mbyte L2 缓存，2 Word宽内存接口，支持32 Mbyte - 512 Mbyte内存。

## 4.1 POWER 2 Core
处理器每周期可以分发6条指令，包括两个定点单元，一个浮点单元，和分支单元。下图展示了逻辑框图:

![Pasted image 20230816175511.png](/assets/images/power/Pasted image 20230816175511.png)

# 5. POWER 3
POWER3是第一个支持32位和64位PowerPC ISA的处理器。下图展示了**POWER 3** 的全芯片图
![Pasted image 20230912085657.png](/assets/images/power/Pasted image 20230912085657.png)

POWER3由7个功能单元组成：
* Instruction processing unit (IPU)
* Instruction ﬂow unit (IFU)
* Fixed-point unit (FXU)
* Floating-point unit (FPU)
* Load/store unit (LSU)
* Data cache unit (DCU)
* Bus interface unit (BIU)

下图展示了*POWER3* 处理器功能模块图：
![Pasted image 20230912085752.png](/assets/images/power/Pasted image 20230912085752.png)

* *Instruction processing unit* 和*instruction ﬂow unit* **IPU** 和**IFU** 负责取指，缓存以及指令分发和完成整个生命流程。IPU有32KB指令缓存和*cache reload buffer(CRB)* 。 指令缓存缓存行大小是128B，因此一共有256行，组织成128路组相联，并且单周期访问。*CRB* 保存从内存读取最新缓存行。另外实现虚拟地址转换，实现了256条目2路组相联*instruction translation lookaside buffer (ITLB)* 和16条目 *instruction segment lookaside buffer (ISLB)* 。每周期可以取8条指令，分发4条指令，并且完成4条指令。为了提高吞吐，指令顺序分发，大部分可以乱序执行和结束，顺序完成。指令分发到不同功能单元指令队列并且由32条目的*completion queue*记录。这些功能单元指令队列确保对应功能单元有足够指令可以选择来执行，并且阻塞的指令不会阻碍IFU的指令分发。*completion queue* 确保处理器的架构状态的正确性，强制指令顺序完成和中断以及异常的正常处理。**POWER3** 采用两种机制来提高分支预测的准确性。首先，通过跟踪所有并发的带条件码的指令，处理器能够在指令分发时就确定分支结果。另外，对于在指令分发时无法确定的分支，会进行投机。当带条件码指令完成并且分支结果投机错误，分支指令之后所有指令会取消并重新分发正确指令。
* *Fixed-point execution units* **POWER3** 包含3个定点执行单元：2个单周期执行单元和一个多周期执行单元。单周期执行单元执行算术，移位，逻辑，比较，*trap* 和 *count leading zero* 指令。其他比如乘法，除法等都由多周期执行单元执行。两个单周期执行单元共享一个6条目的指令队列, 多周期执行单元使用一个3条目的指令队列。不同于**POWER 2** 包含两个执行定点和加载存储的对称的执行单元，**POWER3** 有两个专门的存储加载单元。独立的定点执行单元和存储加载单元对于类似Monte Carlo仿真这样整数操作占比大的应用很显然有比较大性能提升，但是即使是对浮点运算也很重要。像在*sparse-matrix-vector multiply* 中，整数索引必须先转换成字节偏移。
* *Floating-point execution units* *FPU* 包含两个对称的执行融合乘加流水的执行单元。所有浮点指令都需要经过乘法和加法阶段。对于浮点乘，加法阶段0作为一个操作数。对于浮点加，乘法阶段1作为一个操作数。
* *Load/store execution units* 所有的存储加载操作都由两个存储加载执行单元完成。加载指令将数据从内存转移到定点或浮点重命名寄存器，存储指令将数据从寄存器转移到内存。存储指令使用一个16条目*store buffer* 来提高性能. 当存储指令获取到数据之后就可以完成，而不必等到写进数据缓存。两个加载存储执行单元共享一个6条目的*指令队列*。 *LSU* 乱序执行允许加载指令超过存储指令同时记录数据依赖性。存储指令的顺序在执行阶段和存储队列中维护。
* *Data cache unit* **DCU** 主要由数据内存管理单元*MMU* ， L1缓存和数据预取单元组成
	* *Memory management unit* *MMU* 主要负责数据的地址转换，包括一个16条目 *segment lookaside buffer (SLB)* 和两个镜像的256条目两路组相联*data translation lookaside buffers (DTLB)* 以便支持两个存储加载执行单元。*MMU* 支持1T物理内存，64位有效地址和80位的虚拟地址。
	* *L1 data cache* L1数据缓存是单周期访问的64 KB，由4个bank，一共512个128B缓存行组成。每个bank是128路组相联。每个bank，由地址A55/A56决定，根据双字地址(A60)地址分成2个子bank。 *least-recently-used (LRU)* 替换算法不适用于128路组相联缓存，因此，**POWER3** 在L1指令缓存和L1数据缓存中采用*round-robin* 替换方案。为了适应不同位宽和频率以便重建缓存行，需要在传输路径上加入*linewidth buffers* 。L1数据缓存和BIU之间, 每个bank有一个*cache- reload buffers (CRBs)* 和一个*cache-storeback buffers (CSBs)*。 加载操作命中*CRB* 直接提供数据，不必等到缓存行加载。*CSB* 和 *BIU* 之间是64-byte 接口。L1数据缓存可以支持4个缓存缺失，可以有效掩盖内存延迟。当加载操作没有命中L1数据缓存，指令会被放入6条目的*load-miss queue(LMQ)* , 同时*BIU*发出数据加载传输， 后续加载指令可以继续执行。只有当第五个加载操作没有命中L1数据缓存，并且*LMQ* 里已经有4个未命中加载指令，加载操作才会被暂停，直到*LMQ* 里任意一个加载指令执行完成。
	* 数据预取 **POWER3**处理器一大创新是硬件数据预取功能。**POWER3**处理器根据监控到的缓存行缺失及匹配到的模式进行数据预取。当一个模式或数据流被探测到，处理器会对缓存行进行投机预取，假设数据会马上被使用。数据预取对于掩藏内存延迟非常关键。预取的数据流在地址上是连续的缓存行，要么是递增，要么是递减的。 **POWER3**处理器预取引擎包括一个10条目的*stream ﬁlter* 和一个 *stream prefetcher* 。 *stream ﬁlter* 观察所有数据缓存缺失的真实地址(RA)，检测潜在的预取数据流。*stream ﬁlter* 根据加载指令的地址操作数，猜测下一个缓存行的真实地址是增加还是减少，并记录到*FIFO ﬁlter* 。当发生新的缓存缺失时，如果发生缓存缺失的真实地址和猜测的地址一致，这样就检测到一个数据流。如果*stream prefetcher* 当前少于4个数据预取流，当前数据流会被接受，并且预测下一个缓存行加载操作会通过*BIU* 发出。一旦数据流被放入*stream prefetcher* , 数据流保持活跃直到数据流到达终点或者有新的缓存缺失进入*stream ﬁlter*。数据预取引擎会尝试保持预取2个缓存行，前一个缓存行会被放到L1 缓存，后一个缓存行会被放到BIU里一个预取缓冲区。因此，数据预取引擎可以并发预取4个数据流，每个2个缓存行，一共8个预取操作。数据预取引擎监控所有加载操作的地址操作数，当LSU结束当前缓存行并开始加载下一个缓存行，数据预取引擎将预取缓冲区数据传输到L1，并预取下一个缓存行到预取缓冲区。

下图展示了预取引擎框图
![Pasted image 20230912094139.png](/assets/images/power/Pasted image 20230912094139.png)

* *Bus interface unit* **BIU**提供*IPU*，*DCU*，预取引擎和L2缓存之间连接。数据接口位宽128位。
* *L2 cache* **POWER3** 支持1MB-16MB的L2缓存，可以是组相连或直接映射。总线和L2以32B位宽相连，一个128B缓存行需要4个周期传输。

# 6. POWER 4
**POWER 4** 是一个双核乱序处理器，通过一个*core interface unit(CIU)* 共享一个统一的片上二级缓存，下图展示了**POWER4** 芯片的逻辑框图:
![Pasted image 20230912171951.png](/assets/images/power/Pasted image 20230912171951.png)
* *CIU* 是3个L2控制器和2个处理器之间的*crossbar switch* 。每个L2控制器每周期能提供32B数据。*CIU* 通过一个8B的接口接受来自处理器的存储操作
* 每一个处理器有一个相关联的*noncacheable unit(NC unit)*，负责处理指令串行功能和非缓存存储加载操作
* L3控制器和目录在**POWER4** 片上，但是实际L3位于片外
* *fabric controller* 负责L2和L3之间的数据流以及**POWER 4** SMP通信
* *GX controller* 控制器负责IO

每个处理器芯片包括四种类型IO:
* 和同一个模块内其他**POWER4** 芯片通信，4个16-byte接口。物理是线上，这4个逻辑总线有6个总线组成，3个输入，3个输出
* 和不同模块的**POWER4** 芯片通信，2个8-byte的总线，一个输入，一个输出
* 片外L3接口，2个16-byte总线，一个输入，一个输出，运行在1/3处理器频率
* 2个4-byte GX总线, 一个输入，一个输出，运行在1/3处理器频率

同时 **POWER4** 增加了 __logical partition ID__ , __real mode offset register (RMOR)__ , __real mode limit register (RMLR)__ 和 __hypervisor RMOR__ 寄存器，提供将SMP系统分成逻辑分区的功能。这些寄存器只能在hypervisor模式下修改。逻辑分区可以提高系统的稳定性，并且可以用于虚拟化。下图展示了**POWER 4**的全芯片图：
![Pasted image 20230809154420.png](/assets/images/power/Pasted image 20230809154420.png)

## 6.1 POWER4 Core
**POWER 4** 处理器是一个乱序的超标量设计，一共实现了16级流水线，每周期可以取指8条指令，发射5条指令，完成5条指令，可以同时处理超过200条指令。为了增强指令级并行，**POWER 4**处理器拥有8个执行单元，2个相同的可以每周期执行融合乘加的浮点执行单元，2个存储加载单元，2个定点执行单元，一个分支执行单元和一个操作条件寄存器的执行单元。下图展示了**POWER4**处理器的框图
![Pasted image 20230912172922.png](/assets/images/power/Pasted image 20230912172922.png)

### 6.1.1 分支预测
**POWER4** 使用多级分支预测机制来预测条件分支是否发生。每周期直接相连的64KB的指令缓存提供8个指令，分支预测逻辑每周期可以查找两条分支指令。根据找到的分支类别，不同分支预测机制用来预测分支方向或分支目标地址。无条件分支的方向不做预测，所有条件分支都做预测，即使在取指阶段通过 __condition register__ 已知。对于 __branch-to-link-register (bclr)__ 和 __branch-to-count-register (bcctr)__ 指令，分支目标分别通过硬件实现的 *link stack* 和 *count cache*机制预测。 绝对和相对分支目标地址在分支指令扫描时候直接计算。POWER4使用3个分支历史表来预测分支方向：
* 第一个是本地预测器，类似于分支历史表BHT，使用分支指令地址来索引16 384条目的数组，产生1比特预测分支是否发生
* 第二个是全局预测器，通过一个执行过的分支指令的11比特向量，和分支指令地址进行按位异或，来索引16 384条目的全局历史表来产生1比特预测分支是否发生
* 第三个是选择表，记录上面两个预测器预测表现来选择其中一个预测器，16 384条目的选择表和全局预测器使用同样方式索引来产生1比特的选择信号。

动态分支预测能够被软件覆盖，通过设置条件分支指令的2个保留位，一个用来指示软件覆盖，一个用来预测方向。**POWER4** 使用 *link stack* 来预测 __branch-to-link__ 指令的目标地址，一般用于子函数返回。通过设置 __branch-to-link__ 指令的提示位，软件可以将 __branch-to-link__ 是否代表子函数返回，目标地址是否重复等信息通知处理器。当取指逻辑取到 __branch-and-link__ 指令并预测发生分支跳转时，处理器会将下一条指令的地址推入 *link stack* 。当取到一个提示位指示是子函数返回并预测发生分支跳转的 __branch-to-link__ 指令时，*link stack* 会被弹出，并从弹出的地址继续取指。 
**POWER4** 使用一个32-entry, tagless直接映射的 *count cache* 来预测软件提示的重复的目标。*count cache* 每一条能记录62位地址。当软件提示目标地址可重复的 __branch-to-link__ 和 __branch-to-count__ 指令执行时，目标地址被写入 *count cache* 。当再次取指这样指令时，目标地址通过 *count cache* 来预测。

### 6.1.1 取指
一旦 __instruction-fetch address register (IFAR)__ 被加载，指令缓存立即被访问，并每周期提供8条指令。每个指令缓存行是128B，可以提供32个指令，因此每个缓存行被分为4个相等的区域。因为指令缓存缺失比较少，为了省面积，指令缓存只有一个端口，每周期可以读或写一个区域。__I-cache directory (IDIR)__ 每条包含42位的真实地址(RA)，由有效地址访问(EA)。当I-cache缺失时，指令从L2以4个32B传输，最需要的区域在前两个周期传输。缓存行被写入*instruction-prefetch buffer*，指令缓存可以继续被后续指令访问。当取指逻辑不使用指令缓存时，例如发生另一个指令缓存访问缺失，缓存行会被写入指令缓存。这样指令缓存的写入操作可以被掩藏而不影响正常的取指操作。
EA, RA对被保存在128条目的2路组相联的 __effective-to-real address translation (ERAT)__ 表中。**POWER4** 分别实现了*IERAT*和*DERAT*，都是用有效地址(EA)访问。 每个处理器实现了一个1024条目的4路组相联的 *TLB* 。

当指令流水线准备好接受指令时，*IFAR*的值被发送到指令缓存, *IDIR*, *IERAT*, 和分支预测逻辑，同时IFAR被更新为下一个顺序区域的地址。下一个周期，指令从指令缓存转发到译码，分解和分组的指令队列，同时从*IDIR*接收真实地址(RA), 从*IERAT*接收有效地址EA, 真实地址RA对，以及分支方向预测信息。*IERAT*会被检查是否有有效的记录并且真实地址RA和*IDIR*的真实地址RA匹配。如果*IERAT*是无效的记录，EA必须从TLB和SLB进行翻译，取指会被暂停。假设*IERAT*记录是有效的，并且*IERAT*的RA和*IDIR*的真实地址RA匹配，指令缓存访问命中。使用分支预测逻辑重新加载IFAR，然后重复上面过程。填充译码，分解和分组前面的指令队列可以允许取指逻辑提前运行，而且当发生指令缓存缺失时，指令队列可以继续提供指令而不必停止后面的流水线。

如果发生指令缓存缺失，首先，*instruction-prefetch buffers*会被检查是否有请求的指令，如果有，会将指令发送到流水线并写入指令缓存。如果*instruction-prefetch buffer*也不存在请求的指令，取指命令被发送到L2，L2高优先级处理指令重载传输。当数据从L2返回，会尝试写入到I-cache。除了这些取指命令，**POWER4** 会预取指令缓存行到 *instruction-prefetch buffer*，*instruction-prefetch buffer*可以保存4个32条指令。 指令预取逻辑监控取指请求，当*instruction-prefetch buffer*里存在请求缓存行时，会预取下一个缓存行；当*instruction-prefetch buffer*里不存在请求缓存行时，会预取下两个缓存行。这些预取操作需要保证I-cache里不存在预取的缓存行。预取的缓存行会被保存在*instruction-prefetch buffer*里，从而保证不会污染指令缓存。

### 6.1.2 译码，分解和分组
一组包含5个内部指令，称为IOPs。解码阶段，指令被顺序放入一个组。最老的指令放到槽0， 余下依次放入，槽4保留给分支指令。如果必要，`no-ops` 被强制放入槽4。一个周期分发一个组的指令。组按照程序顺序分发，不同的IOPs被发射队列乱序发射到执行单元。一个周期只能完成一个组，并且只有一个组里所有指令都完成，并且更老的组已经完成，这个组才能完成。为了保证正确性，一些指令不允许投机执行，为了确保这些指令不会被投机执行，这些指令只有作为下一个完成的指令时才会被执行，这被称为完成序列化。为了简化实现，这些指令单独组成单指令组。完成序列化例子包括guarded space的存储加载指令和context synchronizing 指令，例如修改处理器状态的 __move-to-machinestate-register__ 指令。

### 6.1.3 组分发和指令发射
一次分发一个指令组到指令队列，当指令组分发时，控制信息被保存在 __group completion table (GCT)__ 。GCT可以保存20个组。GCT会记录指令组里第一条指令的地址。当指令执行结束，也会被记录到对应的GCT，记录会一直维护直到指令组退休。每个指令槽对应不同的执行单元的发射队列。定点执行单元和存储加载单元共享一个发射队列。下表列出了不同发射队列的深度和不同类型的队列的个数：
![Pasted image 20230913150523.png](/assets/images/power/Pasted image 20230913150523.png)

指令被分发到发射队列的顶端，当指令从发射队列发射出去后，队列内指令往下移动。对于两个发射队列对应一个执行单元的，两个队列交织发射，所有的源操作数都准备好且最老的指令被发射到执行单元。当指令组分发时，指令组需要的所有资源都需要可用，不然指令组会被停住直到所有资源可用。成功的分发需要下列资源：
* GCT条目：每一个指令组需要一个GCT条目，当组退休时会被释放
* 发射队列槽： 指令组里每个指令都需要一个对应的指令队列槽，当指令被成功发射到执行单元时会被释放。有些情况下指令发射几个周期之后才能释放，例如，依赖加载操作指令的定点指令可以被投机发射，这时候并不确定加载指令是否会命中L1数据缓存。如果加载指令未命中L1数据缓存，定点指令需要撤回到发射队列直到依赖的数据被加载到寄存器。. 
* 重命名寄存器：指令组里指令必须要有相应的重命名寄存器资源。重命名资源只有在下一个写同样逻辑资源指令完成时才释放。下表总结了 **POWER4** 处理器可用的重命名资源：

![Pasted image 20230913154714.png](/assets/images/power/Pasted image 20230913154714.png)

* __Load reorder queue (LRQ)__ 条目: 指令组里加载指令必须有可用的*LRQ*条目，当指令组完成时被释放。*LRQ*一共32条目
* __Store reorder queue (SRQ)__ 条目: 指令组里存储指令必须有可用的*SRQ*，当指令组完成，并且存储操作成功写到L2之后释放。*SRQ*一共32条目

### 6.1.4 存储加载执行单元
每一个*SRQ*条目关联一个*store data queue (SDQ)*条目，*SDQ*条目维护存储指令的数据直到指令组完成。一旦完成，*SDQ*里的数据会被写入到缓存。另外，三个特别hazards需要避免：
* *Load hit store* 当对同一个内存地址的年轻的读在更老的写把数据写回缓存之前执行时必须从SDQ获取数据。当读指令执行时，会检查SRQ，是否有对同一个地址的更老的写指令，如果有，数据从SDQ转发。当数据不能转发时比如读写地址重叠但是并不完全一致时，包含读指令的指令组被遗弃并重新取指执行。如果更老的写指令还未将数据写到SDQ，读指令被拒绝，并重新发射。
* *Store hit load* 当对同一个内存地址的年轻的读在知道有更老的写之前执行完时，读操作接收到数据是错误的。为了避免这种情况，当写指令执行时，需要检查*LRQ*，如果存在更年轻的读已经执行了，包含读指令的指令组和后续的指令组都被遗弃，并重新取指执行。写指令之后所有指令组也被遗弃。如果读写指令在同一个指令组，指令组被遗弃，并且指令组里的指令单独组成一个指令组。
* *Load hit load* 对同一个内存地址的读必须保证顺序。在*LRQ*条目里有一个位，指示对当前条目里的数据发生了侦听。当读指令执行时，和*LRQ*里的所有地址进行比较，当和一个被侦听过的更年轻的写匹配时，顺序一致性可能存在问题。为了简化实现，老的读指令之后的指令组都被遗弃。如果两个读指令在同一个指令组，当前指令组被遗弃，并且指令组里的指令单独组成指令组。

## 6.2 POWER4 Core Pipeline
下图展示了 **POWER 4**处理器的流水线:
![Pasted image 20230912173004.png](/assets/images/power/Pasted image 20230912173004.png)

在MP(mapper)阶段所有依赖性被确定，资源被分配，指令组被分发到对应的发射队列。在ISS阶段, IOP被发射到对应的执行单元。在RF阶段读取对应寄存器获取源操作数。在EX阶段执行。在WB阶段写回执行的结果到对应寄存器，这个时候，指令结束执行但还未完成。至少经过Xfer和CP两个周期，所有更老的指令组已经完成并且同一个组里其他指令结束执行指令才能完成。如果指令打包成指令组速度没有取指速度快，从指令缓存里取出的进入指令缓存的指令处于D1阶段。 同样，如果没有资源分发指令组到发射队列，指令等待在MP之前；指令在ISS之前在发射队列等待；在CP之前等待完成。

两个存储加载单元流水线是一样的，称为LD/ST流水线。 访问寄存器文件之后，存储加载指令在EA周期生成有效地址。加载指令在DC周期访问*DERAT*，数据缓存目录和数据缓存。如果*DERAT*未命中，加载指令被拒绝，保留在发射队列。同时请求会发送到TLB重新加载*DERAT*。第一次发射最少7个周期之后，被拒绝的指令会重新发射。如果*DERAT*仍然未命中，指令会再次被拒绝。这个过程一直持续直到*DERAT*命中。如果TLB也未命中，地址转换会被投机执行，但是TLB只有在指令确定执行之后才会更新。因此，只有在触发地址缺失的指令所在的指令组是下一个完成的指令组时才会更新TLB。TLB同时支持4 KB和16 MB页表。

对于加载指令，如果L1数据缓存目录指示包含数据，数据会在fmt周期进行格式化并写入到对应的寄存器，依赖的指令也可以使用数据。依赖的指令会假设数据缓存命中并发射，这样这些指令的RF周期和加载指令的WB周期对齐。 如果L1数据缓存提示未命中，请求会发送到L2获取对应缓存行。发送的请求会被保存在*load miss queue(LMQ)*中。*LMQ*可以保存8条请求。如果*LMQ*满了，加载指令会被拒绝并在7个周期之后重新发射。如果已经存在同一个缓存行的请求，第二个请求会被合并到同一个*LMQ*条目。如果这是同一个缓存行第三个请求，加载指令会被拒绝。所有从L2返回的数据都会和LMQ进行匹配，匹配到的数据会被转发到寄存器以便对应的加载操作可以完成，同时对应LMQ条目被释放。

对于存储指令，数据被保存到*SDQ*，一旦存储指令对应的指令组完成，会尝试将*SDQ*数据写到数据缓存。如果数据已经存在L1数据缓存，修改的数据会写回到数据缓存；如果不存在，不会写回到L1。修改的数据都会写回到L2。**POWER4** 的缓存一致性点是L2。另外，L2缓存对L1是包含的，即所有L1数据都在L2中。

## 6.3 POWER4 L2缓存
**POWER4** 上L2由两个处理器共享，下图展示了L2的逻辑视图：
![Pasted image 20230912173042.png](/assets/images/power/Pasted image 20230912173042.png)

L2由3个相同块组成，每一个都自己控制器。缓存行在3个控制器之间做哈希。每个块包括4块SRAM分区，每一个分区每两个周期能提供16B数据；4个分区每周期能提供32B数据，需要4个周期传输一个128B缓存行。数据阵列实现SECDED，并且有冗余的wordline和bitline；L2缓存目录由2个冗余的8路组相联，奇偶保护阵列组成。冗余的阵列除了提供备份，同时也提供了2个非阻塞的读端口，允许侦听而不影响存储加载请求。L2实现了pseudo-LRU替换算法。因为L1是写入设计，到L2的写请求最多8B，L2有2个4条目的64B的队列来合并写请求，减少到L2的请求。每个控制器里有4个一致性处理器来管理L2，每个一致性处理器处理一个请求。请求可能来自两个处理器的L1数据缓存或者取指，或者存储存储队列。一致性处理器负责:
* 负责命中时数据返回，或未命中时从fabric controller返回数据到CIU
* 更新L2目录
* 未命中时发送请求到fabric
* 当发生读未命中或写请求时，控制L2的写入
* 当一个处理器写的缓存行存在另一个处理器L1缓存时，通过CIU发送无效请求到处理器

每个L2控制器有4个侦听处理器负责管理从总线侦听到的一致性操作。当总线操作命中L2时，一个侦听处理器会负责做出相应操作。根据操作类型，L2目录里的包含位，缓存行的一致性状态，会导致：
* 发送无效请求到处理器L1数据缓存
* 从L2读取数据
* 更新缓存行在目录里的状态
* 将修改的数据写回到内存
* 将数据发送到其他L2

除了分配一个侦听处理器，L2对于所有侦听操作提供一个侦听响应。对于侦听请求，L2目录会被访问以确定缓存行是否存在以及对应的一致性状态，同时侦听的地址会和当前活跃的一致性处理器比较来发现可能的地址冲突。基于这些信息，会返回对应的侦听响应。
L2缓存控制器也充当两个处理器的保留站来支持 __load [double] word and reserve indexed (lwarx/ldarx)__ 和 __store [double] word conditional (stwcx/stdcx)__ 指令。每个处理器一个地址寄存器用来保存保留的地址。当 __lwarx__ 或 __ldarx__ 指令执行时会设置一个标志，当侦听到无效操作包括从其他处理器发送的对保留地址的写，或者 __stwcx__ 或 __stdcx__ 执行成功 (通过 __condition register__ 里一位来通知处理器执行结果)，会清除标志。L2实现增强的MESI一致性协议，一共7个状态：
* I (invalid state): 数据无效
* SL (shared state): 数据有效，缓存行可能在其他L2。数据可以传输到同一个MCM内的其他L2。当处理器L1数据缓存加载或指令未命中L2并且数据来自其他缓存或内存时会进入 __SL__ 状态
* S (shared state): 数据有效，缓存行可能在其他L2。数据不可以传输给其他L2。当来自同一个MCM的处理器发出的侦听读命中时进入该状态
* M (modified state): 数据有效，数据被修改且独占，数据可以传输给任意L2。当处理器发出写操作时进入该状态。
* Me (exclusive state): 数据没有修改但被独占，__Me__ 状态缓存行写出只需要无效对应标签，数据不需要写出。当处理器执行 __lwarx__ 或 __ldarx__ 指令并且缓存行时从内存获取时候会进入该状态。
* Mu (unsolicited modified state): 数据有效，数据被修改且独占，当处理器执行 __lwarx__ 或 __ldarx__ 指令并且缓存行时从其他处于 __M__ 状态L2获取时会进入该状态。
* T (tagged state): 数据有效，但是和内存比有修改，并且数据已经传输给其他缓存，在这个状态，数据不能传输给其他L2除非收到回应。当 __M__ 状态时收到侦听读时候会进入该状态。

下表列出了L2和L1可能对应的缓存状态:
![Pasted image 20230913172331.png](/assets/images/power/Pasted image 20230913172331.png)

L2系统里还有两个*noncacheable units (NCU)*， 分别对应两个处理器。NCU处理分缓存的读写，以及缓存同步操作。每个NCU由*NCU master* 和 *NCU snooper*组成。
* *NCU master*负责来自处理器的请求，包含一个深度为4的FIFO队列，处理非缓存的写，包括memory-mapped I/O的写，和缓存以及内存屏障操作。一个深度为1的队列来处理非缓存的读操作。 片上处理器的缓存和同步操作和非缓存的写操作一样处理，不同的是不带数据。这些操作会发送到L2控制器，大部分会被L2侦听到，包括 __icbi__ , __tlbie__ ,  __translation lookaside buffer synchronize (tlbsync)__ , __enforce in-order execution of I/O (eieio)__ , __synchronize (sync)__ , __page table entry synchronize (ptesync)__ , __lsync__ , __data cache block flush (dcbf)__ , __data cache block invalidate (dcbi)__ 指令。
* *NCU snooper*处理来自总线的 __translation lookaside buffer invalidate entry (tlbie)__ 和 __instruction cache block invalidate (icbi)__ 。*NCU snooper*侦听来自总线的 __sync__ , __ptesync__ , __lsync__ , __eieio__ , __icbi__ 和 __tlbie__ 操作，并传递给处理器。

## 6.4 POWER4 L3 Cache
下图展示了L3的逻辑视图：
![Pasted image 20230912173114.png](/assets/images/power/Pasted image 20230912173114.png)

L3由控制器和数据阵列组成，控制器在**POWER4** 芯片上，包含标签目录，仲裁逻辑等。数据阵列在包含两个16MB eDRAM的L3芯片上，后面可以连接单独的内存控制器。为了物理实现和减少bank冲突，L3上的eDRAM组织成8个bank， 每个bank 2M，并分成4个4M的组。L3控制器也分成4个组，每个有两个一致性处理器，可以处理来自总线的请求，L3和内存的访问，以及更新L3标签目录。另外，每个组包含两个处理器来进行内存的踢出，无效操作和IO的DMA操作。  每两个组共享一个L3标签目录。L3是8路组相联，缓存行512B，并以128B大小和L2维持一致性。每个缓存行支持下面5个状态：
 * I (invalid state): 数据无效
 * S (shared state): 数据有效，L3只能传输给L2
 * T (tagged state): 数据有效，并且和内存数据相比是修改过的，数据可能存在其他L2或L3里
 * Trem (remote tagged state): 和T状态类似，但是数据是从其他芯片传输过来的
 * O (prefetch data state): 数据和内存里数据一致，其他L2或L3状态未知

L3缓存数据要么来自直接连接的内存，或者连接的其他处理器芯片的内存。当其中一个处理器的读请求未命中L3时，L3控制器分配一个S状态的缓存行，和L1以及L2的包含性不是必须的。因此，当L3释放数据时，并不需要无效L1或L2缓存。当本地L2弹出M或T状态数据时，L3数据进入*T*或*Trem*状态。当收到侦听时，L3进入*T*或*Trem*状态，这样当L3写出时候. 不需要做内存地址译码。L3使用*T/Trem*来决定数据是否可以写入到直接连接的内存控制器，或者需要将写出操作发送到总线。处在*T*或*Trem*状态时,L3可以给系统上任何请求提供数据。但是在S状态时, L3只能为本地L2提供数据。这样减少片间数据传输，尽量使用本地L3。当处在O状态时, 数据可以传输给任意请求者。

## 6.5 POWER4 Memory System
下图展示了**POWER4**的内存系统视图：
![Pasted image 20230912173158.png](/assets/images/power/Pasted image 20230912173158.png)

每个**POWER4** 芯片有一个可选的内存控制器连接到L3缓存后面，每个内存控制器有1或2个内存接口。每个芯片有2个8-byte总线，一个输入，一个输出。*system memory interface (SMI)*芯片和DRAM芯片由4个4-byte 双向总线连接，工作在400MHz内存控制器有一个64条目的读命令缓存，65条目的写命令缓存和一个16条目的写缓存队列。

## 6.6 POWER4 IO
下图展示了 **POWER4** 的IO结构，GX总线连接远程IO桥芯片。
![Pasted image 20230913093144.png](/assets/images/power/Pasted image 20230913093144.png)

## 6.7 MCM
下图展示了4个 **POWER4** 芯片通过4个逻辑总线组成8路SMP系统。
![Pasted image 20230913100716.png](/assets/images/power/Pasted image 20230913100716.png)

每个芯片写到独立的总线，并和L2，I/O控制器，L3控制器进行仲裁。每个芯片侦听所有的总线，从L2发出的读请求被所有芯片侦听到: 
* 请求数据是否在L2，一致性状态是否允许从L2传输到请求的芯片
* 请求数据是否在L3或内存，如果在，返回数据到请求的芯片

从单芯片角度看，互联拓扑类似总线；从整体看，单个芯片类似交换机。

## 6.8 32路SMP
下图展示了4个MCMs组成更大的SMP系统
![Pasted image 20230913092752.png](/assets/images/power/Pasted image 20230913092752.png)

一到四个MCMs可以进行互联，当多个MCMs互联时，中间的芯片的总线充当repeaters，将请求以环状拓扑从一个模块传输到另一个模块。和单个MCM类似，每个芯片有单独的请求/命令和数据总线，但是侦听所有总线。多个MCMs配置比单个会轻微增加10%的的内存访问延迟，但是提供了统一的内存模型，简化编程。

## 6.9 POWER4 RAS
L1缓存使用奇偶效验保护，L1数据缓存错误会被当成同步的*machine-check*中断。 *machine-check*中断处理函数实现在固件代码里以便支持错误恢复，当中断发生时，固件会保存处理器的架构状态，并检查处理器寄存器来决定恢复和错误状态。如果是可恢复的，固件会无效L1数据缓存来清除错误，并递增错误计数器。如果数据计数器比预设的阈值大，表示是一个不可恢复的错误，固件会禁止发生错误的L1数据缓存部分。固件然后恢复处理器架构状态，以fully recovered的状态回调操作系统的machine-check处理函数。操作系统检查固件返回的状态并继续执行。
L3 标签目录是ECC保护，支持SECDED。不可修正的错误会导致*system checkstop*。如果是可修正的错误，访问会被暂停直到错误被修正，同时会发送recovered attention message到service processor。
L3地址, 内存地址和控制总线有parity，可以发现单bit的错误。L3和内存的数据总线支持ECC的SECDED。不可修正的错误会被发送到requesting processor并导致machine-check中断。
L3 标签目录发生stuck fault或L3 cache-embedded DRAMs发生超过 line-delete控制寄存器范围的stuck faults，包含对应L3缓存的处理器芯片可以被重新配置，从逻辑上在系统里删掉而不影响系统里其他L3缓存。

# 7. POWER 5
**POWER5** 将L3直连到L2，作为victim cache，另外，**POWER5** 还集成了片上内存控制器，以提高主存访问速度。每个处理器核支持双线程，对于操作系统，**POWER5** 是一个4路SMP处理器。两个处理器共享1.875MB L2 缓存，并分为3个分区，每一个分区是10路组相联。 L3分为3个slice，分别作为L2的victim cache，每个slice是12路组相联，缓存行大小为256B，分为2个128B的sector。**POWER5** 芯片和L3通过2个16B的双向总线以1/2处理器速度相连。4 bytes宽的I/O bus称为GX bus，运行在1/3处理频率。下图展示了**POWER 5**的全芯片图：
![Pasted image 20230914085010.png](/assets/images/power/Pasted image 20230914085010.png)

## 7.1 多线程的演进
将超标量微处理器改成SMT需要做下面这些修改：
* 增加 __Program Counter (PC)__ 共享取指逻辑
* 增加*GPR/FPR*重命名资源，高地址位可用来识别不同线程
* 增加完成逻辑来记录不同线程
* 地址和标签中增加线程区别位

下图展示了从ST到Coarse Grain Thread, Fine Grain Thread和Simultaneous Multi Thread的演进：
![Pasted image 20230809154930.png](/assets/images/power/Pasted image 20230809154930.png)

在**POWER5**处理器中, 为了实现SMT，相比**POWER 4**：
* 段表缓存在全相联的64个条目的 __segment lookaside buffer (SLB)__ , 每个线程一个
* 页表缓存在1,024个条目, 4路组相联的 __translation lookaside buffer (TLB)__
* BHT和 __count cache__ 没有因为SMT而修改
* *return address stack*，每个线程一个
* 4个 __instruction prefetch buffers__ ，每个线程一半，每个线程可以独立处理指令缓存缺失和指令预取
* *GCT*修改为 *linked list* 实现，这样每个线程可以独立分配和释放。共享的*GCT*保持和**POWER4**一样的条目数
* 寄存器重命名有一些小修改，每个逻辑寄存器有一个线程位，每种重命名资源增加一些
* 除了浮点发射队列从20增加到24， 其他发射队列保持不变
* 对于*LRQ*和*SRQ*，**POWER4**是32个条目，对于SMT，需要分成每个线程一个，这样会导致*LRQ*和*SRQ*容易用完。为了不增加队列物理大小，每一个队列都提供了32个虚拟队列条目。虚拟队列条目包含足够信息来分辨指令而不是存储加载指令地址和数据。这种方案以很小成本扩展了LRQ和SRQ大小，从而不会停止指令分发。当指令分发时，真实的LRQ或SRQ已满，指令可以分发到虚拟条目。当真实条目可用时，虚拟条目转换成真实条目。在存储加载指令发射时必须使用真实的*LRQ*和*SRQ*
* __Branch Issue Queue(BIQ)__ 保持和**POWER4**一样的16条目大小，每个线程8条
* __load miss queue (LMQ)__ 增加了一个线程位，有2个线程共享

下表总结了重命名寄存器和发射队列的大小比较：
![Pasted image 20230914112838.png](/assets/images/power/Pasted image 20230914112838.png)

下图展示了**POWER5**处理器的逻辑视图：
![Pasted image 20230804094425.png](/assets/images/power/Pasted image 20230804094425.png)

## 7.2 多线程指令流和流水线
**POWER5**指令流水线和**POWER4**一样。所有流水线延迟，包括分支预测错误惩罚，L1数据缓存命中延迟都和**POWER4**一致。**POWER5**流水线如下图所示：
![Pasted image 20230914085221.png](/assets/images/power/Pasted image 20230914085221.png)

在SMT模式下, 每个线程一个程序计数器。取指之后不同线程的指令被放到不同的指令缓存（D1之前)，每个指令缓存可以存放24条指令，比**POWER4**单个指令缓存稍小。在D0阶段，基于线程的优先级，从一个指令缓存中取5条指令，并且在D1和D3阶段组成一个指令组。一个指令组中的指令全部来自同一个线程，并同时解码。下图展示了**POWER5**的指令流程图：
![Pasted image 20230914085156.png](/assets/images/power/Pasted image 20230914085156.png)

### 7.2.1 多线程的调度
**POWER5** 处理器对每个线程，支持8种软件优先级。Level 0表示线程没有激活，Level 1到Level 7优先级从低到高。**POWER5**支持两种ST操作
* 默认启动工作在SMT模式，当某个线程空闲时，软件可以指导硬件将线程进入休眠状态。要唤醒一个休眠的线程，要么活跃的线程执行一个特殊的指令，要么外部或递减中断发生在休眠线程上。
* 当线程进入null状态时，操作系统将无法感知该线程。

多线程支持如下两种调度方式：
* **动态资源平衡** 动态资源平衡的目标是保证两个线程流畅的执行。动态资源平衡通过监控类似GCT和LMQ来决定一个线程是否占用资源。例如，如果一个线程遭遇很多L2缓存缺失，依赖指令可能充满了发射队列，阻止了其他指令组的分发，拖慢了其他线程。为了避免这种情况，动态资源平衡逻辑发现线程超过L2缓存缺失阈值之后可以停止该线程。 动态资源平衡逻辑有下面三种手段停止线程：
	* **降低线程优先级** 当线程用了超过预定的GCT条目时使用
	* **阻止线程解码** 当线程发生超过阈值的L2缺失时使用
	* **遗弃线程所有等待分发的指令并阻止线程解码** 当线程执行类似 __synch__ 这种耗时很长指令时使用
* **可调线程优先级** 可调优先级可以让软件决定线程的优先级，一般下列情况需要调整线程优先级：
	* 等待锁的线程 这种情况软件可以降低线程优先级
	* 线程处于空闲状态
	* 某个线程需要比其他跑的快， 例如，软件可以给实时任务相对后台任务更高的优先级

## 7.3 POWER5 Memory System
内存读数据位宽时16 bytes ，写位宽是8 bytes。POWER5使用 __synchronous memory interface (SMI)__ 来连接DDR-I或DDR-II SDRAM。
* 当使用2个SMI芯片时，每个SMI芯片在控制器侧配置成8-byte的读和2-byte写, DIMMs侧则是2个独立的8-byte端口
* 当使用4个SMI芯片时，每个SMI芯片在控制器侧配置成4-byte的读和2-byte写DIMMs侧则是2个独立的8-byte端口

SMI芯片内部有缓冲来匹配控制器和DIMMs之间频率和位宽的差异。下图展示了内存控制系统的逻辑视图：
![Pasted image 20230914090521.png](/assets/images/power/Pasted image 20230914090521.png)


## 7.4 POWER5 SMP
__fabric bus controller (FBC)__ 控制片间的通信，互联拓扑称为分布式交换，在不同配置下保持一样的行为。高端**POWER5**系统采用MCM封装，基本单元是8个**POWER5** 芯片和8个L3芯片封装成两个MCM。MCMs, 内存, 和连接I/O drawers的桥接芯片组成一个book。下图展示了16路**POWER5**系统：
![Pasted image 20230914090918.png](/assets/images/power/Pasted image 20230914090918.png)

**POWER5** 在MCM内部是通过两个环状总线互联，两个环方向相反。每个总线是8B位宽，运行在处理器频率。一个Book内两个MCM通过4对8B位宽，运行在1/2处理器频率的单向总线互联。一个**POWER5** book是16路SMP，对于SMT2，就是32路SMP。4个books可以互联组成64路SMP系统。每个Book中的**POWER5**芯片和边上芯片通过运行在1/2处理器频率的8B位宽总线互联。下图展示了64路SMP互联示意图：
![Pasted image 20230914091104.png](/assets/images/power/Pasted image 20230914091104.png)

也可以基于DCM来组成1-8个**POWER5**芯片系统。DCM包含一个**POWER5**芯片和一个L3芯片。下图展示了一个基于DCM的16路**POWER5** SMP系统：
![Pasted image 20230914091243.png](/assets/images/power/Pasted image 20230914091243.png)

### 7.4.1 Fabric bus controller
FBC对L2/L3, 内存控制器，MCM的片内总线和缓冲和MCM的片间总线传输进行缓冲和保序。总线地址和数据是分开的，允许乱序返回。
* **地址总线** MCM内部的地址总线是8 bytes位宽，运行在1/2处理器频率，MCM之间地址总线是4 bytes宽，运行在处理器频率，每个地址传输使用4个处理器周期。地址总线使用ECC，支持SECDED。地址传输包括50位真实地址(RA)，地址标签，传输类型和其他信息。MCM内部有12个点对点的地址总线，MCM之间总线使用环拓扑来广播地址，在64路SMP系统中，地址串行经过8个MCM。环上每个芯片负责转发地址到环上的下一个MCM， 并且到MCM内部其他芯片。当发起传输的芯片收到发出的地址之后，停止传播。
* **响应总线** 类似地址总线，MCM内部的响应总线运行在1/2处理器频率.，MCM之间响应总线运行在处理器频率。响应总线是延迟之后的地址总线， 包括每个处理器芯片上内存系统侦听到地址之后缓存一致性相关的操作。响应总线使用奇偶效验保护。响应总线在地址总线之后延迟固定的周期。一旦MCM内部源芯片收到MCM内部其他3个芯片的侦听响应，就将这些响应，自身响应以及来自环前面MCM的响应结合起来并转发到环上下一个MCM。当发起侦听的芯片收到自己发起的侦听响应之后，会生成一个合并的响应并像地址一样广播到系统中。 这个结合的响应详细列出了相应地址上应该做出的动作，比如应该进入什么状态，对于读操作应该哪一个芯片提供数据，那些必须做无效操作等。为了减少cache-to-cache传输延迟，**POWER5**增加了一个early combined response机制，允许remote chip收到地址之后从自己处于S状态的L2或L3发送数据到请求处理器 。early combined response通过和MCM侦听响应来比较找到第一个可以提供数据的一致性侦听者。
* **数据总线** 数据总线用于所有数据传输，比如缓存介入。也用于地址传输之后的数据传输，比如cast-outs, snoop pushes, 和DMA写。 数据总线使用ECC，支持SECDED。在**POWER4**系统上, 数据使用简单的环拓扑进行广播传输。在**POWER5**中，一个MCM内部的数据总线增加到8个，MCM之间也增加到8个，另外，fabric bus路由数据到指定的芯片。

## 7.5 POWER5 RAS
**POWER5**系统上，很多固件能直接升级而不需要重启或不影响系统运行。所有系统互连都实现ECC保护，单bit互联错误会动态修正，如果错误持续，会调度修复程序。对于不可修复的错误，包含错误的部分会被下线而不需要人工干预。内存使用SECDED，并且后台会做memory scrubbing来修正错误。并且每个内存区有一个额外的DRAM可以透明的替换出错的DRAM。

# 8. POWER 6
**POWER6** 是一个双核，每核双线程的顺序处理器，增加了虚拟化，十进制算术运算，向量多媒体运算功能，并且实现了检查点重试和 处理器冗余功能。*recovery unit (RU)* 保存处理器的状态，并且使用ECC保护，当发生错误时，处理器使用这些状态恢复。下图展示了**POWER6**的全芯片图:
![Pasted image 20230726195941.png](/assets/images/power/Pasted image 20230726195941.png)

## 8.1 POWER6 Core流水线
**POWER6**处理器流水线分为*instruction fetch pipe*, *instruction dispatch pipe*, *Floating point pipe*, *BR/FX/load pipe*, 和*Checkpoint recovery pipe*组成，下图展示了**POWER6**处理器的流水图：
![Pasted image 20230915084655.png](/assets/images/power/Pasted image 20230915084655.png)

来自L2的指令在P1到P4阶段进行预解码，然后写入L1。分支预测使用2位16K条目的分支历史表(BHT)来预测分支方向。每周期L1提供8条指令并传输到指令解码流水线的指令缓冲，每个线程有64条目的指令缓冲。所有线程的指令合并成一个指令组，每个指令组包含7条指令。之后会被分发到不同的发射队列，并发射到对应的执行单元执行。十进制和向量多媒体扩展指令通过 __浮点发射队列(FPQ)__ 进行发射并在十进制和向量多媒体指令单元执行。 执行单元的结果送到 __checkpoint recovery (CR) pipe__ 并保存到ECC保护的缓冲中，等待恢复。定点存储加载指令按顺序执行，浮点指令可以和加载指令及定点指令并行执行。为了减少内存延迟影响，会投机预取执行指令预取数据到L1数据缓存，投机预取指令到L1指令缓存，为浮点加载指令增加一个 __load-data buffer__ ，硬件stride预取，软件预取。为了优化一些指令的执行延迟，在分发阶段和执行阶段之间增加了一些缓冲阶段：
* 为了在加载指令和依赖的定点指令之间实现一个周期的加载使用，定点指令在执行之前会额外多两个周期
* 为了避免预测错误导致的分支延迟，分支指令额外增加两周期来和定点指令对齐
* 为了在加载指令和依赖的浮点指令之间实现零周期的加载使用，浮点指令在FPQ会额外多六个周期

### 8.1.1 IFU
IFU从L2加载指令到L1，这些请求要么是L1指令缓存缺失发起，要么是连续两个缓存行加载请求之后导致的指令预取。每个取指请求都是一个缓存行128B，数据以4个32B返回，最多能有32个取指请求从处理器发送到L2。从L1缓存发出的指令会被写到IDU里的指令缓冲I-buffer。**POWER6**和**POWER4**和**POWER5**一个主要差异是把很多译码和分组的功能放到预译码阶段，因此解决了L1指令缓存到指令分发的关键路径。在将指令写入L1指令缓存之前，需要4个周期的预译码和分组，增加了L2的访问延迟。**IFU**主要负责：
* **指令重编码** 在预译码阶段会记录指令信息，例如会调整一些指令的寄存器域以便指令对于执行单元保持一致并减少关键路径上逻辑。**POWER6**上最重要的指令记录发生在分支指令上，相对分支指令包含一个立即数域，是相对当前 __PC__ 的偏移值，指定了分支目标地址。预译码时会将偏移值和 __PC__ 的低位相加，消除了关键分支执行路径上的加法器。预译码也会决定 __PC__ 高位是否需要加，减来计算分支目标地址，这些信息都会编码进记录的分支指令里。绝对分支指令的分支目标地址也可以做符号扩展。条件分支指令也会编码以实现高频。 
* **指令分组** 在预译码阶段指令会被分组并且记录到L1指令缓存，分组可以简化在关键分发路径上的分发仲裁。 同一个指令组只有同一个线程的指令，组里指令条数和分发带宽以及执行单元资源匹配，一个线程5条指令。执行的分支指令结束一个组。L1指令缓存里的每个指令有一个开始位控制在IDU里的分组。如果指令开始为被设置，指令开始一个新的组。 指令组里指令要满足：
	* 一个指令组不能使用超过**POWER6**执行单元资源：两个FXUs, 两个LSUs, 两个FP或VMX和一个分支单元
	* 一个指令组不能有对 __GPR__ , __FPR__ , __CR__ , __XER__ 的*write-after-write*依赖
	* 一个指令组不能有对 __XER__ 的*read-after-write*依赖
	* 一个指令组不能有对 __GPR__ 的*read-after-write*依赖
	* 一个指令组不能有对 __CR__ 的*read-after-write*依赖
	* 一个指令组有读目标 __FPR__ 的指令跟随FMA指令
	* 一个指令组不能有对同一个 __FPR__ 的浮点存储指令跟随浮点加载指令
* 分支执行 **POWER6**每周期能预测8条分支指令，分支预测使用16K条目2位的BHT; 8个条目全相联的 __count cache__ ; 和一个6条目的 __link stack__ ，__BHT__ 和 __count cache__ 由两个线程共享。 每个线程可以由10条分支等待。

### 8.1.2 IDU
IDU负责指令分发，跟踪，发射和完成。在分发阶段，一个线程的指令组里的所有指令一起分发。如果两个线程的指令数不超过可用的执行单元，两个线程可以同时分发。**POWER6**每个线程能分发5条指令，两个线程一共7条指令。为了提高分发带宽，IDU使用两个并行的指令数据流，一个线程一个。每个线程有一个每周期能从指令缓存接收8条指令的I-buffer，每个线程每次最多从I-buffers取5条指令。每个线程指令然后经过非FPU和VMX依赖性跟踪，如果指令组里的指令依赖性解决，指令组被分发。否则，指令组被停在分发阶段直到解决依赖性。非FPU和VMX指令的依赖性使用目标表跟踪，一个线程一个。目标表保存指令当前在流水线里的位置信息，当指令从I-buffer进入到到分发阶段，这些指令的信息写入到目标表。随后的定点指令访问目标表来获取依赖数据，然后被分发。FPU和VMX指令的依赖性由分发阶段后面的FPQ解决。FPU和VMX算术指令被分发到FPQ，每个FPQ能保存8个指令组，每个组可以有2个FPU或VMX算术指令，FPQ每周期发射2个FPU或VMX算术指令。为了在加载指令和依赖的浮点指令之间实现零周期的加载使用，浮点指令在FPQ会额外多六个周期，以便和来自LSU的加载数据对齐。如果加载的数据不能写入FPR，数据被写入32条目的 __load target buffer__ 。__load target buffer__ 允许每个线程有16个加载指令在浮点算术指令之前执行，从而消除额外6个浮点流水周期的影响。 IDU使用一个完成表来记录执行中的大量的指令，每个线程使用10条目的完成表，每个条目可以记录32个指令的信息。当遇到要执行的分支时，预测执行的分支指令之后新分配一个完成表条目。

### 8.1.3 FXU
**POWER6**处理器实现了两个FXU来处理定点指令并为LSU生成地址。大部分定点指令一个周期执行完成，**POWER6**处理器FXU可以实现依赖指令背靠背执行而不需要转发数据到依赖的指令。

### 8.1.4 BFU
**POWER6**处理器有两个BFU，为了对齐执行周期差异，额外增加一个流水周期。和FXU类似，中间结果可以转发给依赖指令而不必做rounding和normalization。

### 8.1.5 LSU
**LSU**由存储加载地址生成和执行逻辑，L1数据缓存，地址转换，*store queue*, *load miss queue (LMQ)*, 和数据预取引擎组成，实现下面功能：
* **存储加载执行** 为了实现**POWER6**高频设计，LSU执行单元采用了相对简单的数据流，保存最少的状态。大部分存储加载指令执行一个操作，一个硬件状态机用来处理 __load/store multiple and string__ 指令。对于跨缓存行的操作，内部分成2个操作。
* **L1数据缓存** **POWER6**处理器包含8路组相联的64KB的L1数据缓存，缓存行大小是128B，由4个32B的区组成。L2数据以32B大小发送，并且缓存行的无效操作也以32B大小处理，加载操作能命中部分有效的缓存行。L1数据缓存由2个端口，可以支持2个读或一个写。缓存行重加载拥有最高的优先级，并且阻塞已经分发的存储加载指令。执行的加载指令拥有第二高优先级。最后如果没有缓存行重新加载或加载指令，完成的存储指令能够从存储队列写到L1数据缓存。 L1数据缓存是store-through，所有的写操作发送到L2 cache。
* **组预测** 为了满足访问路径的周期, 实现了一个组预测。 组预测基于有效地址EA并且可以用作一个小目录来从8个L1数据缓存组里选择。虽然L1数据缓存的目录也可以使用，但是需要ERAT之后的真实地址，要花费更多时间。 组预测使用EA(51:56)索引，8路组相联，每个组或条目包含11个EA哈希位, 2个有效位（每个线程一个）和一个奇偶位,11位EA哈希值使用下面公式生成: `(EA(32:39) XOR EA(40:47)) + EA(48:50)` 。当读缓存时，生成的EA(51:56)来索引组预测, 并且生成11位哈希值和8个组的内容比较，如果匹配并且对应线程有效位是置位，使用预测的信号访问L1数据缓存。

## 8.2 POWER6一致性协议
**POWER5**中使用的广播侦听协议中，一致性传输需要的带宽随着系统规模而增长。因此，基于目录的NUMA (nonuniform memory access)方案更有吸引力，基于目录的协议使用目录来指示内存区域的归属，因此限制广播在一个小的节点内，减少了节点外的传输。**POWER6**开发了基于目录的NUMA方式的广播协议。下表总结了**POWER6**的一致性协议中的缓存状态：
![Pasted image 20230915100402.png](/assets/images/power/Pasted image 20230915100402.png)
![Pasted image 20230915100432.png](/assets/images/power/Pasted image 20230915100432.png)
下表总结了缓存状态对应的scope-state：
![Pasted image 20230915103531.png](/assets/images/power/Pasted image 20230915103531.png)
scope-state位保存在冗余的内存中，对于128B的缓存行，该位指示缓存行是否是本地使用。该scope-state位和数据一起读写，4个新增加的缓存状态提供了一种缓存scope-state的方法。当缓存的scope state被释放时，通常是写回到内存。当scope state隐含全局状态时写回操作时必须的，当scope state是本地时，写回操作是可选的。
scope-state和4个新的缓存状态降低了基于目录的NUMA的成本，并且可以和广播一致性协议集成。**POWER6**使用很多预测器来预测一致性请求是本地还是全局广播操作。

## 8.3 POWER6缓存层次
**POWER6**处理器核包含64KB L1 I-cache和64-KB L1 D-cache分别在IFU和LSU中。每个处理器核还有4-MB L2缓存，片上两个处理器核共享32-MB L3缓存， L3缓存控制器在片上，但是数据阵列在片外，L3缓存行是128B，16路组相联。下表总结了各级缓存的特性和组织形式：
![Pasted image 20230914170540.png](/assets/images/power/Pasted image 20230914170540.png)
4-MB L2缓存是8路组相联，每个缓存行128B，并且和**POWER5**上L2缓存实现三个独立控制器不同，**POWER6**上是一个控制器。L2缓存数据阵列分成4个间隔区，每个包含一个缓存行的32B。处理器发出的缓存行读，castout读和介入读操作使用4个间隔区，但是read-modify-write和L2重新加载时writeback只使用一个缓存行的一个或多个32B的区。目录和一致性管理逻辑分成两个地址哈希区，每个目录区每两个处理器周期能接受一个处理器请求，或一个侦听请求或一根更新操作。不同于**POWER5**上的L2使用不同的读端口处理处理器请求和侦听请求，**POWER6** L2使用滑动窗口来调度侦察请求而不和处理器请求冲突，从而减少面积和功耗。

因为L1 D-cache是store-through，L2每个目录区使用一个8条目128B宽的队列来累积处理器请求。一个条目累积的所有写操作都被L2处理成read-modify-write操作，只影响使用到的间隔区。L2使用每个目录区的16个read/claim (RC) machines中的一个来处理处理器发出的所有读和L2存储队列发出的read-modify-write操作。RC machines 管理所有处理器发出的缓存读写操作。如果L2 cache缺失,L2会等待L3 cache的响应。如果L3命中，数据返回到L2。如果L3未命中，L2 cache会发出请求到SMP一致性互联总线，数据最终通过SMP一致性互联总线从其他L2, L3返回。如果新的缓存行需要释放L2中缓存行，L2会使用每个目录区4个castout machines中一个来将数据或状态转移到L3 victim cache或内存。
为了处理SMP一致性互联总线上的侦听请求，L2首先查询目录，如果需要响应，使用每个目录区4个snoop machines中一个来处理。snoop machine操作包括从缓存读取数据分发送到SMP一致性互联总线，更新目录状态或者从缓存读取数据发送给内存，并无效目录状态，或者只是更新或无效目录状态。
片上两个L2之间有一个高速cache-to-cache接口来提高两个缓存的传输延迟。处理器发出的读操作首先检查自己的L2目录，同时转发到另一个L2，如果自己L2缺失而另一个L2命中，数据会从该L2转发到处理器核和L2。 

L3分为两个目录区，分别对应两个L2，L3使用每个目录区8个read machines中的一个来处理L2发送过来的读操作。如果命中，数据返回L2和处理器核，否则，发送响应到对应的L2指示需要发送请求到SMP一致性互联总线。L3使用每个目录区8个write machines中的一个来处理L2发送过来的castout写操作。如果新的缓存行需要释放已有的缓存行，L3使用和write machine关联的castout machine来将数据写回内存。对于SMP一致性互联总线的侦察请求，L3首先查询目录，如果需要响应，L3使用每个目录区4个snoop machine中的一个来处理。

## 8.4 POWER6内存系统
**POWER6**有两个内存控制器，每个支持4个通道，每个通道支持2B读数据接口，1B 写数据接口和一个运行在4倍DRAM频率的命令接口，最快支持800-MHz DDR2 DRAM。每个通道可以支持1到4个缓冲芯片。

## 8.5 POWER6 I/O系统
**POWER6** I/O控制器和**POWER4**以及**POWER5**保持一致，一个4B读接口和4B写接口与I/O hub芯片相连，这些接口运行在1/2处理器频率。

## 8.6 POWER6 SMP
下图展示了**POWER6**的SMP互联接口
![Pasted image 20230726201124.png](/assets/images/power/Pasted image 20230726201124.png)
4个**POWER6**芯片组成一个基本互联单元Node，每个**POWER6**有5个运行在1/2处理器频率的8B位宽的SMP接口，三个专门用于Node内部互联。如下图所示：
![Pasted image 20230914163617.png](/assets/images/power/Pasted image 20230914163617.png)
依赖于一致性协议上的创新，一致性总线和数据时分复用同一个物理链路，可以67%带宽分配给数据，33%分配给一致性，或者50%分配给数据，50%分配给一致性。 在一个Node内，4个**POWER6**组成全连接网络。

不同于之前的环拓扑，**POWER6**可以使用8个Node组成的全连接拓扑，如下图所示：
![Pasted image 20230914163644.png](/assets/images/power/Pasted image 20230914163644.png)

## 8.7 POWER6 RAS
**POWER6**处理器内部RU包含处理器状态备份，处理器内部的状态检查点不断保存在RU里，而且数据使用ECC保护。L1使用奇偶，L2和L3使用ECC保护。当发生错误时，处理器被停止并阻止和外部通信，上一次成功执行的指令的检查点数据被从RU里读出并恢复到对应处理器上，L1会被清空。处理器从恢复的检查点继续执行。如果是无法修复错误，RU里的数据会被传输到系统中空闲的处理器上并继续执行。

# 9. POWER 7
**POWER7**芯片有8个处理器核，每个核有12个执行单元，并且支持SMT4。**POWER 7**芯片有两个内存控制器，每个内存控制器支持4通道的DDR3，一共8通道内存可以提供100GB/s带宽。芯片上下是SMP接口，提供360 GB/s一致性带宽，可以支持**POWER7**扩展到32片。下图展示了**POWER7**芯片的全芯片图:
![Pasted image 20230915155142.png](/assets/images/power/Pasted image 20230915155142.png)

## 9.1POWER7 Core
**POWER7**能够动态在ST, SMT2和SMT4之间切换。处理器核主要包含6个单元：
* 取指单元IFU
* instruction-sequencing unit (ISU)
* LSU
* FXU
* VSU
* 十进制FPU (DFU)

IFU包含32KB指令缓存, LSU包含32KB数据缓存, 共享256KB L2缓存。每周期能取8条指令，解码和分发6条指令，发射和执行8条指令。每个处理器核有12执行单元：2个定点单元，2个LSU, 4个双精度浮点单元，一个向量单元，一个分支单元，一个条件寄存器单元和一个十进制浮点单元DFP。两个LSU能执行简单的定点操作，每个浮点单元可以执行双精度乘加运算，一共8FLOPS。**POWER6**首次引入的DFU可以加速很多商业应用。下图展示了处理器核的规划图：
![Pasted image 20230915155118.png](/assets/images/power/Pasted image 20230915155118.png)
下图展示了**POWER7**的指令流视图：
![Pasted image 20230915155003.png](/assets/images/power/Pasted image 20230915155003.png)
为了减少面积和功耗，SMT4设计采用了partition方法，一对线程使用一套物理GPR并且使用一个FXU和LSU，另一对线程使用另一套物理GPR和另一个FXU和LSU。**POWER7**能够使用比支持SMT2的**POWER5**更少的物理GPR来对SMT4进行寄存器重命名。在**POWER4**和**POWER5**的乱序处理器中, GPR, FPR和VR的寄存器重命名结构是分开的。在**POWER7**中, 这些都合并成了一个80条目的寄存器重命名结构，匹配分发和完成之间并发的非分支指令数量。另外，在早期处理器中，浮点和定点还有存储加载指令的发射队列都是分开的，但是，在**POWER7**中, 这些队列都合并成一个，叫做统一发射队列UQ。为了实现高频，UQ物理上分成两个24条目的队列UQ0和UQ1。
**POWER6**里FPU和VMX是分开的，但是在**POWER7**里, 这两个合并成了 __vector and scalar unit (VSU)__ , 并实现了 __vector and scalar extension (VSX)__ 架构，使用64个128位的架构寄存器来实现两路 __single-instruction multiple-data (SIMD)__ 。**POWER7**相比**POWER6**没有增加发射端口但是仍然支持新的VSX指令集并且每周期能执行4个浮点运算，8FLOPS。

L1数据和指令缓存使用bank设计，允许读写同时访问，一个bank同一个周期能支持2个读或一个写。

一个**POWER7**处理器核和L2及本地L3组成一个chiplet，并处于单独电源域，和相连接的PowerBus异步。

### 9.1.1 POWER7取指和译码
下图展示了**POWER7**取指和译码逻辑及流水线：
![Pasted image 20230915154836.png](/assets/images/power/Pasted image 20230915154836.png)
**POWER7**处理器核有一个4路组相联32-KB的L1指令缓存，使用了16路bank来避免读写冲突。组预测使用64条目的 __instruction effective address directory (IEADIR)__ 。__I-cache directory (IDIR)__ 同时被访问并在下一个周期确认组预测的正确性。一个64条目的 __instruction effective-to-real-address translation table(IERAT)__ 被用来加速地址转换。__IERAT__ 的前32个条目支持线程0和2，后32个条目支持线程1和2，__IERAT__ 支持4K，64K页表，并根据需要将64 MB和16 GB页表映射成64K。

IFU从L2中取指并写入L1指令缓存，每次取指请求返回4个32B。这些请求要么是L1指令缓存缺失，要么是指令预取。对每一个L1指令缺失请求，预取引擎发出额外两个顺序缓存行请求。4个线程的取指请求都是独立的，处理器核最多发出4笔并发请求。只有 ST和SMT2 支持指令预取。
指令在写入L1指令缓存之前经过两个周期进行预译码并生成奇偶校验，预译码的位用来扫描发生的分支跳转，帮助分组和几个异常情况的处理。 分支指令扫描过程中会对分支指令进行修改以便生成目标地址，修改过的包含部分计算出的目标地址的分支指令，会被保存到L1数据缓存。来自L2的32B数据在3个周期之后写入到L1指令缓存，如果线程正在等待这些指令，它们同时也会被送到 __instruction buffers (IBUF)__ 和分支扫描逻辑。

__Instruction fetch address registers (IFARs)__ 记录每个线程的 __program counter__ ，每周期 __IFAR__ 选择一个线程的PC并作为取指地址送到指令缓存和分支预测逻辑。指令缓存每周期可以提供8条指令，并写入 __IBUF__ , 随后被组装成分发组。线程优先级，缓存缺失，__IBUF__ 满和线程资源平衡都会影响线程的调度。 IBUF有20个条目，每个可保存4条指令。在SMT4模式，每个线程有5个条目，在ST和SMT2 模式，每个线程有10个条目。特殊的线程优先级逻辑每个周期选择一个线程进行分组，从线程对应的IBUF里读取最多4个非分支指令和2个分支指令并组成一个组。和**POWER4**和**POWER5**不同，分支指令不会结束一个组。分组之后，指令要么被解码，要么被送到微码硬件将复杂指令拆分成一系列内部简单操作。

分支的方向使用分支历史表来预测，由8K条目的本地BHT (LBHT)，16K条目的全局BHT(GBHT)和8K条目的全局选择器(GSEL) 组成。这些预测器每周期可以对8条分支指令进行分支预测，由所有活跃线程共享。LBHT由10位取指地址索引，GBHT和GSEL由取指地址和每个线程单独的21位的 __global history vector (GHV)__ 哈希成11位的值索引。GSEL条目里的值用来选择LBHT和GBHT作为分支预测结果。所有BHT条目都是2比特, 高位决定预测的方向，低位提供hysteresis。

分支的目标地址使用下面两个机制预测：
1. 非子函数返回的间接分支指令使用所有线程共享的128条目的 __count cache__ , __count cache__ 使用取指地址和GHV异或生成7位值索引，__count cache__ 每个条目包含62位预测地址和2位置信位。当预测结果不准确时，置信位用来决定何时进行条目的替换。 
2. 子函数返回的分支使用每线程一个的 __link stack__ 来预测, __branch-and-link__ 指令被扫描到时，下一个指令的地址被放入对应线程的 __link stack__ 。当扫描到 __branch-to-link__ 指令时，__link stack__ 会被弹出。当扫描到 __branch-and-link__ 但是被之前预测错误的分支指令被刷掉时，__link stack__ 允许保存一个投机条目。在ST和SMT2模式，每个线程使用16条目的 __link stack__ ，在SMT4模式， 每个线程使用8条目的 __link stack__ 。

在ST模式，当遇到发生的分支时3个流水周期的分支扫描会产生处于空闲，为了消除这种惩罚，增加了一个 __BTAC__ 来记录直接分支的目标地址。__BTAC__ 使用当前取指的地址来预测两个周期之后的取指地址，如果预测正确，发生分支时BTAC能够提供取指地址。当条件分支用来跳过一系列的定点或存储加载指令且分支很难预测时，**POWER7** 可以检测到这样指令，并将其从指令流水线中移除，然后根据条件来执行定点或存储加载指令。条件分支被转换成内部操作，而定点或存储加载指令依赖内部操作。这样阻止了可能的错误的分支预测导致的刷流水线的操作。

### 9.1.2 POWER7 ISU
下图展示了负责分发指令，重命名寄存器，发射指令，完成指令和处理异常的ISU的逻辑视图：
![Pasted image 20230915154927.png](/assets/images/power/Pasted image 20230915154927.png)
在ST和SMT2模式，两个物理的GPR保存相同内容，线程的指令可以分发到任意一个子发射队列UQ0或UQ1，通过每周期轮换子发射队列保证两个子发射队列的负载平衡。在SMT4模式，两个物理的GPR单独使用，线程T0和T1的定点和存储加载指令被分发到UQ0, 使用GPR0, 发射到FX0和LS0； 线程T2和T3的定点和存储加载指令被分发到UQ1, 使用GPR1, 发射到FX1和LS1。

大部分VSU指令可以分发到UQ0或UQ1，不管是在ST, SMT2或SMT4模式，除了:
1. VMX的浮点以及整型指令必须分发到UQ0
2. permute (PM), 十进制浮点, 和128位存储指令必须分发到UQ1
3. 分发到UQ0的VSU指令在VS0执行
4. 分发到UQ1的VSU指令在VS1执行

**POWER7**处理器核以组作为分发单位，一次一个线程的指令组。在指令送到发射队列之前，使用mapper逻辑对寄存器重命名，**GPR**, **VSR**，**XER, **CR**, **FPSCR**，**link**, 和**count**寄存器都会被重命名。**GPR**和**VSR**共享80个重命名寄存器，**CR**使用56个物理寄存器，**XER**使用40个物理寄存器，**Link**和**Count**寄存器使用24个物理寄存器，**FPSCR**使用一个20条目的缓冲来保存每个指令组的**FPSCR**状态。这些资源都是独立并被所有线程共享的。更新多个目的寄存器的指令被拆分成子指令。

 ISU分配 __load tag (LTAG)__ 和 __store tag (STAG)__ 来管理存储加载指令，**LTAG**是一个指向分配给加载指令的 __load-reorder-queue (LRQ)__ 条目的指针；**STAG**是一个指向分配给存储指令的  __store-reorder-queue (SRQ)__ 条目的指针；当耗尽物理的**SRQ**/**LRQ**条目后，会使用虚拟的**STAG**/**LTAG**来减少对指令分发的阻塞。当物理的**LRQ**被释放之后，虚拟的**LTAG**会被转换成真实的**LTAG**；当物理的**SRQ**被释放之后，虚拟的**STAG**会被转换成真实的**STAG**。虚拟的**STAG**或**LTAG**只有转换成真实之后才会被LSU执行。ISU可以为每个线程分配63个**LTAG**和63个**STAG**。

**POWER7**使用三个发射队列：一个48条目的UQ, 一个12条目的 __branch issue queue (BRQ)__ , 和一个8条目的 __CR queue (CRQ)__ 。分发的指令被保存在发射队列，BRQ或CRQ一个周期后发射到执行单元；UQ两个个周期后发射到执行单元。BRQ和CRQ是移动队列，分发的指令被放入队头，然后往队尾移动；为了降低功耗，UQ是非移动队列，而是移动指向队列位置的指针。发射队列可以顺序或乱序发射指令，已经准备好的老的指令有更高的优先级。 当指令的源操作数都可用之后就可以被发射了，对于存储加载指令STAG和LTAGA需要是真实条目。对于BRQ和CRQ, 通过将重命名后的目的物理指针和所有并发的指令的源物理指针比较来检查指令依赖性；对于UQ, 通过一个依赖矩阵来记录队列指针来检查依赖性。发射队列每周期一共可以发射8条指令，包括一条分支指令，一条条件寄存器逻辑操作指令，两个发射到FXU的定点指令，两个发射到LSU的存储加载指令或简单的定点指令，两个发射到VSU的向量标量指令。

BRQ每周期可以接收2条分支指令，并发射一个分支指令到IFU执行；CRQ每周期接收2条条件寄存器逻辑指令或SPR操作指令，并发射一个指令到IFU；UQ一共有48条目，物理上分成2个24条目，包括所有FXU, LSU, VSU, 和DFU执行的指令。队列上半部分包含FX0, LS0, 和VS0(含VMX整数指令)执行的指令；下半部分包含FX1, LS1, 和VS1(含DFP, VMX PM, 和VSU 128位存储指令)执行的指令，UQ0/UQ1分周期分别能接收4条指令。64位的VSU存储指令在分发阶段拆分成地址生成(AGEN)和数据操作。因为UQ是非移动的队列，指令的相对年龄通过一个age matrix决定。UQ0/UQ1每周期分别可以一条定点指令，一条存储加载指令和一个向量标量指令，一共6条指令。指令可以投机发射，比如当定点指令依赖于加载指令，而加载指令还未确定是否数据缓存或 __data effective-to-real-address translation (D-ERAT)__ 命中。当投机错误时，指令被拒绝并在几个周期之后再次发射。简单的定点指令也可以选择发射到LSU从而提高FXU的吞吐。

ISU还负责记录和完成指令，使用一个 __global completion table (GCT)__ 来记录所有分发后的指令，按指令组记录，分发和完成都以指令组为单位。GCT有20条目，所有线程共享，一个条目记录一个指令组，最多包括4个非分支指令和2个分支指令。因此，GCT最多可以记录120条分发后的指令。GCT条目里的指令组里每个指令都有一个结束位，当指令成功执行之后，结束位会被标记。当一个指令组里所有指令都标记结束且指令组是线程里最老的指令组时，指令组就可以完成了；指令组完成时，指令组里所有指令的结果都在架构上可见，__completion group tag (GTAG)__ 被广播到相关联资源以便被释放。 **POWER7** 每周期每线程对可以完成一个指令组(T0和T2是一对，T1和T3是一对)，每周期一共2个指令组。

ISU还需要处理刷流水线操作，像分支预测错误，存储加载执行发生冒险，上下文同步指令或异常处理，都可能需要将投机指令从流水线刷掉。完成单元使用20位的掩码来完成刷流水线操作，每一位对应一个组；同时当分支预测错误的分支指令不是组里第一条指令，需要从线程里刷掉后续指令组时对部分组刷流水线操作会发出 __GTAG__ ，以及4位槽掩码指示是哪些指令。

### 9.1.3 LSU微架构
LSU有两个对称的LS0和LS1流水线，每一个能够一个周期执行存储加载指令。LSU组要功能单元有：存储加载AGEN和执行, __SRQ__ , __store data queue (SDQ)__ , __LRQ__ , __load miss queue (LMQ)__, 包括*D-ERAT* , *ERAT miss queue*, *segment lookaside buffer (SLB)*和TLB的地址转换单元, 包括组预测和数据目录(DDIR)的L1数据缓存和数据预取 __prefetch request queue (PRQ)__ 引擎。下图展示了LSU的微架构：
![Pasted image 20230915154725.png](/assets/images/power/Pasted image 20230915154725.png)
* 存储加载执行 在ST和SMT2模式时，存储加载指令可以在LS0或LS1执行；在SMT4模式，线程T0和T1的指令在LS0流水线执行，线程T2和T3的指令在LS1流水线执行。存储指令发射两次，AGEN发射到LSU, 数据操作发射到FXU或VSU。LSU主要数据流包括32B的L2的重加载总线接口，到L2的16B的写数据接口，到VSU的16B加载数据接口，来自VSU的16B的写数据接口，来自FXU的8B写数据接口。**POWER7** 的L1数据缓存只有32 KB, 减少了数据缓存访问延迟。定点加载指令只有2个周期的加载使用延迟，在加载指令和依赖的定点指令之间只有一个流水线气泡；VSU加载指令有3个周期的加载使用延迟，在加载指令和依赖的VSU指令之间会产生两个流水线气泡。每个LSU流水线也可以执行定点加和逻辑指令，提高了指令发射的灵活性
* 存储加载顺序 尽管指令乱序发射和执行，LSU必须保证存储加载指令遵守架构上的编程顺序。LSU主要依赖SRQ和LRQ来实现编程顺序。SRQ是一个32条目，基于真实地址RA的 __content-addressable memory (CAM)__ 。每个线程有64虚拟条目，允许分发64个并发写；可以发射32个并发的写；SRQ由所有线程共享。SRQ条目在存储指令发射时分配，当数据写到L1数据缓存或发送到L2时释放。每个SRQ条目都有一个对应的16B的SDQ条目，存储指令每周期可以发送最多16B数据到L2。写数据可以转发给后面的读指令，即使读写指令都是投机执行。LRQ也是一个32条目，基于真实地址RA的 __content-addressable memory (CAM)__ 。每个线程有64虚拟条目，允许分发64个并发写；可以发射32个并发的写；LRQ由所有线程共享。LRQ记录读操作，检查是否有同一个地址的年轻的读在更老的读或写之前执行的冒险。当检查到这样冒险发生时，LRQ会将年轻的读和后面同一个线程的指令从流水线刷掉，加载指令重新取指并执行
* 地址转换 指令执行时，EA经过由两个64条目的DERAT缓存和一个64条目的IERAT组成第一级地址转换逻辑转换成46位的真实地址。如果ERAT未命中，由每个线程32条目的SLB和所有线程共享的512条目的TLB组成的第二级地址转换逻辑进行转换。有效地址EA首先使用线段表转换成68位虚拟地址，然后68位虚拟地址经过 __page frame table__ 转换成46位真实地址。**POWER7** 支持256 MB和1 TB两种大小的段, 以及4 KB, 64 KB, 16 MB和16 GB四种页表大小。
	* D-ERAT是基于CAM的全相联的缓存，物理上分成两个，每个LSU流水线一个。在ST和SMT2模式, 两个D-ERAT保持同步；在SMT4模式，T0和1共享一和LS0关联的个64条目的D-ERAT，T2和T3共享一和LS1关联的个64条目的D-ERAT。每个D-ERAT条目支持 4-KB, 64-KB或16-MB页表， 16 GB保存为多个16-MB页表。D-ERAT采用LRU替换算法
	* SLB是每线程32条目，全相联的基于CAM的缓冲，每个SLB条目支持256 MB或1 TB段. **POWER7**支持 __multiple pages per segment (MPSS)__ 扩展；MPSS使一个4KB基础页表的段可以同时有4-KB, 64-KB, 和16-MB页表；一个64KB基础页表的段可以同时有64-KB, 和16-MB页表。SLB由操作系统管理，当SLB缺失时发出数据或指令的段中断
	* TLB是512条目，4路组相联的缓冲，TLB由硬件管理并使用真LRU替换算法。TLB缺失时最多支持2个并发table-walk， TLB支持hit-under-miss功能，即在做table-walk时支持访问。**POWER7**处理器里, 每个TLB条目都有 __logical partition (LPAR) identity__ 标签，TLB条目里的 __LPAR identity__ 处理器里的 __LPAR identity__ 匹配
* L1数据缓存 **POWER7**里L1数据缓存时8路组相联32-KB，每个128B缓存行由4个32B组成。L2有一个专门的32B的重加载数据接口，每周期可以传输32B数据，缓存行可以以32B为单位进行命中。L1数据缓存有两个用于指令加载的读端口和一个用于存储指令或缓存行重加载的写端口。缓存行重加载的写有最高优先级，读的优先级最低。L1数据缓存物理上是4个，每个由16个bank组成，缓存的bank可以支持一个周期同时有一个写和两个读，除非是同一个bank。如果读发生和写发生bank冲突，加载指令被拒绝并重新发射。L1数据缓存是store-through，所有的写都发到L2，而不需要cast-out；写缺失时不会分配L1，而是发到L2；L1包含于L2。L1一个32B区支持字节写，最多16B。L1由有效地址EA索引，L1 D-cache目录使用LRU替换算法， 8路组相联，一共32 KB。 组预测用来减少L1数据缓存命中延迟，使用一个小目录来预测包含数据的组；组预测阵列时8路组相联，使用EA(52:56)索引，每个条目包含EA(33:51)，每个线程有效位和奇偶位哈希得到的11位结果。当读缓存时，生成的EA(52:56)来索引组预测, 并且生成11位哈希值和8个组的内容比较，如果匹配并且对应线程有效位是置位，使用预测的信号访问L1数据缓存。否则，发生缓存缺失
* 加载缺失处理 未命中L1的加载指令发出缓存行重加载请求，并释放发射队列条目，在**LMQ**中创建一个条目来记录缓存行的加载，同时支持将数据转发到目的寄存器。**LMQ**有8个条目，并使用真实地址，在所有线程间共享。**LMQ**支持读合并，最多两个加载指令可以关联到同一个条目，**LMQ**最多支持8个缓存缺失

### 9.1.4 POWER7 FXU
FXU有两个相同的FX0和FX1组成，下图展示了FX0的逻辑视图:
![Pasted image 20230915154552.png](/assets/images/power/Pasted image 20230915154552.png)
每个FXU由一个多端口的**GPR**，一个执行加减，比较和陷阱的ALU，一个执行旋转，移位和选择的rotator，一个count (CNT) leading zeros单元，一个执行按位permute的bit-select unit (BSU)，一个除法器，一个乘法器，和一个执行计数，奇偶，BCD指令的*miscellaneous execution unit (MXU)*组成。
大部分定点指令在一个周期执行完成，并且同一个流水线中的依赖的指令可以背靠背执行。FXU中GPR有112个，最多用于两个线程的架构寄存器和重命名寄存器；并有4个读端口，其中两个为定点单元提供操作数，另两个为相连的LSU的AGEN提供操作数；两个物理写端口每周期操作两次，实现逻辑上4个写端口，两个接收定点单元的结果，另两个接收相连的LSU的数据缓存的数据。在 ST和SMT2模式，两个GPR的内容保持一致，而SMT4则分别用于两对线程对。 FXU的比较单元是专门定制的，能够比ALU更快计算出比较指令里的条件码，因此，允许比较指令和分支指令背靠背执行。

### 9.1.4 VSU
**POWER7** VSU实现了VSX架构，引入了64个架构寄存器。另外，VSU还合并了VMX，BFU和DFU。下图展示了VSU逻辑视图：
![Pasted image 20230918150731.png](/assets/images/power/Pasted image 20230918150731.png)

## 9.2 POWER7 缓存层次
**POWER7**有一个共享的32-MB L3缓存，由8个处理器本地的4-MB L3组成。eDRAM只有SRAM1/5功耗, 而且只占1/3面积。下图展示了**POWER7**的缓存层次图:
![Pasted image 20230915155341.png](/assets/images/power/Pasted image 20230915155341.png)
**POWER7** L2和L3缓存支持同样的13个缓存状态，和**POWER6**缓存状态移植。尽管没有增加新的缓存状态，增加了新的一致性操作。其中一个时缓存注入，I/O设备的direct memory access (DMA)写操作目的地可能是缓存，如果一个处理器核的L2或本地L3拥有缓存行 (M, ME, MU, T, TE, TN, 或TEN状态), 数据会被写到本地L3。另外部分受害缓存管理策略，用于L2和本地L3数据传输，可以减少功耗。下表总结了**POWER7**的缓存状态：
![Pasted image 20230915154407.png](/assets/images/power/Pasted image 20230915154407.png)
**POWER5**和**POWER6**中的 __barrier synchronization register (BSR)__  功能在**POWER7**中被虚拟化了。每个系统里，多个MB的主存空间可以被分为BSR存储区间，BSR提供了并行任务之间低延迟的同步。对BSR区间的写会立即广播到所有的读对象，允许指定的主线程低延迟的编排工作线程，这个功能对HPC里并行计算很有用。 
* **L2缓存** 一个全相连16深度，每个32B的存储缓存用来接收处理器核发出的写，4个32B的条目组成128B可以一起分发到L2
* **L3缓存** 每个4-MB L3由32个eDRAM组成，eDRAM比6T SRAM稍慢。L3区域控制器提供了一个处理器核和L2的分发端口，一个L3分发端口，两个侦听分发端口（分别对应奇偶缓存行）。L2未命中的访问会通过处理器核和L2的分发端口访问本地4-MB L3区域，如果命中则由8个read machine中一个管理。预取和一些L2写缺失也会通过处理器核和L2的分发端口访问L3，这些由24轻量级RC machine中一个处理。每个预取的数据由10个write machine处理，每个write machine有一个关联的cast-out machine来管理由于写而导致的cast-out。本地L3未命中的访问会被广播到一致性总线，并被内存，其他L2，其他L3和其他7个远端的4-MB L3侦听；在28M远端L3命中的请求通过侦听分发端口访问。

## 9.3 POWER7 SMP
**POWER7**采用了和**POWER6**同样的基于非阻塞广播的一致性协议，使用分布式管理*relaxed-order-optimized multiscope*。片上一致性互联通过芯片中间的奇偶仲裁逻辑来路由一致性请求的奇偶缓存行，每个周期可以授予一个even请求和一个odd请求。 一旦请求被授予，请求被广播到芯片左右两边的奇偶侦听总线。需要广播到其他芯片的请求被路由到芯片上下的多片SMP互联。从其他芯片来的请求被奇偶仲裁逻辑管理并广播到侦听总线。 
下图展示了4个**POWER7**组成的Node：
![Pasted image 20230918164544.png](/assets/images/power/Pasted image 20230918164544.png)
每个芯片有4个10B的Node内SMP互联接口，位于芯片顶端。

下图展示了由8个Node组成的SMP系统：
![Pasted image 20230725175956.png](/assets/images/power/Pasted image 20230725175956.png)
每个芯片有2个10B的node外SMP互联接口，位于芯片底部。每个Node一共8个Node外互联，使用其中7个可以组成一个8Node的SMP系统。不同于**POWER6**，数据和一致性传输带宽可以动态分配。**POWER6**引入的投机本地范围一致性广播被广泛使用，例如，将256路SMP系统分成8个Node区域可以使每个32路区域独享SMP互联和侦听带宽。

## 9.4 POWER7 RAS
下图展示了**POWER7**的RAS特性：
![Pasted image 20230918170442.png](/assets/images/power/Pasted image 20230918170442.png)
当错误被处理器核发现和上报之后, **POWER7**处理器核迅速阻止所有指令的完成，以及取指核分发。如果错误条件不足以产生checkstop, 处理器核发起恢复流程。回复流程刷掉流水线里所有的指令，并将处理器核恢复到错误发生之前的架构状态，将处理器核和系统隔离，进行自检，清除和测试SRAM，复位所有逻辑单元，读写GPR和VSR寄存器来使用ECC改正任何单比特错误。然后从恢复状态开始重新执行。

像GPR和VSR大的寄存器使用ECC保护，其他小的寄存器奇偶校验; 所有SRAM使用ECC并支持SECDEC；指令缓存使用奇偶校验。另外，**POWER7**处理器核对各种SPRs和核配置寄存器使用RAS硬化的锁存器。 L1数据缓存使用奇偶校验。当执行加载指令从L1数据缓存读数据并发现奇偶错误时会触发硬件恢复，加载指令会被刷掉，L1数据缓存内容无效，从出错的存储指令所在指令组重新开始执行。当L1数据缓存，目录或组预测阵列发生无法修正错误时，会使用组删除机制来阻止使用出错的组。

# 10. POWER 8
**POWER8**芯片有12个SMT8处理器核，每个处理器核可以分成4个区，每个分区支持2个硬件线程；处理器核每周期可以取8条指令，解码并分发8条指令，发射和执行10条指令，并完成 8条指令。**POWER8**还实现了硬件的TM (Transactional Memory)。每个**POWER8**处理器有2个DDR3/DDR4内存控制器，48条PCIE Gen3通道，512K的L2缓存，96M的L3缓存，以及内存缓冲芯片里的128M的L4。下图展示了**POWER8**处理器的全芯片规划图：
![Pasted image 20230718154255.png](/assets/images/power/Pasted image 20230718154255.png)

## 10.1 POWER 8 Core
**POWER8**处理器核由6部分组成：*instruction fetch unit (IFU)*, *instruction sequencing unit (ISU)*, *load-store unit (LSU)*, *fixed-point unit (FXU)*, *vector and scalar unit (VSU)* 和*decimal floating point unit (DFU)*。取指单元IFU里有32 KB指令缓存，存储加载单元LSU里有64 KB数据缓存，共用512 KB的L2缓存。下图展示了**POWER8**处理器核的规划图：
![Pasted image 20230718155035.png](/assets/images/power/Pasted image 20230718155035.png)
**POWER8**处理器核有16个执行流水线：2个定点流水线，2个存储加载流水线，2个加载流水线，4个双精度浮点流水线（或8个单精度浮点流水线），2个可以执行VMX和VSX的对称的向量流水线，一个加密流水线，一个分支流水线，一个条件寄存器逻辑流水线和一个十进制浮点流水线。 2个存储加载流水线和2个加载流水线可以执行简单的定点指令；4个双精度浮点流水线可以执行双精度乘加运算，可以每周期处理8个双精度，16个单精度，另外还可以执行64位的SIMD指令。

### 10.1.1 POWER8 Core指令流和流水线
下图展示了**POWER8**处理器核的指令流视图：
![Pasted image 20230918172554.png](/assets/images/power/Pasted image 20230918172554.png)
指令从内存系统经过解码，分发和发射队列，然后送到功能单元执行，除了分支和条件寄存器逻辑指令，大部分指令都被分发到 __Unified Issue Queue (UniQueue)__ ，由对称两半UQ0和UQ1组成；GPR和VSR物理上也分为两部分，GPR0和GPR1，VSR0和VSR1；GPR0和VSR0用于UQ0，GPR1和VSR1用于UQ1。相应的定点，浮点，向量，存储加载流水线也分成两部分FX0, FP0, VSX0, VMX0, L0, LS0和FX1, FP1, VSX1, VMX1, L1, LS1，分别对应UQ0和UQ1。在单线程模式，GPR和VSR的两部分是一样的值，指令可以分发到UQ0或UQ1，根据指令类型每周期轮换分发的发射队列来保证负载平衡。在SMT(SMT2, SMT4, SMT8)模式，线程分成两个组，每个组对应一个发射队列和相应的执行流水线。偶数线程 (T0, T2, T4, T6)的定点，浮点，向量和加载存储指令只能使用UQ0，访问GPR0和VSR0，在FX0, LS0, L0, FP0, VSX0, 和VMX0流水线上执行；奇数线程(T1, T3, T5, T7)的定点，浮点，向量和加载存储指令只能使用UQ1, 访问GPR1和VSR1, 在FX1, LS1, L1, FP1, VSX1, 和VMX1流水线上执行。加密和十进制浮点指令也只能发射到线程对应的发射队列，但是在同一个流水线上执行。分支和条件寄存器逻辑指令有自己专门的发射队列，并由所有线程共享。
下图展示了**POWER8**处理器核的流水线：
![Pasted image 20230919095043.png](/assets/images/power/Pasted image 20230919095043.png)
和**POWER7**最大的不同是是分发由两个流水线组成。

### 10.1.2 POWER8 IFU
为了支持SMT8和 __LPARs (logical partitions)__ 需要修改IFU里的很多资源：
* 首先，指令缓存对齐的提高导致了更高的取指数量
* 第二，改善的分支预测提高了分支目标和方向的预测结果
* 第三，指令分组的提高允许一个分发指令组包含更多的指令
* 第四，提高了指令地址转换的命中率
* 第五，实现了指令聚合来提高一些指令序列的性能
* 最后，更好的流水线冒险避免机制减少了刷流水线次数

下图展示了**POWER8**的IFU的逻辑视图和流水线：
![Pasted image 20230918172656.png](/assets/images/power/Pasted image 20230918172656.png)

IFU主要负责：
* **取指和预译码** **POWER8**有8路组相联32 KB的L1指令缓存，使用16路bank避免读写冲突。一个256条目的 __Instruction Effective Address Directory (IEAD)__ 来对组实现预测，指令缓存目录同时被访问以在下个周期确定组预测正确性。指令缓存能够在128B的缓存行内以16B为单位索引。一个64条目全相联的 __ Instruction Effective to Real Address translation Table (IERAT)__ 用来加速地址转换，__IERAT__ 由全部线程共享， __IERAT__ 支持 4 KB, 64 KB, 和16 MB的页表，其他页表可以使用多个相邻的小叶表实现。IFU从L2中取指并写入L1指令缓存，每次取指请求返回4个32B。这些请求要么是L1指令缓存缺失，要么是指令预取。对每一个L1指令缺失请求，预取引擎发出额外顺序缓存行请求。线程的取指请求都是独立的，处理器核最多发出8笔并发请求。 只有 ST，SMT2和SMT4 支持指令预取。ST模式时预取3个顺序缓存行，在SMT2和SMT4预取1个缓存行，SMT8不做预取以便节省带宽。预取不一定会被完成，取决于带宽情况。指令在写入L1指令缓存之前经过两个周期进行预译码并生成奇偶校验，预译码的位用来扫描发生的分支跳转，帮助分组和几个异常情况的处理。 分支指令扫描过程中会对分支指令进行修改以便生成目标地址，修改过的包含部分计算出的目标地址的分支指令，会被保存到L1数据缓存。来自L2的32B数据在3个周期之后写入到L1指令缓存，如果线程正在等待这些指令，它们同时也会被送到*instruction buffers (IBUF)*和分支扫描逻辑。__Instruction Fetch Address Registers (IFARs)__ 记录每个线程的 __program counter__ ,每周期 IFAR 选择一个线程的PC并作为取指地址送到指令缓存和分支预测逻辑。指令缓存每周期可以提供8条指令，并写入 IBUF , 随后被组装成分发组。线程优先级，缓存缺失，IBUF 满和线程资源平衡都会影响线程的调度
* **分组** 指令会被同时送到分支扫描逻辑和指令缓冲IBUF，IBUF有32条目，每个4条指令；在SMT8模式，每个线程有4条目，在SMT4模式，每个线程有8条目；SMT2和ST模式，每个线程有16个条目。指令被分成组，特殊的线程优先级逻辑每个周期选择一个线程进行分组，在ST模式下，从线程对应的IBUF里读取最多6个非分支指令和2个分支指令并组成一个组；在SMT模式下，来自不同线程的指令被分成两个组，每个组最多3个非分支指令和1个分支指令
* **译码** 分组之后，指令要么被解码，要么被送到微码硬件将复杂指令拆分成一系列内部简单操作
* **指令聚合** **POWER8**处理器可以将两个相连的架构指令聚合成一个内部指令
* **分支预测** **POWER8** IFU支持对8条指令进行分支扫描，分支扫描占用3个流水周期。如果有无条件或预测跳转的分支，准备目标地址以便下一周期取指使用；当发生跳转的分支时，需要三个周期才能获取下一个取指地址，有两个周期对当前线程是空闲的；但是在SMT模式，这两个周期会分配给其他线程。条件分支的方向使用分支历史表(BHT)进行预测，包括16K条目的本地BHT(LBHT), 16K条目的全局BHT(GBHT)和16K条目的 __global selection array (GSEL)__ 。每次预测8条指令，这8条都可以是分支指令。LBHT使用取指地址的14位索引，GBHT和GSEL使用取指地址和每个线程单独的21位 __Global History Vector (GHV)__ 哈希之后的11比特索引。GSEL条目的值用来选择LBHT和GBHT的预测结果。所有BHT条目都是2比特, 高位决定预测的方向，低位提供hysteresis
* **流水线冒险** **POWER8** IFU实现了新的机制消除因为流水线冒险导致的性能下降。Store-Hit-Load (SHL)是年轻的读操作在老的写操作之前执行，是乱序执行里一种流水线冒险。当检测到存在这种冒险时，从从流水线刷掉加载指令和后面指令，并标记加载指令的地址；当标记的加载指令再次执行时，会对加载指令设置依赖，确保在存储指令之后执行

### 10.1.3 POWER8 ISU
下图展示了负责分发指令，重命名寄存器，发射指令，完成指令和处理异常的ISU的逻辑视图：
![Pasted image 20230918172731.png](/assets/images/power/Pasted image 20230918172731.png)
**POWER8**处理器核以组作为分发单位，在ST模式, 一个组最多8条指令，6条非分支指令和2条分支指令，并且第二个分支指令在指令组的最后，每周期分发一个组；在SMT模式，每周期分发两个组，每个组最多4条指令，最多3条非分支指令，1条分支指令。在指令分发之前，指令组里所有指令需要的各种重命名资源和队列都要可用，不然会被保持在分发阶段。

ISU使用一个 __Global Completion Table (GCT)__ 来记录所有分发后的指令，按指令组记录，分发和完成都以指令组为单位。GCT有28条目，所有线程共享，一个条目记录一个指令组，在ST模式, 每个GCT是一个指令组，在SMT模式, 每个GCT条目是两个指令组；因此，GCT最多可以记录224条分发后的指令。GCT条目里的指令组里每个指令都有一个结束位，当指令成功执行之后，结束位会被标记。当一个指令组里所有指令都标记结束且指令组是线程里最老的指令组时，指令组就可以完成了；指令组完成时，指令组里所有指令的结果都在架构上可见，__completion group tag (GTAG)__ 被广播到相关联资源以便被释放。

### 10.1.4 POWER8 LSU
**POWER8**LSU有两个对称的LS0和LS1流水线和两个对称的加载(L0和L1)流水线，下图展示了**POWER8** LS0的流水线：
![Pasted image 20230918172837.png](/assets/images/power/Pasted image 20230918172837.png)
LS0和LS1流水线，每一个都能够一个周期执行存储加载指令；L0和L1流水线，每一个都能够一个周期执行加载指令。另外，这四个流水线都能在3个周期执行简单的定点指令。LSU组要功能单元有：存储加载AGEN和执行, **SRQ** , **store data queue (SDQ)** , **LRQ** , **load miss queue (LMQ)** , 包括 **D-ERAT** , ** ERAT Miss Queue(EMQ)** ,  **segment lookaside buffer (SLB)** 和TLB的地址转换单元, 包括组预测和数据目录(DDIR)的L1数据缓存和数据预取 __prefetch request queue(PRQ)__ 引擎。
* *存储加载执行* 在ST模式时，存储加载指令可以在LS0或LS1执行存储指令，在LS0, LS1, L0 和L1中执行加载指令；在SMT2, SMT4, 和SMT8模式，一半线程的指令在LS0和L0执行, 另一半线程的指令在LS1和L1中执行。存储指令发射两次，AGEN发射到LS0或LS1, 数据操作发射到L0或L1。LSU主要数据流包括64B的L2的重加载总线接口，到L2的16B的写数据接口，到VSU的两个16B加载数据接口，来自VSU的两个16B的写数据接口，来自FXU的两个8B写数据接口。定点加载指令只有3个周期的加载使用延迟，在加载指令和依赖的定点指令之间有两个流水线气泡；VSU加载指令有5个周期的加载使用延迟，在加载指令和依赖的VSU指令之间会产生四个流水线气泡。每个LSU流水线也可以执行定点加和逻辑指令，提高了指令发射的灵活性
* *存储加载顺序* 尽管指令乱序发射和执行，LSU必须保证存储加载指令遵守架构上的编程顺序。LSU主要依赖SRQ和LRQ来实现编程顺序。SRQ是一个40条目，基于真实地址RA的CAM结构，每个线程有128虚拟条目，允许分发128个并发写；可以发射32个并发的写；SRQ由所有线程共享。SRQ条目在存储指令发射时分配，当数据写到L1数据缓存或发送到L2时释放。每个SRQ条目都有一个对应的16B的SDQ条目，存储指令每周期可以发送最多16B数据到L2。写数据可以转发给后面的读指令，即使读写指令都是投机执行。LRQ也是一个44条目，基于真实地址RA的CAM。每个线程有128虚拟条目，允许分发64个并发写；可以发射44个并发的写；LRQ由所有线程共享。LRQ记录读操作，检查是否有同一个地址的年轻的读在更老的读或写之前执行的冒险。当检查到这样冒险发生时，LRQ会将年轻的读和后面同一个线程的指令从流水线刷掉，加载指令重新取指并执行
* *地址转换* 指令执行时，EA经过由两个48条目DERAT缓存和一个64条目的IERAT组成第一级地址转换逻辑转换成50位的真实地址。如果ERAT未命中，由每个线程32条目的SLB和所有线程共享的512条目的TLB组成的第二级地址转换逻辑进行转换。有效地址EA首先使用线段表转换成78位虚拟地址，然后78位虚拟地址经过 page frame table 转换成50位真实地址。**POWER8** 支持256 MB和1 TB两种大小的段, 以及4 KB, 64 KB, 16 MB, 和16 GB四种页表大小
	* D-ERAT是基于CAM的全相联的缓存，物理上分成四个，每个LSU流水线一个。在ST模式, 四个D-ERAT保持同步；在SMT模式，一半线程使用LS0和L0关联的两个48条目的D-ERAT，另一半线程使用LS1和L1关联的两个48条目的D-ERAT。每个D-ERAT条目支持 4-KB, 64-KB, 或16-MB页表， 16 GB保存为多个16-MB页表。D-ERAT采用LRU替换算法。第二个DERAT是256条目全相联的CAM结构，由所有线程共享，当主DERAT未命中时查找第二个DERAT。
	* SLB是每线程32条目，全相联的基于CAM的缓冲，每个SLB条目支持256 MB或1 TB段. **POWER8**支持 multiple pages per segment (MPSS) 扩展；MPSS使一个4KB基础页表的段可以同时有4-KB, 64-KB, 和16-MB页表；一个64KB基础页表的段可以同时有64-KB, 和16-MB页表。SLB由操作系统管理，当SLB缺失时发出数据或指令的段中断
	* TLB是2048条目，4路组相联的缓冲，TLB由硬件管理并使用真LRU替换算法。TLB缺失时最多支持4个并发table-walk， TLB支持hit-under-miss功能，即在做table-walk时支持访问。POWER7处理器里, 每个TLB条目都有 logical partition (LPAR) identity 标签，TLB条目里的 LPAR identity 处理器里的 LPAR identity 匹配

### 10.1.5 POWER8 FXU
FXU有两个相同的FX0和FX1组成，下图展示了FX0的逻辑视图:
![Pasted image 20230918172918.png](/assets/images/power/Pasted image 20230918172918.png)
每个FXU由一个多端口的GPR，一个执行加减，比较和陷阱的ALU，一个执行旋转，移位和选择的ROT，一个count (CNT) leading zeros单元，一个执行按位permute的bit-select unit (BSU)，一个除法器，一个乘法器，和一个执行计数，奇偶，BCD指令的*miscellaneous execution unit (MXU)*组成。
FXU中GPR有124个，最多用于四个线程的架构寄存器和重命名寄存器；其余架构寄存器保存在SAR中。GPR有8个读端口，其中两个为定点单元提供操作数，两个为相连的LSU的AGEN提供操作数，两个为相连的LU的AGEN提供操作数，两个为SAR提供数据；6个写口，两个接收定点单元的结果，两个接收相连的LSU的数据缓存的数据，两个接收相连的LU的数据缓存的数据。 **POWER8**处理器核实现了VSU到FXU的总线，减少了VSR 到 GPR传输的延迟。 在 ST模式，两个GPR的内容保持一致，在SMT2, SMT4, 或SMT8 模式，分别用于不同线程。**POWER8** FXU通过增加一倍的寄存器空间保存架构寄存器值来支持 __Transactional Memory (TM)__  ，增加寄存器是SAR。

### 10.1.6 POWER8 VSU
下图展示了**POWER8**的VSU：
![Pasted image 20230918172953.png](/assets/images/power/Pasted image 20230918172953.png)
**POWER8** VSU支持所有标量和向量指令的双发射，VMX/VSX Permute (PM)流水线延迟改进到2个周期, 存储带宽增加到每周期16-byte向量，来匹配每周期32-byte的加载带宽, 使用2个周期的*Simple Unit (XS)*流水线执行浮点比较指令。

## 10.2 POWER 8 片上缓存
**POWER8**片上缓存包括：
* 32K, 8路组相联的L1指令缓存
* 64K, 8路组相联的L1数据缓存，每周期支持4个16B的读或1个16B的写
* 512K，8路组相联的L2缓存，每周期支持64B的读和16B的写
* 96M，8路组相联的L3缓存，每周期32B的读和32B的写
下图展示了**POWER8**的片上缓存层次：
![Pasted image 20230718155501.png](/assets/images/power/Pasted image 20230718155501.png)

## 10.3 POWER8 内存系统
**POWER8**有两个内存控制器，每个支持4个通道，每个接口支持9.6 Gb/s，一共可以达到230 GB/s带宽，每个处理器系统一共支持1TB的内存。下图展示了内存结构：
![Pasted image 20230718160154.png](/assets/images/power/Pasted image 20230718160154.png)
下图展示了定制的DIMM上的memory buffer芯片：
![p8_mem.png](/assets/images/power/p8_mem.PNG)
memory buffer芯片额外提供了128M的L4缓存，同一个缓存的部分写操作可以在L4上合并。

## 10.4 POWER8 CAPI 
__Coherent Accelerator Interface Architecture (CAIA)__ 定义了加速设备和**POWER8**连接的一致性加速接口，__Coherence Attach Processor Interface(CAPI)__ 使用PCIe Gen3连接设备，并和处理器共享缓存一致性的内存以及虚拟地址转换，加速器上的程序可以直接使用有效地址指针，CAPI使加速器在系统上是一个处理器，而不仅仅是一个IO设备。 使用CAPI的好处包括从处理器直接使用共享内存，核处理器缓存直接传输数据，简化编程模型。

# 11. POWER 9
**POWER9**芯片有24个SMT4处理器核或12个SMT8处理器核，每对SMT4处理器核或者一个SMT8处理器核，包含512kB L2缓存和10MB L3缓存。
* SMT8 是为PowerVM优化
* SMT4 适用于Linux
![Pasted image 20230907145848.png](/assets/images/power/Pasted image 20230907145848.png)

## 11.1 POWER9 Core Execution Slice微架构
一个**Slice**是一个集成 **Vector and Scalar Unit(VSU)**和**Load/Store Unit(LSU)**的64位计算模块，VSU混合了整型和浮点的标量和向量操作。IBM宣称提高了资源利用率同时提供了不同slice间高效数据交换。两个slices组成一个**Super-Slice**，每个**super-slice**是一个128位的**POWER9**物理设计模块。两个super-slices，一个**Instruction Fetch Unit(IFU)** 和一个**Instruction Sequencing Unit(ISU)**组成一个**POWER9** SMT4核，SMT8等效于两个SMT4。下图展示了相应的关系：
![Pasted image 20230907160728.png](/assets/images/power/Pasted image 20230907160728.png)

## 11.2 POWER9 Core流水线
**POWER9**减少了5个从取指到计算的周期，对于浮点指令额外减少了3个从映射到退休的周期；增强了指令融合，减少了需要拆分成uOP的指令数， 移除了分发时的指令分组。下图展示了**POWER8**和**POWER9**的流水线对比图：
![Pasted image 20230907162021.png](/assets/images/power/Pasted image 20230907162021.png)
下图展示了**POWER9** SMT4的流水线：
![Pasted image 20230907180319.png](/assets/images/power/Pasted image 20230907180319.png)

## 11.3 POWER9 – Core Compute
**POWER9** 处理器核IFU有8路组相联32K指令缓存；每周期可以取8条指令，译码6条指令；每个VSU有4个64位标量或2个128向量位执行单元，有4个存储加载单元；每个VSU可以执行4个64位算术运算，2个128位Permute，和加解密运算；LSU有8路组相联32K数据缓存，支持4个并发64位读写。下图展示了SMT4的指令流：
![Pasted image 20230907161654.png](/assets/images/power/Pasted image 20230907161654.png)
下图展示了**POWER9** SMT4核的微架构：
![Pasted image 20230907180048.png](/assets/images/power/Pasted image 20230907180048.png)

## 11.4 POWER9 片上加速器和加速器接口
**POWERAccel**是**POWER**提供的所有片上接口和加速协议的集合。**POWER9**提供两套加速接口：
* 48条PCIe Gen4，双向192 GB/s带宽
* 96条25G接口，双向600GB/s带宽

在这两个物理接口之上，主要有4套协议：
- CAPI 2.0 **POWER9**基于PCIe引入的CAPI 2.0，是**POWER8**上的CAPI的带宽的4倍
- OpenCAPI 4.0 基于**POWER9**的25G接口(300 GiB/s)，应用于CPU和加速器连接
- NVLink 2.0 GPU和CPU高带宽互联
- OMI 串行高带宽内存接口
- 片上加速
    - 1个GZip
    - 2个842 Compression
    - 2个AES/SHA

下图展示了上述的接口和应用：
![Pasted image 20230907163735.png](/assets/images/power/Pasted image 20230907163735.png)

## 11.5 POWER9变种
**POWER9**根据内存接口和扩展性分为3个变种，如下图所示：
![Pasted image 20230907165748.png](/assets/images/power/Pasted image 20230907165748.png)
* *POWER9 Scale out* 使用直连内存并支持2路SMP
* *POWER9 Scale up*  使用*DMI Buffered Memory*并支持16路SMP
* *POWER9 Advanced IO*使用*OMI buffered memory*并支持16路SMP

## 11.6 POWER9 SMP
* *16Gbps X-Bus* 用于Drawer内全连接互联，一个drawer由4个**POWER9**组成
* *25Gbps O-Bus* 用于Drawer与Drawer之间互联

下图展示了基于**POWER9**的Power E980系统逻辑架构示意图：
![Pasted image 20230907174711.png](/assets/images/power/Pasted image 20230907174711.png)

下图分别展示了由2-, 3-, 和4-drawer组成的SMP系统的示意图：
![Pasted image 20230907175136.png](/assets/images/power/Pasted image 20230907175136.png)
![Pasted image 20230907175155.png](/assets/images/power/Pasted image 20230907175155.png)

# 12. POWER 10
**POWER10**最多有15个SMT8的处理器核，每个核有2 MB L2缓存和本地8 MB L3缓存，一共120M的 __non-uniform cache access (NUCA)__ L3缓存。**POWER10**在制造时候确定支持SMT8或SMT4；SMT8处理器核是为PowerVM固件和hypervisor设计，支持每线程一个分区或一个核一个分区；SMT4处理器核只支持每线程一个分区。下图展示了**POWER10**的全芯片规划图：
![Pasted image 20230906201517.png](/assets/images/power/Pasted image 20230906201517.png)

## 12.1 Power10 SMT4处理器核微架构
下图展示了**POWER10**的微架构以及相对**POWER9**的对比：
![Pasted image 20230908111004.png](/assets/images/power/Pasted image 20230908111004.png)

* 每周期来自L2 cache的64B，先预译码并以每周期32B保存在6路组相联的48KB的L1指令缓存中。每周期，从L1中取8条指令，然后扫描分支指令，通过分支预测器来预测分支方向和目标地址。
	* 当L1取指未命中时，请求发送到L2，基于一个预取预测器，缓存缺失通常最多会导致7个缓存行的预取
	* 当可用时，指令会被旁路到译码流水线，当译码停滞或切换线程时，指令被缓存在128条目的指令缓冲中。
* 每周期译码8条指令，8B的前缀指令暂用8条指令中的2条；少量指令被分解成2，3或4个内部操作，少数指令会使用微码引擎经过多个周期展开成内部操作。预译码标记的需要融合的指令在译码阶段处理。 
* 每周期可以执行8条指令，分发阶段为指令指定执行资源，`NOP` 在分发阶段直接结束而不需用经过计算流水线，当下列资源不足时，会导致分发停止：
	* 发射队列条目和slice分配
	* 寄存器重命名和依赖顺序
	* 存储加载队列虚拟条目
* 分发指令使用 __Instruction Completion Table (ICT)__ 来记录和跟踪，直到指令结束执行；每个SMT4 拥有 512条目的 __ICT__  ；2个条目称为一个指令对，每周期可以完成64条指令。

### 12.1.1 发射队列
每个slice有20条目的发射队列，每个SMT4处理器核有4个发射队列，一共支持80个依赖指令。 每个发射队列有一个本地计算slice和结对组成的super-slice的第二个发射队列，super-slice包含一套额外的计算流水线和一个读端口和写端口。每周期，每个slice的发射队列能发射3条指令：
* 一个计算指令
* 一个加载指令
* 一个存储或分支或简单指令

每个super-slice选择一个加载和存储/分支/简单指令执行，下图展示了一个SMT4的发射队列组成示意图：
![Pasted image 20230908113518.png](/assets/images/power/Pasted image 20230908113518.png)

### 12.1.2 POWER10处理器核流水线
下图展示了**POWER10**处理器核流水线，包括IFU流水线，LSU流水线和VSU流水线：
![Pasted image 20230908111733.png](/assets/images/power/Pasted image 20230908111733.png)
* __分支流水线__ 分支指令和存储地址计算以及简单加指令公用发射端口；每个SMT4每周期能处理一个分支指令，即使 __CR__ 寄存器还未确定，只要分支目标地址寄存器 __LNK__ , __CNT__ , 或 __TAR__ 准备好之后分支指令就会发射；这些部分执行的分支指令会在 __branch condition queues (BCQ)__ 里等待 __CR__  结果，这样可以使依赖目标寄存器的分支解决目标寄存器依赖。目标寄存器(LNK, CNT, TAR)和GPRs之间通过共享物理寄存器来减少 __move_to__ 和 __move_from__ 指令的延迟；目标地址产生指令和分支指令之间的依赖通过数据转发来减少延迟。
* __简单流水线__ 类似用来操作地址的立即数加指令，既可以在ALU流水线支持，也可以在存储/分支/简单流水线支持。这些指令可以使用每个SMT4里两个简单端口中任意一个，两个周期延迟计算结果；立即数加指令使用动态策略来发射到简单流水线或是ALU流水线。
* __存储加载流水线__ 当存储加载指令的操作数依赖性满足并且对应加载或存储队列条目分配之后，存储加载指令发射，并且相应发射队列条目被释放。存储指令首先发射来生成地址，随后发射来存储数据；存储加载流水线冒险在本地的存储加载流水线及队列中处理。类似读写缓存bank冲突的读冒险可以使用一个周期流水线延迟解决，其他需要刷流水线的冒险使用存储和加载队列解决。

### 12.1.3 POWER10处理器核指令流
下图展示了**POWER10**的简化的指令流图：
![Pasted image 20230908111421.png](/assets/images/power/Pasted image 20230908111421.png)
**POWER10**实现了一个激进投机的顺序流水前端，每周期取指，译码和分发8条指令。

## 12.2 Thread核LPAR管理
对于SMT8的处理器核，一个SMT4的处理器核负责偶数逻辑线程(0, 2, 4, 6), 另一个SMT4的处理器核负责奇数逻辑线程(1, 3, 5, 7)。一个SMT4处理器核支持ST, SMT2核SMT4模式，当在SMT4和SMT2之前切换时，硬件会重新平衡线程资源；线程通过中断激活，使用 __stop__ 指令休眠。下图展示了一个SMT8处理器核的资源划分和组成情况：
![Pasted image 20230908140921.png](/assets/images/power/Pasted image 20230908140921.png)
处理器的资源基于SMT的模式会动态分配，资源的共享以线程对为单位，一般如下:
* 在动态线程优先级的基础上，在所有激活的线程之间切换取指
* 在SMT4模式，译码和分发逻辑通过分割，可以支持每个线程对每周期4条指令，除了微码引擎，每个4指令的流水都是互相独立的
* 在SMT4模式，发射队列，执行单元，存储加载执行单元都是以指令对为单位切分，每个指令对使用一个super-slice
* __load miss queue(LMQ)__ 和L1缓存所有SMT4线程共享
* 对于硬件检测到的预取，预取队列是共享的；对于软件发起的预取，预取队列根据线程数静态分配

## 12.3 L2 Cache
一个SMT4处理器核有8路组相联，1M的L2缓存，缓存行是128B；对于L1指令缓存和L1数据缓存都是inclusive，缓存分为4个bank，不同bank的读写可同时进行；并使用一个8路组相联，4个bank多端口的缓存目录，每个bank有一个处理器核读端口，两个侦听读端口，一个写端口；处理器读端口和侦听端口工作在1/2处理器频率，写操作被安排在第二个周期，因此不会和读操作冲突。L2缓存是全局一致性的点，每个处理器线程有一个保留站；当L2缓存出现错误时，目录支持行删除功能。

## 12.4 L3 Cache
一个SMT4处理器核有16路组相联，4M的私有L3缓存，缓存行128B，使用4个bank，支持并行读写，到L2有64B的读端口，和来自L2的64B的写端口；目录也是16路组相联每两个周期支持不同bank的4个读或2个写；LRU算法使用4个比特来记录使用率和数据类型。L3缓存有4个分发流水线可以处理来自总线的侦听请求，侦察分发流水通过L3目录决定如何处理请求。如果请求需要发送数据或侦听推送或无效相关的缓存行，相应的请求被发送到16个侦听状态机处理；如果请求是 __lateral cast-out (LCO)__ 或接受的缓存注入，请求被发送到侦擦号状态及和写注入状态机。

## 12.5 SMP互联

发送命令的芯片负责发送反射的命令和结合的响应到SMP系统所有其他芯片，其他芯片返回合并的部分响应到发送命令的芯片。数据在点到点之间移动，对于读，包含数据的芯片直接将数据发送给请求芯片；对于写，包含数据的芯片直接将数据发送到*lowest-point of coherency (LPC)*，路由标签里包含芯片的标识。命令请求使用标签并且使用侦听协议广播，多个广播范围用于减少总线的利用率，SMP互联使用缓存状态确定缓存行最后的位置，是否需要更大范围的广播。

1-Hop的广播范围定义如下图所示：
![Pasted image 20230909193458.png](/assets/images/power/Pasted image 20230909193458.png)
2-Hop的广播范围定义如下图所示：
![Pasted image 20230909193512.png](/assets/images/power/Pasted image 20230909193512.png)
**POWER10**真实地址定义如下所示： 
![Pasted image 20230909193549.png](/assets/images/power/Pasted image 20230909193549.png)

### 12.5.1 Power10 SMP互联拓扑
**POWER10**片外SMP互联是高度可扩展，多层，全连接拓扑，片外接口使用18位32Gbps的高速差分接口，可以配置成1-hop或2-hop
* 1-Hop SMP拓扑
在1-Hop配置下，**POWER10**能够和最多8个处理器芯片全连接组成8路SMP系统，每个芯片最多使用7个A-link互联。1-Hop SMP系统拓扑如下图所示：
![Pasted image 20230909093241.png](/assets/images/power/Pasted image 20230909093241.png)
* 2-Hop SMP拓扑
在2-hop配置下，**POWER10**能够和其他3个芯片全连接组成一个4芯片的组，组内使用 __X-links__ ，组内每个**POWER10**芯片和其他组的芯片连接；组内每个芯片提供3个 __A-links__ 和其他组连接，一共可以互联4个组。2-Hop SMP系统拓扑如下图所示：
![Pasted image 20230909093315.png](/assets/images/power/Pasted image 20230909093315.png)

## 12.6 NCU
**POWER10**的*Non-Cacheable Unit (NCU)*负责处理非缓存的读写操作，字和双字的原子读写指令( `lwat` , `ldat` , `stwat` , `stdat` )，, `tlbie`  和 `sync`  and `ptesync` 指令，每一个SMT4有一个NCU单元。 下图展示了*NCU*的逻辑框图：
![Pasted image 20230909194305.png](/assets/images/power/Pasted image 20230909194305.png)
**POWER10**的*NCU*有一个专用的*cache-inhibited load station (LDS)*，每个线程一个，处理非缓存的读写操作，字和双字的原子读写指令( `lwat` , `ldat` )。非缓存的读，无论是否是guarded, 和原子读即不会合并也不会乱序；对于非缓存的写，字和双字的原子读写指令( `stwat` , `stdat` )，由一个16条目的*store gather station*组成的*store queue (STQ)*处理，4个线程共享，一对64B的*station*可以合并成128B。 

**POWER10**的*NCU*有8个*address machines (SAM)*可以支持8个非缓存写或`stwat` , `stdat`。*NCU*还有8个*snoop queues (TLBS)*来处理侦察到的 `TLBIE` 和4个 `snoop queues (SLBS)`处理SLBIE并转发到处理器核。

## 12.7 内存控制器
**POWER10**内存控制器提供了片上SMP互联和*OpenCAPI memory interface (OMI)*接口之间控制逻辑，OMI接口可以连接外部符合OpenCAPI 3.1(Memory Interface Class only)的*memory buffer*芯片，*memory buffer*芯片直接连接业界标准的DIMM。每个内存通道支持2个OMI连接，物理上，内存控制器组织成4个*extended memory OMI (EMO)*实例，每个*EMO*有两条通道。内存控制器可以处理来自处理器，缓存和IO的128B的读请求和64B写请求，1 - 128B部分写和内存原子操作(AMOs)。内存控制也处理地址保护，充当*lowest-point of coherency (LPC)*。

8个内存控制器可以配置成1个或多个地址交织的组，每个组内部，地址空间被分成不同部分，每个连续的部分可以*round-robin*方式由不同内存控制器处理。**POWER10**最大可用地址空间是128 TB。在一个内存控制器通道里，两个OMI子通道总是以128B进行交织。下图展示了**POWER10**的内存系统示意图：
![Pasted image 20230909195300.png](/assets/images/power/Pasted image 20230909195300.png)
为了提高大型系统的RAS，内存控制器支持：
* *Selective Memory Mirroring* 在这个配置下，内存子通道分成镜像对，不同镜像对和非镜像的 __BAR__ 寄存器允许访问镜像或非镜像区域
* *Whole Memory Encryption* 内存控制器支持对所有的访存传输做*Advanced Encryption Standard (AES)*加解密。加密通过固件可访问配置寄存器使能，对OMI配置空间和MMIO空间不会加密。

## 12.8 片上加速器
**Nest Accelerator unit (NX)** 由加解密和压缩解压缩引擎组成，下图展示了**NX**的基本结构框图： 
![Pasted image 20230911083955.png](/assets/images/power/Pasted image 20230911083955.png)
为了支持用户态使用有效地址对加速器调用，访问存储空间以及中断通知任务完成，**NX**包括下列逻辑：
* *SMP interconnect unit (SIU)* 和SMP互联以及DMA控制器的接口，每个16B的数据位宽；使用支持多个并行读写的**SMP interconnect common queue (SICQ)**最大化带宽； *User-mode access control (UMAC)* 协处理器调用模块，当*Virtual Accelerator Switchboard (VAS)*接收到 `copy/paste` 指令发出的*coprocessor request block (CRB)* 后，*UMAC*侦听*VAS*获取可用的*CRB*。每个协处理器分别支持一个高优先级和一个低优先级队列；*SIU*负责从队列获取*CRB*并分发到DMA控制器；*Effective-to-real address translation (ERAT)* 表可以保存最近32个地址转换。
* DMA控制器 解码CRB并代表协处理器搬运数据

下图展示了通过VAS来调用NX的流程图：
![Pasted image 20230911090237.png](/assets/images/power/Pasted image 20230911090237.png)
1. 当操作系统建立发送窗口之后，用户进程就可以使用NX了。首先，创建一个*coprocessor request block (CRB)*，*CRB*的格式由NX的规范定义。然后CRB通过 `copy/paste` 指令发送到*VAS*，`copy` 指令将*CRB*写到*copy buffer*，用户进程随后使用`paste` 指令将128B的*CRB*写到*VAS*。到真实地址的写传输作为*remote memory access write (RMA_write)*发送到SMP互联，真实地址里包含了发送窗口标识。128B的 `RMA_write` 的 *CRB*写到*VAS*的64个数据缓冲中的一个。*VAS*可以维护128个单独的窗口上下文，当侦听到 `RMA_write` 时，VAS使用发送窗口标识来查找 __Send Window Table Entry__ , 如果不在VAS的窗口缓存中，则从内存读取
2. VAS从发送窗口上下文中读取*Receive Window Identifier*，然后判断发送窗口对应的接收窗口。每个*NX coprocessor type (CT)*有一个单独的接收窗口，对应每个加速器的FIFO；如果接收窗口没有被缓存，则会从内存中读取
3. 使用接收窗口上下文中FIFO地址，*VAS*将 `RMA_write` 负载即*CRB*写入到NX的FIFO，VAS将发送和接收窗口标识附加到CRB里。NX在处理CRB时可以使用这些信息，CRB里的发送窗口标识可以被NX用来读取发送窗口和获取CRB里的地址的转换信息；接收FIFO是一个循环队列，当到达FIFO队尾时，VAS回到FIFO队头继续写下一个条目
4. 将*CRB*写到FIFO之后，VAS会发送一个 `ASB_notify` 命令到SMP互联；`ASB_notify` 命令包含*logical partition identifier (LPID)*, *process identifier (PID)*, 和thread identifier (TID)
5. 每个NX FIFO有一个特别的LPID:PID:TID，当NX侦听到 `ASB_notify` 并且匹配到自己的LPID:PID:TID时，会递增相关联的FIFO的计数器，指示有新的工作
6. 当NX CT队列为空但是计数器不为0时，NX从接收FIFO读取下一个CRB；一旦从FIFO里获取到CRB，NX通过对VAS的MMIO的写来返回信用。通过管理信用，VAS保证不会发生接收FIFO溢出。hypervisor基于FIFO的大小来初始化接收窗口的信用，当VAS将CRB写到FIFO时将信用计数器减一，当NX通过MMIO写返回信用时加一。NX使用CRB里附加的发送窗口标识来读取发送窗口上下文并减小内部计数器
7. NX将任务分发到关联的CT，CT可能有多个加速引擎并执行CRB
8. 任务完成之后，NX通过MMIO的写返回一个发送窗口信用给VAS；当hypervisor创建发送窗口时，会分配很多发送信用，这样hypervisor可以对多个用户共享同样加速器资源做调度，实现QoS，防止进程使用允许之外的资源。当VAS收到 `RMA_write` 写命令时，VAS减少关联的发送窗口的发送信用
9. NX写 __ coprocessor status block (CSB)__  并且可选的发送一个中断，通知用户任务已经完成；同时更新发送窗口里的 __ accelerator processed byte count (XPBC)__  来指示处理的字节数。

## 12.9 Nest MMU

![Pasted image 20230911094531.png](/assets/images/power/Pasted image 20230911094531.png)

## 12.10 中断控制器
**POWER10** 中断控制器INT由三部分组成：
* 虚拟化控制器(P3VC)
* 表现层控制器(P3PC)
* 总线接口公共队列(P3CQ)

这些逻辑接收中断并将中断发送到对应的处理器线程，各逻辑之间使用**POWER10**的总线互联，下图展示了中断控制器的逻辑框图及中断之间交互：
![Pasted image 20230911102017.png](/assets/images/power/Pasted image 20230911102017.png)

* **P3VC**通过**POWER10**的总线互联(比如cache-inhibited的写 `ci_wr` )接收来自 __interrupt source controllers (P3SCs)__ 的消息，使用主存里和每个触发源关联的 __event assignment entry (EAE)__ 里的信息进行处理，包括更新事件队列条目，转发消息到**P3PC**，**P3PC**然后将异常发送到一个处理器的线程。如果由于分配的处理器状态变化导致当前没有处理器处理中断，**P3VC**会重新分配中断消息。
* **P3PC**通过一个到处理器核的异常总线来提醒处理器线程，每个线程有三个信号，一个产生hypervisor中断，一个产生操作系统中断，最后一个产生基于事件的分支。P3PC里和三个信号相关联的逻辑是优先级和异常队列，用来防止低优先级事件抢占高优先级事件并且记录每个处理器线程处理的中断数量以便平衡负载；另外还有一个或多个保存在类似CAM的*logical server numbers*，称为 __thread interrupt management area (TIMA)__ ，这些*logical server numbers*用来识别物理线程上的软件实体。 当**P3VC**发送总线消息时，会查找这些 __thread interrupt management area (TIMA)__ 来识别候选处理器线程。
* **P3CQ**充当中断逻辑和系统之间的接口，主要负责中断控制器的消息接收和发送的顺序，比较中断控制器是否是命令的目的地，以及通过总线查询找到当前触发的中断的正确的**P3PC**。


# 参考文献
1. Power ISA (Version 3.1B), 2021.
2. R. R. Oehler and R. D. Groves, "IBM RISC System/6000 processor architecture," in IBM Journal of Research and Development, vol. 34, no. 1, pp. 23-36, Jan. 1990, doi: 10.1147/rd.341.0023.
3. S. W. White and S. Dhawan. 1994. POWER2: next generation of the RISC System/6000 family. IBM J. Res. Dev. 38, 5 (Sept. 1994), 493–502. https://doi.org/10.1147/rd.385.0493
4. F. P. O'Connell and S. W. White, "POWER3: The next generation of PowerPC processors," in IBM Journal of Research and Development, vol. 44, no. 6, pp. 873-884, Nov. 2000, doi: 10.1147/rd.446.0873.
5. pSeries 690 Service Guide, n.d.
6. Andersson, S., Bell, R., Hague, J., Holthoff, H., Mayes, P., Nakano, J., Shieh, D., Tuccillo, J., n.d. RS/6000 Scientific and Technical Computing: POWER3 Introduction and Tuning Guide.
7. J. M. Tendler, J. S. Dodson, J. S. Fields, H. Le and B. Sinharoy, "POWER4 system microarchitecture," in IBM Journal of Research and Development, vol. 46, no. 1, pp. 5-25, Jan. 2002, doi: 10.1147/rd.461.0005.
8. Bossen, D.C., Kitamorn, A., Reick, K.F., Floyd, M.S., 2002. Fault-tolerant design of the IBM pSeries 690 system using POWER4 processor technology. IBM J. Res. & Dev. 46, 77–86. [https://doi.org/10.1147/rd.461.0077](https://doi.org/10.1147/rd.461.0077)
9. B. Sinharoy, R. N. Kalla, J. M. Tendler, R. J. Eickemeyer and J. B. Joyner, "POWER5 system microarchitecture," in IBM Journal of Research and Development, vol. 49, no. 4.5, pp. 505-521, July 2005, doi: 10.1147/rd.494.0505.
10. Kalla, R., 2003. IBM’s POWER5 Microprocessor Design and Methodology.
11. Kalla, R., Sinharoy, B., Tendler, J.M., 2004. IBM power5 chip: a dual-core multithreaded processor. IEEE Micro 24, 40–47. [https://doi.org/10.1109/MM.2004.1289290](https://doi.org/10.1109/MM.2004.1289290)
12. Clabes, J., Friedrich, J., Sweet, M., DiLullo, J., Chu, S., Plass, D., Dawson, J., Muench, P., Powell, L., Floyd, M., Sinharoy, B., Lee, M., Goulet, M., Wagoner, J., Schwartz, N., Runyon, S., Gorman, G., n.d. Design and Implementation of the POWER5 TM Microprocessor.
13. Jiménez, V., Cazorla, F.J., Gioiosa, R., Valero, M., Boneti, C., Kursun, E., Cher, C.-Y., Isci, C., Buyuktosunoglu, A., Bose, P., 2010. Power and thermal characterization of POWER6 system, in: Proceedings of the 19th International Conference on Parallel Architectures and Compilation Techniques. Presented at the PACT ’10: International Conference on Parallel Architectures and Compilation Techniques, ACM, Vienna Austria, pp. 7–18. [https://doi.org/10.1145/1854273.1854281](https://doi.org/10.1145/1854273.1854281)
14. H. Q. Le et al., "IBM POWER6 microarchitecture," in IBM Journal of Research and Development, vol. 51, no. 6, pp. 639-662, Nov. 2007, doi: 10.1147/rd.516.0639.
15. IBM Power 570 and IBM Power 595 (POWER6) System Builder, n.d.
16. Kalla, R., Sinharoy, B., Starke, W.J., Floyd, M., 2010. Power7: IBM’s Next-Generation Server Processor. IEEE Micro 30, 7–15. [https://doi.org/10.1109/MM.2010.38](https://doi.org/10.1109/MM.2010.38)
17. B. Sinharoy et al., "IBM POWER7 multicore server processor," in IBM Journal of Research and Development, vol. 55, no. 3, pp. 1:1-1:29, May-June 2011, doi: 10.1147/JRD.2011.2127330.
18. POWER7 and POWER7+ Optimization and Tuning Guide, n.d.
19. B. Sinharoy et al., "IBM POWER8 processor core microarchitecture," in IBM Journal of Research and Development, vol. 59, no. 1, pp. 2:1-2:21, Jan.-Feb. 2015, doi: 10.1147/JRD.2014.2376112.
20. Caldeira, A., Kahle, M.-E., Saverimuthu, G., Vearner, K.C., n.d. IBM Power Systems S812LC Technical Overview and Introduction.
21. IBM Power System E980: Technical Overview and Introduction, n.d.
22. IBM Power System S822: Technical Overview and Introduction, n.d.
23. S. K. Sadasivam, B. W. Thompto, R. Kalla and W. J. Starke, "IBM Power9 Processor Architecture," in IEEE Micro, vol. 37, no. 2, pp. 40-51, Mar.-Apr. 2017, doi: 10.1109/MM.2017.40.
24. POWER9 Processor User’s Manual, 2019.
25. W. J. Starke, B. W. Thompto, J. A. Stuecheli and J. E. Moreira, "IBM's POWER10 Processor," in IEEE Micro, vol. 41, no. 2, pp. 7-14, March-April 2021, doi: 10.1109/MM.2021.3058632.
26. Power10 Processor Chip User’s Manual, 2021.
27. IBM Power E1050: Technical Overview and Introduction, n.d.
28. IBM Power S1014, S1022s, S1022, and S1024 Technical Overview and Introduction, n.d.
29. W. Starke and B. Thompto, "IBM’s POWER10 Processor," 2020 IEEE Hot Chips 32 Symposium (HCS), Palo Alto, CA, USA, 2020, pp. 1-43, doi: 10.1109/HCS49909.2020.9220618.