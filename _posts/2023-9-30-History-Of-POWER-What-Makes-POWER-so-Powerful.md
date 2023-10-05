---
title : History of POWER | What makes POWER so powerful
categories : [history]
tags : [chip, cpu, architecture, history]
---

# 0. 概述
最近几年系统性的学习并梳理了近30多年的计算技术发展历史，稍有感悟。遂决定将之整理成文，目的有二，一来作为知识沉淀，串联起不同技术，挖掘不同架构之间的渊源，二来通过整理再次审视历史，期望窥见未来发展方向。我将本系列命名为**鉴往知远**, 主要关注**计算与互联**。 本文为第一篇，主要回顾IBM POWER系列。

POWER (Performance Optimization With Enhanced RISC)架构起源于1990年IBM的RISC System/6000产品。1991年，Apple, IBM, 和Motorola一起合作开发了PowerPC架构。1997, Motorola和IBM合作将PowerPC用于嵌入式系统。2006, Freescale和IBM合作制定了POWER ISA 2.03。 2019年8月21, 开源了POWER Instruction Set Architecture (ISA)。PowerISA由必须的基础架构，4组可选特性，一组过时特性组成。OpenPOWER也允许自定义扩展。

本文通过系统性回顾整个POWER系列处理器，试图通过POWER系列处理器发展的历史脉络，来展现近30年计算架构的变迁，技术的演进，进而窥见计算技术发展的未来。

本文组织形式如下:
* 第一章简单介绍POWER指令集架构3.1B版本， 也是最新POWER10处理器使用的指令集。通过本章介绍，可以了解程序在POWER架构上运行的过程及预期结果，掌握异常处理，特权模型，以及POWER调试手段
* 第二章简单回顾整个POWER系列处理器，总结各代处理器的面积，功耗，缓存，IO等基本内容
* 第三章，第四章，第五章分别描述POWER 1和POWER 2的整体架构，简单介绍了POWER 3的微架构，主要是了解这些古老系统结构
* 第六章详细描述POWER 4微架构以及从单核到双核的进化
* 第七章详细描述POWER 5微架构，从单线程到双线程的演进，以及集成的片上内存控制器
* 第八章介绍POWER 6处理器微架构，了解从之前乱序执行变为顺序执行的取舍
* 第九章介绍POWER 7处理器微架构，了解在支持更多线程情况下如何减少面积和功耗，以及内部缓存协议状态
* 第十章介绍POWER 8处理器微架构
* 第十一章主要介绍POWER 9处理器微架构，SMP互联
* 第十二章完整介绍POWER 10处理器微架构，SMP互联，片上加速器，中断
* 第十三章列出了主要的参考文献

# 1. POWER指令集架构
## 1.1 寄存器
* __Condition Register (CR)__ 是32寄存器，记录指令执行结果，供测试和条件分支指令使用
* __Link Register (LR)__ 是64位寄存器，保存 __Branch Conditional to Link Register__ 指令跳转地址, 并且可以保存当 __LK=1__ 时分支指令和 __System Call Vectored__ 指令后的返回地址。
* __Count Register (CTR)__ 是64位寄存器。当执行的分支指令的 __BO__ 编码时候可以作为`for`循环计数寄存器。__Count Register__ 也可以保存 __Branch Conditional to Count Register__ 指令的跳转目标地址。
* __VR Save Register (VRSAVE)__ 是32位寄存器，软件作为SPR使用。
* __Fixed-Point Exception Register (XER)__ 是64位寄存器
	* 0:31 Reserved
	* 32 __Summary Overflow (SO)__ Summary Overflow置位当Overflow置位.
	* 33 __Overflow (OV)__ 指令执行溢出时Overflow置位
	* 34 __Carry (CA)__ 
	* 35:43 Reserved
	* 44 __Overflow32 (OV32)__ OV32 32位运行模式时溢出位
	* 45 __Carry32 (CA32)__ CA32 32位运行模式时溢出位
	* 46:56 Reserved
	* 57:63 指定 __Load String Indexed__  和 __Store String Indexed__  指令传输的字节数
* __FloatingPoint Status and Control Register(FPSCR)__ 控制浮点异常处理和浮点指令执行结果状态。32:55位是状态位, 56:63是控制位

![Pasted image 20230904172658.png](/assets/images/power/Pasted image 20230904172658.png)

* __Logical Partitioning Control Register (LPCR)__  __LPCR__ 控制资源的逻辑分区
* __Logical Partition Identification Register (LPIDR)__  __LPIDR__ 设置逻辑分区ID
* __Machine State Register (MSR)__ 是64位寄存器，控制和定义了线程的状态：
	* 0 __Sixty-Four-Bit Mode (SF)__ 0 线程运行在32位模式，1 线程运行在64位模式。当线程处于ultravisor时软件要保证 __SF=1__ 
	* 1:2 Reserved 
	* 3 __Hypervisor State (HV)__ 具体意义参考特权模型章节
	* 4 Reserved 
	* 5 常0
	* 6:37 Reserved 
	* 38 __Vector Available (VEC)__ 0 线程不能执行任何向量指令 1 线程可以执行向量指令
	* 39 Reserved 
	* 40 __VSX Available (VSX)__ 0 向量不能执行 __Vector Scalar Extension(VSX)__ 指令1 向量可以执行 __VSX__ 指令
	* 41 __Secure (S)__ 0 线程处于非安全态，不能访问安全域线程，且不在ultravisor态 1 线程处于安全态
	* 42:47 Reserved 
	* 48 __External Interrupt Enable (EE)__ 0 __External__ , __Decrementer__ , __Performance Monitor__ , 和 __Privileged Doorbell__ 中断被禁止 1 __External__ , __Decrementer__ , __Performance Monitor__ , 和 __Privileged Doorbell__使能
	* 49 __Problem State (PR)__ 具体意义参考特权模型章节
	* 50 __Floating-Point Available (FP)__ 0 线程不能执行浮点指令 1 线程可以执行浮点指令
	* 51 __Machine Check Interrupt Enable (ME)__ 0 __Machine Check__ 中断禁止 1 __Machine Check__ 中断使能
	* 52 __Floating-Point Exception Mode 0 (FE0)__ 
	* 53:54 __Trace Enable (TE)__
		* 00 Trace Disabled: 线程正常执行指令
		* 01 Branch Trace: 当线程执行完分支指令之后产生 __Branch type Trace__ 中断
		* 10 Single Step Trace: 当线程成功完成下一个指令时产生一个 __Single-Step type Trace__ 中断，__urfid__ , __hrfid__ , __rfid__ , __rfscv__ , 或 __Power-Saving Mode__ 指令除外。
		* 11 Reserved
	* 55 __Floating-Point Exception Mode 1 (FE1)__
	* 56:57 Reserved 
	* 58 __Instruction Relocate (IR)__ 0 禁止指令地址转换； 1 使能指令地址转换
	* 59 __Data Relocate (DR)__ 0 禁止数据地址转换，并且不发生 __Effective Address Overflow (EAO)__ ；1 使能数据地址转换，__EAO__ 产生 __Data Storage__ 中断
	* 60 Reserved 
	* 61 __Performance Monitor Mark (PMM)__ 软件控制Performance Monitor
## 1.2 计算模式
处理器提供两种执行模式, 64位模式和32位模式。 两种模式下，设置64位寄存器指令仍然影响所有64位。计算模式控制有效地址的解释, __Condition Register__ 和 __XER__ 的设置, 当 __LK=1__ 时 __Link Register__ 被分支指令的设置 , 以及 __Count Register__  被条件分支指令的使用。几乎所有指令都可以在两种模式下运行。在两种模式下，有效地址的计算都使用所有相关寄存器的64位( __General Purpose Registers__ , __Link Register__ , __Count Register__ 等) 并且产生64位结果。
## 1.3 指令格式

![Pasted image 20230904175410.png](/assets/images/power/Pasted image 20230904175410.png)

指令前缀格式
前缀指令由4字节前缀和4字节后缀组成。所有前缀的0:5都是*0b000001*.

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
* __addic__ , __addic__ , __subfic__ , __addc__ , __subfc__ , __adde__ , __subfe__ , __addme__ , __subfme__ , __addze__ , 和 __subfze__ 指令设置 __CR.CA__ , 在64位模式下反映位0的进位，在32位模式反映位32的进位
* 对于XO形式的`Multiply Low`和`Divide`指令, __CR.SO/OV/OV32__ 设置依赖计算模式, 反映 __mulld__ , __divd__ , __divde__ , __divdu__ 和 __divdeu__ 的64位溢出, __mullw__ , __divw__ , __divwe__ , __divwu__ 和 __divweu__ 低32位的溢出.
### 1.3.6 定点比较指令
定点比较指令将寄存器 __RA__ 和如下值比较
a 符号扩展SI
b 无符号扩展UI
c __RB__ 寄存器的值
__cmpi__ 和 __cmp__ 是有符号比较, __cmpli__ 和 __cmpl__ 是无符号比较.
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

### 1.6.1 Ultravisor/Hypervisor Offset Real Mode Address
当 __MSR.HV = 1__  且 __EA.0 = 0__ 时, 存储访问由 __Ultravisor Real Mode Offset Register__  或 __Hypervisor Real Mode Offset Register__ 控制, 根据 __MSR.S__ 值：
* 当 __MSR.S=1__ 时, 有效地址4:63位和 __URMOR__ 里的60位偏移值按位或，得到的60位结果就是访问的真实地址
* 当 __MSR.S=0__ 时, 有效地址4:63位和 __HRMOR__ 里的60位偏移值按位或，得到的60位结果就是访问的真实地址

### 1.6.2 Segment Translation
下图展示了64位有效地址通过 __Segment Lookaside Buffer (SLB)__ 转换为78位虚拟地址的过程：
![Pasted image 20230905155844.png](/assets/images/power/Pasted image 20230905155844.png)
如果 __SLB__ 未命中， __LPCR.UPRT=1__ , 且 __MSR.HV=0__  或 __LPID=0__ 时会搜索 __Segment Table__ 。__Segment Lookaside Buffer (SLB)__ 指定了 __Effective Segment IDs (ESIDs)__ 和 __Virtual Segment IDs (VSIDs)__ 的映射。

在 __Paravirtualized HPT__ 模式，使用Page Table来将78位虚拟地址转换为真实地址。__Hashed Page Table (HTAB)__ 指定了虚拟页表和真实页表的映射，__Hashed Page Table (HTAB)__ 由 __Page Table Entry Groups (PTEGs)__ 组成。一个 __PTEG__ 包含8个 __Page Table Entries (PTEs)__ ，每个16。
下图展示了使用 __HPT__ 将78位虚拟地址转换位60位真实地址过程
![Pasted image 20230906084212.png](/assets/images/power/Pasted image 20230906084212.png)

### 1.6.3 Radix Tree Translation
__Radix Tree root descriptor (RTRD)__  指定了转换的地址大小，root table的大小和位置。下图展示了4级Radix Tree在 __PDE.NLS=13__  和 __NLS=9__ 时将52位EA转换成56位RA的过程：
![Pasted image 20230905170905.png](/assets/images/power/Pasted image 20230905170905.png)
Radix Tree Page Directory Entry内容如下：
![Pasted image 20230905171134.png](/assets/images/power/Pasted image 20230905171134.png)
![Pasted image 20230905171203.png](/assets/images/power/Pasted image 20230905171203.png)
Radix Tree Page Table Entry

#### 1.6.3.1 Nested Translation
当 __MSR.HV=0__  且地址转换使能时，对于guest real address必须经过分区的hypervisor的Radix Tree转换。下图展示了Radix on Radix Page Table对52位EA转换过程，一共需要24次内存访问：
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
* __Come From Address Register__  __Come From Address Register (CFAR)__ 是64位寄存器， 当执行 __rfebb__ , __rfid__ , 或 __rfscv__ 执行时，寄存器值设置为当前执行的有效地址。
* __Completed Instruction Address Breakpoint__  __Completed Instruction Address Breakpoint__ 提供了发现完成执行特定地址指令的机制。地址比较是基于有效地址(EA)。__Completed Instruction Address Breakpoint__ 机制是由 __Completed Instruction Address Breakpoint Register (CIABR)__ 控制。
* __Data Address Watchpoint__  __Data Address Watchpoint__ 提供了发现多个双字有效地址(EA)加载存储访问的机制。至少两个独立地址范围可以指定。每个 __Data Address Watchpoint__ 是由一对SPRs控制：__Data Address Watchpoint Register(DAWRn)__ 和 __Data Address Watchpoint Register Extension (DAWRXn)__ 

# 2. POWER处理器概述
* 1975年，IBM Thomas J. Watson Research Center发明了第一个RISC机器，801。801原始设计目标是1 IPC，研究重点是定义一个能够每周期执行多个指令，即超标量的架构。研究的结果是第二代RISC架构，称为"AMERICA architecture"
* 1986年，IBM 位于Austin, Texas的开发RT System的实验室, 基于AMERICA architecture开始开发产品。这个研究最终实现了IBM RISC System/6000* (RS/6000)，即IBM's **POWER**架构
* 1990年，IBM发布RISC System/6000, 包括9种当时工业界最快最强大的工作站。RISC System/6000使用IBM研发的精简指令集，一个新版本的IBM实现UNIX操作系统Advanced Interactive Executive (AIX)。这就是**POWER 1**
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
* 分支单元  分支单元主要负责取指，地址翻译和检查， 中断处理。除非对应定点或浮点单元上指令队列已满或对定点或浮点单元上数据存在依赖性，分支单元能不断对下一个指令进行取指，解码，并执行相应指令，或者将对应定点或浮点指令分发到对应定点或浮点单元。分支单元每周期可以至少获取3条指令，分别对应定点单元，浮点单元，及分支单元。并且每周期可以分发一个定点和浮点指令。**POWER 1**不支持分支延迟槽。
* 定点单元 定点单元除了处理所有定点算术指令外，还需要为浮点单元及自身计算数据地址。因此，定点单元需要负责调度浮点单元和数据缓存之间数据交换。浮点单元寄存器只负责接收或提供数据，因此，浮点单元加载和存储操作是消耗定点单元周期数。
* 浮点单元 浮点单元支持ANSI/IEEE Standard 754-1985. RISC System/6000浮点运算是双精度计算。因此，单精度浮点运算也会被转换为双精度进行运算。

下图展示了**POWER 1**的有效地址(EA)转换过程，32位有效地址首先经过段寄存器转换为52位虚拟地址(VA), 然后52位虚拟地址(VA)经过TLB翻译成32位真实地址(RA)：
![Pasted image 20230912141125.png](/assets/images/power/Pasted image 20230912141125.png)


# 4. POWER 2
**POWER2** 包括高性能的Multi-Chip Module (MCM)和Single Chip Module(SCM)，区别主要是DCU和内存接口，MCM逻辑框图如下所示：

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

* *Instruction processing unit* 和*instruction ﬂow unit* *IPU* 和*IFU* 负责取值，缓存以及指令分发和完成整个生命流程。IPU有32KB指令缓存和*cache reload buffer(CRB)* 。 指令缓存缓存行大小是128B，因此一共有256行，组织成128路组相联，并且单周期访问。*CRB* 保存从内存读取最新缓存行。另外实现虚拟地址转换，实现了256-entry 2路组相联*instruction translation lookaside buffer (ITLB)* 和16-entry *instruction segment lookaside buffer (ISLB)* 。每周期可以取8条指令，分发4条指令，并且完成4条指令。为了调高吞吐，指令顺序分发，大部分可以乱序执行和结束，顺序完成。指令分发到不同功能单元指令队列并且由32-entry *completion queue*记录。这些功能单元指令队列确保对应功能单元有足够指令可以选择来执行，并且阻塞的指令不会阻碍IFU的指令分发。*completion queue* 确保处理器的架构状态的正确性，强制指令顺序完成和中断以及异常的正常处理。**POWER3** 采用两种机制来提高分支预测的准确性。首先，通过跟踪所有并发的带条件码的指令，处理器能够在指令分发时就确定分支结果。另外，对于在指令分发时无法确定的分支，会进行投机。当带条件码指令完成并且分支结果投机错误，分支指令之后所有指令会取消并重新分发正确指令。
* *Fixed-point execution units* **POWER3** 包含3个定点执行单元：2个单周期执行单元和一个多周期执行单元。单周期执行单元执行算术，移位，逻辑，比较，*trap* 和 *count leading zero* 指令。其他比如乘法，除法等都由多周期执行单元执行。两个单周期执行单元共享一个 *six-entry* 指令队列, 多周期执行单元使用一个*three-entry* 的指令队列。不同于**POWER 2** 包含两个执行定点和加载存储的对称的执行单元，**POWER3** 有两个专门的存储加载单元。独立的定点执行单元和存储加载单元对于类似Monte Carlo仿真这样整数操作占比大的应用很显然有比较大性能提升，但是即使是对浮点运算也很重要。像在*sparse-matrix-vector multiply* 中，整数索引必须先转换成字节偏移。
* *Floating-point execution units* *FPU* 包含两个对称的执行融合乘加流水的执行单元。所有浮点指令都需要经过乘法和加法阶段。对于浮点乘，加法阶段0作为一个操作数。对于浮点加，乘法阶段1作为一个操作数。
* *Load/store execution units* 所有的存储加载操作都由两个存储加载执行单元完成。加载指令将数据从内存转移到定点或浮点重命名寄存器，存储指令将数据从寄存器转移到内存。存储指令使用一个*16 entries store buffer* 来提高性能. 当存储指令获取到数据之后就可以完成，而不必等到写进数据缓存。两个加载存储执行单元共享一个 *six-entry* 指令队列。 *LSU* 乱序执行允许加载指令超过存储指令同时记录数据依赖性。存储指令的顺序在执行阶段和存储队列中维护。
* *Data cache unit* *DCU* 主要由数据内存管理单元*MMU* ， L1缓存和数据预取单元组成
	* *Memory management unit* *MMU* 主要负责数据的地址转换，包括一个*16-entry segment lookaside buffer (SLB)* 和两个镜像的*256-entry两路组相联*data translation lookaside buffers (DTLB) 以便支持两个存储加载执行单元。*MMU* 支持1T物理内存，64位有效地址和80位的虚拟地址。
	* *L1 data cache* L1数据缓存是单周期访问的64 KB，由4个bank，一共512个128B缓存行组成。每个bank是128路组相联。每个bank，由地址A55/A56决定，根据双字地址(A60)地址分成2个子bank。 *least-recently-used (LRU)* 替换算法不适用于128路组相联缓存，因此，**POWER3** 在L1指令缓存和L1数据缓存中采用*round-robin* 替换方案。为了适应不同位宽和频率以便重建缓存行，需要在传输路径上加入*linewidth buffers* 。L1数据缓存和BIU之间, 每个bank有一个*cache- reload buffers (CRBs)* 和一个*cache-storeback buffers (CSBs)*。 加载操作命中*CRB* 直接提供数据，不必等到缓存行加载。*CSB* 和 *BIU* 之间是64-byte 接口。L1数据缓存可以支持4个缓存缺失，可以有效掩盖内存延迟。当加载操作没有命中L1数据缓存，指令会被放入*six-entry load-miss queue(LMQ)* , 同时*BIU*发出数据加载传输， 后续加载指令可以继续执行。只有当第五个加载操作没有命中L1数据缓存，并且*LMQ* 里已经有4个未命中加载指令，加载操作才会被暂停，直到*LMQ* 里任意一个加载指令执行完成。
	* 数据预取 **POWER3**处理器一大创新是硬件数据预取功能。**POWER3**处理器根据监控到的缓存行缺失及匹配到的模式进行数据预取。当一个模式或数据流被探测到，处理器会对缓存行进行投机预取，假设数据会马上被使用。数据预取对于掩藏内存延迟非常关键。预取的数据流在地址上是连续的缓存行，要么是递增，要么是递减的。 **POWER3**处理器预取引擎包括一个*ten- entry stream ﬁlter* 和一个 *stream prefetcher* 。 *stream ﬁlter* 观察所有数据缓存缺失的真实地址(RA)，检测潜在的预取数据流。*stream ﬁlter* 根据加载指令的地址操作数，猜测下一个缓存行的真实地址是增加还是减少，并记录到*FIFO ﬁlter* 。当发生新的缓存缺失时，如果发生缓存缺失的真实地址和猜测的地址一致，这样就检测到一个数据流。如果*stream prefetcher* 当前少于4个数据预取流，当前数据流会被接受，并且预测下一个缓存行加载操作会通过*BIU* 发出。一旦数据流被放入*stream prefetcher* , 数据流保持活跃直到数据流到达终点或者有新的缓存缺失进入*stream ﬁlter*。数据预取引擎会尝试保持预取2个缓存行，前一个缓存行会被放到L1 缓存，后一个缓存行会被放到BIU里一个预取缓冲区。因此，数据预取引擎可以并发预取4个数据流，每个2个缓存行，一共8个预取操作。数据预取引擎监控所有加载操作的地址操作数，当LSU结束当前缓存行并开始加载下一个缓存行，数据预取引擎将预取缓冲区数据传输到L1，并预取下一个缓存行到预取缓冲区。

下图展示了预取引擎框图
![Pasted image 20230912094139.png](/assets/images/power/Pasted image 20230912094139.png)

* *Bus interface unit* **BIU**提供*IPU*，*DCU*，预取引擎和L2缓存之间连接。数据接口位宽128位。
* *L2 cache* **POWER3** 支持1MB-16MB的L2缓存，可以是组相连或直接映射。总线和L2以32B位宽相连，一个128B缓存行需要4个周期传输。

# 6. POWER 4
**POWER 4** 是一个双核乱序处理器，通过一个*core interface unit(CIU)* 共享一个统一的片上二级缓存。
* *CIU* 是3个L2控制器和2个处理器之间的*crossbar switch* 。每个L2控制器每周期能提供32B数据。*CIU* 通过一个8B的接口接受来自处理器的存储操作。
* 每一个处理器有一个相关联的*noncacheable unit(NC unit)*，负责处理指令串行功能和非缓存存储加载操作。
* L3控制器和目录在**POWER4** 片上，但是实际L3位于片外。
* *fabric controller* 负责L2和L3之间的数据流以及**POWER 4** SMP通信。
* *GX controller* 控制器负责IO。
每个处理器芯片包括四种类型IO:
* 和同一个模块内其他**POWER4** 芯片通信，4个16-byte接口。物理是线上，这4个逻辑总线有6个总线组成，3个输入，3个输出。
* 和不同模块的**POWER4** 芯片通信，2个8-byte的总线，一个输入，一个输出。
* 片外L3接口，2个16-byte总线，一个输入，一个输出，运行在1/3处理器频率。
* 2个4-byte GX总线, 一个输入，一个输出，运行在1/3处理器频率。

下图展示了**POWER4** 芯片的逻辑框图:
![Pasted image 20230912171951.png](/assets/images/power/Pasted image 20230912171951.png)


同时 **POWER4** 增加了 __logical partition ID__ , __real mode offset register (RMOR)__ , __real mode limit register (RMLR)__ 和 __hypervisor RMOR__ 寄存器，提供将SMP系统分成逻辑分区的功能。这些寄存器只能在hypervisor模式下修改。逻辑分区可以提高系统的稳定性，并且可以用于虚拟化。

下图展示了**POWER 4**的全芯片图：

![Pasted image 20230809154420.png](/assets/images/power/Pasted image 20230809154420.png)

## 6.1 POWER4 Core
**POWER 4** 处理器是一个乱序的超标量设计，一共实现了16级流水线，每周期可以取指8条指令，发射5条指令，完成5条指令，可以同时处理超过200条指令。为了增强指令级并行，**POWER 4**处理器拥有8个执行单元，2个相同的可以每周期执行融合乘加的浮点执行单元，2个存储加载单元，2个定点执行单元，一个分支执行单元和一个操作条件寄存器的执行单元。下图展示了**POWER4**处理器的框图

![Pasted image 20230912172922.png](/assets/images/power/Pasted image 20230912172922.png)

### 6.1.1 分支预测
**POWER4** 使用多级分支预测机制来预测条件分支是否发生。每周期直接相连的64KB的指令缓存提供8个指令，分支预测逻辑每周期可以查找两条分支指令。根据找到的分支类别，不同分支预测机制用来预测分支方向或分支目标地址。无条件分支的方向不做预测，所有条件分支都做预测，即使在取值阶段通过 __condition register__ 已知。对于 __branch-to-link-register (bclr)__ 和 __branch-to-count-register (bcctr)__ 指令，分支目标分别通过硬件实现的 *link stack* 和 *count cache*机制预测。 绝对和相对分支目标地址在分支指令扫描时候直接计算。POWER4使用3个branch-history tables来预测分支方向：
* 第一个是本地预测器，类似于分支历史表BHT，使用分支指令地址来索引16 384-entry数组，产生1-bit预测分支是否发生
* 第二个是全局预测器，通过一个执行过的分支指令的11-bit向量，和分支指令地址进行按位异或，来索引16 384-entry全局历史表来产生1-bit预测分支是否发生
* 第三个是选择表，记录上面两个预测器预测表现来选择其中一个预测器，16 384-entry 选择表和全局预测器使用同样方式索引来产生1bit的选择信号。

动态分支预测能够被软件覆盖，通过设置条件分支指令的2个保留位，一个用来指示软件覆盖，一个用来预测方向。**POWER4** 使用 *link stack* 来预测 __branch-to-link__ 指令的目标地址，一般用于子函数返回。通过设置 __branch-to-link__ 指令的提示位，软件可以将 __branch-to-link__ 是否代表子函数返回，目标地址是否重复通知处理器。当取值逻辑取到 __branch-and-link__ 指令并预测发生分支跳转时，处理器会将下一条指令的地址推入 *link stack* 。当取到一个提示位指示是子函数返回并预测发生分支跳转的 __branch-to-link__ 指令时，*link stack* 会被弹出，并从弹出的地址继续取值。 
**POWER4** 使用一个32-entry, tagless直接映射的 *count cache* 来预测软件提示的重复的目标。*count cache* 每一条能记录62位地址。当软件提示目标地址可重复的 __branch-to-link__ 和 __branch-to-count__ 指令执行时，目标地址被写入 *count cache* 。当再次取值这样指令时，目标地址通过 *count cache* 来预测。

### 6.1.1 取指

一旦 __instruction-fetch address register (IFAR)__ 被加载，I-cache没被访问，并每周期提供8条指令。每个I-cache缓存行是128B，可以提供32个指令，因此每个缓存行被分为4个相等的区域。因为I-cache缺失比较少，为了省面积，I-cache只有一个端口，每周期可以读或写一个区域。__I-cache directory (IDIR)__ 每条包含42位的真实地址(RA)，由有效地址访问(EA)。当I-cache缺失时，指令从L2以4个32B传输，最需要的区域在前两个周期传输。缓存行被写入instruction-prefetch buffer，I-cache可以继续被后续指令访问。当取值逻辑不使用I-cache时，例如发生另一个I-cache访问缺失，缓存行会被写入I-cache。这样 I-cache的写入操作可以被掩藏而不影响正常的取指操作。
EA, RA对被保存在128-entry 2路组相联的 __effective-to-real address translation (ERAT)__ 表中。**POWER4** 分别实现了 IERAT 和 DERAT，都是用有效地址(EA)访问。 每个处理器实现了一个1024-entry 4路组相联的 *TLB* 。

当指令流水线准备好接受指令时，IFAR的值被发送到I-cache, IDIR, IERAT, 和分支预测逻辑，同时IFAR被更新为下一个顺序区域的地址。下一个周期，指令从 I-cache转发到decode, crack, 和group formation的指令队列，同时从IDIR接收真实地址(RA), 从IERAT接收EA, RA对，以及分支方向预测信息。IERAT会被检查是否有有效的记录并且RA和IDIR的RA匹配。如果IERAT是无效的记录，EA必须从TLB和SLB进行翻译，取值会被暂停。假设IERAT记录是有效的，并且 IERAT的RA和IDIR的RA匹配，, I-cache访问命中。使用分支预测逻辑重新加载IFAR，然后重复上面过程。填充decode, crack, and group formation logic前面的指令队列可以允许取指逻辑提前运行，而且当发生I-cache缺失时，指令队列可以继续提供指令而不必停止后面的流水线。

如果发生 I-cache缺失，首先，instruction-prefetch buffers会被检查是否有请求的指令，如果有，会将指令发送到流水线并写入I-cache。如果instruction-prefetch buffer也不存在请求的指令，取指命令被发送到L2，L2高优先级处理指令重载传输。当数据从L2返回，会尝试写入到I-cache。除了这些取指命令，**POWER4** 会预取指令缓存行到 instruction-prefetch buffer，instruction-prefetch buffer可以保存4个32条指令。 指令预取逻辑监控取指请求，当instruction-prefetch buffer里存在请求缓存行时，会预取下一个缓存行；当instruction-prefetch buffer里不存在请求缓存行时，会预取下两个缓存行。这些预取操作需要保证I-cache里不存在预取的缓存行。预取的缓存行会被保存在instruction-prefetch buffer里，从而保证不会污染I-cache。

### 6.1.2 Decode, crack, 和group formation
一组包含5个内部指令，称为IOPs。解码阶段，指令被顺序放入一个组。最老的指令放到槽0， 余下依次放入，槽4保留给分支指令。如果必要，no-ops 被强制放入槽4。一个周期分发一个组的指令。组按照程序顺序分发，不同的IOPs被发射队列乱序发射到执行单元。当组完成时侯，结果被commit。一个周期只能完成一个组，并且只有一个组里所有指令都完成，并且更老的组已经完成，这个组才能完成。为了保证正确性，一些指令不允许投机执行，为了确保这些指令不会被投机执行，这些指令只有作为下一个完成的指令时才会被执行，这被称为completion serialization。为了简化实现，这些指令单独组成单指令组。completion serialization例子包括guarded space的存储加载指令和context synchronizing 指令，例如修改处理器状态的 __move-to-machinestate-register__ 指令。

### 6.1.3 组分发和指令发射
一次分发一个指令组到指令队列，当指令组分发时，控制信息被保存在 __group completion table (GCT)__ 。GCT可以保存20个组。GCT会记录指令组里第一条指令的地址。当指令执行结束，也会被记录到对应的GCT，记录会一直维护直到指令组退休。每个指令槽对应不同的执行单元的发射队列。定点执行单元和存储加载单元共享一个发射队列。下表列出了不同发射队列的深度和不同类型的队列的个数：

![Pasted image 20230913150523.png](/assets/images/power/Pasted image 20230913150523.png)

指令被分发到发射队列的顶端，当指令从发射队列发射出去后，队列内指令往下移动。对于两个发射队列对应一个执行单元的，两个队列间隔发射，所有的源操作数都准备好的最老的指令被发射到执行单元。当指令组分发时，指令组需要的所有资源都需要可用，不然指令组会被停住直到所有资源可用。成功的分发需要下列资源：
* GCT条目：每一个指令组需要一个GCT条目，当组退休时会被释放
* 发射队列槽： 指令组里每个指令都需要一个对应的指令队列槽，当指令被成功发射到执行单元时会被释放。有些情况下指令发射几个周期之后才能释放，例如，依赖加载操作指令的定点指令可以被投机发射，这时候并不确定加载指令是否会命中L1数据缓存。如果加载指令未命中L1数据缓存，定点指令需要撤回到发射队列直到依赖的数据被加载到寄存器。. 
* 重命名寄存器：指令组里指令必须要有相应的重命名寄存器资源。重命名资源只有在下一个写同样逻辑资源指令commit时才释放。下表总结了 **POWER4** 处理器可用的重命名资源：

![Pasted image 20230913154714.png](/assets/images/power/Pasted image 20230913154714.png)

* Load reorder queue (LRQ)条目: 指令组里加载指令必须有可用的LRQ条目，当指令组完成时被释放。LRQ一共32条目
* Store reorder queue (SRQ) 条目: 指令组里存储指令必须有可用的SRQ，当指令组完成，并且存储操作成功写到L2之后释放。SRQ一共32条目

### 6.1.4 存储加载执行单元
每一个SRQ条目关联一个store data queue (SDQ)条目，SDQ条目维护存储指令的数据直到指令组commit。一旦commit，SDQ里的数据会被写入到缓存。另外，三个特别hazards需要避免：
* Load hit store: 当对同一个内存地址的年轻的读在更老的写把数据写回缓存之前执行时必须从SDQ获取数据。当读指令执行时，会检查SRQ，是否有对同一个地址的更老的写指令，如果有，数据从SDQ转发。当数据不能转发时比如读写地址重叠但是并不完全一致时，包含读指令的指令组被遗弃并重新取指执行。如果更老的写指令还未将数据写到SDQ，读指令被拒绝，并重新发射。
* Store hit load: 当对同一个内存地址的年轻的读在知道有更老的写之前执行完时，读操作接收到数据是错误的。为了避免这种情况，当写指令执行时，需要检查LRQ，如果存在更年轻的读已经执行了，包含读指令的指令组和后续的指令组都被遗弃，并重新取指执行。写指令之后所有指令组也被遗弃。如果读写指令在同一个指令组，指令组被遗弃，并且指令组里的指令单独组成一个指令组。
* Load hit load: 对同一个内存地址的读必须保证顺序。在LRQ条目里有一个位，指示对当前条目里的数据发生了侦查。当读指令执行时，和LRQ里的所有地址进行比较，当和一个被侦查过的更年轻的写匹配时，顺序一致性可能存在问题。为了简化实现，老的读指令之后的指令组都被遗弃。如果两个读指令在同一个指令组，当前指令组被遗弃，并且指令组里的指令单独组成指令组。

## 6.2 POWER4 Core Pipeline
下图展示了 **POWER 4**处理器的流水线:

![Pasted image 20230912173004.png](/assets/images/power/Pasted image 20230912173004.png)

在MP(mapper)阶段所有依赖性被确定，资源被分配，指令组被分发到对应的发射队列。在ISS阶段, IOP被发射到对应的执行单元。在RF阶段读取对应寄存器获取源操作数。在EX阶段执行。在WB阶段写回执行的结果到对应寄存器，这个时候，指令结束执行但还未结束。至少经过Xfer和CP两个周期，所有更老的指令组已经完成并且同一个组里其他指令结束执行指令才能完成。如果指令打包成指令组速度没有取指速度快，从指令缓存里取出的进入指令缓存的指令处于D1阶段。 同样，如果没有资源分发指令组到发射队列，指令等待在MP之前；指令在ISS之前在发射队列等待；在CP之前等待完成。

两个存储加载单元流水线是一样的，称为LD/ST流水线。 访问寄存器文件之后，存储加载指令在EA周期生成有效地址。加载指令在DC周期访问DERAT，the data cache directory 和数据缓存。如果DERAT未命中，加载指令被拒绝，保留在发射队列。同时请求会发送到TLB重新加载DERAT。第一次发射最少7个周期之后，被拒绝的指令会重新发射。如果DERAT仍然未命中，指令会再次被拒绝。这个过程一直持续直到DERAT命中。如果TLB也未命中，地址转换会被投机执行，但是TLB只有在指令确定执行之后才会更新。因此，只有在触发地址缺失的指令所在的指令组是下一个完成的指令组时才会更新TLB。TLB同时支持4 KB和16 MB页表。

对于加载指令，如果L1数据缓存目录指示包含数据，数据会在fmt周期进行格式化并写入到对应的寄存器，依赖的指令也可以使用数据。依赖的指令会假设数据缓存命中并发射，这样这些指令的RF周期和加载指令的WB周期对齐。 如果L1数据缓存提示未命中，请求会发送到L2获取对应缓存行。发送的请求会被保存在load miss queue (LMQ)中。LMQ可以保存8条请求。如果LMQ满了，加载指令会被拒绝并在7个周期之后重新发射。如果已经存在同一个缓存行的请求，第二个请求会被合并到同一个LMQ条目。如果这是同一个缓存行第三个请求，加载指令会被拒绝。所有从L2返回的数据都会和LMQ进行匹配，匹配到的数据会被转发到寄存器以便对应的加载操作可以完成，同时对应LMQ条目被释放。

对于存储指令，数据被保存到SDQ，一旦存储指令对应的指令组完成，会尝试将SDQ数据写到数据缓存。如果数据已经存在L1数据缓存，修改的数据会写回到数据缓存；如果不存在，不会写回到L1。修改的数据都会写回到L2。**POWER4** 的缓存一致性点是L2。另外，L2缓存对L1是包含的，即所有L1数据都在L2中。


## 6.3 POWER4 L2 Cache
**POWER4** 上L2由两个处理器共享，下图展示了L2的逻辑视图：

![Pasted image 20230912173042.png](/assets/images/power/Pasted image 20230912173042.png)

L2由3个相同块组成，每一个都自己控制器。缓存行在3个控制器之间做哈希。每个块包括4块SRAM分区，每一个分区每两个周期能提供16B数据；4个分区每周期能提供32B数据，需要4个周期传输一个128B缓存行。数据阵列实现SECDED，并且有冗余的wordline和bitline；L2缓存目录由2个冗余的8路组相联，parity保护阵列组成。冗余的阵列除了提供备份，同时也提供了2个非阻塞的读端口，允许snoop而不影响存储加载请求。L2实现了pseudo-LRU替换算法。因为L1是写入设计，到L2的写请求最多8B，L2有2个4条目的64B的队列来合并写请求，减少到L2的请求。每个控制器里有4个一致性处理器来管理L2，每个一致性处理器处理一个请求。请求可能来自两个处理器的L1数据缓存或者取指，或者存储存储队列。一致性处理器负责:
* 负责命中时数据返回，或未命中时从fabric controller返回数据到CIU
* 更新L2目录
* 未命中时发送请求到fabric
* 当发生读未命中或写请求时，控制L2的写入
* 当一个处理器写的缓存行存在另一个处理器L1缓存时，通过CIU发送无效请求到处理器

每个L2控制器有4个侦查处理器负责管理从总线侦查到的一致性操作。当总线操作命中L2时，一个侦查处理器会负责做出相应操作。根据操作类型，L2目录里的包含位，缓存行的一致性状态，会导致：
* 发送无效请求到处理器L1数据缓存
* 从L2读取数据
* 更新缓存行在目录里的状态
* 将修改的数据写回到内存
* 将数据发送到其他L2

除了分配一个侦查处理器，L2对于所有侦查操作提供一个snoop response。对于侦查请求，L2目录会被访问以确定缓存行是否存在以及对应的一致性状态，同时侦查的地址会和当前活跃的一致性处理器比较来发现可能的地址冲突。基于这些信息，会返回对应的snoop response。
L2缓存控制器也充当两个处理器的保留站来支持 __load [double] word and reserve indexed (lwarx/ldarx)__ 和 __store [double] word conditional (stwcx/stdcx)__ 指令。每个处理器一个地址寄存器用来保存保留的地址。当 __lwarx__ 或 __ldarx__ 指令执行时会设置一个标志，当侦查到无效操作包括从其他处理器发送的对保留地址的写，或者 __stwcx__ 或 __stdcx__ 执行成功 (通过 __condition register__ 里一位来通知处理器执行结果)，会清除标志。L2实现增强的MESI一致性协议，一共7个状态：
* I (invalid state): 数据无效
* SL (shared state): 数据有效，缓存行可能在其他L2。数据可以传输到同一个MCM内的其他L2。当处理器L1数据缓存加载或指令未命中L2并且数据来自其他缓存或内存时会进入 __SL__ 状态
* S (shared state): 数据有效，缓存行可能在其他L2。数据不可以传输给其他L2。当来自同一个MCM的处理器发出的侦查读命中时进入该状态
* M (modified state): 数据有效，数据被修改且独占，数据可以传输给任意L2。当处理器发出写操作时进入该状态。
* Me (exclusive state): 数据没有修改但被独占，__Me__ 状态缓存行写出只需要无效对应标签，数据不需要写出。当处理器执行 __lwarx__ 或 __ldarx__ 指令并且缓存行时从内存获取时候会进入该状态。
* Mu (unsolicited modified state): 数据有效，数据被修改且独占，当处理器执行 __lwarx__ 或 __ldarx__ 指令并且缓存行时从其他处于 __M__ 状态L2获取时会进入该状态。
* T (tagged state): 数据有效，但是和内存比有修改，并且数据已经传输给其他缓存，在这个状态，数据不能传输给其他L2除非收到回应。当 __M__ 状态时收到侦查读时候会进入该状态。

下表列出了L2和L1可能对应的缓存状态:

![Pasted image 20230913172331.png](/assets/images/power/Pasted image 20230913172331.png)

L2系统里还有两个noncacheable units (NCU)， 分别对应两个处理器。NCUs处理分缓存的读写，以及缓存同步操作。每个NCU由NCU master 和 NCU snooper组成。
* NCU master负责来自处理器的请求，包含一个深度为4的FIFO队列，处理非缓存的写，包括memory-mapped I/O的写，和缓存以及内存屏障操作。一个深度为1的队列来处理非缓存的读操作。 片上处理器的缓存和同步操作和非缓存的写操作一样处理，不同的是不带数据。这些操作会发送到L2控制器，大部分会被L2侦查到，包括 __icbi__ , __tlbie__ ,  __translation lookaside buffer synchronize (tlbsync)__ , __enforce in-order execution of I/O (eieio)__ , __synchronize (sync)__ , __page table entry synchronize (ptesync)__ , __lsync__ , __data cache block flush (dcbf)__ , __data cache block invalidate (dcbi)__ 指令。
* NCU snooper处理来自总线的 __translation lookaside buffer invalidate entry (tlbie)__ 和 __instruction cache block invalidate (icbi)__ 。NCU snooper侦查来自总线的 __sync__ , __ptesync__ , __lsync__ , __eieio__ , __icbi__ 和 __tlbie__ 操作，并传递给处理器。

## 6.4 POWER4 L3 Cache
下图展示了L3的逻辑视图：

![Pasted image 20230912173114.png](/assets/images/power/Pasted image 20230912173114.png)

L3由控制器和数据阵列组成，控制器在**POWER4** 芯片上，包含标签目录，仲裁逻辑等。数据阵列在包含两个16MB eDRAM的L3芯片上，后面可以连接单独的内存控制器。为了物理实现和减少bank冲突，L3上的eDRAM组织成8个bank， 每个bank 2M，并分成4个4M的组。L3控制器也分成4个组，每个有两个一致性处理器，可以处理来自总线的请求，L3和内存的访问，以及更新L3标签目录。另外，每个组包含两个处理器来进行内存的踢出，无效操作和IO的DMA操作。  每两个组共享一个L3标签目录。L3是8路组相联，缓存行512B，并以128B大小和L2维持一致性。每个缓存行支持下面5个状态：
 * I (invalid state): 数据无效
 * S (shared state): 数据有效，L3只能传输给L2
 * T (tagged state): 数据有效，并且和内存数据相比是修改过的，数据可能存在其他L2或L3里
 * Trem (remote tagged state): 和T状态类似，但是数据是从其他芯片传输过来的
 * O (prefetch data state): 数据和内存里数据一致，其他L2或L3状态未知

L3缓存数据要么来自直接连接的内存，或者连接的其他处理器芯片的内存。当其中一个处理器的读请求未命中L3时，L3控制器分配一个S状态的缓存行，和L1以及L2的包含性不是必须的。因此，当L3释放数据时，并不需要无效L1或L2缓存。当本地L2弹出M或T状态数据时，L3数据进入 T或Trem状态。当收到侦查时，L3进入T或Trem状态，这样当L3写出时候. 不需要做内存地址译码。L3使用T/Trem来决定数据是否可以写入到直接连接的内存控制器，或者需要将写出操作发送到总线。处在T 或Trem状态时,L3可以给系统上任何请求提供数据。但是在S状态时, L3只能为本地L2提供数据。这样减少片间数据传输，尽量使用本地L3。当处在O状态时, 数据可以传输给任意请求者。

## 6.5 POWER4 Memory System
下图展示了**POWER4**的内存系统视图：

![Pasted image 20230912173158.png](/assets/images/power/Pasted image 20230912173158.png)

每个**POWER4** 芯片有一个可选的内存控制器连接到L3缓存后面，每个内存控制器有1或2个内存接口。每个芯片有2个8-byte总线，一个输入，一个输出。system memory interface (SMI)芯片和DRAM芯片由4个4-byte 双向总线连接，工作在400MHz内存控制器有一个64条目的读命令缓存，65条目的写命令缓存和一个16条目的写缓存队列。

## 6.6 POWER4 IO
下图展示了 **POWER4** 的IO结构，GX总线连接远程IO桥芯片。
![Pasted image 20230913093144.png](/assets/images/power/Pasted image 20230913093144.png)

## 6.7 MCM
下图展示了4个 **POWER4** 芯片通过4个逻辑总线组成8路SMP系统。

![Pasted image 20230913100716.png](/assets/images/power/Pasted image 20230913100716.png)

每个芯片写到独立的总线，并和L2，I/O控制器，L3控制器进行仲裁。每个芯片侦查所有的总线，从L2发出的读请求被所有芯片侦查到: 
* 请求数据是否在L2，一致性状态是否允许从L2传输到请求的芯片
* 请求数据是否在L3或内存，如果在，返回数据到请求的芯片

从单芯片角度看，互联拓扑类似总线；从整体看，单个芯片类似交换机。

## 6.8 32-way SMP
下图展示了4个MCMs组成更大的SMPs

![Pasted image 20230913092752.png](/assets/images/power/Pasted image 20230913092752.png)

一到四个MCMs可以进行互联，当多个MCMs互联时，中间的芯片的总线充当repeaters，将请求以环状拓扑从一个模块传输到另一个模块。和单个MCM类似，每个芯片有单独的请求/命令和数据总线，但是侦查所有总线。多个MCMs配置比单个会轻微增加10%的的内存访问延迟，但是提供了统一的内存模型，简化编程。

## 6.9 POWER4 RAS
L1缓存使用parity保护，L1数据缓存错误会被当成同步的machine-check中断。 machine-check中断处理函数实现在固件代码里以便支持错误恢复，当中断发生时，固件会保存处理器的架构状态，并检查处理器寄存器来决定恢复和错误状态。如果是可恢复的，固件会无效L1数据缓存来清除错误，并递增错误计数器。如果数据计数器比预设的阈值大，表示是一个不可恢复的错误，固件会禁止发生错误的L1数据缓存部分。固件然后恢复处理器架构状态，以fully recovered的状态回调操作系统的machine-check处理函数。操作系统检查固件返回的状态并继续执行。
L3 标签目录是ECC保护，支持SECDED。不可修正的错误会导致system checkstop。如果是可修正的错误，访问会被暂停直到错误被修正，同时会发送recovered attention message到service processor。
L3地址, 内存地址和控制总线有parity，可以发现单bit的错误。L3和内存的数据总线支持ECC的SECDED。不可修正的错误会被发送到requesting processor并导致machine-check中断。
L3 标签目录发生stuck fault或L3 cache-embedded DRAMs发生超过 line-delete控制寄存器范围的stuck faults，包含对应L3缓存的处理器芯片可以被重新配置，从逻辑上在系统里删掉而不影响系统里其他L3缓存。

# 7. POWER 5
**POWER5** 将L3直连到L2，作为victim cache，另外，**POWER5** 还集成了片上内存控制器，以提高主存访问速度。每个处理器核支持双线程，对于操作系统，**POWER5** 是一个4路SMP处理器。两个处理器共享1.875MB L2 缓存，并分为3个分区，每一个分区是10路组相联。 L3分为3个slice，分别作为L2的victim cache，每个slice是12路组相联，缓存行大小为256B，分为2个128B的sector。**POWER5** 芯片和L3通过2个16B的双向总线以1/2处理器速度相连。4 bytes宽的I/O bus称为GX bus，运行在1/3处理频率。下图展示了**POWER 5**的全芯片图：

![Pasted image 20230914085010.png](/assets/images/power/Pasted image 20230914085010.png)

## 7.1 多线程的演进
将超标量微处理器改成SMT需要做下面这些修改：
* 增加 __Program Counter (PC)__ 共享取指逻辑
* 增加GPR/FPR重命名资源，高地址位可用来识别不同线程
* 增加完成逻辑来记录不同线程
* 地址和标签中增加线程区别位

下图展示了从ST到Coarse Grain Thread, Fine Grain Thread和Simultaneous Multi Thread的演进：
![Pasted image 20230809154930.png](/assets/images/power/Pasted image 20230809154930.png)

在**POWER5**处理器中, 为了实现SMT，相比**POWER 4**：
* segment table缓存在全相联的64个条目的 __segment lookaside buffer (SLB)__ , 每个线程一个
* page table缓存在1,024个条目, 4路组相联的 __translation lookaside buffer (TLB)__
* BHT和 __count cache__ 没有因为SMT而修改
* return address stack，每个线程一个
* 4个 __instruction prefetch buffers__ ，每个线程一半，每个线程可以独立处理指令缓存缺失和指令预取
* GCT修改为 *linked list* 实现，这样每个线程可以独立分配和释放。共享的GCT保持和**POWER4**一样的条目数
* 寄存器重命名有一些小修改，每个逻辑寄存器有一个线程位，每种重命名资源增加一些
* 除了浮点发射队列从20增加到24， 其他发射队列保持不变
* 对于LRQ和SRQ，**POWER4**是32个条目，对于SMT，需要分成每个线程一个，这样会导致LRQ和SRQ容易用完。为了不增加队列物理大小，每一个队列都提供了32个虚拟队列条目。虚拟队列条目包含足够信息来分辨指令而不是存储加载指令地址和数据。这种方案以很小成本扩展了LRQ和SRQ大小，从而不会停止指令分发。当指令分发时，真实的LRQ或SRQ已满，指令可以分发到虚拟条目。当真实条目可用时，虚拟条目转换成真实条目。在存储加载指令发射时必须使用真实的LRQ和SRQ
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
* 当线程进入null状态时，操作系统将无法知道该线程。

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
__fabric bus controller (FBC)__ 控制片间的通信，互联拓扑称为分布式交换，在不同配置下保持一样的行为。高端**POWER5**系统采用MCM封装，基本单元是8个**POWER5** 芯片和8个L3芯片封装成两个MCM。MCMs, 内存, 和连接I/O drawers的桥接芯片组成一个book。下图展示了**16-way POWER5**系统：
![Pasted image 20230914090918.png](/assets/images/power/Pasted image 20230914090918.png)

**POWER5** 在MCM内部是通过两个环状总线互联，两个环方向相反。每个总线是8B位宽，运行在处理器频率。一个Book内两个MCM通过4对8B位宽，运行在1/2处理器频率的单向总线互联。一个**POWER5** book是16路SMP，对于SMT2，就是32路SMP。4个books可以互联组成64路SMP系统。每个Book中的**POWER5**芯片和边上芯片通过运行在1/2处理器频率的8B位宽总线互联。下图展示了64路SMP互联示意图：
![Pasted image 20230914091104.png](/assets/images/power/Pasted image 20230914091104.png)

也可以基于DCM来组成1-8个**POWER5**芯片系统。DCM包含一个**POWER5**芯片和一个L3芯片。下图展示了一个基于DCM的16路**POWER5** SMP系统：
![Pasted image 20230914091243.png](/assets/images/power/Pasted image 20230914091243.png)

### 7.4.1 Fabric bus controller
FBC对L2/L3, 内存控制器，MCM的片内总线和缓冲和MCM的片间总线传输进行缓冲和保序。总线地址和数据是分开的，允许乱序返回。
* **Address bus** MCM内部的address buses是8 bytes位宽，运行在1/2处理器频率，MCM之间address buses是4 bytes宽，运行在处理器频率，每个地址传输使用4个处理器周期。Address bus使用ECC，支持SECDED。地址传输包括50位真实地址(RA)，地址标签，传输类型和其他信息。MCM内部有12个点对点的地址总线，MCM之间总线使用环拓扑来广播地址，在64路SMP系统中，地址串行经过8个MCM。环上每个芯片负责转发地址到环上的下一个MCM， 并且到MCM内部其他芯片。当发起传输的芯片收到发出的地址之后，停止传播。
* **Response bus** 类似地址总线，MCM内部的response buses运行在1/2处理器频率.，MCM之间response buses运行在处理器频率。response bus是延迟之后的address bus， 包括每个处理器芯片上内存系统侦查到地址之后缓存一致性相关的操作。response bus使用parity保护。response bus在地址总线之后延迟固定的周期。一旦MCM内部源芯片收到MCM内部其他3个芯片的侦查响应，就将这些响应，自身响应以及来自环前面MCM的响应结合起来并转发到环上下一个MCM。当发起侦查的芯片收到自己发起的侦查响应之后，会生成一个合并的响应并像地址一样广播到系统中。 这个结合的响应详细列出了相应地址上应该做出的动作，比如应该进入什么状态，对于读操作应该哪一个芯片提供数据，那些必须做无效操作等。为了减少cache-to-cache传输延迟，**POWER5**增加了一个early combined response机制，允许remote chip收到地址之后从自己处于S状态的L2或L3发送数据到请求处理器 。early combined response通过和MCM侦查响应来比较找到第一个可以提供数据的一致性侦查者。
* **Data bus** 数据总线用于所有数据传输，比如缓存介入。也用于地址传输之后的数据传输，比如cast-outs, snoop pushes, 和DMA写。 数据总线使用ECC，支持SECDED。在**POWER4**系统上, 数据使用简单的环拓扑进行广播传输。在**POWER5**中，一个MCM内部的数据总线增加到8个，MCM之间也增加到8个，另外，fabric bus路由数据到指定的芯片。

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
IFU从L2加载指令到L1，这些请求要么是L1指令缓存缺失发起，要么是连续两个缓存行加载请求之后导致的指令预取。每个取值请求都是一个缓存行128B，数据以4个32B返回，最多能有32个取值请求从处理器发送到L2。从L1缓存发出的指令会被写到IDU里的指令缓冲I-buffer。**POWER6**和**POWER4**和**POWER5**一个主要差异是把很多译码和分组的功能放到预译码阶段，因此解决了L1指令缓存到指令分发的关键路径。在将指令写入L1指令缓存之前，需要4个周期的预译码和分组，增加了L2的访问延迟。**IFU**主要负责：
* Instruction recoding 在预译码阶段会记录指令信息，例如会调整一些指令的寄存器域以便指令对于执行单元保持一致并减少关键路径上逻辑。**POWER6**上最重要的指令记录发生在分支指令上，相对分支指令包含一个立即数域，是相对当前 __PC__ 的偏移值，指定了分支目标地址。预译码时会将偏移值和 __PC__ 的低位相加，消除了关键分支执行路径上的加法器。预译码也会决定 __PC__ 高位是否需要加，减来计算分支目标地址，这些信息都会编码进记录的分支指令里。绝对分支指令的分支目标地址也可以做符号扩展。条件分支指令也会编码以实现高频。 
* Instruction grouping 在预译码阶段指令会被分组并且记录到L1指令缓存，分组可以简化在关键分发路径上的分发仲裁。 同一个指令组只有同一个线程的指令，组里指令条数和分发带宽以及执行单元资源匹配，一个线程5条指令。执行的分支指令结束一个组。L1指令缓存里的每个指令有一个开始位控制在IDU里的分组。如果指令开始为被设置，指令开始一个新的组。 指令组里指令要满足：
	* 一个指令组不能使用超过**POWER6**执行单元资源：两个FXUs, 两个LSUs, 两个FP或VMX和一个分支单元
	* 一个指令组不能有对 __GPR__ , __FPR__ , __CR__ , __XER__ 的*write-after-write*依赖
	* 一个指令组不能有对 __XER__ 的*read-after-write*依赖
	* 一个指令组不能有对 __GPR__ 的*read-after-write*依赖
	* 一个指令组不能有对 __CR__ 的*read-after-write*依赖
	* 一个指令组有读目标FPR的指令跟随FMA指令
	* 一个指令组不能有对同一个FPR的浮点存储指令跟随浮点加载指令
* Branch execution **POWER6**每周期能预测8条分支指令，分支预测使用16K条目2位的BHT; 8个条目全相联的 __count cache__ ; 和一个6条目的 __link stack__ ，BHT和 __count cache__ 由两个线程共享。 每个线程可以由10条分支等待。

### 8.1.2 IDU
IDU负责指令分发，跟踪，发射和完成。在分发阶段，一个线程的指令组里的所有指令一起分发。如果两个线程的指令数不超过可用的执行单元，两个线程可以同时分发。**POWER6**每个线程能分发5条指令，两个线程一起7条指令。为了提高分发带宽，IDU使用两个并行的指令数据流，一个线程一个。每个线程有一个每周期能从指令缓存接收8条指令的I-buffer，每个线程每次最多从I-buffers取5条指令。每个线程指令然后经过非FPU和VMX依赖性跟踪，如果指令组里的指令依赖性解决，指令组被分发。否则，指令组被停在分发阶段直到解决依赖性。非FPU和VMX指令的依赖性使用目标表跟踪，一个线程一个。目标表保存指令当前在流水线里的位置信息，当指令从I-buffer进入到到分发阶段，这些指令的信息写入到目标表。随后的定点指令访问目标表来获取依赖数据，然后被分发。FPU和VMX指令的依赖性由分发阶段后面的FPQ解决。FPU和VMX算术指令被分发到FPQ，每个FPQ能保存8个指令组，每个组可以有2个FPU或VMX算术指令，FPQ每周期发射2个FPU或VMX算术指令。为了在加载指令和依赖的浮点指令之间实现零周期的加载使用，浮点指令在FPQ会额外多六个周期，以便和来自LSU的加载数据对齐。如果加载的数据不能写入FPR，数据被写入32条目的 __load target buffer__ 。__load target buffer__ 允许每个线程有16个加载指令在浮点算术指令之前执行，从而消除额外6个浮点流水周期的影响。 IDU使用一个完成表来记录执行中的大量的指令，每个线程使用10条目的完成表，每个条目可以记录32个指令的信息。当遇到要执行的分支时，预测执行的分支指令之后新分配一个完成表条目。

### 8.1.3 FXU
**POWER6**处理器实现了两个FXU来处理定点指令并为LSU生成地址。大部分定点指令一个周期执行完成，**POWER6**处理器FXU可以实现依赖指令背靠背执行而不需要转发数据到依赖的指令。

### 8.1.4 BFU
**POWER6**处理器有两个BFU，为了对齐执行周期差异，额外增加一个流水周期。和FXU类似，中间结果可以转发给依赖指令而不必做rounding和normalization。

### 8.1.5 LSU
**LSU**由存储加载地址生成和执行逻辑，L1数据缓存，地址转换，store queue, load miss queue (LMQ), 和数据预取引擎组成，实现下面功能：
	* 存储加载执行 为了实现**POWER6**高频设计，LSU执行单元采用了相对简单的数据流，保存最少的状态。大部分存储加载指令执行一个操作，一个硬件状态机用来处理 __load/store multiple and string__ 指令。对于跨缓存行的操作，内部分成2个操作。
	* L1数据缓存 **POWER6**处理器包含8路组相联的64KB的L1数据缓存，缓存行大小是128B，由4个32B的区组成。L2数据以32B大小发送，并且缓存行的无效操作也以32B大小处理，加载操作能命中部分有效的缓存行。L1数据缓存由2个端口，可以支持2个读或一个写。缓存行重加载拥有最高的优先级，并且阻塞已经分发的存储加载指令。执行的加载指令拥有第二高优先级。最后如果没有缓存行重新加载或加载指令，完成的存储指令能够从存储队列写到L1数据缓存。 L1数据缓存是store-through，所有的写操作发送到L2 cache。
	* 组预测 为了满足访问路径的周期, 实现了一个组预测。 组预测基于有效地址EA并且可以用作一个小目录来从8个L1数据缓存组里选择。虽然L1数据缓存的目录也可以使用，但是需要ERAT之后的真实地址，要花费更多时间。 组预测使用EA(51:56)索引，8路组相联，每个组或条目包含11个EA哈希位, 2个有效位（每个线程一个）和一个奇偶位,11位EA哈希值使用下面公式生成: (EA(32:39) XOR EA(40:47)) + EA(48:50)。当读缓存时，生成的EA(51:56)来索引组预测, 并且生成11位哈希值和8个组的内容比较，如果匹配并且对应线程有效位是置位，使用预测的信号访问L1数据缓存。

## 8.2 POWER6一致性协议
**POWER5**中使用的广播侦查协议中，一致性传输需要的带宽随着系统规模而增长。因此，基于目录的NUMA (nonuniform memory access)方案更有吸引力，基于目录的协议使用目录来指示内存区域的归属，因此限制广播在一个小的节点内，减少了节点外的传输。**POWER6**开发了基于目录的NUMA方式的广播协议。下表总结了**POWER6**的一致性协议中的缓存状态：
![Pasted image 20230915100402.png](/assets/images/power/Pasted image 20230915100402.png)
![Pasted image 20230915100432.png](/assets/images/power/Pasted image 20230915100432.png)
下表总结了缓存状态对应的scope-state：
![Pasted image 20230915103531.png](/assets/images/power/Pasted image 20230915103531.png)
scope-state位保存在冗余的内存中，对于128B的缓存行，该位指示缓存行是否是本地使用。该scope-state位和数据一起读写，4个新增加的缓存状态提供了一种缓存scope-state的方法。当缓存的scope state被释放时，通常是写回到内存。当scope state隐含全局状态时写回操作时必须的，当scope state是本地时，写回操作是可选的。
scope-state和4个新的缓存状态降低了基于目录的NUMA的成本，并且可以和广播一致性协议集成。**POWER6**使用很多预测器来预测一致性请求是本地还是全局广播操作。

## 8.3 POWER6缓存层次
**POWER6**处理器核包含64KB L1 I-cache和64-KB L1 D-cache分别在IFU和LSU中。每个处理器核还有4-MB L2 cache，片上两个处理器核共享32-MB L3缓存， L3缓存控制器在片上，但是数据阵列在片外，L3缓存行是128B，16路组相联。下表总结了各级缓存的特性和组织形式：
![Pasted image 20230914170540.png](/assets/images/power/Pasted image 20230914170540.png)
4-MB L2缓存是8路组相联，每个缓存行128B，并且和**POWER5**上L2缓存实现三个独立控制器不同，**POWER6**上是一个控制器。L2缓存数据阵列分成4个间隔区，每个包含一个缓存行的32B。处理器发出的缓存行读，castout读和介入读操作使用4个间隔区，但是read-modify-write和L2重新加载时writeback只使用一个缓存行的一个或多个32B的区。目录和一致性管理逻辑分成两个地址哈希区，每个目录区每两个处理器周期能接受一个处理器请求，或一个侦查请求或一根更新操作。不同于**POWER5**上的L2使用不同的读端口处理处理器请求和侦查请求，**POWER6** L2使用滑动窗口来调度侦察请求而不和处理器请求冲突，从而减少面积和功耗。
因为L1 D-cache是store-through，L2每个目录区使用一个8条目128B宽的队列来累积处理器请求。一个条目累积的所有写操作都被L2处理成read-modify-write操作，只影响使用到的间隔区。L2使用每个目录区的16个read/claim (RC) machines中的一个来处理处理器发出的所有读和L2存储队列发出的read-modify-write操作。RC machines 管理所有处理器发出的缓存读写操作。如果L2 cache缺失,L2会等待L3 cache的响应。如果L3命中，数据返回到L2。如果L3未命中，L2 cache会发出请求到SMP一致性互联总线，数据最终通过SMP一致性互联总线从其他L2, L3返回。如果新的缓存行需要释放L2中缓存行，L2会使用每个目录区4个castout machines中一个来将数据或状态转移到L3 victim cache或内存。
为了处理SMP一致性互联总线上的侦查请求，L2首先查询目录，如果需要响应，使用每个目录区4个snoop machines中一个来处理。snoop machine操作包括从缓存读取数据分发送到SMP一致性互联总线，更新目录状态或者从缓存读取数据发送给内存，并无效目录状态，或者只是更新或无效目录状态。
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
**POWER7**芯片有8个处理器核，每个核有12个执行单元，并且支持SMT4。**POWER 7**芯片有两个内存控制器，每个内存控制器支持4通道的DDR3，一共8通道内存可以提供100GB/s带宽。芯片上下是SMP接口，提供360 GB/s一致性带宽，可以支持POWER7扩展到32片。下图展示了POWER7芯片的全芯片图:
![Pasted image 20230915155142.png](/assets/images/power/Pasted image 20230915155142.png)

## 9.1POWER7 Core
**POWER7**能够动态在ST, SMT2和SMT4之间切换。处理器核主要包含6个单元：
* 取值单元IFU
* instruction-sequencing unit (ISU)
* LSU
* FXU
* VSU
* 十进制FPU (DFU)

IFU包含32-KB指令缓存, LSU包含32-KB数据缓存, 共享256-KBL2缓存。每周期能取8条指令，解码和分发6条指令，发射和执行8条指令。每个处理器核有12执行单元：2个定点单元，2个LSU, 4个双精度浮点单元，一个向量单元，一个分支单元，一个条件寄存器单元和一个十进制浮点单元DFP。两个LSU能执行简单的定点操作，每个浮点单元可以执行双精度乘加运算，一共8FLOPS。**POWER6**首次引入的DFU可以加速很多商业应用。下图展示了处理器核的规划图：
![Pasted image 20230915155118.png](/assets/images/power/Pasted image 20230915155118.png)
下图展示了**POWER7**的指令流视图：
![Pasted image 20230915155003.png](/assets/images/power/Pasted image 20230915155003.png)
为了减少面积和功耗，SMT4设计采用了partition方法，一对线程使用一套物理GPR并且使用一个FXU和LSU，另一对线程使用另一套物理GPR和另一个FXU和LSU。POWER7能够使用比支持SMT2的**POWER5**更少的物理GPR来对SMT4进行寄存器重命名。在**POWER4**和**POWER5**的乱序处理器中, GPR, FPR和VR的寄存器重命名结构是分开的。在**POWER7**中, 这些都合并成了一个80条目的寄存器重命名结构，匹配分发和完成之间并发的非分支指令数量。另外，在早期处理器中，浮点和定点还有存储加载指令的发射队列都是分开的，但是，在**POWER7**中, 这些队列都合并成一个，叫做统一发射队列UQ。为了实现高频，UQ物理上分成两个24条目的队列UQ0和UQ1。
**POWER6**里FPU和VMX是分开的，但是在**POWER7**里, 这两个合并成了 __vector and scalar unit (VSU)__ , 并采用了 __vector and scalar extension (VSX) architecture__ ，使用64个128位的架构寄存器来实现两路 __single-instruction multiple-data (SIMD)__ 。POWER7相比POWER6没有增加发射端口但是仍然支持新的VSX指令集并且每周期能执行4个浮点运算，8FLOPS。

L1数据和指令缓存使用bank设计，允许读写同时访问，一个bank同一个周期能支持2个读或一个写。

一个**POWER7**处理器核和L2及本地L3组成一个chiplet，并处于单独电源域，和相连接的PowerBus异步。

### 9.1.1 POWER7取指和译码
下图展示了**POWER7**取指和译码逻辑及流水线：
![Pasted image 20230915154836.png](/assets/images/power/Pasted image 20230915154836.png)
**POWER7**处理器核有一个4路组相联32-KB的L1指令缓存，使用了16路bank来避免读写冲突。组预测使用64条目的 __instruction effective address directory (IEADIR)__ 。__I-cache directory (IDIR)__ 同时被访问并在下一个周期确认组预测的正确性。一个64条目的 __instruction effective-to-real-address translation table(IERAT)__ 被用来加速地址转换。__IERAT__ 的前32个条目支持线程0和2，后32个条目支持线程1和2，__IERAT__ 支持4K，64K页表，并根据需要将64 MB和16 GB页表映射成64K。

IFU从L2中取指并写入L1指令缓存，每次取值请求返回4个32B。这些请求要么是L1指令缓存缺失，要么是指令预取。对每一个L1指令缺失请求，预取引擎发出额外两个顺序缓存行请求。4个线程的取值请求都是独立的，处理器核最多发出4笔并发请求。只有 ST和SMT2 支持指令预取。
指令在写入L1指令缓存之前经过两个周期进行预译码并生成奇偶校验，预译码的位用来扫描发生的分支跳转，帮助分组和几个异常情况的处理。 分支指令扫描过程中会对分支指令进行修改以便生成目标地址，修改过的包含部分计算出的目标地址的分支指令，会被保存到L1数据缓存。来自L2的32B数据在3个周期之后写入到L1指令缓存，如果线程正在等待这些指令，它们同时也会被送到 __instruction buffers (IBUF)__ 和分支扫描逻辑。

__Instruction fetch address registers (IFARs)__ 记录每个线程的 __program counter__ ，每周期 __IFAR__ 选择一个线程的PC并作为取指地址送到指令缓存和分支预测逻辑。指令缓存每周期可以提供8条指令，并写入 __IBUF__ , 随后被组装成分发组。线程优先级，缓存缺失，__IBUF__ 满和线程资源平衡都会影响线程的调度。 IBUF有20个条目，每个可保存4条指令。在SMT4模式，每个线程有5个条目，在ST和SMT2 模式，每个线程有10个条目。特殊的线程优先级逻辑每个周期选择一个线程进行分组，从线程对应的IBUF里读取最多4个非分支指令和2个分支指令并组成一个组。和**POWER4**和**POWER5**不同，分支指令不会结束一个组。分组之后，指令要么被解码，要么被送到微码硬件将复杂指令拆分成一系列内部简单操作。

分支的方向使用分支历史表来预测，由8-K条目的本地BHT (LBHT)，16-K条目的全局BHT (GBHT)和8-K条目的全局选择器(GSEL) 组成。这些预测器每周期可以对8条分支指令进行分支预测，由所有活跃线程共享。LBHT由10位取值地址索引，GBHT和GSEL由取值地址和每个线程单独的21位的 __global history vector (GHV)__ 哈希成11位的值索引。GSEL条目里的值用来选择LBHT和GBHT作为分支预测结果。所有BHT条目都是2比特, 高位决定预测的方向，低位提供hysteresis。

分支的目标地址使用下面两个机制预测：
1. 非子函数返回的间接分支指令使用所有线程共享的128条目的 __count cache__ , __count cache__ 使用取值地址和GHV异或生成7位值索引，__count cache__ 每个条目包含62位预测地址和2位置信位。当预测结果不准确时，置信位用来决定何时进行条目的替换。 
2. 子函数返回的分支使用每线程一个的 __link stack__ 来预测, __branch-and-link__ 指令被扫描到时，下一个指令的地址被放入对应线程的 __link stack__ 。当扫描到 __branch-to-link__ 指令时，__link stack__ 会被弹出。当扫描到 __branch-and-link__ 但是被之前预测错误的分支指令被刷掉时，__link stack__ 允许保存一个投机条目。在ST和SMT2模式，每个线程使用16条目的 __link stack__ ，在SMT4模式， 每个线程使用8条目的 __link stack__ 。

在ST模式，当遇到发生的分支时3个流水周期的分支扫描会产生处于空闲，为了消除这种惩罚，增加了一个 __BTAC__ 来记录直接分支的目标地址。__BTAC__ 使用当前取值的地址来预测两个周期之后的取值地址，如果预测正确，发生分支时BTAC能够提供取值地址。当条件分支用来跳过一系列的定点或存储加载指令且分支很难预测时，**POWER7** 可以检测到这样指令，并将其从指令流水线中移除，然后根据条件来执行定点或存储加载指令。条件分支被转换成内部操作，而定点或存储加载指令依赖内部操作。这样阻止了可能的错误的分支预测导致的刷流水线的操作。

### 9.1.2 POWER7 ISU
下图展示了负责分发指令，重命名寄存器，发射指令，完成指令和处理异常的ISU的逻辑视图：
![Pasted image 20230915154927.png](/assets/images/power/Pasted image 20230915154927.png)
在ST和SMT2模式，两个物理的GPR保存相同内容，线程的指令可以分发到任意一个子发射队列UQ0或UQ1，通过每周期轮换子发射队列保证两个子发射队列的负载平衡。在SMT4模式，两个物理的GPR单独使用，线程T0和T1的定点和存储加载指令被分发到UQ0, 使用GPR0, 发射到FX0和LS0； 线程T2和T3的定点和存储加载指令被分发到UQ1, 使用GPR1, 发射到FX1和LS1。

大部分VSU指令可以分发到UQ0或UQ1，不管是在ST, SMT2或SMT4模式，除了:
1. VMX的浮点以及整型指令必须分发到UQ0
2. permute (PM), 十进制浮点, 和128位存储指令必须分发到UQ1
3. 分发到UQ0的VSU指令在VS0执行
4. 分发到UQ1的VSU指令在VS1执行

**POWER7**处理器核以组作为分发单位，一次一个线程的指令组。在指令送到发射队列之前，使用mapper逻辑对寄存器重命名，GPR, VSR，XER, CR, FPSCR，link , 和 count 寄存器都会被重命名。GPR和VSR共享80个重命名寄存器，CR使用56个物理寄存器，XER使用40个物理寄存器，Link和Count寄存器使用24个物理寄存器，FPSCR使用一个20条目的缓冲来保存每个指令组的FPSCR状态。这些资源都是独立并被所有线程共享的。更新多个目的寄存器的指令被拆分成子指令。

 ISU分配 __load tag (LTAG)__ 和 __store tag (STAG)__ 来管理存储加载指令，LTAG是一个指向分配给加载指令的 __load-reorder-queue (LRQ)__ 条目的指针；STAG是一个指向分配给存储指令的  __store-reorder-queue (SRQ)__ 条目的指针；当耗尽物理的SRQ/LRQ条目后，会使用虚拟的STAG/LTAG来减少对指令分发的阻塞。当物理的LRQ被释放之后，虚拟的LTAG会被转换成真实的LTAG；当物理的SRQ被释放之后，虚拟的STAG会被转换成真实的STAG。虚拟的STAG或LTAG只有转换成真实之后才会被LSU执行。ISU可以为每个线程分配63个LTAG和63个STAG。

**POWER7**使用三个发射队列：一个48条目的UQ, 一个12条目的 __branch issue queue (BRQ)__ , 和一个8条目的 __CR queue (CRQ)__ 。分发的指令被保存在发射队列，BRQ或CRQ一个周期后发射到执行单元；UQ两个个周期后发射到执行单元。BRQ和CRQ是移动队列，分发的指令被放入队头，然后往队尾移动；为了降低功耗，UQ是非移动队列，而是移动指向队列位置的指针。发射队列可以顺序或乱序发射指令，已经准备好的老的指令有更高的优先级。 当指令的源操作数都可用之后就可以被发射了，对于存储加载指令STAG和LTAGA需要是真实条目。对于BRQ和CRQ, 通过将重命名后的目的物理指针和所有并发的指令的源物理指针比较来检查指令依赖性；对于UQ, 通过一个依赖矩阵来记录队列指针来检查依赖性。发射队列每周期一共可以发射8条指令，包括一条分支指令，一条条件寄存器逻辑操作指令，两个发射到FXU的定点指令，两个发射到LSU的存储加载指令或简单的定点指令，两个发射到VSU的向量标量指令。

BRQ每周期可以接收2条分支指令，并发射一个分支指令到IFU执行；CRQ每周期接收2条条件寄存器逻辑指令或SPR操作指令，并发射一个指令到IFU；UQ一共有48条目，物理上分成2个24条目，包括所有FXU, LSU, VSU, 和DFU执行的指令。队列上半部分包含FX0, LS0, 和VS0(含VMX整数指令)执行的指令；下半部分包含FX1, LS1, 和VS1(含DFP, VMX PM, 和VSU 128位存储指令)执行的指令，UQ0/UQ1分周期分别能接收4条指令。64位的VSU存储指令在分发阶段拆分成地址生成(AGEN)和数据操作。因为UQ是非移动的队列，指令的相对年龄通过一个age matrix决定。UQ0/UQ1每周期分别可以一条定点指令，一条存储加载指令和一个向量标量指令，一共6条指令。指令可以投机发射，比如当定点指令依赖于加载指令，而加载指令还未确定是否数据缓存或 __data effective-to-real-address translation (D-ERAT)__ 命中。当投机错误时，指令被拒绝并在几个周期之后再次发射。简单的定点指令也可以选择发射到LSU从而提高FXU的吞吐。

ISU还负责记录和完成指令，使用一个 __global completion table (GCT)__ 来记录所有分发后的指令，按指令组记录，分发和完成都以指令组为单位。GCT有20条目，所有线程共享，一个条目记录一个指令组，最多包括4个非分支指令和2个分支指令。因此，GCT最多可以记录120条分发后的指令。GCT条目里的指令组里每个指令都有一个结束位，当指令成功执行之后，结束位会被标记。当一个指令组里所有指令都标记结束且指令组是线程里最老的指令组时，指令组就可以完成了；指令组完成时，指令组里所有指令的结果都在架构上可见，__completion group tag (GTAG)__ 被广播到相关联资源以便被释放。 **POWER7** 每周期可以每线程对完成一个指令组(T0和T2是一对，T1和T3是一对)，每周期一共2个指令组。

ISU还需要处理刷流水线操作，像分支预测错误，存储加载执行发生冒险，上下文同步指令或异常处理，都可能需要将投机指令从流水线刷掉。完成单元使用20位的掩码来完成刷流水线操作，每一位对应一个组；同时当分支预测错误的分支指令不是组里第一条指令，需要从线程里刷掉后续指令组时对部分组刷流水线操作会发出 __GTAG__ ，以及4位槽掩码指示是哪些指令。

### 9.1.3 LSU微架构
LSU有两个对称的LS0和LS1流水线，每一个能够一个周期执行存储加载指令。LSU组要功能单元有：存储加载AGEN和执行, SRQ, __store data queue (SDQ)__ , LRQ, __load miss queue (LMQ)__, 包括D-ERAT, ERAT miss queue, __segment lookaside buffer (SLB)__ 和TLB的地址转换单元, 包括组预测和数据目录(DDIR)的L1数据缓存和数据预取 __prefetch request queue (PRQ)__ 引擎。下图展示了LSU的微架构：
![Pasted image 20230915154725.png](/assets/images/power/Pasted image 20230915154725.png)
* 存储加载执行 在ST和SMT2模式时，存储加载指令可以在LS0或LS1执行；在SMT4模式，线程T0和T1的指令在LS0流水线执行，线程T2和T3的指令在LS1流水线执行。存储指令发射两次，AGEN发射到LSU, 数据操作发射到FXU或VSU。LSU主要数据流包括32B的L2的重加载总线接口，到L2的16B的写数据接口，到VSU的16B加载数据接口，来自VSU的16B的写数据接口，来自FXU的8B写数据接口。**POWER7** 的L1数据缓存只有32 KB, 减少了数据缓存访问延迟。定点加载指令只有2个周期的加载使用延迟，在加载指令和依赖的定点指令之间只有一个流水线气泡；VSU加载指令有3个周期的加载使用延迟，在加载指令和依赖的VSU指令之间会产生两个流水线气泡。每个LSU流水线也可以执行定点加和逻辑指令，提高了指令发射的灵活性
* 存储加载顺序 尽管指令乱序发射和执行，LSU必须保证存储加载指令遵守架构上的编程顺序。LSU主要依赖SRQ和LRQ来实现编程顺序。SRQ是一个32条目，基于真实地址RA的 __content-addressable memory (CAM)__ 。每个线程有64虚拟条目，允许分发64个并发写；可以发射32个并发的写；SRQ由所有线程共享。SRQ条目在存储指令发射时分配，当数据写到L1数据缓存或发送到L2时释放。每个SRQ条目都有一个对应的16B的SDQ条目，存储指令每周期可以发送最多16B数据到L2。写数据可以转发给后面的读指令，即使读写指令都是投机执行。LRQ也是一个32条目，基于真实地址RA的 __content-addressable memory (CAM)__ 。每个线程有64虚拟条目，允许分发64个并发写；可以发射32个并发的写；LRQ由所有线程共享。LRQ记录读操作，检查是否有同一个地址的年轻的读在更老的读或写之前执行的冒险。当检查到这样冒险发生时，LRQ会将年轻的读和后面同一个线程的指令从流水线刷掉，加载指令重新取指并执行
* 地址转换 指令执行时，EA经过由两个64条目的DERAT缓存和一个64条目的IERAT组成第一级地址转换逻辑转换成46位的真实地址。如果ERAT未命中，由每个线程32条目的SLB和所有线程共享的512条目的TLB组成的第二级地址转换逻辑进行转换。有效地址EA首先使用线段表转换成68位虚拟地址，然后68位虚拟地址经过 __page frame table__ 转换成46位真实地址。**POWER7** 支持256 MB和1 TB两种大小的段, 以及4 KB, 64 KB, 16 MB, 和16 GB四种页表大小。
	* D-ERAT是基于CAM的全相联的缓存，物理上分成两个，每个LSU流水线一个。在ST和SMT2模式, 两个D-ERAT保持同步；在SMT4模式，T0和1共享一和LS0关联的个64条目的D-ERAT，T2和T3共享一和LS1关联的个64条目的D-ERAT。每个D-ERAT条目支持 4-KB, 64-KB, 或16-MB页表， 16 GB保存为多个16-MB页表。D-ERAT采用LRU替换算法
	* SLB是每线程32条目，全相联的基于CAM的缓冲，每个SLB条目支持256 MB或1 TB段. **POWER7**支持 __multiple pages per segment (MPSS)__ 扩展；MPSS使一个4KB基础页表的段可以同时有4-KB, 64-KB, 和16-MB页表；一个64KB基础页表的段可以同时有64-KB, 和16-MB页表。SLB由操作系统管理，当SLB缺失时发出数据或指令的段中断
	* TLB是512条目，4路组相联的缓冲，TLB由硬件管理并使用真LRU替换算法。TLB缺失时最多支持2个并发table-walk， TLB支持hit-under-miss功能，即在做table-walk时支持访问。**POWER7**处理器里, 每个TLB条目都有 __logical partition (LPAR) identity__ 标签，TLB条目里的 __LPAR identity__ 处理器里的 __LPAR identity__ 匹配
* L1数据缓存 **POWER7**里L1数据缓存时8路组相联32-KB，每个128B缓存行由4个32B组成。L2有一个专门的32B的重加载数据接口，每周期可以传输32B数据，缓存行可以以32B为单位进行命中。L1数据缓存有两个用于指令加载的读端口和一个用于存储指令或缓存行重加载的写端口。缓存行重加载的写有最高优先级，读的优先级最低。L1数据缓存物理上是4个，每个由16个bank组成，缓存的bank可以支持一个周期同时有一个写和两个读，除非是同一个bank。如果读发生和写发生bank冲突，加载指令被拒绝并重新发射。L1数据缓存是store-through，所有的写都发到L2，而不需要cast-out；写缺失时不会分配L1，而是发到L2；L1包含于L2。L1一个32B区支持字节写，最多16B。L1由有效地址EA索引，L1 D-cache目录使用LRU替换算法， 8路组相联，一共32 KB。 组预测用来减少L1数据缓存命中延迟，使用一个小目录来预测包含数据的组；组预测阵列时8路组相联，使用EA(52:56)索引，每个条目包含EA(33:51)，每个线程有效位和奇偶位哈希得到的11位结果。当读缓存时，生成的EA(52:56)来索引组预测, 并且生成11位哈希值和8个组的内容比较，如果匹配并且对应线程有效位是置位，使用预测的信号访问L1数据缓存。否则，发生缓存缺失
* 加载缺失处理 未命中L1的加载指令发出缓存行重加载请求，并释放发射队列条目，在LMQ中创建一个条目来记录缓存行的加载，同时支持将数据转发到目的寄存器。LMQ有8个条目，并使用真实地址，在所有线程间共享。LMQ支持读合并，最多两个加载指令可以关联到同一个条目，LMQ最多支持8个缓存缺失

### 9.1.4 POWER7 FXU
FXU有两个相同的FX0和FX1组成，下图展示了FX0的逻辑视图:
![Pasted image 20230915154552.png](/assets/images/power/Pasted image 20230915154552.png)
每个FXU由一个多端口的GPR，一个执行加减，比较和陷阱的ALU，一个执行旋转，移位和选择的rotator，一个count (CNT) leading zeros单元，一个执行按位permute的bit-select unit (BSU)，一个除法器，一个乘法器，和一个执行计数，奇偶，BCD指令的miscellaneous execution unit (MXU) 组成。
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
* L2 Cache 一个全相连16深度，每个32B的存储缓存用来接收处理器核发出的写，4个32B的条目组成128B可以一起分发到L2
* L3 Cache 每个4-MB L3由32个eDRAM组成，eDRAM比6T SRAM稍慢。L3区域控制器提供了一个处理器核和L2的分发端口，一个L3分发端口，两个侦查分发端口（分别对应奇偶缓存行）。L2未命中的访问会通过处理器核和L2的分发端口访问本地4-MB L3区域，如果命中则由8个read machine中一个管理。预取和一些L2写缺失也会通过处理器核和L2的分发端口访问L3，这些由24轻量级RC machine中一个处理。每个预取的数据由10个write machine处理，每个write machine有一个关联的cast-out machine来管理由于写而导致的cast-out。本地L3未命中的访问会被广播到一致性总线，并被内存，其他L2，其他L3和其他7个远端的4-MB L3侦查；在28M远端L3命中的请求通过侦查分发端口访问。

## 9.3 POWER7 SMP
**POWER7**采用了和**POWER6**同样的基于非阻塞广播的一致性协议，使用分布式管理relaxed-order-optimized multiscope。片上一致性互联通过芯片中间的奇偶仲裁逻辑来路由一致性请求的奇偶缓存行，每个周期可以授予一个even请求和一个odd请求。 一旦请求被授予，请求被广播到芯片左右两边的奇偶侦查总线。需要广播到其他芯片的请求被路由到芯片上下的多片SMP互联。从其他芯片来的请求被奇偶仲裁逻辑管理并广播到侦查总线。 
下图展示了4个**POWER7**组成的Node：
![Pasted image 20230918164544.png](/assets/images/power/Pasted image 20230918164544.png)
每个芯片有4个10B的Node内SMP互联接口，位于芯片顶端。

下图展示了由8个Node组成的SMP系统：
![Pasted image 20230725175956.png](/assets/images/power/Pasted image 20230725175956.png)
每个芯片有2个10B的node外SMP互联接口，位于芯片底部。每个Node一共8个Node外互联，使用其中7个可以组成一个8Node的SMP系统。不同于**POWER6**，数据和一致性传输带宽可以动态分配。**POWER6**引入的投机本地范围一致性广播被广泛使用，例如，将256路SMP系统分成8个Node区域可以使每个32路区域独享SMP互联和侦查带宽。

## 9.4 POWER7 RAS
下图展示了**POWER7**的RAS特性：
![Pasted image 20230918170442.png](/assets/images/power/Pasted image 20230918170442.png)
当错误被处理器核发现和上报之后, **POWER7**处理器核迅速阻止所有指令的完成，以及取指核分发。如果错误条件不足以产生checkstop, 处理器核发起恢复流程。回复流程刷掉流水线里所有的指令，并将处理器核恢复到错误发生之前的架构状态，将处理器核和系统隔离，进行自检，清除和测试SRAM，复位所有逻辑单元，读写GPR和VSR寄存器来使用ECC改正任何单比特错误。然后从恢复状态开始重新执行。

像GPR和VSR大的寄存器使用ECC保护，其他小的寄存器奇偶校验; 所有SRAM使用ECC并支持SECDEC；指令缓存使用奇偶校验。另外，**POWER7**处理器核对各种SPRs和核配置寄存器使用RAS硬化的锁存器。 L1数据缓存使用奇偶校验。当执行加载指令从L1数据缓存读数据并发现奇偶错误时会触发硬件恢复，加载指令会被刷掉，L1数据缓存内容无效，从出错的存储指令所在指令组重新开始执行。当L1数据缓存，目录或组预测阵列发生无法修正错误时，会使用组删除机制来阻止使用出错的组。

# 10. POWER 8
**POWER8**芯片有12个SMT8处理器核，每个处理器核可以分成4个区，每个分区支持2个硬件线程；处理器核每周期可以取8条指令，解码并分发8条指令，发射和执行10条指令，并commit8条指令。**POWER8**还实现了硬件的TM (Transactional Memory)。下图展示了**POWER8**处理器的全芯片规划图：
![Pasted image 20230718154255.png](/assets/images/power/Pasted image 20230718154255.png)

## 10.1 POWER 8 Core
**POWER8**处理器核由6部分组成：instruction fetch unit (IFU), instruction sequencing unit (ISU), load-store unit (LSU), fixed-point unit (FXU), vector and scalar unit (VSU) 和decimal floating point unit (DFU)。取值单元IFU里有32 KB指令缓存，存储加载单元LSU里有64 KB数据缓存，共用512 KB的L2缓存。

**POWER8**处理器核有16个执行流水线：2个定点流水线，2个存储加载流水线，2个加载流水线，4个双精度浮点流水线（或8个单精度浮点流水线），2个可以执行VMX和VSX的对称的向量流水线，一个加密流水线，一个分支流水线，一个条件寄存器逻辑流水线和一个十进制浮点流水线。 2个存储加载流水线和2个加载流水线可以执行简单的定点指令；4个双精度浮点流水线可以执行双精度乘加运算，可以每周期处理8个双精度，16个单精度，另外还可以执行64位的SIMD指令。下图展示了**POWER8**处理器核的规划图：
![Pasted image 20230718155035.png](/assets/images/power/Pasted image 20230718155035.png)

### 10.1.1 POWER8 Core指令流和流水线
下图展示了**POWER8**处理器核的指令流视图：
![Pasted image 20230918172554.png](/assets/images/power/Pasted image 20230918172554.png)
Instructions flow from the memory hierarchy through various issue queues and then are sent to the functional units for execution. Most instructions (except for branches and condition register logical instructions) are processed through the Unified Issue Queue (UniQueue), which consists of two symmetric halves (UQ0 and UQ1). There are also two copies (not shown) of the general-purpose (GPR0 and GPR1) and vector-scalar (VSR0 and VSR1) physical register files. One copy is used by instructions processed through UQ0 while the other copy is for instructions processed through UQ1. The fixed-point, floating-point, vector, load and load-store pipelines are similarly split into two sets (FX0, FP0, VSX0, VMX0, L0, LS0 in one set, and FX1, FP1, VSX1, VMX1, L1, LS1 in the other set) and each set is associated with one UniQueue half. Which issue queue, physical register file, and functional unit are used by a given instruction depends on the simultaneous multi-threading mode of the processor core at run time. In ST mode, the two physical copies of the GPR and VSR have identical contents. Instructions from the thread can be dispatched to either one of the UniQueue halves (UQ0 or UQ1). Load balance across the two UniQueue halves is maintained by dispatching alternate instructions of a given type to alternating UniQueue halves. In the SMT modes (SMT2, SMT4, SMT8), the two copies of the GPR and VSR have different contents. The threads are split into two thread sets and each thread set is restricted to using only one UniQueue half and associated registers and execution pipelines. Fixed-point, floating-point, vector and load/store instructions from even threads (T0, T2, T4, T6) can only be placed in UQ0, can only access GPR0 and VSR0, and can only be issued to FX0, LS0, L0, FP0, VSX0, and VMX0 pipelines. Fixed-point, floating-point, vector and load/store instructions from odd threads (T1, T3, T5, T7) can only be placed in UQ1, can only access GPR1 and VSR1, and can only be issued to FX1, LS1, L1, FP1, VSX1, and VMX1 pipelines. Cryptographic and decimal floating-point instructions from a thread can only be placed in the corresponding UniQueue half, but since there is only one instance of each of these units, all instructions are issued to the same unit. Branches and condition register logical instructions have their own dedicated issue queues and execution pipelines, which are shared by all threads.

下图展示了**POWER8**处理器核的流水线：
![Pasted image 20230919095043.png](/assets/images/power/Pasted image 20230919095043.png)

### 10.1.2 POWER8 IFU
The POWER8 IFU has several new features relative to the POWER7 processor IFU. Support for SMT8 and additional concurrent LPARs (logical partitions) required changes in sizes for many resources in the IFU. In addition, the following changes were made to improve the overall performance of the POWER8 core: First, instruction cache alignment improvements result in a higher average number of instructions fetched per fetch operation. Second, branch prediction mechanism improvements result in more accurate target and direction predictions. Third, group formation improvements allow more instructions per dispatch group, on average. Fourth, instruction address translation hit rates were improved. Fifth, instruction fusion is used to improve performance of certain common instruction sequences. Finally, better pipeline hazard avoidance mechanisms reduce pipeline flushes.
![Pasted image 20230918172656.png](/assets/images/power/Pasted image 20230918172656.png)
* Instruction fetching and predecoding. The POWER8 core has a dedicated 32 KB, 8-way set associative L1 I-cache. It is based on a 16-way banked design to avoid read and write collisions. A 32 x 8-entry Instruction Effective Address Directory (IEAD) provides fast prediction for way selection to choose one fetch line from the eight ways. A traditional full I-cache directory (I-dir) is accessed in parallel to confirm the way selection prediction in the next cycle. The I-cache can be addressed on any 16-byte boundary within the 128-byte cache line. Fast instruction address translation for instruction fetch is supported by a fully associative 64-entry Instruction Effective to Real Address translation Table (IERAT). The IERAT is shared among all threads. The IERAT directly supports 4 KB, 64 KB, and 16 MB page sizes. Other page sizes are supported by storing entries with the next smaller supported page size. The IFU reads instructions into the I-cache from the L2 unified cache. Each read request for instructions from the L2 returns four sectors of 32 bytes each. These reads are either demand loads that result from I-cache misses or instruction prefetches. For each demand load request, the prefetch engine initiates additional prefetches for sequential cache lines following the demand load. Demand and prefetch requests are made for all instruction threads independently, and instructions may return in any order, including interleaving of sectors for different cache lines. Up to eight instruction read requests can be outstanding from the core to the L2 cache. Instruction prefetching is supported in ST, SMT2, and SMT4 modes only. Up to three sequential lines are prefetched in ST mode and one sequential line per thread in SMT2 and SMT4 modes. There is no instruction prefetching in SMT8 mode to save on memory bandwidth. Prefetches are not guaranteed to be fetched and depending on the congestion in the POWER8 processor nest, some prefetches may be dropped. When instructions are read from the L2 cache, the IFU uses two cycles to create predecode and 奇偶 bits for each of the instructions, before they are written into the I-cache. The predecode bits are used to scan for taken branches, help group formation, and denote several exception cases. Branch instructions are modified in these stages to help generate target addresses during the branch scan process that happens during the instruction fetch stages of the pipeline. The modified branch instruction, with a partially computed target address, is stored in the I-cache. Three cycles after a 32-byte sector of instructions arrives on the I-cache/L2 interface, the sector is written into the I-cache. If the requesting thread is waiting for these instructions, they are bypassed around the I-cache to be delivered to the instruction buffers and the branch scan logic. Instruction Fetch Address Registers (IFARs) track program counter addresses for each thread. On each cycle, the IFAR register for one of the threads is selected to provide the fetch address to the I-cache complex and the branch prediction arrays. The I-cache fetch process reads quad-word aligned block of up to eight instructions per cycle from the I-cache and writes them into the instruction buffers where they are later formed into dispatch groups. Quadword-aligned fetch ensures that for a non-sequential fetch at least one instruction from the first quadword and four instructions from the second quadword are fetched as long as there is a cache hit and both quadwords are within the cache line. Thread priority, pending cache misses, instruction buffer fullness, and thread balancing metrics are used to determine which thread is selected for instruction fetching in a given cycle. The IFU allocates fetch cycles within threads of the same partition based on the priorities associated with each thread.
* Group formation. Fetched instructions are processed by the branch scan logic and are also stored in the instruction buffers (IBUF) for group formation. The IBUF can hold up to 32 entries, each four instructions wide. Each thread can have four entries in SMT8 mode, eight entries in SMT4 mode and 16 entries in SMT2 and ST modes. Instructions are retrieved from the IBUF and collected into groups. Thread priority logic selects one group of up to six non-branch and two branch instructions in ST mode or two groups (from two different threads) of up to three non-branch and one branch instructions in SMT modes per cycle for group formation.
* Instruction decode. After group formation, the instructions are either decoded or routed to microcode hardware that breaks complex instructions into a series of simple internal operations. Simple instructions are decoded and sent to dispatch. Complex instructions that can be handled by two or three simple internal operations are cracked into multiple dispatch slots. Complex instructions requiring more than three simple internal operations are handled in the microcode engine using a series of simple internal operations. Microcode handling continues until the architected instruction is fully emulated. The decode and dispatch section of the IFU also handles illegal special-purpose register (SPR) detection, creation of execution route bits and marking of instructions for debugging and performance monitoring purposes.
* Instruction fusion. For select combinations of instructions, the POWER8 core is capable of fusing two adjacent architected instructions into a single internal operation.
* Branch prediction. The POWER8 IFU supports a three-cycle branch scan mechanism that fetches 32 bytes (corresponding to eight instructions) from the I-cache, scans the fetched instructions for branches that are predicted taken, computes their target addresses (or predicts the target address for a branch-to-link or branch-to-count instruction), determines if any of these branches (in the path of execution) is unconditional or predicted taken and if so, makes the target address of the first such branch available for next fetch for the thread. It takes three cycles to obtain the next fetch address when there is a taken branch, and for two of these cycles there is no fetch for the thread. However, in SMT mode, those two cycles will normally be allocated to other active threads, and thus not lost. If the fetched instructions do not contain any branch that is unconditional or predicted taken, the next sequential address is used for the next fetch for that thread and no fetch cycles are lost. The direction of a conditional branch is predicted using a complex of Branch History Tables (BHT), consisting of a 16K-entry local BHT array (LBHT), a 16K-entry global BHT array (GBHT) and a 16K-entry global selection array (GSEL). These arrays are shared by all active threads and provide branch direction predictions for all the instructions in a fetch sector in each cycle. A fetch sector can have up to eight instructions, all of which can be branches. The LBHT is directly indexed by 14 bits from the instruction fetch address. The GBHT and GSEL arrays are indexed by the instruction fetch address hashed with a 21-bit Global History Vector (GHV) folded down to 11 bits. The value in the GSEL entry is used to choose between the LBHT and GBHT, for the direction prediction of each individual branch. All BHT entries consist of two bits with the higher order bit determining direction (taken or not taken), and the lower order bit providing hysteresis. There is one GHV for every thread in the POWER8 core to track the past branch history for that particular thread.
* Pipeline hazards. The POWER8 IFU also implements mechanisms to mitigate performance degradation associated with pipeline hazards. A Store-Hit-Load (SHL) is an out-of-order pipeline hazard condition, where an older store executes after a younger overlapping load, thus signaling that the load received stale data. The POWER8 IFU has logic to detect when this condition exists and provide control to avoid the hazard by flushing the load instruction which received stale data (and any following instructions). When a load is flushed due to detection of a SHL, the fetch address of the load is saved and the load is marked on subsequent fetches allowing the downstream logic to prevent the hazard. When a marked load instruction is observed, the downstream logic introduces an explicit register dependency for the load to ensure that it is issued after the store operation.

### 10.1.3 POWER8 ISU
The Instruction Sequencing Unit (ISU) dispatches instructions to the various issue queues, renames registers in support of out-of-order execution, issues instructions from the various issues queues to the execution pipelines, completes executing instructions, and handles exception conditions. Below Figure illustrates the logical flow of instructions in the ISU.
![Pasted image 20230918172731.png](/assets/images/power/Pasted image 20230918172731.png)
The POWER8 processor dispatches instructions on a group basis. In ST mode, it can dispatch a group of up to eight instructions per cycle. In SMT mode, it can dispatch two groups per cycle from two different threads and each group can have up to four instructions. All resouces such as the renaming registers and various queue entries must be available for the instructions in a group before the group can be dispatched. Otherwise, the group will be held at the dispatch stage. An instruction group to be dispatched can have at most two branch and six non-branch instructions from the same thread in ST mode. If there is a second branch, it will be the last instruction in the group. In SMT mode, each dispatch group can have at most one branch and three non-branch instructions.
The ISU employs a Global Completion Table (GCT) to track all in-flight instructions after dispatch. The GCT has 28 entries that are dynamically shared by all active threads. In ST mode, each GCT entry corresponds to one group of instructions. In SMT modes, each GCT entry can contain up to two dispatch groups, both from the same thread. This allows the GCT to track a maximum of 224 in-flight instructions after dispatch. Each GCT entry contains finish bits for each instruction in the group. At dispatch, the finish bits are set to reflect the valid instructions. Instructions are issued out of order and executed speculatively. When an instruction has executed successfully (without a reject), it is marked as finished. When all the instructions in a group are marked finished, and the group is the oldest for a given thread, the group can complete. When a group completes, the results of all its instructions are made architecturally visible and the resources held by its instructions are released. In SMT modes, the POWER8 core can complete one group per thread set per cycle, for a maximum total of two group completions per cycle. In ST mode, only one group, consisting of up to eight instructions, can complete per cycle. When a group is completed, a completion group tag (GTAG) is broadcast so that resources associated with the completing group can be released and reused by new instructions.

### 10.1.4 POWER8 LSU
The Load/Store Unit (LSU) is responsible for executing all the load and store instructions, managing the interface of the core with the rest of the systems through the unified L2 cache and the Non-Cacheable Unit (NCU), and implementing address translation as specified in the Power ISA. The POWER8 LSU contains two symmetric load pipelines (L0 and L1) and two symmetric load/store pipelines (LS0 and LS1). Below figure illustrates the microarchitecture of the POWER8 LS0 pipeline.
![Pasted image 20230918172837.png](/assets/images/power/Pasted image 20230918172837.png)
Each of the LS0 and LS1 pipelines are capable of executing a load or a store operation in a cycle. Furthermore, each of L0 and L1 pipelines are capable of executing a load operation in a cycle. In addition, simple fixed-point operations can also be executed in each of the four pipelines in the LSU, with a latency of three cycles. The LSU contains several subunits, including the load/store address generation (AGEN) and execution subunits, the store reorder queue (SRQ), the store data queue (SDQ), the load reorder queue (LRQ), the load miss queue (LMQ), and the L1 data cache array (D-cache) with its supporting set predict and directory arrays (DDIR), and the data prefetch engine (PRQ). The address translation mechanism in the LSU includes the Effective-to-Real Address Translation for data (DERAT), the Effective-to-Real Address Translation (ERAT) Miss Queue (EMQ), the Segment Lookaside Buffer (SLB), and TLB.
* Load/store execution. In ST mode, a given load/store instruction can execute in any appropriate pipeline: LS0, LS1, L0 and L1 for loads, LS0 and LS1 for stores. In SMT2, SMT4, and SMT8 mode, instructions from half of the threads execute in pipelines LS0 and L0, while instructions from the other half of the threads execute in pipelines LS1 and L1. Instructions are issued to the load/store unit out-of-order, with a bias towards the oldest instructions first. Stores are issued twice; an address generation operation is issued to the LS0 or LS1 pipeline, while a data operation to retrieve the contents of the register being stored is issued to the L0 or L1 pipeline. Main dataflow buses into and out of the LSU include a 64-byte reload data bus from and a 16-byte store data bus to the L2 cache, two 16-byte load data buses (one per execution pipeline) to and two 16-byte store data buses from the VSU, and two 8-byte store data buses (one per execution pipeline) from the FXU. The load data buses to the VSU have each a tap off of 8-byte load data to a corresponding FXU execution pipeline. Fixed-point loads have a three-cycle load-to-use latency on a L1 D-cache hit. That is, two cycles of bubbles are introduced between a load and a dependent FXU operation. VSU loads have a five-cycle load-to-use latency on a L1 D-cache hit. That is, four cycles of bubbles are introduced between a load and a dependent VSU operation. Each of the four LSU pipelines can also execute fixed-point add and logical instructions (simple fixed-point), allowing more fixed-point execution capability for the POWER8 core and greater flexibility to the ISU in the issuing of instructions.
* Load/store ordering. The LSU must ensure the effect of architectural program order of execution of the load and store instructions, even though the instructions can be issued and executed out-of-order. To achieve that, the LSU employs two main queues: the store reorder queue (SRQ) and the load reorder queue (LRQ). The SRQ is a 40-entry, real address based CAM structure. Whereas 128 virtual entries per thread are available to allow a total of 128 outstanding stores to be dispatched per thread, only a total of 40 outstanding stores may be issued, since a real, physical SRQ entry is required for the store to be issued. The SRQ is dynamically shared among the active threads. An SRQ entry is allocated at issue time and de-allocated after the completion point when the store is written to the L1 D-cache and/or sent to the L2 Cache. For each SRQ entry, there is a corresponding store data queue (SDQ) entry of 16 bytes. Up to 16 bytes of data for a store instruction can be sent to the L2 Cache (and also written to the L1 D-Cache on a hit) in every processor cycle. Store forwarding is supported, where data from an SRQ entry is forwarded to an inclusive, subsequent load, even if the store and load instructions are speculative. Similar to the SRQ, the LRQ is a 44-entry, real address based, CAM structure. Again, 128 virtual entries per thread are available to allow a total of 128 outstanding loads to be dispatched per thread, but only a total of 44 outstanding loads may be issued, since a real, physical LRQ entry is required for the load to be issued. The LRQ is dynamically shared among the threads. The LRQ keeps track of out-of-order loads, watching for hazards. Hazards generally exist when a younger load instruction executes out-of-order before an older load or store instruction to the same address (in part or in whole). When such a hazard is detected, the LRQ initiates a flush of the younger load instruction and all its subsequent instructions from the thread, without impacting the instructions from other threads. The load is then re-fetched from the I-cache and re-executed, ensuring proper load/store ordering.
* Address translation. During program execution, 64-bit effective addresses are translated by the first level translation into 50-bit real addresses that are used for all addressing in the cache and memory subsystem. The first level translation consists of a primary Data Effective-to-Real Address Translation (DERAT), a secondary DERAT, and an Instruction Effective-to-Real Address Translation (IERAT). When a data reference misses the primary DERAT, it looks up the address translation in the secondary DERAT. If the translation is found in the secondary DERAT, it is then loaded into the primary DERAT. If the translation is not found in either the primary or the secondary DERAT, the second-level translation process is invoked to generate the translation. When an instruction reference misses the IERAT, the second-level translation is also invoked to generate the translation. The second-level translation consists of a per-thread Segment Lookaside Buffer (SLB) and a TLB that is shared by all active threads. Effective addresses are first translated into 78-bit virtual addresses using the segment table and the 78-bit virtual addresses are then translated into 50-bit real addresses using the page frame table. While the architected segment and page frame tables are large and reside in main memory, the SLB and TLB serve as caches of the recently used entries from the segment table and page frame table, respectively. The POWER8 processor supports two segment sizes, 256 MB and 1 TB, and four page sizes: 4 KB, 64 KB, 16 MB, and 16 GB. The primary DERAT is a 48-entry, fully-associative, Content Addressed Memory (CAM) based cache. Physically, there are four identical copies of the primary DERAT, associated with the two load/store pipelines and two load pipelines. In ST mode, the four copies of the primary DERAT are kept synchronized with identical contents. So, in ST mode, logically there are a total of 48 entries available. In the SMT modes, two synchronized primary DERATs (in LS0 and L0 pipes) contain translation entries for half of the active threads while the two other synchronized primary DERATs (in LS1 and L1 pipes) contain translation entries for the other half of the active threads. In the SMT modes, the first two paired primary DERATs contain addresses that can be different from the other two paired primary DERATs, for a total of 96 logical entries. Each Primary DERAT entry translates either 4 KB, 64 KB, or 16 MB pages. The 16 GB pages are broken into 16 MB pages in the primary DERAT. The primary DERAT employs a binary tree Least Recently Used (LRU) replacement policy. The secondary DERAT is a 256-entry, fully associative, CAM-based cache. In single thread mode, all 256 entries are available for that thread. In SMT mode, the secondary DERAT is treated as two 128-entry arrays, one for each thread set. The secondary DERAT replacement policy is a simple First-In First-Out (FIFO) scheme. The SLB is a 32-entry-per-thread, fully associative, CAM-based buffer. Each SLB entry can support 256 MB or 1 TB segment sizes. The Multiple Pages Per Segment (MPSS) extension of Power ISA is supported in the POWER8 processor. With MPSS, a segment with a base page size of 4 KB can have 4 KB, 64 KB, and 16 MB pages concurrently present in the segment. For a segment with a base page size of 64 KB, pages of size 64 KB and 16 MB are allowed concurrently. The SLB is managed by supervisor code, with the processor generating a data or instruction segment interrupt when an SLB entry needed for translation is not found. The TLB is a 2,048-entry, 4-way set associative buffer. The TLB is managed by hardware, and employs a true LRU replacement policy. A miss in the TLB causes a table-walk operation, by which the TLB is reloaded from the page frame table in memory. There can be up to four concurrent outstanding table-walks for TLB misses. The TLB also provides a hit-under-miss function, where the TLB can be accessed and return translation information to the DERAT while a table-walk is in progress. In the POWER8 LSU, each TLB entry is tagged with the LPAR (logical partition) identity. For a TLB hit, the LPAR identity of the TLB entry must match the LPAR identity of the active partition running on the core. When a partition is swapped in, there is no need to explicitly invalidate the TLB entries. If a swapped-in partition has run previously on the same core, there is a chance that some of its TLB entries are still available which reduces TLB misses and improves performance.

### 10.1.5 POWER8 FXU
The Fixed-Point Unit (FXU) is composed of two identical pipelines (FX0 and FX1). As shown in Figure, each FXU pipeline consists of a multiport General Purpose Register (GPR) file, an arithmetic and logic unit (ALU) to execute add, subtract, compares and trap instructions, a rotator (ROT) to execute rotate, shift and select instructions, a count unit (CNT) to execute count leading zeros instruction, a bit select unit (BSU) to execute bit permute instruction, a miscellaneous execution unit (MXU) to execute population count, 奇偶 and binary-coded decimal assist instructions, a multiplier (MUL), and a divider (DIV).
![Pasted image 20230918172918.png](/assets/images/power/Pasted image 20230918172918.png)
At the heart of each FXU pipeline is a GPR file with 124 entries which holds all the rename and a subset of the architected registers for up to four threads. Additional architected registers are kept in the SAR register files. The GPR has eight read ports, two supplying operands for the fixed-point pipeline, two supplying operands to the load/store pipeline, two supplying operands to the load pipeline, and two supplying register data to the SAR. The GPR has six write ports: two for the fixed-point pipelines, two for the load/store pipelines, and two for the load pipelines. (Updates to a particular GPR can come from either set of fixed-point, load/store and load pipelines when the core is in ST mode.) The write ports from the remote fixed-point and load/store pipelines are shared with write ports from the SAR. In SMT modes, writes from remote pipelines are disabled and the ports can be used exclusively to load data from the SAR. The POWER8 core implements a VSU extract bus which is routed to the result multiplexer of each FXU pipe. The extract bus significantly reduces latency for VSR to GPR transfers. The contents of the two GPR register files in each pipeline are managed by the ISU to be identical in ST mode, but distinct in SMT2, SMT4, and SMT8 modes. That is, in SMT2, SMT4, or SMT8 mode the GPR in one pipeline contains the registers for one set of threads, while the GPR in the other pipeline contains the registers for the other set of threads. The POWER8 FXU supports Transactional Memory (TM) by doubling the register space to hold a backup copy of all the architected registers. Rather than doubling the size of the GPR, the SAR was added to expand the state space of the architected GPR registers. The XER, which is the other architected register in the FXU, had to grow for TM support. The XER is implemented as a Reorder Buffer (ROB) and Architected Register File (ARF) structure to accommodate the increase in state space.

### 10.1.6 POWER8 VSU
The POWER8 processor Vector-and-Scalar Unit (VSU), shown in below Figure has been completely redesigned from its initial implementation in the POWER7 processor to support the growing computation and memory bandwidth requirements of business analytics and big data applications.
![Pasted image 20230918172953.png](/assets/images/power/Pasted image 20230918172953.png)
The POWER8 VSU now supports dual issue of all scalar and vector instructions. Further improvements include a two-cycle VMX/VSX Permute (PM) pipeline latency, doubling of the store bandwidth to two 16-byte vectors/cycle to match the 32-byte/cycle load bandwidth, and execution of all floating-point compare instructions using the two-cycle Simple Unit (XS) pipeline to speedup branch execution. Other latencies remain unchanged from the POWER7 processor design point, supporting fast six-cycle bypass within the floating-point unit.
## 10.2 POWER 8 On Chip Cache

![Pasted image 20230718155501.png](/assets/images/power/Pasted image 20230718155501.png)
## 10.3 Power8 Memory Organization
 Up to 8 high speed channels, each running up to 9.6 Gb/s for up to 230 GB/s sustained
 Up to 32 total DDR ports yielding 410 GB/s peak at the DRAM
 Up to 1 TB memory capacity per fully configured processor socket
![Pasted image 20230718160154.png](/assets/images/power/Pasted image 20230718160154.png)

## 10.4 POWER8 CAPI - Coherence Attach Processor Interface
Virtual Addressing
• Accelerator can work with same memory addresses that the processors use
• Pointers de-referenced same as the host application
• Removes OS & device driver overhead
Hardware Managed Cache Coherence
• Enables the accelerator to participate in “Locks” as a normal thread Lowers Latency over IO communication model

# 11. POWER 9
**POWER9**芯片有24个SMT4处理器核或12个SMT8处理器核，每队SMT4处理器核或者一个SMT8处理器核，组成一个slice，每个slice包含512kB L2缓存和10MB L3缓存。
* SMT8 是为PowerVM优化
* SMT4 Linux优化
![Pasted image 20230907145848.png](/assets/images/power/Pasted image 20230907145848.png)

## 11.1 POWER9 Core Execution Slice Microarchitecture
A **Slice** is the basic 64-bit computing block incorporating a single Vector and Scalar Unit(**VSU**) coupled with **Load/Store Unit** (**LSU**). VSU has a heterogeneous mix of computing capabilities including integer and floating point supporting scalar and vector operations. IBM claims this setup allows for higher utilization of resources while providing efficient exchanges of data between the individual slices. Two slices coupled together make up the **Super-Slice**, a 128-bit POWER9 physical design building block. Two super-slices together along with an **Instruction Fetch Unit** (**IFU**) and an **Instruction Sequencing Unit** (**ISU**) form a single POWER9 SMT4 core. The SMT8 variant is effectively two SMT4 units.
![Pasted image 20230907160728.png](/assets/images/power/Pasted image 20230907160728.png)
## 11.2 POWER9 Core Pipeline
POWER9 modular design allowed IBM to reduce fetch-to-compute latency by 5 cycles. Additional 3 cycles were shorten from map-to-retire for floating point instructions. POWER9 furthered increased fusion and reduced the number of instructions cracked (POWER handles complex instructions by 'cracking' them into two or three simple µOPs). Instruction grouping at dispatch that was done in POWER8 has also been entirely removed from POWER9.
![Pasted image 20230907162021.png](/assets/images/power/Pasted image 20230907162021.png)
Below is a more detailed pipeline diagram
![Pasted image 20230907180319.png](/assets/images/power/Pasted image 20230907180319.png)
## 11.3 POWER9 – Core Compute
* Fetch / Branch
	* 32kB, 8-way Instruction Cache
	* 8 fetch, 6 decode
	* 1x branch execution
* Slices issue VSU and AGEN
	* 4x scalar-64b / 2x vector-128b
	* 4x load/store AGEN
* Vector Scalar Unit (VSU) Pipes
	* 4x ALU + Simple (64b)
	* 4x FP + FX-MUL + Complex (64b)
	* 2x Permute (128b)
	* 2x Quad Fixed (128b)
	* 2x Fixed Divide (64b)
	* 1x Quad FP & Decimal FP
	* 1x Cryptography
* Load Store Unit (LSU) Slices
	* 32kB, 8-way Data Cache
	* Up to 4 DW load or store
![Pasted image 20230907161654.png](/assets/images/power/Pasted image 20230907161654.png)
the core microarchitecture diagram is shown in below figure
![Pasted image 20230907180048.png](/assets/images/power/Pasted image 20230907180048.png)
## 11.4 POWER9 – Premier Acceleration Platform
**POWERAccel** is the collective name for all the interfaces and acceleration protocols provided by the POWER microarchitecture. POWER9 offers two sets of acceleration attachments: 
* PCIe Gen4 which offers 48 lanes at 192 GB/s duplex bandwidth
* 25G link which offers 96 lanes delivering up to 600 GB/s of duplex bandwidth.
On top of the two physical interfaces are a set of open standard protocols that integrated onto those signaling interfaces. The four prominent standards are:
- CAPI 2.0 - POWER9 introduces CAPI 2.0 over PCIe which quadruples the bandwidth offered by the original CAPI protocol offered in POWER8
- OpenCAPI 4.0- A new interface that runs on top of the POWER9 25G link (300 GiB/s) interface, designed for CPU-Accelerators applications
- NVLink 2.0 - High bandwidth and integration between the GPU and CPU.
- OMI - serial high bandwidth memory interface
- On-Chip Acceleration
    - 1x GZip
    - 2x 842 Compression
    - 2x AES/SHA
![Pasted image 20230907163735.png](/assets/images/power/Pasted image 20230907163735.png)

## 11.5 POWER9 Variations
* POWER9 Scale out - use direct attach memory and support 2 socket SMP
* POWER9 Scale up  - use DMI Buffered Memory and support 16 socket SMP
* POWER9 Advanced IO - use OMI buffered memory and support 16 socket SMP
![Pasted image 20230907165748.png](/assets/images/power/Pasted image 20230907165748.png)

## 11.6 POWER9 SMP
* 16 Gbps X-Bus Fully connected fabric within a central electronics complex drawer
* 25Gbps O-Bus fabric for Drawer to Drawer interconnect
A Power E980 logical system architecture is shown as below
![Pasted image 20230907174711.png](/assets/images/power/Pasted image 20230907174711.png)
A drawer is consisted of 4 POWER9. Below shows the symmetric multiprocessing (SMP) connections between nodes for 2-, 3-, and 4-drawer configurations
![Pasted image 20230907175136.png](/assets/images/power/Pasted image 20230907175136.png)
![Pasted image 20230907175155.png](/assets/images/power/Pasted image 20230907175155.png)

# 12. POWER 10
The Power10 processor is a superscalar symmetric multiprocessor that is manufactured in samsung 7-nm with 18 layers of metal. The processor contains up to 15 cores that support eight simultaneous multithreading (SMT8) independent execution contexts. Each core has private access to 2 MB L2 cache and local access to 8 MB of L3 cache capacity. The local L3 cache region of a specific core also is accessible from all other cores on the processor chip. The cores of one Power10 processor share up to 120 MB of latency optimized non-uniform cache access (NUCA) L3 cache.
The modular design of the Power10 core provides for two core variants. An “SMT8 core” supports up to eight simultaneously active threads and has double the resources of an “SMT4 core” which supports up to four simultaneously active threads. Each Power10 chip is fabricated to support either SMT8 or SMT4 cores. The SMT8 cores are designed to operate with the PowerVM firmware and hypervisor. The SMT8 cores can operate with either one partition per thread or one partition per core. The SMT4 cores can operate with one partition per thread.
![Pasted image 20230906201517.png](/assets/images/power/Pasted image 20230906201517.png)

## 12.1 Power10 Core Microarchitecture (per SMT4-Core-Resource)
Below shows a block diagram of major Power10 core organizational features. Relative sizes and capacities compared with the POWER9 core are highlighted for reference.
* Instructions are fetched from the L2 cache, 64 bytes to 32 bytes per cycle, then pre-decoded and stored in a 48 KB, 6-way, L1 instruction cache at a rate of up to 32 bytes per cycle. In each cycle, up to eight instructions are read from the L1 instruction cache or bypassed on write. Fetched instructions are scanned for branches and access a set of advanced predictors for both direction and target address prediction. Predicted taken branches redirect subsequent fetches.
	* When a fetch misses the L1 cache, the request is serviced by the L2 cache after under-going address translation. Cache misses conditionally generate up to seven cache-line prefetches based on a prefetch predictor.
	* Fetched instructions are bypassed into the decode pipeline when available. When decode is stalled or another thread is being selected for decode, the fetched instructions are buffered in a 128-entry instruction buffer.
* Instruction decode processes up to eight instructions per cycle. The 8-byte prefixed instructions each use two of the eight decode lanes. A subset of instructions are cracked into two, three, or four internal operations. A limited set of instructions are expanded with a micro-code engine that generates internal operations across multiple cycles of the decode pipeline. Instructions flagged for fusion by pre-decode are processed at instruction decode. Fused instructions can be in one of several categories.
* Up to eight internal operations or instructions are processed per cycle. Dispatch assigns execution resources as needed for each internal operation. A NOP is finished directly at dispatch and does not execute in a computational pipeline. The following resources are allocated at dispatch and when not available can cause dispatch holds.
	* Issue queue entry and slice assignment
	* Register renaming and dependency ordering
	* Load/store queue virtual entry
* Dispatched instructions are tracked in the Instruction Completion Table (ICT) until each operation has finished execution. The ICT holds up to 512 operations per SMT4-core-resource. Operations can be marked as finished either at dispatch or as they are executed in the pipelines. Operations are completed up to 64 instructions per cycle and on the granularity of two entries, called an “instruction-pair”.
![Pasted image 20230908111004.png](/assets/images/power/Pasted image 20230908111004.png)

### 12.1.1 Issue Queue
Each slice has an issue queue with 20 entries. Per SMT4-core-resource, there are four issue queues supporting a total of up to 80 operations awaiting dependency resolution.
Each issue queue is associated with a native computation slice and paired with a second slice to form a super-slice. The super-slice contains an additional set of computation pipelines as well as a load port and a store/branch port. Each cycle, the issue queue in each slice can issue three instructions
* a computational operation
* a load
* a store/branch/simple operation.
Each super-slice selects a single load and a single store/branch/simple operation for execution.
![Pasted image 20230908113518.png](/assets/images/power/Pasted image 20230908113518.png)
### 12.1.2 POWER10 Core Pipeline
The Power10 core pipeline stages show the nominal path through the load and store unit (LSU), arithmetic and logical unit (ALU), and floating-point unit (FPU) pipelines including the shortest latency data forwarding paths.
* Branch Pipeline
	* Branches are issued from an issue port shared with both store address generation and simple addition operations. Each SMT4-core-resource can resolve one branch per cycle. Branches are issued after the branch target-address source register (__ LNK__ , __ CNT__ , or __ TAR__ ), if any, is ready; even when a __ condition register (CR)__  source is still awaiting resolution. These partially executed branches awaiting __ CR__  results are held in the branch condition queues (BCQ). This enables target register dependent branches to resolve target register dependencies and extends the effective capacity of the main issue queues. Move-to and move-from operations between the target registers and GPRs are optimized for latency. The nominal latency of these operations has been reduced by sharing the physical register file between the LNK, CNT, TAR, and the GPRs. Further optimizations include dependency bypass at dispatch to completely eliminate the dependent latency between a target producing instruction and the consuming branch in some scenarios.
* Simple Pipeline
	* Add immediate instructions, such as those used for address manipulation, are supported on either the main ALU pipelines or share the simple pipeline used for some branch instructions by issuing to the store/branch/simple issue port. These operations can use either of the two simple ports per SMT4-core resource to produce a result with a nominal 2-cycle latency. A dynamic policy steers the add immediate instructions to the simple pipelines or the main ALU pipelines.
* Local/Store Pipeline
	* Load and store instructions are issued and released from the slice issue queues once operand dependencies are met and once the assigned entry is available in the load or store queue respectively. Store instructions issue with the address generation first and subsequently issue store data. Load and store pipeline hazards that require re-entry into the pipeline are handled locally by the load and store pipelines and queues. A load hazard such as a read-write cache bank conflict can be accommodated by a single cycle pipeline delay. Other hazards that require pipeline re-entry are managed by the load and store queues and are fully pipelined with operations from the main issue queues.
![Pasted image 20230908111733.png](/assets/images/power/Pasted image 20230908111733.png)
### 12.1.3 Instruction flow
The high-level pipeline segments for instruction flow are depicted. The front end of the Power10 core operates with aggressive speculation and an in-order pipeline; decoding and dispatching up to eight instructions per cycle. After instruction execution resources are assigned at dispatch, internal operations (iops) are executed out-of-order in multi-slice, super-scalar compute and load/store pipelines.
![Pasted image 20230908111421.png](/assets/images/power/Pasted image 20230908111421.png)

## 12.2 Thread and LPAR Management
With SMT8 cores, SMT4-core-resource_0 supports the even logical threads (0, 2, 4, 6) and SMT4-core resource_1 supports the odd logical threads (1, 3, 5, 7). All external interrupt lines and internal interrupts, such as decrementer, hypervisor, and door-bell interrupts, are steered to the correct resources to wake up the correct logical threads.
Within an SMT4-core-resource, the following SMT modes are supported:
* Single-thread (ST) mode: one thread active.
* SMT2 mode: two threads active.
* SMT4 mode: three or four threads active.
When the number of active threads per SMT4-core-resource moves between SMT2 and SMT4 modes, an SMT mode change procedure is executed by the hardware to re-balance resources. Threads become active via enabled interrupts and are de-activated by the stop instruction.
![Pasted image 20230908140921.png](/assets/images/power/Pasted image 20230908140921.png)
a subset of the core resources are dynamically partitioned between threads based on the SMT mode. A number of resources are shared between pairs of active threads (thread-pair). A summary of thread partitioning follows:
• Fetch is toggled between active threads while honoring dynamic thread priority.
• In SMT4 mode, decode and dispatch are split to support up to four instructions per thread-pair per cycle. Each four-instruction pipe flows independently from the others except in the case of the microcode engine, which is shared within an SMT4-core-resource.
• In SMT4 mode, issue queues, execution units, load and store execution units are divided by thread-pair, with each thread-pair using a single super-slice.
• The load miss queue and L1 caches are shared dynamically by threads within an SMT4-core-resource.
• Prefetch queues are shared dynamically for hardware detected streams and are statically partitioned based on the active thread count for software-initiated streams.

## 12.3 L2 Cache
The SMT4 L2 cache features are summarized as follows:
* 1 MB private cache per SMT4-core-resource:
	* 128-byte line, 8-way set associative.
	* Both instruction side (I-side) and data side (D-side) inclusive for a Power10 core.
	* Quad-banked cache design interleaved across a four consecutive cache-line boundary.
	* L2 cache can perform a read from one cache bank while writing to one of the other cache banks.
* 8-way directory, quad-banked multi-ported:
	* One processor read port, two snoop read ports, and one write port per physical bank.
	* The processor port operates at ½ the processor clock rate into a given bank (initiated on a 2:1 clock boundary).
	* The snoop port into a given bank operates at ½ the processor clock rate (initiated on a 2:1 clock boundary) allowing for up to four snoops per 2:1 clock across the four banks.
	* The quad-banked directory can initiate:
		* Up to five directory reads in a given 2:1 cycle (four snoop ports and one on processor port).
		* One write in a given processor clock cycle (where directory writes are scheduled on the second half of a 2:1 cycle, such that they never conflict with directory reads).
* 1024 × 13-bit LRU arrays (logical configuration).
	* 2 × 4 LRU vector tracking tree with cache invalidate state biasing.
	* Supports LRU, direct map, single-member, and pseudo-random modes.
* Point of global coherency.
* Reservation stations: one per processor thread.
* Support for Power10 Synthetic_TM core mode.
* Four snoop-bus ports selected by the cache-line “real-address” bits [55:56].
* Hardware directory line delete capabilities to support faulty L2 cache elements.

## 12.4 L3 Cache
SMT4 L3 cache features:
* Private 4 MB L3 cache; shared L3.1
* 16-way set associativity.
* 128-byte cache lines.
* Data cache consists of 4 banks of high-efficiency SRAMs with interleaving for read/write overlapping.
* 64-byte wide data bus to the L2 cache for reads.
* 64-byte wide data bus from the L2 cache for L2 castouts.
* Sixty-six, 600 kb SRAM macros; two of the SRAM macros are for redundancy.
* All cache accesses have the same latency.
* 16-way directory organized as four banks, with up to four reads or two reads and two writes every 2 pclks to differing banks. Physically implemented as eight 512 × 8 way × 50-bit SRAMs.
* LRU algorithm for victim selection using 4-bit per member utilization tracking using data type and re-use awareness.
The L3 cache has four dispatch pipes that handle incoming reflected-command (snoop) requests from the fabric. The snoop dispatch pipes perform an L3 directory read to determine how to handle the request. If the request requires the sending of intervention data, the executing of a snoop push, or the invalidating of the associated cache line, then the snoop dispatch pipe forwards the request to one of 16 snoop (SN) state machines. If the request is an incoming lateral cast-out (LCO) or a cache-injection that is accepted, the request is forwarded to both a snoop state machine (SN) and a write inject (WI) state machine.

## 12.5 SMP Interconnect
* General Features
	* Master command/data request arbitration.
	* Command requests are tagged and broadcast using a snooping protocol that enables high-speed cache-to-cache transfers.
	* Multiple command scopes are used to reduce the bus-utilizations system wide. The SMP interconnect architecture uses cache states indicating the last known location of a line (sent off chip), information maintained in the system memory (memory domain indicator [MDI] bits), a coarse grained directory that indicates when a line has gone off the chip, and combined response equations that indicate if the scope of the command is sufficient to complete the command or if a larger scope is necessary.
	* The command snoop responses specified by the SMP interconnect implementation are used to create a combined response that is broadcast to maintain system cache state coherency. Combined responses are not tagged. Instead, the order of commands from a chip source, using a specific command-broadcast scope, is the same order that combined responses are issued from that source. The order is also affected by the snoop bus usage as well.
	* Data is tagged and routed along a dynamically selected path using staging/buffering along the way to overcome data routing collisions.
	* Command throttling and retry command back-off mechanisms for livelock prevention.
	* Multiple data links between chips are supported (link aggregation).

### 12.5.1 Power10 Fabric SMP Topology
The Power10 off-chip SMP interconnect is a highly scalable, multi-tiered, fully-connected topology. The off-chip links use 18-bit high-speed differential links running up to 32 Gbps and can be configured in either a 1-hop or 2-hop configuration.
* 1-Hop SMP Topology
In the 1-hop configuration, the Power10 processor chip can fully connect up to seven other processor chips to create an eight-chip SMP system. Each chip is a group using up to seven inter-group A-links for a maximum system of eight processor chips.
![Pasted image 20230909093241.png](/assets/images/power/Pasted image 20230909093241.png)
* 2-Hop SMP Topology
In the 2-hop configuration, the Power10 processor chip can fully connect up to three other processor chips to create a four-chip group. The intra-group links are designated as __ X-links__ . Each Power10 processor in a group connects to its corresponding processor chip in each other group. Three of the inter-group __ A-links__  are provided per chip supporting a total of four groups, each containing four processor chips. A full four-group system of four chips per group comprises a maximum system of 16 processor chips.
![Pasted image 20230909093315.png](/assets/images/power/Pasted image 20230909093315.png)
### 12.5.2 Protocol and Data Routing in Multi-Chip Configurations
The SMP ports configured for coherency are used for both data and control information transport. The buses are used as follows:
1. The chip containing the master that is the source of the command issues the reflected command and the combined response to all other chips in the SMP system. Partial responses are collected and returned to the chip containing the master.
2. Data is moved point-to-point. For read operations, the chip containing the source of the data directs the data to the chip containing the master. For write operations, the chip containing the master directs the data to the LPC that performs the write operation. The routing tag contains the chip and unit identifier information for this purpose.

### 12.5.3 Power10 Coherency Flow
1-Hop Broadcast Scope Definition
![Pasted image 20230909193458.png](/assets/images/power/Pasted image 20230909193458.png)
2-Hop Broadcast Scope Definition
![Pasted image 20230909193512.png](/assets/images/power/Pasted image 20230909193512.png)
Power10 System Real-Address Map
![Pasted image 20230909193549.png](/assets/images/power/Pasted image 20230909193549.png)

## 12.6 NCU
The Power10 Non-Cacheable Unit (NCU) is responsible for processing noncacheable load and store operations, word and doubleword load and store atomic instructions (lwat, ldat, stwat, stdat), and certain other uncacheable operations such as __ tlbie__  and portions of the various __ sync__  and __ ptesync__  instructions. One NCU unit is instantiated per SMT4-core resource.
![Pasted image 20230909194305.png](/assets/images/power/Pasted image 20230909194305.png)
The Power10 NCU provides one dedicated cache-inhibited load station (LDS) per thread to process cache inhibited loads and load word or doubleword atomics (lwat, ldat). Cache-inhibited loads (whether guarded or not) and load atomics are neither gathered nor are they reordered in the Power10 implementation.
For cache-inhibited stores and store word and doubleword atomics (stwat, stdat), a store queue (STQ) consisting of sixteen 64-byte store gather stations is provided. The store gather stations are shared across the four core threads and hardware prevents any thread from blocking other threads in the store queue. A pair of 64-byte stations can “pair” together to gather up to 128 bytes.
The Power10 NCU supports gathering and reordering for cache-inhibited stores in the unguarded caching inhibited (IG = ‘10’) space. In caching-inhibited, but guarded space (IG = ‘11’), cache-inhibited stores are neither reordered nor gathered as required by the architecture. Similarly, atomic word and doubleword stores (stwat, stdat) are never gathered, but might be re-ordered.
The Power10 NCU provides eight store address machines (SAM) that manage the address tenure of the store allowing for up to eight outstanding cache-inhibited or store atomic word or doubleword instructions (stwat, stdat).
Finally, the NCU provides eight snoop queues (TLBS) to process snooped TLBIE operations and four snoop queues (SLBS) to process SLBIE operations and forward these to the core.

## 12.7 Memory Controller
The Power10 memory controller unit (MCU) provides the system memory interface between the on-chip SMP interconnect fabric and the OpenCAPI memory interface (OMI) links. These OMI links can be attached to external memory buffer chips that conform to the OpenCAPI 3.1 specification (Memory Interface Class only).

The buffer chips, in turn, directly connect to industry standard memory DIMM interfaces or other memory media (such as, storage-class memory). Each memory channel supports two OMI links. Physically, the MCUs are grouped into four instances of an extended memory OMI (EMO) unit. Each EMO unit contains two MCU channels. The MCUs process 128-byte read requests and 64-byte or 128-byte write requests from processor cores, caches, and I/O host bridges; 1 - 128-byte partial-line writes; and atomic memory operations (AMOs). The MCU also handles address-only operations for the purpose of address protection, acting as the lowest-point of coherency (LPC).

The eight MCUs on the chip can be configured into one or more address interleave groups. Within each group, the address space is divided into portions, such that each sequential portion is handled by a different MCU in a round-robin fashion. The maximum memory addressing per Power10 chip is 128 TB.
Within a single MCU channel, the two OMI sub-channels are always address interleaved on a 128-byte boundary (assuming both sub-channels are populated with memory).
below is the Power10 System Memory High-Level Diagram
![Pasted image 20230909195300.png](/assets/images/power/Pasted image 20230909195300.png)
To improve memory RAS for large systems, the memory controller supports:
* Selective Memory Mirroring - In this configuration, memory sub-channels are grouped into mirrored pairs. Separate mirrored and non-mirrored BAR registers enable memory access to be targeted to either mirrored or non-mirrored space
* Whole Memory Encryption - The memory controller supports Advanced Encryption Standard (AES) encryption/decryption of all traffic to system memory. Encryption is enabled via configuration bits accessible to firmware. Accesses to OMI configuration and MMIO spaces are never encrypted, because they are not part of the system memory media. Other than that, all traffic to system memory is encrypted (if enabled) or not.


## 12.8 On-Chip Accelerators
The Nest Accelerator unit (NX) consists of cryptographic and memory compression/decompression engines(coprocessors) with support hardware. Below figure shows a block diagram of the NX.
![Pasted image 20230911083955.png](/assets/images/power/Pasted image 20230911083955.png)
To support coprocessor invocation by user code, use of effective addresses, high-bandwidth storage accesses, and interrupt notification of job completion, NX includes the following support hardware:
* SMP interconnect unit (SIU)
	* Interfaces to SMP interconnect and direct memory access (DMA) controller, Provides 16-bytes per cycle data bandwidth per direction to both
	* Employs SMP interconnect common queue (SICQ) multiple parallel read and write machine architecture to maximize bandwidth
	* User-mode access control (UMAC) coprocessor invocation block
		* After the Virtual Accelerator Switchboard (VAS) accepts a CRB that was initiated by a copy/paste instruction, the UMAC snoops the VAS’s notification for an available coprocessor request block (CRB or job).
		* Supports one high- and one low-priority queue per coprocessor type
		* Retrieves CRBs from queues and dispatches CRBs to the DMA controller
	* Effective-to-real address translation (ERAT) table stores 32 recently used translations
* DMA Controller - Decodes CRB to initiate coprocessor and move data on behalf of coprocessors

below figure shows the Flow for NX Invocation through the VAS.
![Pasted image 20230911090237.png](/assets/images/power/Pasted image 20230911090237.png)
1. After a send window has been established, the user process can begin using the NX accelerator. First, it must create a coprocessor request block (CRB). The NX specification defines the format of the CRB. This CRB is sent to the VAS unit by using the Power10 copy/paste instructions. The copy instruction places the CRB into the copy buffer. The user process then issues a paste instruction using the effective address given by the operating system during send window creation to store the copy data to the VAS. The copy data contains the 128-byte CRB. The effective address is translated to a real address by translation hardware in the core. The store to the real address is issued to the SMP interconnect as a remote memory access write (RMA_write) command and has the send window identifier embedded within the real address. The 128-byte RMA_write payload (the CRB) is stored into one of 64 VAS data buffers. The VAS has the ability to hold 128 unique window contexts. Upon snooping the RMA_write, the VAS uses the send window identifier to fetch the Send Window Table Entry from memory if not already resident with the VAS window cache logic.
2. The VAS reads the Receive Window Identifier field in the send window context to determine which receive window the send window from the RMA_write points to. Each NX coprocessor type (CT) has a unique receive window corresponding to a unique FIFO for each of the accelerators. If the receive window is not cached, it will be fetched from memory.
3. Using the FIFO address from the receive window context, VAS stores the RMA_write payload to memory, thereby placing the CRB onto the NX accelerator FIFO. VAS stamps or overlays a portion of the CRB with the send and receive window identifiers. NX uses this information when processing the CRB. In particular, the send window identifier in the CRB is used by NX to fetch the send window and obtain translation information for the addresses contained within the CRB. The receive FIFOs are implemented as circular queues. After reaching the end of the FIFO, VAS wraps back to the beginning of the FIFO and writes the next entry.
4. After writing the CRB to the FIFO, VAS sends an ASB_notify command on the SMP interconnect. The ASB_notify contains a logical partition identifier (LPID), process identifier (PID), and thread identifier (TID).
5. Each NX FIFO has a particular LPID:PID:TID combination associated with it. When NX snoops an ASB_notify that matches its programmed LPID:PID:TID, it increments the corresponding counter for the associated FIFO, indicating a new work item has been placed on the accelerator FIFO.
6. When an NX CT queue is empty and its counter is nonzero, NX reads the next CRB from the receive FIFO. As soon as the CRB is read from the FIFO, NX does a memory mapped (MMIO) store to the VAS unit to return a credit. VAS ensures that the receive FIFO does not overflow by managing credits. The hypervisor initializes the receive window with credits equal to the number of CRBs that can be stored to the receive FIFO based on the size of the FIFO. VAS decrements the receive credit count when it stores a CRB to the receive FIFO and increments the count when NX returns a credit via MMIO store after NX pulls the CRB off of the FIFO. NX uses the stamped information from the CRB to read the send window context from memory and decrements its internal counter.
7. NX dispatches the job to the associated CT, which can have multiple acceleration engines, and executes the CRB.
8. Upon completion of the job, NX returns a send window credit to VAS via an MMIO store. Each send window, when created by the hypervisor, is assigned a number of send credits. This allows the hypervisor to implement quality of service by managing numerous users sharing the same accelerator resource, and preventing one process from using more than its share. When an RMA_write command is received by VAS, VAS decrements the send credit for the associated send window. VAS increments the count when NX completes the CRB and returns a send credit with an MMIO store.
9. NX writes a __ coprocessor status block (CSB)__  and can optionally send an interrupt, which notifies the user that the job has completed. NX also updates the __ accelerator processed byte count (XPBC)__  in the send window indicating the number of bytes that were processed on behalf of the user.


## 12.9 Nest MMU

![Pasted image 20230911094531.png](/assets/images/power/Pasted image 20230911094531.png)

## 12.10 Interrupt Controller
The Power10 interrupt controller (INT) consists of three major units:
* the virtualization controller (P3VC)
* the presentation controller (P3PC)
* the Power10 Fabric bus interface common queue (P3CQ).
These units work together to take triggers from interrupt sources and deliver exceptions to the appropriate processor thread. This section provides an overview of the interrupt architecture, describes the INT units and their interfaces, and also describes how they operate with the interrupt sources and software in the Power10 infrastructure.
The high-level diagram depicts the conceptual interaction among sources and the controller blocks in interrupt signaling and notification. The individual elements are interconnected and communicate via the Power10 Fabric bus.
* The P3VC receives notification triggers from interrupt source controllers (P3SCs) via a Power10 Fabric bus store operation (for example, a cache-inhibited write: ci_wr). It processes the notification using information contained in the event assignment entry (EAE) that is located in main memory and associated with the specific trigger. This processing can include updating an event queue entry and then forwarding the notification to the P3PC, which signals an exception to one of the processor threads. The P3VC also handles notification redistribution if a state change to the assigned processor thread preclude it from handling the interrupt or notification escalation, if there is no processor thread that is currently capable of handling the interrupt.
* The P3PC has an exception bus towards the cores to notify the individual processor threads. Three signals are created for each thread, one to generate hypervisor interrupts, another to generate operating-system interrupts, and a third to generate an event-based branch. Associated with each of the exception-notification signals in the P3PC is prioritization and exception-queuing logic that prevents less favored events from preempting more favored ones or from loss due to dropping an event. Associated with each of the exception notification wires is one or more logical server numbers stored in CAM-like lines. This structure is also referred to as the thread interrupt management area (TIMA). These logical server numbers identify which software entities are currently dispatched on the specific physical processor thread. When the P3VC issues fabric bus operations to route an event notification, these CAM-like lines are searched to identify candidate processor threads. In addition to the CAM-like lines, priority and exception-queuing logic mentioned previously, each interrupt-generating exception has logic to track how much interrupt work has been handled by the associated processor thread. This information is used to evenly distribute interrupt processing load among the candidates.
* The P3CQ serves as the Power10 Fabric bus interface controller between the interrupt logic and the rest of the Power10 chip. This unit is responsible for sequencing the appropriate fabric bus protocol when the interrupt controller drives or receives commands. It performs compares to determine if the interrupt controller is the destination of a command (for example, a store operation used for an interrupt trigger). It is also responsible for driving the fabric bus histogram, poll, and assign commands to find the correct presentation controller for an interrupt trigger. Another key P3CQ function is sending and receiving the AIB interface to the virtualization and presentation controllers.
![Pasted image 20230911102017.png](/assets/images/power/Pasted image 20230911102017.png)

# 13. 参考文献
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