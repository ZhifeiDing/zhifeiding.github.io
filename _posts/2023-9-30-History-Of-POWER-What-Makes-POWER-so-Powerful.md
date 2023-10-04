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
* 第二章 简单回顾整个POWER系列处理器，总结各代处理器的面积，功耗，缓存，IO等基本内容
* 第三章，第四章，第五章分别描述POWER 1和POWER 2的整体架构，简单介绍了POWER 3的微架构，主要是了解这些古老系统结构
* 第六章详细描述POWER 4微架构以及从单核演进到POWER4双核的进化
* 第七章详细描述POWER 5微架构，从单线程到双线程的演进，以及集成片上内存控制器
* 第八章介绍POWER 6处理器微架构，了解从之前乱序执行变为顺序执行的取舍
* 第九章介绍POWER 7处理器微架构，以及内部缓存协议状态
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

L3由控制器和数据阵列组成，控制器在**POWER4** 芯片上，包含tag目录，仲裁逻辑等。数据阵列在包含两个16MB eDRAM的L3芯片上，后面可以连接单独的内存控制器。为了物理实现和减少bank冲突，L3上的eDRAM组织成8个bank， 每个bank 2M，并分成4个4M的组。L3控制器也分成4个组，每个有两个一致性处理器，可以处理来自总线的请求，L3和内存的访问，以及更新L3标签目录。另外，每个组包含两个处理器来进行内存的踢出，无效操作和IO的DMA操作。  每两个组共享一个L3标签目录。L3是8路组相联，缓存行512B，并以128B大小和L2维持一致性。每个缓存行支持下面5个状态：
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
L3 tag目录是ECC保护，支持SECDED。不可修正的错误会导致system checkstop。如果是可修正的错误，访问会被暂停直到错误被修正，同时会发送recovered attention message到service processor。
L3地址, 内存地址和控制总线有parity，可以发现单bit的错误。L3和内存的数据总线支持ECC的SECDED。不可修正的错误会被发送到requesting processor并导致machine-check中断。
L3 tag目录发生stuck fault或L3 cache-embedded DRAMs发生超过 line-delete控制寄存器范围的stuck faults，包含对应L3缓存的处理器芯片可以被重新配置，从逻辑上在系统里删掉而不影响系统里其他L3缓存。

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
**POWER6** 是一个双核，每核双线程的顺序处理器，增加了虚拟化，十进制算术运算，向量多媒体运算功能，并且实现了Checkpoint retry 和 处理器冗余功能。*recovery unit (RU)* 保存处理器的状态，并且使用ECC保护，当发生错误时，处理器使用这些状态恢复。下图展示了**POWER6**的全芯片图:

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
Instruction dispatching, tracking, issuing, and completing are handled by the IDU. At the dispatch stage, all instructions from an instruction group of a thread are always dispatched together. Both threads can be dispatched simultaneously if all of the instructions from both threads do not exceed the total number of available execution units. The POWER6 processor can dispatch up to five instructions from each thread and up to seven instructions from both threads. In order to achieve high dispatch bandwidth, the IDU employs two parallel instruction dataflow paths, one for each thread. Each thread has an I-buffer that can receive up to eight instructions per cycle from the I-cache. Both I-buffers are read at the same time, for a maximum of five instructions per thread. Instructions from each thread then flow to the next stage, in which the non-FP unit (FPU) and VMX dependency-tracking and resource-determination logic is located. If all of the dependencies for instructions in the group are resolved, then the instruction group is dispatched. Otherwise, the group is stalled at the dispatch stage until all of the dependencies are resolved. Tracking of non-FPU and VMX instruction dependencies is performed by a target table, one per thread. The target table stores the information related to the whereabouts of a particular instruction in the execution pipe. As instructions flow from the I-buffer to the dispatch stage, the target information of those instructions is written into the target table. Subsequent FX instructions access the target table to obtain dependency data so that they can be dispatched appropriately. FPU and VMX instruction dependencies are tracked by the FPQ located downstream from the dispatch stage. FPU and VMX arithmetic instructions are dispatched to the FPQ. Each FPQ can hold eight dispatch groups, and each group can have two FPU or VMX arithmetic instructions. The FPQ can issue up to two arithmetic FPU or VMX instructions per cycle. In order to achieve zero-cycle load-to-use for load floats feeding arithmetic FPU instructions, the FPU instructions are staged six cycles after the dispatch stage through the FPQ to line up with load data coming from the LSU. If the load data cannot be written into the FPR upon arrival at the FPU, it is written into a 32-entry load target buffer (16 per thread). The load target buffer allows up to 16 load instructions per thread to execute ahead of arithmetic FP instructions, thus eliminating the effect of the six-cycle FP pipe stages. Arithmetic FPU instructions can also be issued out of order with respect to other FPU instructions. At most, eight FPU instructions can be issued out of order from the FPQ. A completion table is employed by the IDU to track a high number of instructions in flight. Each thread uses a ten-entry completion table. Each completion table entry holds the information necessary to track a cache-line’s worth of instructions (up to 32 sequential instructions). When a taken branch is detected in the fetching instruction stream, a new completion table entry is allocated for the instructions after the predicted taken branch. A new completion table entry is also allocated when an I-cache line is crossed. In effect, the completion table can track up to 320 instructions, or ten taken branches, per thread.

### 8.1.3 FXU
**POWER6**处理器实现了两个FXU来处理定点指令并为LSU生成地址。大部分定点指令一个周期执行完成，**POWER6**处理器FXU可以实现依赖指令背靠背执行而不需要转发数据到依赖的指令。

### 8.1.4 BFU
**POWER6**处理器有两个BFU，为了对齐执行周期差异，额外增加一个流水周期。和FXU类似，中间结果可以转发给依赖指令而不必做rounding和normalization。

### 8.1.5 LSU
**LSU**由存储加载地址生成和执行逻辑，L1数据缓存，地址转换，store queue, load miss queue (LMQ), 和data prefetch engine组成，实现下面功能：
	* 存储加载执行 In support of the POWER6 processor high-frequency design, the LSU execution pipeline employs a relatively simple dataflow with minimal state machines and hold states. Most load/store instructions are handled by executing a single operation. A hardware state machine is employed to assist in the handling of load/store multiple and string instructions and in the handling of a misaligned load or store. As a result, unaligned data within the 128-byte cache-line boundary is handled without any performance penalty. In the case of data straddling a cache line, the instruction is handled with two internal operations, with the partial data from each stitched together to provide the desired result.
	* L1数据缓存 The POWER6 core contains a dedicated 64-KB, eight-way, set-associative L1 D-cache. The cache-line size is 128 bytes, consisting of four sectors of 32 bytes each. The reload data bus from the L2 cache is 32 bytes. The cache line is validated on a sector basis as each 32-byte sector is returned. Loads can hit against a valid sector before the entire cache line is validated. The L1 D-cache has two ports that can support either two reads (for two loads) or one write (for a store or cache-line reload). Writes due to cache-line reloads have the highest priority and they block load/store instructions from being dispatched. Reads for executing loads have the next priority. Finally, if there are no cache-line reloads or load reads occurring, completed stores can be written from the store queue to the L1 D-cache. The L1 D-cache is a store-through design: All stores are sent to the L2 cache, and no L1 castouts are required.
	* Set预测 To meet the cycle time of the access path, a set-predict array is implemented. The set-predict array is based on the EA and is used as a minidirectory to select which one of the eight L1 D-cache sets contains the load data. Alternatively, the L1 D-cache directory array could be used, but it would take more time, as it is based on the real address (RA) and, thus, would require translation results from the ERAT. The set-predict array is organized as the L1 D-cache: indexed with EA(51:56) and eight-way set associative. Each entry or set contains 11 EA hash bits, 2 valid bits (one per thread), and a parity bit. The 11-bit EA hash is generated as follows: (EA(32:39) XOR EA(40:47)) plus EA(48:50). When a load executes, the generated EA(51:56) is used to index into the set-predict array, and EA(32:50) is hashed as described above and compared with the contents of the eight sets of the indexed entry. When an EA hash match occurs and the appropriate thread valid bit is active, the match signal is used as the set select for the L1 D-cache data.

## 8.2 POWER6一致性协议
With a broadcast-based snooping protocol such as that found in the POWER5 processor, coherence traffic and the associated bandwidth required grow proportionally with the square of the system size. As system-packaging cost implications of this bandwidth become more important, alternatives to globally snooped, broadcastbased protocols become more attractive. Approaches such as directory-based NUMA (nonuniform memory access) schemes have become popular because they localize broadcasts to small nodes with directories that indicate when regions of memory owned by a given node are checked out to other nodes. This can greatly restrict traffic flow outside the node. For POWER6 technology, it was necessary to develop a single design that incorporates a robust, global broadcast-based protocol while also integrating a capability styled after directory-based NUMA. Significant innovations have been incorporated into the coherence protocol to address this challenge. In addition to the globally broadcast request, response, and notification transport, with its distributed management using specialized cache states, a localized (or scopelimited) broadcast transport mechanism is also integrated. Thus, a given request can be broadcast globally or locally.
Below table summarizes POWER6 processor coherency protocol of cache states.
![Pasted image 20230915100402.png](/assets/images/power/Pasted image 20230915100402.png)
![Pasted image 20230915100432.png](/assets/images/power/Pasted image 20230915100432.png)
Below tables summarizes the POWER6 processor cache states and scope-state implications.
![Pasted image 20230915103531.png](/assets/images/power/Pasted image 20230915103531.png)
The scope-state bit in memory is integrated into the redundant content for error correction already stored in memory, so no cost is added. For each 128-byte cache line, the bit indicates whether the line might be in use outside of the local scope where the memory resides. Since it is stored with the data bits, the state bit is automatically read or written whenever the data is read or written. The four new cache states provide a means of caching the scope-state bit in the L2 and L3 caches, either by itself or along with the data it covers. Note that when cached scope state is deallocated, it is typically cast out (i.e., written back) to memory. For cases in which the implied scope state might be global, the castout is functionally required to ensure that coherence is maintained. For cases in which the implied scope state is known to be local, the castout is optional, as it is desirable but not necessary to localize the broadcast scope for subsequent operations.
The combination of the scope-state bit in memory and the four new cache states provides a low-cost alternative to a NUMA directory and integrates cleanly into the nonblocking-broadcast distributed-coherence protocol. As some workloads localize well and others do not, the design of the POWER6 processor incorporates a number of predictors to determine whether a given coherence request should make a local attempt or immediately broadcast globally. For workloads that exhibit a high degree of processor-to-memory localization, and for workloads that have varying mixtures of locally resolvable traffic, laboratory results show that scope limited speculative snoop resolution is highly effective.

## 8.3 POWER6缓存层次
**POWER6**处理器核包含64KB L1 I-cache和64-KB L1 D-cache分别在IFU和LSU中。每个处理器核还有4-MB L2 cache，片上两个处理器核共享32-MB L3缓存， L3缓存控制器在片上，但是数据阵列在片外。下表总结了各级缓存的特性和组织形式：
![Pasted image 20230914170540.png](/assets/images/power/Pasted image 20230914170540.png)
4-MB L2缓存是8路组相联，每个缓存行128B，并且和**POWER5**上L2缓存实现三个独立控制器不同，**POWER6**上是一个控制器。L2缓存数据阵列分成4个间隔区，每个包含一个缓存行的32B。处理器发出的缓存行读，castout读和介入读操作使用4个间隔区，但是read-modify-write和L2重新加载时writeback只使用一个缓存行的一个或多个32B的区。目录和一致性管理逻辑分成两个地址哈希区，每个目录区每两个处理器周期能接受一个处理器请求，或一个侦查请求或一根更新操作。不同于**POWER5**上的L2使用不同的读端口处理处理器请求和侦查请求，**POWER6** L2使用滑动窗口来调度侦察请求而不和处理器请求冲突，从而减少面积和功耗。
Because the L1 D-cache is a store-through design in order to reduce accesses to the cache arrays, the L2 accumulates individual core store operations by employing an eight-entry, 128-byte-wide queue per directory slice. All stores gathered by a given entry are presented to the L2 cache with a single read-modify-write operation. This operation uses only the cache interleaves that are affected. To handle all fetch operations initiated by the core and read-modify-write operations initiated by the L2 store queue, the L2 can employ one of 16 read/claim (RC) machines per directory slice. RC machines manage coherence transactions for all core-initiated cacheable operations. The 32 total machines are needed to enable a sufficient number of outstanding prefetch operations to drive the memory interface to saturation while still handling fetch and store traffic. If a fetch operation hits in the L2 cache, the cache interleaves are read, and data is forwarded to the core. If a read-modify-write operation hits in the L2 cache, the impacted cache interleaves are updated.
If either of these operations misses the L2 cache, the L2 waits for a response from the L3 cache. If the operation hits in the L3, data is returned to the L2 (and possibly independently to the core). If the operation misses the L3, the L2 cache sends a request to the SMP coherence interconnect fabric, and data is eventually returned from another L2, L3, or memory via the SMP data interconnect fabric. For cases in which these operations result in the deallocation of a cache line from the L2 in order to install the newly requested cache line, the L2 must additionally employ one of four castout machines per directory slice to move data or state to the L3 victim cache or to memory. The eight total machines are needed to enable sufficient outstanding castout operations to drive the L3 cache write interface to saturation.
If data is returned from the L3 cache or via the SMP data interconnect fabric, in the case of a fetch, it is forwarded to the core and written to the L2 cache; in the case of a read-modify-write, it is written directly to the nonimpacted L2 cache interleaves and merged with store data prior to writing to the impacted L2 cache interleaves. To handle incoming snoop requests from the SMP coherence interconnect fabric, the L2 first consults the directory to determine whether it is required to take any action. If so, it can employ one of four snoop machines per directory slice to perform the required task. The eight total machines are needed to enable enough outstanding interventions to drive the datapath that lies between a likely pair of L2 caches to saturation. A snoop machine task might involve reading the data from the cache to send it, via the SMP data interconnect fabric, to another processor and possibly updating the directory state; or reading the data from the cache to send it to memory and invalidating the directory state; or simply updating or invalidating the directory state.
Since the L2 is private and there are two such L2 caches on a chip, it is possible that the data requested by a given core may be found in the L2 cache associated with the other core, leading to an intervention of data between two L2 caches on the same chip. To improve latency in such a scenario, a high-speed cache-to-cache interface was created. Whenever a fetch operation initiated by a core checks its own L2 directory, the operation is forwarded to the other L2 as well. If the operation misses in its own L2, the other L2 directory is checked, and if the line is found, it is read from the other L2 cache and forwarded on a high-speed interface back to the requesting core and L2. This significantly reduces the latency as compared with a normal intervention case.
The 32-MB POWER6 processor L3 cache, which is shared by both cores on a given chip, is 16-way associative and is composed of 128-byte lines. To handle all fetch and store operations initiated by the core that miss the L2 and must check the L3 cache, the L3 can employ one of eight read machines per directory slice, private to each L2 cache. The 16 total machines per L2 cache are needed to enable enough outstanding L3 fetch hit operations to drive the L3 cache read interface to saturation. If the data resides in the L3 cache, the read machine retrieves it and routes it back to the requesting L2 cache and core. Otherwise, a response is sent to the requesting L2 cache indicating that it should route the request to the SMP coherent interconnect fabric. To handle L2 castout write operations, the L3 can employ one of eight write machines per slice, shared by both L2 caches. The number of write machines is tied to the number of castout machines in order to reduce complexity and functional overhead. For cases in which an L2 castout write to the L3 results in the deallocation of a cache line from the L3 and a copy of the line must be moved to memory, the L3 must additionally employ an L3 castout machine associated with the write machine to move data and state to memory. The 16 total machines are needed to enable enough outstanding L3 castout operations to drive the memory write interface to saturation. Since the rule of exclusivity governing the relationship between the POWER5 processor L2 and L3 caches does not carry over to the POWER6 processor design, there are cases in which an L2 castout write must merge state information with a copy of the same cache line in the L3 cache. To handle incoming snoop requests from the SMP coherence interconnect fabric, the L3 first consults the directory to determine whether it is required to take any action with respect to the snoop operation. If so, it can employ one of four snoop machines per directory subslice to perform the required task. The reason for the number of machines and the possible tasks were described above in the previous section.

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
**POWER6**处理器内部RU包含处理器状态备份，处理器内部的状态检查点不断保存在RU里，而且数据使用ECC保护。L1使用parity，L2和L3使用ECC保护。当发生错误时，处理器被停止并阻止和外部通信，上一次成功执行的指令的检查点数据被从RU里读出并恢复到对应处理器上，L1会被清空。处理器从恢复的检查点继续执行。如果是无法修复错误，RU里的数据会被传输到系统中空闲的处理器上并继续执行。

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

The IFU contains a 32-KB instruction cache (I-cache), and the LSU contains a 32-KB data cache (D-cache), which are each backed up by a tightly integrated 256-KB unified L2 cache. In a given cycle, the core can fetch up to eight instructions, decode and dispatch up to six instructions, and issue and execute up to eight instructions. There are 12 execution units within the core, i.e., two fixed point, two LS, four double-precision (DP) floating-point pipelines, one vector, one branch, one condition register (CR) logical, and one decimal floating-point (DFP) pipeline. The two LS pipes have the additional capability to execute simple FX operations. Each of the four floating-point pipelines is capable of executing DP multiply–add operations, accounting for eight FLOPs per cycle per core. The DFU, which is first introduced in POWER6, accelerates many commercial applications.
![Pasted image 20230915155118.png](/assets/images/power/Pasted image 20230915155118.png)
Below shows the POWER7 processor core instruction flow.
![Pasted image 20230915155003.png](/assets/images/power/Pasted image 20230915155003.png)
To reduce power and area, a partitioned approach to the SMT4 design was incorporated. With this approach, a pair of threads is supported from one physical general-purpose register (GPR) file that feeds one fixed-point unit (FXU) pipeline and one load/store unit (LSU) pipeline, and another pair of threads is supported from a separate physical GPR file that feeds a separate FXU pipeline and LSU pipeline. With this approach, POWER7 can efficiently rename registers for twice as many threads with a total of physical GPR file entries that is less than that of POWER5*, which only supported SMT2.
In earlier out-of-order machines, such as POWER4 and POWER5, the register rename structure for the GPR, floating-point register (FPR), and vector register (VR) was separate, which required a large number of entries. In POWER7, these were all merged into one unified rename structure with a total of 80 entries, matching the maximum number of outstanding nonbranch instructions between instruction dispatch and completion. This significantly reduces the area and power of the out-of-order machine. In addition, in earlier machines, the issue queues for floating-point instructions and fixed-point (FX) (along with load and store) instructions were separate. In POWER7, these have been combined to reduce the area and power. The new issue queue is called unified issue queue (UQ). To achieve high frequency, the large UQ is physically implemented as two 24-entry queues, i.e., UQ0 and UQ1.
The floating-point unit (FPU) and the vector media extension (VMX) unit were separate in the POWER6 design. In POWER7, these two units are merged into one unit called the vector and scalar unit (VSU), which also incorporates the new vector and scalar extension (VSX) architecture that allows two-way single-instruction multiple-data (SIMD) FLOPs out of a 64-entry architected register file, with 128 bits per entry. POWER7 did not increase the number of issue ports over POWER6 but still supports the new VSX instruction sets and can execute four FX operations and eight FLOPs per cycle.
Both of the Level 1 (L1) instruction and data caches are highly banked, which allows concurrent read and write accesses to the cache, whereas an individual bank can only support either two reads or one write in a given cycle. This significantly reduces the area and power for the caches, while most of the time reads and writes (as long as they go to different banks) can occur concurrently.
Each POWER7 chiplet (i.e., a POWER7 core with its Level 2 (L2) and local L3 cache) is designed to be on a separate power domain, with asynchronous boundary with the PowerBus to which it is connected. This allows each chiplet to have independent voltage and frequency slewing for advanced power management.

### 9.1.1 POWER7 instruction fetch and decode pipe stages
The POWER7 core has a dedicated 32-KB four-way set-associative I-cache. It is a 16-way banked design to avoid read and write collisions. Late select of the four ways is predicted using a 64-entry instruction effective address directory (IEADIR), which provides fast prediction for way selection to choose a fetch line from the four ways. A traditional full I-cache directory (IDIR) is also accessed in parallel to confirm the set selection prediction in the next cycle. Fast address translation is supported by a 64-entry instruction effective-to-real-address translation (IERAT) table. The IERAT supports threads 0 and 2 in the first 32 entries and threads 1 and 3 in the bottom 32 entries. The IERAT directly supports 4 and 64 KB, and larger pages (64 MB and 16 GB) are supported by dynamically mapping them into 64-KB pages as needed.
The IFU fetches instructions into the L1 I-cache from the L2 unified cache. Each fetch request for instructions from the L2 returns as four sectors of 32 bytes each. These fetches are either demand fetches that result from L1 I-cache misses or instruction prefetches. For each demand fetch request, the prefetch engine initiates up to two additional L2 prefetches for the two sequential cache lines following the demand fetch. Demand and prefetch requests are made for all four instruction threads independently, and data may return in any order, including interleaving of sectors for different cache lines. Up to four instruction fetch requests can be outstanding from the core to the L2 cache. Instruction prefetching is supported in the ST and SMT2 modes only. Up to two sequential lines are allowed to be prefetched in the ST mode and one per thread in the SMT2 mode.
When instructions are fetched from the memory subsystem, two cycles are taken to create predecode bits and parity for each of the instructions, before the instructions are written into the L1 I-cache. The predecode bits are used to scan for taken branches, help group formation, and denote several exception cases. Branch instructions are modified in these stages to help generate target addresses during the branch scan process. The modified branch instruction, with a partially computed target address, is stored in the L1 I-cache. Three cycles after the data arrives on the L2 interface, the 32 bytes are written into the I-cache. If the requesting thread is waiting for these instructions, they are bypassed around the cache to be delivered to the instruction buffers (IBUFs) and the branch scan logic.
Instruction fetch address registers (IFARs) track program counter addresses for each thread. On each cycle, the IFAR for one of the threads is selected to provide the fetch address to the I-cache complex and the branch prediction arrays. The I-cache fetch process reads up to eight instructions per cycle from the I-cache and writes them into the IBUFs where they are later formed into dispatch groups. Thread priority, cache miss pending, IBUF fullness, and thread balancing metrics are used to determine which thread is selected for fetching in a given cycle.
The direction of a conditional branch is predicted using a complex of branch history tables (BHTs), consisting of an 8-K entry local BHT (LBHT) array, a 16-K entry global BHT (GBHT) array, and an 8-K entry global selection (GSEL) array. These arrays together provide branch direction predictions for all the instructions in a fetch group in each cycle. A fetch group can have up to eight instructions, all of which can be branches. These arrays are shared by all active threads. The local array is directly indexed by 10 bits from the instruction fetch address. The GBHT and GSEL arrays are indexed by the instruction fetch address hashed with a 21-bit global history vector (GHV) folded down to 11 bits, one per thread. The value in the GSEL entry is used to choose between the LBHT and the GBHT for the direction prediction of each individual branch. All the BHT entries consist of 2 bits, with the higher order bit determining direction (taken or not taken) and the lower order bit providing hysteresis.
Branch target addresses are predicted using the following two mechanisms: 1) Indirect branches that are not subroutine returns are predicted using a 128-entry count cache, which are shared by all active threads. The count cache is indexed using an address obtained by doing an XOR of 7 bits, each from the instruction fetch address and the GHV. Each entry in the count cache contains a 62-bit predicted address along with two confidence bits. The confidence bits are used to determine when an entry is replaced if an indirect branch prediction is incorrect. 2) Subroutine returns are predicted using a link stack, one per thread. Whenever a branch-and-link instruction is scanned, the address of the next instruction is pushed down in the link stack for that thread. The link stack is popped whenever a branch-to-link instruction is scanned. The POWER7 link stack allows for one speculative entry to be saved in the case where a branch-and-link instruction is scanned and then flushed due to a mispredicted branch that appeared earlier in the program order. In the ST and SMT2 modes, each thread uses a 16-entry link stack. In the SMT4 mode, each thread uses an eight-entry link stack. In the ST mode, when a taken branch is encountered, the three-cycle branch scan causes two dead cycles where no instruction fetch takes place. To mitigate the penalty incurred by taken branches, a BTAC was added to track the targets of direct branches. The BTAC uses the current fetch address to predict the fetch address two cycles in the future. When correct, the pipelined BTAC will provide a seamless stream of fetch addresses that can handle a taken branch in every cycle. If the effect of a conditional branch is only to conditionally skip over a subsequent FX or LS instruction and the branch is highly unpredictable, POWER7 can often detect such a branch, remove it from the instruction pipeline, and conditionally execute the FX or LS instruction. The conditional branch is converted to an internal resolve operation, and the subsequent FX or LS instruction is made dependent on the resolve operation. When the condition is resolved, depending on the taken or not-taken determination of the condition, the FX or LS instruction is either executed or ignored. This may cause a delayed issue of the FX or LS instruction, but it prevents a potential pipeline flush due to a mispredicted branch.
Fetched instructions go to the branch scan logic and to the IBUFs. An IBUF can hold up to 20 entries, each four instructions wide. In the SMT4 mode, each thread can have five entries, whereas, in ST and SMT2 modes, a thread can have ten entries. Special thread priority logic selects one thread per cycle for group formation. Groups are formed by reading a maximum of four nonbranches and two branches from the IBUF of the thread. Unlike the POWER4 and POWER5 processors, branches do not end groups in POWER7.
After group formation, the instructions are either decoded or routed to special microcode hardware that breaks complex instructions into a series of simple internal operations.
![Pasted image 20230915154836.png](/assets/images/power/Pasted image 20230915154836.png)
### 9.1.2 POWER7 ISU overview
Below shows the ISU which dispatches instructions, renames registers, issues instructions, completes instructions, and handles exception conditions.
![Pasted image 20230915154927.png](/assets/images/power/Pasted image 20230915154927.png)
Which copy of the issue queue, physical register file, and functional unit will be used by an operation depends on the multithreading mode of the processor core. In the ST and SMT2 modes, the two physical copies of the GPR have identical contents. Instructions from the thread(s) can be dispatched to either one of the issue queue halves (UQ0 or UQ1) in these modes. Load balance across the two issue queue halves is maintained by dispatching alternate instructions of a given type from a given thread to a UQ half. In an SMT4 mode, the two copies of the GPR have different contents. FX and load/store (LS) operations from threads T0 and T1 can only be placed in UQ0, can only access GPR0, and can only be issued to FX0 and LS0 pipelines. FX and LS operations from threads T2 and T3 can only be placed in UQ1, can only access GPR1, and can only be issued to FX1 and LS1 pipelines.
most VSU operations can be dispatched to either UQ0 or UQ1 in all modes (single thread, SMT2, SMT4), with the following exceptions: 1) VMX floating point and simple and complex integer operations can only be dispatched to UQ0; 2) permute (PM), decimal floating point, and 128-bit store operations can only be dispatched to UQ1; 3) VSU operations dispatched to UQ0 always execute on vector scalar pipeline 0 (VS0); and 4) VSU operations dispatched to UQ1 always execute on VS1 pipeline.
The POWER7 processor dispatches instructions on a group basis and can dispatch a group from one thread at a time to the ISU. Register renaming is done using the mapper logic (see Figure 4) before the instructions are placed in the issue queues. The following registers are renamed in POWER7: GPR, vector and scalar register (VSR), exception register (XER), CR, floating-point status and control register (FPSCR), link, and count. The GPR and VSR share a pool of 80 rename entries. The CRs are mapped onto 56 physical registers. The XERs are mapped onto 40 physical registers, and one nonrenamed register. The Link and Count registers are mapped onto 24 physical registers. The FPSCR is renamed using a 20-entry buffer to keep the state of the FPSCR associated with each group of instructions. Each of the aforementioned resources has a separate rename pool that can be independently accessed and shared by all active threads. Instructions that update more than one destination register are broken into subinstructions.
The ISU also assigns a load tag (LTAG) and a store tag (STAG) to manage load and store instruction flow. The LTAG corresponds to a pointer to the load-reorder-queue (LRQ) entry assigned to a load instruction. The STAG corresponds to a pointer to the store-reorder-queue (SRQ) entry assigned to a store instruction. This is also used to match the store data instruction with the store address instruction in the SRQ. A virtual STAG/LTAG scheme is used to minimize dispatch holds due to running out of physical SRQ/LRQ entries. When a physical entry in the LRQ is freed, a virtual LTAG will be converted to become a real LTAG. When a physical entry in the SRQ is freed, a virtual STAG will be converted to become a real STAG. Virtual STAGs or LTAGs are not issued to the LSU until they are subsequently marked as being real in the issue queue. The ISU can assign up to 63 LTAGs and 63 STAGs to each thread.
POWER7 employs three separate issue queues: a 48-entry UQ, a 12-entry branch issue queue (BRQ), and an 8-entry CR queue (CRQ). Dispatched instructions are saved in the issue queues and then issued to the execution unit one cycle after dispatch at the earliest for the BRQ or CRQ and two cycles after dispatch at the earliest for the UQ. The BRQ and CRQ are shifting queues, where dispatched instructions are placed at the top of the queue and then trickle downward toward the bottom of the queue. To save power, the UQ is implemented as a nonshifting queue and managed by queue position pointers. The queue position pointers are shifted, but the UQ entries are not shifted, which significantly reduces the switching power in the large UQ. Instructions can issue in order or out of order from all of these queues, with higher priority given to the older ready instructions for maximizing performance. An instruction in the issue queue is selected for issuing when all source operands for that instruction are available. In addition, the STAG and the LTAG must have real entries for a load or store instruction before it can be issued. For the BRQ and CRQ, instruction dependences are checked by comparing the destination physical pointer of the renamed resource against all outstanding source physical pointers. For the UQ, dependences are tracked using queue pointers via a dependence matrix. The issue queues together can issue a total of eight instructions per cycle, i.e., one branch, one CR logical, two FX instructions to the FXU, two LS or two simple FX instructions to the LSU, and two vector–scalar instructions to the VSU.
The BRQ contains only branch instructions, and it receives two branches per cycle from the dispatch stage and can issue one branch instruction per cycle for execution to the IFU. The CRQ contains the CR logical instructions and moves from SPR instructions, for the IFU, the ISU, and the pervasive control unit. The CRQ can receive two instructions per cycle and can issue one instruction per cycle to the IFU. The UQ is implemented as a 48-entry queue that is split into two halves of 24 entries each. It contains all instructions that are executed by the FXU, LSU, VSU, or DFUs. The top half of the queue contains instructions for FX0, LS0, and VS0 pipelines including VMX integer instructions. The bottom half of the queue contains instructions for FX1, LS1, and VS1 pipelines including DFP, VMX PM, and the VSU 128-bit store instructions. Appropriate instructions are steered at the dispatch stage to the appropriate half of the UQ. The UQ can receive up to four instructions per cycle per UQ half. The 64-bit VSU store instructions are split into an address generation (AGEN) operation and a data steering operation during instruction dispatch, and a total of eight such operations can be written into UQ in a given cycle. The relative age of the instructions in the UQ is determined by an age matrix since the UQ is a nonshifting queue, which is written at dispatch time. Each half of the UQ can issue one FX, one LS, and one VS instruction per cycle for a total of six instructions per cycle. Speculative issues can occur, for example, when an FX operation dependent on a load operation is issued before it is known that the load misses the D-cache or the data effective-to-real-address translation (D-ERAT). On a misspeculation, the instruction is rejected and reissued a few cycles later. Simple FX instructions may be selected for issue to the LSU for improved FX throughput, with the same latency as a load operation from L1 D-cache. The ISU is responsible to track and complete instructions. POWER7 employs a global completion table (GCT) to track all in-flight instructions after dispatch. Instructions in the core are tracked as groups of instructions and, thus, will dispatch and complete as a group. The GCT has 20 entries, which are dynamically shared by all active threads. Each GCT entry corresponds to a group of instructions, with up to four nonbranch and up to two branch instructions. This allows the GCT to track a maximum of 120 in-flight instructions after dispatch. Each GCT entry contains finish bits for each instruction in the group. At dispatch, the finish bits are set to reflect the valid instructions. Instructions are issued out of order and speculatively executed. When an instruction has successfully executed (without a reject), it is marked as finished. When all the instructions in a group are marked finished, and the group is the oldest for a given thread, the group can complete. When a group completes, the results of all its instructions are made architecturally visible, and the resources held by its instructions are released. The POWER7 core can complete one group per thread pair (threads 0 and 2 form one pair, whereas threads 1 and 3 form the other pair) per cycle, for a maximum total of two group completions per cycle. When a group is completed, a completion group tag (GTAG) is broadcasted so that resources associated with the completing group can be released and reused by new instructions. Flush generation for the core is handled by the ISU. There are many reasons to flush out speculative instructions from the instruction pipeline such as branch misprediction, LS out-of-order execution hazard detection, execution of a context synchronizing instruction, and exception conditions. The completion unit combines flushes for all groups to be discarded into a 20-bit mask, i.e., 1 bit for each group. The completion unit also sends out the GTAG for partial-group flushes, which occurs when the first branch is not the last instruction in the group, and it mispredicts, causing a need to flush all subsequent instructions from the thread. A 4-bit slot mask accompanies the partial flush GTAG to point out which instructions in the group need to be partially flushed. All operations related to the canceled groups are discarded.
### 9.1.3 LSU microarchitecture
LSU contains two symmetric LS execution pipelines (LS0 and LS1), each capable to execute a load or a store operation in a cycle. Below shows the microarchitecture for an LSU pipeline, which contains several subunits, i.e., LS AGEN and execution, SRQ and store data queue (SDQ), LRQ, load miss queue (LMQ), address translation mechanism, which includes the D-ERAT, ERAT miss queue, segment lookaside buffer (SLB) and TLB, and the L1 D-cache array with its supporting set predict and data directory (DDIR) arrays, and the data prefetch request queue (PRQ) engine.
![Pasted image 20230915154725.png](/assets/images/power/Pasted image 20230915154725.png)
* LS execution. In the ST and SMT2 modes, a given LS instruction can execute in either pipeline. In the SMT4 mode, instructions from threads 0 and 1 execute in pipeline 0, whereas instructions from threads 2 and 3 execute in pipeline 1. Instructions are issued to the LSU out of order, with a bias toward the oldest operations first. Stores are issued twice; an AGEN operation is issued to the LSU, whereas a data steering operation is issued to the FXU or the VSU. Main dataflow buses into and out of the LSU include 32-byte reload data from the L2 cache and 16-byte store data to the L2 cache, 16-byte load data per execution pipeline to the VSU (with a tap off of 8-byte load data per execution pipeline to the FXU), one 16-byte store data from the VSU and 8-byte store data per execution pipeline from the FXU. POWER7 L1 D-cache size is 32 KB, which resulted in a reduction in the D-cache access latency. FX loads have a two-cycle load-to-use latency, that is, only one cycle of bubble (which is a cycle in the pipeline during which no useful work is done) is introduced between a load and a dependent FXU operation. The VSU loads have a three-cycle load-to-use latency, that is, two cycles of bubbles are introduced between a load and a dependent VSU operation. Each LSU pipeline can also execute FX add and logical instructions, allowing more FX execution capability for the POWER7 core and greater flexibility to the ISU in the issuing of instructions.
* LS ordering. The LSU must ensure the effect of architectural program order of execution of the load and store instructions, although the instructions can be issued and executed out of order. To achieve that, LSU employs two main queues: the SRQ and the LRQ. The SRQ is a 32-entry real-address-based content-addressable memory (CAM) structure. Each thread has 64 virtual entries that are available, allowing 64 outstanding stores to be dispatched per thread. A total of 32 outstanding stores may be issued since a real physical SRQ entry is required for the store to be issued. The SRQ is dynamically shared among the active threads. An SRQ entry is allocated at issue time and deallocated after the completion point when the store is written to the L1 D-cache or sent to the L2 cache. For each SRQ entry, there is a corresponding SDQ entry of 16 bytes. Up to 16 bytes of data for a store instruction can be sent to the L2 cache (and also written to the L1 D-cache on a hit) in every processor cycle. Store forwarding is supported, where data from an SRQ entry is forwarded to an inclusive subsequent load, even if the store and load instructions are speculative. Like the SRQ, the LRQ is a 32-entry real-address-based CAM structure. Sixty-four virtual entries per thread are available to allow a total of 64 outstanding loads to be dispatched per thread. A total of 32 outstanding loads may be issued since a real physical LRQ entry is required for the load to be issued. The LRQ is dynamically shared among the threads. The LRQ keeps track of out-of-order loads, watching for hazards. Hazards generally exist when a younger load instruction executes out of order before an older load or store instruction to the same address (in part or in whole). When such a hazard is detected, if specific conditions exist, the LRQ initiates a flush of the younger load instruction and all its subsequent instructions from the thread, without having an impact on the instructions from other threads. The load is then refetched from the I-cache and reexecuted, ensuring proper LS ordering.
* Address translation. During program execution, the EAs are translated by the first level translation into 46-bit real addresses that are used for all addressing in the cache and memory subsystem. The first level translation consists of two 64-entry D-ERAT cache and a 64-entry IERAT. In case of a miss in the ERAT cache (data or instruction), the second level translation is invoked to generate the translation. The second level translation consists of a 32-entry-per-thread SLB and a 512-entry TLB that is shared by all active threads.Effective addresses are first translated into 68-bit virtual addresses using the segment table, and the 68-bit virtual addresses are then translated into 46-bit real addresses using the page frame table. While segment table and page frame tables are large and reside in main memory, a 32-entry-per-thread SLB is maintained to keep entries from the segment table to translate from effective to virtual address, and a 512-entry TLB is maintained to keep the recently used entries from the page frame table to translate from virtual to real addresses. POWER7 supports two segment sizes, i.e., 256 MB and 1 TB, and four page sizes, i.e., 4 KB, 64 KB, 16 MB, and 16 GB. The D-ERAT is a 64-entry fully associative CAM-based cache. Physically, there are two identical copies of the D-ERAT, associated with the two LSU pipelines. In the ST and SMT2 modes, since instructions from the thread(s) can go to either LS0 or LS1 pipeline, the two copies of the D-ERAT are kept in sync with identical contents. Therefore, in the ST and SMT2 modes, logically, there are a total of 64 entries available. In the SMT2 mode, the entries are dynamically shared between the two threads. In the SMT4 mode, since the two LSU pipelines are split between the two thread pairs, the two physical copies of the D-ERAT have different contents, i.e., threads 0 and 1 dynamically share one physical 64-entry D-ERAT (associated with LS0 pipe), and threads 2 and 3 dynamically share the other physical 64-entry D-ERAT (associated with LS1 pipe), for a total of 128 logical entries. Each D-ERAT entry translates 4-KB, 64-KB, or 16-MB pages. Pages of 16 GB are installed as multiple 16-MB pages. The D-ERAT employs a binary tree least recently used (LRU) replacement policy. The SLB is a 32-entry-per-thread fully associative CAM-based buffer. Each SLB entry can support 256 MB or 1 TB segment sizes. The multiple pages per segment (MPSS) extension of PowerPC architecture is supported in POWER7. With MPSS, a segment with a base page size of 4 KB can have 4-KB, 64-KB, and 16-MB pages to be concurrently present in the segment. For a segment base page size of 64 KB, the segment can have 64-KB and 16-MB pages concurrently. The SLB is managed by the operating system, with the processor generating a data or instruction segment interrupt when an SLB entry needed for translation is not found. The TLB is a 512-entry four-way set-associative buffer. The TLB is managed by hardware and employs a true LRU replacement policy. There can be up to two concurrent outstanding table-walks for TLB misses. The TLB also provides a hit-under-miss function, where the TLB can be accessed, and it returns translation information to the D-ERAT, while a table-walk is in progress. In POWER7, each TLB entry is tagged with the logical partition (LPAR) identity. For a TLB hit, the LPAR identity of the TLB entry must match the LPAR identity of the active partition running on the core.
* L1 data cache organization. POWER7 contains a dedicated 32-KB eight-way set-associative banked L1 D-cache. The cache line size is 128 bytes consisting of four sectors of 32 bytes each. There is a dedicated 32-byte reload data interface from the L2 cache, which can supply 32 bytes of data in every processor cycle. The cache line is validated on a sector basis as each 32-byte sector is returned from memory subsystem. Loads can hit against a valid sector before the entire cache line is validated. The L1 D-cache has three ports––two read ports (for two load instructions) and one write port (for a store instruction or a cache line reload). A write has higher priority over a read, and a write for a cache line reload has higher priority than a write for a completed store instruction. The L1 D-cache consists of four physical macros organized by data bytes, each macro partitioned into 16 banks based on the EA bits, for a total of 64 banks. The cache banking allows for one write and two reads to occur in the same cycle, as long as the reads are not to the same bank(s) as the write. If a read has a bank conflict with a write, the load instruction is rejected and reissued. A 32-byte cache line reload spans eight banks, whereas a completed store instruction spans from one to four banks, depending on data length. The L1 D-cache is a store-through design; all stores are sent to the L2 cache, and no L1 cast-outs are required. The L1 D-cache is not allocated on a store miss; the store is just sent to the L2 cache. The L1 D-cache is inclusive of the L2 cache. The L1 D-cache has byte-write capability of up to 16 bytes within a given 32-byte sector in support of store instructions. The L1 D-cache is indexed with the EA bits. The L1 D-cache directory employs a binary tree LRU replacement policy. Being 32 KB and eight-way set-associative results in 4 KB per set, requiring up to EA bit 52 to be used to index into the L1 D-cache. A set predict array is used to reduce the L1 D-cache load hit latency. The set predict array is based on EA and is used as a minidirectory to select which one of the eight L1 D-cache sets contains the load data. The set predict array is organized as the L1 D-cache: indexed with EA(52:56) and eight-way set-associative. Each entry contains 11 hash bits obtained from hashing bits EA(33:51), valid bits per thread, and a parity bit. When a load executes, the generated EA(52:56) is used to index into the set predict array, and EA(33:51) is hashed and compared with the contents of the eight sets of the indexed entry. When an EA hash match occurs and the appropriate thread valid bit is active, the match signal is used as the set select for the L1 D-cache data. If there is no EA hash match, it indicates a cache miss. However, an EA hash match does not necessarily mean a cache hit. For cache hit determination, the EA is used to look up in the L1 data cache directory for the real address and then compare this real address with the real address obtained from the ERAT for the given EA. When a cache line is validated, the default is to enter in a shared mode where all thread valid bits for the line are set. A nonshared mode is dynamically entered on an entry-by-entry basis to allow only one thread valid bit to be active. This is beneficial to avoid thrashing among the threads, allowing the same EA hash to exist for each thread at the same time.
* Load miss handling. Loads that miss the L1 D-cache initiate a cache line reload request to the L2 cache, release the issue queue entry, and create an entry in the LMQ to track the loading of the cache line into the L1 D-cache and also to support the forwarding of the load data to the destination register. When the load data returns from the L2 cache, it gets higher priority in the LSU pipeline, and the data is transferred to the destination register. The LMQ is real address based and consists of eight entries, dynamically shared among the active threads. The LMQ tracks all cache line misses that result in reload data to the L1 D-cache, which also includes data prefetch requests and data touch instructions, in addition to load instructions. The LMQ supports load merging, where up to two load instructions (of the same or different threads) can be associated with a given LMQ entry and cache line reload request. The LMQ can support multiple misses (up to eight) to a given L1 D-cache congruence class.
### 9.1.4 POWER7 FXU overview
The FXU comprises of two identical pipelines (FX0 and FX1). As shown in below figure, each FXU pipeline consists of a multiport GPR file; an arithmetic and logic unit (ALU) to execute add, subtract, compare, and trap instructions; a rotator to execute rotate, shift, and select instructions; a count (CNT) leading zeros unit; a bit-select unit (BSU) to execute bit PM instruction; a divider (DIV); a multiplier (MULT); and a miscellaneous execution unit (MXU) to execute population count, parity, and binary-coded decimal assist instructions. All SPRs that are local to the FXU pipeline are stored in SPR. Certain resources such as the FX XER file are shared between the two pipelines.
![Pasted image 20230915154552.png](/assets/images/power/Pasted image 20230915154552.png)
The most frequent FX instructions are executed in one cycle, and dependent operations may issue back to back to the same pipeline, if they are dispatched to the same UQ half (otherwise, one cycle bubble is introduced). Other instructions may take two, four, or a variable number of cycles. At the heart of each FXU pipeline is a GPR file with 112 entries, which holds the architected registers and the renamed registers for up to two threads. The GPR has four read ports, two supplying operands for the FX pipeline, and two supplying AGEN operands to the attached LSU pipeline. Two physical write ports are clocked twice per cycle (double-pumped), giving four logical write ports, two capturing results from the two FX pipelines, and the other two from the two data cache read ports in the LSU. Double pumping the write ports reduces power consumption and the size of the GPR macro, which is important for shortening the length of critical wires that must traverse it. Contents of the two GPR files in each pipeline are managed by the ISU to be identical in the ST and SMT2 modes but distinct in the SMT4 mode. That is, in the SMT4 mode, the GPR in one pipeline contains the architected and renamed registers for one pair of threads, whereas the GPR in the other pipeline contains the registers for the other pair of threads.The latency between a compare instruction and a dependent branch instruction is often a significant performance detractor for many workloads. To reduce this latency, each FXU pipeline has a fast compare custom macro that calculates the condition code from a compare instruction faster than the ALU, resulting in a back-to-back issue in most cases for a compare, followed by a branch instruction.
* Vector and scalar instruction execution. The POWER7 VSU implements the new VSX architecture introducing 64 architected registers. With dual issue of two-way SIMD floating-point DP instructions, the performance in FLOPs per cycle per core is doubled in comparison to POWER6. In addition, the VSU of the POWER7 processor merges the previously separate VMX unit and binary FPU (BFU) into a single unit for area and power reduction. Furthermore, the POWER6 DFU is attached to the VSU as a separate unit, sharing the issue port with the VS1 pipeline.
Below shows the VSU pipeline diagram.
![Pasted image 20230918150731.png](/assets/images/power/Pasted image 20230918150731.png)
## 9.2 POWER7 Cache hierarchy
the POWER7 cache hierarchy includes a shared 32-MB L3 cache comprised of the 4-MB local L3 regions from the eight cores that reside on the processor chip. The eDRAM requires only one-fifth of the standby energy used by a traditional SRAM cell, while using less than one-third the area.
![Pasted image 20230915155341.png](/assets/images/power/Pasted image 20230915155341.png)
POWER7 L2 and L3 caches support the same 13-cache-state protocol as the POWER6 design point. While no new cache states have been added for POWER7, new coherent operations are supported. One such operation is cache injection. An I/O device performing a direct memory access (DMA) write operation may target the operation to the cache, instead of to memory. If a given core’s L2 cache or local L3 region owns a copy of the targeted cache line (i.e., holds the line in an M, ME, MU, T, TE, TN, or TEN cache state), the data will be installed into the local L3 region. Additionally, new heuristics have been developed, which further exploit the semantic content reflected by the existing cache states. One of these, which is called the partial victim cache management policy, reduces energy usage as data moves between a given L2 cache and its associated local L3 region.
Below table summarizes the POWER7 cache states.
![Pasted image 20230915154407.png](/assets/images/power/Pasted image 20230915154407.png)
the barrier synchronization register (BSR) facility originally implemented in POWER5 and POWER6 processors has been virtualized in the POWER7 processor. Within each system, multiple megabytes of main storage may be classified as BSR storage and assigned to tasks by the virtual memory manager. The BSR facility enables low-latency synchronization for parallel tasks. Writes to BSR storage are instantaneously broadcast to all readers, allowing a designated master thread to orchestrate the activities of workers threads in a low-latency fine-grained fashion. This capability is particularly valuable for improving parallel speedups in certain HPC environments.
* L2 Cache. Store-through traffic from the core represents the bulk of the traffic managed by the L2 cache. A fully associative 16-deep 32-byte entry store cache absorbs every individual store executed in the core or up to 16 bytes of store traffic every core cycle. Up to four of the 32-byte store cache entries, comprising updates to the same 128-byte coherence granule, can be grouped together into a single simultaneous coherence dispatch by the L2 scheduling logic.
* L3 Cache. A 4-MB L3 region is comprised of 32 ultradense high-speed eDRAM macros. The eDRAM macro has access latency and cycle time characteristics slightly worse than conventional 6T SRAM. As such, the combined effects of the eDRAM access latency, the overhead of waiting on the directory result before accessing the cache (to reduce energy usage), and the overhead of traversing the L2 cache prior to accessing the L3 cache are negligible. They are more than counterbalanced by the beneficial latency of the 256-KB L2 cache. Likewise, the slightly higher cycle time and reduction in overall bandwidth per unit of capacity is counterbalanced by the traffic reduction afforded by the 256-KB L2 cache. The refresh overhead, typically associated with DRAM, is hidden by a parallel engine that refreshes unused subarrays within each macro whenever operations exercise the macro. The centralized L3 region controller provides a single core/L2 dispatch port, a single lateral L3 region dispatch port, dual snoop dispatch ports (for even and odd cache lines), and a single pool of operational resources. Storage accesses that miss the L2 cache access the 4-MB local L3 region via the core/L2 dispatch port. Those that hit in the 4-MB local L3 region are managed by a pool of eight read machines. Prefetches and some L2 store misses also access the L3 region via the core/L2 dispatch port and are managed by a pool of 24 lightweight RC machines. When prefetched data is staged to the L3 cache, it is managed by a pool of ten write machines. Each write machine has an associated cast-out machine to manage the eviction of lines displaced by writes. Note that these write machines are also utilized by other operations such as cast-outs and cache-injections. Storage accesses that miss both the L2 cache and the 4-MB local L3 region are broadcast to the coherence fabric and snooped by the memory controller, other L2 caches, possibly other L3 caches, and by the seven remote 4-MB L3 regions that comprise the remaining 28 MB of the on-chip L3 cache. Therefore, operations that miss the 4-MB local L3 region but hit in the remaining 28 MB of the L3 cache access the cache via the snoop dispatch ports. L2 cast-out operations access the 4-MB local L3 region via the core/L2 dispatch port, whereas lateral cast-outs access a given 4-MB L3 region via the lateral L3 region dispatch port or via the snoop dispatch ports.
## 9.3 POWER7 SMP
In order to continue to provide the high-scalability low-latency characteristics of earlier POWER server processors, the POWER7 processor utilizes a similar nonblocking-broadcast-based coherence-transport mechanism, based upon the same distributed management relaxed-order-optimized multiscope enablement provided by the POWER6 platform. The on-chip coherence interconnect routes two sets (even and odd cache line) of coherence requests through the horizontal trunk, inward toward the even/odd arbitration logic located at the center of the chip. Up to one even request and one odd request may be granted each on-chip bus cycle. Once granted, the requests are broadcast within the chip on the even/odd snoop buses outward toward the left and right edges of the chip. Requests that will be routed to other chips are also sent to the multichip SMP interconnect (discussed in the multichip interconnect section) via a central vertical spine toward the top and bottom edges of the chip. Requests that have arrived from other chips are managed by the even/odd arbitration logic and broadcast to the snoop buses. Coherence responses from the snoopers are routed inward along the horizontal trunk toward the even/odd coherence decision logic located at the center of the chip. For requests that have been routed to other chips, additional responses from the off-chip snooper are fed into the coherence decision logic. Once a final coherence decision is made in response to a given request, a notification is broadcast within the chip on the even/odd notification buses outward from the center toward the left and right edges of the chip. Notifications that will be routed to other chips are also sent to the multichip SMP interconnect via the central vertical spine toward the top and bottom edges of the chip.
Because the coherence flow is nonblocking, the rate at which requests may be scheduled onto the snoop buses is restricted by the snooper with the lowest possible snoop processing rate. The central coherence arbitration logic must insure that requests (whether sourced from the chip containing the slowest snooper or from another chip in the system) do not overrun the slowest snooper. To accommodate this, system firmware negotiates a Bfloor frequency. As individual processor frequencies are adjusted upward and downward, none will ever fall beneath the floor frequency. The coherence arbitration logic throttles the rate at which requests are granted to insure that a snooper operating at the floor frequency or higher can process all the requests.
The on-chip data interconnect consists of eight 16-byte buses that span the horizontal trunk. Four flow from left to right, and the other four flow from right to left. These buses are bracketed by memory controllers found at the left and right edges of the chip. They are divided into multiple segments, such that multiple 16-byte data packets may be pipelined within the multiple segments of the same bus at any given time. The buses operate at the on-chip bus frequency. Each memory controller has two 16-byte on-ramps and two 16-byte off-ramps that provide access to the eight buses. Each core’s associated L2 cache and local L3 region share one 16-byte on-ramp/off-ramp pair, as does the pair of I/O controllers. The multichip data interconnect ports, found in the central vertical spines have a total of seven 16-byte on-ramp/off-ramp pairs. In total, there are twenty 16-byte on-ramps and twenty 16-byte off-ramps that provide access to and from the eight horizontal 16-byte trunk buses. Each ramp pair is associated with a bus segment. Note that a source-to-destination on-ramp/off-ramp route may consume only a subset of the segments in the horizontal trunk, depending upon the physical locations of the source and destination. Data transfers are managed by centralized arbitration logic that takes into account source and destination locations, allocates available bus segments to plot one of several possible routes, allocates the on- and off-ramp resources, and manages destination data buffering resources. Note that since transfers may use only a subset of the segments in a given trunk bus, multiple noninterfering source-to-destination transfers may utilize the same horizontal trunk bus simultaneously. The arbitration logic must also account for the differing operating frequencies of the processor cores. For example, a source core operating at a lower frequency will send data via its on-ramp to the trunk buses at a slower rate. Likewise, a destination core operating at a lower frequency will consume data via its off-ramp from the trunk buses at a slower rate. To manage this, the arbitration logic controls speed-matching buffers in all of the on-ramps/off-ramps.
Below depicts a first-level nodal structure, which combines up to four POWER7 chips. Each chip has four 10-B/s on-node SMP links associated with the vertical spine that emanates from the center of the chip toward the top edge. In this manner, the four chips are fully connected.
![Pasted image 20230918164544.png](/assets/images/power/Pasted image 20230918164544.png)
A second-level system structure combines up to eight of the nodes. Each chip has two 10B/s off-node SMP links associated with the vertical spine that emanates from the center of the chip toward the bottom edge. As shown in Figure, in a standard commercial system, up to seven of the eight off-node SMP links (coming from the four POWER7 chips comprising a node) are connected to each of the seven other nodes that comprise an eight-node system. Unlike POWER6 systems, which enforce a strict 50% coherence to 50% data ratio or 33% coherence to 67% data ratio, the POWER7 SMP links enable a dynamic free-form allocation of coherence and data traffic, enabling higher effective utilization of the links. Additionally, to maximize SMP link bandwidth, the on- and off-node SMP links do not operate at the on-chip frequency. They are independently tuned, typically operating at speeds ranging from 2.5 to 3.3 GHz, depending upon system packaging characteristics, and provide increased flexibility over the POWER6 SMP links. POWER7 systems heavily exploit the speculative localized scope coherence broadcast capability introduced in the POWER6 design. The localized regions make use of enhanced scope prediction heuristics to partition the coherence traffic, such that each region has full access to its SMP link and snooper coherence bandwidth. In cases where the speculation is successful, this has the effect of multiplying the link and snooper bandwidths by the number of regions. For example, dividing a large 256-way system into eight nodal regions has the effect (to the degree of successful speculation) of enabling each 32-way region to privately enjoy the SMP link and snooper bandwidth that would otherwise be shared across the whole system.
![Pasted image 20230725175956.png](/assets/images/power/Pasted image 20230725175956.png)

## 9.4 POWER7 RAS
POWER7 reliability and availability features are highlighted in below Figure.
![Pasted image 20230918170442.png](/assets/images/power/Pasted image 20230918170442.png)
When an error is detected and reported by a core unit, the POWER7 core quickly blocks all instruction completion, along with blocking all instruction fetch and dispatch. If the error condition is not severe enough to cause a checkstop, the core initiates the recovery process. The recovery process flushes all the instructions in the pipeline for each thread to put the core in an architected state that existed sometime before the error condition occurred, fence the core from the rest of the system (L2 and nest), run an automatic built-in self-test to clear and test the core SRAM cells, reset each core unit to clean up any errors and reset the state machines, refresh the GPR and VSR files by initiating a state machine that does a read/correct/write operation to each entry in the register files to correct any single bit error through ECC correction mechanism, drop the fence to L2 and nest, and then restart instruction fetching and enable dispatch and completion.
To facilitate error detection and recovery in POWER7, the big register files (such as GPR and VSR) are ECC protected, whereas the smaller register files are protected through parity; all SRAM cells have error detection and recovery mechanisms. The I-cache is parity protected and recoverable. The floating-point pipelines implement residue checking mechanism, and numerous logic units implement additional control checking mechanism. In addition, POWER7 core uses RAS-hardened latches for various SPRs and core configuration latches.
The L1 D-cache is protected by byte parity. Hardware recovery is invoked on detection of a parity error while reading the L1 D-cache for a load instruction. The load instruction in error is not completed but rather flushed, the L1 D-cache contents are invalidated, and the instructions are refetched and reexecuted from the group of the load instruction in error. Additionally, when a persistent hard error is detected either in the L1 D-cache array or in its supporting directory or set predict array, a set delete mechanism is used to prohibit the offending set from being validated again. This allows the processor core to continue execution with slightly degraded performance until a maintenance action is performed.

# 10. POWER 8
**POWER8** core can be put in a split-core mode, so that four partitions can run on one core at the same time, with up to two hardware threads per partition. Also supporting emerging workloads, the POWER8 processor includes an optimized implementation of hardware TM (Transactional Memory). This implementation has a low overhead to start a transaction and additional features that support the exploitation of transactions in Java and other programming languages, in many cases without any changes to the user code.
![Pasted image 20230718154255.png](/assets/images/power/Pasted image 20230718154255.png)
On soft-error detection, the core automatically uses its out-of-order execution features to flush the instructions in the pipeline and re-fetch and re-execute them, so that there is no loss of data integrity.
## 10.1 Power 8 Core
The core consists primarily of the following six units: instruction fetch unit (IFU), instruction sequencing unit (ISU), load-store unit (LSU), fixed-point unit (FXU), vector and scalar unit (VSU) and decimal floating point unit (DFU). The instruction fetch unit contains a 32 KB I-cache (instruction cache) and the load-store unit contains a 64 KB D-cache (data cache), which are both backed up by a tightly integrated 512 KB unified L2 cache. In a given cycle, the core can fetch up to eight instructions, decode and dispatch up to eight instructions, issue and execute up to ten instructions, and commit up to eight instructions.
There are sixteen execution pipelines within the core: two fixed-point pipelines, two load/store pipelines, two load pipelines, four double-precision floating-point pipelines (which can also act as eight single-precision floating-point pipelines), two fully symmetric vector pipelines that execute instructions from both the VMX (Vector eXtensions) and VSX (Vector-Scalar eXtensions) instruction categories in the Power ISA, one cryptographic pipeline, one branch execution pipeline, one condition register logical pipeline, and one decimal floating-point pipeline. The two load/store pipes and the two load pipes have the additional capability to execute simple fixed-point operations. The four floating-point pipelines are each capable of executing double precision multiply-add operations, accounting for eight double-precision, 16 single-precision, floating-point operations per cycle per core. In addition, these pipelines can also execute 64-bit integer SIMD operations.
下图展示了**POWER8**处理器核的floorplan
![Pasted image 20230718155035.png](/assets/images/power/Pasted image 20230718155035.png)

### 10.1.1 POWER8 Core Instruction Flow
Figure shows the instruction flow in POWER8 processor core. Instructions flow from the memory hierarchy through various issue queues and then are sent to the functional units for execution. Most instructions (except for branches and condition register logical instructions) are processed through the Unified Issue Queue (UniQueue), which consists of two symmetric halves (UQ0 and UQ1). There are also two copies (not shown) of the general-purpose (GPR0 and GPR1) and vector-scalar (VSR0 and VSR1) physical register files. One copy is used by instructions processed through UQ0 while the other copy is for instructions processed through UQ1. The fixed-point, floating-point, vector, load and load-store pipelines are similarly split into two sets (FX0, FP0, VSX0, VMX0, L0, LS0 in one set, and FX1, FP1, VSX1, VMX1, L1, LS1 in the other set) and each set is associated with one UniQueue half. Which issue queue, physical register file, and functional unit are used by a given instruction depends on the simultaneous multi-threading mode of the processor core at run time. In ST mode, the two physical copies of the GPR and VSR have identical contents. Instructions from the thread can be dispatched to either one of the UniQueue halves (UQ0 or UQ1). Load balance across the two UniQueue halves is maintained by dispatching alternate instructions of a given type to alternating UniQueue halves. In the SMT modes (SMT2, SMT4, SMT8), the two copies of the GPR and VSR have different contents. The threads are split into two thread sets and each thread set is restricted to using only one UniQueue half and associated registers and execution pipelines. Fixed-point, floating-point, vector and load/store instructions from even threads (T0, T2, T4, T6) can only be placed in UQ0, can only access GPR0 and VSR0, and can only be issued to FX0, LS0, L0, FP0, VSX0, and VMX0 pipelines. Fixed-point, floating-point, vector and load/store instructions from odd threads (T1, T3, T5, T7) can only be placed in UQ1, can only access GPR1 and VSR1, and can only be issued to FX1, LS1, L1, FP1, VSX1, and VMX1 pipelines. Cryptographic and decimal floating-point instructions from a thread can only be placed in the corresponding UniQueue half, but since there is only one instance of each of these units, all instructions are issued to the same unit. Branches and condition register logical instructions have their own dedicated issue queues and execution pipelines, which are shared by all threads.
![Pasted image 20230918172554.png](/assets/images/power/Pasted image 20230918172554.png)

![Pasted image 20230919095043.png](/assets/images/power/Pasted image 20230919095043.png)
### 10.1.2 POWER8 IFU
The POWER8 IFU has several new features relative to the POWER7 processor IFU. Support for SMT8 and additional concurrent LPARs (logical partitions) required changes in sizes for many resources in the IFU. In addition, the following changes were made to improve the overall performance of the POWER8 core: First, instruction cache alignment improvements result in a higher average number of instructions fetched per fetch operation. Second, branch prediction mechanism improvements result in more accurate target and direction predictions. Third, group formation improvements allow more instructions per dispatch group, on average. Fourth, instruction address translation hit rates were improved. Fifth, instruction fusion is used to improve performance of certain common instruction sequences. Finally, better pipeline hazard avoidance mechanisms reduce pipeline flushes.
![Pasted image 20230918172656.png](/assets/images/power/Pasted image 20230918172656.png)
* Instruction fetching and predecoding. The POWER8 core has a dedicated 32 KB, 8-way set associative L1 I-cache. It is based on a 16-way banked design to avoid read and write collisions. A 32 x 8-entry Instruction Effective Address Directory (IEAD) provides fast prediction for way selection to choose one fetch line from the eight ways. A traditional full I-cache directory (I-dir) is accessed in parallel to confirm the way selection prediction in the next cycle. The I-cache can be addressed on any 16-byte boundary within the 128-byte cache line. Fast instruction address translation for instruction fetch is supported by a fully associative 64-entry Instruction Effective to Real Address translation Table (IERAT). The IERAT is shared among all threads. The IERAT directly supports 4 KB, 64 KB, and 16 MB page sizes. Other page sizes are supported by storing entries with the next smaller supported page size. The IFU reads instructions into the I-cache from the L2 unified cache. Each read request for instructions from the L2 returns four sectors of 32 bytes each. These reads are either demand loads that result from I-cache misses or instruction prefetches. For each demand load request, the prefetch engine initiates additional prefetches for sequential cache lines following the demand load. Demand and prefetch requests are made for all instruction threads independently, and instructions may return in any order, including interleaving of sectors for different cache lines. Up to eight instruction read requests can be outstanding from the core to the L2 cache. Instruction prefetching is supported in ST, SMT2, and SMT4 modes only. Up to three sequential lines are prefetched in ST mode and one sequential line per thread in SMT2 and SMT4 modes. There is no instruction prefetching in SMT8 mode to save on memory bandwidth. Prefetches are not guaranteed to be fetched and depending on the congestion in the POWER8 processor nest, some prefetches may be dropped. When instructions are read from the L2 cache, the IFU uses two cycles to create predecode and parity bits for each of the instructions, before they are written into the I-cache. The predecode bits are used to scan for taken branches, help group formation, and denote several exception cases. Branch instructions are modified in these stages to help generate target addresses during the branch scan process that happens during the instruction fetch stages of the pipeline. The modified branch instruction, with a partially computed target address, is stored in the I-cache. Three cycles after a 32-byte sector of instructions arrives on the I-cache/L2 interface, the sector is written into the I-cache. If the requesting thread is waiting for these instructions, they are bypassed around the I-cache to be delivered to the instruction buffers and the branch scan logic. Instruction Fetch Address Registers (IFARs) track program counter addresses for each thread. On each cycle, the IFAR register for one of the threads is selected to provide the fetch address to the I-cache complex and the branch prediction arrays. The I-cache fetch process reads quad-word aligned block of up to eight instructions per cycle from the I-cache and writes them into the instruction buffers where they are later formed into dispatch groups. Quadword-aligned fetch ensures that for a non-sequential fetch at least one instruction from the first quadword and four instructions from the second quadword are fetched as long as there is a cache hit and both quadwords are within the cache line. Thread priority, pending cache misses, instruction buffer fullness, and thread balancing metrics are used to determine which thread is selected for instruction fetching in a given cycle. The IFU allocates fetch cycles within threads of the same partition based on the priorities associated with each thread.
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
The Fixed-Point Unit (FXU) is composed of two identical pipelines (FX0 and FX1). As shown in Figure, each FXU pipeline consists of a multiport General Purpose Register (GPR) file, an arithmetic and logic unit (ALU) to execute add, subtract, compares and trap instructions, a rotator (ROT) to execute rotate, shift and select instructions, a count unit (CNT) to execute count leading zeros instruction, a bit select unit (BSU) to execute bit permute instruction, a miscellaneous execution unit (MXU) to execute population count, parity and binary-coded decimal assist instructions, a multiplier (MUL), and a divider (DIV).
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