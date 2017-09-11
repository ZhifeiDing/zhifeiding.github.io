---
title : SSD Introduction
categories : hardware
tags : [hardware, storage]
---

# SSD简介

> `SSD(Solid State Disk)`是一种使用使用`Non-volatile NAND Flash`的大容量存储器，相比传统的`HDD(Hard Driver Disk)`，
`SSD`由于使用的是集成电路来保存数据，没有机械结构，拥有较高的读写速度，在读写要求较高的数据库领域得到广泛使用。下面主要介绍一下
`SSD`内部存储数据的结构。

# Non-Volatile Flash 分类

根据基本存储结构的不同， 可以分为`NOR Flash`和`NAND Flash`，而根据每一个基本单元存储信息量可以分为`SLC ( single level cell )`, `MLC ( multi - level cell )`, `TLC ( triple level cell )`和`QLC ( quard level cell )`。

## Floating-Gate MOSFET结构

不管怎么分类， *Flash* 里存储信息单元都是 *Floating Gate MOSFET*, 其基本结构如下所示：

![Floating-Gate MOSFET](https://upload.wikimedia.org/wikipedia/commons/2/2c/Flash_cell_structure.svg)

*Floating-Gate MOSFET*和一般场效应管主要区别是栅极是由被*SiO2*包围的*Floating Gate*和正常的*Control Gate*组成， 也就是多了一个*Floating Gate*。


## Floating-Gate MOSFET特性

根据上面结构图， 可以知道， *FLoating Gate*处在 *Control Gate*以及源级和漏级组成的*channel*中间。因此*Floating Gate MOSFET*主要特性和*FLoating Gate*有关。

* 当*Floating Gate*上不存在电荷时，此时*MOSFET*阈值电压是 *VT1*, 管道导通
* 当*Floating Gate*上存在电荷时，此时为了能使管道导通，除了要施加上面*VT1*，还需要抵消*Floating Gate*上电荷， 因此此时阈值电压 *Vt2 > Vt1*
* 因此， 当没有对*Floating Gate MOSFET*做*Programming*操作时，*Floating Gate*上是不存在电荷的，也就是当前数据是`1`。而*Programming*使*Floating Gate*带上电荷之后，保存数据时`0`
* 当需要做*Programming*操作时，需要在*Control Gate*上施加大于*Vt2*电压，此时源级和漏级之间的*Channel*里的电流会导致电子跃迁，进入*Floating Gate*，这叫做热电子注入。此时保存数据是`0`
* 当需要读数据时，在*Control Gate*上施加一个介于*Vt1*和*Vt2*上的，此时如果*Floating*上没有电荷，则管道导通，数据为`1`。否则，管道不导通，数据为`0`
* 上面主要是`SLC`情况， 即每一个*cell*只保存`1bit`的信息，所以只需要判断*MOSFET*是否有电流流过即可。 而对于`MLC`,`TLC`和`QLC`可以根据加到*Control Gate*上不同电压时流过电流不同来区别多bit的信息。


## NOR FLash

将上面介绍的*Floating Gate MOSFET*的漏级接在一起形成*bitline*，相连的源级接地，这样只要一个*MOSFET*的*Control Gate*施加介于*Vt1*和*Vt2*之间的电压，会导致*bitline*被拉低，所以叫做*NOR Flash*. 结构如下所示:

![Nor-Flash layout](https://upload.wikimedia.org/wikipedia/commons/d/dd/NOR_flash_layout.svg)

### Programming

![NOR Flash Programming](https://upload.wikimedia.org/wikipedia/commons/2/28/Flash-Programming.svg)

### Erasing

![NOR Flash Erasing](https://upload.wikimedia.org/wikipedia/commons/1/1f/Flash_erase.svg)

## NAND Flash

![NAND-FLash layout](https://upload.wikimedia.org/wikipedia/commons/f/f5/Nand_flash_structure.svg)

# Reference

1. [Flash Memory - wikipedia](https://en.wikipedia.org/wiki/Flash_memory)
2. [Solid state drive - wikipedia](https://en.wikipedia.org/wiki/Solid-state_drive)
