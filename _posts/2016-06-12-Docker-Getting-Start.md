---
title : Docker and Kubernetes Getting Start
categories : programming
tags : [virtualizer,skills]
---

# What is Docker

> At its core, Docker provides a way to run almost any application securely isolated in a container. The isolation and security allow you to run many containers simultaneously on your host. The lightweight nature of containers, which run without the extra load of a hypervisor, means you can get more out of your hardware.

这是*Docker*官网上描述，我们可以把它理解成一种轻量级的虚拟化。

*Docker*官网的Client-Server architecture图:
[!docker architecture](https://docs.docker.com/engine/article-img/architecture.svg)

可以看到*Docker*主要结构:

* The Docker Daemon : 守护进程，不与用户直接交互。
* The Docker Client : 客户端， 负责用户交互以及和守护进程的通信
* Inside Docker : *Docker*内部又是由_Docker Images_, _Docker Registries_,
  _Docker Containers_组成。
    * Docker Images : 只读，提供实际需要运行的程序
    * Docker Registry : 是_Docker Image_的仓库，提供upload和download功能
    * Docker Container : *Docker*提供的_Docker Image_上指定需要运行的程序

可以把_Docker Images_类比成App, _Docker Registry_当成App Store, _Docker
Container_就是安装之后的App。

# Installation

> 对于*docker*安装相关问题，可以查询[docker官网](https://docs.docker.com/engine/installation).
按照上面要求安装一般不会有什么问题.

* 对于*Ubuntu 14.04 LTS*，可以通过下面command来安装:
```shell
## first add docker GPG key to apt
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
## then add docker source to apt
echo 'deb https://apt.dockerproject.org/repo ubuntu-trusty main' > /etc/apt/sources.list.d/docker.list
## download and install docker engine
sudo apt-get update
sudo apt-get install docker-engine

```

* 然后可以启动*docker*

```shell
sudo service docker start
```

* 测试*Docker*是否安装成功

```shell
sudo docker run hello-world
```

# Basic Usage/Perception

* Namespace

* Control Groups

* Union File Systems

# Reference

* 1. [Docker - User Guide](https://docs.docker.com)

