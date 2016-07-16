---
title : Docker and Kubernetes Getting Start
categories : programming
tags : [virtualizer,skills]
---

# What is Docker

> At its core, Docker provides a way to run almost any application securely isolated in a container. The isolation and security allow you to run many containers simultaneously on your host. The lightweight nature of containers, which run without the extra load of a hypervisor, means you can get more out of your hardware.

这是*Docker*官网上描述，我们可以把它理解成一种轻量级的虚拟化。

*Docker*官网的Client-Server architecture图:
![docker architecture](https://docs.docker.com/engine/article-img/architecture.svg "Architecture")

可以看到*Docker*主要结构:

* The Docker Daemon : 守护进程，不与用户直接交互。
* The Docker Client : 客户端， 负责用户交互以及和守护进程的通信
* Inside Docker : *Docker*内部又是由 _Docker Images_ , _Docker Registries_ ,
  _Docker Containers_ 组成。
    * Docker Images : 只读，提供实际需要运行的程序
    * Docker Registry : 是 _Docker Image_ 的仓库，提供upload和download功能
    * Docker Container : *Docker*提供的 _Docker Image_ 上指定需要运行的程序

可以把 _Docker Images_ 类比成App, _Docker Registry_ 当成App Store, _Docker
Container_ 就是安装之后的App。

# Installation

> 对于*Docker*安装相关问题，可以查询[docker官网](https://docs.docker.com/engine/installation).
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

* 然后可以启动*Docker*

```shell
sudo service docker start
```

* 测试*Docker*是否安装成功

```shell
sudo docker run hello-world
```

# Basic Usage/Perception

## *Docker*使用到的底层技术

* Namespace

> *Docker*利用*Namespace*来创建一个隔离的环境或者*Container*

* Control Groups

> *Docker*利用*Control Group*来对*Container*里的资源(CPU/MEMORY/NETWORKING)来进行限制

* Union File Systems

> *Docker*利用*Union File Systems*来轻量化*Container*

## How To Build An Image

* 创建*Dockerfile*

> `Dockerfile`描述的是*image*里应该包含的应用以及环境,*Dockerfile*里内容如下:
    - 指定基于哪个*image*

    ```shell
    FROM docker/whalesay:latest
    ```

    - *image*中需要安装的应用

    ```shell
    RUN apt-get -y update && apt-get install -y fortunes
    ```

    - 指定运行的应用

    ```shell
    CMD /usr/games/fortune -a | cowsay
    ```

* 基于*Dockerfile*生成*image*

> 在*Dockerfile*所在目录执行下面`command`:

```shell
sudo docker build -t docker-whale .
```

上面`command`会根据*Dockerfile*生成一个名为`docker-whale`的*image*

* 运行上面创建的*imgae*

```shell
sudo docker run docker-whale
```

* 其他*Docker*命令
    * `docker ps` - 列出运行的*container*
    * `docker logs` - 输出*container*标准输出
    * `docker stop` - 停止运行的*Docker*

# What is *Kubernetes* ?

> an open-source system for automating deployment, scaling, and management of containerized applications.It groups containers that make up an application into logical units for easy management and discovery

官网上是这样描述的。简而言之，
我们可以把*Kubernetes*当成*Docker*的一个管理工具。下面图描述了*Kubernetes*的工作状态:
![Kubernetes](http://kubernetes.io/images/hellonode/image_13.png)

# *Kubernetes*基本组成

* Pod

    > *Pod*是一个或多个*Container*组成的，共同管理, 可以使用下面命令来创建*Pod*

    ```shell
    kubectl run hello-node --image=gcr.io/PROJECT_ID/hello-node:v1 --port=8080
    ```
    *Kubernetes*命令和*Docker*类似， 比如:
    * 可以使用`kubectl get pods`来查看创建的*Pod*
    * 可以使用`kubectl logs <POD-NAME>`来查看*Pod*的标准熟输出

* Service

> *Service* 是一组*Pod*以及相关的访问接口集合

# Reference

* 1.[Docker - User Guide](https://docs.docker.com)
* 2.[Kubernetes - User Guide](http://kubernetes.io/docs/user-guide)

