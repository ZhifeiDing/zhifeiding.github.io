# 1. 概述

随着大语言模型支持的上下文长度不断增长，上下文并行（Context Parallelism，CP）已逐渐从一种可选优化演变为支撑超长上下文训练与推理的必要并行手段。与此同时，生产负载中长短请求并存，能够随负载调整并行形态的动态 CP 技术也应运而生。在推理侧，vLLM 与 SGLang 针对 Prefill 和 Decode 两个阶段分别实现了 PCP（Prefill CP）与 DCP（Decode CP）；在训练侧，NVIDIA Megatron 提供了完整的 CP 实现；NVIDIA Helix 与北大、阿里联合提出的 Adaptive Context Parallelism 则代表了动态 CP 的两个研究方向。本文第 2 章介绍 CP 的基本概念，第 3 章比较各类通信方案，第 4 章讨论不同注意力结构下的 CP 实现，第 5 章梳理主流引擎中的工程落地，第 6 章介绍两种动态 CP 系统。

# 2. 什么是CP

张量并行（TP）沿 head 维切分注意力计算，每层需要通过 AllReduce 汇聚结果，通信量大，且并行度受 head 数限制。随着上下文长度增长，即使经过 TP 切分，每张卡上的激活与 KV Cache 仍随序列长度线性膨胀。因此，沿序列维切分成为支撑超长上下文的现实需求。

CP（Context Parallelism，上下文并行）沿序列（token）维切分。长度为 $T$ 的序列被切成 $C$ 份，每个 CP rank 持有 $T/C$ 个 token 的激活以及对应区间的 KV Cache。除注意力（以及线性注意力的循环状态）之外，Linear、MLP、Norm 等算子对 token 逐一独立，天然可以按 token 并行；唯一的跨 token 依赖存在于注意力中，需要通过 rank 间通信解决。下表从五个维度对比 TP 与 CP。

| 维度 | Tensor Parallel | Context Parallel |
| --- | --- | --- |
| 切分维度 | head / hidden 维 | 序列（token）维 |
| 每层通信 | AllReduce，通信量大 | 只传注意力相关数据（KV / Q / latent / 状态） |
| 并行度上限 | 受 head 数限制（GQA/MLA 下更紧） | 受序列长度限制，与 head 数无关 |
| KV Cache | 按 head 切分，head 数少时会复制 | 按 T 切分，显存占用随 CP 度下降 |
| 激活显存 | 按 head 切，长序列仍线性增长 | 按序列切，长序列显存近似除以 $C$ |

CP 与 TP、PP、DP、EP 正交，可以组合成 4D/5D 混合并行。实践中通常先启用 Megatron-SP（TP 组内的序列并行，几乎不引入额外通信），当激活显存仍然不足时再叠加 CP。

因果注意力下，如果按 rank 顺序对序列连续切分，前段 rank 可见的历史 KV 少、计算量小，后段 rank 的历史 KV 多、计算量大，整体产生约两倍的不均衡，而慢的 rank 决定整体速度。通用的解决办法是 zigzag（首尾配对）切分，即将序列切成 $2C$ 个块，rank $r$ 持有块 $r$ 与块 $2C-1-r$，以一个轻载的前段块搭配一个重载的后段块，各 rank 的计算量由此趋于均衡。下图以 C=4 为例给出切分矩阵与因果 mask 矩阵的对应关系。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 590" style="width:100%;height:auto">
<defs>
<linearGradient id="bg0" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot0" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<filter id="gl0" x="-40%" y="-40%" width="180%" height="180%">
<feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect width="680" height="590" fill="url(#bg0)"/>
<rect width="680" height="590" fill="url(#dot0)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">zigzag 切分：把因果负载摊平</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">C = 4 个 rank，序列切成 2C = 8 块；块的颜色表示由哪个 rank 持有</text>

<rect x="24" y="70" width="632" height="232" rx="12" fill="#0E131D" stroke="#232C3A"/>
<circle cx="46" cy="92" r="9" fill="#F4D345"/>
<text x="46" y="96" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#0B0D12">1</text>
<text x="62" y="96" font-size="12.5" fill="#E6E8EC">顺序切分 —— rank 连续拿两块，后段越背越重</text>
<text x="640" y="96" text-anchor="end" font-size="11" fill="#FC6255">最重 / 最轻 = 5×</text>
<g stroke-width="1.2">
<rect x="60" y="108" width="64" height="26" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
<rect x="124" y="108" width="64" height="26" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
<rect x="188" y="108" width="64" height="26" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="252" y="108" width="64" height="26" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="316" y="108" width="64" height="26" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="380" y="108" width="64" height="26" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="444" y="108" width="64" height="26" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/>
<rect x="508" y="108" width="64" height="26" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/>
</g>
<g font-size="10" text-anchor="middle" fill="#DDE3EC">
<text x="92" y="125">B0</text><text x="156" y="125">B1</text><text x="220" y="125">B2</text><text x="284" y="125">B3</text>
<text x="348" y="125">B4</text><text x="412" y="125">B5</text><text x="476" y="125">B6</text><text x="540" y="125">B7</text>
</g>
<g font-size="10" text-anchor="middle">
<text x="124" y="150" fill="#58C4DD">R0</text><text x="252" y="150" fill="#5CD0B3">R1</text>
<text x="380" y="150" fill="#F4D345">R2</text><text x="508" y="150" fill="#FC6255">R3</text>
</g>
<g stroke="#0B0D12" stroke-width="0.8">
<rect x="70" y="160" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/>
<rect x="87" y="160" width="17" height="17" fill="#141A24"/><rect x="104" y="160" width="17" height="17" fill="#141A24"/><rect x="121" y="160" width="17" height="17" fill="#141A24"/>
<rect x="138" y="160" width="17" height="17" fill="#141A24"/><rect x="155" y="160" width="17" height="17" fill="#141A24"/><rect x="172" y="160" width="17" height="17" fill="#141A24"/><rect x="189" y="160" width="17" height="17" fill="#141A24"/>
<rect x="70" y="177" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="87" y="177" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/>
<rect x="104" y="177" width="17" height="17" fill="#141A24"/><rect x="121" y="177" width="17" height="17" fill="#141A24"/><rect x="138" y="177" width="17" height="17" fill="#141A24"/><rect x="155" y="177" width="17" height="17" fill="#141A24"/><rect x="172" y="177" width="17" height="17" fill="#141A24"/><rect x="189" y="177" width="17" height="17" fill="#141A24"/>
<rect x="70" y="194" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="87" y="194" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="104" y="194" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/>
<rect x="121" y="194" width="17" height="17" fill="#141A24"/><rect x="138" y="194" width="17" height="17" fill="#141A24"/><rect x="155" y="194" width="17" height="17" fill="#141A24"/><rect x="172" y="194" width="17" height="17" fill="#141A24"/><rect x="189" y="194" width="17" height="17" fill="#141A24"/>
<rect x="70" y="211" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="87" y="211" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="104" y="211" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="121" y="211" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/>
<rect x="138" y="211" width="17" height="17" fill="#141A24"/><rect x="155" y="211" width="17" height="17" fill="#141A24"/><rect x="172" y="211" width="17" height="17" fill="#141A24"/><rect x="189" y="211" width="17" height="17" fill="#141A24"/>
<rect x="70" y="228" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="87" y="228" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="104" y="228" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="121" y="228" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="138" y="228" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/>
<rect x="155" y="228" width="17" height="17" fill="#141A24"/><rect x="172" y="228" width="17" height="17" fill="#141A24"/><rect x="189" y="228" width="17" height="17" fill="#141A24"/>
<rect x="70" y="245" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="87" y="245" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="104" y="245" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="121" y="245" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="138" y="245" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="155" y="245" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/>
<rect x="172" y="245" width="17" height="17" fill="#141A24"/><rect x="189" y="245" width="17" height="17" fill="#141A24"/>
<rect x="70" y="262" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="87" y="262" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="104" y="262" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="121" y="262" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="138" y="262" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="155" y="262" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="172" y="262" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/>
<rect x="189" y="262" width="17" height="17" fill="#141A24"/>
<rect x="70" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="87" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="104" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="121" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="138" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="155" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="172" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="189" y="279" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/>
</g>
<text x="206" y="164" font-size="9" fill="#5B6678">负载（格）</text>
<g font-size="10" fill="#8B96A8">
<text x="213" y="190">R0</text><text x="213" y="224">R1</text><text x="213" y="258">R2</text><text x="213" y="292">R3</text>
</g>
<g font-size="15" fill="#E6E8EC" text-anchor="end">
<text x="246" y="190">3</text><text x="246" y="224">7</text><text x="246" y="258">11</text><text x="246" y="292" fill="#FC6255">15</text>
</g>

<g stroke="#2A3444" stroke-width="1">
<line x1="207" y1="172" x2="207" y2="300"/>
<line x1="202" y1="186" x2="210" y2="186"/><line x1="202" y1="220" x2="210" y2="220"/>
<line x1="202" y1="254" x2="210" y2="254"/><line x1="202" y1="288" x2="210" y2="288"/>
</g>
<text x="312" y="196" font-size="10.5" fill="#8B96A8">灰色格被因果 mask 屏蔽，</text>
<text x="312" y="212" font-size="10.5" fill="#8B96A8">不计入负载；R3 的 15 格</text>
<text x="312" y="228" font-size="10.5" fill="#FC6255">决定了其他 rank 的空等时间</text>
<g stroke="#FC6255" stroke-width="1.2" fill="none" marker-end="url(#ar0r)">
<line x1="252" y1="288" x2="304" y2="278"/>
</g>

<rect x="24" y="310" width="632" height="232" rx="12" fill="#0E131D" stroke="#232C3A"/>
<circle cx="46" cy="332" r="9" fill="#F4D345"/>
<text x="46" y="336" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#0B0D12">2</text>
<text x="62" y="336" font-size="12.5" fill="#E6E8EC">zigzag 切分 —— rank r 拿块 r 与块 2C−1−r，首尾配对</text>
<g stroke-width="1.2">
<rect x="60" y="348" width="64" height="26" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
<rect x="124" y="348" width="64" height="26" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/>
<rect x="188" y="348" width="64" height="26" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="252" y="348" width="64" height="26" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="316" y="348" width="64" height="26" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="380" y="348" width="64" height="26" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="444" y="348" width="64" height="26" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/>
<rect x="508" y="348" width="64" height="26" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
</g>
<g font-size="10" text-anchor="middle" fill="#DDE3EC">
<text x="92" y="365">B0</text><text x="156" y="365">B1</text><text x="220" y="365">B2</text><text x="284" y="365">B3</text>
<text x="348" y="365">B4</text><text x="412" y="365">B5</text><text x="476" y="365">B6</text><text x="540" y="365">B7</text>
</g>
<g stroke="#0B0D12" stroke-width="0.8">
<rect x="70" y="398" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/>
<rect x="87" y="398" width="17" height="17" fill="#141A24"/><rect x="104" y="398" width="17" height="17" fill="#141A24"/><rect x="121" y="398" width="17" height="17" fill="#141A24"/>
<rect x="138" y="398" width="17" height="17" fill="#141A24"/><rect x="155" y="398" width="17" height="17" fill="#141A24"/><rect x="172" y="398" width="17" height="17" fill="#141A24"/>
<rect x="189" y="398" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/>
<rect x="70" y="415" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="87" y="415" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/>
<rect x="104" y="415" width="17" height="17" fill="#141A24"/><rect x="121" y="415" width="17" height="17" fill="#141A24"/><rect x="138" y="415" width="17" height="17" fill="#141A24"/><rect x="155" y="415" width="17" height="17" fill="#141A24"/>
<rect x="172" y="415" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="189" y="415" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/>
<rect x="70" y="432" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="87" y="432" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="104" y="432" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/>
<rect x="121" y="432" width="17" height="17" fill="#141A24"/><rect x="138" y="432" width="17" height="17" fill="#141A24"/>
<rect x="155" y="432" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="172" y="432" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="189" y="432" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/>
<rect x="70" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="87" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="104" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="121" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/>
<rect x="138" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="155" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="172" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="189" y="449" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/>
<rect x="70" y="466" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="87" y="466" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="104" y="466" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="121" y="466" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/><rect x="138" y="466" width="17" height="17" fill="#FC6255" fill-opacity="0.5"/>
<rect x="155" y="466" width="17" height="17" fill="#141A24"/><rect x="172" y="466" width="17" height="17" fill="#141A24"/><rect x="189" y="466" width="17" height="17" fill="#141A24"/>
<rect x="70" y="483" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="87" y="483" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="104" y="483" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="121" y="483" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="138" y="483" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/>
<rect x="155" y="483" width="17" height="17" fill="#F4D345" fill-opacity="0.5"/><rect x="172" y="483" width="17" height="17" fill="#141A24"/><rect x="189" y="483" width="17" height="17" fill="#141A24"/>
<rect x="70" y="500" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="87" y="500" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="104" y="500" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="121" y="500" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="138" y="500" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/>
<rect x="155" y="500" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="172" y="500" width="17" height="17" fill="#5CD0B3" fill-opacity="0.5"/><rect x="189" y="500" width="17" height="17" fill="#141A24"/>
<rect x="70" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="87" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="104" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="121" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="138" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="155" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="172" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/><rect x="189" y="517" width="17" height="17" fill="#58C4DD" fill-opacity="0.5"/>
</g>
<text x="206" y="402" font-size="9" fill="#5B6678">负载（格）</text>
<g font-size="15" fill="#83C167" text-anchor="end">
<text x="246" y="429">9</text><text x="246" y="463">9</text><text x="246" y="497">9</text><text x="246" y="531">9</text>
</g>
<g stroke="#2A3444" stroke-width="1">
<line x1="207" y1="404" x2="207" y2="540"/>
<line x1="203" y1="428" x2="211" y2="428"/><line x1="203" y1="462" x2="211" y2="462"/>
<line x1="203" y1="496" x2="211" y2="496"/><line x1="203" y1="530" x2="211" y2="530"/>
</g>
<text x="312" y="430" font-size="10.5" fill="#83C167">四个 rank 各 9 格，负载完全相等，</text>
<text x="312" y="446" font-size="10.5" fill="#83C166">慢 rank 不再决定整体速度</text>
<text x="312" y="472" font-size="10.5" fill="#8B96A8">代价只是块的归属被打乱，</text>
<text x="312" y="488" font-size="10.5" fill="#8B96A8">后续按 rank 做 AllGather 时要还原顺序</text>

<g font-size="10" fill="#8B96A8">
<rect x="36" y="556" width="10" height="10" fill="#58C4DD" fill-opacity="0.5" stroke="#58C4DD" stroke-width="1"/>
<text x="52" y="565">R0</text>
<rect x="96" y="556" width="10" height="10" fill="#5CD0B3" fill-opacity="0.5" stroke="#5CD0B3" stroke-width="1"/>
<text x="112" y="565">R1</text>
<rect x="156" y="556" width="10" height="10" fill="#F4D345" fill-opacity="0.5" stroke="#F4D345" stroke-width="1"/>
<text x="172" y="565">R2</text>
<rect x="216" y="556" width="10" height="10" fill="#FC6255" fill-opacity="0.5" stroke="#FC6255" stroke-width="1"/>
<text x="232" y="565">R3</text>
<rect x="292" y="556" width="10" height="10" fill="#141A24" stroke="#2C3748" stroke-width="1"/>
<text x="308" y="565">因果 mask 屏蔽，不计负载</text>
</g>
<defs>
<marker id="ar0r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FC6255"/></marker>
</defs>
</g>
</svg>

▲ 切分矩阵与因果 mask 矩阵：切分矩阵给出每个 query 块所属的 rank；mask 矩阵以 query 块为行、key 块为列，下三角（着色）参与计算，上三角（灰）被因果 mask 屏蔽。顺序切分下各 rank 负载为 3/7/11/15，zigzag 首尾配对后各 rank 均为 9

## 2.1 Prefill CP

Prefill 阶段是算力受限（compute-bound）的。一个长请求包含 $T$ 个新 token，需要计算其 Q/K/V 并完成完整的因果注意力。Prefill CP 的目标是摊薄首 token 延迟（Time-To-First-Token，TTFT），做法是把请求切成 $N$ 个 chunk，每张 GPU 计算一个 chunk 的 Q/K/V 以及该 chunk 内 query 对应的注意力输出。按照 KV 的处理方式，已有做法可以归为两类。

第一类是 Partial Query + Full KV，即 AllGather KV。当请求长度适中、单卡能够容纳完整 KV 时，各 rank 先通过 AllGather 获得全局 KV，再只为本地那部分 query 计算注意力输出。这一策略实现最为简单，无须改动注意力 kernel，是 SGLang Prefill CP、vLLM PCP 与昇腾 DSA-CP 的共同选择；其代价是每层需要临时容纳全局 KV，工程上通常逐层聚合、用完即弃以压低显存峰值。由于 MLA、DSA、CSA 等现代架构把 KV 压缩得很小，AllGather 的代价在多数情况下可以接受。

第二类是 Partial Query + Partial KV，即 Ring Attention。当请求过长、全局 KV 无法容纳时，KV 块沿环形拓扑在各 rank 间点对点传递，每个 rank 用本地 Q 对收到的 KV 块计算部分注意力，并借助在线 softmax（LSE）合并中间结果。该策略的通信可以与计算重叠，但实现复杂度与对 kernel 的侵入性都更高。

在工程实现上，各引擎还有若干共同的处理。其一，头尾配对切分已成为标准配置，vLLM PCP 的 DualChunkSwap、SGLang 的 Zigzag 策略与 Megatron CP 均属此类。序列先被填充到 $2C$ 的倍数并切成 $2C$ 块，rank $i$ 持有 chunk $i$ 与 chunk $2C-1-i$，再通过 restore 索引把 AllGather 后交错的 token 顺序还原为因果顺序。其二，对于带 Indexer 的稀疏注意力（DSA、DeepSeek V4 系列），Indexer 的 top-k 选择计算同样随 query 切分到各 rank，避免了重复计算，这本身也构成 Prefill CP 的收益之一。其三，与 EP 组合时，CP 可以采用 all-to-all 形式的 token dispatch/combine 替代 TP 的 AllReduce，从而降低通信时间。从收益上看，SGLang 合入 CP 后长文本 TTFT 最多降低约 80%（PR #12065）；Moreh 的 SLOPE 引擎在 100K 上下文下将 TTFT 压到 2 秒以内，而同配置下 SGLang TP=8 约为 9 秒。

## 2.2 Decode CP

Decode 阶段是访存受限（memory-bound）的。每一步只有少量 query token，却需要扫描存放在分页 KV Cache 中的全部历史 KV。Decode CP 的目标因此与 Prefill CP 不同，它把 KV Cache 沿序列维切开以腾出显存，从而容纳更多并发请求（更大的 batch），最终提高吞吐。

KV Cache 的分片可以从两个维度理解。当模型有 $H$ 个 KV head 时，TP 可以先沿 head 维切分 KV，即普通的 TP 分片；但 $H$ 由模型结构决定，当 `tp_size > H` 时，每张 GPU 上的 KV Cache 会被复制 `tp_size / H` 份。以采用 MLA 的模型（DeepSeek-R1、Kimi-K2，推理时只有 1 个 KV head）为例，tp=8 意味着 8 倍的 KV 复制，此时需要 DCP 沿序列维进一步切分 KV Cache。具体而言，`dcp_size` 的取值范围是 $[1, tp\_size/H]$。DCP 不增加进程数，只是复用 TP/PCP 的 rank 来消除 KV 复制，`dcp_size` 越大复制越少，相应的通信开销也越大。

Decode 过程中 KV Cache 持续增长，分片方式必须保证各 rank 的显存均匀增长。vLLM 采用交错（interleave）分片，token $i$ 被存到 `dcp_rank (i % dcp_world_size)`（`cp_kv_cache_interleave_size=1`，token 级对齐），或者按 block 级对齐（先填满前序 rank 的 block 再写下一个 rank），使未来 token 的 KV 天然落到正确的 GPU 上。该方案由 Moonshot 的 Chao Hong 提出，Helix 论文（arXiv:2507.07120）给出了详细说明。

通信方式上，decode 与 prefill 恰好相反。decode 的 Q 极小、KV 极大，因此采用传 Q，即把 Q 广播或 AllGather 到各 rank，每个 rank 对本地 KV 分片计算 partial attention 输出与 softmax 统计量（LSE），最后做一次很小的合并（在线 softmax，通信量与上下文长度无关）。vLLM DCP 提供 `ag_rs`（AllGather + ReduceScatter）与 `a2a`（交换 partial output 与 LSE 后用 Triton kernel 合并）两种通信后端。

vLLM 官方文档给出了三个典型案例。DeepSeek-R1（MLA，1 个 KV head）在单机 tp=8 下存在 8 倍 KV 复制，加 `-dcp 8` 可完全消除；Kimi-K2 在 tp=16 下存在 16 倍复制，可用 `-dcp 16` 完全消除，或用 `-dcp 8` 把复制降到 2 倍并把通信限制在节点内；Qwen3-235B-A22B（4 个 KV head）在 tp=8 下存在 2 倍复制，加 `-dcp 2` 即可消除。消除 KV 复制之后，batch size 可以显著放大。Moonshot 在 Kimi K2 的技术分享中报告了 2~3 倍的吞吐提升，这对 RL rollout 等吞吐敏感场景尤为重要。

# 3. 不同的CP通信方案

CP 的通信集中于注意力，已有方案之间的本质差别在于 rank 之间传递什么数据。常见的做法包括传递 KV（Ring Attention 或 AllGather KV）、传递 Q（AllGather Q 或 Pass Q），以及 Ulysses 的 All-to-All 换维（其通信原语常被混称为 AllGather）。本章依次介绍这些方案，随后讨论它们共用的 LSE 合并组件，最后给出选型对比。

## 3.1 传 KV：Ring Attention（P2P 环）

Ring Attention 中 Q 留在本地，KV 块沿环形拓扑以 P2P send/recv 传递一圈。第 $t$ 轮，每个 rank 用手中的 KV 块计算本地 Q 的部分注意力结果，同时把该块发往下一个 rank、从上一个 rank 接收新块，中间结果用 FlashAttention 的 LSE 累加技巧合并，最终结果与单卡稠密注意力精确等价。只要单块的计算时间能够覆盖单块的传输时间，通信即可被完全隐藏。Megatron `cp_comm_type=p2p` 的默认实现就属于这一类，它在 zigzag 切分的基础上引入双 CUDA stream 重叠与 FlashAttention-2 集成，是工业级 Ring Attention 的代表。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 480" style="width:100%;height:auto">
<defs>
<linearGradient id="bg1" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot1" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a1o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FF862F"/></marker>
<marker id="a1g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#5B6678"/></marker>
</defs>
<rect width="680" height="480" fill="url(#bg1)"/>
<rect width="680" height="480" fill="url(#dot1)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">Ring Attention：Q 留在原地，KV 块绕环走一圈</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">C = 4。每轮各 rank 用手中的 K 块算一个 tile，同时把块 P2P 发给下家、从上家收新块</text>

<rect x="24" y="68" width="142" height="146" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="95" y="86" text-anchor="middle" font-size="10.5" fill="#E6E8EC">第 0 轮（初始）</text>
<rect x="28" y="108" width="3" height="16" fill="#58C4DD"/><text x="36" y="120" font-size="9" fill="#58C4DD">R0</text>
<rect x="74" y="108" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="102" y="120" text-anchor="middle" font-size="9" fill="#F0C39B">K0</text>
<rect x="28" y="128" width="3" height="16" fill="#5CD0B3"/><text x="36" y="140" font-size="9" fill="#5CD0B3">R1</text>
<rect x="74" y="128" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="102" y="140" text-anchor="middle" font-size="9" fill="#F0C39B">K1</text>
<rect x="28" y="148" width="3" height="16" fill="#F4D345"/><text x="36" y="160" font-size="9" fill="#F4D345">R2</text>
<rect x="74" y="148" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="102" y="160" text-anchor="middle" font-size="9" fill="#F0C39B">K2</text>
<rect x="28" y="168" width="3" height="16" fill="#FC6255"/><text x="36" y="180" font-size="9" fill="#FC6255">R3</text>
<rect x="74" y="168" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="102" y="180" text-anchor="middle" font-size="9" fill="#F0C39B">K3</text>
<text x="95" y="202" text-anchor="middle" font-size="9" fill="#F4D345">本轮 tile：i − j = 0</text>

<rect x="184" y="68" width="142" height="146" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="255" y="86" text-anchor="middle" font-size="10.5" fill="#E6E8EC">第 1 轮</text>
<rect x="188" y="108" width="3" height="16" fill="#58C4DD"/><text x="196" y="120" font-size="9" fill="#58C4DD">R0</text>
<rect x="234" y="108" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="262" y="120" text-anchor="middle" font-size="9" fill="#F0C39B">K3</text>
<rect x="188" y="128" width="3" height="16" fill="#5CD0B3"/><text x="196" y="140" font-size="9" fill="#5CD0B3">R1</text>
<rect x="234" y="128" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="262" y="140" text-anchor="middle" font-size="9" fill="#F0C39B">K0</text>
<rect x="188" y="148" width="3" height="16" fill="#F4D345"/><text x="196" y="160" font-size="9" fill="#F4D345">R2</text>
<rect x="234" y="148" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="262" y="160" text-anchor="middle" font-size="9" fill="#F0C39B">K1</text>
<rect x="188" y="168" width="3" height="16" fill="#FC6255"/><text x="196" y="180" font-size="9" fill="#FC6255">R3</text>
<rect x="234" y="168" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="262" y="180" text-anchor="middle" font-size="9" fill="#F0C39B">K2</text>
<text x="255" y="202" text-anchor="middle" font-size="9" fill="#F4D345">本轮 tile：i − j = 1</text>

<rect x="344" y="68" width="142" height="146" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="415" y="86" text-anchor="middle" font-size="10.5" fill="#E6E8EC">第 2 轮</text>
<rect x="348" y="108" width="3" height="16" fill="#58C4DD"/><text x="356" y="120" font-size="9" fill="#58C4DD">R0</text>
<rect x="394" y="108" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="422" y="120" text-anchor="middle" font-size="9" fill="#F0C39B">K2</text>
<rect x="348" y="128" width="3" height="16" fill="#5CD0B3"/><text x="356" y="140" font-size="9" fill="#5CD0B3">R1</text>
<rect x="394" y="128" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="422" y="140" text-anchor="middle" font-size="9" fill="#F0C39B">K3</text>
<rect x="348" y="148" width="3" height="16" fill="#F4D345"/><text x="356" y="160" font-size="9" fill="#F4D345">R2</text>
<rect x="394" y="148" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="422" y="160" text-anchor="middle" font-size="9" fill="#F0C39B">K0</text>
<rect x="348" y="168" width="3" height="16" fill="#FC6255"/><text x="356" y="180" font-size="9" fill="#FC6255">R3</text>
<rect x="394" y="168" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="422" y="180" text-anchor="middle" font-size="9" fill="#F0C39B">K1</text>
<text x="415" y="202" text-anchor="middle" font-size="9" fill="#F4D345">本轮 tile：i − j = 2</text>

<rect x="504" y="68" width="142" height="146" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="575" y="86" text-anchor="middle" font-size="10.5" fill="#E6E8EC">第 3 轮</text>
<rect x="508" y="108" width="3" height="16" fill="#58C4DD"/><text x="516" y="120" font-size="9" fill="#58C4DD">R0</text>
<rect x="554" y="108" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="582" y="120" text-anchor="middle" font-size="9" fill="#F0C39B">K1</text>
<rect x="508" y="128" width="3" height="16" fill="#5CD0B3"/><text x="516" y="140" font-size="9" fill="#5CD0B3">R1</text>
<rect x="554" y="128" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="582" y="140" text-anchor="middle" font-size="9" fill="#F0C39B">K2</text>
<rect x="508" y="148" width="3" height="16" fill="#F4D345"/><text x="516" y="160" font-size="9" fill="#F4D345">R2</text>
<rect x="554" y="148" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="582" y="160" text-anchor="middle" font-size="9" fill="#F0C39B">K3</text>
<rect x="508" y="168" width="3" height="16" fill="#FC6255"/><text x="516" y="180" font-size="9" fill="#FC6255">R3</text>
<rect x="554" y="168" width="56" height="16" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1"/><text x="582" y="180" text-anchor="middle" font-size="9" fill="#F0C39B">K0</text>
<text x="575" y="202" text-anchor="middle" font-size="9" fill="#F4D345">本轮 tile：i − j = 3</text>

<g stroke="#FF862F" stroke-width="1.4" fill="none">
<line x1="167" y1="146" x2="180" y2="146" marker-end="url(#a1o)"/>
<line x1="327" y1="146" x2="340" y2="146" marker-end="url(#a1o)"/>
<line x1="487" y1="146" x2="500" y2="146" marker-end="url(#a1o)"/>
</g>
<text x="340" y="232" text-anchor="middle" font-size="10" fill="#8B96A8">每两轮之间，4 对 send/recv 同时发生：块顺时针一跳（上图箭头），与 tile 计算重叠</text>

<text x="36" y="262" font-size="12" fill="#E6E8EC">score tile 矩阵：颜色 = 该 tile 在哪一轮被计算</text>
<g font-size="9.5" text-anchor="middle" fill="#8B96A8">
<text x="151" y="286">K0</text><text x="187" y="286">K1</text><text x="223" y="286">K2</text><text x="259" y="286">K3</text>
<text x="126" y="305" text-anchor="end">Q0</text><text x="126" y="341" text-anchor="end">Q1</text>
<text x="126" y="377" text-anchor="end">Q2</text><text x="126" y="413" text-anchor="end">Q3</text>
</g>
<g stroke="#0B0D12" stroke-width="1">
<rect x="133" y="292" width="36" height="36" fill="#83C167" fill-opacity="0.45"/><rect x="169" y="292" width="36" height="36" fill="#141A24"/><rect x="205" y="292" width="36" height="36" fill="#141A24"/><rect x="241" y="292" width="36" height="36" fill="#141A24"/>
<rect x="133" y="328" width="36" height="36" fill="#58C4DD" fill-opacity="0.45"/><rect x="169" y="328" width="36" height="36" fill="#83C167" fill-opacity="0.45"/><rect x="205" y="328" width="36" height="36" fill="#141A24"/><rect x="241" y="328" width="36" height="36" fill="#141A24"/>
<rect x="133" y="364" width="36" height="36" fill="#F4D345" fill-opacity="0.45"/><rect x="169" y="364" width="36" height="36" fill="#58C4DD" fill-opacity="0.45"/><rect x="205" y="364" width="36" height="36" fill="#83C167" fill-opacity="0.45"/><rect x="241" y="364" width="36" height="36" fill="#141A24"/>
<rect x="133" y="400" width="36" height="36" fill="#FC6255" fill-opacity="0.45"/><rect x="169" y="400" width="36" height="36" fill="#F4D345" fill-opacity="0.45"/><rect x="205" y="400" width="36" height="36" fill="#58C4DD" fill-opacity="0.45"/><rect x="241" y="400" width="36" height="36" fill="#83C167" fill-opacity="0.45"/>
</g>
<g font-size="9" text-anchor="middle" fill="#E6E8EC">
<text x="151" y="314">轮0</text><text x="151" y="350">轮1</text><text x="151" y="386">轮2</text><text x="151" y="422">轮3</text>
<text x="187" y="350">轮0</text><text x="187" y="386">轮1</text><text x="187" y="422">轮2</text>
<text x="223" y="386">轮0</text><text x="223" y="422">轮1</text><text x="259" y="422">轮0</text>
<text x="187" y="314" fill="#46536A">mask</text><text x="223" y="314" fill="#46536A">mask</text><text x="259" y="314" fill="#46536A">mask</text>
<text x="223" y="350" fill="#46536A">mask</text><text x="259" y="350" fill="#46536A">mask</text><text x="259" y="386" fill="#46536A">mask</text>
</g>
<g font-size="10" fill="#8B96A8">
<rect x="316" y="292" width="11" height="11" fill="#83C167" fill-opacity="0.45" stroke="#83C167" stroke-width="1"/>
<text x="336" y="302">轮 0 —— 对角 tile（i = j）</text>
<rect x="316" y="318" width="11" height="11" fill="#58C4DD" fill-opacity="0.45" stroke="#58C4DD" stroke-width="1"/>
<text x="336" y="328">轮 1 —— 次对角（i − j = 1）</text>
<rect x="316" y="344" width="11" height="11" fill="#F4D345" fill-opacity="0.45" stroke="#F4D345" stroke-width="1"/>
<text x="336" y="354">轮 2 —— i − j = 2</text>
<rect x="316" y="370" width="11" height="11" fill="#FC6255" fill-opacity="0.45" stroke="#FC6255" stroke-width="1"/>
<text x="336" y="380">轮 3 —— i − j = 3</text>
<text x="316" y="408" font-size="10" fill="#5B6678">下三角被 4 轮扫完，</text>
<text x="316" y="424" font-size="10" fill="#5B6678">灰格（j &gt; i）被因果 mask 屏蔽，</text>
<text x="316" y="440" font-size="10" fill="#5B6678">块仍照常流转但不算。</text>
</g>
<rect x="24" y="452" width="632" height="20" rx="6" fill="#121826" stroke="#232C3A"/>
<text x="340" y="466" text-anchor="middle" font-size="10" fill="#9AA4B2">通信量 O(T)：KV 沿环走一遍。转满 C 轮后每个 rank 拿齐自己整行 → LSE 合并为精确结果</text>
</g>
</svg>

▲ Ring Attention 每轮通信日程：上排为 4 轮中各 GPU 持有的 K 块与每轮的 P2P 跳转；下排为 score tile 矩阵（颜色表示该 tile 被计算的轮次，灰为因果 mask）

该方案的优点是 CP 度不受 head 数限制，通信量为 $O(T)$（KV 沿环走一遍）；其代价在于多轮流水通信与结果合并要求 kernel 支持 partial output 与 LSE 累加，且块较小时 kernel 效率会下降。

## 3.2 传 KV：AllGather KV

AllGather KV 一次性聚齐完整 KV，本地 Q 对全局 KV 用标准注意力 kernel 计算即可。这一方案实现最简单、不改动 kernel、不受 head 数限制，但需要临时容纳整层的全局 KV，且全量 KV 要在 CP 组内完整传输一遍。由于 MLA、DSA、CSA 等架构把 KV 压缩得非常小（MLA 传 latent，CSA 传压缩条目），先 AllGather KV 再计算往往同时兼顾了效率与实现便利性，SGLang Prefill CP、vLLM PCP 与昇腾 DSA-CP 均采用该方案。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 452" style="width:100%;height:auto">
<defs>
<linearGradient id="bg2" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot2" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a2s" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#5B6678"/></marker>
<marker id="a2o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FF862F"/></marker>
</defs>
<rect width="680" height="452" fill="url(#bg2)"/>
<rect width="680" height="452" fill="url(#dot2)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">AllGather KV：三步把 KV 聚齐再算</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">Q 始终按行块不动，变的是每个 rank 手里的 KV —— 从 1/C 变成完整一份</text>

<rect x="24" y="76" width="196" height="270" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="42" cy="96" r="9" fill="#F4D345"/>
<text x="42" y="100" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="58" y="100" font-size="11.5" fill="#E6E8EC">通信前：各持 1/4</text>
<rect x="34" y="112" width="3" height="20" fill="#58C4DD"/><text x="43" y="126" font-size="9" fill="#58C4DD">R0</text>
<rect x="86" y="112" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.34" stroke="#FF862F" stroke-width="1"/><text x="98" y="126" text-anchor="middle" font-size="8" fill="#F0C39B">K0</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="115" y="112" width="25" height="20" rx="2"/><rect x="144" y="112" width="25" height="20" rx="2"/><rect x="173" y="112" width="25" height="20" rx="2"/></g>
<rect x="34" y="142" width="3" height="20" fill="#5CD0B3"/><text x="43" y="156" font-size="9" fill="#5CD0B3">R1</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="86" y="142" width="25" height="20" rx="2"/></g>
<rect x="115" y="142" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.34" stroke="#FF862F" stroke-width="1"/><text x="127" y="156" text-anchor="middle" font-size="8" fill="#F0C39B">K1</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="144" y="142" width="25" height="20" rx="2"/><rect x="173" y="142" width="25" height="20" rx="2"/></g>
<rect x="34" y="172" width="3" height="20" fill="#F4D345"/><text x="43" y="186" font-size="9" fill="#F4D345">R2</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="86" y="172" width="25" height="20" rx="2"/><rect x="115" y="172" width="25" height="20" rx="2"/></g>
<rect x="144" y="172" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.34" stroke="#FF862F" stroke-width="1"/><text x="156" y="186" text-anchor="middle" font-size="8" fill="#F0C39B">K2</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="173" y="172" width="25" height="20" rx="2"/></g>
<rect x="34" y="202" width="3" height="20" fill="#FC6255"/><text x="43" y="216" font-size="9" fill="#FC6255">R3</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="86" y="202" width="25" height="20" rx="2"/><rect x="115" y="202" width="25" height="20" rx="2"/><rect x="144" y="202" width="25" height="20" rx="2"/></g>
<rect x="173" y="202" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.34" stroke="#FF862F" stroke-width="1"/><text x="185" y="216" text-anchor="middle" font-size="8" fill="#F0C39B">K3</text>
<text x="40" y="252" font-size="9.5" fill="#5B6678">虚线格 = 本 rank 没有的 KV</text>
<text x="40" y="270" font-size="9.5" fill="#5B6678">每行只有 1 块，缺 3 块</text>
<text x="40" y="292" font-size="9.5" fill="#8B96A8">KV 显存占用 T·d / C</text>

<rect x="250" y="76" width="196" height="270" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="268" cy="96" r="9" fill="#F4D345"/>
<text x="268" y="100" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="284" y="100" font-size="11.5" fill="#E6E8EC">AllGather：补齐</text>
<rect x="260" y="112" width="3" height="20" fill="#58C4DD"/><text x="269" y="126" font-size="9" fill="#58C4DD">R0</text>
<g stroke="#FF862F" stroke-width="1">
<rect x="312" y="112" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.18"/><rect x="341" y="112" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.28"/><rect x="370" y="112" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.38"/><rect x="399" y="112" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.52"/></g>
<rect x="260" y="142" width="3" height="20" fill="#5CD0B3"/><text x="269" y="156" font-size="9" fill="#5CD0B3">R1</text>
<g stroke="#FF862F" stroke-width="1">
<rect x="312" y="142" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.18"/><rect x="341" y="142" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.28"/><rect x="370" y="142" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.38"/><rect x="399" y="142" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.52"/></g>
<rect x="260" y="172" width="3" height="20" fill="#F4D345"/><text x="269" y="186" font-size="9" fill="#F4D345">R2</text>
<g stroke="#FF862F" stroke-width="1">
<rect x="312" y="172" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.18"/><rect x="341" y="172" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.28"/><rect x="370" y="172" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.38"/><rect x="399" y="172" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.52"/></g>
<rect x="260" y="202" width="3" height="20" fill="#FC6255"/><text x="269" y="216" font-size="9" fill="#FC6255">R3</text>
<g stroke="#FF862F" stroke-width="1">
<rect x="312" y="202" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.18"/><rect x="341" y="202" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.28"/><rect x="370" y="202" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.38"/><rect x="399" y="202" width="25" height="20" rx="2" fill="#FF862F" fill-opacity="0.52"/></g>
<g font-size="8" text-anchor="middle" fill="#F0C39B">
<text x="324" y="126">K0</text><text x="353" y="126">K1</text><text x="382" y="126">K2</text><text x="411" y="126">K3</text>
<text x="324" y="156">K0</text><text x="353" y="156">K1</text><text x="382" y="156">K2</text><text x="411" y="156">K3</text>
<text x="324" y="186">K0</text><text x="353" y="186">K1</text><text x="382" y="186">K2</text><text x="411" y="186">K3</text>
<text x="324" y="216">K0</text><text x="353" y="216">K1</text><text x="382" y="216">K2</text><text x="411" y="216">K3</text>
</g>
<text x="266" y="252" font-size="9.5" fill="#5B6678">深浅 = 数据来自哪个 rank</text>
<text x="266" y="270" font-size="9.5" fill="#5B6678">4 块全部到手，每 rank 一份副本</text>
<text x="266" y="292" font-size="9.5" fill="#8B96A8">KV 显存占用 T·d（每层临时）</text>

<rect x="476" y="76" width="196" height="270" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="494" cy="96" r="9" fill="#F4D345"/>
<text x="494" y="100" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">3</text>
<text x="510" y="100" font-size="11.5" fill="#E6E8EC">计算：整行块一次算完</text>
<g stroke="#FF862F" stroke-width="0.8">
<rect x="530" y="116" width="22" height="22" fill="#FF862F" fill-opacity="0.2"/><rect x="552" y="116" width="22" height="22" fill="#FF862F" fill-opacity="0.3"/><rect x="574" y="116" width="22" height="22" fill="#FF862F" fill-opacity="0.4"/><rect x="596" y="116" width="22" height="22" fill="#FF862F" fill-opacity="0.52"/>
<rect x="530" y="138" width="22" height="22" fill="#FF862F" fill-opacity="0.2"/><rect x="552" y="138" width="22" height="22" fill="#FF862F" fill-opacity="0.3"/><rect x="574" y="138" width="22" height="22" fill="#FF862F" fill-opacity="0.4"/><rect x="596" y="138" width="22" height="22" fill="#FF862F" fill-opacity="0.52"/>
<rect x="530" y="160" width="22" height="22" fill="#FF862F" fill-opacity="0.2"/><rect x="552" y="160" width="22" height="22" fill="#FF862F" fill-opacity="0.3"/><rect x="574" y="160" width="22" height="22" fill="#FF862F" fill-opacity="0.4"/><rect x="596" y="160" width="22" height="22" fill="#FF862F" fill-opacity="0.52"/>
<rect x="530" y="182" width="22" height="22" fill="#FF862F" fill-opacity="0.2"/><rect x="552" y="182" width="22" height="22" fill="#FF862F" fill-opacity="0.3"/><rect x="574" y="182" width="22" height="22" fill="#FF862F" fill-opacity="0.4"/><rect x="596" y="182" width="22" height="22" fill="#FF862F" fill-opacity="0.52"/>
</g>
<rect x="530" y="108" width="88" height="6" rx="3" fill="#58C4DD" fill-opacity="0.75"/>
<text x="624" y="114" font-size="8.5" fill="#8FC9DC">Q0 行块</text>
<text x="586" y="222" text-anchor="middle" font-size="8.5" fill="#5B6678">Q0 × KV_full → 一次算完</text>
<line x1="586" y1="228" x2="586" y2="226" stroke="#5B6678" stroke-width="0"/>
<rect x="520" y="228" width="132" height="22" rx="4" fill="#58C4DD" fill-opacity="0.18" stroke="#58C4DD" stroke-width="1"/>
<text x="586" y="243" text-anchor="middle" font-size="9" fill="#BEE7F1">O0 → O3 各行块</text>
<text x="490" y="268" font-size="9.5" fill="#5B6678">标准 FlashAttention，不用 LSE 合并</text>
<text x="490" y="286" font-size="9.5" fill="#83C167">不改 kernel、CP 度不受 head 数限制</text>

<line x1="222" y1="200" x2="246" y2="200" stroke="#FF862F" stroke-width="1.5" marker-end="url(#a2o)"/>

<line x1="448" y1="200" x2="472" y2="200" stroke="#FF862F" stroke-width="1.5" marker-end="url(#a2o)"/>


<g font-size="10" fill="#8B96A8">
<rect x="30" y="362" width="11" height="11" fill="#FF862F" fill-opacity="0.34" stroke="#FF862F" stroke-width="1"/>
<text x="48" y="372">已到手的 KV 块</text>
<rect x="176" y="362" width="11" height="11" fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3"/>
<text x="194" y="372">尚未到达</text>
<rect x="300" y="362" width="11" height="11" fill="#58C4DD" fill-opacity="0.4" stroke="#58C4DD" stroke-width="1"/>
<text x="318" y="372">本地 Q 行块</text>
</g>
<rect x="24" y="392" width="632" height="44" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="410" font-size="10" fill="#9AA4B2">代价：每层临时放下整份 KV（T·d），全量 KV 在组内传一遍；收益：实现简单、不改 kernel、CP 度不受 head 数限制。</text>
<text x="44" y="427" font-size="10" fill="#9AA4B2">MLA、DSA、CSA 等把 KV 压得很小，先聚齐再算往往同时兼顾效率与实现便利。</text>
</g>
</svg>

▲ AllGather KV 示意图：通信前每 rank 只持有自己的 KV 分片（虚线格为缺失块），AllGather 后每 rank 补齐全局 KV 矩阵（颜色深浅区分来源 rank），Q 行块始终不动

## 3.3 传 Q：AllGather Q / Pass Q

与传 KV 相对，传 Q 让 KV 按序列维分片留在本地不动，只把（很小的）Q 送达各 rank。每个 rank 对本地 KV 分片计算 partial output $O_j$ 与 LSE $L_j$，再做合并。由于通信量与 KV Cache 大小无关，传 Q 是 decode 阶段的首选。按照 Q 的送达方式，可以进一步分为两种形态。

### 3.3.1 AllGather / Broadcast Q

该形态用一次集合通信把 Q 聚齐到每个 rank。当 DCP 组与 TP 组重叠时，各 rank 只持有自己那部分 head 的 Q 分片，需要先沿 head 维 AllGather 得到完整 Q；当 DCP 组独立时，Q 本来就在各 rank 复制，一次 broadcast 即可。随后各 rank 用完整 Q 对本地 KV 分片计算 score 列段 $S_j$ 并得到 $(O_j, L_j)$，再做一次 LSE 感知的归并（all-reduce 或 gather，例如 vLLM 的 `cp_lse_ag_out_rs`）。全程只需两个集合通信步骤，延迟低，是 SGLang Decode CP（all-gather Q）、vLLM DCP 与 Helix KV Parallelism 的共同选择。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 452" style="width:100%;height:auto">
<defs>
<linearGradient id="bg3" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot3" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a3q" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#58C4DD"/></marker>
<marker id="a3g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#83C167"/></marker>
</defs>
<rect width="680" height="452" fill="url(#bg3)"/>
<rect width="680" height="452" fill="url(#dot3)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">AllGather Q：KV 不动，把小 Q 聚齐到每个 rank</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">三步走完 decode 的一次注意力。通信量只与 q·d 有关，与上下文长度无关</text>

<rect x="24" y="76" width="196" height="270" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="42" cy="96" r="9" fill="#F4D345"/>
<text x="42" y="100" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="58" y="100" font-size="11.5" fill="#E6E8EC">通信前：只有 1/4 head</text>
<text x="70" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h0</text><text x="99" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h1</text>
<text x="128" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h2</text><text x="157" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h3</text>
<rect x="34" y="128" width="3" height="22" fill="#58C4DD"/><text x="43" y="143" font-size="9" fill="#58C4DD">R0</text>
<rect x="58" y="128" width="25" height="22" rx="2" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD" stroke-width="1"/>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="87" y="128" width="25" height="22" rx="2"/><rect x="116" y="128" width="25" height="22" rx="2"/><rect x="145" y="128" width="25" height="22" rx="2"/></g>
<rect x="34" y="162" width="3" height="22" fill="#5CD0B3"/><text x="43" y="177" font-size="9" fill="#5CD0B3">R1</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="58" y="162" width="25" height="22" rx="2"/></g>
<rect x="87" y="162" width="25" height="22" rx="2" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3" stroke-width="1"/>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="116" y="162" width="25" height="22" rx="2"/><rect x="145" y="162" width="25" height="22" rx="2"/></g>
<rect x="34" y="196" width="3" height="22" fill="#F4D345"/><text x="43" y="211" font-size="9" fill="#F4D345">R2</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="58" y="196" width="25" height="22" rx="2"/><rect x="87" y="196" width="25" height="22" rx="2"/></g>
<rect x="116" y="196" width="25" height="22" rx="2" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345" stroke-width="1"/>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="145" y="196" width="25" height="22" rx="2"/></g>
<rect x="34" y="230" width="3" height="22" fill="#FC6255"/><text x="43" y="245" font-size="9" fill="#FC6255">R3</text>
<g fill="none" stroke="#39445A" stroke-width="1" stroke-dasharray="2 3">
<rect x="58" y="230" width="25" height="22" rx="2"/><rect x="87" y="230" width="25" height="22" rx="2"/><rect x="116" y="230" width="25" height="22" rx="2"/></g>
<rect x="145" y="230" width="25" height="22" rx="2" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255" stroke-width="1"/>
<text x="40" y="276" font-size="9.5" fill="#5B6678">DCP 组与 TP 组重叠时，Q 按 head</text>
<text x="40" y="292" font-size="9.5" fill="#5B6678">分片；不重叠时 Q 本有副本，</text>
<text x="40" y="308" font-size="9.5" fill="#5B6678">一次 broadcast 即可</text>
<text x="40" y="330" font-size="9.5" fill="#8B96A8">KV 分片常驻不动</text>

<rect x="250" y="76" width="196" height="270" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="268" cy="96" r="9" fill="#F4D345"/>
<text x="268" y="100" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="284" y="100" font-size="11.5" fill="#E6E8EC">AllGather：补齐 Q</text>
<text x="296" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h0</text><text x="325" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h1</text>
<text x="354" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h2</text><text x="383" y="122" text-anchor="middle" font-size="8.5" fill="#5B6678">h3</text>
<rect x="260" y="128" width="3" height="22" fill="#58C4DD"/><text x="269" y="143" font-size="9" fill="#58C4DD">R0</text>
<g stroke-width="1">
<rect x="284" y="128" width="25" height="22" rx="2" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
<rect x="313" y="128" width="25" height="22" rx="2" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="342" y="128" width="25" height="22" rx="2" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="371" y="128" width="25" height="22" rx="2" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/></g>
<rect x="260" y="162" width="3" height="22" fill="#5CD0B3"/><text x="269" y="177" font-size="9" fill="#5CD0B3">R1</text>
<g stroke-width="1">
<rect x="284" y="162" width="25" height="22" rx="2" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
<rect x="313" y="162" width="25" height="22" rx="2" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="342" y="162" width="25" height="22" rx="2" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="371" y="162" width="25" height="22" rx="2" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/></g>
<rect x="260" y="196" width="3" height="22" fill="#F4D345"/><text x="269" y="211" font-size="9" fill="#F4D345">R2</text>
<g stroke-width="1">
<rect x="284" y="196" width="25" height="22" rx="2" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
<rect x="313" y="196" width="25" height="22" rx="2" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="342" y="196" width="25" height="22" rx="2" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="371" y="196" width="25" height="22" rx="2" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/></g>
<rect x="260" y="230" width="3" height="22" fill="#FC6255"/><text x="269" y="245" font-size="9" fill="#FC6255">R3</text>
<g stroke-width="1">
<rect x="284" y="230" width="25" height="22" rx="2" fill="#58C4DD" fill-opacity="0.38" stroke="#58C4DD"/>
<rect x="313" y="230" width="25" height="22" rx="2" fill="#5CD0B3" fill-opacity="0.38" stroke="#5CD0B3"/>
<rect x="342" y="230" width="25" height="22" rx="2" fill="#F4D345" fill-opacity="0.38" stroke="#F4D345"/>
<rect x="371" y="230" width="25" height="22" rx="2" fill="#FC6255" fill-opacity="0.38" stroke="#FC6255"/></g>
<text x="266" y="276" font-size="9.5" fill="#5B6678">每个 rank 补齐完整 Q，q·d 量级</text>
<text x="266" y="292" font-size="9.5" fill="#5B6678">来源用颜色区分</text>
<text x="266" y="318" font-size="9.5" fill="#8B96A8">通信量 ∝ q·d，与 T 无关</text>

<rect x="476" y="76" width="196" height="270" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="494" cy="96" r="9" fill="#F4D345"/>
<text x="494" y="100" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">3</text>
<text x="510" y="100" font-size="11.5" fill="#E6E8EC">各算列段，再归并</text>
<rect x="500" y="116" width="148" height="26" rx="4" fill="#0E131D" stroke="#3A4557"/>
<text x="510" y="133" font-size="9" fill="#9AA4B2">R0  S₀ = Q·K₀ᵀ</text><text x="642" y="133" text-anchor="end" font-size="9" fill="#58C4DD">(O₀,L₀)</text>
<rect x="500" y="148" width="148" height="26" rx="4" fill="#0E131D" stroke="#3A4557"/>
<text x="510" y="165" font-size="9" fill="#9AA4B2">R1  S₁ = Q·K₁ᵀ</text><text x="642" y="165" text-anchor="end" font-size="9" fill="#5CD0B3">(O₁,L₁)</text>
<rect x="500" y="180" width="148" height="26" rx="4" fill="#0E131D" stroke="#3A4557"/>
<text x="510" y="197" font-size="9" fill="#9AA4B2">R2  S₂ = Q·K₂ᵀ</text><text x="642" y="197" text-anchor="end" font-size="9" fill="#F4D345">(O₂,L₂)</text>
<rect x="500" y="212" width="148" height="26" rx="4" fill="#0E131D" stroke="#3A4557"/>
<text x="510" y="229" font-size="9" fill="#9AA4B2">R3  S₃ = Q·K₃ᵀ</text><text x="642" y="229" text-anchor="end" font-size="9" fill="#FC6255">(O₃,L₃)</text>
<g stroke="#83C167" stroke-width="1.2" fill="none">
<line x1="574" y1="240" x2="574" y2="258" marker-end="url(#a3g)"/>
</g>
<rect x="498" y="264" width="152" height="24" rx="4" fill="#83C167" fill-opacity="0.2" stroke="#83C167" stroke-width="1.2"/>
<text x="574" y="280" text-anchor="middle" font-size="9.5" fill="#C6E3B6">LSE 归并 → 精确 O</text>
<text x="490" y="310" font-size="9.5" fill="#5B6678">全程两次集合通信：AllGather Q</text>
<text x="490" y="326" font-size="9.5" fill="#5B6678">+ LSE 感知归并，延迟低</text>

<line x1="222" y1="190" x2="246" y2="190" stroke="#58C4DD" stroke-width="1.5" marker-end="url(#a3q)"/>

<line x1="448" y1="190" x2="472" y2="190" stroke="#58C4DD" stroke-width="1.5" marker-end="url(#a3q)"/>


<rect x="24" y="362" width="632" height="20" rx="6" fill="#121826" stroke="#232C3A"/>
<text x="340" y="376" text-anchor="middle" font-size="10" fill="#9AA4B2">通信量只与 q·d 成正比，与 KV Cache 大小、上下文长度都无关 —— 这是 decode 阶段首选传 Q 的原因</text>
<rect x="24" y="392" width="632" height="44" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="410" font-size="10" fill="#9AA4B2">归并时按列段还原顺序：第 j 段对应 K 的第 j 个 T 分片，vocab 维无关，因此只合并 (O_j, L_j) 两个小张量。</text>
<text x="44" y="427" font-size="10" fill="#9AA4B2">SGLang Decode CP、vLLM DCP（cp_lse_ag_out_rs）、Helix 的 KV Parallelism 都是这一形态。</text>
</g>
</svg>

▲ AllGather Q 示意图：通信前每 rank 只有 1/C 的 Q head 分片（虚线格为缺失），AllGather 后每 rank 补齐完整 Q（颜色深浅区分来源 rank），KV 分片始终不动；随后各 rank 算 score 列段并 LSE 合并

### 3.3.2 Pass Q（环形传 Q）

Pass Q 可以看作 Ring 传 KV 的镜像。KV 分片常驻不动，Q 连同运行中的 partial 累加器 $(O_{acc}, L_{acc})$ 沿环逐跳以 P2P 传递。每个 rank 收到后用本地 KV 计算 partial，按在线 softmax 并入累加器，再转发给下一个 rank；转满 $C$ 跳回到源 rank 时，累加器即精确输出，最后的归并通信也随之省去。这一形态的通信量同样只与 $q$ 成正比，但 $C$ 跳串行使延迟高于 AllGather Q，因而更适合 KV 常驻缓存的多轮 decode 与 incremental prefill。Meta 在《Context Parallelism for Scalable Million-Token Inference》中提出的 pass-Q 变体即属此类，它与 pass-KV 之间的动态切换将在 3.5 节讨论。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 462" style="width:100%;height:auto">
<defs>
<linearGradient id="bg4" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot4" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a4b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#58C4DD"/></marker>
<marker id="a4g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#83C167"/></marker>
</defs>
<rect width="680" height="462" fill="url(#bg4)"/>
<rect width="680" height="462" fill="url(#dot4)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">Pass Q：让 Q 拖着累加器，沿环走一圈</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">Ring 传 KV 的镜像。KV 分片常驻，Q 与 (O, L) 逐跳下传，就地在线合并</text>

<rect x="24" y="76" width="232" height="226" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="140" y="96" text-anchor="middle" font-size="10.5" fill="#8B96A8">KV 分片常驻，一次都不搬</text>
<rect x="46" y="106" width="88" height="44" rx="6" fill="#FF862F" fill-opacity="0.15" stroke="#FF862F" stroke-width="1.2"/>
<text x="90" y="124" text-anchor="middle" font-size="9.5" fill="#F0C39B">R0</text>
<text x="90" y="140" text-anchor="middle" font-size="8.5" fill="#9AA4B2">KV₀</text>
<rect x="152" y="106" width="88" height="44" rx="6" fill="#FF862F" fill-opacity="0.15" stroke="#FF862F" stroke-width="1.2"/>
<text x="196" y="124" text-anchor="middle" font-size="9.5" fill="#F0C39B">R1</text>
<text x="196" y="140" text-anchor="middle" font-size="8.5" fill="#9AA4B2">KV₁</text>
<rect x="152" y="200" width="88" height="44" rx="6" fill="#FF862F" fill-opacity="0.15" stroke="#FF862F" stroke-width="1.2"/>
<text x="196" y="218" text-anchor="middle" font-size="9.5" fill="#F0C39B">R2</text>
<text x="196" y="234" text-anchor="middle" font-size="8.5" fill="#9AA4B2">KV₂</text>
<rect x="46" y="200" width="88" height="44" rx="6" fill="#FF862F" fill-opacity="0.15" stroke="#FF862F" stroke-width="1.2"/>
<text x="90" y="218" text-anchor="middle" font-size="9.5" fill="#F0C39B">R3</text>
<text x="90" y="234" text-anchor="middle" font-size="8.5" fill="#9AA4B2">KV₃</text>
<g stroke="#58C4DD" stroke-width="1.4" fill="none">
<path d="M136 118 L 148 118" marker-end="url(#a4b)"/>
<path d="M196 152 L 196 196" marker-end="url(#a4b)"/>
<path d="M150 224 L 138 224" marker-end="url(#a4b)"/>
<path d="M90 198 L 90 156" marker-end="url(#a4b)"/>
</g>
<text x="140" y="180" text-anchor="middle" font-size="9" fill="#8FC9DC">Q + acc 顺时针一跳</text>
<text x="140" y="290" text-anchor="middle" font-size="9" fill="#5B6678">每跳 = 一次 P2P，全程 C 跳</text>

<defs><marker id="a4d" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#3A4557"/></marker></defs>
<rect x="280" y="86" width="376" height="44" rx="6" fill="#0E131D" stroke="#3A4557"/>
<circle cx="302" cy="108" r="10" fill="#58C4DD"/>
<text x="302" y="112" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#0B0D12">0</text>
<text x="320" y="103" font-size="10.5" fill="#E6E8EC">跳 0 · R0 用 KV₀ 算出 S₀</text>
<text x="320" y="119" font-size="9" fill="#5B6678">acc = (O₀, L₀)，这是只有一个分片的中间结果</text>
<text x="646" y="108" text-anchor="end" font-size="9" fill="#8FC9DC">→ R1</text>
<line x1="468" y1="130" x2="468" y2="134" stroke="#3A4557" stroke-width="1.2" marker-end="url(#a4d)"/>

<rect x="280" y="138" width="376" height="44" rx="6" fill="#0E131D" stroke="#3A4557"/>
<circle cx="302" cy="160" r="10" fill="#58C4DD"/>
<text x="302" y="164" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#0B0D12">1</text>
<text x="320" y="155" font-size="10.5" fill="#E6E8EC">跳 1 · R1 用 KV₁ 算出 S₁，就地并入</text>
<text x="320" y="171" font-size="9" fill="#5B6678">按在线 softmax 重标定权重，acc 覆盖 2 个分片</text>
<text x="646" y="160" text-anchor="end" font-size="9" fill="#8FC9DC">→ R2</text>
<line x1="468" y1="182" x2="468" y2="186" stroke="#3A4557" stroke-width="1.2" marker-end="url(#a4d)"/>

<rect x="280" y="190" width="376" height="44" rx="6" fill="#0E131D" stroke="#3A4557"/>
<circle cx="302" cy="212" r="10" fill="#58C4DD"/>
<text x="302" y="216" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#0B0D12">2</text>
<text x="320" y="207" font-size="10.5" fill="#E6E8EC">跳 2 · R2 用 KV₂ 算出 S₂</text>
<text x="320" y="223" font-size="9" fill="#5B6678">acc 覆盖 3 个分片，仍未归一化</text>
<text x="646" y="212" text-anchor="end" font-size="9" fill="#8FC9DC">→ R3</text>
<line x1="468" y1="234" x2="468" y2="238" stroke="#3A4557" stroke-width="1.2" marker-end="url(#a4d)"/>

<rect x="280" y="242" width="376" height="44" rx="6" fill="#83C167" fill-opacity="0.12" stroke="#83C167" stroke-width="1.3"/>
<circle cx="302" cy="264" r="10" fill="#83C167"/>
<text x="302" y="268" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#0B0D12">3</text>
<text x="320" y="259" font-size="10.5" fill="#C6E3B6">跳 3 · R3 并入最后一段，acc 回到 R0</text>
<text x="320" y="275" font-size="9" fill="#7FA36F">4 段齐了，acc 就是精确 O —— 归并通信直接省掉</text>
<text x="646" y="264" text-anchor="end" font-size="9" fill="#83C167">输出 O</text>

<rect x="24" y="316" width="632" height="56" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="336" font-size="10.5" fill="#9AA4B2">为什么值得多走几跳：decode 每步只有 1 个 query token，Q 与 acc 都是小张量，一路 P2P 只传 O(C·d)。</text>
<text x="44" y="356" font-size="10.5" fill="#9AA4B2">代价是 C 跳串行，延迟高于一次 AllGather Q，因此更适合 KV 常驻缓存的多轮 decode 与 incremental prefill。</text>

<rect x="24" y="382" width="632" height="60" rx="8" fill="#0E131D" stroke="#232C3A"/>
<text x="44" y="402" font-size="10.5" fill="#8B96A8">同一件事的三种写法</text>
<text x="44" y="422" font-size="9.5" fill="#5B6678">AllGather Q：1 次广播 + 1 次归并，延迟最低；Pass Q：C 跳串行但省归并，缓存友好。</text>
<text x="44" y="436" font-size="9.5" fill="#5B6678">Meta 的 pass-Q 变体即为此形态，与 3.1 的 pass-KV 按 miss rate 动态选择。</text>
</g>
</svg>

▲ Pass Q 示意图：KV 分片常驻各 rank（左），Q 携带 partial 累加器沿环逐 hop 传递并就地在线合并（右，四轮日程全画），通信前 acc 为空，C 跳后回到源 rank 即为完整输出

## 3.4 Ulysses：All-to-All 换维

DeepSpeed-Ulysses 在注意力前后各做一次 All-to-All，把按序列切分的数据转置为按 head 切分。通信之后，每个 rank 持有全序列上的部分 head，可以直接调用普通 FlashAttention，之后再经一次 All-to-All 转回序列切分。该方案不需要修改 kernel，也不需要 LSE 合并；通信量正比于 $T$，但不随 CP 度增长（序列与设备等比例扩展时通信量保持恒定）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 402" style="width:100%;height:auto">
<defs>
<linearGradient id="bg5" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot5" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#9A72AC"/></marker>
</defs>
<rect width="680" height="402" fill="url(#bg5)"/>
<rect width="680" height="402" fill="url(#dot5)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">Ulysses：两次 All-to-All，把序列维换成 head 维</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">矩阵不动，改变的是它被切的方向。注意力夹在两次换维之间的那一段里</text>

<text x="120" y="92" text-anchor="middle" font-size="10.5" fill="#E6E8EC">① 注意力前：按序列（行）切</text>
<text x="328" y="92" text-anchor="middle" font-size="10.5" fill="#E6E8EC">② a2a 之后：按 head（列）切</text>
<text x="536" y="92" text-anchor="middle" font-size="10.5" fill="#E6E8EC">③ 注意力后：换回序列切</text>

<g stroke="#0B0D12" stroke-width="1">
<rect x="64" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="92" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="120" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="148" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="64" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="92" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="120" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="148" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="64" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="92" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="120" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="148" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="64" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="92" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="120" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="148" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
</g>
<text x="106" y="258" text-anchor="middle" font-size="9" fill="#5B6678">每 rank 持 T/4 行 × 全部 H</text>

<g stroke="#0B0D12" stroke-width="1">
<rect x="272" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="300" y="112" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="328" y="112" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="356" y="112" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="272" y="140" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="300" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="328" y="140" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="356" y="140" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="272" y="168" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="300" y="168" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="328" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="356" y="168" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="272" y="196" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="300" y="196" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="328" y="196" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="356" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
</g>
<text x="314" y="258" text-anchor="middle" font-size="9" fill="#5B6678">每 rank 持全部 T × H/4</text>

<g stroke="#0B0D12" stroke-width="1">
<rect x="480" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="508" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="536" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="564" y="112" width="28" height="28" fill="#58C4DD" fill-opacity="0.38"/>
<rect x="480" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="508" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="536" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="564" y="140" width="28" height="28" fill="#5CD0B3" fill-opacity="0.38"/>
<rect x="480" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="508" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="536" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="564" y="168" width="28" height="28" fill="#F4D345" fill-opacity="0.38"/>
<rect x="480" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="508" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="536" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
<rect x="564" y="196" width="28" height="28" fill="#FC6255" fill-opacity="0.38"/>
</g>
<text x="522" y="258" text-anchor="middle" font-size="9" fill="#5B6678">输出按行块归位，供 FFN 继续</text>

<g stroke="#9A72AC" stroke-width="1.6" fill="none">
<line x1="182" y1="168" x2="264" y2="168" marker-end="url(#a5)"/>
<line x1="390" y1="168" x2="472" y2="168" marker-end="url(#a5)"/>
</g>
<text x="223" y="158" text-anchor="middle" font-size="9" fill="#B69BC7">All-to-All</text>
<text x="223" y="186" text-anchor="middle" font-size="8.5" fill="#7B5E8C">行 → 列</text>
<text x="431" y="158" text-anchor="middle" font-size="9" fill="#B69BC7">All-to-All</text>
<text x="431" y="186" text-anchor="middle" font-size="8.5" fill="#7B5E8C">列 → 行</text>

<g font-size="9" fill="#5B6678">



</g>

<rect x="24" y="278" width="632" height="46" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="297" font-size="10" fill="#9AA4B2">中间那一段（②）每个 rank 拿到「全序列 × 本地 head」，可以直接调用普通 FlashAttention，不需要改 kernel、也不需要 LSE 合并。</text>
<text x="44" y="314" font-size="10" fill="#9AA4B2">代价是两次换维都要把整个激活搬一遍，通信量 ∝ T·H；强扩展下（序列与设备同比）每卡通信量恒定。</text>

<rect x="24" y="334" width="632" height="52" rx="8" fill="#0E131D" stroke="#232C3A"/>
<text x="44" y="353" font-size="10" fill="#FC6255">硬约束：a2a 按 head 切，CP 度不能超过 head 数。</text>
<text x="44" y="371" font-size="10" fill="#8B96A8">GQA 只有 2~8 个 KV head、MLA 只有 1 个，Ulysses 很快触顶，这也是它在现代模型上让位于 Ring / AllGather KV 的原因。</text>
</g>
</svg>

▲ Ulysses 示意图：矩阵从按行（序列）切分经 All-to-All 变为按列（head）切分。注意其集合通信原语是 All-to-All，不是 AllGather

这一方案的硬约束是 CP 度不能超过 head 数。GQA、MQA、MLA 的 KV head 极少，该约束在现代模型上非常紧；MLA 的吸收形式下甚至不存在 per-head KV 可供交换，有效 Ulysses 并行度退化为 1。此外需要指出，Ulysses 的集合通信原语是 All-to-All 而非 AllGather，两者在文献与讨论中常被混称。

## 3.5 混合方案：USP / a2a+p2p、pass-KV / pass-Q 动态切换

Ulysses 与 Ring 并不互斥。USP（Unified Sequence Parallelism）在二维网格上组合两者，节点内（NVLink）使用 Ulysses All-to-All，节点间（IB/RoCE）使用 Ring P2P，总 CP 度由此可以超过 head 数。Megatron-Core 的 `cp_comm_type=a2a+p2p` 配合 `hierarchical_context_parallel_sizes`（内层 a2a 尺寸、外层 p2p 尺寸）即对应这一思路。USP 论文还给出了 4D 并行的推荐嵌套顺序（TP < SP-Ulysses < SP-Ring < ZeRO-DP < PP）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 466" style="width:100%;height:auto">
<defs>
<linearGradient id="bg6" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot6" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a6v" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#9A72AC"/></marker>
<marker id="a6o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FF862F"/></marker>
</defs>
<rect width="680" height="466" fill="url(#bg6)"/>
<rect width="680" height="466" fill="url(#dot6)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">USP：节点内 a2a，节点间 Ring，总 CP 度可超过 head 数</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">两种混合在二维网格上各管一个方向，节点内的换维走 NVLink，跨节点只传 KV</text>

<rect x="24" y="84" width="632" height="124" rx="10" fill="#0E131D" stroke="#9A72AC" stroke-width="1.2" stroke-opacity="0.5"/>
<circle cx="44" cy="104" r="9" fill="#F4D345"/>
<text x="44" y="108" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="60" y="108" font-size="11.5" fill="#E6E8EC">阶段一 · 节点内列向 All-to-All（NVLink）</text>
<text x="640" y="108" text-anchor="end" font-size="9.5" fill="#9A72AC">序列维 → head 维</text>
<g>
<rect x="40" y="124" width="130" height="46" rx="6" fill="#58C4DD" fill-opacity="0.12" stroke="#58C4DD" stroke-width="1"/>
<text x="105" y="144" text-anchor="middle" font-size="9.5" fill="#CFEAF2">GPU 0</text>
<text x="105" y="159" text-anchor="middle" font-size="8.5" fill="#7FA9B8">行块 s0 → 列块 h0</text>
<rect x="190" y="124" width="130" height="46" rx="6" fill="#5CD0B3" fill-opacity="0.12" stroke="#5CD0B3" stroke-width="1"/>
<text x="255" y="144" text-anchor="middle" font-size="9.5" fill="#C3E8DC">GPU 1</text>
<text x="255" y="159" text-anchor="middle" font-size="8.5" fill="#79AC9A">行块 s1 → 列块 h1</text>
<rect x="340" y="124" width="130" height="46" rx="6" fill="#F4D345" fill-opacity="0.12" stroke="#F4D345" stroke-width="1"/>
<text x="405" y="144" text-anchor="middle" font-size="9.5" fill="#F3E3AE">GPU 2</text>
<text x="405" y="159" text-anchor="middle" font-size="8.5" fill="#B3A574">行块 s2 → 列块 h2</text>
<rect x="490" y="124" width="130" height="46" rx="6" fill="#FC6255" fill-opacity="0.12" stroke="#FC6255" stroke-width="1"/>
<text x="555" y="144" text-anchor="middle" font-size="9.5" fill="#F6C0BA">GPU 3</text>
<text x="555" y="159" text-anchor="middle" font-size="8.5" fill="#B37C77">行块 s3 → 列块 h3</text>
</g>
<g stroke="#9A72AC" stroke-width="1.3" fill="none">
<line x1="172" y1="140" x2="188" y2="140" marker-end="url(#a6v)"/>
<line x1="188" y1="154" x2="172" y2="154" marker-end="url(#a6v)"/>
<line x1="322" y1="140" x2="338" y2="140" marker-end="url(#a6v)"/>
<line x1="338" y1="154" x2="322" y2="154" marker-end="url(#a6v)"/>
<line x1="472" y1="140" x2="488" y2="140" marker-end="url(#a6v)"/>
<line x1="488" y1="154" x2="472" y2="154" marker-end="url(#a6v)"/>
</g>
<text x="340" y="192" text-anchor="middle" font-size="9" fill="#5B6678">同一节点内的 4 张卡互相交换，每个 rank 此后持有「本节点全部序列 × 1/4 head」</text>

<rect x="24" y="220" width="300" height="120" rx="10" fill="#0E131D" stroke="#FF862F" stroke-width="1.1" stroke-opacity="0.45"/>
<circle cx="44" cy="240" r="9" fill="#F4D345"/>
<text x="44" y="244" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="60" y="244" font-size="11.5" fill="#E6E8EC">阶段二 · 节点 0</text>
<g fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1">
<rect x="56" y="262" width="46" height="34" rx="4"/><rect x="112" y="262" width="46" height="34" rx="4"/>
<rect x="168" y="262" width="46" height="34" rx="4"/><rect x="224" y="262" width="46" height="34" rx="4"/>
</g>
<g font-size="8.5" text-anchor="middle" fill="#F0C39B">
<text x="79" y="283">KV a</text><text x="135" y="283">KV b</text><text x="191" y="283">KV c</text><text x="247" y="283">KV d</text>
</g>
<text x="174" y="318" text-anchor="middle" font-size="9" fill="#5B6678">持序列前半段的全部 KV</text>

<rect x="356" y="220" width="300" height="120" rx="10" fill="#0E131D" stroke="#FF862F" stroke-width="1.1" stroke-opacity="0.45"/>
<text x="372" y="244" font-size="11.5" fill="#E6E8EC">节点 1</text>
<g fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1">
<rect x="388" y="262" width="46" height="34" rx="4"/><rect x="444" y="262" width="46" height="34" rx="4"/>
<rect x="500" y="262" width="46" height="34" rx="4"/><rect x="556" y="262" width="46" height="34" rx="4"/>
</g>
<g font-size="8.5" text-anchor="middle" fill="#F0C39B">
<text x="411" y="283">KV e</text><text x="467" y="283">KV f</text><text x="523" y="283">KV g</text><text x="579" y="283">KV h</text>
</g>
<text x="506" y="318" text-anchor="middle" font-size="9" fill="#5B6678">持序列后半段的全部 KV</text>

<g stroke="#FF862F" stroke-width="1.5" fill="none">
<line x1="327" y1="270" x2="352" y2="270" marker-end="url(#a6o)"/>
<line x1="353" y1="292" x2="328" y2="292" marker-end="url(#a6o)"/>
</g>
<text x="340" y="356" text-anchor="middle" font-size="9" fill="#8B96A8">节点间 Ring P2P（IB / RoCE）：行向传 KV，与本地计算重叠</text>

<text x="24" y="384" font-size="10.5" fill="#E6E8EC">tile 归属：p2p 度 × a2a 度</text>
<g stroke="#0B0D12" stroke-width="1">
<rect x="24" y="394" width="40" height="24" fill="#58C4DD" fill-opacity="0.18"/><rect x="64" y="394" width="40" height="24" fill="#5CD0B3" fill-opacity="0.18"/>
<rect x="104" y="394" width="40" height="24" fill="#F4D345" fill-opacity="0.18"/><rect x="144" y="394" width="40" height="24" fill="#FC6255" fill-opacity="0.18"/>
<rect x="24" y="418" width="40" height="24" fill="#58C4DD" fill-opacity="0.34"/><rect x="64" y="418" width="40" height="24" fill="#5CD0B3" fill-opacity="0.34"/>
<rect x="104" y="418" width="40" height="24" fill="#F4D345" fill-opacity="0.34"/><rect x="144" y="418" width="40" height="24" fill="#FC6255" fill-opacity="0.34"/>
</g>
<text x="190" y="412" font-size="9" fill="#5B6678">行 = 节点（p2p 维，2 个）</text>
<text x="190" y="428" font-size="9" fill="#5B6678">列 = head 组（a2a 维，4 个）</text>
<text x="196" y="446" font-size="9.5" fill="#83C167">总 CP 度 = 4 × 2 = 8</text>
<text x="352" y="404" font-size="9.5" fill="#8B96A8">Megatron：cp_comm_type=a2a+p2p</text>
<text x="352" y="420" font-size="9.5" fill="#8B96A8">hierarchical_context_parallel_sizes = [4, 2]</text>
<text x="352" y="440" font-size="9.5" fill="#5B6678">嵌套顺序建议 TP &lt; Ulysses &lt; Ring &lt; ZeRO-DP &lt; PP</text>
</g>
</svg>

▲ USP 混合示意图：[T, H] 矩阵二维切块，列向走 Ulysses a2a（节点内）、行向走 Ring P2P（节点间）

**pass-KV / pass-Q 动态切换（Meta CP，MLSys 2025）**

USP 把 Ulysses 与 Ring 在拓扑层面组合，Meta 的《Context Parallelism for Scalable Million-Token Inference》（arXiv:2411.01783，MLSys 2025）则在算法层面把 3.1 节的传 KV 与 3.3 节的传 Q 实现为两个无损精确的 ring attention 变体 pass-KV 与 pass-Q，并在运行时按负载动态切换。这篇论文首次公开了 CP 推理系统的实现细节。在 16 节点 128 张 H100 上，Llama3-405B 的 1M 上下文 prefill 仅需 77 秒（并行效率 93%、FLOPS 利用率 63%），128K prefill 为 3.8 秒；在 RDMA 与中低带宽 TCP（约 3 GB/s per rank）下均接近线性扩展。由于跨节点通信量比 TP 小几个数量级，8 节点 128K 的场景下 TP16 已比 CP8 慢约一倍。

pass-KV 对应 3.1 节的 Ring。Q 行块留在本地，KV 沿环以 P2P 传递 $N-1$ 轮，每轮 SendRecv 与 attention 计算重叠，最后用 LSE（论文称 merge attention）合并。其通信量正比于 $T\cdot N_{kv}\cdot d$，在 GQA 下比传 Q 小约 $N_h/N_{kv}$ 倍（405B 约为 16 倍），适合 full prefill 与 miss rate 较高（新 token 占比大）的 partial prefill。decode 阶段不采用这一变体，因为 $T=1$ 时传 KV 几乎总是更贵。

pass-Q 对应 3.3 节的传 Q。KV 分片跨轮常驻（persistent KV cache，多轮会话之间保持不动），Q 沿环传递，各 rank 对本地 KV 分片计算 partial $(O_j, L_j)$，再经 All2All 加 permute 把散落在各 rank 的部分结果归位后做 LSE 合并。其通信量只与新 token 数 $T$ 成正比，与 KV cache 长度无关，适合 decode（$T=1$）与 miss rate 较低（$P \gg T$）的 partial prefill。

两种变体之间的切换由启发式规则决定。记 miss rate 为 $T/(T+P)$（$T$ 为新 token 数，$P$ 为已缓存 KV 长度），则传 Q 的消息更小当且仅当 $\frac{T}{T+P}\le 2\frac{N_{kv}}{N_h}$（405B 的阈值为 12.5%）。实际决策还要叠加 roofline 条件（pass-KV 的 SendRecv 能被 attention 计算遮蔽时优先 pass-KV）、计入 All2All 的精化式，以及经验拟合 $h(T,P)=-1.059\log T+1.145\log\frac{T}{T+P}+12.112$（$h>0$ 时选 pass-KV）。128K、CP4 的实测切换点约在 miss rate 5% 附近。

多轮会话下，KV cache 按 CP rank 分布存储并跨轮保持。序列切成 $2N$ 块、rank $i$ 取 $(C_i, C_{2N-1-i})$ 首尾配对（与 zigzag 相同），同时均衡注意力计算量与 KV 显存；decode 采用 round-robin 每步偏移一个 index，避免某个 rank 既承担 decode 又持有最长 KV 而溢出显存；ring 消息按 $\max_i P_i+\lceil T/N\rceil$ 等长 padding，以满足集合通信接口的要求。

最后是该方案的适用边界。CP 显著降低 TTFT，但 TTIT 略有退化（padding 与 SendRecv/All2All 的开销），论文建议与 PD 分离（DistServe、Mooncake）配合部署。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 486" style="width:100%;height:auto">
<defs>
<linearGradient id="bg7" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot7" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a7o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FF862F"/></marker>
<marker id="a7b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#58C4DD"/></marker>
</defs>
<rect width="680" height="486" fill="url(#bg7)"/>
<rect width="680" height="486" fill="url(#dot7)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">同一套 Ring，两种形状：按 miss rate 切换</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">pass-KV 传历史，pass-Q 传新 token。多轮会话里两者交替出现，判据只有一条</text>

<rect x="24" y="72" width="632" height="164" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="40" y="94" font-size="11.5" fill="#E6E8EC">判据：本轮的 KV 命中率</text>
<text x="40" y="112" font-size="9.5" fill="#5B6678">命中多则只需算新 token，miss 多则历史 KV 也要过一遍</text>
<line x1="56" y1="146" x2="624" y2="146" stroke="#3A4557" stroke-width="1.2"/>
<rect x="56" y="122" width="244" height="24" fill="#58C4DD" fill-opacity="0.16" stroke="#58C4DD" stroke-width="1"/>
<rect x="300" y="122" width="324" height="24" fill="#FF862F" fill-opacity="0.16" stroke="#FF862F" stroke-width="1"/>
<text x="178" y="139" text-anchor="middle" font-size="10.5" fill="#8FC9DC">pass-Q  命中多</text>
<text x="462" y="139" text-anchor="middle" font-size="10.5" fill="#F0C39B">pass-KV  miss 多</text>
<line x1="300" y1="112" x2="300" y2="176" stroke="#F4D345" stroke-width="1.2" stroke-dasharray="4 4"/>
<text x="306" y="112" font-size="9.5" fill="#F4D345">阈值 2·N_kv/N_h，405B 约 12.5%</text>
<text x="56" y="190" font-size="9" fill="#5B6678">0</text>
<text x="624" y="190" text-anchor="end" font-size="9" fill="#5B6678">1（整段都是新的）</text>
<text x="40" y="214" font-size="10" fill="#8B96A8">判据来自通信量之比：Q 侧只有 T·d，KV 侧是 P·d，当 T/(T+P) ≤ 2·N_kv/N_h 时传 Q 更省。</text>
<text x="40" y="228" font-size="9.5" fill="#5B6678">横轴为示意刻度，不是线性。</text>

<rect x="24" y="252" width="304" height="196" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="44" cy="272" r="9" fill="#F4D345"/>
<text x="44" y="276" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="60" y="276" font-size="11" fill="#E6E8EC">pass-KV：KV 每轮环传（C = 3 示意）</text>
<rect x="40" y="288" width="80" height="112" rx="6" fill="#0F1420" stroke="#2C3748"/>
<text x="80" y="303" text-anchor="middle" font-size="8.5" fill="#8B96A8">第 0 轮</text>
<rect x="48" y="310" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="80" y="323" text-anchor="middle" font-size="8" fill="#F0C39B">R0 · K0</text>
<rect x="48" y="332" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="80" y="345" text-anchor="middle" font-size="8" fill="#F0C39B">R1 · K1</text>
<rect x="48" y="354" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="80" y="367" text-anchor="middle" font-size="8" fill="#F0C39B">R2 · K2</text>
<text x="80" y="390" text-anchor="middle" font-size="8" fill="#5B6678">算 i−j = 0</text>

<rect x="136" y="288" width="80" height="112" rx="6" fill="#0F1420" stroke="#2C3748"/>
<text x="176" y="303" text-anchor="middle" font-size="8.5" fill="#8B96A8">第 1 轮</text>
<rect x="144" y="310" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="176" y="323" text-anchor="middle" font-size="8" fill="#F0C39B">R0 · K2</text>
<rect x="144" y="332" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="176" y="345" text-anchor="middle" font-size="8" fill="#F0C39B">R1 · K0</text>
<rect x="144" y="354" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="176" y="367" text-anchor="middle" font-size="8" fill="#F0C39B">R2 · K1</text>
<text x="176" y="390" text-anchor="middle" font-size="8" fill="#5B6678">算 i−j = 1</text>

<rect x="232" y="288" width="80" height="112" rx="6" fill="#0F1420" stroke="#2C3748"/>
<text x="272" y="303" text-anchor="middle" font-size="8.5" fill="#8B96A8">第 2 轮</text>
<rect x="240" y="310" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="272" y="323" text-anchor="middle" font-size="8" fill="#F0C39B">R0 · K1</text>
<rect x="240" y="332" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="272" y="345" text-anchor="middle" font-size="8" fill="#F0C39B">R1 · K2</text>
<rect x="240" y="354" width="64" height="18" rx="3" fill="#FF862F" fill-opacity="0.28" stroke="#FF862F" stroke-width="0.9"/><text x="272" y="367" text-anchor="middle" font-size="8" fill="#F0C39B">R2 · K0</text>
<text x="272" y="390" text-anchor="middle" font-size="8" fill="#5B6678">算 i−j = 2</text>

<g stroke="#FF862F" stroke-width="1.4" fill="none">
<line x1="122" y1="340" x2="134" y2="340" marker-end="url(#a7o)"/>
<line x1="218" y1="340" x2="230" y2="340" marker-end="url(#a7o)"/>
</g>
<text x="40" y="424" font-size="9.5" fill="#5B6678">KV 沿环转一圈，通信量 O(T)；适合首次 prefill 与 miss 高的多轮会话。</text>

<rect x="348" y="252" width="308" height="196" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="368" cy="272" r="9" fill="#F4D345"/>
<text x="368" y="276" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="384" y="276" font-size="11" fill="#E6E8EC">pass-Q：Q 与 acc 逐跳下传</text>
<rect x="364" y="288" width="80" height="112" rx="6" fill="#0F1420" stroke="#2C3748"/>
<text x="404" y="303" text-anchor="middle" font-size="8.5" fill="#8B96A8">跳 0</text>
<rect x="372" y="312" width="64" height="20" rx="3" fill="#58C4DD" fill-opacity="0.25" stroke="#58C4DD" stroke-width="0.9"/><text x="404" y="326" text-anchor="middle" font-size="8" fill="#BFE4EF">Q → R0</text>
<text x="404" y="348" text-anchor="middle" font-size="8" fill="#5B6678">acc = S0 段</text>
<text x="404" y="390" text-anchor="middle" font-size="8" fill="#5B6678">KV0 常驻</text>

<rect x="460" y="288" width="80" height="112" rx="6" fill="#0F1420" stroke="#2C3748"/>
<text x="500" y="303" text-anchor="middle" font-size="8.5" fill="#8B96A8">跳 1</text>
<rect x="468" y="312" width="64" height="20" rx="3" fill="#58C4DD" fill-opacity="0.25" stroke="#58C4DD" stroke-width="0.9"/><text x="500" y="326" text-anchor="middle" font-size="8" fill="#BFE4EF">Q + acc → R1</text>
<text x="500" y="348" text-anchor="middle" font-size="8" fill="#5B6678">acc 两段</text>
<text x="500" y="390" text-anchor="middle" font-size="8" fill="#5B6678">KV1 常驻</text>

<rect x="556" y="288" width="84" height="112" rx="6" fill="#0F1420" stroke="#2C3748"/>
<text x="598" y="303" text-anchor="middle" font-size="8.5" fill="#8B96A8">跳 2</text>
<rect x="564" y="312" width="68" height="20" rx="3" fill="#83C167" fill-opacity="0.25" stroke="#83C167" stroke-width="0.9"/><text x="598" y="326" text-anchor="middle" font-size="8" fill="#C6E3B6">→ 回 R0</text>
<text x="598" y="348" text-anchor="middle" font-size="8" fill="#7FA36F">acc 即 O</text>
<text x="598" y="390" text-anchor="middle" font-size="8" fill="#5B6678">KV2 常驻</text>

<g stroke="#58C4DD" stroke-width="1.4" fill="none">
<line x1="446" y1="340" x2="458" y2="340" marker-end="url(#a7b)"/>
<line x1="542" y1="340" x2="554" y2="340" marker-end="url(#a7b)"/>
</g>
<text x="364" y="424" font-size="9.5" fill="#5B6678">通信量只 ∝ 新 token 数 T；适合 decode 与增量 prefill，省掉归并。</text>

<rect x="24" y="458" width="632" height="20" rx="6" fill="#121826" stroke="#232C3A"/>
<text x="340" y="472" text-anchor="middle" font-size="10" fill="#9AA4B2">两者无损，结果与单卡稠密注意力逐位等价；区别只在把哪一侧搬上环</text>
</g>
</svg>

▲ pass-KV / pass-Q 动态切换示意图：上为多轮会话的 miss rate 与切换阈值；下左为 pass-KV 的三轮环传日程（C = 3 示意），下右为 pass-Q 的三跳下传，KV 始终常驻

## 3.6 公共组件：LSE 合并（在线 softmax）

无论传 KV 分多轮计算，还是传 Q 分片计算，最终都要把各部分注意力的中间结果合并为精确结果。设分片 $j$ 给出归一化输出 $O_j$ 与对数和 $L_j$，则合并公式为

$$m = \max_j L_j,\quad w_j = \exp(L_j - m),\quad O = \frac{\sum_j w_j O_j}{\sum_j w_j}$$

这正是 FlashAttention 在线 softmax 的推广，也是 decode CP 通信量可以极小的原因。每一步只需传递输出与 LSE，无须传递原始 KV。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 420" style="width:100%;height:auto">
<defs>
<linearGradient id="bg8" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot8" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a8g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#83C167"/></marker>
</defs>
<rect width="680" height="420" fill="url(#bg8)"/>
<rect width="680" height="420" fill="url(#dot8)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">LSE 合并：把各段的 partial 结果拼成精确值</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">这是 FlashAttention 的在线 softmax。下面用一组具体数值走一遍（4 个 KV 分片）</text>

<circle cx="30" cy="82" r="9" fill="#F4D345"/>
<text x="30" y="86" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="46" y="86" font-size="10.5" fill="#8B96A8">各段独立算出局部输出与对数和</text>

<g>
<rect x="26" y="96" width="146" height="78" rx="8" fill="#0E131D" stroke="#3A4557"/>
<text x="36" y="112" font-size="8.5" fill="#5B6678">列段 0</text>
<text x="162" y="112" text-anchor="end" font-size="10" fill="#F4D345">L₀ = 2.1</text>
<g><rect x="36" y="120" width="26" height="34" fill="#58C4DD" fill-opacity="0.3"/><rect x="62" y="120" width="26" height="34" fill="#58C4DD" fill-opacity="0.45"/><rect x="88" y="120" width="26" height="34" fill="#58C4DD" fill-opacity="0.6"/></g>
<text x="126" y="140" font-size="9" fill="#8B96A8">O₀</text>
<text x="36" y="166" font-size="8.5" fill="#5B6678">softmax 只在段内归一化</text>
</g>
<g>
<rect x="186" y="96" width="146" height="78" rx="8" fill="#0E131D" stroke="#3A4557"/>
<text x="196" y="112" font-size="8.5" fill="#5B6678">列段 1</text>
<text x="322" y="112" text-anchor="end" font-size="10" fill="#F4D345">L₁ = 1.4</text>
<g><rect x="196" y="120" width="26" height="34" fill="#5CD0B3" fill-opacity="0.3"/><rect x="222" y="120" width="26" height="34" fill="#5CD0B3" fill-opacity="0.45"/><rect x="248" y="120" width="26" height="34" fill="#5CD0B3" fill-opacity="0.6"/></g>
<text x="286" y="140" font-size="9" fill="#8B96A8">O₁</text>
<text x="196" y="166" font-size="8.5" fill="#5B6678">段内和 = 1（各自归一）</text>
</g>
<g>
<rect x="346" y="96" width="146" height="78" rx="8" fill="#0E131D" stroke="#3A4557"/>
<text x="356" y="112" font-size="8.5" fill="#5B6678">列段 2</text>
<text x="482" y="112" text-anchor="end" font-size="10" fill="#F4D345">L₂ = 2.6</text>
<g><rect x="356" y="120" width="26" height="34" fill="#F4D345" fill-opacity="0.3"/><rect x="382" y="120" width="26" height="34" fill="#F4D345" fill-opacity="0.45"/><rect x="408" y="120" width="26" height="34" fill="#F4D345" fill-opacity="0.6"/></g>
<text x="446" y="140" font-size="9" fill="#8B96A8">O₂</text>
<text x="356" y="166" font-size="8.5" fill="#5B6678">L₂ 最大，后面以它为基准</text>
</g>
<g>
<rect x="506" y="96" width="146" height="78" rx="8" fill="#0E131D" stroke="#3A4557"/>
<text x="516" y="112" font-size="8.5" fill="#5B6678">列段 3</text>
<text x="642" y="112" text-anchor="end" font-size="10" fill="#F4D345">L₃ = 0.9</text>
<g><rect x="516" y="120" width="26" height="34" fill="#FC6255" fill-opacity="0.3"/><rect x="542" y="120" width="26" height="34" fill="#FC6255" fill-opacity="0.45"/><rect x="568" y="120" width="26" height="34" fill="#FC6255" fill-opacity="0.6"/></g>
<text x="606" y="140" font-size="9" fill="#8B96A8">O₃</text>
<text x="516" y="166" font-size="8.5" fill="#5B6678">L₃ 最小，权重也最小</text>
</g>

<circle cx="30" cy="196" r="9" fill="#F4D345"/>
<text x="30" y="200" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="46" y="200" font-size="10.5" fill="#8B96A8">取全局最大值 m = max L_j = 2.6，把各段重标定到同一基准</text>

<g>
<rect x="26" y="212" width="146" height="44" rx="6" fill="#161C29" stroke="#3A4557"/>
<text x="99" y="230" text-anchor="middle" font-size="9" fill="#8B96A8">e^(2.1 − 2.6)</text>
<text x="99" y="248" text-anchor="middle" font-size="12" fill="#E6E8EC">w₀ = 0.61</text>
<rect x="186" y="212" width="146" height="44" rx="6" fill="#161C29" stroke="#3A4557"/>
<text x="259" y="230" text-anchor="middle" font-size="9" fill="#8B96A8">e^(1.4 − 2.6)</text>
<text x="259" y="248" text-anchor="middle" font-size="12" fill="#E6E8EC">w₁ = 0.30</text>
<rect x="346" y="212" width="146" height="44" rx="6" fill="#161C29" stroke="#3A4557"/>
<text x="419" y="230" text-anchor="middle" font-size="9" fill="#8B96A8">e^(2.6 − 2.6)</text>
<text x="419" y="248" text-anchor="middle" font-size="12" fill="#E6E8EC">w₂ = 1.00</text>
<rect x="506" y="212" width="146" height="44" rx="6" fill="#161C29" stroke="#3A4557"/>
<text x="579" y="230" text-anchor="middle" font-size="9" fill="#8B96A8">e^(0.9 − 2.6)</text>
<text x="579" y="248" text-anchor="middle" font-size="12" fill="#E6E8EC">w₃ = 0.18</text>
</g>

<circle cx="30" cy="292" r="9" fill="#F4D345"/>
<text x="30" y="296" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">3</text>
<text x="46" y="296" font-size="10.5" fill="#8B96A8">按权相加再归一</text>

<g stroke="#83C167" stroke-width="1.2" fill="none">
<path d="M99 258 C 99 276, 210 282, 218 296" marker-end="url(#a8g)"/>
<path d="M259 258 C 259 276, 240 282, 236 294" marker-end="url(#a8g)"/>
<path d="M419 258 C 419 276, 430 282, 434 294" marker-end="url(#a8g)"/>
<path d="M579 258 C 579 276, 460 282, 452 296" marker-end="url(#a8g)"/>
</g>
<rect x="196" y="300" width="288" height="58" rx="8" fill="#83C167" fill-opacity="0.14" stroke="#83C167" stroke-width="1.4"/>
<text x="340" y="322" text-anchor="middle" font-size="12" fill="#DCECCD">O = (w₀O₀ + w₁O₁ + w₂O₂ + w₃O₃) / Σw</text>
<text x="340" y="344" text-anchor="middle" font-size="10" fill="#8FB27C">Σw = 2.09 → 0.29 / 0.14 / 0.48 / 0.09</text>

<rect x="24" y="374" width="632" height="34" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="340" y="396" text-anchor="middle" font-size="10" fill="#9AA4B2">第 j 段的权重只依赖它自己的 L_j，各段可以完全独立求值 —— 这正是 Ring 多轮、传 Q 多分片能并行的前提</text>
</g>
</svg>

▲ LSE 合并示意图：以四个列段的 L_j = 2.1 / 1.4 / 2.6 / 0.9 为例，先重标定权重再归一，得到与单卡一致的输出

## 3.7 选型小结

下表从通信内容、通信量、head 数约束与 kernel 侵入性四个维度对比上述方案，随后的图给出按场景的选择路径。

| 方案 | 通信内容 | 通信量 | 受 head 数限制 | Kernel 侵入 | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| Ring（传 KV） | KV 块 P2P | $O(T)$，可重叠 | 否 | 高（LSE 合并） | 训练长序列、超长 prefill |
| AllGather KV | 全量 KV | $O(T)$，一次性 | 否 | 低 | Prefill CP、KV 小的架构 |
| 传 Q | Q + 输出/LSE | 与 KV 无关 | 否 | 中 | Decode CP |
| Ulysses（a2a） | Q/K/V/O | $O(T)$，恒定 | 是 | 低 | 节点内、head 多 |
| USP（a2a+p2p） | 组合 | 组合 | 部分 | 中 | 多节点混合拓扑 |

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 400" style="width:100%;height:auto">
<defs>
<linearGradient id="bg9" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot9" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a9" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#5B6678"/></marker>
<marker id="a9b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#58C4DD"/></marker>
</defs>
<rect width="680" height="400" fill="url(#bg9)"/>
<rect width="680" height="400" fill="url(#dot9)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">选型的路径：先看阶段，再看三件事</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">KV 有多大、CP 度会不会超过 head 数、跨不跨节点 —— 三个问题决定走哪条路</text>

<rect x="24" y="168" width="108" height="54" rx="8" fill="#161C29" stroke="#3A4557"/>
<text x="78" y="192" text-anchor="middle" font-size="10.5" fill="#E6E8EC">一层注意力</text>
<text x="78" y="208" text-anchor="middle" font-size="9" fill="#5B6678">要分到多卡</text>

<rect x="160" y="86" width="148" height="56" rx="8" fill="#58C4DD" fill-opacity="0.12" stroke="#58C4DD" stroke-width="1.1"/>
<text x="234" y="109" text-anchor="middle" font-size="11" fill="#CFEAF2">Prefill</text>
<text x="234" y="126" text-anchor="middle" font-size="9" fill="#7FA9B8">T 个新 token，算力受限</text>
<rect x="160" y="238" width="148" height="56" rx="8" fill="#9A72AC" fill-opacity="0.14" stroke="#9A72AC" stroke-width="1.1"/>
<text x="234" y="261" text-anchor="middle" font-size="11" fill="#D8C4E2">Decode</text>
<text x="234" y="278" text-anchor="middle" font-size="9" fill="#9C86AA">每步 1 个 token，访存受限</text>

<g stroke="#5B6678" stroke-width="1.3" fill="none">
<path d="M132 186 C 148 160, 150 130, 158 116" marker-end="url(#a9)"/>
<path d="M132 206 C 148 232, 150 258, 158 268" marker-end="url(#a9)"/>
</g>

<rect x="356" y="66" width="300" height="50" rx="7" fill="#0E131D" stroke="#FF862F" stroke-width="1.1" stroke-opacity="0.6"/>
<text x="370" y="86" font-size="10.5" fill="#F0C39B">AllGather KV</text>
<text x="370" y="104" font-size="9" fill="#5B6678">KV 已被压小（MLA / DSA / CSA）→ 聚齐再算最省事</text>

<rect x="356" y="124" width="300" height="50" rx="7" fill="#0E131D" stroke="#FF862F" stroke-width="1.1" stroke-opacity="0.6"/>
<text x="370" y="144" font-size="10.5" fill="#F0C39B">Ring 传 KV</text>
<text x="370" y="162" font-size="9" fill="#5B6678">KV 大且 CP 度超过 head 数 → 沿环传，通信可重叠</text>

<rect x="356" y="182" width="300" height="50" rx="7" fill="#0E131D" stroke="#9A72AC" stroke-width="1.1" stroke-opacity="0.6"/>
<text x="370" y="202" font-size="10.5" fill="#D8C4E2">Ulysses All-to-All</text>
<text x="370" y="220" font-size="9" fill="#5B6678">head 足够多且都在节点内 → 换维后用普通 kernel</text>

<rect x="356" y="246" width="300" height="58" rx="7" fill="#0E131D" stroke="#58C4DD" stroke-width="1.1" stroke-opacity="0.7"/>
<text x="370" y="266" font-size="10.5" fill="#CFEAF2">传 Q</text>
<text x="370" y="284" font-size="9" fill="#5B6678">AllGather Q 求低延迟；Pass Q 求省归并、缓存友好</text>

<g stroke="#5B6678" stroke-width="1.2" fill="none">
<path d="M310 106 C 332 100, 338 96, 350 91"/>
<path d="M310 120 C 332 124, 338 146, 350 149"/>
<path d="M310 130 C 332 146, 338 194, 350 206"/>
<path d="M310 268 C 330 268, 336 274, 350 275"/>
</g>





<rect x="24" y="316" width="632" height="60" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="336" font-size="10" fill="#9AA4B2">三个判断量：KV 的字节数（决定搬哪一侧更便宜）、CP 度与 head 数的关系（决定 Ulysses 是否可用）、</text>
<text x="44" y="354" font-size="10" fill="#9AA4B2">链路拓扑（节点内 NVLink 可承受 a2a，跨节点只适合 P2P）。三者互斥时按 3.1—3.5 的顺序退化。</text>
<text x="44" y="370" font-size="9.5" fill="#5B6678">GQA 与 MLA 因 KV head 极少，实际多用 AllGather KV 或传 Q，Ulysses 只在训练侧常见。</text>
</g>
</svg>

▲ CP 通信方案选择路径

# 4. 不同Attention的CP

大语言模型的注意力有多种实现，常见的包括 GQA、MLA、DSA、GDN、KDA、CSA 等。不同注意力结构下，CP 需要传递的数据并不相同，本章逐一讨论。

## 4.1 GQA

GQA（Grouped Query Attention）让多个 query head 共享一组 KV head，压缩了 KV Cache，代价是 KV head 数很少（Llama3-8B 为 8 个，Qwen3-235B 为 4 个）。这一结构对 CP 有两方面影响。

一方面，KV head 数直接限制了 Ulysses 的并行度。All-to-All 按 head 切分时 CP 度不能超过 KV head 数，GQA 下很快触顶，这也是 Ulysses 在现代模型上逐渐让位于 Ring、AllGather KV 与传 Q 的原因。另一方面，传 KV 与传 Q 对 GQA 都可行。prefill 常用 AllGather KV（SGLang、vLLM PCP），decode 常用传 Q（AllGather Q、本地 KV 分片与 LSE 合并）。在 vLLM DCP 中，GQA 的 `dcp_size` 上界为 `tp_size / H_kv`；当 DCP 组与 TP 组重叠时，各 rank 的 Q head 不同，需要先沿 head 维 AllGather Q，算完再用 LSE 感知的 all-reduce 或 All-to-All 归并（如 `cp_lse_ag_out_rs`）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 366" style="width:100%;height:auto">
<defs>
<linearGradient id="bg10" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot10" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a10" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FF862F"/></marker>
<marker id="a10b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#58C4DD"/></marker>
</defs>
<rect width="680" height="366" fill="url(#bg10)"/>
<rect width="680" height="366" fill="url(#dot10)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">GQA 的 CP：head 少，于是要么搬 KV，要么搬 Q</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">8 个 Q head 共用 2 个 KV head。KV 小到可以搬，Q 小到可以广播</text>

<rect x="24" y="74" width="206" height="204" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="40" y="94" font-size="10.5" fill="#E6E8EC">结构：Q head 分组共享 KV</text>
<g fill="#58C4DD" fill-opacity="0.35" stroke="#58C4DD" stroke-width="1">
<rect x="36" y="106" width="18" height="20" rx="3"/><rect x="58" y="106" width="18" height="20" rx="3"/>
<rect x="80" y="106" width="18" height="20" rx="3"/><rect x="102" y="106" width="18" height="20" rx="3"/>
</g>
<g fill="#5CD0B3" fill-opacity="0.35" stroke="#5CD0B3" stroke-width="1">
<rect x="126" y="106" width="18" height="20" rx="3"/><rect x="148" y="106" width="18" height="20" rx="3"/>
<rect x="170" y="106" width="18" height="20" rx="3"/><rect x="192" y="106" width="18" height="20" rx="3"/>
</g>
<text x="38" y="142" font-size="8.5" fill="#5B6678">Q head 0–3</text>
<text x="210" y="142" text-anchor="end" font-size="8.5" fill="#5B6678">Q head 4–7</text>
<path d="M36 148 L36 156 L102 156 L102 148" fill="none" stroke="#58C4DD" stroke-width="1" opacity="0.7"/>
<path d="M126 148 L126 156 L192 156 L192 148" fill="none" stroke="#5CD0B3" stroke-width="1" opacity="0.7"/>
<g stroke="#8B96A8" stroke-width="1" fill="none">
<line x1="74" y1="160" x2="74" y2="186"/><line x1="176" y1="160" x2="176" y2="186"/>
</g>
<rect x="58" y="186" width="32" height="26" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1.2"/>
<text x="74" y="203" text-anchor="middle" font-size="8.5" fill="#F0C39B">KV0</text>
<rect x="160" y="186" width="32" height="26" rx="3" fill="#FF862F" fill-opacity="0.3" stroke="#FF862F" stroke-width="1.2"/>
<text x="176" y="203" text-anchor="middle" font-size="8.5" fill="#F0C39B">KV1</text>
<text x="40" y="238" font-size="9" fill="#5B6678">每组 4 个 Q head 共享 1 个 KV head</text>
<text x="40" y="256" font-size="9" fill="#FC6255">Ulysses 的 CP 度 ≤ H_kv = 2</text>

<rect x="246" y="74" width="198" height="204" rx="10" fill="#0E131D" stroke="#FF862F" stroke-width="1.1" stroke-opacity="0.5"/>
<text x="260" y="94" font-size="10.5" fill="#F0C39B">路线一 · 传 KV</text>
<g stroke="#FF862F" stroke-width="0.8">
<rect x="272" y="106" width="20" height="20" fill="#FF862F" fill-opacity="0.18"/><rect x="292" y="106" width="20" height="20" fill="#FF862F" fill-opacity="0.3"/>
<rect x="272" y="126" width="20" height="20" fill="#FF862F" fill-opacity="0.28"/><rect x="292" y="126" width="20" height="20" fill="#FF862F" fill-opacity="0.44"/>
<rect x="272" y="146" width="20" height="20" fill="#FF862F" fill-opacity="0.38"/><rect x="292" y="146" width="20" height="20" fill="#FF862F" fill-opacity="0.56"/>
<rect x="272" y="166" width="20" height="20" fill="#FF862F" fill-opacity="0.48"/><rect x="292" y="166" width="20" height="20" fill="#FF862F" fill-opacity="0.68"/>
</g>
<text x="262" y="202" font-size="8" fill="#5B6678">KV [T, d] × 2</text>
<line x1="330" y1="126" x2="344" y2="126" stroke="#FF862F" stroke-width="1.2" marker-end="url(#a10)"/>
<text x="352" y="120" font-size="9" fill="#8B96A8">每层 AllGather</text>
<text x="352" y="134" font-size="9" fill="#8B96A8">→ 整份 KV</text>
<rect x="352" y="146" width="76" height="46" rx="5" fill="#58C4DD" fill-opacity="0.16" stroke="#58C4DD" stroke-width="1"/>
<text x="390" y="165" text-anchor="middle" font-size="9" fill="#CFEAF2">Q_i × 全 KV</text>
<text x="390" y="181" text-anchor="middle" font-size="9" fill="#CFEAF2">标准 kernel</text>
<text x="260" y="222" font-size="9" fill="#5B6678">通信量 T·d·H_kv；不改 kernel</text>
<text x="260" y="240" font-size="9" fill="#5B6678">CP 度 ≤ tp_size / H_kv</text>
<text x="260" y="262" font-size="9" fill="#83C167">prefill 常用</text>

<rect x="456" y="74" width="200" height="204" rx="10" fill="#0E131D" stroke="#58C4DD" stroke-width="1.1" stroke-opacity="0.6"/>
<text x="470" y="94" font-size="10.5" fill="#CFEAF2">路线二 · 传 Q</text>
<g fill="#58C4DD" fill-opacity="0.3" stroke="#58C4DD" stroke-width="0.8">
<rect x="470" y="106" width="21" height="18"/><rect x="491" y="106" width="21" height="18"/>
<rect x="512" y="106" width="21" height="18"/><rect x="533" y="106" width="21" height="18"/>
<rect x="554" y="106" width="21" height="18"/><rect x="575" y="106" width="21" height="18"/>
<rect x="596" y="106" width="21" height="18"/><rect x="617" y="106" width="21" height="18"/>
</g>
<text x="470" y="140" font-size="8.5" fill="#5B6678">Q [q, 8, d]：decode 时 q = 1</text>
<g stroke="#58C4DD" stroke-width="1.3" fill="none">
<line x1="500" y1="146" x2="500" y2="160" marker-end="url(#a10b)"/>
<line x1="530" y1="146" x2="530" y2="160" marker-end="url(#a10b)"/>
<line x1="560" y1="146" x2="560" y2="160" marker-end="url(#a10b)"/>
<line x1="590" y1="146" x2="590" y2="160" marker-end="url(#a10b)"/>
</g>
<g font-size="8" text-anchor="middle" fill="#5B6678">
<text x="500" y="174">(O₀,L₀)</text><text x="530" y="174">(O₁,L₁)</text><text x="560" y="174">(O₂,L₂)</text><text x="590" y="174">(O₃,L₃)</text>
</g>
<rect x="486" y="184" width="140" height="22" rx="4" fill="#83C167" fill-opacity="0.18" stroke="#83C167" stroke-width="1"/>
<text x="556" y="199" text-anchor="middle" font-size="9" fill="#C6E3B6">LSE 合并 → O</text>
<text x="470" y="222" font-size="9" fill="#5B6678">通信量 ∝ q·H_q·d，与 T 无关</text>
<text x="470" y="240" font-size="9" fill="#5B6678">可与 TP 组重叠，先沿 head gather Q</text>
<text x="470" y="262" font-size="9" fill="#83C167">decode 常用（vLLM DCP）</text>

<rect x="24" y="292" width="632" height="56" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="311" font-size="10" fill="#9AA4B2">DCP 直接消除 KV 复制：DeepSeek-R1 只有 1 个 KV head，tp=8 时 8 份副本，DCP=8 后每卡只留 1/8 的 T。</text>
<text x="44" y="329" font-size="10" fill="#9AA4B2">GQA 的 dcp_size 上界为 tp_size / H_kv；DCP 与 TP 组重叠时各 rank 的 Q head 不同，需先沿 head 维 AllGather Q。</text>
</g>
</svg>

▲ GQA 的 CP：结构（左）与两条通信路径——传 KV 矩阵（中）或传 Q 矩阵（右）

## 4.2 MLA

MLA（Multi-head Latent Attention，DeepSeek-V2/V3）把每个 token 的 K/V 压缩为一个共享的低秩 latent $c_{KV}$（$d_{kv}=512$）外加解耦的 RoPE key $k_{pe}$，比 per-head KV 小约 32 倍。因此 MLA 在 CP 下传递的是 latent 而非 per-head KV。

具体地，AllGather latent（或沿环传递 latent）即可让每个 rank 对全局历史做注意力。配合吸收（absorb）形式，即把 $W_{kv,b}$ 吸收进 Q、直接在 latent 上计算注意力，是推理 decode 的标准做法，也是 vLLM DCP 支持 MLA 的路径；vLLM 还提供 `--dcp-q-replicate` 选项，在 DCP 组内冗余复制 MLA 的 Q 投影，省掉每一步的 query all-gather。如 3.4 节所述，Ulysses 对 MLA 失效，因为吸收形式下不存在可供 All-to-All 交换的 per-head KV，有效并行度退化为 1，MLA 的 CP 因而基本只有 AllGather latent 与环传 latent 两条路。

训练侧另有一个隐性问题。Megatron-Core 明确禁止在训练中使用 absorbed 形式（代码中以 assert 硬性禁止），因为 absorb 的中间激活位于 $n_h \times d_{kv}$ 维，比 per-head K/V 更大，激活显存多出 20~34%（DeepSeek-V3 规模下最多 9.2GB）。LAGA（Latent All-Gather Attention，arXiv:2607.17644）给出了训练侧方案，其通信仍只传 latent，但在本地重建 per-head K/V 后做显式注意力，通信量降低 1.98 倍，显存与显式形式持平，跨节点长序列吞吐提升 1.07~1.24 倍。

MLA 也是 DCP 最典型的应用场景。DeepSeek-R1 推理时只有 1 个 KV head，tp=8 时存在 8 倍复制，DCP=8 可以将其完全消除。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 366" style="width:100%;height:auto">
<defs>
<linearGradient id="bg11" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot11" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a11p" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#9A72AC"/></marker>
<marker id="a11g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#83C167"/></marker>
</defs>
<rect width="680" height="366" fill="url(#bg11)"/>
<rect width="680" height="366" fill="url(#dot11)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">MLA 的 CP：只搬一份 latent</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">KV 被压成 512 维的 latent，per-head 的写法不再成立，Ulysses 也随之失效</text>

<rect x="24" y="74" width="300" height="204" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="40" y="94" font-size="10.5" fill="#E6E8EC">同样 T 行，两种存法差一个数量级</text>
<g fill="#9A72AC" fill-opacity="0.28" stroke="#9A72AC" stroke-width="0.8">
<rect x="44" y="108" width="12" height="96"/><rect x="58" y="108" width="12" height="96"/><rect x="72" y="108" width="12" height="96"/>
<rect x="86" y="108" width="12" height="96"/><rect x="100" y="108" width="12" height="96"/><rect x="114" y="108" width="12" height="96"/>
<rect x="128" y="108" width="12" height="96"/><rect x="142" y="108" width="12" height="96"/>
</g>
<text x="44" y="220" font-size="8.5" fill="#5B6678">per-head KV：128 列 ×（d_c + d_h）</text>
<text x="44" y="236" font-size="8.5" fill="#5B6678">每层每 token 约 128 × 576</text>
<g fill="#9A72AC" fill-opacity="0.5" stroke="#9A72AC" stroke-width="0.9">
<rect x="196" y="108" width="42" height="96"/>
<rect x="244" y="108" width="12" height="96" fill-opacity="0.3"/>
</g>
<text x="196" y="220" font-size="8.5" fill="#5B6678">latent 512 + k_pe 64</text>
<text x="196" y="236" font-size="8.5" fill="#83C167">约为 per-head 的 1/128</text>

<rect x="340" y="74" width="316" height="204" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="356" y="94" font-size="10.5" fill="#E6E8EC">三步：切 latent → 聚齐 → absorb 着算</text>
<g fill="#9A72AC" fill-opacity="0.3" stroke="#9A72AC" stroke-width="0.8">
<rect x="356" y="108" width="54" height="20"/><rect x="356" y="130" width="54" height="20"/>
<rect x="356" y="152" width="54" height="20"/><rect x="356" y="174" width="54" height="20"/>
</g>
<text x="383" y="204" text-anchor="middle" font-size="8" fill="#5B6678">T/4 行</text>
<line x1="416" y1="150" x2="436" y2="150" stroke="#9A72AC" stroke-width="1.4" marker-end="url(#a11p)"/>
<text x="426" y="142" text-anchor="middle" font-size="7.5" fill="#B69BC7">gather</text>
<g fill="#9A72AC" fill-opacity="0.3" stroke="#9A72AC" stroke-width="0.8">
<rect x="444" y="108" width="54" height="20"/><rect x="444" y="130" width="54" height="20"/>
<rect x="444" y="152" width="54" height="20"/><rect x="444" y="174" width="54" height="20"/>


</g>
<text x="471" y="212" text-anchor="middle" font-size="8" fill="#5B6678">T 行</text>
<line x1="506" y1="170" x2="524" y2="170" stroke="#83C167" stroke-width="1.4" marker-end="url(#a11g)"/>
<rect x="528" y="140" width="112" height="58" rx="6" fill="#83C167" fill-opacity="0.14" stroke="#83C167" stroke-width="1.1"/>
<text x="584" y="160" text-anchor="middle" font-size="8.5" fill="#C6E3B6">Q 吸收进权重</text>
<text x="584" y="176" text-anchor="middle" font-size="8.5" fill="#C6E3B6">与 latentᵀ 相乘</text>
<text x="584" y="190" text-anchor="middle" font-size="8" fill="#7FA36F">输出 O 行块</text>
<text x="356" y="236" font-size="8.5" fill="#5B6678">通信量 T×512，约为 per-head 方案的 1/128</text>
<text x="356" y="252" font-size="8.5" fill="#FC6255">Ulysses 失效：只有 1 个 latent，无法按 head 切</text>
<text x="356" y="268" font-size="8.5" fill="#8B96A8">训练时 Megatron-Core 禁 absorbed，LAGA 给出替代</text>

<rect x="24" y="292" width="632" height="56" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="311" font-size="10" fill="#9AA4B2">DeepSeek-R1 推理只有 1 个 KV head，tp=8 时 latent 被复制 8 份；DCP=8 沿 T 维切分后复制被完全消除。</text>
<text x="44" y="329" font-size="10" fill="#9AA4B2">absorb 之后 latent 参与了 Q 的投影，若按 head 切分就会把 latent 复制回去 —— 这是 MLA 不能用 Ulysses 的根因。</text>
</g>
</svg>

▲ MLA 的 CP：latent 矩阵的形状对比（左）与 AllGather latent 通信（右）

## 4.3 DSA

DSA（DeepSeek Sparse Attention，DeepSeek-V3.2-Exp）由两部分组成。Lightning Indexer 是一个轻量的 MQA 式打分器（全层合计约 0.85B 参数），为每个 query 从全序列中选出 top-k 个最相关 token（64K 时 k≈2048），随后再对选中的 token 做细粒度稀疏注意力，复杂度从 $O(L^2)$ 降到 $O(kL)$。

DSA 叠加在 MLA 之上，CP 下需要传递的数据因而有两类，即 MLA 的 latent KV 与 Indexer 的 key（indexer key cache，用于避免 decode 时重算打分）。由于 top-k 选择需要对全序列的 indexer key 打分，各 rank 需要全局的 key 视图。主流做法是 AllGather latent KV 与 indexer key，之后各 rank 只为本地 query 做 indexer 打分与稀疏注意力。在 weight-absorb 模式下，latent 与 indexer key 都很小，AllGather 的内存与通信代价完全可以接受。昇腾在 TorchTitan 适配时对比过 Ring Attention 与 AllGather KV 两条路径，结论是 DSA 场景下 AllGather KV 实现简单且灵活，Ring 形式复杂且收益有限。

工程上还有若干配套做法。负载均衡采用 2C 头尾对称切片加 token 重排，在每层 Indexer 与稀疏注意力之前把 KV 还原回因果顺序，SGLang 的 interleave 策略与 vLLM PCP 的 DualChunkSwap 与此同理。SGLang Prefill CP 的 Interleave 策略对 DSA 有专门集成，按请求切分 query 与 indexer 元数据，并按需 gather indexer key 与 MLA KV。vLLM Ascend 曾实现独立的 DSA-CP 路径（`enable_dsa_cp`），现已由通用的 PCP 路径承接，两者不能同时开启。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 400" style="width:100%;height:auto">
<defs>
<linearGradient id="bg12" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot12" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a12o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FF862F"/></marker>
</defs>
<rect width="680" height="400" fill="url(#bg12)"/>
<rect width="680" height="400" fill="url(#dot12)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">DSA 的 CP：latent 与 indexer key 要一起搬</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">稀疏注意力多了一份「打分用的 key」，CP 时两份数据都要聚齐，再在本地做 top-k</text>

<rect x="24" y="74" width="196" height="252" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="42" cy="94" r="9" fill="#F4D345"/>
<text x="42" y="98" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="58" y="98" font-size="11" fill="#E6E8EC">分片：两份数据</text>
<g fill="#9A72AC" fill-opacity="0.32" stroke="#9A72AC" stroke-width="0.8">
<rect x="40" y="114" width="106" height="20"/><rect x="40" y="136" width="106" height="20"/>
<rect x="40" y="158" width="106" height="20"/><rect x="40" y="180" width="106" height="20"/>
</g>
<text x="93" y="216" text-anchor="middle" font-size="8.5" fill="#B69BC7">latent [T/4, 512]</text>
<g fill="#F4D345" fill-opacity="0.26" stroke="#F4D345" stroke-width="0.8">
<rect x="40" y="230" width="106" height="12"/><rect x="40" y="244" width="106" height="12"/>
<rect x="40" y="258" width="106" height="12"/><rect x="40" y="272" width="106" height="12"/>
</g>
<text x="93" y="298" text-anchor="middle" font-size="8.5" fill="#D6C179">indexer key [T/4, 128]</text>
<text x="40" y="314" font-size="8.5" fill="#5B6678">两份都按 T 维切，不按 head 切</text>

<rect x="250" y="74" width="196" height="252" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="268" cy="94" r="9" fill="#F4D345"/>
<text x="268" y="98" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="284" y="98" font-size="11" fill="#E6E8EC">AllGather：两份聚齐</text>
<g fill="#9A72AC" fill-opacity="0.32" stroke="#9A72AC" stroke-width="0.8">
<rect x="266" y="114" width="106" height="20"/><rect x="266" y="136" width="106" height="20"/>
<rect x="266" y="158" width="106" height="20"/><rect x="266" y="180" width="106" height="20"/>

</g>
<text x="319" y="220" text-anchor="middle" font-size="8.5" fill="#B69BC7">latent [T, 512]</text>
<g fill="#F4D345" fill-opacity="0.26" stroke="#F4D345" stroke-width="0.8">
<rect x="266" y="240" width="106" height="9"/><rect x="266" y="251" width="106" height="9"/>
<rect x="266" y="262" width="106" height="9"/><rect x="266" y="273" width="106" height="9"/>
</g>
<text x="319" y="298" text-anchor="middle" font-size="8.5" fill="#D6C179">indexer key [T, 128]</text>

<rect x="476" y="74" width="196" height="252" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="494" cy="94" r="9" fill="#F4D345"/>
<text x="494" y="98" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">3</text>
<text x="510" y="98" font-size="11" fill="#E6E8EC">本地 top-k，只算选中的列</text>
<text x="490" y="120" font-size="8.5" fill="#5B6678">score 行 = Q 与全部 key 的打分</text>
<g stroke="#0B0D12" stroke-width="1">
<rect x="488" y="128" width="14" height="26" fill="#F4D345" fill-opacity="0.45"/>
<rect x="502" y="128" width="14" height="26" fill="#141A24"/>
<rect x="516" y="128" width="14" height="26" fill="#141A24"/>
<rect x="530" y="128" width="14" height="26" fill="#F4D345" fill-opacity="0.45"/>
<rect x="544" y="128" width="14" height="26" fill="#141A24"/>
<rect x="558" y="128" width="14" height="26" fill="#F4D345" fill-opacity="0.45"/>
<rect x="572" y="128" width="14" height="26" fill="#141A24"/>
<rect x="586" y="128" width="14" height="26" fill="#141A24"/>
<rect x="600" y="128" width="14" height="26" fill="#5CD0B3" fill-opacity="0.4"/>
<rect x="614" y="128" width="14" height="26" fill="#5CD0B3" fill-opacity="0.4"/>
<rect x="628" y="128" width="14" height="26" fill="#5CD0B3" fill-opacity="0.4"/>
<rect x="642" y="128" width="14" height="26" fill="#5CD0B3" fill-opacity="0.4"/>
</g>
<text x="488" y="170" font-size="8" fill="#D6C179">top-k 选中（动态）</text>
<text x="642" y="170" text-anchor="end" font-size="8" fill="#8FC9BB">滑窗（固定）</text>
<g font-size="8.5" fill="#5B6678">
<text x="490" y="196">灰格 = 未选中，不参与 softmax</text>
<text x="490" y="214">选中的列散落在全部 T 上，</text>
<text x="490" y="230">所以必须先把 KV 聚齐才能选</text>
</g>
<rect x="490" y="246" width="164" height="26" rx="4" fill="#83C167" fill-opacity="0.14" stroke="#83C167" stroke-width="1"/>
<text x="572" y="263" text-anchor="middle" font-size="8.5" fill="#C6E3B6">只对被选列做注意力</text>
<text x="490" y="292" font-size="8.5" fill="#8B96A8">Load 极不均衡 → 2C 头尾切分</text>
<text x="490" y="308" font-size="8.5" fill="#8B96A8">算前需还原成因果顺序</text>

<rect x="24" y="336" width="632" height="52" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="355" font-size="10" fill="#9AA4B2">DSA 的 CP 不是「传什么」的问题，而是「什么时候选」：top-k 是 token 级动态的，只有拿到全局 latent 与 indexer key 才能选。</text>
<text x="44" y="373" font-size="10" fill="#9AA4B2">SGLang 的 Interleave 策略按请求切分 query 与 indexer 元数据，vLLM PCP 的 DualChunkSwap 做同样的事。</text>
</g>
</svg>

▲ DSA 的 CP：传递 latent 与 indexer key 两份矩阵（AllGather），本地做 top-k 稀疏注意力（mask 为动态选中列）

## 4.4 GDN

GDN（Gated DeltaNet）是线性注意力的一种，用固定大小的循环状态 $S \in \mathbb{R}^{d_v \times d_k}$ 替代随序列增长的 KV Cache，通过门控遗忘与 delta rule 实现定向覆盖。

$$S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^\top, \qquad o_t = S_t q_t$$

其中 $\alpha_t \in (0,1]$ 控制旧状态的整体遗忘，$\beta_t$ 控制当前 key 关联的写入强度。这一结构决定了 GDN 在 CP 下的形态。由于没有 KV Cache 可传，AllGather KV 类方案不再适用，rank 之间传递的是状态 $S$。状态大小固定（例如 Qwen3-Next 每 head 为 128×128），跨 rank 同步状态的通信量与序列长度无关。另一方面，$S_t$ 依赖 $S_{t-1}$ 的串行依赖必须妥善处理。训练与预填充阶段通常用 chunk-wise 并行（chunk 内矩阵并行、chunk 间串行传状态）摊开；CP 时可以沿序列在 rank 间传递状态（类似 ring 的状态链），也可以各段先独立计算状态转移再精确合成。兜底方案还有重算与 Ulysses（按 head 切），但状态链传递的通信量最优。实际模型多为混合架构，例如 Qwen3.5、Qwen3.8-Flash-Next 的大部分层使用 GDN、少数层保留全局注意力，CP 时两类层分别走状态传递与常规注意力通信。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 420" style="width:100%;height:auto">
<defs>
<linearGradient id="bg13" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot13" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a13" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#9A72AC"/></marker>
<marker id="a13g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#83C167"/></marker>
</defs>
<rect width="680" height="420" fill="url(#bg13)"/>
<rect width="680" height="420" fill="url(#dot13)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">GDN 的 CP：没有 KV 可传，只传一个固定大小的状态</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">线性注意力的段间依赖是仿射的 S_L = S_0Φ + B，这让各段可以先各算各的</text>

<circle cx="32" cy="84" r="9" fill="#F4D345"/>
<text x="32" y="88" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="48" y="88" font-size="10.5" fill="#8B96A8">四段并行：各自算出与输入状态无关的 (Φ, B)</text>
<g>
<rect x="30" y="98" width="146" height="56" rx="6" fill="#9A72AC" fill-opacity="0.12" stroke="#9A72AC" stroke-width="1"/>
<text x="103" y="118" text-anchor="middle" font-size="10" fill="#D8C4E2">段 0 · R0</text>
<text x="103" y="136" text-anchor="middle" font-size="9" fill="#8B96A8">Φ₀, B₀ 独立求值</text>
<rect x="190" y="98" width="146" height="56" rx="6" fill="#9A72AC" fill-opacity="0.12" stroke="#9A72AC" stroke-width="1"/>
<text x="263" y="118" text-anchor="middle" font-size="10" fill="#D8C4E2">段 1 · R1</text>
<text x="263" y="136" text-anchor="middle" font-size="9" fill="#8B96A8">Φ₁, B₁ 独立求值</text>
<rect x="350" y="98" width="146" height="56" rx="6" fill="#9A72AC" fill-opacity="0.12" stroke="#9A72AC" stroke-width="1"/>
<text x="423" y="118" text-anchor="middle" font-size="10" fill="#D8C4E2">段 2 · R2</text>
<text x="423" y="136" text-anchor="middle" font-size="9" fill="#8B96A8">Φ₂, B₂ 独立求值</text>
<rect x="510" y="98" width="146" height="56" rx="6" fill="#9A72AC" fill-opacity="0.12" stroke="#9A72AC" stroke-width="1"/>
<text x="583" y="118" text-anchor="middle" font-size="10" fill="#D8C4E2">段 3 · R3</text>
<text x="583" y="136" text-anchor="middle" font-size="9" fill="#8B96A8">Φ₃, B₃ 独立求值</text>
</g>
<text x="340" y="170" text-anchor="middle" font-size="9" fill="#5B6678">B_i 与输入状态无关，Φ_i 只依赖本段的门控与 key，因此四段可以完全并行</text>

<circle cx="32" cy="196" r="9" fill="#F4D345"/>
<text x="32" y="200" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="48" y="200" font-size="10.5" fill="#8B96A8">状态沿链接力：S_i = S_{i−1}Φ_i + B_i，每次只传一个 d×d 矩阵</text>
<g>
<rect x="30" y="212" width="84" height="34" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="72" y="234" text-anchor="middle" font-size="10" fill="#E6E8EC">S₀</text>
<rect x="150" y="212" width="84" height="34" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="192" y="234" text-anchor="middle" font-size="10" fill="#E6E8EC">S₁</text>
<rect x="270" y="212" width="84" height="34" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="312" y="234" text-anchor="middle" font-size="10" fill="#E6E8EC">S₂</text>
<rect x="390" y="212" width="84" height="34" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="432" y="234" text-anchor="middle" font-size="10" fill="#E6E8EC">S₃</text>
<rect x="510" y="212" width="84" height="34" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="552" y="234" text-anchor="middle" font-size="10" fill="#E6E8EC">S₄</text>
</g>
<g stroke="#9A72AC" stroke-width="1.5" fill="none">
<line x1="116" y1="229" x2="146" y2="229" marker-end="url(#a13)"/>
<line x1="236" y1="229" x2="266" y2="229" marker-end="url(#a13)"/>
<line x1="356" y1="229" x2="386" y2="229" marker-end="url(#a13)"/>
<line x1="476" y1="229" x2="506" y2="229" marker-end="url(#a13)"/>
</g>
<text x="340" y="264" text-anchor="middle" font-size="9" fill="#5B6678">C 次 P2P，传的是固定大小的 S（如 128×128），与序列长度无关</text>

<circle cx="32" cy="290" r="9" fill="#F4D345"/>
<text x="32" y="294" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">3</text>
<text x="48" y="294" font-size="10.5" fill="#8B96A8">拿到上一段的状态后，四段再次并行输出</text>
<g>
<rect x="30" y="306" width="146" height="46" rx="6" fill="#83C167" fill-opacity="0.12" stroke="#83C167" stroke-width="1"/>
<text x="103" y="334" text-anchor="middle" font-size="9.5" fill="#C6E3B6">O₀ = f(S₀, 段 0)</text>
<rect x="190" y="306" width="146" height="46" rx="6" fill="#83C167" fill-opacity="0.12" stroke="#83C167" stroke-width="1"/>
<text x="263" y="334" text-anchor="middle" font-size="9.5" fill="#C6E3B6">O₁ = f(S₀Φ₁+B₁, 段 1)</text>
<rect x="350" y="306" width="146" height="46" rx="6" fill="#83C167" fill-opacity="0.12" stroke="#83C167" stroke-width="1"/>
<text x="423" y="334" text-anchor="middle" font-size="9.5" fill="#C6E3B6">O₂ = f(S₁Φ₂+B₂, 段 2)</text>
<rect x="510" y="306" width="146" height="46" rx="6" fill="#83C167" fill-opacity="0.12" stroke="#83C167" stroke-width="1"/>
<text x="583" y="334" text-anchor="middle" font-size="9.5" fill="#C6E3B6">O₃ = f(S₂Φ₃+B₃, 段 3)</text>
</g>
<text x="340" y="372" text-anchor="middle" font-size="9.5" fill="#5B6678">实际模型多为混合架构：多数层用 GDN 走状态传递，少数全局注意力层仍走 3.1–3.3 的常规通信</text>
<rect x="24" y="386" width="632" height="24" rx="6" fill="#121826" stroke="#232C3A"/>
<text x="340" y="402" text-anchor="middle" font-size="9.5" fill="#9AA4B2">代价：状态链是串行的，段间要等；收益：通信量与 T 完全无关，百万 token 也不涨</text>
</g>
</svg>

▲ GDN 的 CP：无 KV 可传，rank 间只传固定大小的状态矩阵 S

**推导过程**

**① 从线性注意力到循环状态。** 去掉 softmax 后，因果注意力可以写成 $o_t = \sum_{s \le t} (q_t^\top k_s) v_s$。定义状态 $S_t = \sum_{s \le t} v_s k_s^\top \in \mathbb{R}^{d_v \times d_k}$，立刻得到递推形式

$$S_t = S_{t-1} + v_t k_t^\top, \qquad o_t = S_t q_t$$

状态大小固定，不随序列增长。但它只能叠加，同一个 key 先后写入 $v_1$、$v_2$ 之后读回的是 $v_1 + v_2$，旧关联无法被覆盖。

**② Delta rule（定向覆盖）。** 写入前先把 $k_t$ 处的旧关联减掉，只写修正量。

$$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t)\, k_t^\top$$

展开右式得 $S_{t-1} - \beta_t S_{t-1} k_t k_t^\top + \beta_t v_t k_t^\top$，合并后得到等价形式

$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

即先在 $k_t$ 方向擦除旧值、再写入 $v_t$，$\beta_t \in (0,1]$ 是写入步长。同一 key 连续写 $v_1、v_2$ 后读回恰为 $v_2$（可直接验证），这就是 delta rule 的定向覆盖。

**③ 加遗忘门（Gated DeltaNet）。** 对旧状态整体乘衰减 $\alpha_t \in (0,1]$。

$$S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t)\, k_t^\top \;\equiv\; S_{t-1} T_t + \beta_t v_t k_t^\top, \qquad T_t = \alpha_t (I - \beta_t k_t k_t^\top)$$

可以验证两种形式等价，将右式展开即 $S_{t-1}\,\alpha_t(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top = \alpha_t S_{t-1} + \beta_t(v_t - \alpha_t S_{t-1} k_t) k_t^\top$。于是整层递推是右乘线性转移 $T_t$ 再加一次秩一写入，输出 $o_t = S_t q_t$。$\alpha$ 负责整体快速遗忘，delta 负责定向覆盖，两者互补。

**④ 段级分解，即各段可以独立求值再合成的原因。** 对长度为 $L$ 的一段递推展开。

$$S_L = S_0\, \Phi_{0 \to L} + \sum_{s=1}^{L} \beta_s v_s k_s^\top\, \Phi_{s \to L}, \qquad \Phi_{s \to L} = T_{s+1} T_{s+2} \cdots T_L \;(\Phi_{L \to L} = I)$$

即 $S_L = S_0 \Phi + B$，其中 $B = \sum_s \beta_s v_s k_s^\top \Phi_{s \to L}$ 完全由该段的 $(k, v, \alpha, \beta)$ 决定，与输入状态 $S_0$ 无关。这一分解带来 CP 所需的两个关键性质。其一，各 rank 可以零通信地并行求出本段的 $(\Phi, B)$；其二，段间只需把固定大小的状态沿链传播，$S^{(i)}_{\text{out}} = S^{(i)}_{\text{in}} \Phi_i + B_i$，也可以两两合成段级摘要（$\Phi_{12} = \Phi_1 \Phi_2,\ B_{12} = B_1 \Phi_2 + B_2$）后一次性得到末状态。段内各位置输出 $o_t = (S_{\text{in}} \Phi_{\text{in} \to t} + B_{\text{in} \to t}) q_t$ 在 $S_{\text{in}}$ 到位后本地并行计算。

这正是 CP 下只需传递状态 $S$ 的原因。rank 之间要传的只是 $d_v \times d_k$ 的状态（或等价的段级摘要），通信量与序列长度无关。

**⑤ Chunk-wise 并行（GPU 上的落地）。** 标量 $\alpha_t$ 可以提到连乘外面，于是

$$\Phi = \Big(\prod_t \alpha_t\Big) \prod_t (I - \beta_t k_t k_t^\top) = \Big(\prod_t \alpha_t\Big)\,(I - K C K^\top)$$

其中 $K = [k_1, \dots, k_L]$，$C$ 是由 $\beta_t$ 与两两内积 $k_i^\top k_j$ 决定的 $L \times L$ 小矩阵（WY / Woodbury 表示）。段内计算因此归结为几次 GEMM 加一个小三角求解，而不再是 $L$ 步串行递推；chunk 之间再串行传状态。FlashKDA、FLA 系 kernel 的 chunkwise 算法都基于这一结构，CP 时的"chunk 内并行、rank 间传状态"也由此而来。

## 4.5 KDA

KDA（Kimi Delta Attention，Kimi Linear，arXiv:2510.26692）是 Gated DeltaNet 的细粒度升级。它把标量遗忘门换成逐通道遗忘门 $\alpha_t \in [0,1]^{d_k}$（对角转移矩阵），状态仍为固定大小，并配有 DPLR（对角加低秩）转移与 chunkwise 算法。Kimi-K3（2.8T 参数、1M 上下文）采用 3 层 KDA 加 1 层 Gated MLA 的混合结构。

KDA 的 CP 需要采用专门的 KCP（KDA Context Parallelism）。对标准线性注意力，每个序列分段的状态转移可以在不知道输入状态的情况下独立求值，再精确合成递归链；KDA 的逐通道遗忘门在数学上使这种独立求值再合成不再直接成立，具体推导见下文。KCP 的做法是把状态转移中可以独立计算的部分尽可能拆解出来，分发到不同 GPU 进程并行计算，再合成各段应得的状态。除跨设备 KCP 外，KCP 还支持单卡内 SM 级的序列二次切分（intra-device KCP），两者可以同时进行。这一点对百万 token 的长序列尤为重要，因为仅靠 TP 切 head 会让 prefill 的 SM 大量空转。跨设备同步的只是固定大小的状态，通信成本恒定，不随文本变长。工程配套方面，FlashKDA（基于 CUTLASS 的 chunkwise kernel）把 chunk 内并行与跨 chunk 状态传播重叠执行，state-aware prefix caching 则利用了固定大小状态天然适合缓存复用的特性。生态上，SGLang 针对 Kimi-K3 提供了 fused KDA decode kernel、KDA-aware prefix caching、DP attention、MTP 与 PD 分离支持。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 424" style="width:100%;height:auto">
<defs>
<linearGradient id="bg14" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot14" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a14" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#9A72AC"/></marker>
</defs>
<rect width="680" height="424" fill="url(#bg14)"/>
<rect width="680" height="424" fill="url(#dot14)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">KDA 的 CP：门控逐通道，段间就不能各自为政</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">GDN 的仿射分解依赖标量遗忘门；换成逐通道对角门之后，转移矩阵不再对易</text>

<rect x="24" y="74" width="306" height="252" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="40" y="94" font-size="10.5" fill="#E6E8EC">转移矩阵的两块：对角门 − 秩一修正</text>
<g stroke="#0B0D12" stroke-width="1">
<rect x="52" y="120" width="14" height="14" fill="#F4D345" fill-opacity="0.55"/><rect x="66" y="120" width="14" height="14" fill="#141A24"/><rect x="80" y="120" width="14" height="14" fill="#141A24"/><rect x="94" y="120" width="14" height="14" fill="#141A24"/><rect x="108" y="120" width="14" height="14" fill="#141A24"/>
<rect x="52" y="134" width="14" height="14" fill="#141A24"/><rect x="66" y="134" width="14" height="14" fill="#F4D345" fill-opacity="0.55"/><rect x="80" y="134" width="14" height="14" fill="#141A24"/><rect x="94" y="134" width="14" height="14" fill="#141A24"/><rect x="108" y="134" width="14" height="14" fill="#141A24"/>
<rect x="52" y="148" width="14" height="14" fill="#141A24"/><rect x="66" y="148" width="14" height="14" fill="#141A24"/><rect x="80" y="148" width="14" height="14" fill="#F4D345" fill-opacity="0.55"/><rect x="94" y="148" width="14" height="14" fill="#141A24"/><rect x="108" y="148" width="14" height="14" fill="#141A24"/>
<rect x="52" y="162" width="14" height="14" fill="#141A24"/><rect x="66" y="162" width="14" height="14" fill="#141A24"/><rect x="80" y="162" width="14" height="14" fill="#141A24"/><rect x="94" y="162" width="14" height="14" fill="#F4D345" fill-opacity="0.55"/><rect x="108" y="162" width="14" height="14" fill="#141A24"/>
<rect x="52" y="176" width="14" height="14" fill="#141A24"/><rect x="66" y="176" width="14" height="14" fill="#141A24"/><rect x="80" y="176" width="14" height="14" fill="#141A24"/><rect x="94" y="176" width="14" height="14" fill="#141A24"/><rect x="108" y="176" width="14" height="14" fill="#F4D345" fill-opacity="0.55"/>
</g>
<text x="85" y="206" text-anchor="middle" font-size="8.5" fill="#D6C179">D_t = diag(α_t)</text>
<text x="130" y="158" font-size="14" fill="#8B96A8">−</text>
<g stroke="#0B0D12" stroke-width="1">
<rect x="146" y="120" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/><rect x="160" y="120" width="14" height="14" fill="#9A72AC" fill-opacity="0.55"/><rect x="174" y="120" width="14" height="14" fill="#9A72AC" fill-opacity="0.28"/><rect x="188" y="120" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/><rect x="202" y="120" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/>
<rect x="146" y="134" width="14" height="14" fill="#9A72AC" fill-opacity="0.55"/><rect x="160" y="134" width="14" height="14" fill="#9A72AC" fill-opacity="0.9"/><rect x="174" y="134" width="14" height="14" fill="#9A72AC" fill-opacity="0.55"/><rect x="188" y="134" width="14" height="14" fill="#9A72AC" fill-opacity="0.34"/><rect x="202" y="134" width="14" height="14" fill="#9A72AC" fill-opacity="0.28"/>
<rect x="146" y="148" width="14" height="14" fill="#9A72AC" fill-opacity="0.28"/><rect x="160" y="148" width="14" height="14" fill="#9A72AC" fill-opacity="0.55"/><rect x="174" y="148" width="14" height="14" fill="#9A72AC" fill-opacity="0.55"/><rect x="188" y="148" width="14" height="14" fill="#9A72AC" fill-opacity="0.34"/><rect x="202" y="148" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/>
<rect x="146" y="162" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/><rect x="160" y="162" width="14" height="14" fill="#9A72AC" fill-opacity="0.34"/><rect x="174" y="162" width="14" height="14" fill="#9A72AC" fill-opacity="0.34"/><rect x="188" y="162" width="14" height="14" fill="#9A72AC" fill-opacity="0.28"/><rect x="202" y="162" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/>
<rect x="146" y="176" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/><rect x="160" y="176" width="14" height="14" fill="#9A72AC" fill-opacity="0.28"/><rect x="174" y="176" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/><rect x="188" y="176" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/><rect x="202" y="176" width="14" height="14" fill="#9A72AC" fill-opacity="0.2"/>
</g>
<text x="179" y="206" text-anchor="middle" font-size="8.5" fill="#B69BC7">β·(α⊙k)kᵀ（秩一）</text>
<text x="40" y="232" font-size="9.5" fill="#FC6255">对角阵与秩一项不对易：D(I−βkkᵀ) ≠ (I−βkkᵀ)D</text>
<text x="40" y="250" font-size="9.5" fill="#8B96A8">于是 WY 那种把整段乘积写成紧凑闭式的做法不封闭，</text>
<text x="40" y="266" font-size="9.5" fill="#8B96A8">GDN 的「各段独立求值再合成」不能原样套用。</text>
<text x="40" y="290" font-size="9.5" fill="#5B6678">可拆的部分仍在：对角门本身可并行作用在状态上。</text>
<text x="40" y="308" font-size="9.5" fill="#5B6678">KCP 就是把这部分拆出来分发，再合成整段状态。</text>

<rect x="348" y="74" width="308" height="252" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="364" y="94" font-size="10.5" fill="#E6E8EC">KCP 的做法：两级并行</text>
<text x="364" y="118" font-size="9.5" fill="#D8C4E2">① 跨设备：切序列段，段间传状态 S</text>
<g>
<rect x="364" y="128" width="60" height="30" rx="4" fill="#9A72AC" fill-opacity="0.18" stroke="#9A72AC" stroke-width="1"/>
<text x="394" y="147" text-anchor="middle" font-size="9" fill="#D8C4E2">R0</text>
<rect x="438" y="128" width="60" height="30" rx="4" fill="#9A72AC" fill-opacity="0.18" stroke="#9A72AC" stroke-width="1"/>
<text x="468" y="147" text-anchor="middle" font-size="9" fill="#D8C4E2">R1</text>
<rect x="512" y="128" width="60" height="30" rx="4" fill="#9A72AC" fill-opacity="0.18" stroke="#9A72AC" stroke-width="1"/>
<text x="542" y="147" text-anchor="middle" font-size="9" fill="#D8C4E2">R2</text>
<rect x="586" y="128" width="60" height="30" rx="4" fill="#9A72AC" fill-opacity="0.18" stroke="#9A72AC" stroke-width="1"/>
<text x="616" y="147" text-anchor="middle" font-size="9" fill="#D8C4E2">R3</text>
</g>
<g stroke="#9A72AC" stroke-width="1.3" fill="none">
<line x1="426" y1="143" x2="434" y2="143" marker-end="url(#a14)"/>
<line x1="500" y1="143" x2="508" y2="143" marker-end="url(#a14)"/>
<line x1="574" y1="143" x2="582" y2="143" marker-end="url(#a14)"/>
</g>
<text x="364" y="174" font-size="8.5" fill="#5B6678">传的只有 S（与 T 无关），C 次 P2P</text>

<text x="364" y="198" font-size="9.5" fill="#D8C4E2">② 卡内：单卡的序列再切一次，分发到 SM</text>
<rect x="364" y="208" width="274" height="66" rx="6" fill="#161C29" stroke="#3A4557"/>
<g fill="#9A72AC" fill-opacity="0.2" stroke="#9A72AC" stroke-width="0.8">
<rect x="372" y="216" width="62" height="50"/><rect x="438" y="216" width="62" height="50"/>
<rect x="504" y="216" width="62" height="50"/><rect x="570" y="216" width="62" height="50"/>
</g>
<g font-size="8" text-anchor="middle" fill="#B69BC7">
<text x="403" y="240">SM 0</text><text x="469" y="240">SM 1</text><text x="535" y="240">SM 2</text><text x="601" y="240">SM 3</text>
</g>
<g font-size="7.5" text-anchor="middle" fill="#7B5E8C">
<text x="403" y="256">tile 0</text><text x="469" y="256">tile 1</text><text x="535" y="256">tile 2</text><text x="601" y="256">tile 3</text>
</g>
<text x="364" y="292" font-size="8.5" fill="#5B6678">两级可以同时进行：设备间传 S，设备内切 tile</text>
<text x="364" y="310" font-size="8.5" fill="#83C167">对百万 token 的 prefill 尤其关键</text>

<rect x="24" y="340" width="632" height="62" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="359" font-size="10" fill="#9AA4B2">为什么不能只靠 TP：TP 切 head 不切序列，单卡仍要放下整条 prefill 的激活与中间结果；序列到百万级时，只有沿 T 切才有用。</text>
<text x="44" y="377" font-size="10" fill="#9AA4B2">KCP 把「可独立算的转移量」并行、把「串行的状态合成」固定成常数次通信，从而在逐通道门下仍拿到接近线性的扩展。</text>
<text x="44" y="393" font-size="9.5" fill="#5B6678">SGLang 对 Kimi-K3 提供 fused KDA decode kernel、KDA 感知的前缀缓存与 PD 分离支持。</text>
</g>
</svg>

▲ KDA 的 CP（KCP）：DPLR 转移矩阵（左）与"独立量并行 + 固定状态合成"两级并行（右）

**推导过程**

**① KDA 的更新律与 DPLR 转移。** 把 GDN 的标量遗忘门换成逐通道（对角）遗忘门 $\alpha_t \in [0,1]^{d_k}$，记 $D_t = \mathrm{Diag}(\alpha_t)$。

$$S_t = S_{t-1}\, D_t (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top \;=\; S_{t-1}\, T_t + \beta_t v_t k_t^\top, \qquad o_t = S_t q_t$$

单步转移可以整理成

$$T_t = D_t - \beta_t (D_t k_t)\, k_t^\top$$

即对角加秩一的形式，这正是 DPLR（Diagonal-Plus-Low-Rank）转移矩阵。Kimi Linear 的 chunkwise 算法对这一特殊 DPLR 形式做了专门化，比通用 DPLR 求解省掉大量计算，同时保持与经典 delta rule 一致的语义。逐通道门的意义在于，不同 key 维度可以有自己的遗忘速率，固定大小状态的记忆利用率更高。

**② 段级仿射结构（形式上仍然成立）。** 与 GDN 完全相同地展开得

$$S_L = S_0\, \Phi_{0 \to L} + \sum_{s=1}^{L} \beta_s v_s k_s^\top\, \Phi_{s \to L}, \qquad \Phi_{s \to L} = T_{s+1} \cdots T_L$$

每段先算 $(\Phi, B)$ 再合成，在代数上依旧成立，问题变成如何高效地表示与计算 $\Phi$。

**③ GDN 的独立求值再合成不能直接搬到 KDA 的原因。** GDN 的 $T_t = \alpha_t(I - \beta_t k_t k_t^\top)$ 中标量 $\alpha_t$ 与一切对易，可以提到连乘外面。

$$\Phi_{\text{GDN}} = \Big(\prod_t \alpha_t\Big)\,(I - K C K^\top)$$

紧凑结构（对角缩放与秩一修正的连乘）在段内 keys 张成的空间里封闭，段间合成只是小矩阵运算。KDA 中各 $D_t$ 是互不相同的对角阵，与秩一项不对易。

$$D(I - \beta k k^\top) = D - \beta (Dk) k^\top \;\neq\; (I - \beta k k^\top) D = D - \beta k (Dk)^\top$$

（除非 $k$ 落在退化方向上）。于是 $D_t$ 不能像标量那样外提，$(I - KCK^\top)$ 的 WY 紧凑形式不再封闭，连乘 $D_1R_1D_2R_2\cdots$（$R_t = I - \beta_t k_t k_t^\top$）会不断混合对角与低秩结构，直接套用通用 DPLR 合成的代价随段长增长。这正是公开资料所指出的，KDA 的状态转移函数带有记忆遗忘机制，在数学上引入了额外困难，使标准线性注意力的独立求值再合成不再直接可行。

**④ KCP 的拆解思路。** KCP 的做法是把状态转移中可以独立计算的部分尽可能拆解出来，再分发并行，具体分为四个层面。第一，段内与输入状态无关的量照常并行求值，包括 key/value 统计、累积对角衰减 $\prod_t D_t$、DPLR 转移中的低秩因子等，它们被拆到不同 GPU 进程独立计算。第二，段链上只传递固定大小的状态（或等价的段级转移摘要），再与各段的独立计算结果合成，跨设备同步的就是 $d_v \times d_k$ 的状态，通信成本恒定，不随文本变长。第三，除跨设备 KCP 外还有单卡内 SM 级的序列二次切分（intra-device KCP），百万 token 时只靠 TP 切 head 会让 prefill 的 SM 空转，KCP 把序列在 SM 之间再切分，各自算完状态再合并，与跨设备 KCP 同时进行。第四，工程配套上，FlashKDA 的 chunkwise kernel 把 chunk 内并行与跨 chunk 状态传播重叠，state-aware prefix caching 直接缓存固定大小的状态。

KCP 的完整数学细节以 Kimi K3 技术资料为准，公开材料给出的核心即上述可独立计算部分的拆解与固定状态的合成。

## 4.6 CSA

CSA（Compressed Sparse Attention，压缩稀疏注意力，DeepSeek-V4）是 DSA 思路的进一步演化，其核心是先压缩、再稀疏选择，整个机制由三部分构成。首先，KV 压缩用带学习位置偏差的 softmax-gated pooling，把每 $m=4$ 个 token 的 KV 条目压缩为一条压缩条目，序列长度缩为 $1/m$。其次，稀疏选择由 FP4 精度的 Lightning Indexer 对压缩条目打分，每个 query 只取 top-k（512~1024）个压缩块做注意力。最后，SWA 分支保留最近窗口内的精细局部依赖。CSA 还与 HCA（Heavily Compressed Attention，压缩率约 128 倍后做稠密注意力）在层间交替堆叠（V4-Pro 为 61 层 CSA/HCA 交替），兼顾近处细节、中程精准检索与远程全局覆盖。在 1M 上下文下，KV Cache 降到 V3 的 10%，单 token FLOPs 降到 27%。

CP 下传递的是压缩 KV 条目（以及独立缓存时的 indexer key），体量只有原始 KV 的 $1/m$，比 DSA 的 AllGather 更轻，因此传 KV 与 AllGather KV 几乎是自然选择。top-k 选择同样需要对全部压缩条目打分，即需要全局（压缩后的）key 视图，做法与 DSA 相同。需要注意的是，压缩组（每 $m$ 个 token）应作为 CP 切分的最小粒度，避免把一个压缩组切断到两个 rank，HCA 的高压缩摘要同理；SWA 分支只依赖局部窗口，除窗口跨界外基本不需要跨 rank 通信。

从演化路线看，NSA（2025-02，压缩、选择、滑窗三分支与块级稀疏）开其端，DSA（V3.2，token 级动态稀疏加 Lightning Indexer）继其后，CSA（V4，KV 压缩加稀疏选择）是最新形态。SGLang 的 DeepSeek-V4 后端（multi-ratio compressed KV pools、FlashMLA 与稀疏索引）以及 Prefill CP 的 DeepSeek V4 interleave 路径已有对应集成。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 396" style="width:100%;height:auto">
<defs>
<linearGradient id="bg15" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot15" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a15" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#FF862F"/></marker>
</defs>
<rect width="680" height="396" fill="url(#bg15)"/>
<rect width="680" height="396" fill="url(#dot15)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">CSA 的 CP：先压成条目，再在条目上做稀疏</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">KV 被压到 1/m，通信量随之下降；但 top-k 的候选变成了条目，选择粒度变粗</text>

<rect x="24" y="74" width="196" height="242" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="42" cy="94" r="9" fill="#F4D345"/>
<text x="42" y="98" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="58" y="98" font-size="11" fill="#E6E8EC">压缩：m 个 token → 1 条</text>
<g stroke="#0B0D12" stroke-width="0.8">
<rect x="44" y="114" width="22" height="14" fill="#FF862F" fill-opacity="0.22"/><rect x="66" y="114" width="22" height="14" fill="#FF862F" fill-opacity="0.22"/><rect x="88" y="114" width="22" height="14" fill="#FF862F" fill-opacity="0.22"/><rect x="110" y="114" width="22" height="14" fill="#FF862F" fill-opacity="0.22"/>
<rect x="44" y="130" width="22" height="14" fill="#FF862F" fill-opacity="0.3"/><rect x="66" y="130" width="22" height="14" fill="#FF862F" fill-opacity="0.3"/><rect x="88" y="130" width="22" height="14" fill="#FF862F" fill-opacity="0.3"/><rect x="110" y="130" width="22" height="14" fill="#FF862F" fill-opacity="0.3"/>
<rect x="44" y="146" width="22" height="14" fill="#FF862F" fill-opacity="0.38"/><rect x="66" y="146" width="22" height="14" fill="#FF862F" fill-opacity="0.38"/><rect x="88" y="146" width="22" height="14" fill="#FF862F" fill-opacity="0.38"/><rect x="110" y="146" width="22" height="14" fill="#FF862F" fill-opacity="0.38"/>
<rect x="44" y="162" width="22" height="14" fill="#FF862F" fill-opacity="0.46"/><rect x="66" y="162" width="22" height="14" fill="#FF862F" fill-opacity="0.46"/><rect x="88" y="162" width="22" height="14" fill="#FF862F" fill-opacity="0.46"/><rect x="110" y="162" width="22" height="14" fill="#FF862F" fill-opacity="0.46"/>
</g>
<text x="138" y="140" font-size="8" fill="#5B6678">4 个</text>
<text x="138" y="152" font-size="8" fill="#5B6678">token</text>
<line x1="88" y1="182" x2="88" y2="198" stroke="#FF862F" stroke-width="1.4" marker-end="url(#a15)"/>
<text x="96" y="194" font-size="8" fill="#8B96A8">压缩 m = 4</text>
<g stroke="#FF862F" stroke-width="1">
<rect x="44" y="202" width="22" height="16" fill="#FF862F" fill-opacity="0.62"/><rect x="66" y="202" width="22" height="16" fill="#FF862F" fill-opacity="0.62"/><rect x="88" y="202" width="22" height="16" fill="#FF862F" fill-opacity="0.62"/><rect x="110" y="202" width="22" height="16" fill="#FF862F" fill-opacity="0.62"/>
</g>
<text x="44" y="236" font-size="8.5" fill="#5B6678">1 条压缩 KV，携带 4 个 token 的信息</text>
<text x="44" y="254" font-size="8.5" fill="#5B6678">条目数 = T / m，KV 显存与通信量同比降</text>
<text x="44" y="276" font-size="8.5" fill="#FC6255">代价：分辨率变粗，只能整条选或整条弃</text>
<text x="44" y="296" font-size="8.5" fill="#5B6678">另有 FP4 indexer 负责给条目打分</text>

<rect x="250" y="74" width="196" height="242" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="268" cy="94" r="9" fill="#F4D345"/>
<text x="268" y="98" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="284" y="98" font-size="11" fill="#E6E8EC">AllGather：条目聚齐</text>
<g stroke="#FF862F" stroke-width="0.8">
<rect x="266" y="114" width="70" height="16" fill="#FF862F" fill-opacity="0.3"/>
<rect x="266" y="134" width="70" height="16" fill="#FF862F" fill-opacity="0.3"/>
<rect x="266" y="154" width="70" height="16" fill="#FF862F" fill-opacity="0.3"/>
</g>
<text x="342" y="150" font-size="8" fill="#5B6678">本 rank 的条目</text>
<line x1="344" y1="166" x2="344" y2="182" stroke="#FF862F" stroke-width="1.2" marker-end="url(#a15)"/>
<g stroke="#FF862F" stroke-width="0.7">
<rect x="266" y="188" width="126" height="11" fill="#FF862F" fill-opacity="0.24"/>
<rect x="266" y="200" width="126" height="11" fill="#FF862F" fill-opacity="0.28"/>
<rect x="266" y="212" width="126" height="11" fill="#FF862F" fill-opacity="0.32"/>
<rect x="266" y="224" width="126" height="11" fill="#FF862F" fill-opacity="0.36"/>
<rect x="266" y="236" width="126" height="11" fill="#FF862F" fill-opacity="0.4"/>
<rect x="266" y="248" width="126" height="11" fill="#FF862F" fill-opacity="0.44"/>
<rect x="266" y="260" width="126" height="11" fill="#FF862F" fill-opacity="0.48"/>
<rect x="266" y="272" width="126" height="11" fill="#FF862F" fill-opacity="0.52"/>
</g>
<text x="272" y="298" font-size="8.5" fill="#5B6678">每个 rank 持全部条目，后续选择才有全局视野</text>

<rect x="476" y="74" width="180" height="242" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="494" cy="94" r="9" fill="#F4D345"/>
<text x="494" y="98" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">3</text>
<text x="510" y="98" font-size="11" fill="#E6E8EC">稀疏：条目级选择</text>
<g stroke="#0B0D12" stroke-width="0.8">
<rect x="490" y="114" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="508" y="114" width="18" height="18" fill="#141A24"/><rect x="526" y="114" width="18" height="18" fill="#141A24"/><rect x="544" y="114" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="562" y="114" width="18" height="18" fill="#141A24"/><rect x="580" y="114" width="18" height="18" fill="#141A24"/><rect x="598" y="114" width="18" height="18" fill="#5CD0B3" fill-opacity="0.45"/><rect x="616" y="114" width="18" height="18" fill="#5CD0B3" fill-opacity="0.45"/>

<rect x="490" y="132" width="18" height="18" fill="#141A24"/><rect x="508" y="132" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="526" y="132" width="18" height="18" fill="#141A24"/><rect x="544" y="132" width="18" height="18" fill="#141A24"/><rect x="562" y="132" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="580" y="132" width="18" height="18" fill="#141A24"/><rect x="598" y="132" width="18" height="18" fill="#141A24"/><rect x="616" y="132" width="18" height="18" fill="#5CD0B3" fill-opacity="0.45"/>

<rect x="490" y="150" width="18" height="18" fill="#141A24"/><rect x="508" y="150" width="18" height="18" fill="#141A24"/><rect x="526" y="150" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="544" y="150" width="18" height="18" fill="#141A24"/><rect x="562" y="150" width="18" height="18" fill="#141A24"/><rect x="580" y="150" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="598" y="150" width="18" height="18" fill="#141A24"/><rect x="616" y="150" width="18" height="18" fill="#141A24"/>

<rect x="490" y="168" width="18" height="18" fill="#141A24"/><rect x="508" y="168" width="18" height="18" fill="#141A24"/><rect x="526" y="168" width="18" height="18" fill="#141A24"/><rect x="544" y="168" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="562" y="168" width="18" height="18" fill="#141A24"/><rect x="580" y="168" width="18" height="18" fill="#141A24"/><rect x="598" y="168" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="616" y="168" width="18" height="18" fill="#141A24"/>

<rect x="490" y="186" width="18" height="18" fill="#141A24"/><rect x="508" y="186" width="18" height="18" fill="#141A24"/><rect x="526" y="186" width="18" height="18" fill="#141A24"/><rect x="544" y="186" width="18" height="18" fill="#141A24"/><rect x="562" y="186" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/><rect x="580" y="186" width="18" height="18" fill="#141A24"/><rect x="598" y="186" width="18" height="18" fill="#141A24"/><rect x="616" y="186" width="18" height="18" fill="#F4D345" fill-opacity="0.5"/>
</g>
<text x="490" y="222" font-size="8" fill="#D6C179">金 = top-k 选中</text>
<text x="490" y="236" font-size="8" fill="#8FC9BB">绿 = 滑窗保底</text>
<text x="490" y="258" font-size="8.5" fill="#5B6678">条目内仍需完整 KV 才能算出</text>
<text x="490" y="274" font-size="8.5" fill="#5B6678">精确结果，故条目与原始 KV</text>
<text x="490" y="290" font-size="8.5" fill="#5B6678">要一起分片、一起搬</text>
<text x="490" y="308" font-size="8.5" fill="#83C167">压缩比越高，CP 越轻松</text>

<rect x="24" y="328" width="632" height="56" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="346" font-size="9.5" fill="#9AA4B2">演化路线：NSA（块级稀疏，压缩 / 选择 / 滑窗三分支）→ DSA（token 级动态稀疏）→ CSA（KV 压缩 + 稀疏选择）</text>
<text x="44" y="364" font-size="9.5" fill="#9AA4B2">对 CP 而言，压缩比是关键变量：条目越小，AllGather 越便宜，可承受的 CP 度也越高。</text>
</g>
</svg>

▲ CSA 的 CP：KV 压缩矩阵（左）、AllGather 压缩条目（中）、稀疏选择与 SWA 两种 mask（右）

# 5. 推理和训练引擎的实现

本章分别介绍 vLLM、SGLang 与 Megatron 中 CP 的工程实现。

## 5.1 vLLM

vLLM 把 PCP 与 DCP 设计为两个正交的命令行维度，相关讨论集中在 vLLM Slack 的 `#sig-context-parallel` 频道。

PCP（Prefill Context Parallel）由 `--prefill-context-parallel-size`（`-pcp`）开启。它按序列切分 prefill 计算、扩展进程的 world size，但不增加 KV cache 的分片数。官方文档给出两种策略，即 Partial query + full key/value（AllGather KV，摊薄 TTFT）与 Partial query + partial key/value（ring 式传 KV，应对放不下全局 KV 的超长请求），两者均在持续开发中。token 分区采用头尾式的 DualChunkSwap 保证负载均衡，并用 restore 索引（`pcp_allgather_restore_idx`）恢复原始 token 顺序。在 PD 分离部署中，PCP 只开在 prefill（kv_producer）引擎上，decode 侧的 `prefill_context_parallel_size` 为 1。

DCP（Decode Context Parallel）由 `--decode-context-parallel-size`（`-dcp`）开启。它沿序列维分片 decode 的 KV Cache，不扩大 world size（复用 TP/PCP rank），`dcp_size` 的取值范围是 $[1, tp\_size/H_{kv}]$。交错分片由 `--cp_kv_cache_interleave_size` 控制，可以按 token 级（$i \bmod$ `dcp_world_size`）或 block 级对齐；涉及 KV 传输（KV pooling、PD 分离）时建议设为 KV cache 的 block_size（默认 128）。通信后端由 `--dcp-comm-backend` 选择，`ag_rs` 为 AllGather 加 ReduceScatter，`a2a` 则交换 partial output 与 LSE 后用 Triton kernel 合并，MLA 模型下每层 NCCL 调用从 3 次降到 2 次。`--dcp-q-replicate` 在 DCP 组内冗余复制 MLA 的 Q 投影，使 decode 跳过每步的 query all-gather。目前 DCP 支持 MLA 与 GQA 模型，部分注意力后端还支持 DCP 与 MTP（multi-token prediction）组合加速解码。

## 5.2 SGLang

SGLang 的 Prefill CP 于 2025 年 12 月由百度百舸 AIAK 贡献合入主线（PR #12065）。它把 prompt 的 token 分布到 attention CP 组，各 rank 为本地 query 计算注意力（含 indexer），并通过 AllGather KV cache 获得所需的 K/V。实现上，`ContextParallelStrategy` 把 token 布局策略与模型执行、注意力 kernel 解耦，每进程一个策略实例，每次前向的状态放在 `ForwardBatch.attn_cp_metadata` 中。切分策略有两种。Zigzag 策略把每个请求的新增 token 切成 $2C$ 个连续块，rank $r$ 拿块 $r$ 与块 $2C-1-r$（每个请求独立重启切分，长度不整除时前段块拿余数），既保留连续的 query 块又保证因果负载均衡。Interleave 策略让 rank $r$ 拿展平索引 $r, r+C, r+2C, \dots$（索引跨 batch 的新增 token 连续、不按请求重置，position id 保持原值），query 分布更均匀，并配合 DSA 与 DeepSeek V4 的稀疏注意力集成（gather indexer key 与 MLA KV）。后端集成路径包括 Zigzag 加 FlashAttention（gather 并重排新 KV）、Zigzag 加 trtllm_mha（zigzag page table 的变长注意力）、Interleave 加 DSA，以及 DeepSeek V4 interleave。收益上，该实现把 indexer（DSA、DeepSeek V4）这类原本由各 rank 重复执行的工作切开；与 KV cache 分片（DSA cache LayerSplit）叠加可以进一步节省显存；与 EP 组合时可用 all-to-all 的 token dispatch/combine 替代 TP 的 AllReduce。

Decode CP 采用传 Q 策略，即 decode 阶段先对 Q 做 AllGather，各 rank 对本地 KV 分片算注意力后再以在线 softmax（LSE）合并。RadixTree 的处理上，逻辑树留在中心 scheduler，物理存储按 rank 分片（一条上下文序列的 KV cache 切到各 dcp rank），调度时把逻辑节点映射到物理 GPU rank。PD 分离下还支持 CP 感知的 KV 传输，`SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 用于控制由 CP rank 0 单独传输还是所有 CP rank 各传自己那部分。

模型支持面上，SGLang 已适配多种混合注意力模型（GLM-5.3-Flash 的 MLA、DSA 与 KDA，Kimi-K3 的 fused KDA decode 与 KDA-aware prefix caching，Qwen3.8-Flash-Next 的 GDN 与稀疏注意力等），CP 需要按层类型分别处理 KV、latent、状态与 indexer key。

## 5.3 Megatron

Megatron-LM / Megatron-Core 的 CP 是训练侧的工业级实现。入口参数包括 `--context-parallel-size`、`--cp-comm-type`（可选 `p2p`、`all_gather`、`a2a`、`a2a+p2p`）、`hierarchical_context_parallel_sizes`（内层 a2a 尺寸乘以外层 p2p 尺寸）与 `max_seqlen_per_dp_cp_rank`。默认的 `p2p` 即 Ring Attention，zigzag 切分成 $2 \times cp\_size$ 块（rank $k$ 拿块 $k$ 与 $2cp-1-k$），用双 CUDA stream 做计算与通信重叠，与 FlashAttention-2 深度集成，并以 LSE 合并部分结果。`a2a` 即 Ulysses，`all_gather` 为 AllGather KV，`a2a+p2p` 为层级混合（节点内 a2a、节点间 p2p），适合 CP 度超过 head 数的多节点场景。NVIDIA 文档给出的经验是，能调 TP 使 `CP ≤ num_kv_heads / TP` 时优先使用纯 a2a；KV head 很少或需要最大 CP 灵活性时使用 p2p。

使用中还需注意三点。其一，CP 应与 Megatron-SP 区分，后者是 TP 组内对 norm 与 dropout 的序列并行，Reduction 与 AllGather 几乎零额外开销，通常先启用 SP，激活仍不够时再叠加 CP。其二，MLA 注意力在 Megatron-Core 中有 explicit（训练用，per-head K/V 参与 a2a）与 absorbed（推理 decode 用，只 gather latent）两种实现，训练路径以 assert 禁用 absorbed（激活显存会回归 20~34%），低通信的训练方案见 LAGA 论文。其三，在 TorchTitan 侧，torch 原生 CP（序列切分加 RingAttention 或 AllGatherKV）对普通注意力可用；DSA 这类需要同时处理 KV 与 indexer key 的模型则要自定义 CP（`CustomContextParallelContext`，在 forward 中加入 KV AllGather）。

# 6. 动态CP

固定 CP 配置难以同时适配长短悬殊的上下文负载，动态调整 CP 的并行度因此逐渐成为研究与工程的热点。在两个代表性方向中，NVIDIA Helix 在执行时让同一组 GPU 于不同并行布局之间动态切换，北大与阿里合作的 Vertumnus 则在调度时按负载动态调整 CP 度与 worker 组成。

## 6.1 Helix

NVIDIA 在论文《Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding》（arXiv:2507.07120）中提出了面向交互式百万级 token 解码的混合执行策略。

Helix 针对的是自回归 decode 的两大瓶颈，即 FFN 权重读取与长 KV Cache 读取。TP 能缓解 FFN 权重读取，但对注意力扩展很差，TP 宽度超过 KV head 数后 KV Cache 被低效复制，并行度与 batch 上限都被锁死；与此同时，长 KV 的 DRAM 读取随 batch 线性增长。

Helix 的核心做法是把注意力与 FFN 的并行映射在时间维上解耦，而不是像 AFD 那样把两者空分到不同 GPU。整个过程分六步。第一，Attention 阶段使用 KV Parallelism（KVP），KV Cache 沿序列维分片到 CP rank，每个 rank 对本地 KV 分片计算 partial attention。第二，一次轻量通信步对 partial 结果做 LSE 式合并与校正，保证注意力结果精确。第三，FFN 阶段同一组 GPU 被重配置为 TP（dense 模型）或 TP×EP（MoE 模型）来计算 FFN/MoE，充分利用 GPU。第四，拓扑组织为 $N = KVP \times TP_A$，注意力 TP 度 $TP_A$ 不超过 KV head 数，QKV 投影冗余计算使每个 KV 分片可做完全本地的 FlashAttention。第五，Helix HOP-B 用 batchwise overlap 把上述通信开销隐藏起来，在保持低 TTL 的同时提高 GPU 效率。第六，KV 块按 rank 轮转（交错）分配，保证 KVP 组内所有 GPU 的显存均匀增长。这一交错分片方案也是 vLLM DCP 交错分片的出处，由 Moonshot 的 Chao Hong 提出。

进一步拆解其动态机制。Helix 的动态并不体现在运行时修改 CP 度，而体现在同一组 GPU 在每个 decoder layer 内被重配置两次，Attention 与 FFN 各用最优的并行映射，时分复用同一批卡。其布局基座为 $N$ 块 GPU 按 $N = KVP \times TP_A$ 组织，KV Cache 沿序列维交错分片常驻各 rank（显存均匀增长），整个生成过程保持不变。每个 layer 的执行按一拍进行。在 Attention 段（KVP 布局），每 rank 对本地 KV 分片做完整 FlashAttention（QKV 投影冗余计算换取本地性），得到 partial 输出 $(O_j, L_j)$；在轻量合并步，沿 KVP 组做一次 LSE 式合并与校正（只传 $O$ 与 LSE，通信量很小），注意力结果与单卡精确等价；在 FFN/MoE 段（TP/EP 布局），同一批 GPU 立刻切换映射，dense 模型走 TP，MoE 走 TP×EP，伴随权重 AllReduce 或路由 All-to-All；随后回到 Attention 段进入下一个 layer，逐层重复 KVP、合并、TP/EP 这一拍。HOP-B（Helix Overlapped P2P-Batch）进一步把合并通信与其余计算在 batch 维交错，请求 A 通信时请求 B 正在做 Attention 或 FFN 计算，通信被完全藏进计算，既保低 TTL 又保 GPU 利用率。

与传统做法相比，AFD 把注意力与 FFN 空分到不同 GPU（各干各的，但卡约有一半时间空闲），Helix 把两者时分到同一组 GPU；TP 单独使用时 KV Cache 随 KV head 复制，batch 上限被锁死。形象地说，Helix 等于在 Attention 时是 CP、在 FFN 时是 TP/EP，每层只有一次轻量 LSE 合并通信。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 430" style="width:100%;height:auto">
<defs>
<linearGradient id="bg16" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot16" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
</defs>
<rect width="680" height="430" fill="url(#bg16)"/>
<rect width="680" height="430" fill="url(#dot16)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">Helix：同一组卡，Attention 时当 CP、FFN 时当 TP</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">不改并行度，只改映射。每层重配置两次，中间只夹一次轻量合并</text>

<rect x="24" y="70" width="632" height="112" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="40" y="90" font-size="10.5" fill="#E6E8EC">一层的三段：KV 分片算注意力 → 合并 → 权重切分算 FFN</text>
<rect x="44" y="102" width="326" height="34" rx="4" fill="#58C4DD" fill-opacity="0.22" stroke="#58C4DD" stroke-width="1"/>
<text x="207" y="123" text-anchor="middle" font-size="9.5" fill="#CFEAF2">KV Parallelism：KV 交错分片，本地算 partial</text>
<rect x="374" y="102" width="62" height="34" rx="4" fill="#F4D345" fill-opacity="0.22" stroke="#F4D345" stroke-width="1"/>
<text x="405" y="117" text-anchor="middle" font-size="8.5" fill="#F3E3AE">LSE</text>
<text x="405" y="130" text-anchor="middle" font-size="8.5" fill="#F3E3AE">合并</text>
<rect x="440" y="102" width="200" height="34" rx="4" fill="#83C167" fill-opacity="0.2" stroke="#83C167" stroke-width="1"/>
<text x="540" y="123" text-anchor="middle" font-size="9.5" fill="#C6E3B6">TP / EP 映射：算 FFN</text>
<g font-size="8.5" text-anchor="middle" fill="#5B6678">
<text x="207" y="152">Attention 段的显存按 T 切，batch 上限不再被 KV 复制锁死</text>
<text x="405" y="152">唯一的跨卡通信</text>
<text x="540" y="152">FFN 段的权重按 hidden 切，卡不空转</text>
</g>
<text x="44" y="172" font-size="9" fill="#8B96A8">逐层循环：第 l 层的这三段跑完，第 l+1 层重新按同样的顺序重配置一次</text>

<rect x="24" y="192" width="308" height="150" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="40" y="212" font-size="10.5" fill="#E6E8EC">HOP-B：用另一个请求盖住合并的通信</text>
<text x="40" y="234" font-size="8.5" fill="#8B96A8">请求 A</text>
<g>
<rect x="76" y="224" width="70" height="16" fill="#58C4DD" fill-opacity="0.3" stroke="#58C4DD" stroke-width="0.8"/>
<rect x="146" y="224" width="26" height="16" fill="#F4D345" fill-opacity="0.3" stroke="#F4D345" stroke-width="0.8"/>
<rect x="172" y="224" width="80" height="16" fill="#83C167" fill-opacity="0.28" stroke="#83C167" stroke-width="0.8"/>
<rect x="252" y="224" width="26" height="16" fill="#F4D345" fill-opacity="0.3" stroke="#F4D345" stroke-width="0.8"/>
<rect x="278" y="224" width="34" height="16" fill="#58C4DD" fill-opacity="0.3" stroke="#58C4DD" stroke-width="0.8"/>
</g>
<text x="40" y="266" font-size="8.5" fill="#8B96A8">请求 B</text>
<g>
<rect x="76" y="256" width="26" height="16" fill="#F4D345" fill-opacity="0.3" stroke="#F4D345" stroke-width="0.8"/>
<rect x="102" y="256" width="70" height="16" fill="#58C4DD" fill-opacity="0.3" stroke="#58C4DD" stroke-width="0.8"/>
<rect x="172" y="256" width="26" height="16" fill="#F4D345" fill-opacity="0.3" stroke="#F4D345" stroke-width="0.8"/>
<rect x="198" y="256" width="80" height="16" fill="#83C167" fill-opacity="0.28" stroke="#83C167" stroke-width="0.8"/>
<rect x="278" y="256" width="34" height="16" fill="#58C4DD" fill-opacity="0.3" stroke="#58C4DD" stroke-width="0.8"/>
</g>
<g stroke="#F4D345" stroke-width="0.9" stroke-dasharray="3 3" fill="none">
<line x1="172" y1="242" x2="172" y2="254"/>
<line x1="252" y1="242" x2="252" y2="254"/>
</g>
<text x="40" y="292" font-size="8.5" fill="#5B6678">A 的合并（黄）对齐 B 的计算（蓝/绿），</text>
<text x="40" y="308" font-size="8.5" fill="#5B6678">B 的合并再对齐 A 的下一段 —— 通信被藏掉。</text>
<text x="40" y="328" font-size="8.5" fill="#83C167">副作用：同时在线两个请求，KV 需双份预算。</text>

<rect x="348" y="192" width="308" height="150" rx="10" fill="#0E131D" stroke="#232C3A"/>
<text x="364" y="212" font-size="10.5" fill="#E6E8EC">同一次重配置，两种切法</text>
<text x="364" y="234" font-size="8.5" fill="#8B96A8">KVP：KV 按 T 交错切</text>
<g stroke="#0B0D12" stroke-width="1">
<rect x="364" y="242" width="16" height="16" fill="#58C4DD" fill-opacity="0.4"/><rect x="380" y="242" width="16" height="16" fill="#5CD0B3" fill-opacity="0.4"/><rect x="396" y="242" width="16" height="16" fill="#F4D345" fill-opacity="0.4"/><rect x="412" y="242" width="16" height="16" fill="#FC6255" fill-opacity="0.4"/><rect x="428" y="242" width="16" height="16" fill="#58C4DD" fill-opacity="0.4"/><rect x="444" y="242" width="16" height="16" fill="#5CD0B3" fill-opacity="0.4"/><rect x="460" y="242" width="16" height="16" fill="#F4D345" fill-opacity="0.4"/><rect x="476" y="242" width="16" height="16" fill="#FC6255" fill-opacity="0.4"/><rect x="364" y="262" width="16" height="16" fill="#58C4DD" fill-opacity="0.4"/><rect x="380" y="262" width="16" height="16" fill="#5CD0B3" fill-opacity="0.4"/><rect x="396" y="262" width="16" height="16" fill="#F4D345" fill-opacity="0.4"/><rect x="412" y="262" width="16" height="16" fill="#FC6255" fill-opacity="0.4"/><rect x="428" y="262" width="16" height="16" fill="#58C4DD" fill-opacity="0.4"/><rect x="444" y="262" width="16" height="16" fill="#5CD0B3" fill-opacity="0.4"/><rect x="460" y="262" width="16" height="16" fill="#F4D345" fill-opacity="0.4"/><rect x="476" y="262" width="16" height="16" fill="#FC6255" fill-opacity="0.4"/>
</g>
<text x="364" y="300" font-size="8" fill="#5B6678">块 i → rank (i mod 4)，每卡隔 4 取 1</text>
<text x="364" y="316" font-size="8" fill="#5B6678">显存均匀，逻辑顺序需重排</text>
<text x="530" y="234" font-size="8.5" fill="#8B96A8">TP：权重按列切</text>
<g stroke="#83C167" stroke-width="1">
<rect x="530" y="242" width="24" height="56" fill="#83C167" fill-opacity="0.28"/>
<rect x="554" y="242" width="24" height="56" fill="#83C167" fill-opacity="0.2"/>
<rect x="578" y="242" width="24" height="56" fill="#83C167" fill-opacity="0.14"/>
<rect x="602" y="242" width="24" height="56" fill="#83C167" fill-opacity="0.1"/>
</g>
<text x="530" y="316" font-size="8" fill="#5B6678">每卡一份完整激活</text>
<text x="530" y="330" font-size="8" fill="#5B6678">显存占用是另一套账</text>

<rect x="24" y="354" width="632" height="62" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="373" font-size="10" fill="#9AA4B2">与 AFD 的对比：AFD 把 Attention 与 FFN 空分到不同 GPU，两类卡各有一半时间闲着；Helix 时分复用同一组卡，没有闲着的那一半。</text>
<text x="44" y="391" font-size="10" fill="#9AA4B2">代价是每层一次重配置与一次合并通信，且两套映射的显存预算必须同时成立 —— 这就是它需要 HOP-B 来填通信空隙的原因。</text>
<text x="44" y="407" font-size="9.5" fill="#5B6678">在 DeepSeek-R1、Blackwell NVL72、FP4、1M 上下文的模拟中，固定 batch 下 TTL 最多降 1.5 倍，同延迟预算下 batch 最多放大 32 倍。</text>
</g>
</svg>

▲ Helix 动态 CP 机制：每层一拍的 KVP、合并、TP/EP 时序，HOP-B 的 batch 交错，两种映射下的分片形态

效果方面，在 DeepSeek-R1、Blackwell NVL72、FP4、1M token 上下文的模拟中，固定 batch 下 TTL 最多降低 1.5 倍，同等延迟预算下 batch 最多放大 32 倍。其目标场景是 PD 分离架构中的 generation 服务器，以及输入超过 64K、低 batch、延迟敏感（高 tokens/s/user）的负载。

实现方面，TensorRT-LLM 已支持 Helix（`context_parallel_size ≥ 2`、`cp_config.cp_type = "HELIX"`，`cp_config.tokens_per_block` 需与 KV cache 配置一致），目前支持 Blackwell 上的 MLA 模型（DeepSeek-V3、V3-Lite），并配有 MLA Helix 单测与分离式服务精度测试。

## 6.2 Adaptive Context Parallelism

北大与阿里联合发表的《Adaptive Context Parallelism for Production LLM Serving》（系统名 Vertumnus，arXiv:2609.04774，2026-09）把 CP 度变成生产调度的一等公民。

其出发点在于生产负载的三个特征。第一，请求长度高度异构，从几百 token 的问答到几十万 token 的文档分析与 Agent 请求并存。第二，负载随时间动态变化，到达率与长短请求占比有数倍波动。第三，前缀复用普遍（系统提示、多轮对话、Agent 工作流），前缀缓存必不可少。CP 的收益与输入长度强相关，长请求从高 CP 度明显获益，短请求却因通信开销反而变慢，静态 CP 配置注定顾此失彼。已有方案的不足也相应分为三类。有的完全不把 CP 度当作调度维度；有的采用静态 CP 配置（部署时定死，最多按长度路由，不考虑缓存命中与 worker 负载）；还有的虽能动态调整 CP 度，但只针对活跃请求或批次，没有稳定的持久化 worker 集群，前缀缓存无法跨请求复用。

Vertumnus 的设计相应分为三层。第一层是请求级路由，用 placement cost 在不同 CP 度的 worker 之间路由请求，代价函数由预测排队延迟、cache-aware 的 prefill 时间与 GPU-time 成本构成。第二层是集群级重组，随负载变化以秒级 split/merge 调整 worker 组成，短请求潮到来时把大 CP worker 拆成多个小 worker，长请求潮到来时把小 worker 合并成大 worker，全程无须重启进程或重新加载模型。第三层是全局前缀缓存管理，跨 worker（无论 CP 度是否相同）协调缓存的放置与复制，在 worker 重组、请求改派时保持缓存局部性，缺失的缓存块可以从其他 worker（甚至是不同 CP 度的 worker）按需复制。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 412" style="width:100%;height:auto">
<defs>
<linearGradient id="bg17" x1="0" y1="0" x2="0.3" y2="1">
<stop offset="0" stop-color="#101728"/><stop offset="1" stop-color="#080A10"/></linearGradient>
<pattern id="dot17" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="0.9" fill="#2C3748"/></pattern>
<marker id="a17" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#5B6678"/></marker>
<marker id="a17g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#83C167"/></marker>
</defs>
<rect width="680" height="412" fill="url(#bg17)"/>
<rect width="680" height="412" fill="url(#dot17)" opacity="0.55"/>
<g font-family="Georgia, 'Times New Roman', 'Source Han Serif SC', 'Songti SC', SimSun, serif">
<text x="36" y="34" font-size="19" fill="#F2F4F8">Vertumnus：让 CP 度跟着负载走</text>
<text x="36" y="55" font-size="11.5" fill="#8B96A8">请求级路由、秒级 split/merge、跨 worker 的前缀缓存，三层各管一件事</text>

<rect x="24" y="72" width="204" height="190" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="42" cy="92" r="9" fill="#F4D345"/>
<text x="42" y="96" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">1</text>
<text x="58" y="96" font-size="10.5" fill="#E6E8EC">请求级路由</text>
<rect x="36" y="110" width="72" height="24" rx="4" fill="#58C4DD" fill-opacity="0.18" stroke="#58C4DD" stroke-width="0.9"/>
<text x="72" y="126" text-anchor="middle" font-size="8.5" fill="#CFEAF2">短请求</text>
<rect x="36" y="144" width="72" height="24" rx="4" fill="#5CD0B3" fill-opacity="0.18" stroke="#5CD0B3" stroke-width="0.9"/>
<text x="72" y="160" text-anchor="middle" font-size="8.5" fill="#C3E8DC">中等请求</text>
<rect x="36" y="178" width="72" height="24" rx="4" fill="#F4D345" fill-opacity="0.18" stroke="#F4D345" stroke-width="0.9"/>
<text x="72" y="194" text-anchor="middle" font-size="8.5" fill="#F3E3AE">超长请求</text>
<g stroke="#5B6678" stroke-width="1.2" fill="none">
<line x1="110" y1="122" x2="132" y2="116" marker-end="url(#a17)"/>
<line x1="110" y1="156" x2="132" y2="152" marker-end="url(#a17)"/>
<line x1="110" y1="190" x2="132" y2="188" marker-end="url(#a17)"/>
</g>
<rect x="136" y="104" width="78" height="24" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="175" y="120" text-anchor="middle" font-size="8.5" fill="#E6E8EC">worker · CP=1</text>
<rect x="136" y="140" width="78" height="24" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="175" y="156" text-anchor="middle" font-size="8.5" fill="#E6E8EC">worker · CP=2</text>
<rect x="136" y="176" width="78" height="24" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="175" y="192" text-anchor="middle" font-size="8.5" fill="#E6E8EC">worker · CP=4</text>
<text x="36" y="224" font-size="8.5" fill="#5B6678">代价 = 预测排队 + cache-aware 的</text>
<text x="36" y="238" font-size="8.5" fill="#5B6678">prefill 时间 + GPU-time 成本</text>
<text x="36" y="254" font-size="8.5" fill="#83C167">按请求粒度挑 CP 度，而不是全局一个</text>

<rect x="240" y="72" width="204" height="190" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="258" cy="92" r="9" fill="#F4D345"/>
<text x="258" y="96" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">2</text>
<text x="274" y="96" font-size="10.5" fill="#E6E8EC">秒级 split / merge</text>
<rect x="256" y="108" width="72" height="28" rx="4" fill="#F4D345" fill-opacity="0.16" stroke="#F4D345" stroke-width="1"/>
<text x="292" y="126" text-anchor="middle" font-size="9" fill="#F3E3AE">CP=4</text>
<line x1="332" y1="122" x2="352" y2="112" stroke="#83C167" stroke-width="1.3" marker-end="url(#a17g)"/>
<line x1="332" y1="122" x2="352" y2="136" stroke="#83C167" stroke-width="1.3" marker-end="url(#a17g)"/>

<rect x="356" y="100" width="72" height="24" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="392" y="116" text-anchor="middle" font-size="8.5" fill="#E6E8EC">CP=2</text>
<rect x="356" y="128" width="72" height="24" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="392" y="144" text-anchor="middle" font-size="8.5" fill="#E6E8EC">CP=2</text>

<rect x="256" y="176" width="72" height="24" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="292" y="192" text-anchor="middle" font-size="8.5" fill="#E6E8EC">CP=1</text>
<rect x="256" y="206" width="72" height="24" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="292" y="222" text-anchor="middle" font-size="8.5" fill="#E6E8EC">CP=1</text>
<line x1="332" y1="188" x2="352" y2="194" stroke="#58C4DD" stroke-width="1.3" marker-end="url(#a17)"/>
<line x1="332" y1="218" x2="352" y2="204" stroke="#58C4DD" stroke-width="1.3" marker-end="url(#a17)"/>

<rect x="356" y="186" width="72" height="26" rx="4" fill="#58C4DD" fill-opacity="0.16" stroke="#58C4DD" stroke-width="1"/>
<text x="392" y="203" text-anchor="middle" font-size="9" fill="#CFEAF2">CP=2</text>
<text x="256" y="248" font-size="8" fill="#5B6678">不重启进程、不重载权重</text>

<rect x="456" y="72" width="200" height="190" rx="10" fill="#0E131D" stroke="#232C3A"/>
<circle cx="474" cy="92" r="9" fill="#F4D345"/>
<text x="474" y="96" text-anchor="middle" font-size="10" font-weight="bold" fill="#0B0D12">3</text>
<text x="490" y="96" font-size="10.5" fill="#E6E8EC">全局前缀缓存</text>
<g>
<rect x="470" y="108" width="172" height="30" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="480" y="127" font-size="8" fill="#9AA4B2">w0 · CP=4</text>
<g stroke="#83C167" stroke-width="0.8">
<rect x="540" y="115" width="16" height="16" fill="#83C167" fill-opacity="0.3"/><rect x="558" y="115" width="16" height="16" fill="#83C167" fill-opacity="0.3"/>
<rect x="576" y="115" width="16" height="16" fill="#83C167" fill-opacity="0.18"/><rect x="594" y="115" width="16" height="16" fill="#83C167" fill-opacity="0.18"/>
<rect x="612" y="115" width="16" height="16" fill="#83C167" fill-opacity="0.18"/>
</g>
<rect x="470" y="146" width="172" height="30" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="480" y="165" font-size="8" fill="#9AA4B2">w1 · CP=2</text>
<g stroke="#83C167" stroke-width="0.8">
<rect x="540" y="153" width="16" height="16" fill="#83C167" fill-opacity="0.18"/><rect x="558" y="153" width="16" height="16" fill="#83C167" fill-opacity="0.3"/>
<rect x="576" y="153" width="16" height="16" fill="#83C167" fill-opacity="0.3"/><rect x="594" y="153" width="16" height="16" fill="#83C167" fill-opacity="0.18"/>
<rect x="612" y="153" width="16" height="16" fill="#83C167" fill-opacity="0.18"/>
</g>
<rect x="470" y="184" width="172" height="30" rx="4" fill="#161C29" stroke="#3A4557"/>
<text x="480" y="203" font-size="8" fill="#9AA4B2">w2 · CP=1</text>
<g stroke="#83C167" stroke-width="0.8">
<rect x="540" y="191" width="16" height="16" fill="#83C167" fill-opacity="0.3"/><rect x="558" y="191" width="16" height="16" fill="#83C167" fill-opacity="0.18"/>
<rect x="576" y="191" width="16" height="16" fill="#83C167" fill-opacity="0.18"/><rect x="594" y="191" width="16" height="16" fill="#83C167" fill-opacity="0.3"/>
<rect x="612" y="191" width="16" height="16" fill="#83C167" fill-opacity="0.18"/>
</g>
</g>
<g stroke="#83C167" stroke-width="1" fill="none" stroke-dasharray="3 3" opacity="0.8">
<path d="M566 174 C 566 180, 566 182, 566 188" marker-end="url(#a17g)"/>
<path d="M602 140 C 602 146, 602 148, 602 152" marker-end="url(#a17g)"/>
</g>
<text x="470" y="228" font-size="8" fill="#5B6678">缺块可从别的 worker（哪怕 CP 度不同）</text>
<text x="470" y="242" font-size="8" fill="#5B6678">复制过来，重组后仍命中缓存</text>
<text x="470" y="256" font-size="8" fill="#83C167">浅格 = 缺失，虚线 = 跨 worker 复制</text>

<rect x="24" y="280" width="632" height="58" rx="8" fill="#121826" stroke="#232C3A"/>
<text x="44" y="299" font-size="10" fill="#9AA4B2">三层合起来的效果：请求按自身长度落到合适的 CP 度上，负载形态变化时 worker 自动重组，重组不丢前缀缓存。</text>
<text x="44" y="317" font-size="10" fill="#FC6255">代价：重组期间有迁移开销，缓存复制占用带宽，代价函数需要在线校准。</text>

<rect x="24" y="348" width="632" height="52" rx="8" fill="#0E131D" stroke="#232C3A"/>
<text x="44" y="367" font-size="10" fill="#8B96A8">实测（64 GPU，公开集 + 生产流量）：最高负载下平均 TTFT 降 28.1%，token 加权 SLO 达成率提升 13.3 个百分点。</text>
<text x="44" y="385" font-size="10" fill="#8B96A8">与 Helix 的分工：Helix 在单次前向内改映射，Vertumnus 在集群层面改 worker 组成，两者可以叠加。</text>
</g>
</svg>

▲ Vertumnus 三层机制：请求级路由（左）、集群级 split/merge（中）、全局前缀缓存（右）

效果方面，在 64 GPU 集群、公开与生产混合负载下，最高负载时平均 TTFT 最多降低 28.1%，token 加权 SLO 达成率最多提升 13.3 个百分点（对比最强基线）。

# 参考文献

1. Jacobs, S. A. et al. **DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**. arXiv:2309.14509, 2023. https://arxiv.org/abs/2309.14509
2. Liu, H., Zaharia, M., Abbeel, P. **Ring Attention with Blockwise Transformers for Near-Infinite Context**. arXiv:2310.01889, 2023. https://arxiv.org/abs/2310.01889
3. Yang, A. et al. (Meta). **Context Parallelism for Scalable Million-Token Inference**（pass-KV / pass-Q ring attention）. arXiv:2411.01783, 2024（MLSys 2025）. https://arxiv.org/abs/2411.01783
4. Fang, J., Zhao, S. **USP: A Unified Sequence Parallelism Approach for Long Context Generative AI**. arXiv:2405.07719, 2024. https://arxiv.org/abs/2405.07719
5. Korthikanti, V. A. et al. **Reducing Activation Recomputation in Large Transformer Models**（Megatron-SP）. arXiv:2205.05198, 2022. https://arxiv.org/abs/2205.05198
6. Bhatia, N. et al. (NVIDIA). **Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding**. arXiv:2507.07120, 2025. https://arxiv.org/abs/2507.07120
7. Guo, J., Wang, R., Huang, P., Lv, Z., Wang, Z., Liu, K., Lan, T., Qu, L., Wang, X., Yang, T. **Adaptive Context Parallelism for Production LLM Serving**（Vertumnus，北大 & 阿里）. arXiv:2609.04774, 2026. https://arxiv.org/abs/2609.04774
8. DeepSeek-AI. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**（MLA）. arXiv:2405.04434, 2024. https://arxiv.org/abs/2405.04434
9. Yuan, J. et al. **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention**（NSA）. arXiv:2502.11089, 2025. https://arxiv.org/abs/2502.11089
10. Yang, S. et al. **Gated Delta Networks: Improving Mamba2 with Delta Rule**. arXiv:2412.06464, 2024（ICLR 2025）. https://arxiv.org/abs/2412.06464
11. Kimi Team. **Kimi Linear: An Expressive, Efficient Attention Architecture**（KDA）. arXiv:2510.26692, 2025. https://arxiv.org/abs/2510.26692
12. Ma, C. **A Training-Memory Regression in MLA Sequence Parallelism: Why Megatron-Core Forbids Absorption, and LAGA**. arXiv:2607.17644, 2026. https://arxiv.org/abs/2607.17644
13. DeepSeek-AI. **DeepSeek-V3.2-Exp**（DSA / Lightning Indexer）. 2025-12. https://github.com/deepseek-ai/
14. DeepSeek-AI. **DeepSeek-V4**（CSA / HCA 混合注意力）. 2026-04（官方发布资料）.
15. vLLM 官方文档. **Context Parallel Deployment**（PCP/DCP）. https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html
16. SGLang 官方文档. **Prefill Context Parallelism**. https://docs.sglang.io/docs/advanced_features/prefill_cp ；CP 合入 PR（百度百舸 AIAK）：https://github.com/sgl-project/sglang/pull/12065
17. NVIDIA 文档. **Hierarchical Context Parallel**（Megatron-Bridge，cp_comm_type / a2a+p2p）. https://docs.nvidia.com/nemo/megatron-bridge/
18. NVIDIA 文档. **Helix Parallelism**（TensorRT-LLM）. https://nvidia.github.io/TensorRT-LLM/features/helix.html
19. Moonshot AI（Yuxin Wu）. **Introduction to Kimi K2**（Decode Context Parallel 交错分片由 Chao Hong 提出；2~3 倍吞吐收益）. PyTorch 2025 演讲纪要.
