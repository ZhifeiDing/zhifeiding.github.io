# 1. 概述

在大语言模型的上下文越来越长背景下，Context Parallelism逐渐成为一种必要的模型并行方案。同时，为了能同时适配长上下文和短上下文的需求，Dynamic CP技术也应运而生。vLLM和SGLang推理引擎针对Prefill和Decode分别实现了PCP(Prefill CP)和DCP(Decode CP)，NVIDIA Megatron也实现了CP，以及Helix的Dynamic CP，Alibaba也发布了Adaptive Context Parallelism相关论文。

# 2. 什么是CP

传统的TP并行针对head_num进行切分，需要通过AllReduce汇聚结果，通信量大，而且受限于head_num。而且随着上下文越来越大，即使进行TP切分之后，激活和KVCache也越来越大。因此，针对超长上下文的需求，针对序列进行切分就是个迫切的需求了。

CP（Context Parallelism，上下文并行）沿**序列（token）维度**切分。长度为 $T$ 的序列被切成 $C$ 份，每个 CP rank 持有 $T/C$ 个 token 的激活以及对应区间的 KV Cache。除 Attention（以及线性注意力的循环状态）外，Linear、MLP、Norm 等算子都是逐 token 独立的，天然可以按 token 并行；唯一的跨 token 依赖在注意力上，需要通过 rank 间通信解决。

与 TP 的对比：

| 维度 | Tensor Parallel | Context Parallel |
| --- | --- | --- |
| 切分维度 | head / hidden 维 | 序列（token）维 |
| 每层通信 | AllReduce，通信量大 | 只传注意力相关数据（KV / Q / latent / 状态） |
| 并行度上限 | 受 head 数限制（GQA/MLA 下更紧） | 受序列长度限制，与 head 数无关 |
| KV Cache | 按 head 切分，head 数少时会复制 | 按 T 切分，显存占用随 CP 度下降 |
| 激活显存 | 按 head 切，长序列仍线性增长 | 按序列切，长序列显存近似除以 $C$ |

CP 与 TP / PP / DP / EP 正交，可以组合成 4D/5D 混合并行。实践中的经验顺序是先用 Megatron-SP（TP 内的序列并行，几乎零额外开销），激活显存仍然不够时再叠加 CP。

因果注意力下如果按 rank 顺序连续切分，前段 rank 历史 KV 少、计算量小，后段 rank 历史 KV 多、计算量大，约有 2 倍的不均衡，慢的 rank 决定整体速度。通用解法是 **zigzag（首尾配对）切分**。序列切成 $2C$ 个块，rank $r$ 拿块 $r$ 和块 $2C-1-r$，即一个轻载前段块配一个重载后段块，各 rank 计算量趋衡。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 550" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="550" fill="#ffffff"/>
<text x="340" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">zigzag 切分：切分矩阵与因果 mask 矩阵（C=4，序列切成 2C=8 块）</text>
<rect x="16" y="40" width="648" height="230" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="32" y="62" font-size="13" font-weight="bold" fill="#334155">① 顺序切分：rank r 拿块 2r、2r+1 —— 后段 rank 越背越重</text>
<text x="32" y="92" font-size="11" fill="#475569">切分矩阵</text>
<text x="32" y="108" font-size="10" fill="#94a3b8">query 块 → rank</text>
<g font-size="9" text-anchor="middle">
<rect x="200" y="82" width="17" height="17" fill="#93c5fd" stroke="#2563eb"/><text x="208.5" y="94" fill="#0f172a">B0</text>
<rect x="218" y="82" width="17" height="17" fill="#93c5fd" stroke="#2563eb"/><text x="226.5" y="94" fill="#0f172a">B1</text>
<rect x="236" y="82" width="17" height="17" fill="#86efac" stroke="#16a34a"/><text x="244.5" y="94" fill="#0f172a">B2</text>
<rect x="254" y="82" width="17" height="17" fill="#86efac" stroke="#16a34a"/><text x="262.5" y="94" fill="#0f172a">B3</text>
<rect x="272" y="82" width="17" height="17" fill="#fcd34d" stroke="#d97706"/><text x="280.5" y="94" fill="#0f172a">B4</text>
<rect x="290" y="82" width="17" height="17" fill="#fcd34d" stroke="#d97706"/><text x="298.5" y="94" fill="#0f172a">B5</text>
<rect x="308" y="82" width="17" height="17" fill="#fca5a5" stroke="#dc2626"/><text x="316.5" y="94" fill="#0f172a">B6</text>
<rect x="326" y="82" width="17" height="17" fill="#fca5a5" stroke="#dc2626"/><text x="334.5" y="94" fill="#0f172a">B7</text>
</g>
<text x="352" y="94" font-size="9" fill="#94a3b8">← 同时是 mask 的列（key 块）</text>
<text x="32" y="170" font-size="11" fill="#475569">因果 mask 矩阵</text>
<text x="32" y="186" font-size="10" fill="#94a3b8">行 = query 块</text>
<text x="32" y="200" font-size="10" fill="#94a3b8">着色 = 参与计算</text>
<text x="32" y="214" font-size="10" fill="#94a3b8">灰 = 因果屏蔽</text>
<text x="32" y="232" font-size="10" fill="#94a3b8">行着色 = 该行归属 rank</text>
<g stroke="#ffffff" stroke-width="1">
<rect x="200" y="112" width="18" height="18" fill="#bfdbfe"/><rect x="218" y="112" width="126" height="18" fill="#e2e8f0"/>
<rect x="200" y="130" width="36" height="18" fill="#bfdbfe"/><rect x="236" y="130" width="108" height="18" fill="#e2e8f0"/>
<rect x="200" y="148" width="54" height="18" fill="#bbf7d0"/><rect x="254" y="148" width="90" height="18" fill="#e2e8f0"/>
<rect x="200" y="166" width="72" height="18" fill="#bbf7d0"/><rect x="272" y="166" width="72" height="18" fill="#e2e8f0"/>
<rect x="200" y="184" width="90" height="18" fill="#fde68a"/><rect x="290" y="184" width="54" height="18" fill="#e2e8f0"/>
<rect x="200" y="202" width="108" height="18" fill="#fde68a"/><rect x="308" y="202" width="36" height="18" fill="#e2e8f0"/>
<rect x="200" y="220" width="126" height="18" fill="#fecaca"/><rect x="326" y="220" width="18" height="18" fill="#e2e8f0"/>
<rect x="200" y="238" width="144" height="18" fill="#fecaca"/>
</g>
<g font-size="9" text-anchor="end" fill="#64748b">
<text x="196" y="125">B0</text><text x="196" y="143">B1</text><text x="196" y="161">B2</text><text x="196" y="179">B3</text>
<text x="196" y="197">B4</text><text x="196" y="215">B5</text><text x="196" y="233">B6</text><text x="196" y="251">B7</text>
</g>
<text x="430" y="118" font-size="11" fill="#475569">各 rank 负载（着色格数）</text>
<g font-size="10">
<text x="442" y="148" text-anchor="end" fill="#334155">R0</text>
<rect x="450" y="136" width="34" height="16" fill="#93c5fd" stroke="#2563eb"/><text x="490" y="148" fill="#334155">3</text>
<text x="442" y="174" text-anchor="end" fill="#334155">R1</text>
<rect x="450" y="162" width="79" height="16" fill="#86efac" stroke="#16a34a"/><text x="535" y="174" fill="#334155">7</text>
<text x="442" y="200" text-anchor="end" fill="#334155">R2</text>
<rect x="450" y="188" width="125" height="16" fill="#fcd34d" stroke="#d97706"/><text x="581" y="200" fill="#334155">11</text>
<text x="442" y="226" text-anchor="end" fill="#334155">R3</text>
<rect x="450" y="214" width="170" height="16" fill="#fca5a5" stroke="#dc2626"/><text x="626" y="226" fill="#334155">15</text>
</g>
<text x="340" y="262" text-anchor="middle" font-size="10" fill="#64748b">rank 负载 = 其拥有的 query 行在下三角中的着色格数（∝ 计算量）：3 / 7 / 11 / 15，约 5 倍不均衡</text>
<rect x="16" y="280" width="648" height="230" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="32" y="302" font-size="13" font-weight="bold" fill="#334155">② zigzag：rank r 拿块 r 与块 2C−1−r —— 首尾配对，负载相等</text>
<text x="32" y="332" font-size="11" fill="#475569">切分矩阵</text>
<text x="32" y="348" font-size="10" fill="#94a3b8">query 块 → rank</text>
<g font-size="9" text-anchor="middle">
<rect x="200" y="322" width="17" height="17" fill="#93c5fd" stroke="#2563eb"/><text x="208.5" y="334" fill="#0f172a">B0</text>
<rect x="218" y="322" width="17" height="17" fill="#86efac" stroke="#16a34a"/><text x="226.5" y="334" fill="#0f172a">B1</text>
<rect x="236" y="322" width="17" height="17" fill="#fcd34d" stroke="#d97706"/><text x="244.5" y="334" fill="#0f172a">B2</text>
<rect x="254" y="322" width="17" height="17" fill="#fca5a5" stroke="#dc2626"/><text x="262.5" y="334" fill="#0f172a">B3</text>
<rect x="272" y="322" width="17" height="17" fill="#fca5a5" stroke="#dc2626"/><text x="280.5" y="334" fill="#0f172a">B4</text>
<rect x="290" y="322" width="17" height="17" fill="#fcd34d" stroke="#d97706"/><text x="298.5" y="334" fill="#0f172a">B5</text>
<rect x="308" y="322" width="17" height="17" fill="#86efac" stroke="#16a34a"/><text x="316.5" y="334" fill="#0f172a">B6</text>
<rect x="326" y="322" width="17" height="17" fill="#93c5fd" stroke="#2563eb"/><text x="334.5" y="334" fill="#0f172a">B7</text>
</g>
<text x="352" y="334" font-size="9" fill="#94a3b8">R0←B0+B7，R1←B1+B6，R2←B2+B5，R3←B3+B4</text>
<text x="32" y="410" font-size="11" fill="#475569">因果 mask 矩阵</text>
<text x="32" y="426" font-size="10" fill="#94a3b8">行 = query 块</text>
<text x="32" y="440" font-size="10" fill="#94a3b8">着色 = 参与计算</text>
<text x="32" y="454" font-size="10" fill="#94a3b8">灰 = 因果屏蔽</text>
<g stroke="#ffffff" stroke-width="1">
<rect x="200" y="352" width="18" height="18" fill="#bfdbfe"/><rect x="218" y="352" width="126" height="18" fill="#e2e8f0"/>
<rect x="200" y="370" width="36" height="18" fill="#bbf7d0"/><rect x="236" y="370" width="108" height="18" fill="#e2e8f0"/>
<rect x="200" y="388" width="54" height="18" fill="#fde68a"/><rect x="254" y="388" width="90" height="18" fill="#e2e8f0"/>
<rect x="200" y="406" width="72" height="18" fill="#fecaca"/><rect x="272" y="406" width="72" height="18" fill="#e2e8f0"/>
<rect x="200" y="424" width="90" height="18" fill="#fecaca"/><rect x="290" y="424" width="54" height="18" fill="#e2e8f0"/>
<rect x="200" y="442" width="108" height="18" fill="#fde68a"/><rect x="308" y="442" width="36" height="18" fill="#e2e8f0"/>
<rect x="200" y="460" width="126" height="18" fill="#bbf7d0"/><rect x="326" y="460" width="18" height="18" fill="#e2e8f0"/>
<rect x="200" y="478" width="144" height="18" fill="#bfdbfe"/>
</g>
<g font-size="9" text-anchor="end" fill="#64748b">
<text x="196" y="365">B0</text><text x="196" y="383">B1</text><text x="196" y="401">B2</text><text x="196" y="419">B3</text>
<text x="196" y="437">B4</text><text x="196" y="455">B5</text><text x="196" y="473">B6</text><text x="196" y="491">B7</text>
</g>
<text x="430" y="358" font-size="11" fill="#475569">各 rank 负载（着色格数）</text>
<g font-size="10">
<text x="442" y="388" text-anchor="end" fill="#334155">R0</text>
<rect x="450" y="376" width="102" height="16" fill="#93c5fd" stroke="#2563eb"/><text x="558" y="388" fill="#334155">9</text>
<text x="442" y="414" text-anchor="end" fill="#334155">R1</text>
<rect x="450" y="402" width="102" height="16" fill="#86efac" stroke="#16a34a"/><text x="558" y="414" fill="#334155">9</text>
<text x="442" y="440" text-anchor="end" fill="#334155">R2</text>
<rect x="450" y="428" width="102" height="16" fill="#fcd34d" stroke="#d97706"/><text x="558" y="440" fill="#334155">9</text>
<text x="442" y="466" text-anchor="end" fill="#334155">R3</text>
<rect x="450" y="454" width="102" height="16" fill="#fca5a5" stroke="#dc2626"/><text x="558" y="466" fill="#334155">9</text>
</g>
<text x="340" y="502" text-anchor="middle" font-size="10" fill="#64748b">轻载前段块（行短）配重载后段块（行长），每个 rank 的着色格数均为 9，慢 rank 不再拖累整体</text>
</svg>

▲ 切分矩阵与因果 mask 矩阵：切分矩阵给出每个 query 块归属的 rank；mask 矩阵行为 query 块、列为 key 块，下三角（着色）参与计算、上三角（灰）被因果 mask 屏蔽。顺序切分下负载 3/7/11/15，zigzag 首尾配对后各 rank 均为 9

## 2.1 Prefill CP

Prefill 阶段是算力受限（compute-bound）的，一个长请求有 $T$ 个新 token 要计算 Q/K/V 并做完整的因果注意力。Prefill CP 的目标是**摊薄 TTFT**（Time-To-First-Token），把请求切成 $N$ 个 chunk，每个 GPU 计算一个 chunk 的 Q/K/V 及该 chunk query 对应的注意力输出。按 KV 的处理方式分两种策略。

1. **Partial Query + Full KV（AllGather KV）**：请求长度适中、能放得下完整 KV 时，各 rank AllGather 得到完整 KV，每个 GPU 只为自己那部分 query 计算注意力输出。实现最简单，无需改注意力 kernel，是 SGLang Prefill CP、vLLM PCP、DSA-CP 的主流做法；代价是每层需要临时放得下全局 KV（可逐层聚合、用完即弃以压峰值）。现代架构（MLA / DSA / CSA）把 KV 压得很小，AllGather 的代价可接受。
2. **Partial Query + Partial KV（Ring Attention）**：请求太长、全局 KV 放不下时，KV 块沿环 P2P 传/收，每个 rank 用本地 Q 对收到的 KV 块做部分注意力，用在线 softmax（LSE）合并结果。通信可与计算重叠，但实现复杂度和 kernel 侵入性更高。

工程要点：

- **头尾配对切分**（如 vLLM PCP 的 DualChunkSwap、SGLang Zigzag 策略、Megatron CP）：序列填充到 $2C$ 的倍数后切成 $2C$ 块，rank $i$ 拿 chunk $i$ 与 chunk $2C-1-i$，并通过 restore 索引把 AllGather 后交错的 token 顺序还原回因果顺序。
- 对带 **Indexer 的稀疏注意力**（DSA、DeepSeek V4 系列），Indexer 的 top-k 选择计算也随 query 切分到各 rank，避免重复计算——这本身也是 Prefill CP 的收益之一。
- 与 EP 组合时，CP 可以用 all-to-all 的 token dispatch/combine 替代 TP 的 AllReduce，降低通信时间。
- 收益参考：SGLang 合入 CP 后长文本 TTFT 最多降低约 80%（PR #12065）；Moreh 的 SLOPE 引擎在 100K 上下文做到 TTFT < 2s，而同配置 SGLang TP=8 约 9s。

## 2.2 Decode CP

Decode 阶段是访存受限（memory-bound）的。每步只有少量 query token，却要扫描大量存在 paged KV cache 里的历史 KV。Decode CP 的目标是**把 KV Cache 按序列维切开，腾出显存放更多请求（更大 batch），从而提高吞吐**。

先看 KV Cache 的分片路径。模型有 $H$ 个 KV head 时，TP 可以先按 head 维切 KV（就是普通的 TP 分片）；但 $H$ 由模型结构决定，当 `tp_size > H` 时每个 GPU 上的 KV Cache 会**复制** `tp_size / H` 份。以 MLA 模型（DeepSeek-R1 / Kimi-K2，推理时只有 1 个 KV head）为例，tp=8 就是 8 倍 KV 复制。此时就需要 DCP 再沿 **T 维**切 KV Cache。

- `dcp_size` 取值范围 $[1, tp\_size/H]$，DCP 不增加进程数，只是复用 TP/PCP 的 rank 去掉 KV 复制。`dcp_size` 越大复制越少，但通信开销越大。
- **交错（interleave）分片**：decode 过程中 KV Cache 持续增长，vLLM 用交错策略沿 T 维分片——token $i$ 存到 `dcp_rank (i % dcp_world_size)`（`cp_kv_cache_interleave_size=1`，token 级对齐），或按 block 级对齐（先填满前序 rank 的 block 再写下一个 rank），使未来 token 的 KV 天然落到正确的 GPU 上，各 rank 显存均匀增长。该方案由 Moonshot 的 Chao Hong 提出，Helix 论文（arXiv:2507.07120）中有详细说明。
- 通信方式与 prefill 相反，decode 的 Q 极小、KV 极大，所以走**传 Q**——把 Q 广播/AllGather 到各 rank，每个 rank 对本地 KV 分片算 partial attention 输出和 softmax 统计量（LSE），再做一次很小的合并（在线 softmax trick，通信量与上下文长度无关）。vLLM DCP 提供 `ag_rs`（AllGather + ReduceScatter）与 `a2a`（交换 partial output + LSE 后用 Triton kernel 合并）两种通信后端。

案例（vLLM 官方文档）：

- **DeepSeek-R1**（MLA，1 个 KV head）：单机 tp=8 → 8 倍 KV 复制，加 `-dcp 8` 可完全消除复制。
- **Kimi-K2**：tp=16 → 16 倍复制，可加 `-dcp 16` 完全消除，或 `-dcp 8` 把复制降到 2 倍（通信限制在节点内，开销更小）。
- **Qwen3-235B-A22B**（4 个 KV head）：tp=8 → 2 倍复制，加 `-dcp 2` 消除。

去掉 KV 复制之后 batch size 可以显著放大，Moonshot 在 Kimi K2 分享中给出的经验是 2~3 倍吞吐提升，对 RL rollout 这类吞吐敏感场景尤其重要。

# 3. 不同的CP通信方案

针对CP并行，有多种不同的通信实现方案，常见的有传递Q、传递KV，或者Ulysses的All-to-All（常被混称为 AllGather）。下面分别介绍。

## 3.1 传 KV：Ring Attention（P2P 环）

Q 留在本地，KV 块沿环形拓扑 P2P send/recv 传一圈。第 $t$ 轮每个 rank 用手里的 KV 块算本地 Q 的部分注意力结果，同时把该块发给下一个 rank、从上一个 rank 收新块，用 FlashAttention 的 LSE 累加技巧合并，最终结果与单卡稠密注意力等价。只要单块计算时间能盖住单块传输时间，通信可以完全隐藏。这是 Megatron `cp_comm_type=p2p` 的默认实现（工业级 Ring Attention，即 zigzag 切分 + 双 CUDA stream 重叠 + FA2 集成）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 430" style="width:100%;height:auto">
<defs>
<marker id="ringA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#ea580c"/></marker>
</defs>
<rect x="0" y="0" width="680" height="430" fill="#ffffff"/>
<text x="340" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">Ring Attention 每轮通信日程（C=4）：KV 块顺时针 P2P，一轮一跳</text>
<rect x="23" y="36" width="142" height="132" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="94" y="54" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">第 0 轮（初始）</text>
<g font-size="9" text-anchor="middle">
<rect x="33" y="62" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="60" y="75" fill="#1e3a8a">GPU 0</text>
<rect x="93" y="62" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="119" y="75" fill="#7c2d12">K₀</text>
<rect x="33" y="84" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="60" y="97" fill="#14532d">GPU 1</text>
<rect x="93" y="84" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="119" y="97" fill="#7c2d12">K₁</text>
<rect x="33" y="106" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="60" y="119" fill="#92400e">GPU 2</text>
<rect x="93" y="106" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="119" y="119" fill="#7c2d12">K₂</text>
<rect x="33" y="128" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="60" y="141" fill="#7f1d1d">GPU 3</text>
<rect x="93" y="128" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="119" y="141" fill="#7c2d12">K₃</text>
</g>
<text x="94" y="162" text-anchor="middle" font-size="8" fill="#64748b">本轮算 tile：i = j（对角）</text>
<rect x="187" y="36" width="142" height="132" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="258" y="54" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">第 1 轮</text>
<g font-size="9" text-anchor="middle">
<rect x="197" y="62" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="224" y="75" fill="#1e3a8a">GPU 0</text>
<rect x="257" y="62" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="283" y="75" fill="#7c2d12">K₃</text>
<rect x="197" y="84" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="224" y="97" fill="#14532d">GPU 1</text>
<rect x="257" y="84" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="283" y="97" fill="#7c2d12">K₀</text>
<rect x="197" y="106" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="224" y="119" fill="#92400e">GPU 2</text>
<rect x="257" y="106" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="283" y="119" fill="#7c2d12">K₁</text>
<rect x="197" y="128" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="224" y="141" fill="#7f1d1d">GPU 3</text>
<rect x="257" y="128" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="283" y="141" fill="#7c2d12">K₂</text>
</g>
<text x="258" y="162" text-anchor="middle" font-size="8" fill="#64748b">本轮算 tile：i − j = 1</text>
<rect x="351" y="36" width="142" height="132" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="422" y="54" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">第 2 轮</text>
<g font-size="9" text-anchor="middle">
<rect x="361" y="62" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="388" y="75" fill="#1e3a8a">GPU 0</text>
<rect x="421" y="62" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="447" y="75" fill="#7c2d12">K₂</text>
<rect x="361" y="84" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="388" y="97" fill="#14532d">GPU 1</text>
<rect x="421" y="84" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="447" y="97" fill="#7c2d12">K₃</text>
<rect x="361" y="106" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="388" y="119" fill="#92400e">GPU 2</text>
<rect x="421" y="106" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="447" y="119" fill="#7c2d12">K₀</text>
<rect x="361" y="128" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="388" y="141" fill="#7f1d1d">GPU 3</text>
<rect x="421" y="128" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="447" y="141" fill="#7c2d12">K₁</text>
</g>
<text x="422" y="162" text-anchor="middle" font-size="8" fill="#64748b">本轮算 tile：i − j = 2</text>
<rect x="515" y="36" width="142" height="132" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="586" y="54" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">第 3 轮</text>
<g font-size="9" text-anchor="middle">
<rect x="525" y="62" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="552" y="75" fill="#1e3a8a">GPU 0</text>
<rect x="585" y="62" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="611" y="75" fill="#7c2d12">K₁</text>
<rect x="525" y="84" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="552" y="97" fill="#14532d">GPU 1</text>
<rect x="585" y="84" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="611" y="97" fill="#7c2d12">K₂</text>
<rect x="525" y="106" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="552" y="119" fill="#92400e">GPU 3</text>
<rect x="585" y="106" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="611" y="119" fill="#7c2d12">K₀</text>
<rect x="525" y="128" width="54" height="18" fill="#e0e7ff" stroke="#6366f1"/><text x="552" y="141" fill="#7f1d1d">GPU 3</text>
<rect x="585" y="128" width="52" height="18" fill="#fdba74" stroke="#ea580c"/><text x="611" y="141" fill="#7c2d12">K₃</text>
</g>
<text x="586" y="162" text-anchor="middle" font-size="8" fill="#64748b">本轮算 tile：i − j = 3</text>
<g stroke="#ea580c" stroke-width="2" fill="none">
<line x1="167" y1="100" x2="183" y2="100" marker-end="url(#ringA)"/>
<line x1="331" y1="100" x2="347" y2="100" marker-end="url(#ringA)"/>
<line x1="495" y1="100" x2="511" y2="100" marker-end="url(#ringA)"/>
</g>
<text x="340" y="186" text-anchor="middle" font-size="9" fill="#c2410c">↑ 每轮的通信：4 对 P2P send/recv 同时进行（各 GPU 把手里的 K 块发给下家、从上家收新块），与本轮 tile 计算重叠</text>
<text x="120" y="212" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">score tile 矩阵：颜色 = 计算发生的轮次</text>
<g font-size="10" text-anchor="middle" fill="#64748b">
<text x="140" y="230">K₀</text><text x="180" y="230">K₁</text><text x="220" y="230">K₂</text><text x="260" y="230">K₃</text>
<text x="112" y="255" text-anchor="end" fill="#1d4ed8">Q₀</text><text x="112" y="295" text-anchor="end" fill="#15803d">Q₁</text>
<text x="112" y="335" text-anchor="end" fill="#b45309">Q₂</text><text x="112" y="375" text-anchor="end" fill="#b91c1c">Q₃</text>
</g>
<g stroke="#ffffff" stroke-width="2">
<rect x="120" y="235" width="40" height="40" fill="#86efac"/><rect x="160" y="235" width="40" height="40" fill="#e2e8f0"/><rect x="200" y="235" width="40" height="40" fill="#e2e8f0"/><rect x="240" y="235" width="40" height="40" fill="#e2e8f0"/>
<rect x="120" y="275" width="40" height="40" fill="#93c5fd"/><rect x="160" y="275" width="40" height="40" fill="#86efac"/><rect x="200" y="275" width="40" height="40" fill="#e2e8f0"/><rect x="240" y="275" width="40" height="40" fill="#e2e8f0"/>
<rect x="120" y="315" width="40" height="40" fill="#fcd34d"/><rect x="160" y="315" width="40" height="40" fill="#93c5fd"/><rect x="200" y="315" width="40" height="40" fill="#86efac"/><rect x="240" y="315" width="40" height="40" fill="#e2e8f0"/>
<rect x="120" y="355" width="40" height="40" fill="#fca5a5"/><rect x="160" y="355" width="40" height="40" fill="#fcd34d"/><rect x="200" y="355" width="40" height="40" fill="#93c5fd"/><rect x="240" y="355" width="40" height="40" fill="#86efac"/>
</g>
<g font-size="9" text-anchor="middle" fill="#334155">
<text x="140" y="259">轮0</text><text x="140" y="299">轮1</text><text x="140" y="339">轮2</text><text x="140" y="379">轮3</text>
<text x="180" y="299">轮0</text><text x="180" y="339">轮1</text><text x="180" y="379">轮2</text>
<text x="220" y="339">轮0</text><text x="220" y="379">轮1</text>
<text x="260" y="379">轮0</text>
<text x="180" y="259" fill="#94a3b8">mask</text><text x="220" y="259" fill="#94a3b8">mask</text><text x="260" y="259" fill="#94a3b8">mask</text>
<text x="220" y="299" fill="#94a3b8">mask</text><text x="260" y="299" fill="#94a3b8">mask</text>
<text x="260" y="339" fill="#94a3b8">mask</text>
</g>
<g font-size="9">
<rect x="310" y="238" width="12" height="12" fill="#86efac" stroke="#16a34a"/><text x="330" y="248" fill="#334155">轮 0：每 rank 用起始 K 块算对角 tile（i=j）</text>
<rect x="310" y="262" width="12" height="12" fill="#93c5fd" stroke="#2563eb"/><text x="330" y="272" fill="#334155">轮 1：块移一跳后算次对角（i−j=1）</text>
<rect x="310" y="286" width="12" height="12" fill="#fcd34d" stroke="#d97706"/><text x="330" y="296" fill="#334155">轮 2：算 i−j=2</text>
<rect x="310" y="310" width="12" height="12" fill="#fca5a5" stroke="#dc2626"/><text x="330" y="320" fill="#334155">轮 3：算 i−j=3，下三角全部算完</text>
</g>
<g font-size="9" fill="#64748b">
<text x="310" y="348">每轮每 rank：① 用手里的 K 块算 1 个 tile → partial (O, LSE)</text>
<text x="310" y="364">② P2P 发块、收新块（双 CUDA stream 与 ① 重叠）</text>
<text x="310" y="380">灰 = causal mask 的 tile（j&gt;i），不算；转满 C 轮后 LSE 合并 = 精确注意力</text>
</g>
<rect x="40" y="400" width="600" height="22" rx="6" fill="#f1f5f9" stroke="#e2e8f0"/>
<text x="340" y="415" text-anchor="middle" font-size="10" fill="#475569">通信量：每轮每链路 1 个 KV 块，共 C 轮 = KV 沿环走一遍 O(T)；zigzag 切分只改行归属，轮转日程不变</text>
</svg>

▲ Ring Attention 每轮通信日程：上排 = 4 轮中各 GPU 持有的 K 块与每轮 P2P 跳转；下排 = score tile 矩阵（颜色 = 该 tile 被计算的轮次，灰 = 因果 mask）

- 优点：CP 度不受 head 数限制，通信量 $O(T)$（KV 沿环走一遍）。
- 缺点：多轮流通信 + 结果合并，kernel 需要支持 partial output/LSE 合并；小块时 kernel 效率下降。

## 3.2 传 KV：AllGather KV

一次性 AllGather 完整 KV，本地 Q 对全局 KV 用标准注意力 kernel 算完即可。实现最简单、不改 kernel、不受 head 数限制，但需要临时放得下整层全局 KV，且全量 KV 要在 CP 组内传一遍。由于 MLA/DSA/CSA 等架构把 KV 压缩得非常小（MLA 传 latent，CSA 传压缩条目），"先 AllGather KV 再算"反而常常是又快又好写的选择。SGLang Prefill CP、vLLM PCP、昇腾 DSA-CP 都采用该方案。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 395" style="width:100%;height:auto">
<defs>
<marker id="m32n" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#ea580c"/></marker>
</defs>
<rect x="0" y="0" width="680" height="395" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">AllGather KV：通信前每 rank 只有本地 KV 分片，通信后每 rank 补齐全局 KV</text>
<text x="160" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">通信前（虚线格 = 缺失的 KV 块）</text>
<text x="520" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">AllGather 后（颜色深浅 = 来源 rank）</text>
<g>
<rect x="20" y="58" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="30" y="73" font-size="9.5" font-weight="bold" fill="#1d4ed8">GPU 0</text>
<rect x="30" y="80" width="40" height="28" fill="#93c5fd" stroke="#1d4ed8" stroke-width="1.2"/>
<text x="50" y="98" text-anchor="middle" font-size="9.5" fill="#1e3a8a">Q₀</text>
<rect x="76" y="80" width="50" height="28" fill="#fed7aa" stroke="#ea580c" stroke-width="1"/>
<text x="101" y="98" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₀</text>
<rect x="132" y="80" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="188" y="80" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="244" y="80" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
</g>
<g>
<rect x="20" y="124" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="30" y="139" font-size="9.5" font-weight="bold" fill="#15803d">GPU 1</text>
<rect x="30" y="146" width="40" height="28" fill="#86efac" stroke="#15803d" stroke-width="1.2"/>
<text x="50" y="164" text-anchor="middle" font-size="9.5" fill="#14532d">Q₁</text>
<rect x="76" y="146" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="132" y="146" width="50" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1"/>
<text x="157" y="164" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₁</text>
<rect x="188" y="146" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="244" y="146" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
</g>
<g>
<rect x="20" y="190" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="30" y="205" font-size="9.5" font-weight="bold" fill="#b45309">GPU 2</text>
<rect x="30" y="212" width="40" height="28" fill="#fcd34d" stroke="#b45309" stroke-width="1.2"/>
<text x="50" y="230" text-anchor="middle" font-size="9.5" fill="#78350f">Q₂</text>
<rect x="76" y="212" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="132" y="212" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="188" y="212" width="50" height="28" fill="#fb923c" stroke="#ea580c" stroke-width="1"/>
<text x="213" y="230" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₂</text>
<rect x="244" y="212" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
</g>
<g>
<rect x="20" y="256" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="30" y="271" font-size="9.5" font-weight="bold" fill="#b91c1c">GPU 3</text>
<rect x="30" y="278" width="40" height="28" fill="#fca5a5" stroke="#b91c1c" stroke-width="1.2"/>
<text x="50" y="296" text-anchor="middle" font-size="9.5" fill="#7f1d1d">Q₃</text>
<rect x="76" y="278" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="132" y="278" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="188" y="278" width="50" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="244" y="278" width="50" height="28" fill="#f97316" stroke="#ea580c" stroke-width="1"/>
<text x="269" y="296" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₃</text>
</g>
<g>
<rect x="380" y="58" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="390" y="73" font-size="9.5" font-weight="bold" fill="#1d4ed8">GPU 0</text>
<rect x="390" y="80" width="40" height="28" fill="#93c5fd" stroke="#1d4ed8" stroke-width="1.2"/>
<text x="410" y="98" text-anchor="middle" font-size="9.5" fill="#1e3a8a">Q₀</text>
<rect x="436" y="80" width="50" height="28" fill="#fed7aa" stroke="#ea580c" stroke-width="1"/>
<text x="461" y="98" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₀</text>
<rect x="492" y="80" width="50" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1"/>
<text x="517" y="98" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₁</text>
<rect x="548" y="80" width="50" height="28" fill="#fb923c" stroke="#ea580c" stroke-width="1"/>
<text x="573" y="98" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₂</text>
<rect x="604" y="80" width="50" height="28" fill="#f97316" stroke="#ea580c" stroke-width="1"/>
<text x="629" y="98" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₃</text>
</g>
<g>
<rect x="380" y="124" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="390" y="139" font-size="9.5" font-weight="bold" fill="#15803d">GPU 1</text>
<rect x="390" y="146" width="40" height="28" fill="#86efac" stroke="#15803d" stroke-width="1.2"/>
<text x="410" y="164" text-anchor="middle" font-size="9.5" fill="#14532d">Q₁</text>
<rect x="436" y="146" width="50" height="28" fill="#fed7aa" stroke="#ea580c" stroke-width="1"/>
<text x="461" y="164" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₀</text>
<rect x="492" y="146" width="50" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1"/>
<text x="517" y="164" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₁</text>
<rect x="548" y="146" width="50" height="28" fill="#fb923c" stroke="#ea580c" stroke-width="1"/>
<text x="573" y="164" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₂</text>
<rect x="604" y="146" width="50" height="28" fill="#f97316" stroke="#ea580c" stroke-width="1"/>
<text x="629" y="164" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₃</text>
</g>
<g>
<rect x="380" y="190" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="390" y="205" font-size="9.5" font-weight="bold" fill="#b45309">GPU 2</text>
<rect x="390" y="212" width="40" height="28" fill="#fcd34d" stroke="#b45309" stroke-width="1.2"/>
<text x="410" y="230" text-anchor="middle" font-size="9.5" fill="#78350f">Q₂</text>
<rect x="436" y="212" width="50" height="28" fill="#fed7aa" stroke="#ea580c" stroke-width="1"/>
<text x="461" y="230" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₀</text>
<rect x="492" y="212" width="50" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1"/>
<text x="517" y="230" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₁</text>
<rect x="548" y="212" width="50" height="28" fill="#fb923c" stroke="#ea580c" stroke-width="1"/>
<text x="573" y="230" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₂</text>
<rect x="604" y="212" width="50" height="28" fill="#f97316" stroke="#ea580c" stroke-width="1"/>
<text x="629" y="230" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₃</text>
</g>
<g>
<rect x="380" y="256" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="390" y="271" font-size="9.5" font-weight="bold" fill="#b91c1c">GPU 3</text>
<rect x="390" y="278" width="40" height="28" fill="#fca5a5" stroke="#b91c1c" stroke-width="1.2"/>
<text x="410" y="296" text-anchor="middle" font-size="9.5" fill="#7f1d1d">Q₃</text>
<rect x="436" y="278" width="50" height="28" fill="#fed7aa" stroke="#ea580c" stroke-width="1"/>
<text x="461" y="296" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₀</text>
<rect x="492" y="278" width="50" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1"/>
<text x="517" y="296" text-anchor="middle" font-size="9.5" fill="#7c2d12">KV₁</text>
<rect x="548" y="278" width="50" height="28" fill="#fb923c" stroke="#ea580c" stroke-width="1"/>
<text x="573" y="296" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₂</text>
<rect x="604" y="278" width="50" height="28" fill="#f97316" stroke="#ea580c" stroke-width="1"/>
<text x="629" y="296" text-anchor="middle" font-size="9.5" fill="#ffffff">KV₃</text>
</g>
<line x1="306" y1="185" x2="374" y2="185" stroke="#ea580c" stroke-width="2.5" marker-end="url(#m32n)"/>
<text x="340" y="160" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#c2410c">AllGather KV</text>
<text x="340" y="212" text-anchor="middle" font-size="9" fill="#c2410c">沿 T 维聚齐</text>
<text x="340" y="336" text-anchor="middle" font-size="11" fill="#334155">通信后：每 rank 用本地 Q 行块对全局 KV 做标准 FlashAttention，整行块一次算完，无需 LSE 合并</text>
<rect x="60" y="350" width="560" height="28" rx="6" fill="#f1f5f9" stroke="#e2e8f0"/>
<text x="340" y="368" text-anchor="middle" font-size="10.5" fill="#475569">KV 占用 T/C·d → T·d（逐层聚合、用完即弃）；通信量 = 全量 KV 在 CP 组内传一遍；不改 kernel，CP 度不受 head 数限制</text>
</svg>

▲ AllGather KV 示意图：通信前每 rank 只持有自己的 KV 分片（虚线格为缺失块），AllGather 后每 rank 补齐全局 KV 矩阵（颜色深浅区分来源 rank），Q 行块始终不动

## 3.3 传 Q：AllGather Q / Pass Q

KV 按 T 维分片留在本地不动，把（很小的）Q 送达各 rank，每个 rank 对本地 KV 分片算 partial output $O_j$ 和 LSE $L_j$，再合并。通信量与 KV Cache 大小**无关**，因此是 decode 阶段的首选。按 Q 的送达方式分为两种形态。

### 3.3.1 AllGather / Broadcast Q

一次性集合通信把 Q 聚齐到每个 rank。DCP 组与 TP 组重叠时各 rank 只持有自己那部分 head 的 Q 分片，需要先沿 head 维 AllGather 得到完整 Q；DCP 组独立时 Q 本来就在各 rank 复制，一次 broadcast 即可。之后各 rank 用完整 Q 对本地 KV 分片算 score 列段 $S_j$ 并得到 $(O_j, L_j)$，再做一次 LSE 感知的归并（all-reduce 或 gather，如 vLLM 的 `cp_lse_ag_out_rs`）。全程两个集合通信拍、延迟低，是 SGLang Decode CP（all-gather Q）、vLLM DCP、Helix 的 KV Parallelism 的共同选择。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 400" style="width:100%;height:auto">
<defs>
<marker id="m33a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2563eb"/></marker>
<marker id="m33ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#16a34a"/></marker>
</defs>
<rect x="0" y="0" width="680" height="400" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">AllGather / Broadcast Q：通信前 Q 按 head 维分片，通信后每 rank 持有完整 Q</text>
<text x="160" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">通信前（虚线格 = 缺失的 head 分片）</text>
<text x="520" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">AllGather Q 后（颜色深浅 = 来源 rank）</text>
<g>
<rect x="20" y="58" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="28" y="73" font-size="9.5" font-weight="bold" fill="#1d4ed8">GPU 0</text>
<rect x="28" y="80" width="44" height="28" fill="#bfdbfe" stroke="#2563eb" stroke-width="1"/>
<text x="50" y="98" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₀</text>
<rect x="76" y="80" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="124" y="80" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="172" y="80" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="224" y="80" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="258" y="98" text-anchor="middle" font-size="9" fill="#7c2d12">KV₀ 常驻</text>
</g>
<g>
<rect x="20" y="124" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="28" y="139" font-size="9.5" font-weight="bold" fill="#15803d">GPU 1</text>
<rect x="28" y="146" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="76" y="146" width="44" height="28" fill="#93c5fd" stroke="#2563eb" stroke-width="1"/>
<text x="98" y="164" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₁</text>
<rect x="124" y="146" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="172" y="146" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="224" y="146" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="258" y="164" text-anchor="middle" font-size="9" fill="#7c2d12">KV₁ 常驻</text>
</g>
<g>
<rect x="20" y="190" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="28" y="205" font-size="9.5" font-weight="bold" fill="#b45309">GPU 2</text>
<rect x="28" y="212" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="76" y="212" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="124" y="212" width="44" height="28" fill="#60a5fa" stroke="#2563eb" stroke-width="1"/>
<text x="146" y="230" text-anchor="middle" font-size="9.5" fill="#ffffff">q₂</text>
<rect x="172" y="212" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="224" y="212" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="258" y="230" text-anchor="middle" font-size="9" fill="#7c2d12">KV₂ 常驻</text>
</g>
<g>
<rect x="20" y="256" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="28" y="271" font-size="9.5" font-weight="bold" fill="#b91c1c">GPU 3</text>
<rect x="28" y="278" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="76" y="278" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="124" y="278" width="44" height="28" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="172" y="278" width="44" height="28" fill="#3b82f6" stroke="#2563eb" stroke-width="1"/>
<text x="194" y="296" text-anchor="middle" font-size="9.5" fill="#ffffff">q₃</text>
<rect x="224" y="278" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="258" y="296" text-anchor="middle" font-size="9" fill="#7c2d12">KV₃ 常驻</text>
</g>
<g>
<rect x="380" y="58" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="388" y="73" font-size="9.5" font-weight="bold" fill="#1d4ed8">GPU 0</text>
<rect x="388" y="80" width="44" height="28" fill="#bfdbfe" stroke="#2563eb" stroke-width="1"/>
<text x="410" y="98" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₀</text>
<rect x="436" y="80" width="44" height="28" fill="#93c5fd" stroke="#2563eb" stroke-width="1"/>
<text x="458" y="98" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₁</text>
<rect x="484" y="80" width="44" height="28" fill="#60a5fa" stroke="#2563eb" stroke-width="1"/>
<text x="506" y="98" text-anchor="middle" font-size="9.5" fill="#ffffff">q₂</text>
<rect x="532" y="80" width="44" height="28" fill="#3b82f6" stroke="#2563eb" stroke-width="1"/>
<text x="554" y="98" text-anchor="middle" font-size="9.5" fill="#ffffff">q₃</text>
<rect x="584" y="80" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="618" y="98" text-anchor="middle" font-size="9" fill="#7c2d12">KV₀ 常驻</text>
</g>
<g>
<rect x="380" y="124" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="388" y="139" font-size="9.5" font-weight="bold" fill="#15803d">GPU 1</text>
<rect x="388" y="146" width="44" height="28" fill="#bfdbfe" stroke="#2563eb" stroke-width="1"/>
<text x="410" y="164" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₀</text>
<rect x="436" y="146" width="44" height="28" fill="#93c5fd" stroke="#2563eb" stroke-width="1"/>
<text x="458" y="164" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₁</text>
<rect x="484" y="146" width="44" height="28" fill="#60a5fa" stroke="#2563eb" stroke-width="1"/>
<text x="506" y="164" text-anchor="middle" font-size="9.5" fill="#ffffff">q₂</text>
<rect x="532" y="146" width="44" height="28" fill="#3b82f6" stroke="#2563eb" stroke-width="1"/>
<text x="554" y="164" text-anchor="middle" font-size="9.5" fill="#ffffff">q₃</text>
<rect x="584" y="146" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="618" y="164" text-anchor="middle" font-size="9" fill="#7c2d12">KV₁ 常驻</text>
</g>
<g>
<rect x="380" y="190" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="388" y="205" font-size="9.5" font-weight="bold" fill="#b45309">GPU 2</text>
<rect x="388" y="212" width="44" height="28" fill="#bfdbfe" stroke="#2563eb" stroke-width="1"/>
<text x="410" y="230" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₀</text>
<rect x="436" y="212" width="44" height="28" fill="#93c5fd" stroke="#2563eb" stroke-width="1"/>
<text x="458" y="230" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₁</text>
<rect x="484" y="212" width="44" height="28" fill="#60a5fa" stroke="#2563eb" stroke-width="1"/>
<text x="506" y="230" text-anchor="middle" font-size="9.5" fill="#ffffff">q₂</text>
<rect x="532" y="212" width="44" height="28" fill="#3b82f6" stroke="#2563eb" stroke-width="1"/>
<text x="554" y="230" text-anchor="middle" font-size="9.5" fill="#ffffff">q₃</text>
<rect x="584" y="212" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="618" y="230" text-anchor="middle" font-size="9" fill="#7c2d12">KV₂ 常驻</text>
</g>
<g>
<rect x="380" y="256" width="280" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="388" y="271" font-size="9.5" font-weight="bold" fill="#b91c1c">GPU 3</text>
<rect x="388" y="278" width="44" height="28" fill="#bfdbfe" stroke="#2563eb" stroke-width="1"/>
<text x="410" y="296" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₀</text>
<rect x="436" y="278" width="44" height="28" fill="#93c5fd" stroke="#2563eb" stroke-width="1"/>
<text x="458" y="296" text-anchor="middle" font-size="9.5" fill="#1e3a8a">q₁</text>
<rect x="484" y="278" width="44" height="28" fill="#60a5fa" stroke="#2563eb" stroke-width="1"/>
<text x="506" y="296" text-anchor="middle" font-size="9.5" fill="#ffffff">q₂</text>
<rect x="532" y="278" width="44" height="28" fill="#3b82f6" stroke="#2563eb" stroke-width="1"/>
<text x="554" y="296" text-anchor="middle" font-size="9.5" fill="#ffffff">q₃</text>
<rect x="584" y="278" width="68" height="28" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="618" y="296" text-anchor="middle" font-size="9" fill="#7c2d12">KV₃ 常驻</text>
</g>
<line x1="306" y1="185" x2="374" y2="185" stroke="#2563eb" stroke-width="2.5" marker-end="url(#m33a)"/>
<text x="340" y="160" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#1d4ed8">AllGather Q</text>
<text x="340" y="212" text-anchor="middle" font-size="9" fill="#1d4ed8">沿 head 维聚齐</text>
<text x="340" y="334" text-anchor="middle" font-size="11" fill="#334155">之后：各 rank 用完整 Q 对本地 KV 分片算 score 列段 S_j [q, T/C]，得到 (O_j, L_j)</text>
<g stroke="#ffffff" stroke-width="1.5">
<rect x="90" y="346" width="80" height="20" fill="#fed7aa"/><rect x="180" y="346" width="80" height="20" fill="#fdba74"/>
<rect x="270" y="346" width="80" height="20" fill="#fed7aa"/><rect x="360" y="346" width="80" height="20" fill="#fdba74"/>
</g>
<g font-size="9" text-anchor="middle" fill="#7c2d12">
<text x="130" y="359">S₀ (O₀,L₀)</text><text x="220" y="359">S₁ (O₁,L₁)</text><text x="310" y="359">S₂ (O₂,L₂)</text><text x="400" y="359">S₃ (O₃,L₃)</text>
</g>
<line x1="446" y1="356" x2="486" y2="356" stroke="#16a34a" stroke-width="2" marker-end="url(#m33ag)"/>
<rect x="492" y="346" width="122" height="20" rx="6" fill="#dcfce7" stroke="#16a34a"/>
<text x="553" y="359" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#0f172a">LSE 合并 → O</text>
<text x="340" y="388" text-anchor="middle" font-size="10" fill="#64748b">通信：AllGather Q（∝ q·d）+ 一次 LSE 归并（all-reduce / gather）；与 KV Cache 大小、上下文长度均无关</text>
</svg>

▲ AllGather Q 示意图：通信前每 rank 只有 1/C 的 Q head 分片（虚线格为缺失），AllGather 后每 rank 补齐完整 Q（颜色深浅区分来源 rank），KV 分片始终不动；随后各 rank 算 score 列段并 LSE 合并

### 3.3.2 Pass Q（环形传 Q）

Ring 传 KV 的镜像版本。KV 分片常驻不动，Q 连同运行中的 partial 累加器 $(O_{acc}, L_{acc})$ 沿环逐 hop P2P 传递，每个 rank 收到后用本地 KV 算 partial、按在线 softmax 合并进累加器再发给下家，转满 $C$ 跳回到源 rank 时累加器即精确输出，连最后的归并通信都省掉。通信量同样只正比于 $q$（与 KV 无关），但 $C$ 跳串行，延迟高于 AllGather Q。适合 KV 常驻缓存的多轮 decode / incremental prefill，即 Meta《Context Parallelism for Scalable Million-Token Inference》提出的 pass-Q 变体（动态切换见 3.5）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 420" style="width:100%;height:auto">
<defs>
<marker id="m33p2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#ea580c"/></marker>
<marker id="m33g3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#16a34a"/></marker>
</defs>
<rect x="0" y="0" width="680" height="420" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">Pass Q：KV 分片常驻，Q 携带 partial 累加器沿环逐 hop 传递</text>
<text x="160" y="48" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">环形拓扑（KV 分片常驻）</text>
<g font-size="10" text-anchor="middle">
<rect x="48" y="72" width="84" height="44" rx="7" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<text x="90" y="90" font-weight="bold" fill="#0f172a">GPU 0</text><text x="90" y="106" font-size="9" fill="#334155">KV₀ 常驻</text>
<rect x="188" y="72" width="84" height="44" rx="7" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<text x="230" y="90" font-weight="bold" fill="#0f172a">GPU 1</text><text x="230" y="106" font-size="9" fill="#334155">KV₁ 常驻</text>
<rect x="188" y="196" width="84" height="44" rx="7" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<text x="230" y="214" font-weight="bold" fill="#0f172a">GPU 2</text><text x="230" y="230" font-size="9" fill="#334155">KV₂ 常驻</text>
<rect x="48" y="196" width="84" height="44" rx="7" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<text x="90" y="214" font-weight="bold" fill="#0f172a">GPU 3</text><text x="90" y="230" font-size="9" fill="#334155">KV₃ 常驻</text>
</g>
<g stroke="#ea580c" stroke-width="2" fill="none">
<line x1="134" y1="94" x2="184" y2="94" marker-end="url(#m33p2)"/>
<line x1="230" y1="118" x2="230" y2="192" marker-end="url(#m33p2)"/>
<line x1="186" y1="218" x2="136" y2="218" marker-end="url(#m33p2)"/>
<line x1="90" y1="192" x2="90" y2="118" marker-end="url(#m33p2)"/>
</g>
<text x="160" y="146" text-anchor="middle" font-size="10" fill="#64748b">KV 不动</text>
<text x="160" y="162" text-anchor="middle" font-size="9" fill="#64748b">Q 包沿环逐 hop</text>
<text x="480" y="48" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">每轮通信日程（C=4，源 = GPU 0）</text>
<g font-size="9" fill="#334155">
<rect x="300" y="58" width="172" height="88" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="310" y="75" font-size="10" font-weight="bold" fill="#0f172a">轮 0</text>
<text x="310" y="93">Q 在 GPU 0（源），acc = ∅</text>
<text x="310" y="109">算本地 KV₀ 的 partial</text>
<text x="310" y="125">acc ← (O₀, L₀)</text>
<text x="310" y="141" fill="#c2410c">send Q+acc → GPU 1</text>
<rect x="488" y="58" width="172" height="88" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="498" y="75" font-size="10" font-weight="bold" fill="#0f172a">轮 1</text>
<text x="498" y="93">Q 到 GPU 1，acc = (O₀,L₀)</text>
<text x="498" y="109">算本地 KV₁ 的 partial</text>
<text x="498" y="125">acc ⊕ (O₁, L₁)（在线合并）</text>
<text x="498" y="141" fill="#c2410c">send → GPU 2</text>
<rect x="488" y="160" width="172" height="88" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="498" y="177" font-size="10" font-weight="bold" fill="#0f172a">轮 2</text>
<text x="498" y="195">Q 到 GPU 2，acc 含 KV₀~₁</text>
<text x="498" y="211">算本地 KV₂ 的 partial</text>
<text x="498" y="227">acc ⊕ (O₂, L₂)</text>
<text x="498" y="243" fill="#c2410c">send → GPU 3</text>
<rect x="300" y="160" width="172" height="88" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="310" y="177" font-size="10" font-weight="bold" fill="#0f172a">轮 3</text>
<text x="310" y="195">Q 到 GPU 3，acc 含 KV₀~₂</text>
<text x="310" y="211">算本地 KV₃ 的 partial</text>
<text x="310" y="227">acc ⊕ (O₃, L₃) = 完整 O</text>
<text x="310" y="243" fill="#c2410c">send → GPU 0（闭环）</text>
</g>
<g stroke="#ea580c" stroke-width="2" fill="none">
<line x1="474" y1="102" x2="484" y2="102" marker-end="url(#m33p2)"/>
<line x1="574" y1="148" x2="574" y2="156" marker-end="url(#m33p2)"/>
<line x1="484" y1="204" x2="474" y2="204" marker-end="url(#m33p2)"/>
</g>
<text x="380" y="262" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#334155">通信前</text>
<text x="583" y="262" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#334155">通信后（C 跳之后）</text>
<rect x="300" y="268" width="160" height="44" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
<text x="380" y="286" text-anchor="middle" font-size="9" fill="#0f172a">Q 在 GPU 0，acc = ∅</text>
<text x="380" y="302" text-anchor="middle" font-size="9" fill="#334155">尚无 KV 信息</text>
<line x1="466" y1="290" x2="500" y2="290" stroke="#16a34a" stroke-width="2" marker-end="url(#m33g3)"/>
<rect x="506" y="268" width="154" height="44" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
<text x="583" y="286" text-anchor="middle" font-size="9" fill="#0f172a">Q 回到 GPU 0</text>
<text x="583" y="302" text-anchor="middle" font-size="9" fill="#334155">acc = 完整 O（免归并）</text>
<rect x="40" y="330" width="600" height="76" rx="6" fill="#f1f5f9" stroke="#e2e8f0"/>
<text x="340" y="352" text-anchor="middle" font-size="10" fill="#475569">传递的数据：Q [q, d] + 累加器 (O_acc [q, d]，L_acc [q])；通信方式：逐 hop P2P send/recv，串行 C 跳</text>
<text x="340" y="372" text-anchor="middle" font-size="10" fill="#475569">通信量 C·(2qd + q)，与 KV Cache 无关；结果在源 rank 直接成型，省掉最后的归并通信</text>
<text x="340" y="392" text-anchor="middle" font-size="10" fill="#475569">适合 KV 常驻的多轮 decode / incremental prefill（即 3.5 节 Meta 论文的 pass-Q）；缺点：串行延迟高于 AllGather Q</text>
</svg>

▲ Pass Q 示意图：KV 分片常驻各 rank（左），Q 携带 partial 累加器沿环逐 hop 传递并就地在线合并（右，四轮日程全画），通信前 acc 为空，C 跳后回到源 rank 即为完整输出

## 3.4 Ulysses：All-to-All 换维

DeepSpeed-Ulysses 在注意力前后各做一次 All-to-All，把"按序列切分"转置成"按 head 切分"，通信后每个 rank 持有**全序列**的部分 head，直接用普通 FlashAttention，再 All-to-All 转回序列切分。不需要改 kernel、不需要 LSE 合并；通信量正比于 $T$ 但不随 CP 度增长（序列与设备等比例扩展时通信量恒定）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 360" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="360" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">Ulysses：两次 All-to-All，把 [T, H] 矩阵在「行切分」与「列切分」间转置</text>
<text x="115" y="52" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">① 序列切分（按行）</text>
<text x="340" y="52" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">② head 切分（按列）</text>
<text x="565" y="52" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">③ 转回序列切分</text>
<g font-size="9" text-anchor="middle" fill="#64748b">
<text x="115" y="72">Q / K / V [T, H]，每个 GPU 持 T/4 行 × 全部 H 列</text>
<text x="340" y="72">每个 GPU 持全部 T 行 × H/4 列</text>
<text x="565" y="72">输出 [T/4, H] 各回各位</text>
</g>
<g stroke="#ffffff" stroke-width="1.5">
<rect x="40" y="82" width="150" height="38" fill="#bfdbfe"/><rect x="40" y="120" width="150" height="38" fill="#bbf7d0"/><rect x="40" y="158" width="150" height="38" fill="#fde68a"/><rect x="40" y="196" width="150" height="38" fill="#fecaca"/>
<rect x="265" y="82" width="38" height="152" fill="#bfdbfe"/><rect x="303" y="82" width="38" height="152" fill="#bbf7d0"/><rect x="341" y="82" width="38" height="152" fill="#fde68a"/><rect x="379" y="82" width="38" height="152" fill="#fecaca"/>
<rect x="490" y="82" width="150" height="38" fill="#bfdbfe"/><rect x="490" y="120" width="150" height="38" fill="#bbf7d0"/><rect x="490" y="158" width="150" height="38" fill="#fde68a"/><rect x="490" y="196" width="150" height="38" fill="#fecaca"/>
</g>
<g font-size="11" text-anchor="middle">
<text x="115" y="106" fill="#1d4ed8">GPU 0：[B, T/4, H]</text>
<text x="115" y="144" fill="#15803d">GPU 1：[B, T/4, H]</text>
<text x="115" y="182" fill="#b45309">GPU 2：[B, T/4, H]</text>
<text x="115" y="220" fill="#b91c1c">GPU 3：[B, T/4, H]</text>
<text x="284" y="162" fill="#1d4ed8">H/4</text>
<text x="322" y="162" fill="#15803d">H/4</text>
<text x="360" y="162" fill="#b45309">H/4</text>
<text x="398" y="162" fill="#b91c1c">H/4</text>
<text x="565" y="106" fill="#1d4ed8">GPU 0：[B, T/4, H]</text>
<text x="565" y="144" fill="#15803d">GPU 1：[B, T/4, H]</text>
<text x="565" y="182" fill="#b45309">GPU 2：[B, T/4, H]</text>
<text x="565" y="220" fill="#b91c1c">GPU 3：[B, T/4, H]</text>
</g>
<text x="340" y="252" text-anchor="middle" font-size="10" fill="#64748b">列方向示意：T 全长（↑↓）× H/4（每列色块 = 一个 GPU 的 head 子集）</text>
<g stroke="#7c3aed" stroke-width="2.2" fill="none">
<line x1="196" y1="158" x2="258" y2="158"/><line x1="417" y1="158" x2="483" y2="158"/>
</g>
<g font-size="11" fill="#6d28d9" text-anchor="middle">
<text x="227" y="144">All-to-All</text>
<text x="227" y="176">行切 → 列切</text>
<text x="450" y="144">All-to-All</text>
<text x="450" y="176">列切 → 行切</text>
</g>
<rect x="140" y="272" width="400" height="34" rx="7" fill="#f0fdf4" stroke="#16a34a"/>
<text x="340" y="294" text-anchor="middle" font-size="11" fill="#14532d">中间状态：普通 FlashAttention 算「全序列 × 本地 head 列块」，无需 LSE 合并</text>
<rect x="60" y="316" width="560" height="30" rx="6" fill="#f1f5f9" stroke="#e2e8f0"/>
<text x="340" y="336" text-anchor="middle" font-size="11" fill="#475569">硬约束：CP 度 ≤ head 数 —— GQA/MQA/MLA 的 KV head 极少，Ulysses 很快到顶</text>
</svg>

▲ Ulysses 示意图：矩阵从「按行（序列）切分」经 All-to-All 变为「按列（head）切分」（注意：其集合通信原语是 All-to-All，不是 AllGather）

- 硬约束：CP 度 ≤ head 数（GQA/MQA/MLA 下 KV head 极少，约束非常紧；MLA 吸收式下甚至没有 per-head KV 可交换，有效 Ulysses 度塌缩为 1）。
- 注意：Ulysses 的集合通信原语是 **All-to-All**，不是 AllGather，常被混称。

## 3.5 混合方案：USP / a2a+p2p、pass-KV / pass-Q 动态切换

Ulysses 与 Ring 不互斥，USP（Unified Sequence Parallelism）在 2D 网格上组合两者——节点内（NVLink）用 Ulysses All-to-All，节点间（IB/RoCE）用 Ring P2P，总 CP 度可超过 head 数。Megatron-Core 的 `cp_comm_type=a2a+p2p` 配合 `hierarchical_context_parallel_sizes`（内层 a2a 尺寸、外层 p2p 尺寸）就是该思路。USP 论文还给出 4D 并行的推荐嵌套顺序（TP < SP-Ulysses < SP-Ring < ZeRO-DP < PP）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 350" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="350" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">USP / a2a+p2p：[T, H] 矩阵二维切块 —— 行向 Ring，列向 Ulysses</text>
<text x="195" y="52" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">矩阵切分（行 = 序列块，列 = head 块）</text>
<rect x="90" y="62" width="212" height="212" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>
<g stroke="#ffffff" stroke-width="1.5">
<rect x="92" y="64" width="104" height="104" fill="#bfdbfe"/><rect x="196" y="64" width="104" height="104" fill="#bbf7d0"/>
<rect x="92" y="168" width="104" height="104" fill="#fde68a"/><rect x="196" y="168" width="104" height="104" fill="#fecaca"/>
</g>
<g font-size="11" text-anchor="middle">
<text x="144" y="112" fill="#1d4ed8">GPU 0</text><text x="144" y="128" fill="#1d4ed8">T/2 × H/2</text>
<text x="248" y="112" fill="#15803d">GPU 1</text><text x="248" y="128" fill="#15803d">T/2 × H/2</text>
<text x="144" y="216" fill="#b45309">GPU 2</text><text x="144" y="232" fill="#b45309">T/2 × H/2</text>
<text x="248" y="216" fill="#b91c1c">GPU 3</text><text x="248" y="232" fill="#b91c1c">T/2 × H/2</text>
</g>
<g stroke="#7c3aed" stroke-width="2" fill="none">
<line x1="92" y1="116" x2="300" y2="116" stroke-dasharray="6 3"/>
<line x1="92" y1="220" x2="300" y2="220" stroke-dasharray="6 3"/>
</g>
<text x="310" y="80" font-size="10" fill="#6d28d9">← 列方向：a2a 组</text>
<text x="310" y="200" font-size="10" fill="#c2410c">← 行方向：p2p 环</text>
<text x="195" y="296" text-anchor="middle" font-size="10" fill="#64748b">每块 = 一个 GPU 持有的 (序列段 × head 子集)</text>
<text x="510" y="52" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">两层通信（示例：节点内 4 GPU × 节点间 2）</text>
<g>
<rect x="360" y="64" width="300" height="96" rx="9" fill="#faf5ff" stroke="#a78bfa" stroke-width="1.5"/>
<text x="510" y="84" text-anchor="middle" font-size="11" font-weight="bold" fill="#6d28d9">节点 0（NVLink）：Ulysses a2a</text>
<g font-size="10" text-anchor="middle">
<rect x="380" y="96" width="58" height="28" rx="4" fill="#ddd6fe"/><text x="409" y="114" fill="#3b0764">GPU 0</text>
<rect x="448" y="96" width="58" height="28" rx="4" fill="#ddd6fe"/><text x="477" y="114" fill="#3b0764">GPU 1</text>
<rect x="380" y="130" width="58" height="28" rx="4" fill="#ddd6fe"/><text x="409" y="148" fill="#3b0764">GPU 2</text>
<rect x="448" y="130" width="58" height="28" rx="4" fill="#ddd6fe"/><text x="477" y="148" fill="#3b0764">GPU 3</text>
</g>
<rect x="530" y="90" width="110" height="60" rx="6" fill="#fff7ed" stroke="#fdba74"/>
<text x="585" y="114" text-anchor="middle" font-size="10" fill="#7c2d12">节点 1：GPU 4-7</text>
<text x="585" y="132" text-anchor="middle" font-size="10" fill="#7c2d12">同构 a2a</text>
</g>
<g stroke="#7c3aed" stroke-width="1.8" fill="none">
<line x1="380" y1="110" x2="442" y2="110"/><line x1="380" y1="144" x2="442" y2="144"/>
</g>
<g stroke="#ea580c" stroke-width="2" fill="none">
<line x1="510" y1="182" x2="510" y2="212"/>
<line x1="360" y1="198" x2="360" y2="198"/>
</g>
<rect x="360" y="214" width="300" height="52" rx="9" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>
<text x="510" y="236" text-anchor="middle" font-size="11" font-weight="bold" fill="#c2410c">节点间（IB / RoCE）：Ring P2P 传 KV 块</text>
<text x="510" y="254" text-anchor="middle" font-size="10" fill="#9a3412">KV 块沿 p2p 环跨节点流转，与计算重叠</text>
<rect x="60" y="310" width="560" height="28" rx="6" fill="#f1f5f9" stroke="#e2e8f0"/>
<text x="340" y="329" text-anchor="middle" font-size="11" fill="#475569">总 CP 度 = a2a 度 × p2p 度（例 4×2=8），可超过 head 数；对应 Megatron hierarchical_context_parallel_sizes</text>
</svg>

▲ USP 混合示意图：[T, H] 矩阵二维切块，列向走 Ulysses a2a（节点内）、行向走 Ring P2P（节点间）

**另一种混合：pass-KV / pass-Q 动态切换（Meta CP，MLSys 2025）**

USP 是把 3.4 的 Ulysses 与 3.1 的 Ring 在**拓扑层面**组合；Meta 的《Context Parallelism for Scalable Million-Token Inference》（arXiv:2411.01783，MLSys 2025）则是在**算法层面**把 3.1 的传 KV 与 3.3 的传 Q 做成两个无损精确的 ring attention 变体 **pass-KV** 与 **pass-Q**，运行时按负载动态切换。这是首篇公开 CP 推理系统实现细节的论文，16 节点 128 张 H100 上 Llama3-405B 的 1M 上下文 prefill 仅 **77 秒**（并行效率 93%、FLOPS 利用率 63%），128K prefill 3.8 秒；RDMA 与中低带宽 TCP（约 3 GB/s per rank）下均近线性扩展——跨节点通信量比 TP 小数个量级（8 节点 128K 时 TP16 已比 CP8 慢约 100%）。

- **pass-KV（= 3.1 Ring）**：Q 行块留本地，KV 沿环 P2P 传 $N-1$ 轮，每轮 SendRecv 与 attention 计算重叠，LSE（论文称 merge attention）合并。通信量 $\propto T\cdot N_{kv}\cdot d$，GQA 下比传 Q 小约 $N_h/N_{kv}$ 倍（405B 约 16 倍）。适合 full prefill 与 miss rate 高（新 token 占比大）的 partial prefill；decode 不用（$T=1$ 时传 KV 几乎总是更贵）。
- **pass-Q（= 3.3 传 Q）**：KV 分片**跨轮常驻**（persistent KV cache，多轮会话间不动），Q 沿环传，各 rank 对本地 KV 分片算 partial $(O_j, L_j)$，经 **All2All + permute** 把散落在各 rank 的部分结果归位后再 LSE 合并。通信量只与新 token 数 $T$ 成正比，与 KV cache 长度无关。适合 decode（$T=1$）与 miss rate 低（$P \gg T$）的 partial prefill。
- **动态切换启发式**：记 miss rate $=T/(T+P)$（$T$=新 token 数，$P$=已缓存 KV 长度），传 Q 消息更小当且仅当 $\frac{T}{T+P}\le 2\frac{N_{kv}}{N_h}$（405B 阈值 12.5%）；实际再叠加 roofline 条件（pass-KV 的 SendRecv 能被 attention 遮蔽时优先 pass-KV）、计入 All2All 的精化式，以及经验拟合 $h(T,P)=-1.059\log T+1.145\log\frac{T}{T+P}+12.112$（$h>0$ 选 pass-KV）。128K / CP4 实测切换点约在 miss rate 5% 附近。
- **多轮 persistent KV 与负载均衡**：KV cache 按 CP rank 分布存储、跨轮保持；序列切 $2N$ 块、rank $i$ 取 $(C_i,\ C_{2N-1-i})$ 首尾配对（同 zigzag 切分），同时均衡 attention 计算与 KV 显存；decode 用 round-robin 每步偏移 1 个 index，防某 rank 既背 decode 又持最长 KV 而 OOM；ring 消息按 $\max_i P_i+\lceil T/N\rceil$ 等长 padding 以满足集合通信接口。
- **适用边界**：CP 显著降低 TTFT，但 TTIT 略有退化（padding + SendRecv/All2All 开销），论文建议与 PD 分离（DistServe / Mooncake）配合部署。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 440" style="width:100%;height:auto">
<defs>
<marker id="m35s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#6366f1"/></marker>
<marker id="m35q" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2563eb"/></marker>
</defs>
<rect x="0" y="0" width="680" height="440" fill="#ffffff"/>
<text x="340" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">pass-KV / pass-Q 动态切换（Meta CP）：同一 Ring 拓扑，按 KV miss rate 决定传什么</text>
<rect x="20" y="34" width="640" height="44" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="34" y="52" font-size="10" font-weight="bold" fill="#334155">多轮会话：</text>
<rect x="100" y="42" width="250" height="18" rx="3" fill="#fed7aa" stroke="#ea580c"/>
<text x="225" y="55" text-anchor="middle" font-size="9" fill="#7c2d12">P：已缓存 KV（persistent，跨轮常驻）</text>
<rect x="350" y="42" width="92" height="18" rx="3" fill="#fdba74" stroke="#c2410c"/>
<text x="396" y="55" text-anchor="middle" font-size="9" fill="#7c2d12">T：新 token</text>
<text x="646" y="55" text-anchor="end" font-size="10" fill="#334155">miss rate = T/(T+P)</text>
<text x="34" y="72" font-size="9" fill="#64748b">decode 每步 T=1 → miss rate≈0；full prefill P=0 → miss rate=100%；多轮 partial prefill 介于两者之间</text>
<rect x="110" y="86" width="460" height="44" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
<text x="340" y="104" text-anchor="middle" font-size="11" font-weight="bold" fill="#312e81">切换规则：T/(T+P) ≤ 2·N_kv/N_h → pass-Q；否则 → pass-KV</text>
<text x="340" y="121" text-anchor="middle" font-size="10" fill="#3730a3">Llama3-405B：N_h=128、N_kv=8 → 阈值 12.5%；实测切换点 ≈ miss rate 5%（128K / CP4）</text>
<line x1="255" y1="130" x2="180" y2="150" stroke="#6366f1" stroke-width="1.5" marker-end="url(#m35s)"/>
<line x1="425" y1="130" x2="500" y2="150" stroke="#6366f1" stroke-width="1.5" marker-end="url(#m35s)"/>
<rect x="20" y="152" width="312" height="196" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="176" y="172" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a5f">pass-KV（miss rate 高 / full prefill）— 即 3.1 Ring</text>
<g font-size="8" text-anchor="middle" fill="#64748b">
<text x="66" y="190">K₀</text><text x="92" y="190">K₁</text><text x="118" y="190">K₂</text><text x="144" y="190">K₃</text>
<text x="46" y="212" text-anchor="end" fill="#1d4ed8">Q₀</text>
<text x="46" y="238" text-anchor="end" fill="#15803d">Q₁</text>
<text x="46" y="264" text-anchor="end" fill="#b45309">Q₂</text>
<text x="46" y="290" text-anchor="end" fill="#b91c1c">Q₃</text>
</g>
<g stroke="#ffffff" stroke-width="1.5">
<rect x="53" y="194" width="26" height="26" fill="#bfdbfe"/><rect x="79" y="194" width="26" height="26" fill="#e2e8f0"/><rect x="105" y="194" width="26" height="26" fill="#e2e8f0"/><rect x="131" y="194" width="26" height="26" fill="#e2e8f0"/>
<rect x="53" y="220" width="26" height="26" fill="#bbf7d0"/><rect x="79" y="220" width="26" height="26" fill="#bbf7d0"/><rect x="105" y="220" width="26" height="26" fill="#e2e8f0"/><rect x="131" y="220" width="26" height="26" fill="#e2e8f0"/>
<rect x="53" y="246" width="26" height="26" fill="#fde68a"/><rect x="79" y="246" width="26" height="26" fill="#fde68a"/><rect x="105" y="246" width="26" height="26" fill="#fde68a"/><rect x="131" y="246" width="26" height="26" fill="#e2e8f0"/>
<rect x="53" y="272" width="26" height="26" fill="#fecaca"/><rect x="79" y="272" width="26" height="26" fill="#fecaca"/><rect x="105" y="272" width="26" height="26" fill="#fecaca"/><rect x="131" y="272" width="26" height="26" fill="#fecaca"/>
</g>
<rect x="77" y="192" width="30" height="108" fill="none" stroke="#ea580c" stroke-width="2.5"/>
<text x="92" y="312" text-anchor="middle" font-size="8" fill="#c2410c">KV 列块沿环轮转</text>
<g font-size="8.5">
<rect x="170" y="188" width="152" height="18" rx="4" fill="#fff7ed" stroke="#fdba74"/>
<text x="246" y="201" text-anchor="middle" fill="#7c2d12">轮 0：算 K₀ → 发 K₀ / 收 K₃</text>
<rect x="170" y="210" width="152" height="18" rx="4" fill="#fff7ed" stroke="#fdba74"/>
<text x="246" y="223" text-anchor="middle" fill="#7c2d12">轮 1：算 K₃ → 发 K₃ / 收 K₂</text>
<rect x="170" y="232" width="152" height="18" rx="4" fill="#fff7ed" stroke="#fdba74"/>
<text x="246" y="245" text-anchor="middle" fill="#7c2d12">轮 2：算 K₂ → 发 K₂ / 收 K₁</text>
<text x="246" y="266" text-anchor="middle" fill="#9a3412">GPU0 视角（C=4）：末块 K₁ 收到后算完</text>
<text x="246" y="282" text-anchor="middle" fill="#9a3412">N−1 轮 SendRecv 与计算重叠 → LSE 合并</text>
</g>
<rect x="30" y="316" width="292" height="24" rx="5" fill="#fff7ed" stroke="#fdba74"/>
<text x="176" y="332" text-anchor="middle" font-size="9" fill="#7c2d12">传 KV：∝ T·N_kv·d，GQA 下比传 Q 小约 N_h/N_kv 倍（405B：16×）</text>
<rect x="348" y="152" width="312" height="196" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="504" y="172" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a5f">pass-Q（miss rate 低 / decode）— 即 3.3 传 Q</text>
<g font-size="8" text-anchor="middle" fill="#64748b">
<text x="381" y="190">KV₀</text><text x="407" y="190">KV₁</text><text x="433" y="190">KV₂</text><text x="459" y="190">KV₃</text>
<text x="360" y="212" text-anchor="end" fill="#1d4ed8">Q₀</text>
<text x="360" y="238" text-anchor="end" fill="#94a3b8">Q₁</text>
<text x="360" y="264" text-anchor="end" fill="#94a3b8">Q₂</text>
<text x="360" y="290" text-anchor="end" fill="#94a3b8">Q₃</text>
</g>
<g stroke="#ffffff" stroke-width="1.5">
<rect x="368" y="194" width="26" height="26" fill="#fed7aa"/><rect x="394" y="194" width="26" height="26" fill="#fdba74"/><rect x="420" y="194" width="26" height="26" fill="#fed7aa"/><rect x="446" y="194" width="26" height="26" fill="#fdba74"/>
<rect x="368" y="220" width="26" height="26" fill="#fed7aa"/><rect x="394" y="220" width="26" height="26" fill="#fdba74"/><rect x="420" y="220" width="26" height="26" fill="#fed7aa"/><rect x="446" y="220" width="26" height="26" fill="#fdba74"/>
<rect x="368" y="246" width="26" height="26" fill="#fed7aa"/><rect x="394" y="246" width="26" height="26" fill="#fdba74"/><rect x="420" y="246" width="26" height="26" fill="#fed7aa"/><rect x="446" y="246" width="26" height="26" fill="#fdba74"/>
<rect x="368" y="272" width="26" height="26" fill="#fed7aa"/><rect x="394" y="272" width="26" height="26" fill="#fdba74"/><rect x="420" y="272" width="26" height="26" fill="#fed7aa"/><rect x="446" y="272" width="26" height="26" fill="#fdba74"/>
</g>
<rect x="366" y="192" width="108" height="30" fill="none" stroke="#2563eb" stroke-width="2.5"/>
<path d="M 480 207 L 500 207" stroke="#2563eb" stroke-width="1.6" marker-end="url(#m35q)"/>
<text x="420" y="312" text-anchor="middle" font-size="8" fill="#1d4ed8">Q 行块沿环流转，KV 列块常驻</text>
<g font-size="8.5">
<rect x="496" y="188" width="156" height="18" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
<text x="574" y="201" text-anchor="middle" fill="#1e3a81">轮 0：Q₀@R0×KV₀ → 发 Q₀</text>
<rect x="496" y="210" width="156" height="18" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
<text x="574" y="223" text-anchor="middle" fill="#1e3a81">轮 1：Q₀@R1×KV₁ → 发 Q₀</text>
<rect x="496" y="232" width="156" height="18" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
<text x="574" y="245" text-anchor="middle" fill="#1e3a81">轮 2：Q₀@R2×KV₂ → 发 Q₀</text>
<text x="574" y="266" text-anchor="middle" fill="#1d4ed8">每轮只传小 Q（decode 每步 1 token）</text>
<text x="574" y="282" text-anchor="middle" fill="#1d4ed8">4 个 partial 散在各 rank → All2All 归位 + LSE 合并</text>
</g>
<rect x="358" y="316" width="292" height="24" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
<text x="504" y="332" text-anchor="middle" font-size="9" fill="#1e3a81">传 Q：∝ T·d·e，与 KV cache 大小无关；KV 分片跨轮常驻（persistent）</text>
<rect x="20" y="356" width="640" height="34" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="340" y="370" text-anchor="middle" font-size="10" fill="#334155">负载均衡：序列切 2N 块、首尾配对 rank i ← (C_i, C_{2N−1−i})（同 zigzag），计算与 KV 显存双均衡；</text>
<text x="340" y="384" text-anchor="middle" font-size="10" fill="#64748b">decode 用 round-robin 每步偏移 1 个 index 防 OOM；ring 消息按 max_i P_i + ⌈T/N⌉ 等长 padding</text>
<rect x="20" y="396" width="640" height="34" rx="7" fill="#f1f5f9" stroke="#e2e8f0"/>
<text x="340" y="410" text-anchor="middle" font-size="10" fill="#475569">传递的数据 / 通信方式：pass-KV = KV 张量 P2P 环传（N−1 轮 SendRecv 与计算重叠）；pass-Q = 小 Q P2P 环传 + All2All 归位；</text>
<text x="340" y="424" text-anchor="middle" font-size="10" fill="#475569">两者均以 LSE 合并收尾、无损精确，按 miss rate T/(T+P) 与 roofline 自动切换，覆盖 full prefill / persistent KV prefill / decode</text>
</svg>

▲ pass-KV / pass-Q 动态切换示意图：上 = 多轮会话的 miss rate 与切换规则；下左 = pass-KV 的 KV 列块每轮环传（GPU0 视角），下右 = pass-Q 的 Q 行块每轮环传 + KV 列块常驻

## 3.6 公共组件：LSE 合并（在线 softmax）

"传 KV 分多轮算"与"传 Q 分片算"最终都要把部分注意力结果合并成精确结果。设分片 $j$ 给出归一化输出 $O_j$ 和对数和 $L_j$。

$$m = \max_j L_j,\quad w_j = \exp(L_j - m),\quad O = \frac{\sum_j w_j O_j}{\sum_j w_j}$$

这就是 FlashAttention 的 online softmax 推广，也是 decode CP 通信量极小的原因——每步只需传输出和 LSE，无需传原始 KV。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 330" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="330" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">LSE 合并：score 行分段的 (O_j, L_j) 加权合成精确输出</text>
<text x="150" y="52" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">各分片的 partial 结果</text>
<text x="430" y="52" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">合并（在线 softmax）</text>
<g>
<rect x="40" y="64" width="220" height="44" rx="6" fill="#dcfce7" stroke="#16a34a"/>
<rect x="52" y="76" width="60" height="20" fill="#86efac" stroke="#16a34a" stroke-width="1.2"/>
<text x="82" y="90" text-anchor="middle" font-size="10" fill="#14532d">O₀</text>
<text x="190" y="90" text-anchor="middle" font-size="11" fill="#14532d">L₀（logsumexp）</text>
<rect x="40" y="118" width="220" height="44" rx="6" fill="#dcfce7" stroke="#16a34a"/>
<rect x="52" y="130" width="60" height="20" fill="#86efac" stroke="#16a34a" stroke-width="1.2"/>
<text x="82" y="144" text-anchor="middle" font-size="10" fill="#14532d">O₁</text>
<text x="190" y="144" text-anchor="middle" font-size="11" fill="#14532d">L₁（logsumexp）</text>
<rect x="40" y="172" width="220" height="44" rx="6" fill="#dcfce7" stroke="#16a34a"/>
<rect x="52" y="184" width="60" height="20" fill="#86efac" stroke="#16a34a" stroke-width="1.2"/>
<text x="82" y="198" text-anchor="middle" font-size="10" fill="#14532d">O₂</text>
<text x="190" y="198" text-anchor="middle" font-size="11" fill="#14532d">L₂（logsumexp）</text>
</g>
<g stroke="#16a34a" stroke-width="1.8" fill="none">
<path d="M 262 86 L 316 130"/><path d="M 262 140 L 316 140"/><path d="M 262 194 L 316 150"/>
</g>
<rect x="320" y="80" width="230" height="120" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
<text x="435" y="106" text-anchor="middle" font-size="12" font-weight="bold" fill="#14532d">合并公式（在线 softmax）</text>
<text x="435" y="132" text-anchor="middle" font-size="12" fill="#14532d">m = max_j L_j</text>
<text x="435" y="156" text-anchor="middle" font-size="12" fill="#14532d">w_j = exp(L_j − m)</text>
<text x="435" y="180" text-anchor="middle" font-size="12" fill="#14532d">O = Σ_j w_j·O_j / Σ_j w_j</text>
<line x1="552" y1="140" x2="586" y2="140" stroke="#16a34a" stroke-width="1.8"/>
<rect x="588" y="112" width="72" height="56" rx="8" fill="#f1f5f9" stroke="#94a3b8"/>
<rect x="600" y="126" width="48" height="14" fill="#86efac" stroke="#16a34a" stroke-width="1.2"/>
<rect x="600" y="142" width="48" height="14" fill="#86efac" stroke="#16a34a" stroke-width="1.2"/>
<text x="624" y="182" text-anchor="middle" font-size="10" fill="#334155">输出 O</text>
<text x="340" y="232" text-anchor="middle" font-size="11" fill="#475569">矩阵视角：score 行 S[q, T] 列向切成 j 段 → 各段独立 softmax 得 (O_j, L_j) → 加权拼回整行</text>
<g>
<rect x="180" y="244" width="70" height="22" fill="#fed7aa"/><rect x="250" y="244" width="70" height="22" fill="#fdba74"/><rect x="320" y="244" width="70" height="22" fill="#fed7aa"/><rect x="390" y="244" width="70" height="22" fill="#fdba74"/>
<text x="215" y="259" text-anchor="middle" font-size="9" fill="#7c2d12">段 0</text>
<text x="285" y="259" text-anchor="middle" font-size="9" fill="#7c2d12">段 1</text>
<text x="355" y="259" text-anchor="middle" font-size="9" fill="#7c2d12">段 2</text>
<text x="425" y="259" text-anchor="middle" font-size="9" fill="#7c2d12">段 3</text>
</g>
<text x="340" y="286" text-anchor="middle" font-size="10" fill="#64748b">↓ 各段 (O_j, L_j) ↓</text>
<rect x="230" y="296" width="220" height="20" fill="#86efac" stroke="#16a34a" stroke-width="1.2"/>
<text x="340" y="310" text-anchor="middle" font-size="9" fill="#14532d">合并后的完整 score 行 / 输出 O</text>
</svg>

▲ LSE 合并示意图：各列段 partial 结果 (O_j, L_j) 与整行 score 矩阵的对应关系

## 3.7 选型小结

| 方案 | 通信内容 | 通信量 | 受 head 数限制 | Kernel 侵入 | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| Ring（传 KV） | KV 块 P2P | $O(T)$，可重叠 | 否 | 高（LSE 合并） | 训练长序列、超长 prefill |
| AllGather KV | 全量 KV | $O(T)$，一次性 | 否 | 低 | Prefill CP、KV 小的架构 |
| 传 Q | Q + 输出/LSE | 与 KV 无关 | 否 | 中 | Decode CP |
| Ulysses（a2a） | Q/K/V/O | $O(T)$，恒定 | 是 | 低 | 节点内、head 多 |
| USP（a2a+p2p） | 组合 | 组合 | 部分 | 中 | 多节点混合拓扑 |

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 360" style="width:100%;height:auto">
<defs>
<marker id="m37" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker>
</defs>
<rect x="0" y="0" width="680" height="360" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">CP 通信方案选择路径</text>
<rect x="260" y="40" width="160" height="34" rx="8" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<text x="340" y="62" text-anchor="middle" font-size="12" font-weight="bold" fill="#0f172a">CP 通信方案选择</text>
<g stroke="#64748b" stroke-width="1.6" fill="none">
<line x1="340" y1="74" x2="340" y2="98" marker-end="url(#m37)"/>
</g>
<rect x="275" y="100" width="130" height="32" rx="6" fill="#f8fafc" stroke="#94a3b8"/>
<text x="340" y="121" text-anchor="middle" font-size="12" fill="#0f172a">哪个阶段？</text>
<g stroke="#64748b" stroke-width="1.6" fill="none">
<line x1="300" y1="132" x2="185" y2="168" marker-end="url(#m37)"/>
<line x1="380" y1="132" x2="500" y2="168" marker-end="url(#m37)"/>
</g>
<text x="220" y="152" text-anchor="middle" font-size="11" fill="#475569">Prefill</text>
<text x="460" y="152" text-anchor="middle" font-size="11" fill="#475569">Decode</text>
<rect x="95" y="170" width="170" height="32" rx="6" fill="#f8fafc" stroke="#94a3b8"/>
<text x="180" y="191" text-anchor="middle" font-size="11" fill="#0f172a">全局 KV 放得下吗？</text>
<rect x="415" y="170" width="180" height="32" rx="6" fill="#f8fafc" stroke="#94a3b8"/>
<text x="505" y="191" text-anchor="middle" font-size="11" fill="#0f172a">Q 小、KV 大 →</text>
<g stroke="#64748b" stroke-width="1.6" fill="none">
<line x1="130" y1="202" x2="95" y2="238" marker-end="url(#m37)"/>
<line x1="230" y1="202" x2="270" y2="238" marker-end="url(#m37)"/>
<line x1="505" y1="202" x2="505" y2="238" marker-end="url(#m37)"/>
</g>
<text x="98" y="224" text-anchor="middle" font-size="11" fill="#15803d">放得下</text>
<text x="262" y="224" text-anchor="middle" font-size="11" fill="#c2410c">放不下</text>
<rect x="30" y="240" width="150" height="38" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
<text x="105" y="264" text-anchor="middle" font-size="12" font-weight="bold" fill="#0f172a">AllGather KV</text>
<g stroke="#ffffff" stroke-width="1"><rect x="38" y="248" width="10" height="10" fill="#fdba74"/><rect x="38" y="258" width="10" height="10" fill="#fdba74"/></g>
<rect x="205" y="240" width="150" height="38" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
<text x="280" y="264" text-anchor="middle" font-size="12" font-weight="bold" fill="#0f172a">Ring（传 KV 块）</text>
<g stroke="#ffffff" stroke-width="1"><rect x="213" y="248" width="10" height="10" fill="#fdba74"/><rect x="223" y="248" width="10" height="10" fill="#fed7aa"/></g>
<rect x="415" y="240" width="180" height="38" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
<text x="505" y="264" text-anchor="middle" font-size="12" font-weight="bold" fill="#0f172a">传 Q + LSE 合并</text>
<g stroke="#ffffff" stroke-width="1"><rect x="423" y="248" width="20" height="8" fill="#93c5fd"/></g>
<g stroke="#ffffff" stroke-width="1"><rect x="423" y="258" width="10" height="8" fill="#fdba74"/><rect x="433" y="258" width="10" height="8" fill="#fed7aa"/></g>
<rect x="30" y="300" width="620" height="48" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="5 4"/>
<text x="340" y="320" text-anchor="middle" font-size="11" fill="#334155">补充判断：head 富余 + 节点内高带宽、想少改 kernel → Ulysses（a2a）；CP 度需超过 head 数的多节点场景 → USP（a2a+p2p）</text>
<text x="340" y="338" text-anchor="middle" font-size="11" fill="#64748b">无论哪条路径，多轮/多分片的部分结果最终都用 LSE 合并收尾</text>
</svg>

▲ CP 通信方案选择路径

# 4. 不同Attention的CP

大语言模型的Attention有多种实现，比较常见的包括GQA, MLA, DSA, GDN, KDA, CSA等。而不同的Attention在CP时需要传递的数据有所不同。下面分别针对不同的Attention进行详细介绍。

## 4.1 GQA



GQA（Grouped Query Attention）让多个 query head 共享一组 KV head，KV Cache 省了，但代价是 KV head 数很少（Llama3-8B 8 个、Qwen3-235B 4 个）。这带来两个 CP 特征。

- **Ulysses 度被 KV head 数卡死**：a2a 按 head 切分时 CP 度 ≤ KV head 数，GQA 下很快到顶；这也是 Ulysses 在现代模型上逐渐让位给 Ring/AllGather KV/传 Q 的原因。
- **传 KV 或传 Q 都可行**：prefill 常用 AllGather KV（SGLang/vLLM PCP），decode 常用传 Q（AllGather Q + 本地 KV 分片 + LSE 合并）。vLLM DCP 中 GQA 的 `dcp_size` 上界是 `tp_size / H_kv`；当 DCP 组与 TP 组重叠时各 rank 的 Q head 不同，需要先沿 head 维 AllGather Q，算完再用 LSE 感知的 all-reduce/a2a 归并（如 `cp_lse_ag_out_rs`）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 320" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="320" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">GQA 的 CP：传 KV 矩阵或传 Q 矩阵</text>
<text x="120" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">GQA 结构（4 个 Q head 共享 1 个 KV head）</text>
<g font-size="9" text-anchor="middle">
<rect x="40" y="62" width="44" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/><text x="62" y="77" fill="#1e3a8a">Q h0</text>
<rect x="92" y="62" width="44" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/><text x="114" y="77" fill="#1e3a8a">Q h1</text>
<rect x="40" y="92" width="44" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/><text x="62" y="107" fill="#1e3a8a">Q h2</text>
<rect x="92" y="92" width="44" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/><text x="114" y="107" fill="#1e3a8a">Q h3</text>
<rect x="52" y="132" width="72" height="24" rx="3" fill="#fdba74" stroke="#ea580c"/><text x="88" y="148" fill="#7c2d12">KV head 0</text>
</g>
<path d="M 62 84 L 80 132" stroke="#94a3b8" stroke-width="1.2" fill="none"/>
<path d="M 114 84 L 96 132" stroke="#94a3b8" stroke-width="1.2" fill="none"/>
<path d="M 62 114 L 80 132" stroke="#94a3b8" stroke-width="1.2" fill="none"/>
<path d="M 114 114 L 96 132" stroke="#94a3b8" stroke-width="1.2" fill="none"/>
<text x="120" y="178" text-anchor="middle" font-size="9" fill="#64748b">KV head 数 H_kv 少 → Ulysses 度 ≤ H_kv 很快到顶</text>
<text x="355" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">路径 A（Prefill）：传 KV —— AllGather KV 矩阵</text>
<g>
<rect x="250" y="62" width="60" height="34" fill="#bfdbfe"/><rect x="250" y="96" width="60" height="34" fill="#bbf7d0"/><rect x="250" y="130" width="60" height="34" fill="#fde68a"/><rect x="250" y="164" width="60" height="34" fill="#fecaca"/>
<text x="280" y="83" text-anchor="middle" font-size="9" fill="#1e3a8a">Q₀</text><text x="280" y="117" text-anchor="middle" font-size="9" fill="#14532d">Q₁</text><text x="280" y="151" text-anchor="middle" font-size="9" fill="#92400e">Q₂</text><text x="280" y="185" text-anchor="middle" font-size="9" fill="#7f1d1d">Q₃</text>
<rect x="330" y="62" width="34" height="136" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="347" y="134" text-anchor="middle" font-size="9" fill="#7c2d12" transform="rotate(90 347 134)">KV₀</text>
<rect x="376" y="62" width="34" height="136" fill="#fed7aa" stroke="#ea580c" stroke-width="1.2"/>
<text x="393" y="134" text-anchor="middle" font-size="9" fill="#7c2d12" transform="rotate(90 393 134)">KV₁</text>
<rect x="422" y="62" width="34" height="136" fill="#fdba74" stroke="#ea580c" stroke-width="1.2"/>
<text x="439" y="134" text-anchor="middle" font-size="9" fill="#7c2d12" transform="rotate(90 439 134)">KV₂</text>
<rect x="468" y="62" width="34" height="136" fill="#fed7aa" stroke="#ea580c" stroke-width="1.2"/>
<text x="485" y="134" text-anchor="middle" font-size="9" fill="#7c2d12" transform="rotate(90 485 134)">KV₃</text>
</g>
<text x="399" y="216" text-anchor="middle" font-size="10" fill="#c2410c">AllGather（T 维）→ 全局 KV [T, d] × H_kv</text>
<text x="399" y="234" text-anchor="middle" font-size="9" fill="#64748b">Q 行块 [T/C, H_q] 不动，对全局 KV 标准 FlashAttention</text>
<text x="580" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">路径 B（Decode）：传 Q</text>
<g>
<rect x="530" y="62" width="100" height="18" fill="#93c5fd" stroke="#2563eb" stroke-width="1.2"/>
<text x="580" y="75" text-anchor="middle" font-size="9" fill="#1e3a8a">Q [q, H_q, d]（很小）</text>
<text x="580" y="98" text-anchor="middle" font-size="9" fill="#1d4ed8">broadcast / all-gather ↓</text>
<rect x="530" y="106" width="46" height="40" fill="#fdba74"/><rect x="584" y="106" width="46" height="40" fill="#fed7aa"/>
<text x="553" y="130" text-anchor="middle" font-size="8" fill="#7c2d12">KV₀</text><text x="607" y="130" text-anchor="middle" font-size="8" fill="#7c2d12">KV₁</text>
<rect x="530" y="150" width="46" height="40" fill="#fed7aa"/><rect x="584" y="150" width="46" height="40" fill="#fdba74"/>
<text x="553" y="174" text-anchor="middle" font-size="8" fill="#7c2d12">KV₂</text><text x="607" y="174" text-anchor="middle" font-size="8" fill="#7c2d12">KV₃</text>
<text x="580" y="208" text-anchor="middle" font-size="9" fill="#15803d">各 rank 对本地 KV 分片 → (O_j, L_j)</text>
<rect x="532" y="218" width="96" height="24" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="580" y="234" text-anchor="middle" font-size="9" fill="#14532d">LSE 合并（很小）</text>
</g>
<rect x="60" y="252" width="560" height="52" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="340" y="272" text-anchor="middle" font-size="10" fill="#334155">传递的数据：KV 矩阵 [T, d] × H_kv（路径 A）或 Q 矩阵 [q, H_q, d] + partial (O, LSE)（路径 B）</text>
<text x="340" y="292" text-anchor="middle" font-size="10" fill="#64748b">通信方式：AllGather（KV 沿 T）/ broadcast+AllGather（Q 沿 head，DCP 与 TP 重叠时）+ LSE 感知归并</text>
</svg>

▲ GQA 的 CP：结构（左）与两条通信路径——传 KV 矩阵（中）或传 Q 矩阵（右）

## 4.2 MLA

传递Latent

MLA（Multi-head Latent Attention，DeepSeek-V2/V3）把每 token 的 K/V 压缩成一个共享的低秩 latent $c_{KV}$（$d_{kv}=512$）外加解耦 RoPE key $k_{pe}$，比 per-head KV 小约 32 倍。CP 时传递的是 **latent** 而非 per-head KV。

- **AllGather latent**（或 ring 传递 latent）即可让每个 rank 对全局历史做注意力；配合 absorb 形式（把 $W_{kv,b}$ 吸收进 Q，直接在 latent 上算注意力）是推理 decode 的标准做法，也是 vLLM DCP 支持 MLA 的路径（可选 `--dcp-q-replicate`：在 DCP 组内冗余复制 MLA 的 Q 投影，省掉每步的 query all-gather）。
- **Ulysses 对 MLA 失效**：吸收式下不存在 per-head KV 可供 a2a scatter，有效 Ulysses 并行度塌缩为 1，因此 MLA 的 CP 基本走 AllGather latent 或 ring 传 latent。
- **训练侧的坑**：Megatron-Core 明确禁止在训练中使用 absorbed 形式（代码里硬 assert）——absorb 的中间激活在 $n_h \times d_{kv}$ 维，比 per-head K/V 更大，激活显存多 20~34%（DeepSeek-V3 规模最多 9.2GB）。LAGA（Latent All-Gather Attention，arXiv:2607.17644）给出训练方案：通信仍只传 latent，但本地重建 per-head K/V 再做显式注意力，通信量降 1.98 倍、显存与显式形式持平，跨节点长序列吞吐提升 1.07~1.24 倍。
- KV Cache 复制案例：DeepSeek-R1 推理只有 1 个 KV head，tp=8 时 8 倍复制，DCP=8 消除——这是 DCP 最典型的应用场景。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 320" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="320" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">MLA 的 CP：传 latent 矩阵（而非 per-head KV）</text>
<text x="130" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">对比：per-head KV vs MLA latent</text>
<g>
<rect x="40" y="62" width="70" height="26" fill="#fca5a5" stroke="#dc2626" stroke-width="1.2"/>
<rect x="40" y="88" width="70" height="26" fill="#fca5a5" stroke="#dc2626" stroke-width="1.2"/>
<rect x="40" y="114" width="70" height="26" fill="#fca5a5" stroke="#dc2626" stroke-width="1.2"/>
<rect x="40" y="140" width="70" height="26" fill="#fca5a5" stroke="#dc2626" stroke-width="1.2"/>
<text x="75" y="79" text-anchor="middle" font-size="9" fill="#7f1d1d">K/V h0</text><text x="75" y="105" text-anchor="middle" font-size="9" fill="#7f1d1d">K/V h1</text>
<text x="75" y="131" text-anchor="middle" font-size="9" fill="#7f1d1d">K/V h2</text><text x="75" y="157" text-anchor="middle" font-size="9" fill="#7f1d1d">K/V h3</text>
<text x="75" y="184" text-anchor="middle" font-size="9" fill="#b91c1c">per-head KV：大（H × T）</text>
<rect x="140" y="62" width="36" height="104" fill="#c4b5fd" stroke="#7c3aed" stroke-width="1.2"/>
<rect x="182" y="62" width="18" height="104" fill="#ddd6fe" stroke="#7c3aed" stroke-width="1.2"/>
<text x="158" y="118" text-anchor="middle" font-size="9" fill="#4c1d95" transform="rotate(90 158 118)">c_KV [T, 512]</text>
<text x="191" y="118" text-anchor="middle" font-size="8" fill="#5b21b6" transform="rotate(90 191 118)">k_pe</text>
<text x="166" y="184" text-anchor="middle" font-size="9" fill="#6d28d9">MLA latent：小约 32 倍</text>
</g>
<text x="390" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">CP 通信：AllGather latent 矩阵（或 ring 传 latent）</text>
<g>
<rect x="250" y="62" width="46" height="30" fill="#ddd6fe"/><text x="273" y="81" text-anchor="middle" font-size="8" fill="#4c1d95">c⁰</text>
<rect x="250" y="92" width="46" height="30" fill="#c4b5fd"/><text x="273" y="111" text-anchor="middle" font-size="8" fill="#4c1d95">c¹</text>
<rect x="250" y="122" width="46" height="30" fill="#ddd6fe"/><text x="273" y="141" text-anchor="middle" font-size="8" fill="#4c1d95">c²</text>
<rect x="250" y="152" width="46" height="30" fill="#c4b5fd"/><text x="273" y="171" text-anchor="middle" font-size="8" fill="#4c1d95">c³</text>
<text x="273" y="198" text-anchor="middle" font-size="9" fill="#6d28d9">latent 分片 [T/C, 512]</text>
<text x="330" y="122" text-anchor="middle" font-size="10" fill="#6d28d9">AllGather →</text>
<rect x="360" y="62" width="120" height="120" rx="4" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
<g stroke="#ffffff" stroke-width="1.2">
<rect x="362" y="64" width="116" height="28" fill="#ddd6fe"/><rect x="362" y="92" width="116" height="28" fill="#c4b5fd"/>
<rect x="362" y="120" width="116" height="28" fill="#ddd6fe"/><rect x="362" y="148" width="116" height="28" fill="#c4b5fd"/>
</g>
<text x="420" y="112" text-anchor="middle" font-size="9" fill="#4c1d95">全局 latent [T, 512]</text>
<text x="420" y="198" text-anchor="middle" font-size="9" fill="#6d28d9">+ 解耦 k_pe [T, 64] 一并聚合</text>
<text x="545" y="86" text-anchor="middle" font-size="10" fill="#334155">absorb 形式：</text>
<text x="545" y="106" text-anchor="middle" font-size="10" fill="#334155">W_kv,b 吸进 Q</text>
<text x="545" y="126" text-anchor="middle" font-size="10" fill="#334155">→ 直接在 latent</text>
<text x="545" y="146" text-anchor="middle" font-size="10" fill="#334155">上算注意力</text>
<rect x="505" y="162" width="80" height="26" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="545" y="179" text-anchor="middle" font-size="9" fill="#14532d">decode 标准做法</text>
</g>
<rect x="60" y="216" width="560" height="88" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="340" y="238" text-anchor="middle" font-size="10" fill="#334155">传递的数据：latent c_KV [T, 512] + k_pe [T, 64]（每 token 一条，与 head 数无关）</text>
<text x="340" y="258" text-anchor="middle" font-size="10" fill="#334155">通信方式：AllGather（prefill / decode）或 ring P2P 传 latent；vLLM DCP 可选 --dcp-q-replicate 省掉每步 query all-gather</text>
<text x="340" y="278" text-anchor="middle" font-size="10" fill="#64748b">Ulysses 失效：absorb 式下无 per-head KV 可做 a2a scatter，有效 Ulysses 度塌缩为 1</text>
<text x="340" y="296" text-anchor="middle" font-size="10" fill="#64748b">训练侧：Megatron-Core 禁用 absorbed（激活回归 20~34%）；LAGA 通信只传 latent、本地重建 per-head K/V</text>
</svg>

▲ MLA 的 CP：latent 矩阵的形状对比（左）与 AllGather latent 通信（右）

## 4.3 DSA

DSA（DeepSeek Sparse Attention，DeepSeek-V3.2-Exp）由两部分组成。**Lightning Indexer**（轻量 MQA 式打分器，全层合计约 0.85B 参数）为每个 query 从全序列中选出 top-k 个最相关 token（64K 时 k≈2048），再对选中 token 做细粒度稀疏注意力，复杂度从 $O(L^2)$ 降到 $O(kL)$。

CP 下需要传递的数据有两类。

1. **MLA latent KV**（DSA 叠加在 MLA 之上）；
2. **Indexer 的 Key**（indexer key cache，避免 decode 重算）。

top-k 选择需要对全序列的 indexer key 打分，因此需要全局 key 视图——主流做法是 **AllGather latent KV + indexer key**，之后各 rank 只为本地 query 做 indexer 打分与稀疏注意力。在 weight-absorb 模式下 latent 与 indexer key 都很小，AllGather 的内存与通信代价完全可以接受；昇腾 TorchTitan 适配时明确对比过 RingAttention 与 AllGather KV 两条路，结论是 DSA 场景下 AllGather KV 实现简单且灵活，Ring 形式复杂且不划算。工程上还有以下配套做法。

- 用 **2C 头尾对称切片 + token 重排**做负载均衡（每层 Indexer 与稀疏注意力前把 KV 还原回因果顺序），SGLang 的 interleave 策略、vLLM PCP 的 DualChunkSwap 同理；
- SGLang Prefill CP 的 Interleave 策略对 DSA 有专门集成（按请求切分 query/indexer 元数据，按需 gather indexer key 与 MLA KV）；
- vLLM Ascend 曾实现独立的 `DSA-CP`（`enable_dsa_cp`），现已由通用的 PCP 路径承接（两者不能同时开启）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 330" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="330" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">DSA 的 CP：传 latent 矩阵 + indexer key 矩阵</text>
<text x="340" y="48" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">需要全局视图的两份数据（top-k 打分要扫全部 indexer key）</text>
<g>
<text x="150" y="72" text-anchor="middle" font-size="10" fill="#4c1d95">MLA latent [T, 512]</text>
<rect x="90" y="78" width="120" height="70" rx="3" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.2"/>
<g stroke="#ffffff" stroke-width="1">
<rect x="92" y="80" width="116" height="16" fill="#ddd6fe"/><rect x="92" y="96" width="116" height="16" fill="#c4b5fd"/>
<rect x="92" y="112" width="116" height="16" fill="#ddd6fe"/><rect x="92" y="128" width="116" height="16" fill="#c4b5fd"/>
</g>
<text x="340" y="72" text-anchor="middle" font-size="10" fill="#9a3412">Indexer key [T, d_idx]</text>
<rect x="280" y="78" width="120" height="70" rx="3" fill="#fff7ed" stroke="#ea580c" stroke-width="1.2"/>
<g stroke="#ffffff" stroke-width="1">
<rect x="282" y="80" width="116" height="16" fill="#fed7aa"/><rect x="282" y="96" width="116" height="16" fill="#fdba74"/>
<rect x="282" y="112" width="116" height="16" fill="#fed7aa"/><rect x="282" y="128" width="116" height="16" fill="#fdba74"/>
</g>
<text x="520" y="72" text-anchor="middle" font-size="10" fill="#1d4ed8">2C 头尾配对切分（负载均衡）</text>
<g font-size="9" text-anchor="middle">
<rect x="440" y="78" width="34" height="18" fill="#bfdbfe"/><rect x="476" y="78" width="34" height="18" fill="#bbf7d0"/><rect x="512" y="78" width="34" height="18" fill="#fde68a"/><rect x="548" y="78" width="34" height="18" fill="#fecaca"/>
<rect x="440" y="98" width="34" height="18" fill="#fecaca"/><rect x="476" y="98" width="34" height="18" fill="#fde68a"/><rect x="512" y="98" width="34" height="18" fill="#bbf7d0"/><rect x="548" y="98" width="34" height="18" fill="#bfdbfe"/>
<text x="510" y="132" fill="#64748b">query/indexer 按块 r 与块 2C−1−r 配对</text>
</g>
</g>
<text x="340" y="168" text-anchor="middle" font-size="10" fill="#c2410c">通信：AllGather latent + AllGather indexer key（weight-absorb 下两者都很小；Ring 形式复杂不划算）</text>
<text x="340" y="190" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">之后各 rank 本地完成：indexer top-k 打分 → 稀疏注意力（示意 mask 矩阵）</text>
<g>
<rect x="130" y="200" width="144" height="88" fill="#f8fafc" stroke="#94a3b8"/>
<g>
<rect x="132" y="202" width="16" height="20" fill="#e2e8f0"/><rect x="164" y="202" width="16" height="20" fill="#e2e8f0"/><rect x="180" y="202" width="16" height="20" fill="#86efac"/><rect x="212" y="202" width="16" height="20" fill="#e2e8f0"/><rect x="228" y="202" width="16" height="20" fill="#86efac"/><rect x="260" y="202" width="16" height="20" fill="#e2e8f0"/>
<rect x="132" y="222" width="16" height="20" fill="#86efac"/><rect x="148" y="222" width="16" height="20" fill="#e2e8f0"/><rect x="196" y="222" width="16" height="20" fill="#e2e8f0"/><rect x="212" y="222" width="16" height="20" fill="#86efac"/><rect x="244" y="222" width="16" height="20" fill="#e2e8f0"/><rect x="260" y="222" width="16" height="20" fill="#86efac"/>
<rect x="132" y="242" width="16" height="20" fill="#e2e8f0"/><rect x="148" y="242" width="16" height="20" fill="#86efac"/><rect x="180" y="242" width="16" height="20" fill="#e2e8f0"/><rect x="196" y="242" width="16" height="20" fill="#86efac"/><rect x="228" y="242" width="16" height="20" fill="#e2e8f0"/><rect x="244" y="242" width="16" height="20" fill="#86efac"/>
<rect x="132" y="262" width="16" height="20" fill="#86efac"/><rect x="164" y="262" width="16" height="20" fill="#e2e8f0"/><rect x="180" y="262" width="16" height="20" fill="#86efac"/><rect x="212" y="262" width="16" height="20" fill="#e2e8f0"/><rect x="228" y="262" width="16" height="20" fill="#86efac"/><rect x="260" y="262" width="16" height="20" fill="#e2e8f0"/>
</g>
<text x="202" y="304" text-anchor="middle" font-size="9" fill="#64748b">行 = query，列 = 全序列 key；绿 = top-k 选中（k≈2048 / 64K）</text>
</g>
<rect x="380" y="216" width="220" height="66" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="490" y="238" text-anchor="middle" font-size="10" fill="#334155">每 rank 只为本地 query 做：</text>
<text x="490" y="256" text-anchor="middle" font-size="10" fill="#334155">① indexer 打分（对全局 indexer key）</text>
<text x="490" y="274" text-anchor="middle" font-size="10" fill="#334155">② top-k 稀疏注意力（对全局 latent）</text>
</svg>

▲ DSA 的 CP：传递 latent 与 indexer key 两份矩阵（AllGather），本地做 top-k 稀疏注意力（mask 为动态选中列）

## 4.4 GDN

需要传递状态S

GDN（Gated DeltaNet）是线性注意力，用固定大小的循环状态 $S \in \mathbb{R}^{d_v \times d_k}$ 替代随序列增长的 KV Cache，门控遗忘 + delta rule 定向覆盖。

$$S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^\top, \qquad o_t = S_t q_t$$

$\alpha_t \in (0,1]$ 控制旧状态整体遗忘，$\beta_t$ 控制当前 key 关联的写入强度。CP 下的特征：

- **需要传递的是状态 $S$**：没有 KV Cache 可传，也不能用 AllGather KV 方案；状态大小固定（例如 Qwen3-Next 每 head 128×128），跨 rank 同步状态的通信量**与序列长度无关**。
- **串行依赖**：$S_t$ 依赖 $S_{t-1}$，训练/预填充用 **chunk-wise 并行**（chunk 内矩阵并行、chunk 间串行传状态）摊开；CP 时可以沿序列在 rank 间传递状态（类似 ring 的状态链），或者各段先独立算"状态转移"再精确合成。
- 兜底方案还有重算和 Ulysses（按 head 切），但状态链传递是通信量最优的。
- 实际模型多为混合架构，如 Qwen3.5/Qwen3.8-Flash-Next 大部分层用 GDN、少数层保留全局注意力，CP 时两类层分别走状态传递与常规注意力通信。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 320" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="320" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">GDN 的 CP：只传固定大小的状态矩阵 S</text>
<text x="340" y="48" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">序列分段 → 各段独立求 (Φᵢ, Bᵢ) → 状态链传播（无 KV Cache 可传）</text>
<g font-size="10" text-anchor="middle">
<rect x="70" y="62" width="120" height="26" fill="#bfdbfe"/><text x="130" y="79" fill="#1e3a8a">段 0（rank 0）</text>
<rect x="200" y="62" width="120" height="26" fill="#bbf7d0"/><text x="260" y="79" fill="#14532d">段 1（rank 1）</text>
<rect x="330" y="62" width="120" height="26" fill="#fde68a"/><text x="390" y="79" fill="#92400e">段 2（rank 2）</text>
<rect x="460" y="62" width="120" height="26" fill="#fecaca"/><text x="520" y="79" fill="#7f1d1d">段 3（rank 3）</text>
</g>
<g font-size="9" text-anchor="middle" fill="#475569">
<text x="130" y="108">本地算 (Φ₀, B₀)</text><text x="260" y="108">本地算 (Φ₁, B₁)</text><text x="390" y="108">本地算 (Φ₂, B₂)</text><text x="520" y="108">本地算 (Φ₃, B₃)</text>
<text x="130" y="122">零通信、可并行</text><text x="260" y="122">零通信、可并行</text><text x="390" y="122">零通信、可并行</text><text x="520" y="122">零通信、可并行</text>
</g>
<g>
<rect x="96" y="142" width="48" height="48" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.2"/>
<g stroke="#ffffff" stroke-width="1"><rect x="98" y="144" width="22" height="22" fill="#a5b4fc"/><rect x="120" y="144" width="22" height="22" fill="#c7d2fe"/><rect x="98" y="166" width="22" height="22" fill="#c7d2fe"/><rect x="120" y="166" width="22" height="22" fill="#a5b4fc"/></g>
<text x="120" y="206" text-anchor="middle" font-size="9" fill="#3730a3">S₀ [d_v, d_k]</text>
<rect x="276" y="142" width="48" height="48" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.2"/>
<g stroke="#ffffff" stroke-width="1"><rect x="278" y="144" width="22" height="22" fill="#c7d2fe"/><rect x="300" y="144" width="22" height="22" fill="#a5b4fc"/><rect x="278" y="166" width="22" height="22" fill="#a5b4fc"/><rect x="300" y="166" width="22" height="22" fill="#c7d2fe"/></g>
<text x="300" y="206" text-anchor="middle" font-size="9" fill="#3730a3">S₁ = S₀Φ₀ + B₀</text>
<rect x="456" y="142" width="48" height="48" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.2"/>
<g stroke="#ffffff" stroke-width="1"><rect x="458" y="144" width="22" height="22" fill="#a5b4fc"/><rect x="480" y="144" width="22" height="22" fill="#c7d2fe"/><rect x="458" y="166" width="22" height="22" fill="#c7d2fe"/><rect x="480" y="166" width="22" height="22" fill="#a5b4fc"/></g>
<text x="480" y="206" text-anchor="middle" font-size="9" fill="#3730a3">S₂ = S₁Φ₁ + B₁</text>
</g>
<g stroke="#7c3aed" stroke-width="2" fill="none">
<path d="M 152 166 L 266 166"/><path d="M 332 166 L 446 166"/><path d="M 512 166 L 590 166"/>
</g>
<text x="210" y="158" text-anchor="middle" font-size="9" fill="#6d28d9">P2P 传 S（固定大小）</text>
<text x="390" y="158" text-anchor="middle" font-size="9" fill="#6d28d9">P2P 传 S（固定大小）</text>
<text x="556" y="158" text-anchor="middle" font-size="9" fill="#6d28d9">→ S₃ → S_out</text>
<rect x="90" y="228" width="500" height="72" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="340" y="250" text-anchor="middle" font-size="10" fill="#334155">传递的数据：状态矩阵 S ∈ ℝ^(d_v×d_k)（或等价段级摘要 (Φᵢ, Bᵢ)）—— 大小固定，例如每 head 128×128</text>
<text x="340" y="270" text-anchor="middle" font-size="10" fill="#334155">通信方式：沿序列顺序的 P2P 状态链（类 ring，但传的是 S 而非 KV）；也可两两合成摘要后一次性求末状态</text>
<text x="340" y="290" text-anchor="middle" font-size="10" fill="#64748b">通信量与序列长度无关；段内输出 o_t = S_t q_t 在 S_in 到位后本地并行算</text>
</svg>

▲ GDN 的 CP：无 KV 可传，rank 间只传固定大小的状态矩阵 S

**推导过程**

**① 从线性注意力到循环状态。** 去掉 softmax 后，因果注意力可以写成 $o_t = \sum_{s \le t} (q_t^\top k_s) v_s$。定义状态 $S_t = \sum_{s \le t} v_s k_s^\top \in \mathbb{R}^{d_v \times d_k}$，立刻得到递推形式

$$S_t = S_{t-1} + v_t k_t^\top, \qquad o_t = S_t q_t$$

状态大小固定，不随序列增长。但它只能**叠加**，同一个 key 写入 $v_1$、$v_2$ 之后读回的是 $v_1 + v_2$，旧关联无法被覆盖。

**② Delta rule（定向覆盖）。** 写入前先把 $k_t$ 处的旧关联减掉，只写"修正量"。

$$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t)\, k_t^\top$$

展开右式 $= S_{t-1} - \beta_t S_{t-1} k_t k_t^\top + \beta_t v_t k_t^\top$，合并得等价形式

$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

即"先在 $k_t$ 方向擦除旧值、再写入 $v_t$"，$\beta_t \in (0,1]$ 是写入步长。同一 key 连续写 $v_1、v_2$ 后读回恰为 $v_2$（可直接验证），这就是 delta rule 的"定向覆盖"。

**③ 加遗忘门（Gated DeltaNet）。** 对旧状态整体乘衰减 $\alpha_t \in (0,1]$。

$$S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t)\, k_t^\top \;\equiv\; S_{t-1} T_t + \beta_t v_t k_t^\top, \qquad T_t = \alpha_t (I - \beta_t k_t k_t^\top)$$

等价性：右式展开 $S_{t-1}\,\alpha_t(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top = \alpha_t S_{t-1} + \beta_t(v_t - \alpha_t S_{t-1} k_t) k_t^\top$。于是整层递推是**右乘线性转移 $T_t$ 加一次秩一写入**，输出 $o_t = S_t q_t$。$\alpha$ 负责"整体快速遗忘"，delta 负责"定向覆盖"，两者互补。

**④ 段级分解——为什么各段可以独立求值再合成。** 对长度为 $L$ 的一段递推展开。

$$S_L = S_0\, \Phi_{0 \to L} + \sum_{s=1}^{L} \beta_s v_s k_s^\top\, \Phi_{s \to L}, \qquad \Phi_{s \to L} = T_{s+1} T_{s+2} \cdots T_L \;(\Phi_{L \to L} = I)$$

即 $S_L = S_0 \Phi + B$，其中 $B = \sum_s \beta_s v_s k_s^\top \Phi_{s \to L}$ **完全由该段的 $(k, v, \alpha, \beta)$ 决定，与输入状态 $S_0$ 无关**。这带来 CP 需要的两个关键性质。

1. 各 rank 可先**零通信地并行**求出本段的 $(\Phi, B)$；
2. 段间只需把**固定大小**的状态沿链传播，$S^{(i)}_{\text{out}} = S^{(i)}_{\text{in}} \Phi_i + B_i$；也可以两两合成段级摘要（$\Phi_{12} = \Phi_1 \Phi_2,\ B_{12} = B_1 \Phi_2 + B_2$）后一次性得到末状态。段内各位置输出 $o_t = (S_{\text{in}} \Phi_{\text{in} \to t} + B_{\text{in} \to t}) q_t$ 在 $S_{\text{in}}$ 到位后本地并行算。

**这就是"需要传递状态 $S$"的来源**。CP 之间要传的只是 $d_v \times d_k$ 的状态（或等价的段级摘要），通信量与序列长度无关。

**⑤ Chunk-wise 并行（GPU 上的落地）。** 标量 $\alpha_t$ 可以提到连乘外面，于是

$$\Phi = \Big(\prod_t \alpha_t\Big) \prod_t (I - \beta_t k_t k_t^\top) = \Big(\prod_t \alpha_t\Big)\,(I - K C K^\top)$$

其中 $K = [k_1, \dots, k_L]$，$C$ 是由 $\beta_t$ 与两两内积 $k_i^\top k_j$ 决定的 $L \times L$ 小矩阵（WY / Woodbury 表示）。段内计算因此归结为几次 GEMM 加一个小三角求解，而不再是 $L$ 步串行递推；chunk 之间再串行传状态。FlashKDA、FLA 系 kernel 的 chunkwise 算法都基于这一结构，CP 时的"chunk 内并行、rank 间传状态"也由此而来。

## 4.5 KDA

需要采用KCP

KDA（Kimi Delta Attention，Kimi Linear，arXiv:2510.26692）是 Gated DeltaNet 的细粒度升级，把标量遗忘门换成**逐通道**遗忘门 $\alpha_t \in [0,1]^{d_k}$（对角转移矩阵），状态仍是固定大小，配 DPLR（对角加低秩）转移与 chunkwise 算法。Kimi-K3（2.8T 参数、1M 上下文）采用 3 层 KDA + 1 层 Gated MLA 的混合结构。

KDA 的 CP 必须用专门的 **KCP（KDA Context Parallelism）**。

- 对标准线性注意力，每个序列分段的状态转移可以在不知道输入状态的情况下独立求值，再精确合成递归链；但 KDA 的逐通道遗忘门在数学上使这种"独立求值再合成"不能直接成立。
- KCP 把状态转移中**可独立计算的部分尽可能拆解出来**，分发到不同 GPU 进程并行计算，再合成各段应得的状态；此外还支持**单卡内 SM 级**的序列二次切分（intra-device KCP）与跨设备 KCP 两条线并行——长序列（百万 token）时仅靠 TP 切 head 会让 prefill 的 SM 大量空转。
- 跨设备只同步**固定大小的状态**，通信成本恒定，不随文本变长。
- 配套工程：FlashKDA（基于 CUTLASS 的 chunkwise kernel，chunk 内并行与跨 chunk 状态传播重叠）、state-aware prefix caching（固定大小状态天然适合缓存复用）。
- 生态：SGLang 针对 Kimi-K3 提供 fused KDA decode kernel、KDA-aware prefix caching、DP attention、MTP 与 PD 分离支持。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 330" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="330" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">KDA 的 CP（KCP）：独立量并行求值 + 固定状态合成</text>
<text x="170" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">KDA 转移：D_t（对角）+ 秩一</text>
<g>
<rect x="90" y="62" width="44" height="44" fill="#f8fafc" stroke="#94a3b8"/>
<g stroke="#ffffff" stroke-width="1">
<rect x="92" y="64" width="12" height="12" fill="#fbbf24"/><rect x="104" y="64" width="12" height="12" fill="#e2e8f0"/><rect x="116" y="64" width="12" height="12" fill="#e2e8f0"/>
<rect x="92" y="76" width="12" height="12" fill="#e2e8f0"/><rect x="104" y="76" width="12" height="12" fill="#fbbf24"/><rect x="116" y="76" width="12" height="12" fill="#e2e8f0"/>
<rect x="92" y="88" width="12" height="12" fill="#e2e8f0"/><rect x="104" y="88" width="12" height="12" fill="#e2e8f0"/><rect x="116" y="88" width="12" height="12" fill="#fbbf24"/>
</g>
<text x="112" y="122" text-anchor="middle" font-size="9" fill="#92400e">D_t = Diag(α_t)</text>
<text x="112" y="136" text-anchor="middle" font-size="9" fill="#64748b">逐通道遗忘门</text>
<rect x="150" y="62" width="44" height="44" fill="#f8fafc" stroke="#94a3b8"/>
<g stroke="#ffffff" stroke-width="1">
<rect x="152" y="64" width="12" height="12" fill="#93c5fd"/><rect x="164" y="64" width="12" height="12" fill="#93c5fd"/><rect x="176" y="64" width="12" height="12" fill="#93c5fd"/>
<rect x="152" y="76" width="12" height="12" fill="#93c5fd"/><rect x="164" y="76" width="12" height="12" fill="#93c5fd"/><rect x="176" y="76" width="12" height="12" fill="#93c5fd"/>
<rect x="152" y="88" width="12" height="12" fill="#93c5fd"/><rect x="164" y="88" width="12" height="12" fill="#93c5fd"/><rect x="176" y="88" width="12" height="12" fill="#93c5fd"/>
</g>
<text x="172" y="122" text-anchor="middle" font-size="9" fill="#1d4ed8">−β(D_tk)kᵀ</text>
<text x="172" y="136" text-anchor="middle" font-size="9" fill="#64748b">秩一修正</text>
<text x="135" y="158" text-anchor="middle" font-size="10" fill="#334155">T_t = D_t − β_t(D_tk_t)k_tᵀ（DPLR）</text>
</g>
<text x="460" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">KCP 两级并行</text>
<g font-size="9" text-anchor="middle">
<rect x="300" y="62" width="100" height="40" rx="5" fill="#dcfce7" stroke="#16a34a"/><text x="350" y="80" fill="#14532d">独立量求值</text><text x="350" y="95" fill="#14532d">∏D_t / 低秩因子 / KV 统计</text>
<rect x="412" y="62" width="100" height="40" rx="5" fill="#dcfce7" stroke="#16a34a"/><text x="462" y="80" fill="#14532d">独立量求值</text><text x="462" y="95" fill="#14532d">拆到不同 GPU 进程</text>
<rect x="524" y="62" width="100" height="40" rx="5" fill="#dcfce7" stroke="#16a34a"/><text x="574" y="80" fill="#14532d">独立量求值</text><text x="574" y="95" fill="#14532d">（零通信、并行）</text>
</g>
<g stroke="#7c3aed" stroke-width="2" fill="none">
<path d="M 350 106 L 420 148"/><path d="M 462 106 L 455 148"/><path d="M 574 106 L 490 148"/>
</g>
<rect x="392" y="150" width="126" height="34" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
<text x="455" y="166" text-anchor="middle" font-size="9" fill="#4c1d95">状态合成</text>
<text x="455" y="180" text-anchor="middle" font-size="9" fill="#4c1d95">只传 S [d_v, d_k]</text>
<rect x="300" y="202" width="324" height="48" rx="6" fill="#f8fafc" stroke="#94a3b8"/>
<text x="462" y="220" text-anchor="middle" font-size="9" fill="#334155">跨设备 KCP：沿段链 P2P 传固定大小状态</text>
<text x="462" y="238" text-anchor="middle" font-size="9" fill="#334155">卡内 KCP：SM 级序列二次切分，各 SM 算完再合并状态</text>
<rect x="90" y="262" width="500" height="52" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="340" y="282" text-anchor="middle" font-size="10" fill="#334155">传递的数据：状态 S（或段级转移摘要）+ 各段独立计算的中间量就地合成</text>
<text x="340" y="302" text-anchor="middle" font-size="10" fill="#64748b">通信方式：跨设备 P2P 状态链（成本恒定）；与 FlashKDA chunkwise kernel、state-aware prefix caching 配套</text>
</svg>

▲ KDA 的 CP（KCP）：DPLR 转移矩阵（左）与"独立量并行 + 固定状态合成"两级并行（右）

**推导过程**

**① KDA 的更新律与 DPLR 转移。** 把 GDN 的标量遗忘门换成**逐通道（对角）**遗忘门 $\alpha_t \in [0,1]^{d_k}$，$D_t = \mathrm{Diag}(\alpha_t)$。

$$S_t = S_{t-1}\, D_t (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top \;=\; S_{t-1}\, T_t + \beta_t v_t k_t^\top, \qquad o_t = S_t q_t$$

单步转移可以整理成

$$T_t = D_t - \beta_t (D_t k_t)\, k_t^\top$$

即 **"对角 + 秩一"**——这正是 DPLR（Diagonal-Plus-Low-Rank）转移矩阵；Kimi Linear 的 chunkwise 算法对这一特殊 DPLR 形式做了专门化（比通用 DPLR 求解省掉大量计算，同时保持与经典 delta rule 一致的语义）。逐通道门的意义：不同 key 维度可以有自己的遗忘速率，固定大小状态的记忆利用率更高。

**② 段级仿射结构（形式上仍然成立）。** 与 GDN 完全相同地展开：

$$S_L = S_0\, \Phi_{0 \to L} + \sum_{s=1}^{L} \beta_s v_s k_s^\top\, \Phi_{s \to L}, \qquad \Phi_{s \to L} = T_{s+1} \cdots T_L$$

"每段先算 $(\Phi, B)$、再合成"在代数上依旧成立——问题变成**如何高效地表示与计算 $\Phi$**。

**③ 为什么 GDN 的"独立求值再合成"不能直接搬到 KDA。** GDN 的 $T_t = \alpha_t(I - \beta_t k_t k_t^\top)$ 中标量 $\alpha_t$ 与一切对易，可以提到连乘外面。

$$\Phi_{\text{GDN}} = \Big(\prod_t \alpha_t\Big)\,(I - K C K^\top)$$

紧凑结构（对角缩放 × 秩一修正的连乘）在段内 keys 张成的空间里**封闭**，段间合成只是小矩阵运算。KDA 中各 $D_t$ 是互不相同的对角阵，与秩一项**不对易**。

$$D(I - \beta k k^\top) = D - \beta (Dk) k^\top \;\neq\; (I - \beta k k^\top) D = D - \beta k (Dk)^\top$$

（除非 $k$ 落在退化方向上）。于是 $D_t$ 不能像标量那样外提，$(I - KCK^\top)$ 的 WY 紧凑形式不再封闭，连乘 $D_1R_1D_2R_2\cdots$（$R_t = I - \beta_t k_t k_t^\top$）会不断混合对角与低秩结构，直接套用通用 DPLR 合成的代价随段长增长。这就是公开材料里说的"KDA 的状态转移函数具有记忆遗忘机制，在数学上导致额外困难，（标准线性注意力的）独立求值再合成并不直接可行"。

**④ KCP 的拆解思路。** KCP 的做法是把状态转移中**可以独立计算的部分尽可能拆解出来**，再分发并行。

1. **段内与输入状态无关的量照常并行求值**：key/value 统计、累积对角衰减 $\prod_t D_t$、DPLR 转移中的低秩因子等，拆到不同 GPU 进程独立计算；
2. **只沿段链传递固定大小的状态**（或等价的段级转移摘要），把它与各段的独立计算结果合成——跨设备同步的就是 $d_v \times d_k$ 的状态，**通信成本恒定，不随文本变长**；
3. **两级并行**：除跨设备 KCP 外，还有单卡内 **SM 级**的序列二次切分（intra-device KCP）——百万 token 时只靠 TP 切 head 会让 prefill 的 SM 空转，KCP 把序列在 SM 之间再切一段，各自算完状态再合并，与跨设备 KCP 同时进行；
4. **工程配套**：FlashKDA 的 chunkwise kernel 把 chunk 内并行与跨 chunk 状态传播重叠；state-aware prefix caching 直接缓存固定大小的状态。

（KCP 完整数学细节以 Kimi K3 技术资料为准；公开材料给出的核心即上述"可独立计算部分拆解 + 固定状态合成"。）

## 4.6 CSA

CSA（Compressed Sparse Attention，压缩稀疏注意力，DeepSeek-V4）是 DSA 思路的进一步演化，核心是"先压缩、再稀疏选择"。

1. **KV 压缩**：用带学习位置偏差的 softmax-gated pooling，把每 $m=4$ 个 token 的 KV 条目压缩成 1 条压缩条目，序列长度缩为 $1/m$；
2. **稀疏选择**：Lightning Indexer（FP4 精度）对压缩条目打分，每个 query 只取 top-k（512~1024）个压缩块做注意力；
3. **SWA 分支**保留最近窗口内的精细局部依赖；与 HCA（Heavily Compressed Attention，压缩率约 128 倍后做稠密注意力）在层间交替堆叠（V4-Pro 61 层 CSA/HCA 交替），兼顾"近处细节、中程精准检索、远程全局覆盖"。1M 上下文下 KV Cache 降到 V3 的 10%、单 token FLOPs 降到 27%。

CP 下需要传递的数据与注意事项：

- 传递的是**压缩 KV 条目**（以及独立缓存时的 **indexer key**），体量只有原始 KV 的 $1/m$，比 DSA 的 AllGather 更轻，"传 KV / AllGather KV"几乎是自然选择；
- top-k 选择同样需要对**全部压缩条目**打分，需要全局（压缩后的）key 视图，做法与 DSA 相同；
- 压缩组（每 $m$ 个 token）应作为 CP 切分的最小粒度，避免把一个压缩组切断到两个 rank；HCA 的高压缩摘要同理；
- SWA 分支只依赖局部窗口，基本不需要跨 rank 通信（窗口跨界除外）。
- 演化路线：NSA（2025-02，压缩/选择/滑窗三分支、块级稀疏）→ DSA（V3.2，token 级动态稀疏 + Lightning Indexer）→ CSA（V4，KV 压缩 + 稀疏选择）。SGLang 的 DeepSeek-V4 后端（multi-ratio compressed KV pools + FlashMLA + 稀疏索引）与 Prefill CP 的 "DeepSeek V4 interleave" 路径已有对应集成。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 330" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="330" fill="#ffffff"/>
<text x="340" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">CSA 的 CP：传压缩 KV 矩阵（1/m）+ indexer key</text>
<text x="140" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">KV 压缩：每 m=4 条 → 1 条</text>
<g>
<rect x="70" y="62" width="40" height="12" fill="#fdba74"/><rect x="70" y="74" width="40" height="12" fill="#fed7aa"/><rect x="70" y="86" width="40" height="12" fill="#fdba74"/><rect x="70" y="98" width="40" height="12" fill="#fed7aa"/>
<rect x="70" y="118" width="40" height="12" fill="#fdba74"/><rect x="70" y="130" width="40" height="12" fill="#fed7aa"/><rect x="70" y="142" width="40" height="12" fill="#fdba74"/><rect x="70" y="154" width="40" height="12" fill="#fed7aa"/>
<text x="90" y="180" text-anchor="middle" font-size="9" fill="#9a3412">原始 KV [T, d]</text>
<path d="M 116 84 L 148 84" stroke="#64748b" stroke-width="1.5" fill="none"/>
<path d="M 116 136 L 148 136" stroke="#64748b" stroke-width="1.5" fill="none"/>
<rect x="152" y="68" width="40" height="26" fill="#f97316" stroke="#ea580c" stroke-width="1.2"/>
<rect x="152" y="120" width="40" height="26" fill="#f97316" stroke="#ea580c" stroke-width="1.2"/>
<text x="172" y="86" text-anchor="middle" font-size="9" fill="#7c2d12">条目 0</text>
<text x="172" y="138" text-anchor="middle" font-size="9" fill="#7c2d12">条目 1</text>
<text x="172" y="180" text-anchor="middle" font-size="9" fill="#c2410c">压缩 KV [T/4, d]</text>
</g>
<text x="330" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">CP 通信：AllGather 压缩条目 + indexer key</text>
<g>
<rect x="240" y="62" width="56" height="20" fill="#fdba74"/><text x="268" y="76" text-anchor="middle" font-size="8" fill="#7c2d12">压缩 0</text>
<rect x="240" y="82" width="56" height="20" fill="#fed7aa"/><text x="268" y="96" text-anchor="middle" font-size="8" fill="#7c2d12">压缩 1</text>
<rect x="240" y="102" width="56" height="20" fill="#fdba74"/><text x="268" y="116" text-anchor="middle" font-size="8" fill="#7c2d12">压缩 2</text>
<rect x="240" y="122" width="56" height="20" fill="#fed7aa"/><text x="268" y="136" text-anchor="middle" font-size="8" fill="#7c2d12">压缩 3</text>
<text x="268" y="160" text-anchor="middle" font-size="9" fill="#c2410c">各 rank 本地分片</text>
<text x="340" y="106" font-size="10" fill="#c2410c">AllGather →</text>
<rect x="392" y="62" width="90" height="80" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
<g stroke="#ffffff" stroke-width="1">
<rect x="394" y="64" width="86" height="18" fill="#fed7aa"/><rect x="394" y="82" width="86" height="18" fill="#fdba74"/><rect x="394" y="100" width="86" height="18" fill="#fed7aa"/><rect x="394" y="118" width="86" height="18" fill="#fdba74"/>
</g>
<text x="437" y="160" text-anchor="middle" font-size="9" fill="#c2410c">全局压缩 KV [T/4, d]</text>
<text x="560" y="86" text-anchor="middle" font-size="9" fill="#334155">体量只有原始 KV 的 1/m</text>
<text x="560" y="106" text-anchor="middle" font-size="9" fill="#334155">+ indexer key（FP4）一并聚合</text>
<text x="560" y="126" text-anchor="middle" font-size="9" fill="#334155">压缩组（m 个 token）为切分</text>
<text x="560" y="142" text-anchor="middle" font-size="9" fill="#334155">最小粒度，不可切断</text>
</g>
<text x="340" y="188" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">之后本地：indexer 对压缩条目 top-k → 稀疏注意力 + SWA 局部分支</text>
<g>
<rect x="120" y="200" width="100" height="72" fill="#f8fafc" stroke="#94a3b8"/>
<g>
<rect x="122" y="202" width="18" height="16" fill="#e2e8f0"/><rect x="140" y="202" width="18" height="16" fill="#86efac"/><rect x="158" y="202" width="18" height="16" fill="#e2e8f0"/><rect x="176" y="202" width="18" height="16" fill="#86efac"/><rect x="194" y="202" width="18" height="16" fill="#e2e8f0"/>
<rect x="122" y="218" width="18" height="16" fill="#86efac"/><rect x="140" y="218" width="18" height="16" fill="#e2e8f0"/><rect x="158" y="218" width="18" height="16" fill="#86efac"/><rect x="176" y="218" width="18" height="16" fill="#e2e8f0"/><rect x="194" y="218" width="18" height="16" fill="#86efac"/>
<rect x="122" y="234" width="18" height="16" fill="#e2e8f0"/><rect x="140" y="234" width="18" height="16" fill="#86efac"/><rect x="158" y="234" width="18" height="16" fill="#e2e8f0"/><rect x="176" y="234" width="18" height="16" fill="#86efac"/><rect x="194" y="234" width="18" height="16" fill="#e2e8f0"/>
<rect x="122" y="250" width="18" height="16" fill="#86efac"/><rect x="140" y="250" width="18" height="16" fill="#e2e8f0"/><rect x="158" y="250" width="18" height="16" fill="#86efac"/><rect x="176" y="250" width="18" height="16" fill="#e2e8f0"/><rect x="194" y="250" width="18" height="16" fill="#86efac"/>
</g>
<text x="170" y="288" text-anchor="middle" font-size="9" fill="#64748b">稀疏选择：行=query，列=压缩条目</text>
<rect x="270" y="200" width="100" height="72" fill="#f8fafc" stroke="#94a3b8"/>
<g>
<rect x="272" y="202" width="18" height="16" fill="#86efac"/><rect x="290" y="202" width="18" height="16" fill="#86efac"/><rect x="308" y="202" width="18" height="16" fill="#e2e8f0"/><rect x="326" y="202" width="18" height="16" fill="#e2e8f0"/><rect x="344" y="202" width="18" height="16" fill="#e2e8f0"/>
<rect x="272" y="218" width="18" height="16" fill="#86efac"/><rect x="290" y="218" width="18" height="16" fill="#86efac"/><rect x="308" y="218" width="18" height="16" fill="#86efac"/><rect x="326" y="218" width="18" height="16" fill="#e2e8f0"/><rect x="344" y="218" width="18" height="16" fill="#e2e8f0"/>
<rect x="272" y="234" width="18" height="16" fill="#e2e8f0"/><rect x="290" y="234" width="18" height="16" fill="#86efac"/><rect x="308" y="234" width="18" height="16" fill="#86efac"/><rect x="326" y="234" width="18" height="16" fill="#86efac"/><rect x="344" y="234" width="18" height="16" fill="#e2e8f0"/>
<rect x="272" y="250" width="18" height="16" fill="#e2e8f0"/><rect x="290" y="250" width="18" height="16" fill="#e2e8f0"/><rect x="308" y="250" width="18" height="16" fill="#86efac"/><rect x="326" y="250" width="18" height="16" fill="#86efac"/><rect x="344" y="250" width="18" height="16" fill="#86efac"/>
</g>
<text x="320" y="288" text-anchor="middle" font-size="9" fill="#64748b">SWA 分支：对角带状 mask，基本不通信</text>
<rect x="430" y="208" width="180" height="60" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="520" y="228" text-anchor="middle" font-size="9" fill="#334155">传递的数据：压缩 KV [T/4, d] + indexer key</text>
<text x="520" y="248" text-anchor="middle" font-size="9" fill="#334155">通信方式：AllGather（比 DSA 更轻）</text>
<text x="520" y="264" text-anchor="middle" font-size="9" fill="#64748b">SWA 窗口跨界时少量补传</text>
</g>
</svg>

▲ CSA 的 CP：KV 压缩矩阵（左）、AllGather 压缩条目（中）、稀疏选择与 SWA 两种 mask（右）

# 5. 推理和训练引擎的实现

## 5.1 vLLM

vLLM 把 PCP 与 DCP 做成了两个正交的命令行维度（讨论集中在 vLLM Slack `#sig-context-parallel`）。

**PCP（Prefill Context Parallel）**：`--prefill-context-parallel-size` / `-pcp`

- 按序列切分 prefill 计算，**扩展进程 world size**，但不增加 KV cache 的分片数；
- 两种策略（官方文档）："Partial query, full key/value"（AllGather KV，摊薄 TTFT）与 "Partial query, partial key/value"（ring 式传 KV，应对放不下全局 KV 的超长请求），均在持续开发中；
- 头尾式（DualChunkSwap）token 分区保证负载均衡，并用 restore 索引恢复原始 token 顺序（`pcp_allgather_restore_idx`）；
- PD 分离部署中 PCP 只开在 prefill（kv_producer）引擎上，decode 侧 `prefill_context_parallel_size=1`。

**DCP（Decode Context Parallel）**：`--decode-context-parallel-size` / `-dcp`

- 沿 T 维分片 decode KV Cache，**不扩大 world size**（复用 TP/PCP rank），`dcp_size ∈ [1, tp_size / H_kv]`；
- 交错分片由 `--cp_kv_cache_interleave_size` 控制（token 级 `i % dcp_world_size` 或 block 级对齐）；KV 传输（KV pooling / PD 分离）时建议设为 KV cache 的 block_size（默认 128）；
- 通信后端 `--dcp-comm-backend`：`ag_rs`（AllGather + ReduceScatter）或 `a2a`（交换 partial output + LSE 后用 Triton kernel 合并，MLA 模型每层 NCCL 调用从 3 次降到 2 次）；
- `--dcp-q-replicate`：在 DCP 组内冗余复制 MLA 的 Q 投影，让 decode 跳过每步的 query all-gather；
- 支持 MLA 与 GQA 模型，部分注意力后端还支持 DCP + MTP（multi-token prediction）组合加速解码。

## 5.2 SGLang

**Prefill CP**（2025-12 由百度百舸 AIAK 贡献合入主线，PR #12065）：

- 把 prompt 的 token 分布到 attention CP 组，各 rank 为本地 query 计算注意力（含 indexer），通过 **all-gather KV cache** 拿到所需 K/V；
- `ContextParallelStrategy` 把 token 布局策略与模型执行/注意力 kernel 解耦，每进程一个策略实例，per-forward 状态放在 `ForwardBatch.attn_cp_metadata`；
- 两种切分策略：
  - **Zigzag**：每请求新增 token 切成 $2C$ 个连续块，rank $r$ 拿块 $r$ 与块 $2C-1-r$（每请求独立重启切分；长度不整除时前段块拿余数），保留连续 query 块且因果负载均衡；
  - **Interleave**：rank $r$ 拿展平索引 $r, r+C, r+2C, \dots$（索引跨 batch 新增 token 连续、不按请求重置，position id 保持原值），query 分布更均匀，配合 DSA/DeepSeek V4 的稀疏注意力集成（gather indexer key 与 MLA KV）；
- 后端集成路径：Zigzag + FlashAttention（gather 并重排新 KV）、Zigzag + trtllm_mha（zigzag page table 的变长注意力）、Interleave + DSA、DeepSeek V4 interleave；
- 收益：把 indexer（DSA/DeepSeek V4）这类原本各 rank 重复的工作切开；与 KV cache 分片（DSA cache LayerSplit）叠加省显存；与 EP 组合时用 all-to-all token dispatch/combine 替代 TP all-reduce。

**Decode CP**：

- 采用**传 Q** 策略，即"an all_gather of q is done for decode stage"，各 rank 对本地 KV 分片算注意力后合并（在线 softmax/LSE）；
- RadixTree 的处理：逻辑树留在中心 scheduler，物理存储按 rank 分片（一条上下文序列的 KV cache 切到各 dcp rank），调度时把逻辑节点映射到物理 GPU rank；
- PD 分离下 CP 感知的 KV 传输：`SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 控制由 CP rank 0 单独传输还是所有 CP rank 各传自己那部分。

模型支持面上，SGLang 已适配混合注意力模型（GLM-5.3-Flash 的 MLA + DSA + KDA、Kimi-K3 的 fused KDA decode + KDA-aware prefix caching、Qwen3.8-Flash-Next 的 GDN + 稀疏注意力等），CP 需要按层类型分别处理 KV / latent / 状态 / indexer key。

## 5.3 Megatron

Megatron-LM / Megatron-Core 的 CP 是训练侧的工业级实现。

- 入口参数：`--context-parallel-size`、`--cp-comm-type`（`p2p` / `all_gather` / `a2a` / `a2a+p2p`）、`hierarchical_context_parallel_sizes`（内层 a2a 尺寸 × 外层 p2p 尺寸）、`max_seqlen_per_dp_cp_rank`；
- **p2p（默认）**即 Ring Attention，zigzag 切分成 $2 \times cp\_size$ 块（rank $k$ 拿块 $k$ 与 $2cp-1-k$），双 CUDA stream 做计算-通信重叠，与 FlashAttention-2 深度集成，LSE 合并部分结果；
- **a2a** 即 Ulysses；**all_gather** 为 AllGather KV；**a2a+p2p** 为层级混合（节点内 a2a、节点间 p2p），适合 CP 度超过 head 数的多节点场景。NVIDIA 文档的经验是能调 TP 使 `CP ≤ num_kv_heads / TP` 时优先纯 a2a；KV head 很少或要最大 CP 灵活性时用 p2p；
- 与 Megatron-SP（TP 内对 norm/dropout 的序列并行，Reduction/AllGather 零额外开销）区分，SP 先上，激活仍不够再上 CP；
- MLA 注意力在 Megatron-Core 中有 explicit（训练用，per-head K/V 参与 a2a）与 absorbed（推理 decode 用，只 gather latent）两种实现，训练路径被 assert 禁用 absorbed（激活显存回归 20~34%），低通信训练方案见 LAGA 论文；
- TorchTitan 侧：torch 原生 CP（序列切分 + RingAttention 或 AllGatherKV）对普通注意力可用；DSA 这类需要同时处理 KV 与 indexer key 的模型要自定义 CP（`CustomContextParallelContext`，在 forward 里加 KV AllGather）。

# 6. 动态CP

为了适应不同的上下文，动态调整CP的并行度也逐渐成为研究的热点。两个代表性方向中，NVIDIA Helix 在**执行时**于不同并行布局间动态切换同一组 GPU；北大与阿里的 Vertumnus 则在**调度时**按负载动态调整 CP 度与 worker 组成。

## 6.1 Helix

NVIDIA 在 Helix 论文（*Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding*，arXiv:2507.07120）中提出面向交互式百万级 token 解码的混合执行策略。

**要解决的问题**：自回归 decode 有两大瓶颈——FFN 权重读取与长 KV Cache 读取。TP 能缓解 FFN 权重读取，但对注意力扩展很差：TP 宽度超过 KV head 数后 KV Cache 被低效复制，锁死并行度与 batch 上限；同时长 KV 的 DRAM 读取随 batch 线性增长。

**核心做法**：把 Attention 与 FFN 的并行映射在**时间维度**上解耦（对比传统 AFD 把两者空分到不同 GPU）。

1. **Attention 阶段用 KV Parallelism（KVP）**：KV Cache 沿 T 维分片到 CP rank，每 rank 对本地 KV 分片计算 partial attention；
2. **轻量通信步**合并/校正 partial 结果（LSE 式合并），保证注意力结果精确；
3. **FFN 阶段同一组 GPU 重配置**为 TP（dense 模型）或 TP×EP（MoE）去算 FFN/MoE，充分利用 GPU；
4. 拓扑为 $N = KVP \times TP_A$（注意力 TP 度 $TP_A$ ≤ KV head 数），QKV 投影冗余计算使每个 KV 分片可做完全本地的 FlashAttention；
5. **Helix HOP-B**：用 batchwise overlap 把上述通信开销藏起来，保持低 TTL 的同时提高 GPU 效率；
6. KV 块按 rank 轮转（交错）分配，保证 KVP 组内所有 GPU 显存均匀增长——这也是 vLLM DCP 交错分片的出处（由 Moonshot 的 Chao Hong 提出）。

**动态 CP 怎么并行起来（机制拆解）**：Helix 的"动态"不是运行时改 CP 度，而是**同一组 GPU 在每个 decoder layer 内被重配置两次**，Attention 与 FFN 各用各的最优并行映射，时分复用同一块卡：

1. **布局基座**：$N$ 块 GPU 按 $N = KVP \times TP_A$ 组织；KV Cache 沿 T 维交错分片常驻各 rank（显存均匀增长），**整个生成过程不动**；
2. **Attention 段（KVP 布局）**：每 rank 对本地 KV 分片做完整 FlashAttention（QKV 投影冗余计算换本地性），得到 partial 输出 $(O_j, L_j)$；
3. **轻量合并步**：沿 KVP 组做一次 LSE 式合并/校正（只传 $O$ 与 LSE，量很小），注意力结果精确等价；
4. **FFN/MoE 段（TP/EP 布局）**：**同一批 GPU** 立刻换映射——dense 模型走 TP，MoE 走 TP×EP，权重 AllReduce / 路由 All-to-All；
5. 回到 2，进入下一个 decoder layer——每个 layer 都重复"KVP → 合并 → TP/EP"这一拍；
6. **HOP-B（Helix Overlapped P2P-Batch）**：第 3 步的合并通信与第 2/4 步的计算在 **batch 维**交错——请求 A 通信时，请求 B 正在做 Attention/FFN 计算，通信被完全藏进计算，既保低 TTL 又保 GPU 利用率。

与传统做法相比，AFD 把 Attention 与 FFN **空分**到不同 GPU（各干各的，但卡有一半时间闲），Helix 把两者**时分**到同一组 GPU；TP 单独用则 KV Cache 随 KV head 复制、batch 上限被锁死。Helix 等于"Attention 时是 CP、FFN 时是 TP/EP"，通信只有每层一次轻量 LSE 合并。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 400" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="400" fill="#ffffff"/>
<text x="340" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">Helix 动态 CP：同一组 GPU 时分复用 KVP 与 TP/EP 两种映射</text>
<text x="60" y="48" font-size="11" font-weight="bold" fill="#334155">时间轴（每层一拍）→</text>
<g font-size="10" text-anchor="middle">
<rect x="60" y="58" width="130" height="30" fill="#c7d2fe" stroke="#6366f1"/><text x="125" y="77" fill="#312e81">Attention（KVP）</text>
<rect x="194" y="58" width="70" height="30" fill="#fdba74" stroke="#ea580c"/><text x="229" y="77" fill="#7c2d12">LSE 合并</text>
<rect x="268" y="58" width="120" height="30" fill="#86efac" stroke="#16a34a"/><text x="328" y="77" fill="#14532d">FFN（TP）</text>
<rect x="392" y="58" width="130" height="30" fill="#c7d2fe" stroke="#6366f1"/><text x="457" y="77" fill="#312e81">Attention（KVP）</text>
<rect x="526" y="58" width="70" height="30" fill="#fdba74" stroke="#ea580c"/><text x="561" y="77" fill="#7c2d12">LSE 合并</text>
<rect x="600" y="58" width="60" height="30" fill="#86efac" stroke="#16a34a"/><text x="630" y="77" fill="#14532d">FFN</text>
</g>
<text x="340" y="108" text-anchor="middle" font-size="10" fill="#64748b">layer i 的一拍：KVP 算注意力 → 轻量合并 → 换 TP/EP 映射算 FFN → 进入 layer i+1（同一批 GPU，反复重配置）</text>
<text x="60" y="136" font-size="11" font-weight="bold" fill="#334155">HOP-B：batch 维交错，把合并通信藏进另一个 batch 的计算</text>
<g font-size="9" text-anchor="middle">
<rect x="60" y="146" width="70" height="24" fill="#bfdbfe"/><text x="95" y="162" fill="#1e3a8a">A: Attn</text>
<rect x="132" y="146" width="50" height="24" fill="#fdba74"/><text x="157" y="162" fill="#7c2d12">A: 合并</text>
<rect x="184" y="146" width="70" height="24" fill="#bbf7d0"/><text x="219" y="162" fill="#14532d">A: FFN</text>
<rect x="256" y="146" width="70" height="24" fill="#bfdbfe"/><text x="291" y="162" fill="#1e3a8a">A: Attn</text>
<rect x="328" y="146" width="50" height="24" fill="#fdba74"/><text x="353" y="162" fill="#7c2d12">A: 合并</text>
<rect x="100" y="176" width="70" height="24" fill="#fde68a"/><text x="135" y="192" fill="#92400e">B: Attn</text>
<rect x="172" y="176" width="50" height="24" fill="#fecaca"/><text x="197" y="192" fill="#7f1d1d">B: 合并</text>
<rect x="224" y="176" width="70" height="24" fill="#ddd6fe"/><text x="259" y="192" fill="#4c1d95">B: FFN</text>
<rect x="296" y="176" width="70" height="24" fill="#fde68a"/><text x="331" y="192" fill="#92400e">B: Attn</text>
<rect x="368" y="176" width="50" height="24" fill="#fecaca"/><text x="393" y="192" fill="#7f1d1d">B: 合并</text>
</g>
<text x="530" y="162" text-anchor="middle" font-size="9" fill="#334155">请求 A 通信（橙）时</text>
<text x="530" y="180" text-anchor="middle" font-size="9" fill="#334155">请求 B 正在计算 →</text>
<text x="530" y="198" text-anchor="middle" font-size="9" fill="#15803d">通信 100% 被藏起来</text>
<text x="340" y="226" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">两种映射下的矩阵/权重分片（4 GPU 例）</text>
<g font-size="9" text-anchor="middle">
<text x="175" y="246" fill="#312e81">KVP：KV Cache 沿 T 交错分片（Attention 段）</text>
<rect x="60" y="252" width="52" height="20" fill="#c7d2fe"/><rect x="116" y="252" width="52" height="20" fill="#a5b4fc"/><rect x="172" y="252" width="52" height="20" fill="#c7d2fe"/><rect x="228" y="252" width="52" height="20" fill="#a5b4fc"/>
<rect x="60" y="272" width="52" height="20" fill="#a5b4fc"/><rect x="116" y="272" width="52" height="20" fill="#c7d2fe"/><rect x="172" y="272" width="52" height="20" fill="#a5b4fc"/><rect x="228" y="272" width="52" height="20" fill="#c7d2fe"/>
<text x="175" y="306" fill="#64748b">KV 块按 rank 轮转交错 → 各卡显存均匀增长</text>
<text x="500" y="246" fill="#14532d">TP/EP：权重按 head / expert 切分（FFN 段）</text>
<rect x="400" y="252" width="52" height="40" fill="#86efac"/><rect x="456" y="252" width="52" height="40" fill="#bbf7d0"/><rect x="512" y="252" width="52" height="40" fill="#86efac"/><rect x="568" y="252" width="52" height="40" fill="#bbf7d0"/>
<text x="500" y="306" fill="#64748b">FFN/MoE：TP（dense）或 TP×EP（MoE），结果 AllReduce</text>
</g>
<g stroke="#7c3aed" stroke-width="1.6" fill="none" stroke-dasharray="5 3">
<rect x="292" y="248" width="26" height="48"/>
</g>
<text x="340" y="330" text-anchor="middle" font-size="9" fill="#6d28d9">⇄ 每拍在两种映射间切换（同一批 GPU 重配置，KV 分片不动）</text>
<rect x="60" y="344" width="560" height="42" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="340" y="362" text-anchor="middle" font-size="10" fill="#334155">传递的数据：partial (O, LSE) 的轻量合并（每层一次）+ FFN 权重 AllReduce / MoE 路由</text>
<text x="340" y="380" text-anchor="middle" font-size="10" fill="#64748b">通信方式：KVP 组内 LSE 合并 + TP AllReduce；HOP-B 让两者都被另一个 batch 的计算覆盖</text>
</svg>

▲ Helix 动态 CP 机制：每层一拍的 KVP→合并→TP/EP 时序、HOP-B 的 batch 交错、两种映射下的分片形态

**效果**（DeepSeek-R1、Blackwell NVL72、FP4、1M token 上下文的模拟）：固定 batch 下 TTL 最多降低 1.5 倍；同等延迟预算下 batch 最多放大 32 倍。目标场景是 PD 分离架构的 generation 服务器、输入 >64K、低 batch、延迟敏感（高 tokens/s/user）的负载。

**实现**：TensorRT-LLM 已支持（`context_parallel_size ≥ 2`、`cp_config.cp_type = "HELIX"`、`cp_config.tokens_per_block` 与 KV cache 配置一致），目前支持 Blackwell 上的 MLA 模型（DeepSeek-V3 / V3-Lite），有 MLA Helix 单测与分离式服务精度测试。

## 6.2 Adaptive Context Parallelism

北大与阿里联合发表 *Adaptive Context Parallelism for Production LLM Serving*（系统名 **Vertumnus**，arXiv:2609.04774，2026-09），把 CP 度变成生产调度的一等公民。

**背景——生产负载的三个特征**：① 请求长度高度异构（几百 token 的问答到几十万 token 的文档分析/Agent 请求）；② 负载随时间动态变化（到达率与长短请求占比数倍波动）；③ 前缀复用普遍（系统提示、多轮对话、Agent 工作流），前缀缓存必不可少。CP 的收益与输入长度强相关：长请求从高 CP 度明显获益，短请求却会因通信开销反而变慢——静态 CP 配置注定顾此失彼。

**现有方案的不足**：① 完全不把 CP 度当调度维度；② 静态 CP 配置（部署时定死，最多按长度路由，不考虑缓存命中与 worker 负载）；③ 虽能动态调 CP 度，但只针对活跃请求/批次，没有稳定的持久化 worker 集群，前缀缓存无法跨请求复用。

**Vertumnus 的三层设计**：

1. **请求级路由**：用 placement cost 在不同 CP 度的 worker 之间路由请求，代价函数 = 预测排队延迟 + cache-aware 的 prefill 时间 + GPU-time 成本；
2. **集群级重组**：随负载变化以**秒级** split/merge 调整 worker 组成——短请求潮来了把大 CP worker 拆成多个小 worker，长请求潮来了把小 worker 合并成大 worker，无需重启进程或重新加载模型；
3. **全局前缀缓存管理**：跨 worker（无论 CP 度相同与否）协调缓存的放置与复制，worker 重组、请求改派时保持缓存局部性——缺的缓存块可以从别的 worker（甚至是不同 CP 度的 worker）按需复制过来。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 380" style="width:100%;height:auto">
<rect x="0" y="0" width="680" height="380" fill="#ffffff"/>
<text x="340" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">Vertumnus：调度时动态调 CP 度（请求级路由 + 秒级 split/merge + 全局前缀缓存）</text>
<text x="150" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">① 请求级路由（placement cost）</text>
<g font-size="9" text-anchor="middle">
<rect x="30" y="60" width="86" height="24" fill="#bbf7d0" stroke="#16a34a"/><text x="73" y="76" fill="#14532d">短请求 512</text>
<rect x="30" y="90" width="86" height="24" fill="#bfdbfe" stroke="#2563eb"/><text x="73" y="106" fill="#1e3a8a">长请求 128K</text>
<rect x="30" y="120" width="86" height="24" fill="#bbf7d0" stroke="#16a34a"/><text x="73" y="136" fill="#14532d">短请求 256</text>
<rect x="30" y="150" width="86" height="24" fill="#fecaca" stroke="#dc2626"/><text x="73" y="166" fill="#7f1d1d">超长 1M</text>
</g>
<rect x="140" y="72" width="120" height="88" rx="7" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="200" y="94" text-anchor="middle" font-size="9" fill="#334155">代价函数 =</text>
<text x="200" y="110" text-anchor="middle" font-size="9" fill="#334155">排队延迟 +</text>
<text x="200" y="126" text-anchor="middle" font-size="9" fill="#334155">cache-aware</text>
<text x="200" y="142" text-anchor="middle" font-size="9" fill="#334155">prefill 时间 + GPU·s</text>
<g stroke="#64748b" stroke-width="1.5" fill="none">
<path d="M 118 72 L 138 100"/><path d="M 118 102 L 138 112"/><path d="M 118 132 L 138 124"/><path d="M 118 162 L 138 138"/>
</g>
<text x="330" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">② 集群级 split / merge（秒级，不重启、不重载）</text>
<g font-size="9" text-anchor="middle">
<rect x="280" y="62" width="150" height="30" fill="#e0e7ff" stroke="#6366f1"/><text x="355" y="81" fill="#1e3a8a">worker CP=4（大）</text>
<rect x="280" y="122" width="70" height="26" fill="#ddd6fe" stroke="#7c3aed"/><text x="315" y="139" fill="#4c1d95">CP=2</text>
<rect x="360" y="122" width="70" height="26" fill="#ddd6fe" stroke="#7c3aed"/><text x="395" y="139" fill="#4c1d95">CP=2</text>
<rect x="280" y="162" width="150" height="30" fill="#e0e7ff" stroke="#6366f1"/><text x="355" y="181" fill="#1e3a8a">worker CP=1（小）</text>
<rect x="280" y="212" width="150" height="30" fill="#e0e7ff" stroke="#6366f1"/><text x="355" y="231" fill="#1e3a8a">worker CP=1（小）</text>
<rect x="280" y="262" width="150" height="30" fill="#e0e7ff" stroke="#6366f1"/><text x="355" y="281" fill="#1e3a8a">worker CP=2（合并后）</text>
</g>
<g stroke="#7c3aed" stroke-width="1.8" fill="none">
<path d="M 320 94 L 315 118"/><path d="M 390 94 L 395 118"/>
<path d="M 300 244 L 320 258"/><path d="M 410 244 L 390 258"/>
</g>
<g font-size="9" fill="#6d28d9" text-anchor="middle">
<text x="452" y="88">短请求潮 → split</text>
<text x="452" y="104">大 worker 拆小</text>
<text x="452" y="236">长请求潮 → merge</text>
<text x="452" y="252">小 worker 合大</text>
</g>
<g stroke="#64748b" stroke-width="1.5" fill="none">
<path d="M 262 110 L 278 80"/><path d="M 262 126 L 278 136"/><path d="M 262 158 L 278 176"/><path d="M 262 138 L 278 228"/>
</g>
<text x="590" y="50" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">③ 全局前缀缓存</text>
<g>
<rect x="500" y="62" width="180" height="26" fill="#fef3c7" stroke="#d97706"/><text x="590" y="79" text-anchor="middle" font-size="9" fill="#92400e">system prompt 前缀块</text>
<rect x="500" y="94" width="180" height="26" fill="#fef3c7" stroke="#d97706"/><text x="590" y="111" text-anchor="middle" font-size="9" fill="#92400e">多轮对话前缀块</text>
<rect x="500" y="126" width="180" height="26" fill="#fef3c7" stroke="#d97706"/><text x="590" y="143" text-anchor="middle" font-size="9" fill="#92400e">Agent 工作流前缀块</text>
<text x="590" y="172" text-anchor="middle" font-size="9" fill="#334155">跨 worker / 跨 CP 度统一管理</text>
<text x="590" y="188" text-anchor="middle" font-size="9" fill="#334155">缺块按需复制，重组时保局部性</text>
<g stroke="#d97706" stroke-width="1.5" fill="none">
<path d="M 498 100 L 434 240"/><path d="M 498 132 L 434 272"/>
</g>
</g>
<rect x="60" y="310" width="560" height="54" rx="7" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="340" y="330" text-anchor="middle" font-size="10" fill="#334155">传递的数据：按 placement cost 选择的请求流 + 前缀缓存块的跨 worker 复制</text>
<text x="340" y="350" text-anchor="middle" font-size="10" fill="#64748b">"通信"方式：请求改派（秒级）与 worker split/merge（秒级），缓存块按需搬移而非全量重算</text>
</svg>

▲ Vertumnus 三层机制：请求级路由（左）、集群级 split/merge（中）、全局前缀缓存（右）

**效果**：64 GPU 集群、公开 + 生产负载，最高负载下平均 TTFT 最多降低 **28.1%**，token 加权 SLO 达成率最多提升 **13.3 个百分点**（对比最强基线）。

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
