# KataGo RTX 3080 Ti / SM86 CUDA 优化审查与最终结果

> 整理时间：2026-08-12（Asia/Shanghai）
> 状态：本轮 E022 已完成并停止；不再启动新的优化实验。
> 目标机器：NVIDIA GeForce RTX 3080 Ti，compute capability 8.6（SM86），80 SM，12 GiB。
> 代码基础：`doomoooo/KataGomo_fork` 的 `final-migration@5dfd8cb16bc0393518bdadcd1fe55ee1252da1a8`。
> 云端源码：`/root/katagomo-final-upstream-5dfd8cb1-sm86-fullscan-experiment`。
> 云端证据根目录：`/root/katagomo-final-c192821a-results`。
> E022 证据目录：`/root/katagomo-final-c192821a-results/e022-sm86-fullscan`。

本文参考 [`final-migration`](https://github.com/doomoooo/KataGomo_fork/tree/final-migration/final-migration) 的“性能总览、运行图、优化项、正确性、实验历史、适用边界、证据索引”风格整理，但只报告这台 RTX 3080 Ti 上实际核对或实际运行过的结果。上游已有实现、此次任务早期 E001–E021 的改动、本次 E022 新增改动严格分开，避免把别人的既有成果重复计算为本轮产出。

## 1. 结论摘要

### 1.1 最终严格性能

最终一轮将官方 CUDA、官方 TensorRT、本次 E022 前的优化版和 E022 最终版放在同一个热状态窗口，以两个互补回文块运行。四者均使用：

- 同一 RTX 3080 Ti；
- 同一 70M 模型、同一配置、同一默认 benchmark 局面；
- `19×19`、FP16/NHWC；
- 固定最大 batch `B=8`；
- NN server 数 `S=4`；
- 搜索线程 `T=48`；
- 每个局面 `v=3200` visits，每进程 `n=10` 个局面；
- 每种实现 4 个独立进程；
- 主指标只使用日志最终打印的有效 `nnEvals/s`，不使用会随搜索并发浮动的 `visits/s`。

| 实现 | 4 次 `nnEvals/s` | 均值 | 中位数 | 样本标准差 | 平均实际 batch | 相对官方 CUDA |
|---|---|---:|---:|---:|---:|---:|
| 官方 CUDA | `735.70 / 730.74 / 726.58 / 726.28` | **729.825** | 728.660 | 4.414 | 7.8750 | 基线 |
| 官方 TensorRT | `1029.16 / 1029.83 / 1026.50 / 1023.73` | **1027.305** | 1027.830 | 2.784 | 7.8475 | `+40.760%` |
| E022 前 incumbent | `1464.73 / 1465.66 / 1465.84 / 1460.23` | **1464.115** | 1465.195 | 2.635 | 7.5925 | `+100.612%` |
| E022 最终版 | `1483.14 / 1484.47 / 1480.55 / 1479.54` | **1481.925** | 1481.845 | 2.275 | 7.6150 | **`+103.052%`** |

最终版的直接比较为：

- 相对官方 CUDA：**`+103.052%`**，即 `2.0305×`；
- 相对官方 TensorRT：**`+44.254%`**，即 `1.4425×`；
- 相对 E022 前 incumbent：**`+1.216%`**；
- E022 前 incumbent 相对官方 CUDA已是 `+100.612%`，所以 E022 是在约翻倍的基础上继续获得约 1.2%，不是把早期百分比简单串联。

E022 的 incumbent/final 相邻配对分别提升：

`+1.2569% / +1.2834% / +1.0035% / +1.3224%`

四对方向全部为正；log-ratio 配对几何提升为 **`+1.2165%`**，单侧 95% Student-t 下界仍为 **`+1.0464%`**。最终版平均实际 batch 只比 incumbent 大约 `0.30%`，而 `nnEvals/s` 高 `1.216%`；收益主要来自每批推理变快，不是通过扩大 batch 制造的表观提升。

严格四方验证报告为 `strict-b8-s4-t48-v2/e022-strict-fourway-validation-v2.json`，SHA-256：

`e079c7d81e4a8f41efee4b3defbd3feec435e868c3fd3018ed7d980dbd6d1ac5`

### 1.2 本次 E022 新接受的性能改动

完整 123 候选扫描和依赖感知 refinement 最终只接受一个新坐标：

`cudaLinear2CutlassTacticSm89=m128-n128-k32-w64-n32-s3-sw1`

这里的配置名沿用了上游 `Sm89` 实现命名空间，但产物实际编译到 SM86，最终二进制中包含 SM86 cubin；设备分类、搜索历史和性能证书均明确属于 `sm86/rtx3080ti`，没有冒充 RTX 4090 或继承其性能历史。

在固定 B8/S4 的 8 段 `ABBA-BAAB`、每段 500 正式迭代中：

| 项目 | 结果 |
|---|---:|
| incumbent 算术均值 | `1478.019709 nnEval/s` |
| challenger 算术均值 | `1500.017250 nnEval/s` |
| 四对提升 | `+0.8929% / +1.4683% / +1.3552% / +2.2481%` |
| 配对几何提升 | **`+1.48996%`** |
| 单侧 95% 下界 | **`+0.83076%`** |
| 最小效应门槛 | `+0.5%` |
| 方向一致 | 是，4/4 为正 |
| 决策 | 接受 |

严格 T48 搜索中的实际落地收益为 `+1.216%`，小于固定形状确认的 `+1.490%`，属于预期差异：真实搜索还包含请求生成、批次形成和 CPU/search 协同。本文采用严格 T48 的 `+1.216%` 作为最终 E022 性能结论。

### 1.3 Elo 边界

本文没有把吞吐提升直接换算成 Elo。`nnEval/s` 提升允许同一墙钟内做更多神经网络评估，但最终棋力还取决于搜索参数、并行重复、虚拟损失、树复用和时间控制。要声明 Elo，仍需在固定模型、规则、时间/计算预算和搜索参数下做足量自博弈或对局 SPRT。当前可确认的是 GPU 推理与真实搜索评估吞吐提高，不是已经测得的 Elo 数值。

## 2. 代码来源与成果归属

### 2.1 上游 `final-migration` 已有的能力

本任务没有把以下上游设计重新包装成本轮原创：

- plan-driven 的 CUDA backend、精确 19×19 FP16/NHWC shape contract；
- 每个 NN server 独立 CUDA stream，并把 cuBLAS/cuDNN handle 绑定到所属 stream；
- wide QKV/FFN、fused QKV+RoPE、dual-FFN、fused residual、RMSNorm、post-conv、FlashAttention 等实现目录和候选框架；
- offline whole-graph tactic scanner、activation marker、plan apply mapping、8192 行 FP32 replay 思路；
- async/event pipeline、batch-aware dispatch、CUDA Graph 等可选运行能力；
- source/runtime packaging 以及 GPU plan registry 的基本框架。

本任务以 `final-migration@5dfd8cb1` 为代码起点，先审查其上游记录，之后只补 RTX 3080 Ti/SM86 缺口、测量/统计/证据缺口，以及在此硬件上实际有证据的策略。这样避免重复别人已完成的 SM89/SM120 优化，也不把 RTX 4090/5080 的数据直接外推到 SM86。

### 2.2 本任务早期 E001–E018 完成的 GPU 优化与审查

在 E022 完整扫描之前，本任务已经在 RTX 3080 Ti 上验证并保留了以下关键路径：

1. **多流提交和 handle 绑定。** per-thread default stream 与 cuBLAS/cuDNN handle 绑定必须原子启用；只启用前者会退化。该层解释了迁移版相对官方 CUDA 的大部分基础收益。
2. **SM86 portable dual-FFN。** 将 `linear1 + linearGate + SwiGLU` 合入一个 shared-A CUTLASS 路径；严格 S4/T48 中相对当时基础组合约 `+13.74%`。
3. **fused QKV+RoPE。** 合并 Q/K/V 投影并在 epilogue 计算 Q/K RoPE；在 dual-FFN 上再提高约 `+5.10%`，完整组合相对基础约 `+19.55%`。
4. **post-conv BN+SiLU。** 将 transformer block 的第二个 projection、residual 和下一层 BN+SiLU 组合；当时严格搜索约 `+1.27%`，并通过动态 batch FP32 对照。
5. **真实 SM86 FlashAttention T5。** 修正可选目标硬编码 arch 89 的问题，把固定 D32/361 Flash 内核实际编译为 SM86；T5 `d32-m64-n96-w4-pack0-both16` 相对同二进制 disabled 严格提高 `+16.36%`，将旧 headline 从约 1268 推到约 1479 `nnEval/s`。
6. **机制验证。** Nsight 显示 T5 将单次 attention kernel 从约 `80.470 µs` 降到 `40.284 µs`（`-49.94%`），attention 累计热点占比从 `24.2%` 降到 `13.3%`，且并未少运行 attention 层。
7. **动态 shape 正确性修复。** CUTLASS `update()` 不能改变 GEMM problem shape；token/problem shape 改变时必须重新 initialize。修复前约 `1182 nnEval/s` 的假高值和接近 100% 的 batched winrate 错误已明确废弃。
8. **精确 19×19 测试入口。** `testgpuerror` 原来会绕过 exact-shape 后端；增加显式 exact-NN-len 测试路径后，669 局面动态 batch 才真正覆盖优化实现。

这些改动和负结果的逐实验记录位于云端 `katago-gpu-experiment-ledger.md` 与 `katago-gpu-audit-progress.md`。本轮最终四方基准重新测了官方 CUDA/TRT 和最终版，因此最终 headline 不依赖跨轮百分比拼接。

### 2.3 E019–E022 新增的 infra 与本轮策略

E019–E022 的重点是让“看起来更快”升级为“可重算、可追溯、可拒绝假阳性”的证据链：

- **E019：schema-v2 统一墙钟指标。** 新增 `timedWallNNEvals`、`timedWallSeconds` 和 `aggregateWallNNEvalsPerSec`，严格满足 `work=B×S×iterations` 与 `rate×seconds=work`。旧 `combinedNNEvalsPerSec` 只保留诊断兼容，不再负责生产排序。
- **E020：配对统计门禁。** 候选必须通过 8 段 `ABBA-BAAB`、4 对方向全正、几何收益至少 0.5%、单侧 95% 下界大于 0；long gate 收紧为 4×1000 和 2% 最大相对极差。
- **E021：真实 SM86 身份。** 增加 `sm86/rtx3080ti` 分类；允许复用已实际编译运行的 SM89 实现命名空间，但不继承 RTX 4090 性能历史。缩减 microspace 永远不能生成 production-ready plan。
- **E022：完整生产空间。** 构建 19 个策略族、123 个候选的完整 B8/S4 空间；实际生成、编译并链接 10 个 TileLang AOT 候选；对 AOT 候选先做 8192 行真实 linked-path 正确性，再做完整 discovery、依赖 refinement、long gate、最终联合 FP32 replay 和严格四方 T48 基准。

## 3. 最终优化版的数据流

最终 B8/S4 图的关键路径是：

```text
每个 NN server 独立 stream/handles
        |
        v
精确 19x19、FP16/NHWC 输入
        |
        v
wide QKV/FFN 权重布局
        |
        +--> fused QKV GEMM + Q/K RoPE
        |         |
        |         v
        |    SM86 FlashAttention T5
        |         |
        |         v
        |    attention out projection + residual
        |
        +--> dual-FFN CUTLASS sw4 + SwiGLU
                  |
                  v
             E022 linear2 CUTLASS
        m128-n128-k32 / warp64x32 / stage3
                  |
                  v
          post-conv projection + BN + SiLU
                  |
                  v
             RMSNorm warps4
                  |
                  v
          policy/value/score/ownership heads
```

最终 runtime activation marker 共 11 项：

- `cudaDualFfnCutlassTacticSm89=m128-n64-k32-w64-n32-s3-sw4-exp`；
- `cudaFlashAttentionTacticSm89=d32-m64-n96-w4-pack0-both16`；
- `cudaLinear2CutlassTacticSm89=m128-n128-k32-w64-n32-s3-sw1`；
- `cudaPostConvCutlassTacticSm89=m128-n128-k32-w64-n64-s3-sw1`；
- fused QK RoPE、fused residual、post-conv BN+SiLU、QKV+RoPE GEMM、RMSNorm、wide FFN、wide QKV。

本轮严格 incumbent 与 final 的配置和激活集合只相差 `cudaLinear2CutlassTacticSm89` 一个键；验证器逐份核对四个 incumbent 和四个 final 日志，没有把其他策略变化混入 E022 的 `+1.216%`。

## 4. E022 完整空间扫描

### 4.1 固定身份

| 项目 | SHA-256 / 值 |
|---|---|
| 完整空间 | `d84f51fa5e99783f2ce7541206d55c13253b3be934c111ba30ae8e6a13798afc` |
| E022 二进制 | `87c7767a7d3aee5b5bc39f47025542b3d66b74c423b581336fe2a78196e32747` |
| 压缩模型 | `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6` |
| 解压执行模型 | `38d03bb990f774c0b1676b0a00feee4b05b61ed1a900cfa93a8f123af52e48ae` |
| 配置 | `71b33bc7031d5552ab0bda586ec1b9def81c386395c523777faca4465a2c971c` |
| AOT artifact bundle | `9b2c0659a6ae830fc914b51f388eedec48e2144dfab78a9ca4fc3dc5bef25310` |
| linked AOT replay certificate | `9e31be1b29bb9235879f85456404498de9d6a5c995234116b4a57d6f993283f4` |
| 冻结 workflow | `6879bb5e362dd6df180efeb9d5083d0fe703d3aa8491826d1a946944ff798718` |
| 锁定语料 | `0b2f2838df51ff98847f5bf595f9670350e993c5e178a92855c21e80e75762c5` |
| FP32 参考 KRNN | `c4c702192c667ad4d072ee00e1d31aac9cbf1bee610fbc7141a60bce0a6ba817` |

### 4.2 19 个策略族和最终选择

完整空间不是只围绕预想中的赢家做窄扫，而是覆盖 19 个有序、显式依赖的策略族。最终赢家如下：

| 策略族 | 候选数 | 最终选择 | E022 是否改变 |
|---|---:|---|---|
| wide projection | 3 | `wide-projection-both` | 否，保留既有宽 QKV/FFN 结构 |
| QKV + RoPE | 14 | `qkv-rope-gemm-epilogue` | 否 |
| dual-FFN | 11 | `m128-n64-k32-w64-n32-s3-sw4-exp` | 否 |
| fused residual | 3 | `on` | 否 |
| linear2 | 13 | `m128-n128-k32-w64-n32-s3-sw1` | **是，唯一新增** |
| attention out-projection | 7 | keep incumbent | 否 |
| post-conv + BN/SiLU | 12 | `m128-n128-k32-w64-n64-s3-sw1-bn-silu` | 否 |
| pointwise | 7 | keep incumbent | 否 |
| RMSNorm | 4 | warps4 | 否 |
| FlashAttention | 7 | T5 `M64/N96/W4/both16` | 否 |
| pre-conv | 8 | keep incumbent | 否 |
| persisting L2 | 11 | keep incumbent | 否 |
| model-weight sharing | 3 | keep incumbent | 否 |
| initial convolution | 3 | keep incumbent | 否 |
| wide head projection | 4 | keep incumbent | 否 |
| initial global path | 3 | keep incumbent | 否 |
| policy P1 | 4 | keep incumbent | 否 |
| head BN | 3 | keep incumbent | 否 |
| value terminal | 3 | keep incumbent | 否 |

候选数合计为 123。每个 discovery 候选使用 B8/S4、100 正式迭代、50 warmup，schema-v2 物理工作量恰为 `8×4×100=3200` 行。完整 discovery 从 10:35:42 运行到 10:51:25，123/123 成功；JSON SHA-256：

`18663f55a00924b089f9e0dce789efe94b41e1ca196c96579c6e8762f72e2fb2`

独立验证器重新核对 19 个 family、123 个 candidate、每行 B/S/iterations、schema、原始 rate/time/work 恒等式以及唯一 final-joint 行，输出 `valid=true`。

### 4.3 refinement 如何排除短测假阳性

refinement 对首轮 top-K 在已经改进的整张图上重测，最多三轮；本次第一轮有 98 个短测行，第二轮对依赖相关 top-3 再测 10 行，合计 108 个 refinement 测量，并形成 11 次正式决策摘要。

一些候选在单次短测中看似更快，但长配对不成立：

| 策略族/候选 | 配对几何变化 | 95% 下界 | 结果 |
|---|---:|---:|---|
| linear2 新 CUTLASS | `+1.490%` | `+0.831%` | **接受** |
| outproj-off（第一轮） | `-0.077%` | `-0.353%` | 拒绝 |
| RMSNorm warps8（第一轮） | `+0.194%` | `-0.276%` | 拒绝 |
| preconv CUTLASS（第一轮） | `+0.295%` | `-1.305%` | 拒绝 |
| value-terminal-off（第一轮） | `-0.129%` | `-0.335%` | 拒绝 |
| wide-projection-off（第二轮） | `+0.458%` | `-0.272%` | 拒绝 |
| dual-FFN sw2（第二轮） | `+0.148%` | `-1.013%` | 拒绝 |
| outproj CUTLASS（第二轮） | `+0.376%` | `-0.022%` | 拒绝 |
| RMSNorm warps8（第二轮） | `-0.102%` | `-0.297%` | 拒绝 |
| preconv CUTLASS（第二轮） | `+0.339%` | `-0.324%` | 拒绝 |
| value-terminal-off（第二轮） | `-0.122%` | `-0.372%` | 拒绝 |

这张表是本轮 infra 价值最直观的证据：如果只选单次最高值，outproj、RMSNorm、preconv、wide projection 等都会被错误推进；正式门槛只接受了方向一致且下界为正的 linear2。

canonical refinement JSON SHA-256：

`916ecc8909b161920888451ad063a7ca73f0095cc52b4a14cfb6796e5ca40e4f`

独立 validator 从原始 stdout 重取 benchmark JSON，重新哈希 128 份正式确认原始文件，复算 `B×S×iterations`、`rate×seconds`、四对 log-ratio、几何收益和 t 下界；结果 `valid=true`，SHA-256：

`278516f4433aac81048d1ebeabb42984d5f1ece40706b00455be1db597d90da7`

## 5. 长稳态和最终正确性

### 5.1 固定形状长稳态 gate

最终组合以 B8/S4、50 warmup、每次 1000 正式迭代运行 4 次：

`1512.307688 / 1508.175579 / 1499.925423 / 1495.357122 nnEval/s`

- 中位数：**`1504.050501 nnEval/s`**；
- 相对极差：`1.12699%`；
- 生产上限：`2%`；
- 每次物理工作量：`8×4×1000=32000` 行；
- 外部 SM 占用 PID：无；
- 四次均首次成功，无超时或重试。

这只是固定形状 `benchmarknn` 长稳态值，不与官方 CUDA/TRT 的 T48 搜索值混为一张性能表。long-gate JSON SHA-256：

`0b45e7f92be26ff117c6794b309329d62f7f2ef40bad22c022ce1c9c4b2001a0`

### 5.2 8192 行整图 FP32 联合回放

最终正确性不是把 10 个 AOT 单候选证书拼起来，而是重新运行最终 19-family 联合 overrides：

- candidate：B8/S4，8192 行，固定 batch tail padding；
- reference：官方 full-FP32 路径，B13/S1，8192 行；
- candidate 的 11 个 activation marker 与四次 long gate 完全相同；
- 输入和 target sections byte-exact；
- policy、value、score、ownership、weighted P0 loss 和逐请求最坏误差均通过门槛。

| 指标 | 最终结果 | 门槛 |
|---|---:|---:|
| policy top-1 vs FP32 | `0.9974365` | `>= 0.995` |
| policy probability RMSE | `0.00013367` | `<= 0.001` |
| weighted P0-loss delta | `0.00007033` | `<= 0.001` |
| value outcome RMSE | `0.00259713` | `<= 0.01` |
| score mean RMSE | `0.00224982` | `<= 0.01` |
| ownership sigmoid RMSE | `0.00029969` | `<= 0.001` |
| per-request policy max abs / max RMSE | `0.015276 / 0.000885` | `<= 0.025 / 0.002` |
| per-request value max abs / max RMSE | `0.038590 / 0.031505` | `<= 0.06 / 0.05` |
| per-request score max abs / max RMSE | `0.546282 / 0.236626` | `<= 0.60 / 0.30` |
| per-request ownership max abs / max RMSE | `0.018626 / 0.001817` | `<= 0.025 / 0.006` |

比较 JSON SHA-256：

`7d117bff54079c553f15dcfd73bfb409867f4fb193c97e9f1b18c681d9755cee`

最终 candidate KRNN SHA-256：

`ec9470c82f3d5398d226dfcd54a6a63815b8f5b23ed80ca5816419a159f9614e`

certified gate SHA-256：

`97eefe5b0110337a09f52fc0360a1b1925c0457e47723e212a461b9153380cf3`

发布前复核发现旧加载器只接受 SM89/SM120，并会把 `nnBatchAwareDispatch` 强制改成 `true`；这与本报告实际认证的 SM86、Graph 关闭、batch-aware 关闭配置不一致。发布版因此增加了显式 runtime execution contract，并让 C++ 加载器校验及应用认证值。最终 production plan ID 为 `sm86-rtx3080ti-abc187f1c89a74d4`，`production_ready=true`、`ready_for_scan_bypass=true`、missing groups 为 0；内部 canonical plan SHA-256：

`abc187f1c89a74d4ef2af34b7b72a4cf1d685eb6a45db8dc306e076f6d5371e7`

发布文件 SHA-256：

`933f50fb95fb0857a5f76191046e7b58997c98e235496d92d5a5e7a758ec6ff6`

新二进制在同一 RTX 3080 Ti 上实际加载该 plan，确认 `B8/S4`、FP16/NHWC、`cudaUseGraphInference=false`、`nnBatchAwareDispatch=false` 和全部 11 个 activation marker。50 次短跑的物理墙钟吞吐为 `1533.891099 nnEval/s`；它只证明发布加载路径可用，不替换第 1 节的正式同轮四方基准。

## 6. 官方 CUDA 和 TensorRT 基线如何保证正确

### 6.1 同轮四方设计

最终顺序为：

```text
official-a, trt-a, incumbent-a, final-a,
final-b, incumbent-b, trt-b, official-b,
final-c, incumbent-c, trt-c, official-c,
official-d, trt-d, incumbent-d, final-d
```

两个八段块互补，使四种实现都占据内侧、外侧以及早晚位置。全部运行期间 GPU 为 P0；除一次起点采到 1920 MHz 外，SM clock 基本为 1935 MHz，稳态温度约 56–58°C。driver 同时记录每段开始/结束的温度、SM/memory clock、功耗和 P-state。

### 6.2 TensorRT 隔离 plan

官方 TRT 使用全新目录：

`/root/katago-official-trt-home-e022-strict-b8-v2`

正式测量前由官方 TRT 二进制构建精确 `ex19x19_b8_fp16` plan；四个 TRT 进程中的每个 NN server thread 都记录使用同一路径。plan SHA-256：

`fed5a91515ec531a9bcc060fc8b5f4b4a8150e68b88f75dab648031e7b77e42b`

这样避免了之前发现的“官方 TRT 错误复用实验分支 plan”污染。TRT 每个进程仍会通过内部 ONNX emitter 重建 network 描述，但四个 context 都加载上述隔离 B8 engine；plan 构建/模型加载时间不进入最终 `nnEvals/s`。

### 6.3 身份和口径核对

| 对象 | SHA-256 |
|---|---|
| 官方 CUDA 二进制 | `1753662c3ea684024dad56b2a848ff0bd32e603bbad3682e97d4a03071dd4123` |
| 官方 TensorRT 二进制 | `8515ba33467b44f9a0e540f9b9a44fa93ae427619bd3501a71d20db4dba5f74a` |
| candidate 二进制 | `87c7767a7d3aee5b5bc39f47025542b3d66b74c423b581336fe2a78196e32747` |
| 模型 | `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6` |
| 配置 | `71b33bc7031d5552ab0bda586ec1b9def81c386395c523777faca4465a2c971c` |
| certified gate | `97eefe5b0110337a09f52fc0360a1b1925c0457e47723e212a461b9153380cf3` |

独立 strict validator 要求每份日志恰有一个 T48/n10 终值、S4、fixed B8 warning、相同模型/配置加载身份、平均 batch 不超过 8、完整 telemetry；final 和 incumbent 的 config/activation 差只能是 linear2 一个键。所有检查通过。

## 7. 代码层面做了哪些改动

相对 `final-migration@5dfd8cb1`，当前实验树有 18 个 tracked 文件改变，约 `+2275 / -271` 行；大量新增位于 workflow 和测试，不是 2000 行 CUDA kernel。最终 `git diff --check` 通过，workflow/autotune 65 项单元测试全部通过；最终测试日志 SHA-256：

`59ff116bf3e840c965839f2b7f4451f5e75a4c5d4c29c43915d3802ee4722148`

工作树还保留 6 个未跟踪的 E020/E021 审计辅助脚本：`e020-gpu-smoke.py`、`e020-statgate-validation.py`、`e021-inspect-space.py`、`e021-space-summary.py`、`python/e021_make_microspace.py`、`python/e021_validate_pipeline.py`。它们没有链接进最终 KataGo 二进制，也不包含在“18 个 tracked 文件”的增删行统计中；保留它们是为了复查早期统计门禁和 SM86 pipeline，而不是把工作树描述成干净提交。

### 7.1 构建与 SM86 AOT

- `cpp/CMakeLists.txt`
  - SM8x Flash/tactic target 要求唯一实际 CUDA architecture；
  - 不再硬编码 89；RTX 3080 Ti 构建到 `sm_86`；
  - 将 portable dual-FFN、linear2 AOT artifacts 纳入最终链接。
- `cpp/neuralnet/cudabackend_sm89_flash.cu`
  - launcher 的 architecture identity 改为真实 SM86；
  - 保留已验证 T5 和相邻候选的明确 tactic 名称/activation。
- `cpp/neuralnet/cudabackend_sm89_dual_gemm.cu`、`cudabackend_sm89_linear2_gemm.cu`
  - token/problem shape 改变时重新 initialize；
  - 修复把 CUTLASS `update()` 当作 shape 更新的错误；
  - 允许精确 B8/S4 portable artifact 与现有 CUTLASS fallback 共存。
- `python/portable_generate_tilelang_aot.py`、`portable_prepare_tilelang_fat_scan.py`
  - 从 SM89-only 扩展为真实 SM8x；
  - 根据 space 的 compute capability 生成 `sm_86` 编译参数；
  - 绑定 space、candidate projection 和 artifact identity。

### 7.2 测量口径

- `cpp/command/benchmarknn.cpp`
  - 输出 schema-v2：`timedWallNNEvals`、`timedWallSeconds`、`aggregateWallNNEvalsPerSec`；
  - 文本和 JSON 都明确报告 aggregate timed-wall physical throughput。
- `cpp/neuralnet/nneval.cpp/.h`、`nninterface.h`、`cudabackend.cpp`
  - 在 warmup 完成后、正式迭代开始前记录每线程起点；
  - 在所有正式迭代完成并同步后记录终点；
  - 跨 server 取最早 start 到最晚 end，分子严格为 `B×S×iterations`；
  - 保留旧字段兼容，但生产选择强制 schema-v2。

### 7.3 正确性入口和后端安全

- `cpp/command/gputest.cpp`
  - 增加 exact-NN-len 测试开关，使固定 19×19 优化后端真正进入 GPU error test；
  - 防止测试实际上落入官方 fallback 而产生假通过。
- `cpp/neuralnet/cudabackend.cpp` / `cudabackend_sm89.cpp`
  - 显式 tactic 不可用时 fail-fast；
  - activation marker 只在真实 launch 后记录；
  - 保存实际 batch/stream/architecture 约束和 graph 边界。

### 7.4 自动调优和证据链

- `final-migration/autotune/detect_gpu.py`
  - 增加 `sm86`、`rtx3080ti`（以及通用 SM86）分类；
  - 实现命名空间可复用不等于性能历史可继承。
- `final-migration/autotune/autotune.py`
  - SM86 进入合法 workflow；
  - 接线 schema-v2 指标、0.5% 最小效应和 production 安全状态。
- `python/cuda_tactic_workflow.py`
  - 真实 SM86 positive-history closure；
  - 完整空间与缩减空间的 production eligibility 分层；
  - 依赖/所有权 map，禁止 later family 隐式重写 earlier family；
  - artifact bundle、binary、space、model、config、replay certificate 的 SHA 绑定；
  - stale resume 身份校验；
  - `ABBA-BAAB` 8 段运行、log-ratio、0.5% 效应、一侧 95% t 下界；
  - origin confirmation 和 decision history；
  - 4×1000、2% long gate；
  - individual AOT correctness 只允许进入 scan，不能冒充最终组合正确性；
  - final joint comparison 通过后才能 certify 与生成 production plan；
  - 发布版从 final-joint evidence 提取并绑定运行契约，防止调度、Graph 或精度开关在加载时漂移。
- `cpp/neuralnet/cudatacticplan.cpp`
  - 接受并严格校验 SM86/RTX 3080 Ti plan；
  - 对 SM86 强制要求显式 runtime contract，并应用认证的 Graph/batch-aware/FP16/NHWC/warmup 值；
  - 保持旧 SM89/SM120 plan 的兼容加载路径。
- `python/tests/test_cuda_tactic_workflow.py`、`test_autotune_entrypoint.py`、`test_checked_in_tactic_plan.py`
  - 覆盖设备分类、物理工作量恒等式、篡改拒绝、配对统计、origin evidence、reduced-space 拒绝、artifact closure、最终 certify、runtime contract 和 production plan registry。

E022 性能冻结点的 18 个 tracked 源文件 SHA 列表保存在：

`e022-final-source-hashes.sha256`

该列表自身 SHA-256：

`c17047624ab7c94f8bf2893c5cefc9c5f1052c805106d6019329f579c9303429`

发布阶段只增加方案加载契约、registry 测试、文档和 plan 资产，不改变已认证的 CUDA kernel/tactic 图。相关 67 项发布测试全部通过；全目录 90 项导入中另有 2 项因默认环境未安装 `cutlass`/`pytest` 而未运行，与本次改动无关。

## 8. 已否决、暂停或限制使用的路线

### 8.1 运行调度与 CUDA Graph

- **单独 pinned async H2D/D2H：** batches/s 上升但实际 batch 下降，有效 `nnEvals/s` 约 `-2.1%`；不能单独合入。
- **event-pipeline Graph：** 严格搜索约 `-1.39%`。
- **普通路径按实际 batch 的 compute-only Graph：** T48 约 `-1.81%`；T56 `-2.16%`；T64 `-0.51%`。Graph 减少每批开销，却更快抽空请求队列，平均 batch 下降。
- **Graph + batch-aware padding：** 能凑满 batch，但 padding 使有效吞吐下降，组合约 `-1.80%`。
- **phase staggering：** 负收益。

结论：Graph 本身不是 Elo/吞吐保证；必须与不 padding 的有界聚合等待或更好的 queue scheduler 联合设计。

### 8.2 TensorRT 路线

- Q/K/V ONNX 宽 MatMul：约 `-0.29%`；
- FFN `linear1+gate` ONNX 宽 MatMul：约 `-0.20%`；
- builder optimization level 5：约 `-1.87%`；
- 放开全部 tactic sources：约 `-0.50%`；
- 缩减 RMSNorm FP32 约 `-0.57% / -0.60%`；
- TensorRT CUDA Graph 和 head FP16 在早期实验中有小幅正收益，但本任务目标后来转为更高的自定义 CUDA 上限，最终 CUDA 已比官方 TRT 高 `44.254%`。

TensorRT cache identity 仍是重点工程风险：早期官方对照曾错误命中实验 plan，该数据已废弃。最终轮通过隔离目录和 plan hash 修复测量纪律，但 upstream cache key 仍应进一步绑定网络图、逐层精度、builder/tactic、TRT/CUDA 版本与生成代码身份。

### 8.3 kernel/tactic 负结果

- 早期 linear2/outproj 小扫描曾显示 linear2 不足 `+0.1%`、outproj 无收益；E022 在完整 graph 和更全面 CUTLASS 几何下发现了新的 linear2 赢家，说明策略要在实际整图上下文重测。
- E022 的 4 个 TileLang linear2 AOT 与 6 个 dual-FFN AOT 均通过正确性，但性能不如 retained CUTLASS；正确不等于更快。
- E022 outproj-off 和另一个 CUTLASS outproj 都被配对门槛拒绝。
- RMSNorm warps8 两轮均未通过；保留 warps4。
- preconv CUTLASS、value-terminal-off、wide-projection-off、dual-FFN sw2 都没有稳定证据。
- model-weight sharing 首块约正、确认块中性，最终未复现；吞吐不启用，显存场景可保留。
- persisting L2 trunk/inner 在 SM86 单流明显退化，完整空间继续保留关闭。
- B10 9+1 split 仅 S1 正、S2 负，不适合当前 S4 默认路径。
- token-keyed CUTLASS operator cache 比简单 shape reinitialize 约 `-0.09%`，复杂度更高，否决。

### 8.4 Flash 微调停止条件

- T5 邻域 M64/N64 的对称均值仅 `+0.08%`，远低于 0.5% 门槛和约 1.55% 时间漂移；其他 tile 都更慢。
- 删除未消费 LSE 写出的 combined 值约 `+0.064%`，两时间块方向相反，schema-v2/配对回放明确拒绝；寄存器还从 117 增至 118，没有改变 occupancy 档位。
- 因此不继续盲扫 Flash tile/epilogue；最终保留 T5。

## 9. 审查中发现但尚未解决的改进空间

以下是后续最有价值的方向，但本轮按用户要求已经停止，没有继续实现或测试。

### 9.1 搜索侧：最可能直接转化为 Elo 的方向

GPU 推理已经接近把官方 CUDA 翻倍，下一阶段不能只盯单 kernel。建议先增加只读 instrumentation，量化：

- concurrent same-hash NN miss：多个搜索线程是否同时请求相同 position/hash；
- `EVALUATING` 状态的重试和空转；
- root 并发、virtual loss 导致的重复探索；
- node-output store 失败、被覆盖或等待超时；
- subtree reuse 在 ponder、连续落子和多分析请求中的实际命中率；
- NN queue age、batch fill、从请求产生到 GPU launch 的分布。

如果重复 NN 评估占比显著，可设计带 generation、timeout 和接管机制的 in-flight coalescing：第一个线程成为 owner，其余 waiter 复用结果；owner 失败或超时后允许安全接管。必须防止死锁、永久 `EVALUATING`、错误跨规则/komi/model 复用。性能先看固定 nnEval 数下的搜索节点质量与去重率，最终用固定模型和预算的自博弈 Elo/胜率验证。

### 9.2 NN 调度：批次形成和吞吐的联合优化

Graph 路线说明“单批更快”可能让平均 batch 下降，从而使有效 `nnEvals/s` 变差。后续可尝试：

- 不 padding 的短微秒 bounded wait；
- 基于 queue age、GPU idle 和近期 arrival rate 的自适应 batch deadline；
- 不同 NN lane 之间避免同时抢小批；
- 将 batch fill 和 GPU completion 反馈给 search producer；
- 对首局延迟、稳态吞吐和尾延迟分别设目标。

任何 scheduler 改动都必须保持 B/S/T 对照或明确声明拓扑改变，不能拿 batches/s 单独定案。

### 9.3 GPU 后端新热点

Flash T5 后的搜索期热点大致为 dual-FFN `21.5%`、主 cuBLAS GEMM `15.9%`、QKV+RoPE `15.1%`、另一组 GEMM `14.4%`、Flash `13.3%`、RMSNorm `7.9%`、post-conv `5.4%`。可优先：

- 对 dual-FFN 做 SM86 专属 occupancy、stage、warp layout 和 epilogue 调优，而不是搬 SM89/SM120 固定答案；
- 对 QKV+RoPE 检查内存布局、sincos 生成、寄存器压力与多流 SM 竞争；
- 分解剩余两个 cuBLAS GEMM，确认是否有稳定 exact-B8 shape 适合 CUTLASS/CuTe epilogue fusion；
- 用 NCU 分析 E022 linear2 新赢家为何在 SM86/S4 更好，并围绕相邻 warp/stage 做小而有统计门槛的局部搜索；
- 扩大到多个模型宽度、block 数和 batch，避免对当前 28b 模型过拟合。

### 9.4 infra 仍可改进的点

1. **confirmation 历史保留。** E022 日志有 11 次正式决策摘要，但 refinement JSON 对同一候选第二轮重测时会覆盖第一轮 rejection 对象，只留下 8 个不同完整边界。唯一 accepted linear2 证据完整，最终选择不受影响；但未来应把每轮 confirmation 追加到不可变 history，而不是覆盖 row 字段。
2. **多重比较。** 当前 19 family/123 candidate 先 discovery 再 top-K confirmation，已显著降低假阳性；如果继续扩大空间，应考虑 family-wise false discovery 或更严格的下界门槛，例如要求 95% 下界本身高于某个最小效应。
3. **plan/certificate canonical digest。** 继续减少 JSON 自证字段，使用 canonical serialization 和 Merkle/manifest 式证据闭包；plan loader 应验证而不是只保存 producer metadata。
4. **第三方依赖 provenance。** 上游新归档删除了部分 FlashAttention/CUTLASS commit 兼容检查。构建可允许本地 source bundle，但产物必须记录实际 source tree hash、patch hash、编译器与 flags。
5. **TRT cache lock/identity。** 增加进程间构建锁以及完整 network/build identity；plan 内部元数据二次验证。
6. **显式配置死开关。** `cudaUseBatchSharedRoPE`、`cudaUseFusedFFN` 等“解析但未实际消费”的键应删除、实现或启动时警告。
7. **官方 CUDA n20 析构内存破坏。** 早期官方 n20 在输出指标后出现 `malloc_consolidate(): invalid chunk size`；最终基线改用 4×n10 规避并保留现场。仍应以 ASan/UBSan 和析构路径二分定位。

## 10. 失败尝试与测量事故留痕

本任务没有覆盖失败结果或把失败路径当性能证据：

1. **错误 CUTLASS dynamic shape：** 约 1182 的假高值全部废弃。
2. **TRT 共享 cache 污染：** 第一轮官方 TRT 对照废弃，重建隔离 plan。
3. **旧 `combinedNNEvalsPerSec` 排序：** 多流相位下可与真实 aggregate 相差约 1%，不再用于生产选择。
4. **官方 CUDA n20 析构 crash：** 即使打印了 731.69，也因进程 SIGABRT 而废弃；正式轮改为四个 n10。
5. **E022 discovery attempt 1：** `wide-projection-off` 没有关闭其 QKV/FFN 消费者，触发 fail-closed tactic unavailable；修复显式 ownership/dependencies 后从新身份重跑。
6. **旧 artifact 绑定 v2 space：** candidate projection 改变后 binder 正确拒绝；重新生成全部 10 个 AOT artifacts、重建并重新做 linked replay，没有给旧证书改名冒充。
7. **replay topology mismatch：** 第一版回放驱动误用 S1，而 artifacts 绑定 S4；runtime 正确拒绝。修正 driver 后重新跑 10/10。
8. **独立 discovery validator v1：** 错把容器 schema 当 schema-v2，产生统一假拒绝；保留失败 JSON/log，v2 按实际序列化结构和 shell `pipefail` 重跑通过。
9. **严格四方脚本 v1：** incumbent 辅助配置漏写 `nnMaxBatchSize`，在任何性能段开始前触发构造/断言失败；v2 从 certified final 复制并只还原 linear2，断言 config diff 恰为一个键，随后完整运行 16 次。
10. **GPU clock lock：** 云平台 NVML 拒绝，即使 root 也无法锁频；实验采用交错回文、telemetry 和配对统计处理自然 boost 漂移，没有假装锁频成功。

## 11. 适用边界

当前生产证书只支持：

- RTX 3080 Ti / compute capability 8.6；
- 当前 GPU product/memory identity；
- 当前 70,442,025 参数 transformer 模型及其 SHA；
- 精确 19×19；
- FP16/NHWC；
- 物理 B8、S4；
- 当前 19-family apply map；
- 当前二进制、配置、artifact bundle 和正确性语料。

它不能自动外推到：

- RTX 3090、A10、A40 或其他同为 SM86 的卡；
- RTX 4090/SM89 或 RTX 5080/SM120；
- 另一个模型宽度、block 数、head 数或模型版本；
- 9×9/13×13、动态棋盘大小；
- B4–B32 的其他 batch 或不同 server 数；
- FP32、BF16、不同 layout；
- 多 GPU；
- 不同 search T、不同时间控制下的 Elo。

其他硬件/模型应重新执行设备分类、完整候选空间、linked artifact correctness、refinement、long gate、最终整图 FP32 replay 和严格真实搜索对照。

## 12. 关键证据索引

以下路径均相对于 `/root/katagomo-final-c192821a-results/e022-sm86-fullscan`：

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 完整空间 | `full-space-b8-s4-v2.json` | `d84f51fa...98afc` |
| artifact bundle | `e022-artifact-bundle-v2.json` | `9b2c0659...25310` |
| linked AOT correctness | `replay-correctness-v2/e022-linked-aot-replay-correctness.json` | `9e31be1b...283f4` |
| 完整 discovery | `discovery-b8-s4.json` | `18663f55...2fb2` |
| refinement | `refinement-b8-s4.json` | `916ecc89...e4f` |
| refinement 独立验证 | `e022-independent-refinement-validation-v2.json` | `278516f4...da7` |
| long gate | `long-gate-b8-s4.json` | `0b45e7f9...01a0` |
| final joint comparison | `final-joint-replay-b8-s4/final-joint-b8-vs-fp32.json` | `7d117bff...5cee` |
| final joint manifest | `final-joint-replay-b8-s4/e022-final-joint-replay-manifest.json` | `b0697cf3...abe5` |
| certified gate | `certified-long-gate-b8-s4.json` | `97eefe5b...0cf3` |
| production plan（发布版） | `production-plan-b8-s4-runtime-v2.json` | `933f50fb...6ff6` |
| plan 加载 GPU 烟测 | `e022-publish-runtime-contract-gpu-smoke.log` | 见发布留痕清单 |
| strict 四方 driver | `strict-b8-s4-t48-v2/e022-strict-fourway-driver.log` | 见 validation identities |
| strict 四方独立验证 | `strict-b8-s4-t48-v2/e022-strict-fourway-validation-v2.json` | `e079c7d8...1ac5` |
| 最终单元测试 | `e022-final-unit-tests.log` | `59ff116b...2148` |
| 最终源码 hash 清单 | `e022-final-source-hashes.sha256` | `c1704762...429` |
| E022 实时审计账本 | `e022-sm86-fullscan-report.md` | `c0a2cbcd...5a8d` |

历史总账与阶段报告：

- `/root/katagomo-final-c192821a-results/katago-gpu-experiment-ledger.md`；
- `/root/katagomo-final-c192821a-results/katago-gpu-audit-progress.md`；
- `/root/katagomo-final-c192821a-results/e020-statistical-selection-gate-report.md`；
- `/root/katagomo-final-c192821a-results/e021-sm86-pipeline-report.md`；
- `/root/katagomo-final-c192821a-results/e022-sm86-fullscan/e022-sm86-fullscan-report.md`。

## 13. 最终保留状态

本轮最终保留：

- 上游 `final-migration` 的多流/handle 和 plan-driven CUDA 基础；
- RTX 3080 Ti 上实际验证的 wide QKV/FFN、fused QKV+RoPE、dual-FFN sw4、fused residual、post-conv BN+SiLU、RMSNorm warps4；
- 真正编译到 SM86 的 Flash T5 both16；
- E022 新 linear2 CUTLASS `m128-n128-k32-w64-n32-s3-sw1`；
- schema-v2 physical metric；
- SM86/RTX3080Ti 独立身份；
- artifact/certificate/resume closure；
- ABBA-BAAB 配对统计和 4×1000 long gate；
- 8192 行最终整图 full-FP32 正确性证书；
- 同轮官方 CUDA/TRT/incumbent/final 严格基线。

本轮明确不保留为性能默认项：CUDA Graph、batch-aware padding、pinned async 单独方案、weight sharing、persisting L2、RMSNorm warps8、outproj/preconv 新 tactic、Flash no-LSE 和相邻 tile、TRT builder 激进设置。

最终性能结论是：在本报告固定的 RTX 3080 Ti、模型、配置和 `B8/S4/T48` 合同下，最终 CUDA 版本达到 **`1481.925 nnEval/s`**，相对官方 CUDA **`+103.052%`**，相对官方 TensorRT **`+44.254%`**；E022 本轮新增 linear2 对 E022 前版本贡献 **`+1.216%`** 的严格搜索吞吐。当前没有经对局测得的 Elo 数字。
