# 2026-09 关键负面结果:teacher 轨迹 SFT 使 8B 模式坍缩

> 一句话:用强模型解题轨迹对 Qwen3-8B 做 per-turn SFT,会让它退化成
> 反复复读同一个探索动作、无视观测,solve 率被打到 0,比不训练的 base 更差。
> 已复现两次(v1/v2),清洗数据和降超参都救不了。结论:**放弃 SFT 冷启动,
> 直接在 base 上做 DPO**(即 [PLAN-trajectory.md](PLAN-trajectory.md) 原定的 List-wise DPO)。

## 实验设置
- 目标模型:Qwen3-8B(LoRA SFT,r=32)。GPU 平台 A800,vLLM 起服务,rollout 评测走隧道回环境机。
- 评测:4 个任务 × 每模型 5 次 rollout,最多 40 步。任务含 2 个 held-out(arvo-10628/11007)
  + 2 个训练见过(10013/10999)。对照 = 同一台机上的 base Qwen3-8B(不训练)。
- SFT 数据:pilot 的 solved 轨迹拆成 per-turn(prompt=系统+任务+历史,target=该轮动作),
  只对模型输出算 loss。v1 用 raw_output 当 target;v2 用清洗后的规范 JSON。

## 结果(SFT 一致地比 base 差)
| | solved | degenerate(退化) | no_crash |
|---|---|---|---|
| base 8B | 1–4 / 20 | ~20% | 多数 |
| SFT v1(3ep, lr1e-4, raw target) | 0/20 | 75% | 少数 |
| SFT v2(2ep, lr5e-5, 干净 target) | 0/19 | 84% | 少数 |

## 根因:模式坍缩(已挖轨迹逐步确认)
SFT 后的模型把某个高频动作连发几十次、无视环境返回。实测 arvo-10628 上
`list_dir {"path":"src-vul"}` 连发 34 次,40 步只有 2 种不同输入,永远走不到 write_poc/submit。
base 在同任务上会 grep/list_dir/read_file 混着来、卡住会换动作。

机制:训练集 704 个轮次里 56% 是 list_dir/read_file 两个探索动作,真正决定性的
write_poc+submit 仅 20%。per-turn 下一动作 SFT 把小模型压到强预测高频动作上,
丢掉了「按观测调整、该收手时收手」的能力。**这是 SFT 目标本身的问题,不是数据脏。**

## 排除的两个假设(都验证过、都不是主因)
1. **坏 target**:v1 训练数据 376 轮里 73 个(19%)无法解析——强模型常在一个回合里粘两个
   JSON(如 `{...}{...`),`raw_output` 存了全文当学习目标,教会 8B 偶尔吐空/垃圾。
   已修:`traj/export_sft.py` 新增 `canonical()`,从环境解析好的 `thought/action/action_input`
   字段重建规范 JSON。v2 数据 0% 坏 target。**修完仍坍缩。**
2. **训练太猛**:v1(3ep/1e-4)→ v2(2ep/5e-5)。**仍坍缩。**

## 结论与下一步
- SFT 冷启动在这个任务上是负资产:格式从来不是瓶颈(base 吐格式就对),SFT 唯一效果是坍缩。
- 转向 **DPO on base**(不复制 teacher token,而是给轨迹排优劣,教「别循环、要推进」):
  - 第一轮 off-policy:复用现有 qwen3-8b(≈base 架构)rollout,19 个任务可形成同任务优劣对。
  - 粗粒度先行(整条轨迹级 DPO,见 PLAN 第八节),有信号再细化到步级 / 采 on-policy 数据。
- 价值:这个负面结果在扩数据、上 DPO 前就否掉了 SFT 冷启动,避免在坏 base 上做偏好优化白烧算力。

## 复现命令
- 干净 SFT 数据:`PYTHONPATH=. CYBERGYM_SRC=/root/cybergym-src python -m traj.export_sft
  --dest /data/traj_batch/sft_v2 --min-tier solved --per-task-cap 12 --val-frac 0`
- 一键训练+评测+释放卡:`/data/traj_batch/auto_eval.sh <sft_jobid> <tag> [adapter]`
- 坑:vLLM 本地关思考要 `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`(顶层
  enable_thinking 是硅基流动约定,vLLM 不认);节点 CUDA 12.4 vs torch cu130 → serve 加
  `--enforce-eager` + `VLLM_USE_FLASHINFER_SAMPLER=0` 关掉 JIT 编译路径。

---

## 补记:DPO v1 结果(2026-09-18)——方向成立

按上面的方向,在 base 8B 上跑了第一轮 trajectory-level DPO(56 个跨档偏好对,来自
qwen3-8b≈base 的 rollout,LoRA r32 / 2ep / lr5e-6 / beta0.1 / max_len8192)。同一次评测
(4 任务 ×5,base 与 DPO 同台对比):

| 模型 | solved | degenerate | no_crash |
|---|---|---|---|
| base 8B | 1/20 | 12 (60%) | 7 |
| SFT | 0/~20 | 84% | 少 |
| **DPO v1** | **2/20** | **5 (25%)** | 13 |

- **退化率:base 60% / SFT 84% → DPO 25%**,坍缩(反复复读一个动作)被治好,模型恢复正常
  探索+提交;held-out arvo-10628 上 DPO 有一条 15 步真 solved。
- solve 1→2(n 小,信号弱但方向对);行为从 degenerate 大量转为 no_crash(有进展未崩)。
- 结论:**DPO on base 成立,SFT 冷启动确定放弃。**

工程坑:DPO 显存 ≈ SFT 2–3 倍,80G 卡 max_len 12288 会 OOM → 用 8192 +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;TRL 1.13 `DPOConfig` 无 `max_prompt_length`。

下一步:solve 率要往上顶,需 (a) 迭代 on-policy DPO(用 DPO 模型自采样再造对)或
(b) 扩偏好对;(c) 评测扩到更多 held-out 任务提高统计力。
