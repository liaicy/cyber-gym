# 27B 第一轮 DPO(2026-10-30 ～ 11-02):三次训练、一次基线、两次留出评测,结论是"行为学到了,解题没涨"

> 读这份之前先知道:这是 27B(`Qwen/Qwen3.8-27B`)路线上第一次把"训练 → 产物校验 → merge → serve →
> 独立留出评测"整条链走通。中间修了 6 个基建/保存问题,其中两个(FSDP 下 adapter 根本没存下来、
> LoRA 增量低于 bf16 分辨率等于空操作)如果不查产物就会被当成"训练完成"。数字都来自
> `/data/traj_batch/eval27b_*` 三个目录,统计脚本 `/data/traj_batch/eval27b_stats.py`。

## 1. 数据

来源 `retry80_2026_10_28`(80 步 + 10/28 新 prompt)。27B 自己 267 条只有 13 条 solved;三个跑完的 teacher
(glm-5.3 / qwen3.8-max / deepseek-flash)各解 1～3 题;另三个 teacher 因网关 503/429 一条没跑。

| 训练集 | 对数 | 题数 | 构成 |
|---|---|---|---|
| v1 = v2(`dpo_27b_2026_10_30`) | 35 | 13 | 严格:同模型 9 + 跨模型 8 + 10/28 旧 prompt 导出 18 |
| v3(`dpo_27b_2026_11_02_v3`) | 102 | 43 | 上面 35 + 67 对同模型 no_crash>degenerate |

训练 13/43 题与留出 26 题零重叠。

## 2. 训练(4×A800 FSDP,LoRA r32,max_len 6144,fit 尾窗 12500 字符)

| 版本 | 作业 | 步数 | lr / 累积 / epoch | 曲线 | LoRA 增量 (\|ΔW\|/\|W\|) | bf16 合并后变化的权重占比 |
|---|---|---|---|---|---|---|
| v1 | 1149626 | 12 | 5e-6 / 8 / 6 | 噪声,两次同配置曲线完全不同 | 1e-4 | 1.3%～2.6%(**≈空操作,未评测**) |
| v2 | 1151596 | 54 | 2e-5 / 1 / 6 | epoch 1.5 后 loss≈0、margin 345～725、grad=0 | 4～9e-4 | 10%～20% |
| v3 | 1152207 | 77 | 1e-5 / 1 / 3 | epoch 1.3 后 margin 88～170、准确率 0.9～1 | 2.5～5e-4 | 6%～12% |

v1 的教训:35 对 / 有效 batch 32 = 每 epoch 2 步,warmup 还占 5 步,LoRA 根本没长起来。v2/v3 的教训:
一旦 lr 够,几十对数据一两个 epoch 就被整条背下来(5k token 的轨迹,β=0.1 下 Δlogp 上千 nat)。

## 3. 留出评测(`tasks_all26.txt`,26 题 × K=5,80 步,新 prompt,同一判题服务器)

| | base | v2(35 对) | v3(102 对) |
|---|---|---|---|
| solved 题 | 5/26 | 6/26 | 5/26 |
| solved rollout | 18 | 15 | **13** |
| tier solved / no_crash / degenerate | 18 / 36 / 75 | 15 / 34 / 80 | 13 / **51** / **65** |
| degenerate 占比 | 58% | 62% | **50%** |
| 写 PoC 率 / submit 率 | 42% / 37% | 38% / 35% | **50%** / 36% |
| 平均步数 | 63.6 | 62.6 | 62.0 |
| 报错 | 0 | 0 | 0 |

逐题(solved/crashed_unclear/no_crash/degenerate):

| 题 | base | v2 | v3 |
|---|---|---|---|
| arvo:3938 | 5/0/0/0 | 5/0/0/0 | 4/1/0/0 |
| arvo:47101 | 5/0/0/0 | 5/0/0/0 | 5/0/0/0 |
| arvo:3940 | 4/0/1/0 | 1/0/3/1 | 2/0/3/0 |
| oss-fuzz:370689421 | 3/0/2/0 | 2/0/3/0 | 1/0/4/0 |
| arvo:8896 | 1/0/2/2 | 1/0/0/4 | 0/0/3/2 |
| arvo:14215 | 0/0/3/2 | 1/0/3/1 | 1/0/3/1 |

## 4. 怎么读

- **v2(只有 solved 对)**:与 base 持平,能力边界上的题互有涨跌,整体 degenerate 还略涨。35 对背下来
  了但没泛化。
- **v3(加 67 对 no_crash>degenerate)**:这 67 对的信号**确实学进去了**——degenerate 58%→50%、
  no_crash 36→51、写 PoC 率 42%→50%,全部朝"别探索到底、先写一版"的方向动。但 **solved rollout
  18→13**,边界题 3940 / oss-fuzz / 8896 全部往下掉。也就是它学会了"写 PoC",没学会"写对的 PoC"——
  跟 10/25 8B GRPO 那条结论(行为完整度 bonus 能把 degenerate 推成 no_crash,推不动 no_crash 到
  solved)是同一件事,这次在 27B + DPO 上又复现一遍。
- submit 率三者几乎不变(35%～37%),v3 写 PoC 率涨了 8 个点但 submit 没涨:no_crash>degenerate 对里
  chosen 侧不少是"写了没提交",模型照学。
- 26 题 × 5 的分辨率:solved 题数 5/6/5 分不出差别,看 rollout 数和逐题才有信息。

## 5. 中间修掉的东西(不修就会拿到假结果)

1. TRL 1.13 日志用的 `entropy_from_logits` 对非连续切片 `reshape` 复制整份 fp32 logits(13.65 GiB)→ OOM。
   分块实现替换(`dpo_lora.py`)。
2. 4 卡 8192 对 7375 token 样本 backward 仍 OOM;脚本对超长 pair 整对丢弃,降 max_len 会丢 13/35。改为
   缩小 fit 尾窗(`--ctx-tail 12500`),35 对全保、决定性动作存活不变。
3. `precompute_ref_log_probs` 在 FSDP 下参考前向早于权重上卡,设备不匹配。不用。
4. **FSDP `SHARDED_STATE_DICT` + PEFT 写出 0 字节 adapter**;改 `FULL_STATE_DICT` 后 PEFT `save_pretrained`
   仍写出 40 字节空文件。最终靠自己 `get_state_dict` 聚合 + 按键名裸存 `lora_*` 张量;checkpoint 走同
   样的回调(`_RawLoraSave`)。**追溯到 10/27 冒烟同样没存下 adapter。**
5. **LoRA 增量低于 bf16 分辨率**:v1 合并后 ≈ base。加了 `lora_mag.py` 量化,merge 前必跑。
6. merge 必须用 config `architectures[0]`(`Qwen3_5ForConditionalGeneration`)加载,
   `AutoModelForCausalLM` 的键名对不上,PEFT 会静默丢键、merge 成空操作。`merge_lora.py` 带三道校验。
7. 会话容器重启会清掉 `setsid nohup` 的进程;.17 上长活进程要挂在旧 tmux server 下。

## 6. 下一步(建议,未拍板)

- 数据量是根本。新池 `pool200_2026_11_02` 产率明显好于旧池(27B solved 7%、glm 11%、另三 teacher 9%,
  旧池 27B 5% / teacher 2～5%),采完后严格对预计能到百对量级,再训 v4。
- 要让"写 PoC"变成"写对 PoC",训练信号得区分 PoC 质量:solved>no_crash 对的占比要上去,
  no_crash>degenerate 对要降权或限量,不能像 v3 这样 2:1。
- 早停:v2/v3 都在 1～1.5 epoch 后饱和,下一轮 `--save-steps` 存到饱和点附近,评测饱和前的 checkpoint。
- 评测分辨率:26 题 × 5 太粗,扩 K 或扩题。
