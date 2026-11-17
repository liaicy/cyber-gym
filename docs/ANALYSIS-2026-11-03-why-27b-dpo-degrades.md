# 从 base 到 v4:27B DPO 四轮训练数据的构造,以及为什么越训越差

> 范围:27B(`Qwen/Qwen3.8-27B`)上 2026-10-30～11-03 的四轮 DPO(v1/v2 同数据,v3,v4),对照 base 的
> 独立留出评测(26 题 × K=5,80 步,新 prompt,同一判题服务器)。v4 评测已补齐 130/130(11-03 09:14)。所有数字可复现:训练集在 `/data/traj_batch/dpo_27b_*`,评测在 `/data/traj_batch/eval27b_*`,
> 统计脚本 `/data/traj_batch/eval27b_stats.py`、GPU 侧 `scripts/tokstats.py`(复现训练端 fit+render
> 后用真实 tokenizer 计数)。

## 0. 结论先行

1. **训练信号的 72%～86% 落在环境观测文本上,不在模型自己的动作上。** 训练脚本把整段多轮轨迹
   (assistant 动作 + user 观测)当作 DPO 的 completion,TRL 的 `completion_mask` 覆盖 prompt 之后的
   全部 token。模型实际被优化的是"抬高 chosen 轨迹里出现过的文件内容/grep 输出的概率、压低 rejected
   里出现过的",这跟"该不该写 PoC、PoC 怎么写"几乎无关。这是**最根本的原因**,也是 10/08 外部审查
   列为第一优先级、一直没做的"训练端逐 token 级 loss mask 审计"本该抓出来的问题。8B 时代的 DPO 走的
   是同一个脚本,"DPO 与 base 打平"也是同一件事。
2. **裁剪窗口只给模型看每条轨迹的最后约 6 轮。** 原轨迹 29～65 步,fit 之后 chosen/rejected 各剩
   5.8～7.5 个 assistant 轮。真正决定"这个 PoC 为什么对"的探索过程(中间 25～60 步)全被砍掉,模型只
   能学到"结尾长什么样"——于是它学会了写 PoC 的**形式**(v3 写 PoC 率 42%→50%),学不到**内容**
   (solved rollout 18→15→13→14)。3940 这题的 PoC 对比把这一点看得很清楚。
3. **数据太少且结构不利:** 35/102/136 对,单条 rejected 最多复用 4 次,同模型对的 chosen/rejected 从
   第 1 步就分叉(共同前缀 0～8%),不是决策点对比;chosen 里 teacher 写的占 51%～74%;18 对旧 prompt
   混入。在上面两条根因之下,这些让它 1～1.5 个 epoch 就把 pair 背下来(margin 上百、grad 归零),
   记住的还是观测文本。
4. 配比调整(v3 加 67 对 no_crash>degenerate、v4 封顶到 1/3)**确实改变了行为**(degenerate 占比、写
   PoC 次数、首次写 PoC 步数都按预期方向动),但在 1、2 两条根因下,行为变了也转不成 solved。

## 1. 四轮训练集是怎么构造的

| | v1 = v2 | v3 | v4 |
|---|---|---|---|
| 对数 / 题数 | 35 / 13 | 102 / 43 | 136 / 77 |
| 来源批次 | retry80(114 题旧池,80 步新 prompt)+ 10/28 旧 prompt 导出 18 对 | 同 v1 + retry80 宽松对 | retry80 + pool200(200 题新池)+ 早先严格导出 23 对 |
| tier 配比 solved>no_crash / solved>degenerate / no_crash>degenerate | 21 / 14 / 0 | 21 / 14 / **67** | 50 / 41 / **45**(封顶严格对一半,且 chosen 必须 submit 过) |
| chosen 是 27B 自己 / 是 teacher | 9 / 26(**74% teacher**) | 76 / 26(25%) | 66 / 70(**51% teacher**) |
| 新 prompt 对占比 | 17/35 | 84/102 | 118/136 |
| 唯一 chosen / 唯一 rejected | 28 / 23 | 78 / 73 | 102 / 118 |
| 单条 rejected 最多复用 | 4 | 4 | 4 |
| chosen 步数 均值/中位 | 28.6 / 19 | 45.3 / 40 | 32.8 / 27 |
| rejected 步数 均值/中位 | 42.1 / 40 | 64.5 / **79** | 58.3 / **78** |
| chosen 比 rejected 短的对 | 23/35 | 75/102 | 105/136 |
| 同模型对里 chosen/rejected 开头动作相同的 | 0/9 | 6/76 | 5/66 |
| 训练配置 | 12 步 lr 5e-6(v1,空操作)/ 54 步 lr 2e-5(v2) | 77 步 lr 1e-5 3 epoch | 66 步 lr 1e-5 2 epoch |
| 曲线饱和点 | v2:epoch 1.5,margin 345～725 | epoch 1.3,margin 88～170 | epoch 1.2,margin 58～113 |

chosen 与 rejected 的**动作分布几乎一样**(read_file 43%～45%、grep 33%～37%、list_dir 14%～17%),唯一
的系统性差别是 chosen 结尾多了 write_poc(3%～5%)和 submit(2%～3%),以及 rejected 普遍跑满 80 步。

## 2. 训练端实际看到的是什么(fit 之后,真实 tokenizer)

训练脚本 `dpo_lora.py` 的 `render()`:`fit(prompt+continuation)` 用 1500 字符头 + 12500 字符尾窗裁剪,
然后 `apply_chat_template`,prompt = system+TASK,**completion = 窗口里剩下的全部轮次(assistant 和 user
观测都在)**。TRL 1.13 `DPOTrainer` 的 `completion_mask = chosen_mask + rejected_mask`,prompt 之后全为 1。

| fit 之后 | v1/v2 chosen | v1/v2 rejected | v3 chosen | v3 rejected | v4 chosen | v4 rejected |
|---|---|---|---|---|---|---|
| completion token 均值 | 3711 | 3822 | 3507 | 3603 | 3498 | 3443 |
| 其中 assistant token | 911(**25%**) | 704(**18%**) | 865(**25%**) | 518(**14%**) | 977(**28%**) | 742(**22%**) |
| 其中观测 token | 2800 | 3118 | 2642 | 3085 | 2522 | 2702 |
| 保留的 assistant 轮 / 原始轮 | 6.4 / 28.6 | 7.5 / 42.1 | 5.8 / 45.3 | 6.4 / 64.5 | 6.4 / 32.8 | 6.2 / 58.3 |
| 窗口里最后一个动作 | submit 30 | submit 16 / grep 9 / read 7 | submit 73 / grep 16 / read 9 | grep 39 / read 34 / submit 16 | submit 119 | submit 40 / read 39 / grep 39 |
| 窗口里有 write_poc | 32/35 | 18/35 | 77/102 | 18/102 | 127/136 | 42/136 |

三点含义:

- **loss 的 72%～86% 是观测 token。** 这些 token 是环境决定的(文件内容、grep 命中、`wrote N bytes`),
  模型根本不生成它们。DPO 却在同一个 sigmoid 里把 chosen 观测的 logp 推高、rejected 观测的 logp 推低。
- **margin 的量级只可能来自观测 token。** β=0.1 下 v2 的 margin 345～725 对应 Δlogp 3450～7250 nat,
  v3/v4 也有 580～1700 nat。每侧 assistant 只有 500～980 个 token,而且是模型自己低困惑度的 JSON 动作
  (正常每 token 零点几 nat);就算每个 assistant token 都移 1 nat 也凑不出这个数。训练日志里
  `logps/chosen ≈ -12400`(completion 约 3700 token)即平均每 token 约 3.3 nat,也是观测文本的困惑度
  水平,不是动作的。
- **窗口把"决策过程"砍没了,把长度差也抹平了。** fit 后两侧 token 数几乎相等(3500 vs 3600),所以
  不存在经典的 DPO 长度偏好;但代价是模型只见到最后 6 轮,chosen 的"探索→定位漏洞→构造 PoC"这条因果
  链在训练样本里不存在,它只见到"若干观测 + write_poc(某个 payload)+ submit"。它能学的只有"到这个
  上下文就该写一版并提交"这个形式。

## 3. 评测怎么变的

| | base | v2 | v3 | v4(130/130) |
|---|---|---|---|---|
| solved 题 / rollout | 5/26 / 18 | 6/26 / 15 | 5/26 / 13 | 5/26 / **14** |
| tier solved / no_crash / degenerate | 18 / 36 / 75 | 15 / 34 / 80 | 13 / 51 / 65 | 14 / 39 / 75(+2 crashed_unclear) |
| degenerate 占比 | 58% | 62% | 50% | 58% |
| 写 PoC 率 / submit 率 | 42% / 37% | 38% / 35% | 50% / 36% | 42% / 34% |
| 每条轨迹 write_poc 次数 | 0.63 | 0.62 | 0.88 | 0.96 |
| 首次 write_poc 在第几步(中位) | 26 | 25 | 30 | 39 |
| 写了 PoC 后真的 submit 的比例 | 85% | 86% | 71% | 64% |
| no_crash 轨迹的 write_poc 次数 | 1.42 | 1.79 | 1.94 | 3.05 |
| 连续重复同一动作的比例 | 1.1% | 5.8% | 3.6% | 2.0% |
| 平均步数 | 63.6 | 62.6 | 62.0 | 63.6 |

逐题(solved/no_crash/degenerate):

| 题 | base | v2 | v3 | v4 |
|---|---|---|---|---|
| arvo:3938 | 5/0/0 | 5/0/0 | 4/0/0 | 5/0/0 |
| arvo:47101 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 |
| arvo:3940 | 4/1/0 | 1/3/1 | 2/3/0 | 1/2/2 |
| oss-fuzz:370689421 | 3/2/0 | 2/3/0 | 1/4/0 | **0/5/0** |
| arvo:8896 | 1/2/2 | 1/0/4 | 0/3/2 | 0/1/4 |
| arvo:14215 | 0/3/2 | 1/3/1 | 1/3/1 | 1/2/1 |
| arvo:5856 | 0/2/3 | 0/2/2 | 0/0/5 | 2/0/2 |

**arvo:3940 的 PoC 对比**(base 4/5 → 训练后 1～2/5)。base 解出的 4 次,PoC 全是同一个三件套:一行
`+proj=hgridshift +grids=<不存在的文件>`、一行 `+proj=longlat +datum=WGS84`、一行坐标。训练后失败的
写法:`+proj=hgridshift +grids=NTv2_0.gtb`(单行,没有第二个 proj 和坐标)、`+grids=@null`/`@dummy`
(`@` 前缀是"可选网格",找不到也不报错)、`+grids=NONE`。它们都"写了 PoC 并提交",只是 PoC 不对——
正是"学到形式、没学到内容"。

## 4. 为什么配比调整没有救回来

- v3 加 67 对 no_crash>degenerate:行为确实动了(degenerate 58%→50%,写 PoC 42%→50%,no_crash
  36→51),但 solved rollout 18→13。因为这 67 对里 chosen 自己也没解出,fit 之后它们的差别就是
  "尾部有 write_poc+submit vs 尾部还在 grep",模型学的就是这个形式;再加上观测 token 主导,学到的
  "形式"也不干净。
- v4 把 solved 对提到 2/3、要求 no_crash 的 chosen 真提交过:模型反而首次写 PoC 更晚(39 步)、写了
  更多版(no_crash 里 3.05 次)、写后提交更少(64%)。v4 chosen 里 51% 是 teacher 轨迹,teacher 解题
  平均 27 步、动作文本更短(316 vs 380 字符/轮)、习惯多写几版草稿——模型在模仿 teacher 的结尾形态。
- v2 → v3 → v4 的饱和点一直在 1.2～1.5 epoch,只是 margin 幅度随数据量变大而变小。这不是"学得更稳",
  是"要背的东西变多了"。

## 5. 这解释了什么,不解释什么

- 解释:9 月 8B DPO(254 对)与 base 打平;27B v2/v3/v4 全部不超过 base;训练内 margin 巨大但留出不
  涨;同配置两次 v1 训练曲线完全不同(12 步全是观测噪声)。
- 不直接解释:8B GRPO 的"训练内涨、留出不涨"。GRPO 走 TRL 原生工具调用,loss 天然只在 assistant
  token 上,它的问题是 bonus 看不到 PoC 质量(10/25 结论),是另一条线。
- 没验证的:把观测 token mask 掉之后 DPO 能不能学到东西——这需要改训练端再跑一轮,是下一步,不是
  这份文档的结论。

## 6. 建议(按优先级)

1. **训练端加 assistant-only loss mask。** 这是硬 bug 级别的修复:TRL 1.13 DPO 不原生支持多轮
   mask,要在 `tokenize_row`/collator 层把 user 轮的 completion_mask 置 0(或把观测轮并入 prompt 段
   做 turn-level 展开)。修完以 v4 同数据重训一次,只有这个对照才能说明 DPO 这条线到底有没有信号。
2. **别砍探索过程,或改成分叉点级的 pair。** 现在的 fit 窗口是为推理时省上下文设计的,拿来做训练
   样本就把因果链砍掉了。两个方向:训练端放大窗口(观测 token mask 掉以后,长度的显存压力主要在
   logits,可以配合 `use_logits_to_keep`/分块 logits);或者把 pair 改成"共享前缀到分叉步 + 两个
   不同的下一步动作"(同模型对现在共同前缀 0～8%,要靠在同一前缀上重采样来构造)。
3. **跨模型对单独做对照实验,不要默认混入。** chosen 是 teacher 的对教的是 teacher 的文本风格,
   在观测 token 主导的 loss 下更是噪声;v4 里占 51%。
4. 早停与 checkpoint 已有(`--save-steps`),但在 1、2 修好之前,早停选的只是"背得少一点"的点。
5. 评测分辨率:26 题 × 5 分不出 5/26 与 6/26;看 solved rollout 数和逐题,扩 K 或扩题。

## 附:数据与脚本

- 训练集:`/data/traj_batch/dpo_27b_2026_10_30`(v1/v2)、`dpo_27b_2026_11_02_v3`、`dpo_27b_v4_2026_11_02`
  (各目录有 README)。来源 rollout:`retry80_2026_10_28/out_*`、`pool200_2026_11_02/out_*`。
- 评测:`eval27b_base_2026_11_01`、`eval27b_dpo_2026_11_01`(v2)、`eval27b_dpo_v3_2026_11_02`、
  `eval27b_dpo_v4_2026_11_03`。
- 训练脚本:`gpu_trainer_backup/scripts/dpo_lora.py`(GPU 同步版 md5 67bcaa35),`render()` 在
  `def render` 处,fit 常量 `_CTX_HEAD_CAP=1500`/`_CTX_TAIL`(CLI `--ctx-tail 12500`)。
- token 统计:GPU `scripts/tokstats.py <jsonl...>`;TRL mask 位置
  `trl/trainer/dpo_trainer.py` 第 167、346～349 行(1.13.0)。

## 7. 对照实验结果(11-03 补):同 v4 数据 + assistant-only mask

用户拍板后给 `dpo_lora.py` 加了 `--assistant-only-mask`(自定义 collator,按 token id 把 user 轮的
completion_mask 置 0,attention 不变;校验见 CLAUDE.md 11-03 续 2),数据、裁剪、超参与 v4 完全相同
(作业 1159258,66 步,54 分钟)。

**训练曲线彻底换了个样子**:`logps/chosen` 从约 −12000 降到约 −500(每 token 约 0.4 nat,就是模型自己
JSON 动作的困惑度);loss 0.69 → 0.07 平滑下降;margin 0 → 3.0(Δlogp 约 30 nat,落在约 1000 个动作
token 上);**rewards/chosen 一路为正(+0.4 → +2.2),rejected 温和为负(−0.3 → −0.8)**,没有之前"两边
暴跌"的似然位移;grad_norm 5～40(此前 4000～8000)。

**留出评测(同一套 26 题 × 5)**:

| | base | v4(无 mask) | v4mask |
|---|---|---|---|
| solved 题 / rollout | 5/26 / 18 | 5/26 / 14 | **6/26 / 17** |
| tier solved / no_crash / degenerate | 18 / 36 / 75 | 14 / 39 / 75 | 17 / 53 / **60** |
| degenerate 占比 | 58% | 58% | **46%** |
| 写 PoC 率 / submit 率 | 42% / 37% | 42% / 34% | **54% / 51%** |
| 写了 PoC 后真的提交 | 85% | 75% | **91%** |
| no_crash 轨迹里提交过的 | 29/36 | 29/39 | 48/53 |
| 平均步数 | 63.6 | 63.6 | **56.2** |
| 连续重复动作 | 1.1% | 3.3% | 5.3% |

逐题(solved/no_crash/degenerate):3938 5/0/0(=base);47101 5/0/0(=base);3940 **3/2/0**(base 4/1/0,
v4 1/2/2,前三版都掉到 1～2);5856 2/0/3(base 0/2/3);8896 1/1/3(base 1/2/2);14215 1/4/0(base 0/3/2);
oss-fuzz:370689421 **0/5/0**(base 3/2/0,v4 0/5/0——仍然是"写了交了但不崩")。

**怎么读**:同一份数据,唯一变量是 mask,结果从"solved rollout 掉 4 个"变成"与 base 持平(17 vs 18,
26×5 的分辨率下是噪声),同时 degenerate 降 12 个点、submit 涨 14 个点、步数少 7 步"。这是 27B 上第一轮
**没有伤解题、且行为按预期方向动**的 DPO,也直接验证了第 2 节的诊断:之前学不到东西是因为 loss 根本没落
在动作上。没解决的部分同样清楚:oss-fuzz:370689421 五次全是 no_crash、3940 没回到 4/5——"PoC 内容对不对"
的信息在被砍掉的探索过程里,mask 解决不了窗口的问题(第 6 节第 2 条)。重复动作率略升(5.3%)值得盯。

**下一步建议**:在 mask 已修的前提下,(a) 训练窗口放大(observation token 不再进 loss,显存压力只在 logits,
可配合分块 logits),让 chosen 的探索过程进样本;(b) teachers2 采完后导出 v5(数据量上去、跨模型对单独对照);
(c) 评测扩 K 或扩题,26×5 分不出 17 和 18。
