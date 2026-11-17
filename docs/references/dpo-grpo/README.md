# DPO、GRPO 与稀疏成功信号：引用论文与阅读说明

整理日期：2026-10-20。收录本轮讨论中直接给出链接的 6 篇论文，按本清单首次出现顺序编号。

## 文件与引用格式

- `pdf/`：6 篇论文的本地 PDF；优先采用正式出版版。
- [references.bib](references.bib)：正式出版信息的 BibTeX，可按目标会议或期刊切换引用样式。
- [downloads.json](downloads.json)：下载来源、版本说明、文件大小和 SHA-256；保留此前引用的 arXiv 来源以供追溯。

**PDF 是文件格式；IEEE 是论文排版／参考文献样式。** 这 6 篇论文分别发表于 NeurIPS、ICLR、ICML、TMLR，各自采用其出版方模板，原文没有统一改排为 IEEE。下面的书目按 IEEE 编号引用样式整理；这不表示论文发表于 IEEE。

在本项目的研究文档目录中，目前找到的是 Markdown 设计、实验和审查记录，未发现 LaTeX、IEEE 类文件或既有 BibTeX 论文稿。因此，现阶段不能把项目论文称为“已采用 IEEE 模板”。正式投稿时应采用目标 venue 的要求。

ReST 的 OpenReview PDF 下载被拒绝，本地保存的是 arXiv v4（页面注明已被 TMLR 接收的 camera-ready 版本）。该文初稿为 2023 年，正式发表年为 2024 年；Rewarding Progress 初稿为 2024 年，正式会议年为 2025 年。文件名和参考文献使用正式发表年份。

## 论文索引

| 编号 | 论文／主题 | 正式发表 | 本地全文 | 本轮引用的预印本或作者站 |
|---|---|---|---|---|
| [1] | DAPO；GRPO 组内优势、动态采样 | NeurIPS 2025 | [PDF](pdf/01-dapo-2025.pdf) | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| [2] | Rewarding Progress；基于后续成功概率变化的过程监督 | ICLR 2025 | [PDF](pdf/02-rewarding-progress-2025.pdf) | [arXiv:2410.08146](https://arxiv.org/abs/2410.08146) |
| [3] | R³；成功示范前缀与逆向课程 | ICML 2024 | [PDF](pdf/03-reverse-curriculum-r3-2024.pdf) | [arXiv:2402.05808](https://arxiv.org/abs/2402.05808) |
| [4] | IRPO；迭代偏好优化与正例 NLL | NeurIPS 2024 | [PDF](pdf/04-iterative-reasoning-preference-optimization-2024.pdf) | [arXiv:2404.19733](https://arxiv.org/abs/2404.19733) |
| [5] | ReSTᵉᵐ；真实反馈筛选后的自训练 | TMLR 2024 | [PDF](pdf/05-rest-em-2024.pdf) | [arXiv:2312.06585](https://arxiv.org/abs/2312.06585) |
| [6] | Potential-based reward shaping；保持最优策略的条件 | ICML 1999 | [PDF](pdf/06-policy-invariance-reward-shaping-1999.pdf) | [作者站原文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) |

建议阅读顺序：[1] → [6] → [5] → [3] → [4] → [2]。先理解当前奖励和采样瓶颈，再比较自训练、课程和过程监督。

## 每篇与本次问题的关系

### [1] DAPO

**用于支持的论点：** 标准 GRPO 使用同一题多条输出的组内相对奖励计算优势。只有终局成败奖励时，全成功或全失败组缺少奖励差异；DAPO 用动态采样过滤这些组，维持具有有效奖励优势的训练样本。

**与项目的关系：** 应观察当前 checkpoint 的逐题成功分布与组内奖励方差。历史上其他模型的成功率，不能直接替代当前 8B 的成功概率。

**适用边界：** 论文主要在数学推理设置中评估。它不能证明 CyberGym 上 GRPO 必然优于 DPO／SFT，也不保证在环境调用昂贵时，额外采样过滤一定更省总成本。文中“零优势”指任务奖励对应的策略信号，独立 KL 等项仍可能产生梯度。

### [2] Rewarding Progress

**用于支持的论点：** 有价值的过程监督可以衡量一个步骤使后续成功概率发生了多少变化，而不只是判断这一步表面上是否合理。该工作提出用 prover policy 与过程优势验证器构造这类信号。

**与项目的关系：** 不同动作参数的数量、是否调用 `write_poc`、内容是否非空，都没有直接测量后续成功概率。不能因为这些指标可确定计算，就称其为可验证的任务进展。

**适用边界：** 该工作涉及模型化过程验证器，其训练与推断成本并不等于当前项目的纯规则 bonus。论文不证明覆盖率、crash 种类或任一行为里程碑天然与目标成功一致。

### [3] R³：逆向课程强化学习

**用于支持的论点：** 从成功示范的中间状态开始，让模型完成剩余部分，再逐步把起点向任务开头移动，可以缩短稀疏终局奖励下的探索距离。

**与项目的关系：** 可以研究利用已有成功轨迹或可靠纠正后的状态，训练短后续，再回到完整任务。

**适用边界：** 原文是推理任务实验。对有文件、工具和执行副作用的 agent，单独提供文字前缀不等于恢复环境状态；前缀课程的评估仍应包含从完整初始状态出发的任务。

### [4] Iterative Reasoning Preference Optimization（IRPO）

**用于支持的论点：** 迭代收集推理偏好对，并在偏好优化中加入正例的负对数似然（NLL）项，是值得比较的训练方案。

**与项目的关系：** 完整成功轨迹和同一状态下的可靠纠正动作，可以用于比较纯 DPO、正例 SFT、DPO＋NLL；偏好排序改善不自动等价于正确动作生成能力已充分学会。

**适用边界：** 其推理实验不能证明 8B 多轮工具任务一定需要 SFT，也不能证明已有 DPO 数据继续训练必然有效或必然导致模式坍缩。单步纠正的可靠性与完整任务成功应分别衡量。

### [5] ReSTᵉᵐ

**用于支持的论点：** 从模型采样，按外部反馈筛选样本，再用保留样本微调并迭代，是一条不依赖额外偏好打分器的自训练路线。

**与项目的关系：** 可以把 `verify()` 通过的成功轨迹作为正例 SFT 对照，尤其在已有可用成功轨迹时，新增数据收集成本较低。

**适用边界：** 原文在数学和代码任务上验证方法；它不能仅凭本项目 13% 的观测成功率，推出比在线 GRPO 更省样本。持续筛选新正例仍需采样，并继承终局验证器的正确性与数据覆盖限制。

### [6] Policy Invariance under Reward Transformations

**用于支持的论点：** 在相应 MDP、折扣和终止条件下，势函数形式的奖励变换可以保持最优策略。非 potential 的奖励缺少这种一般保证，但不能据此断言其在每个具体环境里一定改变最优策略。

**与项目的关系：** 不能把动作计数、非空提交等加性 bonus 自动视为无害整形。使用 potential shaping 时，需要匹配实际折扣、终态、截断和奖励聚合方式。

**适用边界：** 保持最优策略的理论保证，不等于有限样本训练一定更快、更稳定；把任意指标写成差分形式，也不能跳过这些条件。

## 本轮讨论的推导与项目事实

以下是结合算法公式做的分析，不是上述论文在本项目上的实验结论。

### 小 bonus 仍可能主导全失败组

若轨迹总奖励为

\[
R_i=S_i+\lambda B_i,
\]

在所有终局成功分量 \(S_i=0\) 的组中，按组标准差归一化后（忽略很小的数值稳定项，且 \(\lambda>0\)、\(\operatorname{std}(B)>0\)）：

\[
A_i\approx\frac{B_i-\overline B}{\operatorname{std}(B)}.
\]

因此，只缩小原始 bonus 系数，并不保证它对这类组的更新变弱。若 bonus 也完全相同，则中心化后仍为零优势。当前行为 bonus 可先作诊断指标；若要加入训练，应单独检验优势、loss 权重和相关组的更新占比。

### Potential shaping 不自动补上轨迹级 GRPO 的方差

若

\[
F_t=\gamma\Phi(s_{t+1})-\Phi(s_t),
\]

则完整轨迹上的折扣和满足

\[
\sum_{t=0}^{T-1}\gamma^tF_t=-\Phi(s_0)+\gamma^T\Phi(s_T).
\]

当终态 potential 为零、同组轨迹初始状态相同时，它只增加一个相同常数。如果实现按同一 \(\gamma\) 计算完整轨迹的折扣回报（或 \(\gamma=1\) 时直接求和），再生成单个组内 advantage，该常数会被中心化抵消，无法让全失败组产生新的奖励方差。若 shaping 使用 \(\gamma<1\)，聚合回报却直接无折扣求和，则不能直接套用这个常数抵消结论。逐步信用分配是否受益，需要另看具体实现。

### 当前数字的解释范围

用户报告：26 个任务各采 5 条，共 130 条，17 条终局成功，6 个任务至少成功过一次。该数据在本次文献整理中未重新审计。

- rollout 级经验成功率为 17/130，约 13.1%；任务级至少一次成功比例为 6/26，约 23.1%。二者不能直接互换。
- 仅使用二元终局奖励时，这批 26 个任务组中有 4～6 个组同时包含成功和失败：20 组全失败，另外 6 组中最多 2 组可以全成功。
- 0/5 不证明不可达。若某题单次真实成功率为 10%，5 次独立采样都失败的概率仍为约 59%。
- 历史多模型混合成功率、成功轨迹步数和模型种类数，可以帮助安排复测顺序，但不能直接充当当前策略的成功概率。

与本项目已有记录交叉阅读：[SFT 负结果记录](../../FINDINGS-2026-09-sft-collapse.md)、[2026-10-08 外部审查](../../REVIEW-2026-10-08-codex.md)、[轨迹项目设计](../../PLAN-trajectory.md)。通用论文结论不能代替对本地数据、训练实现和评测协议的核对。

## IEEE 编号样式参考文献

[1] Q. Yu *et al*., “DAPO: An Open-Source LLM Reinforcement Learning System at Scale,” in *Advances in Neural Information Processing Systems*, vol. 38, 2025, pp. 113222–113244, doi: [10.52202/085713-3775](https://doi.org/10.52202/085713-3775). [正式出版页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html).

[2] A. Setlur *et al*., “Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning,” in *Proc. 13th Int. Conf. Learning Representations (ICLR)*, 2025. [正式出版页](https://openreview.net/forum?id=A6Y7AqlzLW).

[3] Z. Xi *et al*., “Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning,” in *Proc. 41st Int. Conf. Machine Learning*, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 54030–54048. [正式出版页](https://proceedings.mlr.press/v235/xi24a.html).

[4] R. Y. Pang, W. Yuan, K. Cho, H. He, S. Sukhbaatar, and J. Weston, “Iterative Reasoning Preference Optimization,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024, pp. 116617–116637, doi: [10.52202/079017-3702](https://doi.org/10.52202/079017-3702). [正式出版页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d37c9ad425fe5b65304d500c6edcba00-Abstract-Conference.html).

[5] A. Singh *et al*., “Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models,” *Transactions on Machine Learning Research*, Apr. 2024. [正式出版页](https://openreview.net/forum?id=lNAyUngGFK).

[6] A. Y. Ng, D. Harada, and S. Russell, “Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping,” in *Proc. 16th Int. Conf. Machine Learning*, 1999, pp. 278–287. [作者站全文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

出版信息按正式版整理，未核实到正式 DOI 的条目不填 DOI。DAPO 的正式版与 arXiv 版作者名单不同，本地保存正式版，BibTeX 也采用正式版作者。ReST 的正式版与 arXiv 元数据中一位作者的姓名拼写不同，BibTeX 按正式版写作 Kshitiej Mahajan。

如目标 venue 要求 IEEE 参考文献，且 LaTeX 工程已具备相应样式文件，可使用：

```tex
\bibliographystyle{IEEEtran}
\bibliography{references}
```

若目标 venue 为 NeurIPS、ICLR、ICML 等，直接保留这份 `.bib`，改用对应模板要求的引用样式即可。
