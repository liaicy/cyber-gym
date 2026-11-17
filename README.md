# ExamSociety — 多智能体对抗式出题社会

输入一个学科领域或关键词，一个由多个 LLM 组成的"出题社会"持续运转：互相出题、互相作答、评估难度、对抗加难，并可控地注入"专家可识破"的误导信息作为鲁棒性探针——不人工停止就一直生成，产出带完整来源链（provenance）和心理测量学质检的高难度考题。

## 架构

```
领域关键词
   │
   ▼
TopicMapper ──► 子话题池（循环取用）
   │
   ▼
Proposer（出题）──► Validator（效度守门：盲解 + 良构性审查，从不看答案键）
   │                     │ 不通过 → 重出
   ▼                     ▼
Solver 面板（异构模型并行盲答）──► solve rate
   │
   │ solve rate > 阈值（太简单）
   ▼
Hardener（对抗加难 / 按概率注入误导信息探针）──► 再验证 ──► 再答
   │  （最多 max_revisions 轮）
   ▼
落盘 items.jsonl（全部版本、全部作答、全部验证记录）
```

设计要点：

- **效度守门独立于答案键**：Validator 只看题干和选项，独立求解后由代码比对答案键（`key_agrees`），不是让模型自查自评。
- **误导信息是受控探针，不是陷阱**：注入的错误声明必须"原则上可被领域专家识破"，且注入后仍有唯一正确答案；每次注入记录错误类型分类（false_premise / distractor_evidence / plausible_misattribution / spurious_constraint）、错误声明原文与可识破理由。
- **心理测量学质检**：内置经典指标（p 值、point-biserial 区分度）和 2PL IRT（区分度 a、难度 b、solver 能力 θ），自动旗标 a≤0 / 负区分度的病态题。
- **全程可追溯**：每道题保留所有历史版本、每个 solver 的作答与理由、每轮验证结论。

## 快速开始

```bash
# 需要环境变量 ANTHROPIC_API_KEY
pip install -e .

# 无限生成（Ctrl-C 停止，可断点续跑——子话题游标持久化在 state.json）
examsociety run --domain "量子力学" --out data/qm

# 只生成 5 道（调试用）
examsociety run --domain "量子力学" --out data/qm --limit 5

# 便宜模型冒烟配置
examsociety run --domain "线性代数" --out data/smoke --config configs/smoke.yaml --limit 1

# baseline 对照（单次出题、无对抗循环，用于论文对比实验）
examsociety run --domain "量子力学" --out data/qm-baseline --pipeline baseline --limit 5

# 配对模式：不做对抗加难，只产出「干净版 / 注入版」匹配对
# 用于测量误导鲁棒性（详见下节）
examsociety run --domain "量子力学" --out data/qm-pair --pipeline paired --limit 100

# 心理测量学分析 / 多组对比 / 导出干净题库
examsociety irt data/qm
examsociety report society=data/qm baseline=data/qm-baseline
examsociety export data/qm > qm_items.jsonl

# 误导鲁棒性（需 --pipeline paired 的数据）
examsociety misinfo data/qm-pair

# 知识产能：概念饱和度、还能榨出多少、重复率（表层 + 同概念改写）
examsociety coverage data/qm

# 对抗加难是不是在用「更冷僻」换难度
examsociety drift data/qm

# 出题者自我偏好 / 注入者交互（需轮换过角色的数据）
examsociety leakage data/qm

# 导出时保留难度分布，而不是只要 accepted
examsociety export data/qm --statuses all --min-solve-rate 0.2 --max-solve-rate 0.8
```

## 两种测量

**难度 / 区分度（`irt`）** 依赖 solver 面板作为"考生群体"。面板只有 3 个同族模型时，θ 被标准化成均值 0 标准差 1 后只剩一个自由度，测不出能力差距的大小，只剩排序；IRT 的 a 同样不可识别。要得到可用的区分度估计，面板需要扩到 8+ 个跨家族、能力真正拉开的模型。

**误导鲁棒性（`misinfo`）** 以**配对**为测量单位，不依赖大面板。`--pipeline paired` 对每道题产出三次面板作答：干净版、干净版对照重跑、注入版。三个量按可靠性递增：

| 指标 | 含义 | 问题 |
|---|---|---|
| `rawD` | 干净版准确率 − 注入版准确率 | 受天花板约束（弱模型掉不了那么多），与能力机械相关 |
| `derail` | P(注入版错 \| 干净版对) | 去掉了天花板，但仍混入模型自身作答不稳定性 |
| `FRAGIL` | 用对照重跑去除不稳定性后的比值估计 | **报告用这个** |
| `baited` | P(选中诱导选项 \| 干净版对) | 最锋利：排除"注入只是让题变难了"这一替代解释 |

`FRAGIL = 1 − (1−derail)/(1−unstab)`。用比值而非差值，是因为差值残留与准确率成正比的偏置——恰好是配对设计本想消除的能力混杂。

`dimension test` 报告 Spearman(准确率, 鲁棒性)：**相关弱 = 鲁棒性是独立于能力的维度**，这是这套方法是否有独立价值的判据。少于 8 个 solver 时该值不可解读。

**答案键可信度（`report`）** —— `loop.validator_consensus: N` 时，效度闸门抽 N 个**不同家族**的验证者各自盲解，首个异议即拒绝。因为只有第一个通过时才会问第二个，每次异议都对应一道**单验证者设计会放行**的题，于是分歧率就是**答案键错误率的下界**：

```
validator consensus: 6/120 second opinions overturned the first -> key error rate >= 5.0%
```

单验证者设计产不出这个数字——它与唯一验证者共享的错误对自己不可见。仍是下界：两个验证者一起犯的错要靠专家审计。

跨家族面板配置见 [configs/cross_family.yaml](configs/cross_family.yaml)；待办清单见 [TODO.md](TODO.md)。

## 配置

见 `configs/default.yaml`。关键参数：

- `solvers`：解题者面板。默认 opus 4.6/4.7/4.8 三档——故意跨版本形成能力梯度（IRT 需要能力梯度），且同族模型在本网关上稳定可用。做论文级 IRT 请扩到 30+ 个跨家族异构模型。
- `loop.target_max_solve_rate`：面板通过率降到该阈值以下才接受（默认 0.5，即多数解题者要被难倒）。
- `loop.misinfo_probability`：每轮加难时注入误导信息探针的概率。
- `loop.max_revisions`：对抗加难最大轮数。

## 数据格式

`items.jsonl` 每行一个 `ItemRecord`：`versions[]`（含 misinfo 元数据）、`validations[]`、`panel_runs[]`（逐 solver 作答）、`final_status`（accepted / too_easy / suspect_too_hard / rejected_invalid / panel_error）。`suspect_too_hard`（无人答对）的题需人工复核后才可采信。

`panel_run.solve_rate` 的分母是**实际给出答案的 solver 数**（`n_scored`），不是面板规模。当 `n_scored` 低于法定人数（面板过半，且至少 2 人）时，该题记为 `panel_error`——这表示"没测出来"而非"很难"，不设 `final_version`，因此自动被 report / export / IRT 排除。上游网关中断不会产生这类记录：生成循环会暂停等待恢复。

## 评测设计

论文实验蓝本（baseline 消融链 B0/B1/B2、专家审计协议、收敛效度方案、已知限制）见 [docs/EVALUATION.md](docs/EVALUATION.md)。

## 测试

```bash
python tests/test_offline.py   # 无 API 调用：JSON 解析 + 2PL 参数恢复 + 序列化往返
```
