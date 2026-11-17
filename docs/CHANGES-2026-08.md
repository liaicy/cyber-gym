# 改动记录：缺陷修复 + 测量能力重构（2026-08）

本次工作分三段：先修两个会污染数据的缺陷，再围绕"这个项目有没有研究意义"补上两类测量能力，最后打通多 provider 并把角色轮换起来。所有改动均有离线测试覆盖（27 项，无 API 调用）。

---

## 状态一览

本文档提出的问题**不都已解决**。每个问题在正文出现处都标了状态，这里是索引。

图例：**已修复** = 问题不存在了 ｜ **待验证** = 代码已就绪，尚未在真实数据上确认 ｜ **仅诊断** = 只能测出严重程度，问题本身还在 ｜ **未处理** = 什么都没做

| # | 问题 | 提出于 | 状态 | 处理在 |
|---|---|---|---|---|
| 1 | 断线重试逻辑从未生效 | 1 | ✅ 已修复 | 第一部分 |
| 2 | 面板全挂被判"极难题" | 1 | ✅ 已修复 | 第一部分 |
| 3 | 全 NaN 行污染 IRT 统计 | 1 | ✅ 已修复 | 第一部分 |
| 4 | 配对不匹配则 Δ 不可解释 | 2.2 | ✅ 已修复 | 2.2 最小编辑校验 |
| 5 | rawD / derail 估计量有偏 | 2.2 | ✅ 已修复 | 2.2 `FRAGIL` |
| 6 | 共享盲区导致错误答案键 | 2.3 | ✅ 可测下界 | 3.6 双验证者共识 |
| 7 | `provider` 命名歧义、family 匹配碰撞 | 3.4 | ✅ 已修复 | 3.4 |
| 8 | 12 个子话题循环到永远 | 2.3 | ✅ 已修复 | 5.1 `subtopic_refresh_every` |
| 9 | `avoid` 只看最近 15 条题干 | 2.3 | ✅ 已修复 | 5.1 按子话题过滤 + 加宽 |
| 10 | `concepts` 收了不用 | 2.3 | ✅ 已修复 | 5.1 回灌作覆盖约束 |
| 11 | 换皮不换核（改写后的同概念重复） | 2.3 | ✅ 已修复 | 5.2 `concept_duplicates` |
| 12 | **难度漂向冷僻（会伪装成成功）** | 2.3 | ✅ 已修复 | 5.3 carryover / `examsociety drift` |
| 13 | 分不清"难"与"随机" | 2.1 | ✅ 已修复 | 5.4 `solve_replicates` + 一致性 |
| 14 | 接受判据把题库压向盲猜线附近 | 2.1 | ✅ 已修复 | 5.5 难度带 + `export` 筛选 |
| 15 | `paired` 记录不进 IRT | — | ✅ 已修复 | 5.6（用干净版） |
| 16 | `near_duplicate_stems` 是 O(n²) | — | ✅ 已修复 | 5.2 倒排索引 |
| 17 | 无成本上限 / 预算开关 | — | ✅ 已修复 | 5.7 `max_api_calls` |
| 18 | EVALUATION.md 主张排序已过时 | — | ✅ 已修复 | 已重写，主张 1 换成误导鲁棒性 |
| 19 | n=3 同族面板，θ 只剩排序 | 2.1 | ⏳ 待验证 | 3.2（代码就绪，**待填端点**） |
| 20 | 出题者自我偏好泄漏 | 2.1 | ⏳ 待验证 | 3.3 轮换 + 第四部分检验 |
| 21 | 注入者身份混杂 `FRAGIL` | 3.3 坑 3 | ⏳ 待验证 | 3.3 轮换 + 第四部分检验 |
| 22 | OpenAI 兼容后端未打过真实网络 | 3.2 | ⏳ 待验证 | — |

**18 项已修复，4 项待验证，0 项未处理。**

剩下的 4 项**全部卡在同一件事上**：没有凭据，跑不了真实数据。代码早已就绪，合成数据也验证过尺子本身有效——但尺子有效不等于知道被测物是什么。这 4 项只能等 [TODO.md](../TODO.md) 第 1、2 节填完。


---

## 第一部分：两个缺陷

### 缺陷 1 — 断线重试逻辑从未生效

**症状。** `_with_outage_retry` 的设计意图是"上游网关中断时暂停等待、不要中断长跑"，但它包裹的 `generate_item` 内部把四条会抛 `LLMError` 的路径全部自行吞掉了（出题 `continue`、验证 `return verdict="fail"`、作答 `return correct=None`、加难 `break`），因此外层永远收不到异常。该函数在生成循环里是死代码，唯一真正生效的地方是只在启动时调用一次的 topic mapper。

**后果。** `llm.complete` 的内层退避约 6.3 分钟（7 次，`min(20*attempt, 90)`）。超过这个窗口后，网关中断被当作"题目不合格"：三次出题尝试全失败 → 落盘一条 `rejected_invalid` → 游标推进 → 计入 `--limit` 配额 → 继续下一题。跑一夜网关抽风得到的是一批废记录，而不是暂停。EVALUATION.md 的实验协议是三个 arm 各 `--limit 100`，若中断只砸中其中一个 arm，三组有效样本量不同而 `n_records` 仍显示 100。

**修法。** 不能简单放行 `LLMError`，因为它同时代表两类语义完全不同的失败：

| 类别 | 例子 | 正确处理 |
|---|---|---|
| 基建故障 | 重试耗尽、连接失败、持续 5xx | 等待后重试（重试有意义） |
| 请求级故障 | 无法解析出 JSON、被安全分类器拒绝、空响应、misinfo 元数据缺失 | 换一题重出（重试无意义） |

直接放行会让"模型死活不输出合法 JSON"变成无限循环。因此拆分异常层次：

```
LLMError                    请求级：重试无用
├── TruncatedError          输出被 max_tokens 截断（已有）
└── UpstreamUnavailable     基建：值得等（新增）
```

- `llm.complete` 的 `for...else` 分支（重试耗尽）改抛 `UpstreamUnavailable`
- 四个角色统一在原 catch 之前加 `except UpstreamUnavailable: raise`
- `_with_outage_retry` 只对 `UpstreamUnavailable` 无限等待；请求级 `LLMError` 给 3 次有界重试后抛出（原实现若收到请求级错误会永远空转，这是修复时顺带堵上的反向缺陷）

**Solver 的设计选择。** 单个 solver 遇到 `UpstreamUnavailable` 也向上抛，会中断整题。理由：`UpstreamUnavailable` 只在约 6.3 分钟退避耗尽后才产生，此时静默降级成 2 人面板产生的数据比暂停更糟。永久失效的模型（如模型名拼错 → 404）不属于 5xx，走的是请求级路径，由下面的法定人数机制兜住。

### 缺陷 2 — 面板全体失败被判定为"极难题"

**症状。** `solve_rate` 的分母是**实际给出答案的 solver 数**，全体失败时 `scored` 为空、`solve_rate = 0.0`；而分类逻辑把 `solve_rate == 0.0` 判为 `suspect_too_hard`。即：API 全挂 = 产出一道"无人答对的神级难题"。

**更常见的隐蔽变体。** 代码没有任何最小样本量检查，分母会悄悄缩水：

| 3 个 solver 的实际情况 | solve_rate | 系统判断 |
|---|---|---|
| 2 个报错，1 个答对 | 1/1 = 1.00 | 太简单 → 触发对抗加难 |
| 2 个报错，1 个答错 | 0/1 = 0.00 | suspect_too_hard |
| 1 个报错，2 个答对 | 2/2 = 1.00 | 太简单 → 触发对抗加难 |

用 n=1 的样本判定"这题太简单，需要加难"，然后把加难后的版本写进数据集——噪声在驱动整条流水线，而 `report` 层面完全看不见。

**下游污染。** 废记录进入 `response_matrix`（`suspect_too_hard` 在白名单内），整行为 NaN：

- `p` 保持 NaN（有 `row_has_data` 保护）
- `pb` 走 `x.size < 2` 分支 → `0.0`，拉低 `mean_point_biserial`
- `fit_2pl` 中 `if not mask[i].any(): continue` 跳过该题，但 `log_a = log_a + math.log(sd)` 这行的尺度归一化对**被跳过的题照样累加**，30 轮后把它推向 `a ≈ 0`，并被打上 `low_discrimination` 旗标
- `mean_final_solve_rate` 被 `0.0` 拉低

最后一条危害最大：它是 EVALUATION.md 主张 1 的头号指标（"solve rate 越低越难"）。**API 挂得越多，报告显示的题目越难。** 同时 `low_discrimination` 旗标会虚高——而"被旗标剔除的比例"正是用来衡量"对抗循环产生病态题的速率"的指标。

**修法。**

- `schemas.PanelRun` 增加 `n_scored: Optional[int]`（`None` 表示本字段出现前的旧记录）
- `society.py` 抽出三个模块级纯函数，使分类规则不依赖 LLM 即可测试：
  - `panel_quorum(n)` → `min(n, max(2, (n+1)//2))`，即面板过半且至少 2 人（3 人→2，2 人→2，1 人→1）
  - `aggregate_panel(version, results)` → 记录 `n_scored`
  - `classify_panel(panel, n_solvers, target)` → 不足法定人数返回 `panel_error`
- 不足法定人数时 `final_version` **保持 `None`**。这是关键设计：`final_item()` 返回 `None`，于是 report / export / psychometrics 三处自动排除，与 `rejected_invalid` 走同一条路，无需改动那三个文件。题面与每个 solver 的报错保留在记录中供诊断
- `psychometrics.response_matrix` 追加一道防线：全 NaN 行直接不进响应矩阵，使已产生的旧数据重新分析时也不会再污染

### 一处更正

对话中我曾称废记录在 IRT 表里显示为 `a = 1.00` 且无旗标、会**虚增** `mean_irt_a`。实测为 `a ≈ 0.00` 且被标记 `low_discrimination`，方向相反，是**拉低**平均区分度。原因是上文所述 `log_a += log(sd)` 对被跳过的题持续累加。危害结论不变（且多了一条"虚高病态题率"），但具体表现以本节为准。

---

## 第二部分：两类测量能力

背景问题是"这个项目有没有研究意义"，拆成两条：题目能否有区分度、模型能否蒸馏出足够多的有效知识。

### 2.1 为什么难度主线拿不到区分度

三条结构性原因，记录在此以免后续反复讨论：

1. **难与有区分度在底部互相冲突。** 接受判据 `solve_rate ≤ 0.5` 把题库压向 20%–35%（5 选 1 盲猜是 20%）。全对和全错的方差都是 0，`a = 0`；区分度最大在 `p ≈ 0.5` 附近，且题库需要 `b` **有分布**。而 `export` 只导出 `accepted`，恰好丢掉了提供测量精度的中等难度题。
2. **分不清"难"和"随机"。** 3 个 solver 各答 1 次、接受条件 ≤1 个答对。一道所有模型真实正确率均为 0.3 的**纯随机**题（歧义、选项区分不清），被接受的概率约 0.78。筛出来的是**回答不稳定**的题，而不稳定 = 低信度 = 区分度趋近 0。唯一的辨别办法是同一模型对同一题**重复作答 k 次**。
3. **n=3 且同族，θ 不含信息。** 3 个数被约束成均值 0、标准差 1 后只剩一个自由度，取值锁在 ±√2 内。实测：真实正确率 `[0.40, 0.50, 0.60]` 与 `[0.10, 0.50, 0.90]` 标准化后得到**完全相同**的 θ = `[-1.22, 0, 1.22]`。"差多少"这个信息被彻底抹掉，只剩排序。且 opus 4.6/4.7/4.8 能力序几乎处处单调，"有区分度"约等于"新版本会、旧版本不会"，测的是版本新旧。

> **状态**
> - 第 1 条（压向盲猜线）**✅ 已修复**（5.5）。接受判据改成难度**带**（`target_min_solve_rate`），`export` 支持按状态与 solve rate 筛选，不再只吐 `accepted`
> - 第 2 条（难 vs 随机）**✅ 已修复**（5.4）。`solve_replicates` 让同一模型对同一题答 k 次，`response_consistency` 把不稳定题暴露出来
> - 第 3 条（n=3 同族）**⏳ 待验证**。3.2 的多 provider 让跨家族面板成为可能，但要等端点填好才知道实际梯度

另有一条会污染全部结论的泄漏：默认配置 proposer / validator 均为 `claude-opus-4-8`，而 4.8 **也在 solver 面板里**——题是它出的、参考解是它写的、然后它被这道题打分。自我偏好效应会表现为漂亮的正区分度并被误读为"测出了能力差异"。


> **状态 ⏳ 待验证。** 3.3 让出题者跨家族轮换、把作者身份记进 `ItemVersion.author_family`；第四部分实现了 Mantel–Haenszel 检验并在合成数据上验证过。但**尚未在真实运行上跑过**，所以泄漏实际有多大仍然未知。

### 2.2 配对误导鲁棒性（新增，`--pipeline paired` + `examsociety misinfo`）

把测量单位从"一道题"换成"一对题"，同一道题的干净版与注入版由同一面板作答，**相减时题目难度抵消**。这使得测量不再依赖大面板或高难度题，绕开了 2.1 的全部三条问题。

**实现要点：**

- `Misinfo.target_option_index` — 记录虚假声明诱导向哪个错误选项。`_item_from_obj` 校验它在范围内且**不等于**正确答案
- `Hardener.SYSTEM_INJECT` + `Hardener.inject()` — 最小编辑注入模式，要求逐字保留选项、不改变答案位置、不加深题目本身
- `agents.check_minimal_edit(clean, injected)` — **代码级**配对校验（选项集合与顺序、答案位置、题干确实变了、misinfo 元数据完整）。仅靠 prompt 约束不够：若允许改写选项或移动答案键，Δ 里就混进了未受控的难度变化，配对设计的前提失效。校验不过直接抛出
- `society._make_pair()` — 干净版 → **对照重跑** → 注入 → 再验证 → 注入版
- `config.LoopConfig.pair_control_replicate: bool = True`

**估计量的三次迭代**（这是本次工作中最重要的方法论修正，务必保留）：

| 记号 | 定义 | 问题 |
|---|---|---|
| `rawD` | 干净版准确率 − 注入版准确率 | 受天花板机械约束（30% 的模型最多掉 30pt，90% 的能掉 90pt），与能力人为相关 |
| `derail` | P(注入版错 \| 干净版对) | 去掉天花板，但仍混入模型**自身作答不稳定性**——本来就飘的模型不注入也会翻，导致弱模型被带偏率系统性虚高 |
| `FRAGIL` | `1 − (1−derail)/(1−unstab)` | **报告用这个** |
| `baited` | P(选中诱导选项 \| 干净版对) | 最锋利：排除"注入只是让题变难了"这一替代解释——变难不会把答案推向某个预定选项 |

`FRAGIL` 用**比值而非差值**的推导：把每次作答建模为以某个逐题概率独立答对。记 `B = P(干净版重跑对 | 干净版对)`、`C = P(注入版对 | 干净版对)`，则噪声项在两者中相同并在比值中约掉：

```
C = (1 − φ) · B      ⟹      φ = 1 − C/B
```

用差值会残留一个与准确率成正比的偏置，恰好是配对设计本想消除的能力混杂。

**合成数据验证。** 构造 6 个模型，能力与脆弱度**故意交叉**（如 `m_b` 强但脆、`m_e` 弱但稳），真实脆弱度已知，400 对：

| 估计量 | 平均绝对误差 | 维度检验 rho | 相对真值偏差 |
|---|---|---|---|
| rawD | 12.8% | +0.09 | **+0.80（符号都错）** |
| derail（未校正） | 26.8% | −0.89 | −0.17 |
| **FRAGIL** | **6.7%** | **−0.60** | **+0.11** |

真值 `Spearman(准确率, 脆弱度) = −0.71`。关键是最后一列：**用未校正的 derail 会得到 −0.89，据此判定"鲁棒性只是能力的代理、没有独立价值"——而这个结论是错的。** 这正是决定项目走不走的判断，因此对照重跑不是可选项。

`misinfo.dimension_test()` 报告 `Spearman(准确率, 鲁棒性)`：相关弱 = 鲁棒性是独立于能力的维度 = 项目有独立价值。少于 8 个 solver 时该值不可解读，代码会警告。

### 2.3 知识产能（新增，`examsociety coverage`）

回答"模型能否蒸馏出足够多的有效知识"。前提认知：proposer 写题、validator 盲解，二者是同一模型——**不是在创造知识，是把权重里的隐性知识外化成可测形式**。题库的知识天花板即 proposer 的能力，这个方法做不出"超越前沿"的 benchmark，但用于区分现有模型是够的。

**三处硬约束限制了覆盖度：**

1. 12 个子话题循环到永远（`subtopics[cursor % len(subtopics)]`），TopicMapper 只在开头跑一次。跑 10000 题 = 12 个子话题各摊 833 题
2. 去重只看**最近 15 条**题干（`avoid[-15:]`，截断到 120 字符）。写到第 100 道时 proposer 不知道第 10 道写了什么。而全部历史题干被读进内存——付了全量解析成本只用了 15 条
3. `concepts` 字段**收集了但从不使用**（`schemas.py` 定义、`cli.py` 导出，中间无任何消费）

> **状态：三条都已修复（5.1）。**
> - 第 1 条 **✅**：`subtopic_refresh_every` 每 N 题让 TopicMapper 针对已覆盖概念扩展子话题池，天花板从"开头那一池"变成"这个领域"
> - 第 2 条 **✅**：`avoid` 改为**按子话题过滤**并加宽到 `avoid_window`（默认 40）。原来的全局最近 15 条在子话题轮转下几乎全是别的话题，约束不了任何东西
> - 第 3 条 **✅**：`concepts` 回灌给 proposer 作覆盖约束（"这些概念已经考过，换一个"）

**预期退化模式**，其中第 2 种最需警惕：

1. 换皮不换核——同一概念换数字换情境。IRT 完全看不出来，两道考同一概念的题 `a` 和 `b` 都正常
2. **难度来源从"深"漂向"偏"**——这是对抗加难循环的**系统性偏置**：当更深的推理写不动了，让题变难最省力的办法是换更冷僻的事实。此时 solve rate 漂亮下降，指标显示成功，但测的东西已从推理能力变成记忆广度，且违反 `SYSTEM_MISINFO` 自己写的 "never rely on obscure trivia"。**它会伪装成成功**
3. 共享盲区导致的错误答案键——同族 validator 给出的 `key_agrees` 是虚高的

> **状态：三种都已覆盖。**
> - 第 1 种 **✅**（5.2）：新增 `concept_duplicates`，按概念集 Jaccard 比对。改写措辞会换掉每一个字符 bigram 却留下同一个概念标签，这正是 stem 相似度结构上看不见的那一半
> - 第 2 种 **✅**（5.3）：新增 carryover 检验（`examsociety drift`）。**注意最初设想的"Δ区分度"是错的**——单次作答时每个 solver 的正确率只能是 0/1，标准差完全由均值决定，不含额外信息。实际用的是关联度：`P(改后对|改前对) − P(改后对|改前错)`
> - 第 3 种 **✅ 可测下界**：3.6 的双验证者共识给出 `key_error_lower_bound`。仍是下界，两个验证者一起犯的错要靠专家审计

**实现——注意：这是诊断，不是解决方案。** 上面三处硬约束一个都没修，本节只是让它们的后果变得可见（修法在末尾的工程队列）。退化模式 1 只能拓到表层，**退化模式 2 完全没有度量**，退化模式 3 在 3.6 双验证者共识里。

把概念当物种、题目当采样，用标准的物种丰富度估计量：

- `accumulation_curve` — 不重复概念数 vs 已生成题数，拐平处即该领域的有效产能
- `richness` — Chao1（偏差校正型 `S_obs + f1(f1−1)/(2(f2+1))`，因经典式 `f1²/(2f2)` 在无 doubleton 时未定义）+ Good–Turing 覆盖率 `1 − f1/n`
- `near_duplicate_stems` — 字符 bigram 的 Jaccard 相似度（对中文和英文都适用；词切分会把整个中文子句变成一个 token）。只能抓表层近重复，改写过的概念重复需要 embedding，故计数是重复率的**下界**
- `new_concepts_last_quarter` — 最后 25% 的边际产出，回答"还在发现新材料还是在换措辞"

---

## 第三部分：多 provider 与角色轮换

### 3.1 改动前的网关设计

`LLMClient` 只构造**一个** `anthropic.Anthropic()`，所有角色和 solver 共用；端点与凭据完全由环境变量决定（`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`）；模型是逐次调用传的参数。因此"异构面板"在原实现里的含义仅是"换模型名字符串"，`config` 里的 `provider` 字段从未被读取。`_needs_header_scrub` 会在 base_url 非 `*.anthropic.com` 时抹掉 `user-agent` 与 `x-stainless-*` 指纹头——说明这个仓从一开始就是冲第三方网关设计的。

判断部署形态：把非 Claude 模型名填进 `solvers` 能跑通 ⇒ 聚合网关（只需改配置）；返回 404/400 ⇒ 各家官方 API（需要 provider 抽象）。

### 3.2 provider 抽象

`llm.py` 重构为「后端分发」：

- `ModelRef` — 纯 dataclass，承载 id / model / provider / base_url / api_key_env / token_param / family。放在 `llm.py` 以免该模块反向依赖 config
- `_AnthropicBackend` — 按 (base_url, api_key_env) 缓存客户端，保留原有的指纹头清洗
- `_OpenAICompatBackend` — 裸 `httpx` 实现 chat-completions。覆盖 GPT、Gemini（OpenAI 兼容端点）、DeepSeek、Qwen、Kimi、GLM 及多数聚合网关。不用 openai SDK 是因为不引新依赖，且网关对请求头挑剔时需要完全控制
- 两个后端把各自的错误**归一**为 `_Retryable` / `TruncatedError` / `LLMError` 三种结果，重试与退避逻辑上移到 `LLMClient.complete` 共用

最后一条是关键：**若新后端不做这层归一，第一部分修好的断线语义会在新家族上静默失效。** 具体映射：

| Anthropic | OpenAI 兼容 | 归一结果 |
|---|---|---|
| `RateLimitError` / `InternalServerError` / `APIConnectionError` / 5xx | HTTP 429 / 5xx / `TransportError` | `_Retryable` → 耗尽后 `UpstreamUnavailable` |
| `stop_reason == "max_tokens"` | `finish_reason == "length"` | `TruncatedError` |
| `stop_reason == "refusal"` | `finish_reason == "content_filter"` | `LLMError` |
| 4xx（非 429） | HTTP 4xx（非 429） | `LLMError` |

`config.ModelSpec` 支持裸字符串简写，`RolesConfig` 接受旧的单数键（`proposer:` → `proposers: [...]`），三个既有 YAML 无需修改即可加载。

### 3.3 角色轮换

出题者轮换既堵掉自我偏好泄漏，又扩大知识池（不同家族知识分布不同，Chao1 的池子会变大）。实现时避开了三个坑：

**坑 1 — 取模轮换会把出题者和子话题绑死。** 原代码是 `subtopics[cursor % 12]`；若照搬加一句 `proposers[cursor % 4]`，因 `gcd(12,4)=4`，出题者 0 永远只拿到子话题 0/4/8，四个家族被锁在互不相交的子集上，家族效应与子话题效应无法分离。`ExamSociety.assign()` 改为遍历叉积（`cell % n_sub`, `cell // n_sub`），均衡由构造保证。有回归测试。

**坑 2 — 验证者必须跨家族。** 否则 Claude 出的题被同族验证（门槛松、`key_agrees` 虚高），别家出的题被跨族验证，通过率的家族差异成了验证配置的产物。`pick_validator()` 在 `loop.cross_family_validation` 为真时优先选异族；`ValidationResult` 记录 `validator_id` / `validator_family` / `cross_family`。

**坑 3 — 注入者身份会混杂主指标。** 虚假声明由 Hardener 撰写，若固定一家，那些假话带着该家族对"什么听起来可信"的先验；同族模型完全可能因共享先验而更易（或更难）识破，直接污染 `FRAGIL`。因此 hardener 独立于 proposer 轮换（`index % len(hardeners)`，与叉积周期错开），`ItemVersion` 记录 `author_id` / `author_family`。分析时需检验「注入者家族 × 应试者家族」交互——若显著，`FRAGIL` 就不是应试者的纯属性。

**出题者是否排除出自己题目的面板：不排除。** 记录作者身份、分析时两种口径各算一遍——自我偏好效应的大小本身就是可报告的结果。确认存在后再在导出题库时剔除作者格。

### 3.4 命名更正与一个真 bug

**`provider` → `api`。** 原命名有歧义：它选的是**协议**不是**厂商**——DeepSeek / Qwen / Kimi / GLM / Gemini 都说 OpenAI 的 chat-completions 协议，共用同一份客户端，区别只在 `base_url` 与 key。厂商身份由 `family` 承载，作者优势与注入者交互分析按 `family` 分组。旧的 `provider:` 键作为别名保留。

**family 匹配顺序 bug。** `_FAMILIES` 按顺序做子串匹配，而 `"doubao-1.5"` 含子串 `"o1"`，豆包会被归入 `o1` 家族。已改为长名优先（`sorted(key=len, reverse=True)`）并加验证。

### 3.5 为什么角色一律用强档，弱模型只进 solvers

**原则：应试面板是被测群体，需要能力梯度；出题 / 验证 / 注入是仪器，仪器必须一致。**

- **validator 混编强弱最危险**：弱验证者会拒掉强验证者会放行的题，过闸率随抽到谁而变，存活下来的题被"恰好抽到哪个验证者"非随机筛选
- **injector 弱了也不行**：写一句"够可信、能骗到人、专家能识破、且不破坏唯一正确答案"的假话是很难的生成任务；弱模型要么写得太明显（无效果）要么把题写坏（过不了闸）。「攻击强度」本身值得研究，但要单独设臂，不能混进主测量
- **成本上没有省的必要**：配对模式每题约 37 次调用（1 出题 + 1 验证 + 11 干净 + 11 对照 + 1 注入 + 1 验证 + 11 注入），角色只占约 1/10，面板才是大头

**唯一例外是 proposer 可以偏中档。** 配对模式下 FRAGIL 的样本量是 `n_eligible`（干净版答对的题数），题越难弱 solver 的 eligible 越少——恰好让最需要测的模型样本量最小。合成验证里 `m_f` 就只有 130/400 对 eligible，估计噪声明显偏大。中等难度题统计功效更高，而且更便宜。

### 3.6 双验证者共识（`loop.validator_consensus`）

`validator_consensus: N` 时，效度闸门抽 N 个**不同家族**的验证者，各自盲解并与答案键比对，**首个异议即拒绝并短路**（省掉剩余调用）。

短路带来的统计性质正是想要的：只有第一个验证者通过时才会问第二个，所以每一次异议都对应**一道单验证者设计会放行的题**。因此：

```
key_error_lower_bound = 异议数 / 第二意见总数
```

这是**答案键错误率的下界**——单验证者设计根本产不出这个数字，因为它与唯一的验证者共享的错误对它自己不可见。EVALUATION.md 主张 2 原本只能靠人工抽检估计这一项。仍然只是下界：两个验证者一起犯的错依然隐藏，那部分要靠专家审计。

实现：`ExamSociety.pick_validators(author_family, n)`（异族优先、彼此不同族，家族不够时才允许重复）、`validate_item()`、模块级 `consensus_passed()`；`ValidationResult.version` 用于把同一轮共识的多条验证归组（`record.validations` 是跨版本的平铺列表）；`report.validation_agreement()` 计算并在 `examsociety report` 里输出：

```
validator consensus: 6/120 second opinions overturned the first -> key error rate >= 5.0%
  cross-family second opinions: 120, overturn rate 5.0%
```

`configs/cross_family.yaml` 是可填的模板（两种部署形态的说明 + 4 出题者轮换 + 11 个跨 7 家族的 solver 面板 + `validator_consensus: 2`）。

## 第四部分：两个泄漏检验

轮换（3.3）只是让检验**成为可能**，本节是检验本身。两个问题是同一个统计形状：

| 检验 | 问题 | 若成立的后果 |
|---|---|---|
| 作者优势 | solver 在**自己家族出的题**上是否更强？ | `a` / `θ` 测的是"这题谁写的"而不是能力 |
| 注入者交互 | solver 是否更不容易被**自己家族写的假话**带偏？ | `FRAGIL` 是「应试者 × 注入者」配对的属性，不是应试者的属性 |

两者都是 differential item functioning：同一个结局、两个组、外加一个必须**条件化而非平均掉**的干扰变量（这道题 / 这句假话有多难）。所以用 DIF 的标准估计量 **Mantel–Haenszel**，连同 ETS 的 delta 分级（`|Δ|<1.0` 可忽略 / `1.0–1.5` 中等 / `≥1.5` 大）——这样"多大算大"有现成标准，而不是我随口定的阈值。

**分层变量一律是留一法**：用**其他** solver 在同一题上的表现来度量难度。用 solver 自己的结果会让比较变成循环论证。

- 作者优势：用 **v0 及 v0 的首次面板作答**——题目刚被提出、还没被加难或注入改写过作者身份的状态。按"其他 solver 答对了几个"分层
- 注入者交互：只取该 solver **干净版答对**的配对（只有这些才谈得上被带偏），按"其他合格 solver 有几个被带偏"分层，即按这句假话有多强分层

**合成数据验证**（4 个家族轮流出题 / 注入，500 对，真实效应已植入）：

| | 植入泄漏（作者 +12pt，注入者盾 20pt） | 无泄漏对照 |
|---|---|---|
| 作者优势 POOLED | OR=1.72, Δ=−1.28, p<0.0001 → **B moderate** | OR=0.95, Δ=+0.13, p=0.65 → A negligible |
| 注入者交互 POOLED | OR=0.63, Δ=+1.08, p=0.0006 → **B moderate** | OR=0.84, Δ=+0.42, p=0.22 → A negligible |

**POOLED 行不是顺手加的。** 第一版只做逐 solver 检验，结果注入者效应虽然真实存在、4 个 solver 的 OR 全是 0.62（方向完全一致），却因为每人只有约 86 对同族注入样本而**全部不显著**（p=0.08–0.12）。原因是作者身份在 k 个家族间轮换时只有约 1/k 的题是同族的，逐 solver 检验功效严重不足。把 solver 本身也作为分层变量合并，比较仍严格保持在 solver 内部，功效则恢复。**逐 solver 全 A 而 POOLED 是 B/C，是典型情形，不是矛盾。**

配套：`SolverResult.family`（旧记录回退到 `guess_family(model)`）、`config.guess_family()` 提取为公共函数、CLI 新增 `examsociety leakage <run>`。

## 第五部分：把仅诊断和未处理的问题清掉

第二部分只给出了诊断，本部分是治疗。

### 5.1 知识产能的三处硬约束

- **子话题池会长大**。`TopicMapper.expand()` 拿已用子话题 + 已考概念去要新的，`loop.subtopic_refresh_every`（默认 50 题）触发。原来的天花板是"开头那一池的大小"，与跑多久无关
- **`avoid` 按子话题过滤**。原来是全局最近 15 条题干截断到 120 字符；子话题轮转之后，最近 15 条几乎全属于别的子话题，对当前这道题不构成任何约束。现在 `run()` 维护 `history[subtopic] = {stems, concepts}`，只喂同子话题的最近 `avoid_window` 条
- **`concepts` 回灌**。proposer 的 prompt 里现在有"这些概念在本子话题已经考过，测别的、不要改写"。数据早就在收集，只是从没被消费过

### 5.2 同概念改写重复 + 索引化

`concept_duplicates()` 按概念集 Jaccard 比对。改写场景会换掉每一个字符 bigram，却留下同一组概念标签——这正是 stem 相似度**结构上**看不见的那一半。有回归测试：两道措辞完全不同、概念相同的中文题，`near_duplicate_stems` 判 0，`concept_duplicates` 判 1。

两者共用 `_jaccard_pairs()`，改为倒排索引：Jaccard ≥ t 的一对必然至少共享一个元素，索引只提出可能合格的候选，再用 `Jaccard ≤ min/max` 的尺寸界在算交集之前刷掉大部分。

### 5.3 难度漂向冷僻：carryover

**最初的设想是错的。** 我原本打算按 revision round 算「Δ难度 vs Δ区分度」，用每个 solver 正确率的标准差衡量区分度。写测试时发现：`solve_replicates = 1` 时每个 solver 的正确率只能是 0 或 1，其标准差是均值的确定性函数——`d_spread` 不含任何超出难度本身的信息。3 对 1 和 1 对 3 的标准差完全相同。

换成对二值数据有效的统计量——**关联度**：

```
retention = P(改后对 | 改前对)
flip_in   = P(改后对 | 改前错)
carryover = retention − flip_in
```

靠**深度**加难，能做原版的人依然领先，carryover 保持高位；靠**冷僻**加难，成败取决于是否恰好背过某个事实，与谁能做原版关系不大，carryover 塌向 0——而 solve rate 照样漂亮地掉，被其他所有指标记为成功。

`examsociety drift` 输出，`carryover < 0.15` 判为 OBSCURITY DRIFT。测试用两组构造数据验证：深度组 carryover > 0.5，冷僻组（v1 成败均匀分布在各 solver 上，与 v0 无关）carryover ≈ 0。

### 5.4 分不清难与随机：重复作答

`loop.solve_replicates`（默认 1，不改变现有行为）。`run_panel_replicated()` 答 k 次，`pool_runs()` 把 k 次合并后再分类——一道 4 次里对 2 次的题落在 0.5，而不是取决于恰好存了哪一次。`psychometrics.response_consistency()` 读出重复间的一致率。

低一致性意味着筛出来的是**回答不稳定**的题而非难题，而不稳定 = 低信度 = 区分度趋近 0。这是 2.1 第 2 条唯一的解法。

### 5.5 接受判据改成难度带

`classify_panel` 增加 `target_min_solve_rate`（默认 0.0，保持原行为），低于下界记为新状态 `too_hard`。只有天花板会把整个题库推向盲猜线，那里方差为 0、区分度为 0——**题库"分数"越好，测到的越少**。

`examsociety export` 增加 `--statuses`（默认仍是 `accepted`，可传 `all`）与 `--min-solve-rate` / `--max-solve-rate`，并把 `final_status` / `solve_rate` / `author_family` 一起导出，下游可以自行分带。

### 5.6 paired 记录进 IRT

`SCORABLE_STATUSES` 增加 `paired` 与 `too_hard`。新增 `_scored_run()` 决定一条记录用哪次面板作答代表自己：

- **paired 记录用干净版**——注入版是被刻意扰动的变体，它的难度是探针的属性而不是题目的属性
- 其余用 final version 的**第一次**作答（`solve_replicates > 1` 时有多次，取第一次是一次无偏测量，而不是原来的"隐式取最后一次"）

### 5.7 成本上限

`loop.max_api_calls`（0 = 不限）。`LLMClient._charge()` 逐次计数，含重试——那才是真正花钱的地方。超限抛 `BudgetExhausted`，它**刻意不继承 `LLMError`**：否则会被某个角色的降级处理吞掉，或被断线重试循环无限重试。`run()` 捕获后干净退出。

## 逐文件改动

| 文件 | 改动 |
|---|---|
| `examsociety/llm.py` | 新增 `UpstreamUnavailable`；重试耗尽改抛该异常；**重构为 `ModelRef` + 双后端分发（`_AnthropicBackend` / `_OpenAICompatBackend`），退避逻辑上移共用** |
| `examsociety/agents.py` | Validator / Solver 放行 `UpstreamUnavailable`；`target_option_index` 解析与校验；`SYSTEM_INJECT` + `Hardener.inject()`；`check_minimal_edit()`；`revise` 的 misinfo schema 补 `target_option_index` |
| `examsociety/society.py` | 新增 `panel_quorum` / `aggregate_panel` / `panel_is_scorable` / `classify_panel` 四个纯函数；`run_panel` 改用 `aggregate_panel`；`generate_item` 两处法定人数检查 + Proposer/Hardener 处放行 outage；`_make_pair()`；`_abandon_unscorable()`；`_with_outage_retry` 区分两类异常；**`assign()` 叉积轮换、`pick_validators()`、`validate_item()` 共识闸门、模块级 `consensus_passed()`** |
| `examsociety/schemas.py` | `PanelRun.n_scored`；`Misinfo.target_option_index`；`final_status` 增加 `panel_error` 说明；**`ItemVersion.author_id/author_family`、`ValidationResult.version/validator_id/validator_family/cross_family`** |
| `examsociety/config.py` | `LoopConfig.pair_control_replicate` / `cross_family_validation` / `validator_consensus`；`provider` → `api`（协议而非厂商，旧键保留为别名）；family 匹配改长名优先；**`SolverSpec` → `ModelSpec`（含 provider/base_url/api_key_env/family），角色改为列表并兼容旧单数键** |
| `examsociety/psychometrics.py` | `response_matrix` 丢弃全 NaN 行；**`_scored_run()`、`response_consistency()`、`revision_drift()` + carryover 判据** |
| `examsociety/report.py` | 新增 `validation_agreement()`（答案键错误率下界），并入 `report` 输出 |
| `examsociety/coverage.py` | **新增** — 知识产能诊断；**`concept_duplicates()`、`_jaccard_pairs()` 倒排索引** |
| `examsociety/misinfo.py` | **新增** — 配对误导鲁棒性测量 |
| `examsociety/leakage.py` | **新增** — 作者优势 / 注入者交互的 Mantel–Haenszel DIF 检验 |
| `examsociety/cli.py` | `--pipeline` 增加 `paired`；新增 `coverage` / `misinfo` / `leakage` / `drift` 子命令；`export` 增加 `--statuses` 与 solve-rate 筛选 |
| `tests/test_offline.py` | 3 项 → 27 项 |
| `configs/cross_family.yaml` | **新增** — 跨家族轮换模板 |
| `README.md` | 状态列表补 `panel_error`；`solve_rate` 分母说明；新增「两种测量」一节与新命令 |

**新增测试（18 项）：** 法定人数与分类规则（含"1 人答对 → 不得触发加难"）、全 NaN 行不进 IRT、outage 穿透 vs 请求级降级、重试策略两分支、最小编辑校验的五种破坏、配对统计（含一个 `rawD = 0.0` 但 `derail = 50%` 的用例，把"raw Δ 会骗人"固化下来）、维度检验、覆盖度估计量、近重复检测、出题者轮换均衡性（含叉积 vs 取模的对照）、跨家族验证者选取、OpenAI 兼容后端错误映射（httpx.MockTransport）、新后端上的持续 503 → `UpstreamUnavailable`、共识验证者池（异族且互不同族）、答案键错误率下界的计数逻辑、Mantel–Haenszel 手算校验、泄漏检验对已知植入效应的检出与无效应时的不误报、family 回退。

---

## 已知限制与未完成项

### 阻塞性

1. ~~跨家族面板在代码层面不被支持~~ — 已由第三部分解决。**剩余动作：把 `configs/cross_family.yaml` 里的 `<...>` 换成真实端点与模型名。** OpenAI 兼容后端仅经 `httpx.MockTransport` 验证过，**未打过真实网络**。
2. **EVALUATION.md 仍是以"难度"为主张 1 的旧版本**，与 2.1 / 2.2 的结论不一致。
3. ~~两个新分析尚未落地~~ — 已由第四部分解决。
4. 需要对方手动补的凭据 / 端点 / 决定见 [TODO.md](../TODO.md)；本节其余各条是本侧的工程队列。

### 非阻塞

3. paired 记录不进 IRT（`response_matrix` 白名单无 `paired`）。若要同时用配对数据估难度需放开
4. `near_duplicate_stems` 是 O(n²)，上万题会慢（有长度预筛，但仍需索引化）
5. 难度主线若保留，需要**重复作答 k 次**才能分离"难"与"随机"（见 2.1 第 2 条）
6. `--limit` 仍把 `rejected_invalid` / `panel_error` 计入配额。缺陷 1 修复后网关中断已不再刷废记录，问题基本自愈，剩下的是真实内容级失败，计入配额是合理语义——但需要时可改
7. 无成本上限 / 预算开关。默认配置每题约 20+ 次 Opus 调用（1 出题 + 1 验证 + 3 作答，再乘最多 3 轮加难），paired 模式约 12 次
8. 提高知识产能的杠杆均未实现：`concepts` 反馈给 proposer 作覆盖约束（免费，数据已有）、递归展开子话题、轮换 proposer 跨家族（顺带解决自我偏好泄漏）、**用源文档接地生成**（唯一能突破 proposer 知识天花板的办法，且能让 README 承诺的 provenance 变成真正的外部来源链）

### 验证状态

- 27 项离线测试通过，无 API 调用
- 两个新子命令用合成数据端到端跑通
- **未做端到端真实运行**（需 `ANTHROPIC_API_KEY`）。建议先跑 `examsociety run --domain "线性代数" --out data/smoke --config configs/smoke.yaml --limit 1`。注意 `smoke.yaml` 只有 2 个 solver，法定人数即为 2，任一失败即 `panel_error`——对 2 人面板这是正确的（挂 1 个后通过率只能是 0 或 1）
- 环境：系统默认 `python` 为 3.9，低于 `pyproject.toml` 要求的 3.10；conda base 有 numpy/scipy/pydantic/yaml 但无 `anthropic`。测试在临时 venv `C:\Users\localmgr\AppData\Local\Temp\claude\esv` 中运行（继承 site-packages），可随时删除
