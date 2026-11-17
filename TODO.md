# TODO —— 只有你能做的事

只有你有凭据、只有你知道网关实际情况、只有你能拍板的，都在这里。
代码侧已清零，改动与状态见 [docs/CHANGES-2026-08.md](docs/CHANGES-2026-08.md) 顶部的状态一览表。

---

## 1. 先做这个（5 分钟，决定后面要不要填一堆东西）

把一个非 Claude 的模型名直接填进 `configs/default.yaml` 的 `solvers` 里跑一次：

```bash
examsociety run --domain "线性代数" --out data/probe --limit 1
```

- **跑通** → 你走的是聚合网关，用 Anthropic 协议代理所有家族。**第 2 节整节跳过**，只要把 `configs/cross_family.yaml` 里所有条目保持 `api: anthropic`、改 `model:` 名、删掉 `base_url` / `api_key_env`
- **返回 404 / 400** → 各家官方 API，按第 2 节逐项填

---

## 2. 凭据与端点（各家官方 API 时才需要）

全部填在 [configs/cross_family.yaml](configs/cross_family.yaml)，把 `<...>` 换掉。

### 2.1 环境变量

| 变量 | 家族 | ☐ |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | ☐ |
| `ANTHROPIC_BASE_URL` | 走网关才需要，否则留空 | ☐ |
| `OPENAI_API_KEY` | GPT | ☐ |
| `GEMINI_API_KEY` | Gemini | ☐ |
| `DEEPSEEK_API_KEY` | DeepSeek | ☐ |
| `DASHSCOPE_API_KEY` | Qwen | ☐ |
| `MOONSHOT_API_KEY` | Kimi | ☐ |
| `ZHIPUAI_API_KEY` | GLM | ☐ |

变量名不固定——每个模型条目的 `api_key_env:` 指向谁就读谁，随便改。

### 2.2 端点与模型名

配置里注释了常见端点，但**版本会变，务必自己核对**：

| 家族 | base_url | 要填的模型名 | ☐ |
|---|---|---|---|
| Claude | 网关或官方 | 三档版本号已填，确认网关支持 | ☐ |
| GPT | `https://api.openai.com/v1` | `<gpt-model>` `<gpt-mini-model>` | ☐ |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `<gemini-model>` `<gemini-flash>` | ☐ |
| DeepSeek | `https://api.deepseek.com/v1` | `<deepseek-model>` | ☐ |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `<qwen-model>` | ☐ |
| Kimi | `https://api.moonshot.cn/v1` | `<kimi-model>` | ☐ |
| GLM | `https://open.bigmodel.cn/api/paas/v4` | `<glm-model>` | ☐ |

**三个坑：**

- 部分较新的 OpenAI 推理模型要 `max_completion_tokens` 而不是 `max_tokens`，那种条目上加 `token_param: max_completion_tokens`
- **`family` 推错了，两个泄漏检验就废了。** 它靠模型名子串推断，名字太怪就在条目上显式写 `family: xxx`
- 面板要**刻意包含弱模型**。没有能力梯度估不出区分度，`dimension test` 也要 ≥ 8 个 solver 才可解读

---

## 3. 要你拍板的决定

| 决定 | 说明 | ☐ |
|---|---|---|
| **成本上限** | 配对模式每题约 37 次调用（11 人面板），跑 400 对 ≈ 15000 次。开关已实现：在配置里设 `loop.max_api_calls`，超限干净退出。**填个数字给我** | ☐ |
| **Python 环境** | 系统默认 `python` 是 3.9，低于 `pyproject.toml` 要求的 3.10；conda base 没装 `anthropic`。你想用哪个环境正式跑？ | ☐ |
| **Pilot 领域** | 先跑哪个学科？建议选你能找到人工审核专家的领域 | ☐ |
| **重复作答次数** | `loop.solve_replicates`（默认 1）。设为 2–3 才能分辨“难”与“随机”，代价是面板调用翻倍 | ☐ |

---

## 4. 只有你能做的判断（Pilot 结果出来之后）

我会把命令跑完、把数字整理出来，但**这四个判断是你的**：

| 看什么 | 数字在哪 | 决定 |
|---|---|---|
| 知识产能够不够 | `coverage` 的 Chao1 与饱和度 | 一个领域产不出 300+ 不重复概念 → 要不要继续，还是先做「源文档接地」 |
| **鲁棒性是不是独立维度** | `misinfo` 的 `dimension test` rho | **相关弱 = 项目成立**。这是整个项目的生死判断 |
| 答案键可信度 | `report` 的 `key error rate >= X%` | 下界多高你能接受？超过就要改出题或验证流程 |
| 泄漏是否需要处理 | `leakage` 的 POOLED 行 | 出现 B/C → 要不要把作者家族排除出面板（会损失数据）、FRAGIL 要不要分层报告 |

第 2 项是核心：**它和题目难不难完全无关**。

---

## 5. 人工专家审计（只能你安排）

自动流程给出的答案键错误率是**下界**——两个验证者一起犯的错它看不见。EVALUATION.md 主张 2 要求的分层抽样 100–200 题、2–3 名领域专家、报告 Cohen's κ，这部分没法自动化。

☐ 决定要不要做、找谁做、什么时候做

---

## 已经不需要你操心的

冒烟验证、跑 pilot 命令、写分析代码、改文档——都在我这边。

代码侧已经清零：CHANGES 状态表里 22 项问题中 18 项已修复，剩下 4 项（跨家族梯度、自我偏好泄漏、注入者混杂、OpenAI 后端真实连通）**全部只差凭据**。给了就能往下推。
