# traj 项目第二意见评审：证据、问题与下一步

七项代码修复后的当前状态见 [修复复审](REVIEW-2026-10-08-fixes.md)。本文保留初次评审时的代码与数据结论；已修复项以复审记录为准。

评审日期：2026-10-08，时间使用 Asia/Shanghai。重点材料为 `STATUS-2026-10-08-for-review.md`；交叉核对了 README、CLAUDE.md、PLAN、SETUP、SFT 负面记录、traj 代码、本机实验产物和 CyberGym/TRL 官方资料。项目目录及父目录未发现 AGENTS.md。本目录没有 Git 元数据。

本轮仅新增评审文件，未改功能代码、未启动模型 API、训练、Docker 或远端判分，未干预正在运行的采样。结构化核对结果与关键输入 SHA256 见 [审计快照](REVIEW-2026-10-08-audit.json)。GPU 侧实际 trainer、完整超参数与在线 GRPO 日志未在本机找到，因此实际 loss mask、截断规则、KL 系数、训练 seed 和“150 步、38 轮中 2 轮有方差”未独立验证。

## 1. 总体判断

当前活跃目标是：在 CyberGym 的真实漏洞复现任务上，通过环境验证的轨迹增强 Qwen3-8B，并控制解题成本。`examsociety/` 已封存，不应继续用 README 的出题社会目标解释当前工作。

这条路线具备继续做研究的基础：有真实环境、目标尺寸模型存在非零成功率、有完整轨迹、愿意保留并复查负面结果。但目前还没有证据证明 SFT、DPO 或 GRPO 带来了可靠的能力提升。优先级应为：**修正标签链路 → 审计实际训练目标和交互协议 → 冻结评测 → 最小因果对照 → 有依据地扩量**。“数据太少”合理，但不是目前已确认的首要根因。

## 2. 做得好的地方及其研究价值

- **环境裁判替代模型自评。** 独立的 vulnerable/fixed 运行结果使结果可外部核验。代码仅在 fix=0 时接受 solved，没有把未知 fix 直接算成功。它提高标签可信度，但标签传输、超时和异常退出仍需审计。
- **能主动否定早期正面结论。** 从 teacher-SFT 坍缩到更严格留出评测，保留原始轨迹和负面结果，有助于发现评价与训练目标之间的偏差。当前 STATUS 对“扩数据”明确写为假设，这个习惯应保留。
- **并发隔离与来源记录有基础。** batch rollout 使用独立目录和新 agent_id；保存逐步 action、observation、raw_output、usage、退出码与停止原因，便于重判和行为诊断。DPO v2 实际配对全部来自同一 qwen3-8b model_id，降低了跨模型表面风格混杂。
- **已意识到训练信号稀疏和 slice 差异。** 做了未训练目标模型、14B 和强模型对照，区分近分布训练题与留出题，并尝试 level2 信息干预。下一步应强化这些诊断，而不是只增加 trainer 步数。
- **GRPO 复用同一环境实现。** 比复制一套判分代码更易保证一致性；adapter 也考虑到了 pooled environment 的重置问题。

外部核对：CyberGym 官方论文确有 1,507 个任务、188 个项目，报告 OpenHands+GPT-5 high reasoning 在 300 题子集约 22% 成功率；官方 10 题 demo 明确是 5 个可解、5 个较难任务，不是随机代表样本。[论文](https://arxiv.org/html/2506.02548v3)、[官方仓库](https://github.com/sunblaze-ucb/cybergym)。这些公开结果支持“任务很难、demo 有选择性”，不能直接校准本项目不同工具与预算下的成功率。

## 3. P0：本地未取得新远端判分的最终结果，先处理这一项

**已验证事实。** 2026-10-08 16:40:26 的本地快照有 186 个可读新批轨迹文件，标为 solved 的数量为 0；其中 16 个 `crashed_unclear_fix` 全部 `fix_exit_code=null`，全部提示 fix 结果未出现在数据库，对应 poc_id 在本地数据库中命中 0/16。这个数字来自尚在运行的采样，不是最终实验结果。

**代码证据。** [env.py:425](/data/projects/exam-society/traj/env.py:425) 仍从本地 SQLite 获取 fix；STATUS 说明新判分已通过隧道移到另一台机器。`_trigger_fix_verification()` 对异常静默处理，也没有检查 HTTP status。这足以确认“本地标签端没有拿到最终判分”，与远端架构不相容是最直接的解释。

**尚未验证。** 未查询远端，不能判断 fix 是否运行成功，更不能判断这 16 条中实际有多少 solved。当前 0 solved 不应被解释为模型能力或新池难度。

**建议。** 从同一个 judge server 查询最终结果，检查 HTTP status，按 agent_id/task_id 查询，再用 poc_id 精确匹配。当前本机 CyberGym [server:198](/root/cybergym-src/src/cybergym/server/__main__.py:198) 提供 POST `/query-poc`。记录 judge 地址标识、版本、agent_id、task_id、poc_id、PoC hash、原始 verdict；对已有轨迹回填标签，并保留修改前后的来源链。缺结果应是 `pending_verification/infra_error`，不能作为一个有序的能力档位直接训练。在重判前，保留探索数据，但不要依据当前 tier 选择可达池或导出新训练集。

## 4. 最新评测：点估计下降成立，机制与方法排名尚未成立

重算 `/data/traj_batch/heldout_eval_v2/out/` 的 78 条记录：

| 模型 | solved / 完成 rollout | pooled solve | 按任务等权 macro solve | 有至少一次 solved 的任务 |
|---|---:|---:|---:|---:|
| 未做本任务适配的 base | 7/20 | 35.0% | 35.0% | 3/4 |
| 自蒸馏 SFT | 5/20 | 25.0% | 25.0% | 2/4 |
| DPO v2 | 4/19 | 21.1% | 20.0% | 1/4 |
| GRPO | 2/19 | 10.5% | 10.0% | 1/4 |

STATUS 总体表正确。arvo:21316 的 DPO 和 GRPO 都是 **0/4**，不是 0/5：缺失 unit 是上游连接重置，见 [batch.log:102](/data/traj_batch/heldout_eval_v2/batch.log:102)。应补齐或明确 missing 规则；如果评价完整运行服务的预算成功率，计作失败时总体是 4/20、2/20，但与模型条件成功率要分开报告。

暂按独立 rollout 算出的描述性 Wilson 95% 区间分别是 18.1–56.7%、11.2–46.9%、8.5–43.3%、2.9–31.4%。这些不是跨任务泛化区间：泛化只有 **4 个 task clusters**。同题多跑主要提高该题成功概率的估计精度，不能代替增加不同任务。

arvo:21316 的 4/5 对 0/5 或 0/4，nominal 双侧 Fisher p≈0.0476；任务为事后突出展示，三个模型共用 base，多重比较后不能称稳定显著。它值得优先诊断，不能单凭这一题确认灾难性遗忘。最新数据中 GRPO 的空 action 比例约 4.7%，base 约 8.4%，也不支持直接把 GRPO 最差归因于格式全面崩坏。

推荐表述：**“本次固定四题 slice 上，三个训练产物点估计均低于 base，存在值得复查的能力下降；样本不足以可靠排名方法，下降机制未确定。”**

## 5. 数据规模、切分和“均匀可达”需要重新核准

**DPO v2 实际数据。** [train.jsonl](/data/traj_batch/dpo_v2/train.jsonl:1) 有 254 对、44 个有 pair 的任务，非 56。源池任务数与有效训练任务数被混淆的解释合理：[export_dpo.py:157](/data/projects/exam-society/traj/export_dpo.py:157) 统计的是输入 trajectory 所属任务，不是实际写出 pair 的任务。

| 偏好类型 | pair 数 | 比例 | 能证明什么 |
|---|---:|---:|---|
| no_crash > degenerate | 184 | 72.4% | 多数监督在区分形成尝试与未形成有效尝试 |
| solved > no_crash/degenerate | 69 | 27.2% | 存在经环境确认的成功对照 |
| crashed_unclear_fix > degenerate | 1 | 0.4% | fix 未确认，不应视为严格成功 |

44 个任务只有 12 个提供 solved chosen。254 对共使用 261 条不同轨迹，其中 34 条 solved；pair 共享轨迹，不能把 254 当独立成功样本数。当前自蒸馏 SFT 的 17 条是 17 条不同 trajectory、7 题、163 个 assistant 轮次，也不能与 254 对或 150 trainer steps 直接比较数据量。

**切分。** 最新四个 heldout 未出现在本地已查 SFT/DPO 训练文件中，这是正面结果。但 DPO 的 44 题有 28 题不在 `task_splits.json` 声明的 train23 中，说明 manifest 已滞后，尚不能保证所有方法用同一训练范围。SFT 原始 v1/v2 的 train.jsonl 实际含旧记录称 heldout 的 10628/11007，这与针对这两题的旧 SFT heldout 声明冲突；若 GPU trainer 曾额外过滤，需要提供相应记录，当前不能将这两题作为已验证的干净 SFT 留出。这不等于最新四题发生泄漏，也不说明旧 22 题评测受这两题直接泄漏影响。

**旧 22 题不应称已验证的均匀随机样本。** [fetch_heldout.py:73](/data/traj_batch/fetch_heldout.py:73) 从按 ID 排序的候选列表顺序挑镜像可用任务；ready 集是前面的 22 题，未找到随机采样与 seed。它是一个受可用性影响的留出集。旧 base 是 1/110、DPO 2/110；DeepSeek 3/51 只完成 17/22 题，GPT 只完成 10/22 题。DeepSeek 的 48/51 失败均为 40 步耗尽、没写 PoC。低成功率说明的是**该 harness、信息级别与预算下的结果**，不能推出“谁都解不了”“不是训练方法失败”。ID 相邻也不是分布接近的充分证据，应看项目、漏洞类型、代码相似性和输入格式。

**可达筛选。** 未进入训练只能支持“train-clean”，不自动支持代表性。根据 probe 成功再挑题会得到条件分布；这种 reachable slice 可以有价值，但须预先固定规则，报告目标模型、harness、difficulty、预算、K，并用新的独立 rollout 评价。新批每 task 总 K≈7 是 qwen3-8b×3、14B×2、GPT×2，不是目标 8B 的 K=7。

若单次真实成功概率 p=10%，K=13 仍有 25.4% 的机会零命中，K=20 仍有 12.2%；0/13、0/20 的单侧 95% p 上界仍为 20.6%、13.9%。因此应称“在此探测预算下未观测到成功”，保留不确定任务，不硬标不可达。抽 100 个新任务也不等于增加 100 个成功可达训练任务。

## 6. SFT/DPO/GRPO 的结论和实际目标需要对齐

**SFT：坍缩行为有证据，单一根因没有被识别。** 重复动作、忽视反馈、两个训练版本失败是观测事实；分布不匹配是合理机制。离线 SFT 在 teacher 状态上训练，未见状态的误差累积主要出现在模型自己交互时，“第一步就分叉”不能单独证明学习到了动作边际。[DAgger 原论文](https://arxiv.org/abs/1011.0686) 支持一般的 sequential covariate shift 问题，不证明本项目的具体根因。两组较接近的 LR/epoch 设置，也不足以排除更新过强、mask、context、数据选择等实现因素。当前自蒸馏也下降，表明“teacher 身份”不能作为唯一解释。建议把 FINDINGS 中“SFT 目标本身的问题”“不是超参数”“DPO 方向成立”改成与证据相称的阶段性判断。

**DPO：当前导出是 trajectory-level pairwise，未证明实现了 list-wise loss。** 保存 ranked list 是支持未来 list-wise 的数据结构；实际 exporter 写 chosen/rejected。DPO 仍用两端 token log-probabilities 更新策略，不会因为采用偏好标签就自动免除 off-policy 状态问题或风格拟合。[DPO 原论文](https://arxiv.org/abs/2305.18290)。

需要从实际 GPU trainer 逐项审计：environment observation 是否全部 loss-masked；每条轨迹的 likelihood 是否只累计 assistant action；训练 history/template/thinking 与推理是否一致；截断后是否保留决定成败的动作；长轨迹在 loss 中如何归一化。还应核对实际 serving 的 adapter/模型 hash，并检查零更新 LoRA 的保存、加载与原始 base 是否等价，以排除加载或模板变化带来的下降。原始 continuation 中 137/254 chosen、169/254 rejected 的字符数超过推理约 17,500 字符预算，这只是优先审计的理由，**不是已经证明被截断或 mask 错误**。

另外，[rank_score:58](/data/projects/exam-society/traj/preference.py:58) 在失败档内奖励非空 action 数，重复 grep、无效工具名也能加分，不等于有效推进。实际 DPO exporter 不调用 `is_scorable()`，已有 10 对 rejected 的 stop_reason=error，可能把 API/基础设施故障当负例。主实验建议使用严格 solved-vs-valid-failed；no_crash-vs-degenerate 单独当行为辅助实验或消融，不混成相同强度的漏洞能力监督。

**GRPO：稀疏 advantage 能解释缺少学习，尚不能单独解释明显退化。** 在线 reward 为四档 4/3/2/1，非零方差可以只来自写 PoC 与没写 PoC，甚至判分基础设施差异，不自动意味着组内有真实 solved。相同 reward 的组在标准归一化下 advantage=0，但“其余全靠 KL”需实际 beta、loss 与梯度日志确认。KL 本身约束向 reference 靠近；优化器动量等也会影响更新，不能把 reward 稀疏直接当遗忘的充分原因。[TRL 官方文档](https://huggingface.co/docs/trl/grpo_trainer) 确认当前支持 environment_factory，也明确 beta 决定是否使用 KL；本机未验证训练端具体版本与参数。

二元终局 reward 下，同题 K 条独立采样出现成功/失败对照的概率为 `1-p^K-(1-p)^K`。p=1%、K=4 时仅 3.94%，p=5% 时 18.55%，p=20% 时 58.88%。所以增加任务数和增加有效组比例是不同的轴；优先用有适中成功概率的**训练题**、受预算限制的 adaptive K/重采样。记录 solved 对照组比例、valid verdict 比例、advantage 分布、policy/KL loss、entropy、completion 截断率和每有效组成本，而不只看 trainer steps。对筛选/跳过组的规则也做记录，以说明优化的是哪个任务分布。

## 7. 工程与实验协议：几个具体可复现的问题

| 优先级 | 事实与位置 | 建议 |
|---|---|---|
| P1 | verify 把无 PoC、缺 Bash、提交超时、坏响应都归 degenerate；缺 exit_code 的 JSON 被判 no_crash，[env.py:332](/data/projects/exam-society/traj/env.py:332) | 拆开行为失败、infra_error、invalid_verdict、pending；后几类重试或 mask |
| P1 | GRPO `_step()` 只返回 observation，丢弃 done；submit 后仍能调用工具，[grpo_env.py:130](/data/projects/exam-society/traj/grpo_env.py:130) | 实现明确 terminal contract，核查 trainer 停止语义 |
| P1 | GRPO native tools 与评测 JSON action 不同，训练/评测 token/tool 预算也不同 | 同一 checkpoint 在一致协议下评测；另做协议交叉消融，不能先假定已解释负面结果 |
| P2 | 路径用字符串 startswith 判断祖先，`../_repo_sibling/` 可越界，[env.py:191](/data/projects/exam-society/traj/env.py:191) | 用实际路径祖先判断；添加边界回归 |
| P2 | 最新单条 observation 超大时 `_fit_messages` 仍完整保留；50,000 字符输入可突破窗口，[agent.py:100](/data/projects/exam-society/traj/agent.py:100) | tokenizer 限制总上下文和工具输出，并记录截断 |
| P2 | HTTP client timeout=120s，而 task generation/submit 最长300/600s；失败 session 缺 TTL，[grpo_env_client.py:25](/data/projects/exam-society/traj/grpo_env_client.py:25) | 对齐超时层级、回收故障 session、保证查询 reward 幂等 |
| P2 | 桥接只返回 float reward 并删除工作目录，未落完整在线轨迹与 verdict | reward 与审计日志分开；失败也能追溯 |
| P2 | README/CLAUDE/PLAN 存在过期方向与环境信息，pyproject 只打包 examsociety，实际 trainer 不在本机 | 当前入口指向 traj；收录 GPU trainer、锁定依赖与完整实验 manifest |

本地低成本验证：Python 3.12.14 的 `cybergym-py312` 环境下 `tests/test_traj.py` 7/7 通过，但这七项只覆盖偏好排序与配对。另用临时文件和 mock 复现了路径越界、context 超限、verifier 缺字段/infra 分类、submit 后继续工具调用。旧 `tests/test_offline.py` 因缺 scipy 停在 import，未运行通过。`/data/venv-cyber` 当前不能作为可用环境；未为审查安装依赖。

文档成功条件还应包含排除 timeout=300。非零退出是 benchmark operational criterion，不能单凭 exit code 保证 sanitizer 确认了描述中的漏洞；异常退出与潜在其他漏洞应保留 stderr/signature 审核。严格 solved 与环境审计后的部分信号分开报告。

## 8. 对 STATUS 五个问题的直接建议

1. **扩大多样性值得测试，但排在判分与目标审计之后。** 用同一种算法、同样的有效 action token / update 预算和标签构成，比较少题多采样与多题少采样；另比较当前 pairs 与 solved-only pairs。否则多样性、成功监督比例、训练强度同时变化，改善也无法归因。
2. **GRPO 先提升有效组比例。** 课程并非只按易到难；全成功和全失败都没有 binary group advantage，应优先有中等 p 的训练题。增 K 要按单位有效对照组成本决定，而不是固定无限扩采样。先证实这个组内差异来自可核验成功，再扩大训练。
3. **留出应先扩任务数。** 10–15 题可做筛查；更稳定的跨任务结论需要更多独立题、多个训练 seed 和按 task 配对/cluster 的不确定性报告。具体样本量须按最小有意义改善与预算确定，不能宣称某个题数天然足够。原四题现已影响研究决策，宜作为 diagnostic/dev；正式最终 test 另行冻结。筛选 probe 不复用为最终评测样本。
4. **保留能力应做干预，而非默认灾难性遗忘成立。** 先统一协议、评估中间 checkpoints，测每题动作/状态分叉。用更小更新、早停、保持对 reference 的约束、加入 base 擅长的其他训练题做对照。DPO 的 beta 是目标中的 reference-relative 系数，不能与 GRPO 显式 KL coefficient 当相同旋钮；比较实际策略漂移。不要把 arvo:21316 加训练后继续称它 heldout。
5. **可达用概率与预算描述。** 对零命中题保留不确定性，用固定规则分配额外探测预算，并记录模型各自 K。训练筛选可以追求产率；评测要同时保留代表性全体样本与预定义 reachable slice 的分母。

## 9. 最小下一步与论文/应用定位

建议下一轮只集中做一个可解释的小实验：先完成 verdict 重判和 trainer token-mask 审计，再固定开发集，比较 base、当前 DPO、严格 solved-vs-failed DPO。若后者有信号，再在保持训练预算与成功标签比例相近的条件下扩任务多样性；用开发集选 checkpoint，最终在冻结 test 验证。并行跑所有算法的大量超参数会消耗预算，也难定位原因。

“环境裁判”是测量基础，尚不足以单独成为新的方法贡献。更可检验的问题是：**在固定数据生成与推理成本下，哪些轨迹选择和偏好目标能提高小模型的真实多步成功率，同时保留原有能力？** 若最终只能证明 reachable slice 内改善，就清楚限定适用范围；若要跨项目泛化，需 project/CVE family 分组切分，不能只排除 task ID。也可把环境失败诊断与轨迹数据集作为可信交付物，但产品化性能仍须实测。

成本至少分开记 API prompt/completion tokens、GPU-hours、judge 时间与失败重试；报告 task-macro pass@1、固定 K 的任务解决率、每成功任务成本和实际预算分布。与 base 的 best-of-K 或更好提示/检索工具基线比较，才能判断训练是否比直接多采样有价值。

实验 manifest 应包含代码/模型/adapter hash、task 与项目 split、screening 与训练/解码 seeds、difficulty、system prompt 与 chat template、thinking、温度/top-p、步数/token/观测截断、pair 来源与奖励版本、trainer 全配置、missing 处理、逐任务结果与费用。当前审计 JSON 保存了一部分输入 hash，不能替代未来每次实验的完整 manifest。

待明确的成功标准是：优先证明 reachable slice 在同预算下稳定超越 base、代表性全体集的泛化提升，还是相近成功率下显著降低成本。三者需要不同筛选与评测设计。本轮未得到偏好前，建议先把 reachable slice 当开发诊断，同时保留一份预先抽样、未经模型结果筛选的最终 test。
