# 下一批采集与训练建议（2026-10-09）

> 后续确认本轮只采新任务，采集侧修复和可执行配置已完成，见 [采集启动说明](COLLECT-2026-10-09.md)。下文保留修复前审计与训练建议；其中判分/字符预算缺口已有新修复，GPU 训练问题仍未解决。

基于当前源码、本地 GPU 脚本备份、采集日志、原始 JSON 和离线验证。此次没有访问模型 API、远端 judge、GPU 平台或启动任何付费作业。当前配置与数据 hash、100 题计划清单及旧 DPO pair 源记录见 [审计快照](NEXT-BATCH-2026-10-09-audit.json)。用户没有指定本轮直接提交任务；以下配置是待采用的实验建议。

## 启动判断

**可以准备受控续采；不能将当前配置认定为可以直接放量或正式训练。** 新数据的主要问题是目标模型严格成功监督仍很少；训练入口还有确定的输入构造错误。过去训练产物的点估计下降，不能唯一归因于数据规模、SFT/DPO 方法或灾难性遗忘。

| 环节 | 当前事实 | 下一步条件 |
|---|---|---|
| Judge | `env.py` 仍从配置 server 的 `/query-poc` 读 fix，原本地 SQLite 错位没有回退 | 明确采用远端隧道 `http://127.0.0.1:18667`；本轮只确认 listener，没有探活 |
| 采集 | 默认 server 是本机 8666；默认 10 个旧 demo 任务、7 个模型、每题 15 条、25 步、workers=6 | 显式给 task manifest、server、POOL/REPS、预算和全新 OUT/WORK/LOG；不要直接运行默认命令 |
| DPO | 原 254 对仍有 97 对裁剪后两侧 prompt 不同；裁剪后的 chosen 142 条/rejected 173 条均从 user 开始 | 修复奇数长度轨迹配对及共同 prompt；验证真实 tokenizer 与 actor token mask |
| GRPO | validity 已透传，但备份 trainer 未消费；没有 valid-only 组统计/loss mask | 接入实际 trainer 并做混合 valid/invalid 组验证；同时对齐 native-tool 训练/评测协议 |
| 静态数据 | 旧 DPO 254 对中，当前源记录有 15 对不可评分（包括已迁移 infra 对） | 从来源重新构建新版本；源码和标签修改不会自动更新 JSONL/checkpoint |

`gpu_trainer_backup` 是本地备份，本轮没有证明它与当前远端脚本完全一致。下一次正式训练还需在 GPU 上确认实际脚本、依赖版本与模型路径。

## 数据实际规模与能学到的信号

新池 `/data/traj_batch/pilot_newpool/out`：**220 条、34 个不同任务、215 条可评分、14 条成功、10 个成功任务**。计划的 100 题中仍有 66 题没有输出，采集进程当前未运行。14 条成功不是 14 个可达任务。

| 模型 | 轨迹 | 可评分 | 成功轨迹 | 成功任务 |
|---|---:|---:|---:|---:|
| Qwen3-8B | 95 | 93 | 2 | 2 |
| Qwen14B | 66 | 65 | 6 | 4 |
| GPT strong（历史批次） | 59 | 57 | 6 | 5 |

历史 GPT 轨迹保留 provenance；下一批当前 roster 已排除 GPT，不应把两个批次视为同一采样条件。所有现有 11 条 unclear 都有非零 fix，无 `fix=None`；这是文件内判据一致性核查，本轮没有重新获取服务器原始结果。

目标 8B 的成功在 `arvo:27936` r3 和 `arvo:28458` r1。前者只有 r1 的有效 no_crash 可配对，r2 缺失；后者只有一条成功记录。因此当前新池只有 **1 对严格同题、同目标模型的 solved-vs-valid-failed 对比**。

按当前 exporter 规则模拟可导出约 76 对/30 题，但其中仅 1 对涉及 solved；39 对 no_crash>degenerate、29 对同档启发式比较、7 对涉及 unclear。不能将这些 pair 的增加直接解释为漏洞复现能力的监督增加。

旧 DPO 源文件关联核查进一步表明：239 对源记录当前可评分，其中 61 对是 solved>no_crash/degenerate，覆盖 11 题；但 **45 对 degenerate rejected 的 detail 都是歧义的 `no submit.sh or no poc written`**，无法仅凭字符串证明是模型失败。剩下 16 对 solved>no_crash 覆盖 4 题，rejected 的 vul 均为 0。这 16 对可作为更严格的候选核心；仍需去重、任务上限、完整 provenance 与 tokenizer 验证。61 对不能全部叫作“已清洗的严格训练集”。

## 下一批怎样采

1. **先冻结任务角色。** 将 `/tmp/batch100.txt` 的计划清单版本化；原采样 seed 没有查到，不能补写成已知值。旧 4 题只作持续查看的 dev/回归集。新池先划定额外 reachable dev；从尚无输出的任务中按固定 seed（建议 42）预留独立 test，之后不按模型是否成功删除 test 题。reachable slice 与未经结果筛选的整体测试分别报告；核查项目/重复漏洞关系，而不只 task ID。
   划分必须进入导出和 trainer 的实际过滤：当前 `export_dpo.py` 默认只排除旧 `10628/11007`，未读取新的 split；`export_sft.py` 的 hash val split 也不等同于四题及新 dev/test 的强制排除。新导出需显式排除 `20652/17607/5856/21316` 和新增 dev/test，并在实际训练行上断言任务交集为空。不能仅保存 split JSON 后假定代码自动使用。
2. **新配置用独立目录。** roster、backend、thinking、temperature、difficulty 或预算改变，都生成新 run manifest。`run_unit()` 仅按文件存在断点恢复，不能在同一 OUT 下变更条件后宣称完成新配置。旧池保留原样。
3. **先缩小模型组合和并发。** 建议从 Qwen3-8B 加两个非 GPT 教师开始，例如 `glm-5.3,deepseek-flash`；这是减少请求量的工程起点，不是已验证的最优教师选择。初始可设 8B K=3、两个教师各 K=1，总 workers=2；避免重演两个进程各 4 workers 共享 SiliconFlow 配额导致大量 TPM 429。需要按 provider 的实测 TPM/429 再调并发。
4. **优先追加目标模型样本。** split 冻结后，在训练部分中先对“8B 曾成功”和“教师成功、8B 尚未成功”的题追加同一目标策略采样，预算允许时逐题达到实际有效 K=8–12。这些是寻找成功/失败对比的训练课程，不是均匀代表样本；K=3 全失败只能叫“尚未观察到成功”。不要继续在所有题上机械地给五个教师都采 K=2。
5. **统一并记录条件。** 主线保持 level1、JSON-action、目标模型 thinking-off；正式训练数据采样与评测建议使用相同 40 步上限及每轮 4000 output-token 上限。若探测阶段为省成本使用 25 步，单独记录，并在主对比前补齐同预算评测。API seed 若不支持或无法确认就记 null；记录实际请求参数、模型 ID/backend、calls、tokens、有效 K、任务缺失原因和文件 hash。

受控端到端检查至少应覆盖环境判据、动作解析、真实 fix 返回、错误分类和数据落盘；“API 返回 200”不足以证明整个链路成功。每题 5 条×40 步的 100 题计划，在不计重试时的请求上限是 20,000；这是请求数量上限，不是预计实际用量或金额。重试会占预算，因此 call budget 应显式限制，并允许预算不足时分段续采；不能承诺旧 2,000/4,000 次预算能完成全部任务。

查询边界尚需收尾：HTTP 200 但无目标 poc/缺 fix 字段不能当作合法 null；非法 fix 必须归 invalid；trigger 失败需记录。主数据只收有完整证据的最终标签，不能把缺 verdict 作为 rejected。

## 怎样训练：先做一条清晰的主线

**第一轮从未训练 Qwen3-8B base 新建 LoRA，先做 DPO；旧 adapter 保留作对照。** 直接在已经下降的旧 adapter 上追加训练，会混入旧数据、旧准备逻辑和先前更新的影响。旧 7/20、5/20、4/19、2/19 仍是小样本历史点估计；新配置的 base 应用同一 harness 重新测量，不能把 35% 当固定真值。

主 DPO 的 chosen 只取严格 solved；rejected 优先取有有效 vul=0 判据的 no_crash。真正的“模型没有写 PoC”可另保留，但旧歧义 degenerate 要先查证。两侧同题、同目标策略/版本/采样条件，每题最多 4 对，避免单个成功轨迹通过笛卡尔积放大成大量看似独立的监督。教师成功用于寻找训练题和诊断，教师 imitation/跨模型 pair 如要做，单独成实验分支。

DPO 原始目标比较同一个 prompt 下的偏好响应，当前两侧 prefix 错配必须先修复。[DPO 原论文](https://arxiv.org/html/2305.18290v3) 对这里的多轮环境任务，还需明确条件策略只对模型生成的动作累计 log-prob，工具观测用作条件输入；不能未经核对就把整份带 user 观测的 transcript 当普通 completion 计算策略概率。

| 设置 | 建议首轮 |
|---|---|
| 初始化 | 未训练 base + 新 LoRA，reference 为同一冻结 base |
| lr / epoch / beta | **1e-6 / 1 epoch / 0.1** |
| LoRA | r=32，alpha=64，dropout=0.05 |
| batch | per-device=1，gradient accumulation=8 |
| max_length | 8192；先确认显存与保留的完整动作，不直接套默认 12288 |
| seed | 初轮 0；出现可信方向后再重复 1、2 |
| 输出 | 新的 DATA/RUN/checkpoint 路径；保存早期 checkpoint |

这些是降低首轮干预强度的**建议起点，尚未验证最优**。若 8192 tokens 无法容纳完整关键动作，应记录丢弃并调整数据设计/长度；不通过错误切片强行保留。不要将 `--max-prompt-len` 当已生效上限，当前脚本只解析而未使用它。

当前新池的两条 8B 成功/一对 DPO 只足以做准备链路诊断，不能支持扩规模效果判断。先补充不同训练任务上的严格成功对比。建议将至少 20 个这样的任务作为小规模能力实验的数据准备目标；这是预算规划阈值，不是统计充分性或训练有效性的保证。保留经过审计的旧训练任务作 replay，并按任务均衡，而不是让某几个容易题占据多数 pair。

SFT 暂不沿用 teacher SFT 默认 lr=1e-4、3 epoch。目标策略自身成功 SFT 可以作为后续单独的低更新强度对照；当前只有 2 条新成功时不优先投入。历史失败支持谨慎选择，不足以永久否定 SFT 方法。GRPO 先解决 invalid mask、终止规则和协议对照，再决定 K/课程；不将教师轨迹混成目标策略的 GRPO 组。[GRPO 原始方法来源](https://arxiv.org/abs/2402.03300)

## 验证与可解释的比较

先验证准备正确，再测试是否超过 base。固定至少 10–15 个 dev 任务，每题 K=5，按 task 汇总 solve，附每题分子/分母和 missing/infra；额外报告有效提交、循环、步数和 token/call 成本。failed/infra 的处理规则预先固定；同时报告 attempted-denominator 与 valid-denominator，避免删除失败请求后产生选择偏差。旧 `arvo:21316` 作为回归题独立查看，但不由单题 K=5 决定整体结论。

第一轮只对比 base 与一份修正后的主线 DPO。若要说明“数据多样性带来提升”，需要后续将清洗旧数据与旧+新数据在相同准备逻辑、更新/token 预算下比较；一次同时修代码、换 teacher、增加 K、换 difficulty、加数据和改超参，只能证明整套配置表现，不能定位根因。训练 loss 下降和训练题 solve 不能作为泛化提升的成功标准；出现 dev 能力下降时停止扩训并检查任务分布、模板、mask 和实际更新幅度。

本轮现有离线测试 7/7 通过；当前 DPO 反例持久脚本和两个独立审查均仍能复现。没有新训练效果结果。下一步应先补齐采集 manifest 与 trainer 准备正确性，之后才启动有预算上限的小实验。
