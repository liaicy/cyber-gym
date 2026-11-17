# 七项代码修复复审

2026-10-08，Asia/Shanghai。基于当前源码、现有数据、离线测试及 mock 验证；没有启动 API rollout、Docker、训练或远端判分。功能代码未修改。对应证据与代码 SHA256 见 [复审审计快照](REVIEW-2026-10-08-fixes-audit.json)。

结论：**P0 的本地/远端读回错位已修复，新标准 rollout 的 infra 分流与导出过滤已接通，路径越界和提交后环境状态污染已堵住。在线 GRPO 的 invalid reward、判分响应分类、上下文合计预算及完整超时预算仍未闭合，旧数据需要另行迁移。**

## 七项修复的实际覆盖

| 用户列出的修复 | 复审结果 | 范围与限制 |
|---|---|---|
| 1. fix 从远端读回 | 主要故障已修复 | trigger/query 使用同一 server，按 poc_id 匹配；查询失败的原因仍统一折叠为 None |
| 2. 新增 infra_error | 新标准路径生效 | 缺 submit/bash、提交超时/坏响应已分流，agent 写 stop_reason=error；缺 exit_code 仍误判 |
| 3. GRPO infra reward 调整 | 仍有训练信号问题 | 从0改为1避免“低于最差档”，但没有排除 invalid，仍污染组内 advantage |
| 4. exporter 加 is_scorable | 新 error 过滤生效 | 显式 stop_reason=error 会被排除；schema不直接查tier，旧误标数据不会自动迁移 |
| 5. GRPO submit 后 done 门 | 环境状态冻结生效 | 后续工具不再执行；trainer generation/token/loss 的停止仍未通过该接口实现 |
| 6. reset/reward 单独超时 | 改善但未完全对齐 | reset300/reward650仍没有覆盖下游完整的顺序流程及处理余量 |
| 7. 路径和超长观测 | 路径已修，上下文部分修复 | 单条50k观测能裁剪；消息对合计超限与超长task仍有反例 |

## 已验证的进展

- [env.py:453](/data/projects/exam-society/traj/env.py:453) 已使用 POST `/query-poc`，不再猜测本地 SQLite 路径。
- 上海时间17:45左右的新池快照有220条记录，25条带corrected标记：14 solved、11 unclear；`unclear + fix=None` 为0。14条solved的记录内判据全部符合 vul不属于0/300、fix=0。这里验证的是现有文件的一致性，远端原始返回本轮没有独立请求。
- 临时task目录验证：缺submit → `infra_error/error/不可评分`；有submit但没写PoC → `degenerate/step_budget/可评分`。[agent.py:217](/data/projects/exam-society/traj/agent.py:217) 的接线正确。
- [export_dpo.py:110](/data/projects/exam-society/traj/export_dpo.py:110)、[export_grpo.py:73](/data/projects/exam-society/traj/export_grpo.py:73) 的过滤位于实际候选收集路径，确实排除stop_reason=error。
- [env.py:203](/data/projects/exam-society/traj/env.py:203) 已按真实祖先关系拒绝路径越界。
- GRPO mock确认submit后write_poc被拦截，backend只收到submit；不会再修改已提交episode的PoC。
- 本轮重新运行 `tests/test_traj.py`，7/7通过。它们依旧只覆盖偏好排序与配对，因此不能代替下列边界验证。

## P1：infra reward=1并不中性

[grpo_env.py:48](/data/projects/exam-society/traj/grpo_env.py:48) 将infra_error设成degenerate的1分；bridge/client仍只传float，没有validity信息。

本地例子：三个有效结果都是no_crash=2，第四条是judge错误=1。用population std归一化：

```text
reward    = [2, 2, 2, 1]
advantage = [0.577, 0.577, 0.577, -1.732]
```

三个有效结果原本没有差异，基础设施错误制造了正负训练信号。online trainer若用sample std，数值会变化，这个问题仍成立。未取得GPU trainer代码，不能排除它另有外部mask；但当前接口没有提供validity，不能声称已经消除了infra奖励污染。

建议把结果有效性传到trainer，**先用valid rollout计算组内mean/std，再屏蔽invalid rollout的训练loss**；只在最后把invalid自己的advantage设0仍不足以消除其他样本的污染。有效组太小或无方差时应按明确规则跳过/重采样。未知tier也不应静默获得0分。

## P1：判分字段与查询失败仍会变成可训练标签

Mock验证：[env.py:392](/data/projects/exam-society/traj/env.py:392) 对有poc_id、没有exit_code的响应，仍判`no_crash`；非整数exit_code则可抛ValueError。应严格校验字段、类型与允许的状态，非法响应归invalid/infra，并保留轨迹。

[_trigger_fix_verification:443](/data/projects/exam-society/traj/env.py:443) 没有检查HTTP status；query返回503、错poc或缺fix字段时，[_fix_exit_code:479](/data/projects/exam-society/traj/env.py:479) 都返回None，随后变成可评分的`crashed_unclear_fix`。vul崩溃证据仍有效，但这不足以说明fix已经完成或最终裁决有效。

建议结果同时包含query status/reason，区分查询失败、pending、fix合法非零、最终solved。trigger失败但query已返回合法缓存结果时，可采信缓存结果；不能简单丢掉已有证据。对于仅有vul信号的轨迹，若继续用于部分奖励，须另设明确的辅助实验与来源标记。

## P1：新过滤器不清理旧数据，schema也没有独立防线

[schemas.py:81](/data/projects/exam-society/traj/schemas.py:81) 仍只看stop_reason，以下记录返回True：

```python
Trajectory(task_id="audit", model_id="m",
           tier="infra_error", stop_reason="step_budget").is_scorable()
```

新标准agent会把infra写成error，因此标准路径受保护；schema注释所称“直接排除infra”还没有成立。建议明确tier和stop_reason两项不变量，拒绝显式infra/未知状态。

更实际的是旧记录：

- [arvo-58428](/data/traj_batch/pilot_newpool/out/arvo-58428__qwen3-8b__r1.json:307)
- [oss-fuzz-42538001](/data/traj_batch/pilot_newpool/out/oss-fuzz-42538001__gpt-strong__r1.json:167)

两条detail明确是`unparseable server response`，但tier仍为degenerate、stop_reason仍为step_budget，新is_scorable都接受。

旧DPO源轨迹 `/data/traj_batch/dpo_v2_merged/out_arvo-10341__qwen3-8b__r2.json` 也有同样误标，并作为rejected出现在[已有训练文件:30](/data/traj_batch/dpo_v2/train.jsonl:30)起的5对中。原有254对不会因为exporter源码修改就自动重导出。旧混合detail“no submit.sh or no poc”则不足以可靠重判；不要靠字符串批量猜测无法确认的历史状态。

建议版本化迁移明确可判的旧infra，保留原记录与证据；对无法区分的设unknown/待审计。完成迁移后重导出到新版本目录，复核pair provenance、split、计数与hash。SFT阈值确实排除显式infra，但不能由此断言它排除了所有stop_reason=error或旧误标数据。

## P2：上下文裁剪仍没有合计硬上限

[agent.py:108](/data/projects/exam-society/traj/agent.py:108) 按单条消息超过16,000字符才截断，没有让最近一对消息的**合计**满足tail预算。使用100字符system和100字符task，实际结果为：

| assistant / observation字符数 | 返回总字符数 | 相对于名义17,700预算 |
|---|---:|---|
| 100 / 50,000 | 16,385 | 原始单条反例已修复 |
| 8,000 / 10,000 | 18,250 | 仍超限，两条单独都不超16,000 |
| 20,000 / 20,000 | 32,320 | 每条分别截16,000后，合计仍超限 |

另外50,000字符的task本身也不会被head预算限制。这些是字符预算反例，不是在未测tokenizer的情况下宣称实际必定超过模型context length。

建议按system/task占用计算剩余预算，对消息对合计分配预算，并将截断提示计入；最终检查总token数。优先截断工具观测，避免破坏assistant JSON。GRPO native工具路径不走这个JSON `_fit_messages`，它的工具输出与context预算须在实际trainer单独检查。

## P2：超时需要覆盖完整操作而不只是最长子调用

mock确认client确实分别使用reset300s、reward650s、普通工具120s，这个改动生效。但reward服务内部可先完成接近600s的submit，再执行120s fix trigger和30s query；650s仍可能早于服务端结束。reset300s也与gen_task子进程300s上限相等，尚无额外解压和HTTP处理余量。

建议由共同的wall-time deadline约束各阶段，将client超时设在server完整deadline加余量之外，或改成job/status轮询。不应将各库read timeout误当严格的全流程wall-time上限。session cleanup/TTL与reward幂等也仍需补齐，单纯增timeout不解决这些问题。

## 后续次序与记录要求

优先补GRPO valid mask、判分响应分类与旧数据迁移，然后完成context和timeout边界；之后进入GPU端loss mask、模板、截断和adapter审计。submit guard已保护环境状态，但不能通过字符串返回值强制结束trainer的generation，需核对trainer停止规则。

回填脚本已救回14个标签，但覆盖原detail后丢失agent_id，未保存server、查询时间、原标签和原始返回；14条corrected solved的stop_reason仍为step_budget。建议保留修订前后的结构化provenance、按约定同步语义字段，并使用原子写入。这个问题目前不阻止评分，但影响停止原因统计和远端结果复核。

七项既有测试可继续保留；本轮反例应沉淀为持久回归，覆盖缺字段/503、组内invalid mask、旧标签迁移、合计context预算和超时契约。新的代码检查不能追溯性证明历史训练数据已干净，修复后也还没有新的训练效果结论。
