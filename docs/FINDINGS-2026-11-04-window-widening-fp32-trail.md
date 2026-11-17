# 放大训练窗口排查记录(2026-11-04 ~ 11-11):六次 OOM,三个真 bug,一个还在查

> 背景:11-03 的分析([docs/ANALYSIS-2026-11-03-why-27b-dpo-degrades.md](ANALYSIS-2026-11-03-why-27b-dpo-degrades.md))
> 确认 assistant-only mask 是硬修复(v4mask 把 solved rollout 从 14 拉回到 17,与 base 的 18 打平,
> degenerate 58%→46%)。但有一类失败(`oss-fuzz:370689421`,所有 DPO 版本都是写了 PoC、提交了、
> 就是不崩)mask 修不了,根因是训练样本的裁剪窗口(`_CTX_TAIL`)只留最后约 6 轮,模型探索定位漏洞
> 的过程不在样本里。这份文档记放大窗口这条线踩的坑,截至写这份文档时**仍未跑通**。

## 时间线

| 作业 | 配置 | 结果 | 根因 |
|---|---|---|---|
| 1163831/1164327(冒烟) | 6144(v4mask 原窗口)+ 稀疏 logits | 通过 | — |
| `w32k` 1165260 | max_len 16384 / tail 32000 | 第 7 步 backward OOM(76.5G 满,再要 4.78G) | 序列太长,激活线性增长 |
| `w22k` 1165383 | max_len 12288 / tail 22000 | 首步 30+ 分钟卡死,NCCL 超时整体被杀 | 节点被共享邻居挤爆主机内存(个案,非配置问题) |
| `w22k` 1170330(重跑,换节点) | 同上 | 第 26 步 OOM(13.08 GiB) | accelerate `ConvertOutputsToFp32` 把整份 logits `.float()`(词表 248320,不是之前估的 15 万) |
| `w20k` 1170780 | max_len 11264 / tail 20000 | 第 1 步 backward OOM(13.08 GiB,同一个数字) | 绕过 fp32 转换的尝试没生效(见下) |
| `w16k` 1171737 | max_len 9216 / tail 16000 | 第 26 步 OOM | `SPARSE_DIAG` 确认 `logits dtype=torch.float32`——lm_head 输出一直是 fp32 |
| 冒烟 `smoke_sparse4` 1173613 + `w20k` 1173614 | 加 `torch.autocast(bf16)` 包 forward | 冒烟"通过"但诊断仍显示 fp32;正式训练仍在第 1 步同一数字 OOM | autocast 包了个空壳(见下) |

## 根因链(从表象到底层)

1. **显存账本一直对不上**:4 卡 FSDP、assistant-only mask 下稀疏 logits,6144 窗口能稳定跑在 45～53 GiB,
   但拉长窗口后 OOM 发生的时点和数字每次都不一样,说明有一笔"跟窗口长度线性相关但不该在场"的显存开销。
2. **第一层真相(11-05 续 2)**:accelerate 的 `ConvertOutputsToFp32` 包装器把模型输出的整份 logits 升到
   fp32 再返回给调用方——这是 accelerate 为 fp16 训练设计的保护(fp16 数值不稳定,算 loss 前转回 fp32),
   但不管 mixed_precision 配的是 fp16 还是 bf16 都会触发。真实词表是 **248320**(此前一直按 15 万估,
   不对),2 行 × keep(约 9000)× 248320 × 4 字节 = 18.3 GiB,一次性申请,这是当时那次 OOM 的直接原因。
3. **尝试绕过(没生效,11-05 续 5 → 11-11 发现)**:原计划是拿到 "被 fp32 转换包裹之前" 的原始 forward
   直接调,再在外面套 `torch.autocast(bf16)`,让 lm_head 输出 bf16(减半)。实现时假设
   `model.forward.__func__.model_forward` 这条路径总能拿到那层——**这个假设只在 accelerate 内部走
   `MethodType` 包装那条分支时成立**。读 `accelerate/accelerator.py` 源码确认 native_amp 包装
   `model.forward` 实际有两条分支:(a) `model.forward` 本就不是 bound method 时直接赋值整个
   `ConvertOutputsToFp32(...)` 实例给 `model.forward`,这个结果没有 `__func__` 属性;(b) 是 bound
   method 时才走 `MethodType` 包装。原来的代码只处理了 (b),走 (a) 分支时 `getattr(f, "__func__",
   None)` 直接是 `None`,fallback 回 `f(**kw)`——**仍然调用了被 fp32 转换包裹的那层**,外面套的
   autocast 纯粹是空摆设(fp32 转换发生在 autocast 退出之后)。11-05 续 4/5 和 11-11 的 `w20k` 两次
   OOM 在同一个数字(13.08 GiB)上,就是因为这个绕过逻辑从头到尾没真正生效过。
4. **修法(11-11,未经 GPU 验证)**:不再猜走的是哪条分支,直接加运行时诊断(`RAWFWD_DIAG`,打印
   `model.forward` 的类型、有没有 `__func__`、两条取值路径各自拿到了什么);并换一个更可靠的兜底——
   accelerate 在两条分支分叉之前,会先执行 `model._original_forward = model.forward`,这是**两条分支
   共享的、完全未包装的原始 forward**,理论上不管走哪条分支都该有效。诊断冒烟
   `smoke_rawfwd_diag_2026_11_11`(1197183)已提交,写这份文档时仍在排队,结果未知。

## 还没定论的

- `_original_forward` 这条兜底本身没有在 accelerate 1.15.0 源码里反复确认过是否在 FSDP 场景下也会被设置
  (目前只确认了它在 native_amp 包装前被无条件赋值一次,FSDP wrap 发生在这段代码的前面还是后面没有
  完整走过一遍调用链)。诊断冒烟的结果会直接回答这个问题。
- 这台集群没装 `flash-linear-attention`,Qwen3.5 的 Gated DeltaNet 走 PyTorch 参考实现,长序列下
  本身就慢、占显存更多,窗口放大到 32k 这件事本身代价会比短窗口大得多,即便 fp32 bug 修好,也要留意
  这部分开销。
- 主机内存基线 700 GB(4 rank 各自按 fp32 把整模型读进主机内存再分片上卡)是另一个独立发现,解释了
  "同样 22k 窗口,一次在 g02 卡死、一次在 g58 顺利跑完 3 步"的差异——是节点共享邻居挤爆内存的偶发
  因素,不是配置 bug,记在案但没有计划修(真正干净的修法是 FSDP2,是另一次独立改配置)。

## 下一步

1. 等 `smoke_rawfwd_diag_2026_11_11`(1197183)结果,确认 `_original_forward` 兜底生效、logits 真的
   变成 bf16 后,重新从 w16k(9216/16000)这一档往上测,而不是直接冲 32k。
2. 如果这条兜底也不生效,备选是放弃"绕过 accelerate 包装"的思路,改成在 `_sparse_compute_loss` 里
   拿到 fp32 logits 后立刻 `.to(torch.bfloat16)` 再传进 `_chunked_logps`(牺牲一次已经发生的内存峰值,
   但至少不会在反向时再放大)——这是退路,不是首选,首选仍是从源头避免产出 fp32 logits。
3. 窗口放大跑通后,用它重训一版(v4mask 同数据,只变窗口),与 v4mask 的 solved rollout 17/26 题 6
   做对照,重点看 `oss-fuzz:370689421` 这道此前所有版本都是 0/5 的题是否松动。
