# 2026-09-25 pairwise ranking and proxy OOD continuation

本轮沿用已冻结的 TVSum nested source-group protocol；没有读取或调节 comparison_holdout_v1 的 7 条视频。新增的 pairwise objective 只改变 temporal loss：每个训练视频从有效 timestep 有放回抽取 1024 个有序 pair，过滤 human importance gap < 0.1，使用 `BCE + 0.1 * pairwise_logistic`。A0 与 DeiT-S 使用相同 2×5 outer folds、3 seeds、20 epochs 和 inner-dev checkpoint/threshold 选择。每次训练约 12–88 秒，60 个 outer evaluations 全部完成；官方 `F_video` 与 competition score 仍为 null。

## Pairwise nested-CV

| representation | outer runs | binary proxy F1 mean ± sd | Spearman mean ± sd | NDCG@15% mean ± sd | author-style summary F1 mean ± sd | empty prediction |
|---|---:|---:|---:|---:|---:|---:|
| A0 ResNet18 | 30 | 0.155797 ± 0.042412 | 0.426467 ± 0.108241 | 0.651274 ± 0.064576 | 0.217504 ± 0.028750 | 0.0300 |
| DeiT-S | 30 | **0.172148 ± 0.031270** | 0.424183 ± 0.065153 | **0.669796 ± 0.045305** | **0.233615 ± 0.013983** | 0.0033 |

Per-video averaging across the six repeated observations gives DeiT-S minus A0 deltas of +0.01635 binary F1, -0.00228 Spearman, +0.01852 NDCG@15%, and +0.01611 summary F1. A 200,000-replicate paired video bootstrap gives F1 CI `[-.00684,.04461]`, Spearman `[-.06504,.06303]`, NDCG@15 `[-.01741,.05856]`, and summary F1 `[-.00573,.04064]`; every interval crosses zero. The raw per-video paired distributions contain only 52%, 44%, 52%, and 50% positive videos respectively. The result supports a narrow finding: the ranking objective improves the aggregate DeiT-S summary/NDCG direction in this protocol, while threshold-free Spearman remains tied. It does not justify replacing A0 fallback or reopening DeiT hyperparameter search.

## Acquisition status

The remote Clash/Mihomo HTTP proxy at `127.0.0.1:7890` successfully served Hugging Face range requests. YouTube Highlights indexing reached 358 members and recovered README/annotation metadata; the 9.9 GB media tar is still incomplete, so it is not counted as OOD video evidence. DHF1K 021–030 RAR archives were recovered through the proxy, but the available `7z` cannot extract their `video/021.AVI` member (`Unsupported Method`); no spatial transfer score was produced. Existing 20-video RetargetVid/DHF1K GT results remain the only spatial IoU evidence.

## Decision

Keep A0 center as engineering fallback and DeiT-S as a promising S-tier challenger. Pairwise ranking is retained as a measured experimental branch, not an accepted default, because it has no independent raw-video OOD confirmation and its ranking correlation gain is absent. Do not call YouTube Highlights or DHF1K 021–030 acquired benchmarks until media extraction, frame alignment, and native evaluator checks pass.
