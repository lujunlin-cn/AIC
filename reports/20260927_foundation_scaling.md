# Foundation scaling continuation — 2026-09-27

The InternVideo2 Stage1-1B K700 candidate received official platform score 34.43, versus SUB_C 6.64. The platform did not provide a raw/size decomposition, so 34.43 is recorded only as the reported official score.

VideoMAE-Large (about 304M parameters, M tier) completed the 96/24 native-QVH protocol: Spearman 0.17574, NDCG 0.96752. Paired deltas versus matched DeiT were +0.06211 and +0.00465; bootstrap CIs crossed zero.

InternVideo2 Stage2-1B (about 1.05B vision-plus-head parameters; source bundle 2,827,511,457 bytes) completed the same protocol: Spearman 0.21433, NDCG 0.96962. Paired deltas were +0.10069 and +0.00675; CIs crossed zero but the direction is stronger than VideoMAE-Large. A frozen raw-video release is running with the unchanged YuNet spatial path and no official-test tuning.

K710 was downloaded and a bounded 12-video probe reached Spearman 0.33586; it remains a small-sample action-pretraining control and has not been promoted without the full 96/24 run.
