# Valen-2B crop-choice SFT verdict — 2026-10-20

## Verdict (amended after RV 601--700 confirmation)

The first Valen-2B crop-choice SFT run completed and restored successfully for evaluation. The media/option/label contract passes for all 12,822 public native-crop rows. The newly completed RetargetVid 601--700 exposed transfer pool gives a positive source-level confirmation (`+0.018466` IoU versus the existing head, 95% bootstrap CI `[+0.010735,+0.026371]`). This validates the direction of native crop-choice supervision across 100 held-out sources within the exposed project universe. It still is not an independent fresh-source or official-score confirmation, and it does not by itself justify a submission package.

The next action is one utility-aware candidate-choice objective on public TRAIN, evaluated with the frozen four-candidate contract and source-level held-out data. Keep the prompt, temperature, fusion weight, and time policy fixed. Batch size 16 is required for all model inference. If the utility objective does not improve the frozen source-level gate, close this Valen deployment route rather than scan prompts or calibration.

## Runtime and training evidence

- Model: Valen-2B revision `fb9205c2541abc97ce276670e66f79b78d9068fc`.
- Runtime parameter count after loading the checkpoint: `2,248,463,977`; trainable parameters: `60,396,329`. This is below the 9B rule when deployed with the fixed center-time policy alone.
- Trainable groups were present and updated: decision head `1,583,913`, LoRA `33,638,400`, visual merger `25,174,016`.
- The final checkpoint contains 411 trainable tensors and optimizer state for all 411 tensors. All 186 LoRA-B tensors are nonzero after training. The checkpoint progress is epoch 0, step/optimizer step 244, cursor 11276.
- The run used the actual remote batch configuration: BF16 autocast, SDPA, `microbatch_size=16`, `tokens_per_step=32768`, one epoch, gradient checkpointing and prefetching.
- Effective progress was `11,276` states/questions and `7,904,252` compute tokens over 244 steps. Final elapsed training time was `5,152.50 s` (85.9 minutes); peak allocated CUDA memory was `19,160,744,960` bytes.
- Evaluation loaded the checkpoint through Valen's `load_checkpoint` path and completed 1,546/1,546 questions in 1,466.50 s. No official test media or labels were read.

The local configuration was synchronized after the run to match the remote batch-16 configuration. The remote configuration SHA is `a2b08dfd5372c5aee294221c671c8fbd64e0351d357570e6366cad2a303e190d`.

## Contract audit

The audit uses `(row_key, ratio)` as the row identity because one source frame occurs at both target aspect ratios. It checks the original frame, candidate image path, candidate index, A/B/C/D key, legal box, target ratio and soft-IoU target.

- Public source records: 12,822 rows, 1,663 sources.
- SFT train: 11,276 rows, 1,495 sources.
- SFT dev: 1,546 rows, 168 sources.
- Candidate images checked: 51,288.
- Contract errors: zero.
- The existing full-head choice is inside the four-candidate pool on all 12,822 rows; center candidate index 64 is also present in every row.

The final audit is [contract_audit_v2.json](contract_audit_v2.json). Its native-record SHA is `e2dca06c1771edf53d27a8f8a202ddfe8f4cce5f9455a39e24fedb8b52ccfed7`.

## Source-macro results

All values below average rows within each source before averaging sources. `oracle129` is the best of all 129 legal windows; `oracle4` is the best of the four candidates actually shown to Valen. Candidate recall loss is `oracle129 - oracle4`; candidate selection loss is `oracle4 - Valen`.

| Domain | Sources | Valen | Head | Center | Oracle-4 | Oracle-129 | Valen−Head, 95% CI | Recall loss | Selection loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RetargetVid | 20 | 0.699242 | 0.693241 | 0.622034 | 0.769392 | 0.826340 | +0.006001, [-0.010304,+0.022171] | 0.056948 | 0.070150 |
| LIVE | 148 | 0.558478 | 0.549693 | 0.480771 | 0.690038 | 0.777764 | +0.008784, [-0.006178,+0.024297] | 0.087727 | 0.131560 |
| All | 168 | 0.575236 | 0.566782 | 0.497588 | 0.699485 | 0.783547 | +0.008453, [-0.004944,+0.022082] | 0.084062 | 0.124249 |

Coverage is 1,546/1,546 rows and 168/168 sources. Severe degradation, defined before looking at the result as source delta at most -0.05, occurs for 1/20 RetargetVid sources and 30/148 LIVE sources; delta at most -0.10 occurs for 0/20 and 13/148 respectively. The largest LIVE declines are `video481` (-0.277), `video436` (-0.269), `video620` (-0.205), `video1607` (-0.160) and `video834` (-0.159).

The large gap between `oracle4` and `oracle129` shows a real candidate-coverage limit, especially in LIVE. However, the larger current loss is inside the four-candidate decision: `0.124249` source-macro IoU on average, compared with `0.084062` candidate recall loss. Valen closes only `0.008453` relative to the existing head. This is evidence for a candidate-internal utility problem, but the exposed development split cannot establish that it will transfer.

## Exposed RetargetVid 601--700 confirmation

The confirmation uses 4,081 rows from 100 sources, with 2,039 `1:3` rows and 2,042 `3:1` rows. All 4,081 predictions are covered and the official test is unused. Source-macro results are Valen `0.675226`, existing head `0.656760`, center `0.580568`, four-candidate oracle `0.762726`, and 129-candidate oracle `0.833979`. Candidate recall loss is `0.071254`; candidate-internal selection loss is `0.087500`.

Valen improves 71 sources and has two declines at or below `-0.05`; no source declines at or below `-0.10`. The 1:3 split improves by `+0.014766` (CI `[+0.001868,+0.028027]`) and the 3:1 split by `+0.022196` (CI `[+0.013865,+0.030809]`). These numbers are a source-level transfer diagnostic over an exposed pool: the sources have appeared in older project diagnostics or training branches, so they are not claimed as fresh independent confirmation.

The remote run used `batch=16` throughout: 255 complete batches and one tail batch, 614.5 seconds total, 6.64 rows/s. Runtime details are in [confirm_runtime_metrics.json](confirm_runtime_metrics.json), and the full source analysis is in [confirm_analysis_rv601_700.json](confirm_analysis_rv601_700.json).

## Exposure and interpretation boundary

The train/dev split is source-disjoint within this run, but the 168-source dev pool is the historical RetargetVid 20 + LIVE 148 development pool already used by R70 and the DINOv3 matched diagnostics. It must remain named exposed development. It is not fresh independent confirmation, and these IoU results cannot be converted into an AIC joint official score.

The zero-shot Valen result remains negative relative to the existing head on the earlier 168-source pool. The SFT run demonstrates that native crop supervision can move the decision in the intended direction; the 100-source RV confirmation strengthens that result but is still exposed project data, so it does not demonstrate a robust competition gain.

## Reproduction fingerprints

- Training data SHA from the run manifest: `5b96b69201a46079b623f121150442e6fe11e7367edc08461b31d2fffa938396`.
- Final checkpoint SHA: `0063f07cafbb5379a1c80bfab27ef50a28e10d25932e82d7e9ad5284b20dd561`.
- Evaluation metrics SHA: `4226345859466038b484118696dd420ea12dc4a94fd629e932259383796c990f`.
- Source metrics SHA: `59074bebaa7c4507a629876dd1e57852692430cabff5aaf36cbe5b7f965e2621`.
- Environment: RTX 4080 32GB, torch 2.8.0+cu128, transformers 5.4.0, peft 0.18.1, av 16.1.0.

