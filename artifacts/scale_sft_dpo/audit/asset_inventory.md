# Verified assets — 2026-11-08

Remote: `ssh -p 10045 root@connect.westb.seetacloud.com`. Read-only audit used the previously pinned ED25519 key; no security check was disabled. Weight and public-frame audits completed in 97.19 s. Raw results: `remote_assets.json`.

| Asset | Verified identity / count |
|---|---|
| Compact base | `/root/autodl-tmp/videoora42_clone/models/compact`; eight shard hashes agree with `MODEL_FREEZE.json`; 8,996,674,800 tensor elements |
| P1 portrait adapter | `semifinal_20261107/code/adapters/portrait`; SHA256 `d18f187d630cc7854859858a827f105e278df4205960857da411d4fb22181013`; 1,966,080 parameters |
| P1 vertical adapter | `semifinal_20261107/code/adapters/vertical`; SHA256 `c89b05dd209295543e1b07de388dd24dba6e8fabca590f859bba6475948f9b1d`; 983,040 parameters |
| Parent deployment | Base+A+B = 8,999,623,920 parameters; conservative original buffer count 147,538; total 8,999,771,458 |
| Isolated Scale adapter | Rank 1 on q/v/o of eight full-attention layers; 204,800 parameters; exact names/shapes in `../smoke_test/metadata_probe.json` |
| Complete planned inference | 8,999,976,258 including buffers; 23,742 below 9B. Structural count verified with actual architecture and adapter configurations; real-weight loading still pending |
| Runtime | Python 3.12.3; torch 2.8.0+cu128; transformers 5.5.4; peft 0.20.0; safetensors 0.6.2 |
| No-card allocation | `memory.max=2147483648`, `cpu.max=50000 100000`; `nvidia-smi -L` permission denied. No CUDA job launched |

P1/P2 share the same parent base and A/B weights. Inspected `infer_candidates.py` uses P1 QP floor 0.15 and P2 0.6. The user-reported semifinal scores are P2=44.65, P1=44.63, B_MIX=42.88. Package hashes and model hashes are recorded separately; no package-level score is assigned to a single adapter.

Position outputs are per-mille `crop_center` coordinates in the original prompt; `generation.parse_center` projects a maximum legal window. The full P1 submission then applies its existing smoothing/projection. Scale uses the immutable **final deployed P1 center**, with both its original `[x,y,w]` and complete parent identity retained. Raw requested centers and final crop centers must not be silently substituted. The historical public-dev raw cache permits a limited geometry audit, not a new forward pass or a full training-center cache.

The historical `load_bundle(train_arm=...)` creates fresh A/B training adapters, so it is deliberately not used for Scale continuation. The new loader verifies base/runtime/tokenizer hashes, loads the actual frozen P1 A/B banks, and adds the small isolated Scale delta. Position activates only the original parent bank; Scale activates parent+Scale. Every activation reapplies the exact allowlist; checkpoint saving verifies that parent tensors are unchanged. No parent file is overwritten. This resolves the Position-freeze conflict within the 9B cap without adding an independent model or reward model.

Public LIVE cache: 54,000/54,000 files exist and match recorded hashes; 1,600/200 historical video sources are disjoint. One identical-image group crosses the split. Five complete train sources are quarantined in `../config/live_source_split.json`, leaving **1,595 train sources / 47,850 frames**, and the original 200 validation sources / 6,000 frames. Source-video hashes are unique across all 1,800 sources. Recorded original-video paths are absent remotely; cached frame bytes and extraction provenance are verified, but complete original-video PTS re-decoding has not been repeated.

Real-weight smoke remains blocked by the no-card resource allocation. A separate local environment was prepared without changing the shared Python installation; an incomplete weight transfer was stopped and was never used as a valid model. The metadata probe is explicitly not counted as real-model loading or training success.
