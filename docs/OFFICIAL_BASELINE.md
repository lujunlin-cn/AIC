# Public baseline contract reference

At 2026-09-25, the public repository `HVision-NKU/TempSamp-R1` was reachable at
commit `3fd4e851aa3310156dcef1f933751c3860aa34ac`. Its `baseline/README.md`
and `test_index.json` are public format references, not the current detailed
rules source. The README describes 174 compact index entries (119 `[16,9]`, 55
`[9,16]`) and one JSONL row per entry.

The current detailed rules in `01_赛事规则与评分标准.md` remain authoritative
where the old baseline differs. In particular, the baseline README describes a
fixed largest crop and integer coordinates; the current rules document records
that this is historical baseline behavior and allows the detailed `[x,y,w]`
contract. We therefore use the public index only as a compact input reference,
then probe each local video before validation.

Use:

```bash
PYTHONPATH=. python scripts/build_video_index.py \
  --compact-index /path/to/test_index.json \
  --video-dir /path/to/video \
  --output /path/to/enriched_index.jsonl
```

The generated index records actual decoded frame count, PTS-related video
metadata, coded dimensions, rotation, and the path used for inference. It does
not include labels and does not download or infer competition test data.
