# InternVideo2 Stage2 release — 2026-09-27

`SUB_F_INTERNVIDEO2_STAGE2_NATIVE_V1` uses InternVideo2 Stage2-1B vision encoder, the fixed native-QVH temporal head, threshold `.35`, 2 FPS sampling, and the unchanged YuNet `true_face_smooth` spatial path. No official-test statistic was used for model selection or parameter changes.

The frozen raw-video run covered all 174 official videos and produced 87,781 predictions with zero empty videos. Loaded neural weight bytes are `2,827,511,457` and parameter count is `1,020,930,561`, placing it in the 500M–9B tier (`k_size=0.90`). The merged project validator, independent checker, and a fresh post-unzip independent check all returned valid with zero errors.

Artifacts:

- ZIP: `/data/aic/official_test_20260926/submissions/SUB_F_INTERNVIDEO2_STAGE2_NATIVE_V1_FINAL/upload.zip`
- ZIP SHA256: `a7622bae7b78712549a344f5b5d61fd1762c395a734d82a2c85173bc344a6537`
- Predictions SHA256: `1322320b486b6a3595870b30e9ebc8b561751d139094deb0dc237570557e4e58`
- `run_manifest.json` and `upload.zip.sha256` are stored beside the ZIP.

The K710 comparison completed full 96/24 feature extraction and fit-only training. Its best native-QVH DEV values were F1 `.76543`, Spearman `.19818`, and NDCG `.96576`, below Stage2 `.21433/.96962`; it is therefore retained as a diagnostic control and was not released for official inference.

Official platform score for SUB_F was subsequently reported as `34.42`; the raw/size decomposition was not provided.
