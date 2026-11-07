# Scale SFT/DPO Asset Inventory

Audit time: 2026-11-07T17:59:19.290980+00:00

No AIC official test media, labels, predictions, or hidden-test-derived information were read.
Public-dataset test labels were not used for fitting, candidate selection, or threshold choice.

## GAICD

- source_url: `"https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch"`
- license_status: `"not recorded in local dataset card; verify original data terms before external use"`
- competition_use_authorized: `null`
- media_count: `3336`
- unique_source_count: `3336`
- official_split_source_counts: `{"train": 2636, "val": 200, "test": 500}`
- journal_train_candidate_count: `231489`
- journal_train_unrated_candidate_count: `3734`
- journal_train_annotation_missing: `0`
- manifest_hashes: `{"dataset_card.json": "76deec38f4cad478e2fcd55762751bc13dd23a53e898537670f503f1ef54542c", "media.jsonl": "b8df764a4c7221a5385fa0df0acd3d23b3354982aabadf3da24e626bbd2fdf98", "annotations.jsonl": "90b378f15c8bfb6695ffe23c41f1bc0fa2f169dd22200fef346c91a7f13b422f"}`
- raw_tree: `{"files": 9149, "bytes": 900897198, "tree_sha256": "f4b1bc7820f82e113bb41ecc43cf56cc9e758c3c2cd70720f0ceec2b71fcfe79"}`
- blockers: `[]`

## SACD

- license_status: `"not recorded in local provenance metadata; verify before external use"`
- competition_use_authorized: `null`
- images: `2777`
- subject_annotations: `2906`
- split_annotation_counts: `{"train": 2326, "val": 290, "test": 290}`
- split_image_overlap_counts: `{"train_val": 22, "train_test": 20, "val_test": 4}`
- explicit_ranked_pair_count: `0`
- source_tree_sha256_from_prior_validation: `"0f61926d885062cd36717f8ee53dcacac6e262ff9c9e3d1e57fcd966af2438c4"`
- blockers: `["split files share source images", "no explicit subject-linked ranked-pair records"]`

## LIVE_YT_VC

- source_url: `"https://github.com/steven413d/LIVE-YT-VideoCropping"`
- license_status: `"not confirmed in local asset metadata"`
- competition_use_authorized: `null`
- labeled_sources: `1800`
- video_files_locally_present: `37`
- source_split_manifest: `{"provided": false, "source_counts": null, "source_disjoint": null, "invalid_split_sources": 0, "missing_required_splits": ["train", "val"]}`
- blockers: `["LIVE source split manifest is missing", "LIVE manifest must contain both train and val sources"]`

## GNMC

- source_url: `"https://doi.org/10.5281/zenodo.6228834"`
- license: `"PolyForm-Noncommercial-1.0.0"`
- competition_use_authorized: `null`
- split_image_counts: `{"train": 7000, "validation": 1000, "test": 2000}`
- source_split_disjoint: `true`
- blockers: `[]`

## P1 Parent

{
  "path_supplied": null,
  "present": false,
  "manifest_or_weight_files": [],
  "sha256": null,
  "parameter_count": null,
  "blocker": "P1 parent model, tokenizer, and adapter are not present in the local dataset/code checkout"
}
