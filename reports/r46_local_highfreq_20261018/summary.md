# R46 D3 local-state residual package

Remote shards 0--3 were merged after complete inference. The output covers
all 426 videos and 199,037 predictions with the exact `round(.85*N)` budget.
It copies R26 B3 boxes frame-for-frame and changes only the temporal set using
the fixed 4 fps local-state residual (`eta=.20`, Ridge `alpha=10`).

The independent checker passes at 450.283377 MB parameter units, and the ZIP
round trip is byte-identical. A fixed two-sided 5% edge-band derivative was
also packaged as a risk-control candidate; no boundary or coefficient sweep
was run.

- D3 full: `submissions/r46_local_packaged_20261018/R46_HIGHFREQ_LOCAL_RESIDUAL_B3/`
  - predictions SHA `ebdf0f1869a83ec425a52c127535cd341e0ea5a546221e4dc4997f46c39004dc`
  - ZIP SHA `b21797bfc812726958084bda2986c55335f13a83b919b63a11438f9536ef7717`
- D3 edge guarded: `.../R46_HIGHFREQ_LOCAL_BOUNDARY_GUARDED_B3/`
  - predictions SHA `c4126f7d818c60aa7bd4da44fb0d406ff3db1daf9fc8b418f7ec0791d3332e3b`
  - ZIP SHA `8e868ca389ae8db1c99ca17792c8ab977ca2f53d6e4f69d0f7c344777014e461`

Both are packaged and unuploaded; official scores are unknown.
