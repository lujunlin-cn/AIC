# R70 spatial utility readout pilot

This experiment tests a high-leverage spatial hypothesis after R67: the legal
129-window candidate set is adequate, while the candidate utility readout may
be the limiting layer. The original V8 B3 checkpoint is absent from the
current AutoDL clone, so every result here is explicitly a **new R70 readout**.

The V9 vision tower and its processor contract are frozen. Public native crop
tables from RetargetVid and LIVE are split by source into train and dev. The
readout receives the same 2,305-dimensional candidate feature contract used by
the V9 student (`window`, `outside`, `difference`, normalized position) and
predicts native IoU for all 129 legal max-windows. Official test data and
labels are never used.

The promotion decision requires held-out source-level improvement over the
center candidate on both public source pools. A positive result would justify
re-extracting the 426 competition videos and making one spatial-only package
with the R26 center85 temporal mask. A negative result closes this readout
route for the current vision representation; it does not prove that the
missing B3 head or official-domain crop policy is optimal.

The implementation is [r70_spatial_utility_pilot.py](../../scripts/r70_spatial_utility_pilot.py),
and the frozen protocol is [R70_SPATIAL_UTILITY_PILOT.json](../../configs/R70_SPATIAL_UTILITY_PILOT.json).
