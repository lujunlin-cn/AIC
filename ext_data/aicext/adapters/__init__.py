"""Reader adapters, one per supervision type (see aicext.schema.ANNOTATION_TYPES)."""
from .base import Index, TorchDataset, collate_pad  # noqa: F401
from .spatial import (CropCandidatesAdapter, DenseCropAdapter, MaskletAdapter, MaskSequenceAdapter,  # noqa: F401
                      ObjectTrackAdapter, SaliencyMapAdapter, SparseCropAdapter)
from .temporal import GridScoreAdapter, SegmentLabelAdapter, ShotBoundaryAdapter, UserSelectionAdapter  # noqa: F401,E501
