---
license: apache-2.0
task_categories:
- summarization
language:
- en
tags:
- cross-modal-video-summarization
- video-summarization
- video-captioning
pretty_name: VideoXum
size_categories:
- 10K<n<100K
---


# Dataset Card for VideoXum

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Splits](#data-splits)
  - [Data Resources](#data-resources)
  - [Data Fields](#data-fields)
  - [Annotation Sample](#annotation-sample)
- [Citation](#citation)

## Dataset Description
- **Homepage:** https://videoxum.github.io/
- **Paper:** https://arxiv.org/abs/2303.12060

### Dataset Summary
The VideoXum dataset represents a novel task in the field of video summarization, extending the scope from single-modal to cross-modal video summarization. This new task focuses on creating video summaries that containing both visual and textual elements with semantic coherence. Built upon the foundation of ActivityNet Captions, VideoXum is a large-scale dataset, including over 14,000 long-duration and open-domain videos. Each video is paired with 10 corresponding video summaries, amounting to a total of 140,000 video-text summary pairs.

### Languages
The textual summarization in the dataset are in English.


## Dataset Structure

### Dataset Splits
|             |train  |validation| test  | Overall |
|-------------|------:|---------:|------:|--------:|
| # of videos | 8,000 |  2,001   | 4,000 | 14,001  |

### Dataset Resources
- `train_videoxum.json`: annotations of training set
- `val_videoxum.json`: annotations of validation set
- `test_videoxum.json`: annotations of test set

### Dataset Fields
- `video_id`: `str` a unique identifier for the video.
- `duration`: `float` total duration of the video in seconds.
- `sampled_frames`: `int` the number of frames sampled from source video at 1 fps with a uniform sampling schema.
- `timestamps`: `List_float` a list of timestamp pairs, with each pair representing the start and end times of a segment within the video.
- `tsum`: `List_str` each textual video summary provides a summarization of the corresponding video segment as defined by the timestamps.
- `vsum`: `List_float` each visual video summary corresponds to key frames within each video segment as defined by the timestamps. The dimensions (3 x 10) suggest that each video segment was reannotated by 10 different workers.
- `vsum_onehot`: `List_bool` one-hot matrix transformed from 'vsum'. The dimensions (10 x 83) denotes the one-hot labels spanning the entire length of a video, as annotated by 10 workers.

### Annotation Sample
For each video, We hire workers to annotate ten shortened video summaries.
``` json
{
    'video_id': 'v_QOlSCBRmfWY',
    'duration': 82.73,
    'sampled_frames': 83
    'timestamps': [[0.83, 19.86], [17.37, 60.81], [56.26, 79.42]],
    'tsum': ['A young woman is seen standing in a room and leads into her dancing.',
             'The girl dances around the room while the camera captures her movements.',
             'She continues dancing around the room and ends by laying on the floor.'],
    'vsum': [[[ 7.01, 12.37], ...],
             [[41.05, 45.04], ...],
             [[65.74, 69.28], ...]] (3 x 10 dim)
    'vsum_onehot': [[[0,0,0,...,1,1,...], ...],
                    [[0,0,0,...,1,1,...], ...],
                    [[0,0,0,...,1,1,...], ...],] (10 x 83 dim)
}
```

### File Structure of Dataset
The file structure of VideoXum looks like:
```
dataset
└── ActivityNet
    ├── anno
    │   ├── test_videoxum.json
    │   ├── train_videoxum.json
    │   └── val_videoxum.json
    └── feat
        ├── blip
        │   ├── v_00Dk03Jr70M.npz
        │   └── ...
        └── vt_clipscore
            ├── v_00Dk03Jr70M.npz
            └── ...
```


## Citation

```bibtex
@article{lin2023videoxum,
  author    = {Lin, Jingyang and Hua, Hang and Chen, Ming and Li, Yikang and Hsiao, Jenhao and Ho, Chiuman and Luo, Jiebo},
  title     = {VideoXum: Cross-modal Visual and Textural Summarization of Videos},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2023},
}
```
