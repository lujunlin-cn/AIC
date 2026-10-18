# R59 TimeLens2 4fps mask + edge ensemble

R59 reuses the R58 fixed multi-view decoder but uses a fresh 4fps TimeLens2 run (`max_pixels=50176`) over all 426 official videos. It combines interval membership (`eta_inside=.20`) and interval endpoint evidence (`eta_edge=.50`) around the exact centre-85 rank, with V9 full-frame B3 boxes. This is an aggressive observation-density intervention and a separate candidate from R58.

- Dense response SHA256: `bf712c77d94cdfd41d07e1d9aa0ec2719bd33328787df43662d96d6a64dc1906`
- ZIP SHA256: `ef46457ab64b9a6d2a160cf45ba60ec75f2139fe2f74209470cc4da162cdad76`
- Predictions SHA256: `1f9c54c2783a8e290d33a21847a5d6c47edfa1d9c54ae387de79063efc1507ac`
- 426 videos / 199037 predictions; independent checker valid.
- Total inference parameters: 8,854,531,425 (8.8545B), under the 9B limit if V9 artifact is accepted.
- Compared to the V9 center parent, 424/426 videos changed; 14,028 frame swaps (32.93/video mean). R58 2fps changed 421/426 and 12,933 swaps, so R59 is a genuinely different dense candidate.

The 4fps run used the existing TimeLens2-8B, `fps=4`, `max_pixels=50176`, generic highlight prompt, and completed 426/426 responses without OOM.
