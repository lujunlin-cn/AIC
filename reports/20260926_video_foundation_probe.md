# Video foundation probe

VideoMAEv2-Base loaded on V100 with 86,227,200 parameters and a 344,924,592-byte source checkpoint. The corrected native-QVH candidate run completed 120 train/dev feature caches and 20 head epochs; it did not improve the matched DeiT ranking metrics.

InternVideo2 Stage1-1B K700 loaded with 1,020,710,144 encoder parameters. The source checkpoint is 2,042,600,861 bytes, SHA256 `a615568ca9f386509373e4943a5924adfb36f640c2e7c460930c277072e48caf`. The native candidate head has 1,663,745 parameters, for 1,022,373,889 parameters including the temporal head; YuNet adds 232,589 bytes. V100 compatibility passed at 8 frames (`0.2095 s`, 2267 MiB) and 16 frames (`0.5385 s`, 3084 MiB).

Candidate release remains frozen at threshold `0.35`, 2 FPS, native DEV-selected checkpoint, and dense YuNet `true_face_smooth`. No official-test statistic is used for selection. The full raw-video release is running only as an engineering artifact; official score and competition score remain null until the platform evaluates it.
