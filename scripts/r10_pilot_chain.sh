#!/bin/bash
# R10 audio pilot chain: 8 CPU shards over the 64 pre-registered pilot
# sources (132 frags), then a coverage sanity summary.  Absolute paths only;
# no pgrep/pkill; incremental npz per frag (resume-safe).
set -u
D=/data/aic/experiments_910a/LFM_V11
SRC=/root/AIC/scripts/r10_audio_extract.py
PILOT=$D/r10_audio/pilot_sources_64.txt
echo "pilot chain start $(date)" > $D/r10_pilot_chain.log
for S in 0 1 2 3 4 5 6 7; do
  python3 $SRC --shard $S --nshards 8 --sources $PILOT \
    --out $D/r10_audio > $D/r10_audio_s$S.log 2>&1 &
done
wait
echo "extract done $(date)" >> $D/r10_pilot_chain.log
python3 - <<'PYEOF' >> $D/r10_pilot_chain.log 2>&1
import numpy as np, glob
fs = sorted(glob.glob('/data/aic/experiments_910a/LFM_V11/r10_audio/p*/*.npz'))
av = np.concatenate([np.load(f)['avalid'] for f in fs])
ha = np.array([int(np.load(f)['has_audio']) for f in fs])
s527 = np.concatenate([np.load(f)['ast_527'].astype(np.float32) for f in fs])
# AudioSet class indices: 0 Speech, 137 Music, 285 Crowd? -> report top named coverage
print('pilot frags', len(fs), 'slots', len(av), 'avalid_frac', round(float(av.mean()), 4),
      'has_audio_frac', round(float(ha.mean()), 4))
for name, idx in (('Speech', 0), ('Music', 137), ('Cheering', 285), ('Crowd', 292)):
    p = 1.0 / (1.0 + np.exp(-s527[:, idx]))
    print(f'  {name}: mean_sigmoid {round(float(p.mean()), 4)} frac>0.5 {round(float((p > 0.5).mean()), 4)}')
PYEOF
echo "pilot chain done $(date)" >> $D/r10_pilot_chain.log
