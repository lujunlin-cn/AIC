#!/usr/bin/env bash
# Source on the remote host before running any ext_data script.
#   source /home/supie/AIC/ext_data/scripts/env.sh
export AIC_EXT_ROOT=${AIC_EXT_ROOT:-/data/aic/external_datasets}
# Reverse tunnel to a workstation proxy (only used for hosts the server cannot
# reach directly; see aicext/download.py PROXY_HOSTS). Leave empty to disable.
export AIC_EXT_PROXY=${AIC_EXT_PROXY:-http://127.0.0.1:18890}
# Free space that downloads/extracts must never consume (GB).
export AIC_EXT_MIN_FREE_GB=${AIC_EXT_MIN_FREE_GB:-700}
export AIC_EXT_PY=/opt/miniconda3/envs/cv/bin/python
# Tool overlay: yt-dlp, pyarrow, rarfile, pycocotools (numpy comes from the env).
export PYTHONPATH=$AIC_EXT_ROOT/_tools/pyoverlay:/home/supie/AIC/ext_data${PYTHONPATH:+:$PYTHONPATH}
# The cv env takes cv2/h5py/av from ~/.local; do not set PYTHONNOUSERSITE here.
export PATH=$AIC_EXT_ROOT/_tools/pyoverlay/bin:/data/aic/tmp/unrar/rar:$PATH
# Keep CPU/IO polite towards running experiments.
export AIC_EXT_NICE="nice -n 15 ionice -c2 -n7"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
