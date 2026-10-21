"""Bounded read-only probes of public official model endpoints; no full weights."""
import datetime
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path('/tmp/aic_visual_primary_sources/weight_access_audit.json')
URLS = [
    ('raft_author_bundle', 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip'),
    ('raft_small_torchvision_C_T_V1', 'https://download.pytorch.org/models/raft_small_C_T_V1-ad48884c.pth'),
    ('transnet_author_raw_lfs_pointer', 'https://raw.githubusercontent.com/soCzech/TransNetV2/master/inference/transnetv2-weights/variables/variables.data-00000-of-00001'),
    ('transnet_author_data', 'https://media.githubusercontent.com/media/soCzech/TransNetV2/master/inference/transnetv2-weights/variables/variables.data-00000-of-00001'),
    ('transnet_author_index', 'https://media.githubusercontent.com/media/soCzech/TransNetV2/master/inference/transnetv2-weights/variables/variables.index'),
    ('transnet_author_saved_model', 'https://media.githubusercontent.com/media/soCzech/TransNetV2/master/inference/transnetv2-weights/saved_model.pb'),
    ('gmflow_author_scale1_things', 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth'),
    ('tsm_author_mobilenetv2', 'https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth'),
]

def probe(name, url):
    row = {'name': name, 'url': url}
    for method in ('HEAD', 'GET'):
        req = urllib.request.Request(url, method=method, headers={
            'User-Agent': 'AIC-public-source-audit/1.0',
            **({'Range': 'bytes=0-511'} if method == 'GET' else {})})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read(512) if method == 'GET' else b''
                result = {
                    'status': response.status, 'final_url': response.geturl(),
                    'headers': {k: response.headers.get(k) for k in
                                ('Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges')},
                    'bytes_read': len(data),
                }
                if data:
                    low = data.lstrip().lower()
                    result.update({'prefix_sha256': hashlib.sha256(data).hexdigest(),
                                   'prefix_hex_16': data[:16].hex(),
                                   'looks_html': low.startswith((b'<!doctype html', b'<html')),
                                   'looks_git_lfs_pointer': data.startswith(b'version https://git-lfs.github.com/spec/v1'),
                                   'looks_zip': data.startswith(b'PK\x03\x04')})
                    if result['looks_git_lfs_pointer']:
                        result['public_lfs_pointer'] = data.decode('ascii')
                row[method] = result
        except Exception as exc:
            row[method] = {'error_type': type(exc).__name__, 'message': str(exc)[:240]}
    return row

records = [probe(name, url) for name, url in URLS]
OUT.write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           'max_bytes_read_per_get': 512,
                           'full_weight_downloaded_or_loaded': False,
                           'records': records}, ensure_ascii=False, indent=2) + '\n')
for row in records:
    print(json.dumps(row, ensure_ascii=False))
