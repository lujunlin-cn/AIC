"""Read-only HEAD probes of author-linked public weights; never downloads a body.

This verifies W2 endpoint availability only, not full hashes, load, or inference.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import subprocess
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

TARGETS = {
    'u2netp': 'https://drive.google.com/uc?export=download&id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
    'unisal': 'https://raw.githubusercontent.com/rdroste/unisal/master/training_runs/pretrained_unisal/weights_best.pth',
    'yolox_nano': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth',
}

def shareable_headers(raw):
    """Omit ephemeral signed redirect URLs; retained transcript is not byte-original."""
    lines=[]
    redacted=False
    for line in raw.splitlines(keepends=True):
        if line.lower().startswith('location:') and ('sig=' in line.lower() or 'jwt=' in line.lower()):
            endpoint=urlsplit(line.split(':',1)[1].strip())
            line='location: '+endpoint.scheme+'://'+endpoint.netloc+' [temporary signed URL omitted]\n'
            redacted=True
        lines.append(line)
    return ''.join(lines),redacted

def probe(name, url, directory):
    cmd = ['curl', '-sS', '-I', '-L', '--connect-timeout', '10', '--max-time', '25', url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    transcript,redacted=shareable_headers(result.stdout)
    (directory / (name + '_headers.txt')).write_text(transcript)
    (directory / (name + '_stderr.txt')).write_text(result.stderr)
    statuses = re.findall(r'^HTTP/[^ ]+ +(\d+)', result.stdout, re.M)
    final_block = re.split(r'(?m)^HTTP/', result.stdout)[-1]
    def header(key):
        found = re.findall(r'^' + re.escape(key) + r':\s*(.*)$', final_block, re.M | re.I)
        return found[-1].strip() if found else None
    status = int(statuses[-1]) if statuses else None
    typ = header('content-type')
    return {
        'name': name, 'author_link_or_standard_download_url': url,
        'method': 'HEAD with redirects; -I; no body requested',
        'header_transcript': 'temporary signed redirect Location redacted; not byte-original' if redacted else 'headers as returned by curl text mode',
        'exit_code': result.returncode, 'status_chain': [int(x) for x in statuses],
        'final_status': status, 'content_type': typ,
        'content_length': header('content-length'),
        'content_disposition': header('content-disposition'),
        'weight_content_bytes_read': 0,
        'access_level': 'W2' if result.returncode == 0 and status == 200 and typ and 'application/octet-stream' in typ.lower() else 'W1',
        'complete_weight_hash_verified': False, 'weight_loaded': False,
    }

if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    logs = base / 'spatial_weight_heads'
    logs.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    out = {
        'timestamp_utc': now.isoformat(),
        'timestamp_asia_shanghai': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'author weight endpoint HEAD only; W2 not W3; shareable headers omit ephemeral signed redirect URLs',
        'results': [probe(n, u, logs) for n, u in TARGETS.items()],
    }
    (base / 'spatial_weight_access_audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
