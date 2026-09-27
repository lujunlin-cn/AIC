"""Create deterministic parity shards for parallel frozen inference."""
import argparse, hashlib, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--index',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in Path(a.index).read_text().splitlines() if x.strip()]
    base=json.loads(Path(a.manifest).read_text())
    for part in (0,1):
        p=out/f'shard{part}.jsonl'
        p.write_text('\n'.join(json.dumps(r,separators=(',',':')) for i,r in enumerate(rows) if i%2==part)+'\n')
        m=dict(base,submission_id=f'{base["submission_id"]}_SHARD{part}',index_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        (out/f'shard{part}_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__': main()
