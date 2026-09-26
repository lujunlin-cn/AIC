"""Merge fixed index shards and run the final submission checks."""
import argparse, json, zipfile
from pathlib import Path
from aic.contract import load_index, load_jsonl, write_submission
from scripts.independent_submission_check import check

def main():
    p=argparse.ArgumentParser(); p.add_argument('--index',required=True); p.add_argument('--shard',action='append',required=True); p.add_argument('--output',required=True); p.add_argument('--zip',required=True); p.add_argument('--weight-bytes',type=int,required=True); a=p.parse_args()
    index=load_index(a.index); order=list(index)
    by_id={}
    for shard in a.shard:
        for row in load_jsonl(shard):
            vid=row['video_id']
            if vid in by_id: raise ValueError('duplicate video_id '+vid)
            by_id[vid]=row
    if set(by_id)!=set(order): raise ValueError(f'coverage mismatch: {len(by_id)} vs {len(order)}')
    rows=[by_id[v] for v in order]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=False)
    report=write_submission(out,rows,index,stage='preliminary',actual_model_size_mb=a.weight_bytes/1_000_000).to_dict()
    independent=check(a.index,out,a.weight_bytes)
    z=Path(a.zip); z.parent.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED) as f: f.write(out,'predictions.jsonl')
    print(json.dumps({'validator':report,'independent':independent,'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'zip_bytes':z.stat().st_size},indent=2))
if __name__=='__main__': main()
