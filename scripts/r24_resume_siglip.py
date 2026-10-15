"""Resume a known public LFS blob; require range contract and complete SHA256."""
import hashlib,json,os,time
from pathlib import Path
import requests
base=Path('/root/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384')
expected='ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a'
part=base/'blobs'/(expected+'.incomplete'); complete=base/'blobs'/expected
total=3511950624; initial=part.stat().st_size if part.exists() else complete.stat().st_size
start=time.time()
while not complete.exists() and part.stat().st_size<total:
    offset=part.stat().st_size
    u='https://hf-mirror.com/google/siglip-so400m-patch14-384/resolve/main/model.safetensors?resume='+str(offset)
    try:
        with requests.get(u,headers={'Range':f'bytes={offset}-{min(offset+(4<<20)-1,total-1)}'},stream=True,timeout=(20,30)) as r:
            assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {offset}-'),(r.status_code,r.headers.get('Content-Range'))
            with part.open('ab') as f:
                for b in r.iter_content(1<<20):
                    f.write(b); print('bytes',f.tell(),'/',total,flush=True)
    except requests.RequestException as e:
        print(type(e).__name__,flush=True)
        if time.time()-start>600: raise
p=complete if complete.exists() else part
assert p.stat().st_size==total
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
assert h.hexdigest()==expected, h.hexdigest()
if p==part:os.replace(part,complete)
snapshot=(base/'refs/main').read_text().strip(); link=base/'snapshots'/snapshot/'model.safetensors'
if not link.exists():link.symlink_to('../../blobs/'+expected)
print(json.dumps(dict(status='PASS',initial_bytes=initial,total_bytes=total,sha256=h.hexdigest(),seconds=time.time()-start,snapshot=snapshot)),flush=True)
