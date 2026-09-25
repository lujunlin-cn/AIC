"""Bounded resumable tar indexing; no video or label-based selection."""
import argparse,json,tarfile,time,urllib.request,hashlib
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seconds',type=int,default=900)
    a=ap.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    url='https://huggingface.co/datasets/jhanglee/youtube-highlights-full/resolve/main/youtube_highlights_full.tar'
    dest=root/'index.json';state=json.loads(dest.read_text()) if dest.exists() else {'url':url,'offset':0,'members':[],'complete':False,'pax':{}}
    cache_start=-1;cache=b''
    def get(start,n):
        nonlocal cache_start,cache
        if cache_start<=start and start+n<=cache_start+len(cache):return cache[start-cache_start:start-cache_start+n]
        size=max(n,65536)
        req=urllib.request.Request(url+f'?aic_range={start}_{size}',headers={'Range':f'bytes={start}-{start+size-1}'})
        with urllib.request.urlopen(req,timeout=25) as r:
            cr=r.headers.get('Content-Range','')
            if r.status!=206 or not cr.startswith(f'bytes {start}-'):raise ValueError('range mismatch: '+cr)
            data=r.read(size)
        if len(data)<n:raise ValueError('short range')
        cache_start=start;cache=data
        return data[:n]
    begin=time.monotonic()
    while not state['complete'] and time.monotonic()-begin<a.seconds:
        offset=state['offset'];block=get(offset,512)
        if block==b'\0'*512:
            state['complete']=True
        else:
            info=tarfile.TarInfo.frombuf(block,'utf-8','surrogateescape')
            if info.type in (tarfile.XHDTYPE,tarfile.XGLTYPE):
                raw=get(offset+512,info.size);pos=0;pax={}
                while pos<len(raw):
                    end=raw.index(b' ',pos);size=int(raw[pos:end]);key,value=raw[end+1:pos+size-1].split(b'=',1)
                    pax[key.decode()]=value.decode();pos+=size
                state['pax']=pax
            elif info.type==tarfile.GNUTYPE_LONGNAME:
                state['pax']={'path':get(offset+512,info.size).rstrip(b'\0').decode()}
            else:
                name=state['pax'].get('path',info.name);state['pax']={}
                row={'name':name,'offset':offset+512,'size':info.size,'regular':info.isfile()}
                state['members'].append(row)
                if info.isfile() and not name.endswith('.mp4') and info.size<2000000:
                    data=get(offset+512,info.size);p=(root/'metadata'/name).resolve()
                    if (root/'metadata').resolve() not in p.parents:raise ValueError('unsafe path')
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                    row['sha256']=hashlib.sha256(data).hexdigest()
                print(len(state['members']),name,info.size,flush=True)
            state['offset']=offset+512+((info.size+511)//512)*512
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(dest)
    print('INDEX_STOP',state['complete'],len(state['members']),flush=True)
if __name__=='__main__':main()
