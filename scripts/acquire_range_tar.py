#!/usr/bin/env python3
"""Index an uncompressed tar over strict HTTP ranges; extract selected members."""
import argparse,hashlib,json,time,tarfile,urllib.request
from pathlib import Path
class Remote:
    def __init__(self,url):self.url=url;self.start=-1;self.data=b"";self.total=None
    def get(self,start,n):
        if self.start<=start and start+n<=self.start+len(self.data):
            return self.data[start-self.start:start-self.start+n]
        req=urllib.request.Request(self.url,headers={"Range":f"bytes={start}-{start+n-1}"})
        with urllib.request.urlopen(req,timeout=45) as r:
            cr=r.headers.get("Content-Range","")
            if r.status!=206 or not cr.startswith(f"bytes {start}-"):raise RuntimeError(f"bad range {r.status} {cr}")
            data=r.read()
            self.total=int(cr.split("/")[-1])
        if len(data)!=n:raise ValueError("short HTTP range")
        self.start=start;self.data=data;return data
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--url",required=True);ap.add_argument("--root",required=True)
    ap.add_argument("--index-only",action="store_true");ap.add_argument("--include",default="summe");ap.add_argument("--max-mb",type=int,default=900)
    a=ap.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);r=Remote(a.url);offset=0;rows=[];pending=None
    index=root/"tar_index.json"
    if index.exists():rows=json.loads(index.read_text())["members"]
    else:
        while True:
            block=r.get(offset,512)
            if block==b"\0"*512:break
            member=tarfile.TarInfo.frombuf(block,encoding="utf-8",errors="surrogateescape")
            name=pending or member.name;pending=None
            if member.type in (tarfile.GNUTYPE_LONGNAME,tarfile.GNUTYPE_LONGLINK):
                pending=r.get(offset+512,member.size).rstrip(b"\0").decode()
            else:
                row={"name":name,"offset":offset+512,"size":member.size,"regular":member.isfile()}
                rows.append(row);print(json.dumps(row),flush=True)
            offset+=512+((member.size+511)//512)*512
            index.write_text(json.dumps({"source":a.url,"members":rows},indent=2)+"\n")
        index.write_text(json.dumps({"source":a.url,"complete":True,"members":rows},indent=2)+"\n")
    if a.index_only:return
    total=0;manifest=[]
    for row in rows:
        name=row["name"]
        if not row["regular"] or a.include.lower() not in name.lower():continue
        # First restore small labels/features and raw SumMe videos, within a fixed byte budget.
        if total+row["size"]>a.max_mb*1000000:continue
        dest=(root/name).resolve()
        if root.resolve() not in dest.parents:raise ValueError("unsafe tar path")
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or dest.stat().st_size!=row["size"]:
            partial=dest.with_suffix(dest.suffix+".partial")
            with partial.open("wb") as out:
                for i in range(0,row["size"],4*1024*1024):
                    out.write(r.get(row["offset"]+i,min(4*1024*1024,row["size"]-i)))
            partial.rename(dest)
        total+=row["size"]
        manifest.append({**row,"path":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"source":a.url})
        print("EXTRACTED",name,row["size"],flush=True)
        (root/"download_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":main()
