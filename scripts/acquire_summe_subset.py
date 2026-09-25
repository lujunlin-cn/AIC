#!/usr/bin/env python3
"""Restore a bounded SumMe subset from a ZIP member inside the Zenodo tar."""
import argparse,io,json,zipfile,hashlib,time
from pathlib import Path
from scripts.acquire_range_tar import Remote
class Slice(io.RawIOBase):
    def __init__(self,remote,base,size):self.r=remote;self.base=base;self.size=size;self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,pos,whence=0):
        self.pos=pos if whence==0 else self.pos+pos if whence==1 else self.size+pos
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b""
        chunks=[]
        for start in range(0,n,4*1024*1024):
            chunk=self.r.get(self.base+self.pos+start,min(4*1024*1024,n-start));chunks.append(chunk)
        self.pos+=n
        return b"".join(chunks)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--count",type=int,default=8);ap.add_argument("--index-only",action="store_true")
    a=ap.parse_args();root=Path(a.root);meta=json.loads((root/"tar_index.json").read_text());row=next(x for x in meta["members"] if x["name"].endswith("SumMe.zip"))
    remote=Remote(meta["source"]);remote.start=-1;remote.data=b""
    archive=zipfile.ZipFile(Slice(remote,row["offset"],row["size"]))
    info=[dict(name=x.filename,size=x.file_size,compressed=x.compress_size) for x in archive.infolist()]
    (root/"summe_zip_index.json").write_text(json.dumps(info,indent=2))
    videos=sorted([x for x in archive.infolist() if x.filename.lower().endswith((".mp4",".avi",".mov"))],key=lambda x:(x.compress_size,x.filename))[:a.count]
    selected=videos+[x for x in archive.infolist() if x.filename.lower().endswith((".mat",".m",".txt",".pdf"))]
    print(json.dumps({"selected":[(v.filename,v.file_size) for v in videos],"total_bytes":sum(v.compress_size for v in selected)}),flush=True)
    if a.index_only:return
    manifest=[]
    for item in selected:
        dest=(root/"extracted"/item.filename).resolve()
        if (root/"extracted").resolve() not in dest.parents:raise ValueError("unsafe ZIP name")
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or dest.stat().st_size!=item.file_size:
            data=archive.read(item) # zipfile verifies member CRC.
            dest.write_bytes(data)
        manifest.append(dict(path=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),zip_member=item.filename,source=meta["source"],selection="8_smallest_compressed_videos_no_label_inspection"))
        (root/"subset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print("EXTRACTED",item.filename,item.file_size,flush=True)
if __name__=="__main__":main()
