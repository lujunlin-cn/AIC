#!/usr/bin/env python3
"""Bounded non-solid RAR4 member recovery via HTTP ranges, preserving CRCs."""
import argparse,concurrent.futures,hashlib,json,struct,urllib.request
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--url",required=True);ap.add_argument("--root",required=True);ap.add_argument("--count",type=int,default=20)
    a=ap.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    def fetch(offset,size):
        for attempt in range(3):
            try:
                req=urllib.request.Request(a.url,headers={"Range":f"bytes={offset}-{offset+size-1}"})
                with urllib.request.urlopen(req,timeout=40) as r:
                    if r.status!=206 or not r.headers.get("Content-Range","").startswith(f"bytes {offset}-"):raise ValueError("server ignored range")
                    data=r.read()
                if len(data)!=size:raise ValueError("truncated range")
                return data
            except Exception:
                if attempt==2:raise
    prefix=fetch(0,20)
    if prefix[:7]!=b"Rar!\x1a\x07\x00":raise ValueError("only RAR4 supported")
    pos=20;rows=[]
    while len(rows)<a.count:
        h=fetch(pos,7);crc,typ,flags,n=struct.unpack("<HBHH",h)
        h=fetch(pos,n)
        if typ==0x74:
            pack,unpack,host,fcrc,ftime,ver,method,nlen,attr=struct.unpack_from("<IIBIIBBHI",h,7)
            if flags & 0x100:raise ValueError("large RAR member unsupported")
            if flags & 0x10:raise ValueError("solid RAR unsupported")
            name=h[32:32+nlen].decode().replace("\\","/")
            if name.lower().endswith(".avi"):
                rows.append({"name":name,"header_offset":pos,"data_offset":pos+n,"packed":pack,"unpacked":unpack,"header_hex":h.hex()})
                print("INDEX",name,pack,flush=True)
            pos+=n+pack
        elif typ==0x7b:break
        else:
            extra=struct.unpack_from("<I",h,7)[0] if flags & 0x8000 else 0
            pos+=n+extra
    (root/"index.json").write_text(json.dumps({"url":a.url,"members":rows},indent=2))
    for row in rows:
        dest=root/(Path(row["name"]).stem+".rar")
        if not dest.exists():
            pieces=[(row["data_offset"]+i,min(2*1024*1024,row["packed"]-i)) for i in range(0,row["packed"],2*1024*1024)]
            with concurrent.futures.ThreadPoolExecutor(4) as ex:
                chunks=list(ex.map(lambda p:fetch(*p),pieces))
            dest.write_bytes(prefix+bytes.fromhex(row["header_hex"])+b"".join(chunks))
        print("DOWNLOADED",row["name"],dest.stat().st_size,flush=True)
if __name__=="__main__":main()
