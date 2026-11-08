"""Reuse AVI files, or extract the requested RetargetVid members exactly once."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

RETARGET_IDS = tuple(list(range(1, 101)) + list(range(601, 701)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");tmp.replace(path)


def existing_avi(root):
    root=Path(root);found={};duplicates=[]
    for directory in (root/"video",root/"raw/video",root/"raw",root):
        if not directory.is_dir():continue
        for p in sorted(directory.iterdir()):
            if not p.is_file() or not re.fullmatch(r"\d{3,4}\.avi",p.name,re.I):continue
            n=int(p.stem)
            if n not in RETARGET_IDS:continue
            if n in found and found[n].resolve()!=p.resolve():
                if digest(found[n])!=digest(p):raise ValueError("conflicting duplicate AVI: "+str(n))
                duplicates.append(str(p))
            else:found[n]=p.resolve()
    return found,duplicates


def ensure_selected_media(root,receipt_path):
    root=Path(root).resolve();found,duplicates=existing_avi(root)
    missing=set(RETARGET_IDS)-set(found);archive=root/"video.rar"
    receipt={"requested_ids":list(RETARGET_IDS),"preexisting_AVI":len(found),"duplicate_copies_created":0,
             "preexisting_equal_byte_duplicates":duplicates,"extracted":[],"archive":str(archive) if archive.exists() else None}
    if not missing:
        receipt["action"]="reuse_existing_AVI_no_archive_unpack";write_json(receipt_path,receipt);return found
    if not archive.exists():
        raise FileNotFoundError(f"missing {len(missing)} AVI and no video.rar; no substitute videos allowed")
    import libarchive
    receipt.update(archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,action="first_extract_selected_200_only")
    output=root/"video";output.mkdir(exist_ok=True)
    seen=set();uncompressed=0
    with libarchive.file_reader(str(archive)) as stream:
        for entry in stream:
            name=PurePosixPath(str(entry.pathname).replace("\\","/"))
            if name.is_absolute() or ".." in name.parts:raise ValueError("unsafe archive member")
            if not re.fullmatch(r"\d{3,4}\.avi",name.name,re.I):continue
            n=int(Path(name.name).stem)
            if n not in missing:continue
            if n in seen:raise ValueError("duplicate selected AVI in archive")
            if shutil.disk_usage(output).free < int(entry.size)+2*(1<<30):raise ValueError("insufficient disk; no historical files deleted")
            destination=output/f"{n:03d}.AVI";temporary=destination.with_suffix(".AVI.extracting")
            if destination.exists() or temporary.exists():raise FileExistsError("preserve existing/uncertain extraction: "+str(destination))
            with temporary.open("xb") as f:
                for block in entry.get_blocks():f.write(block)
            if temporary.stat().st_size!=entry.size:raise ValueError("archive member size mismatch")
            temporary.replace(destination);found[n]=destination.resolve();seen.add(n);uncompressed+=entry.size
            receipt["extracted"].append({"video_id":f"{n:03d}","archive_member":str(name),"bytes":entry.size})
            write_json(receipt_path,receipt)
            if len(seen)%25==0:print(json.dumps({"selected_AVI_extracted":len(seen),"requested_missing":len(missing)}),flush=True)
    if set(found)!=set(RETARGET_IDS):raise ValueError("archive lacks requested AVI; actual missing IDs recorded, never substituted")
    receipt.update(extracted_bytes=uncompressed,complete=True);write_json(receipt_path,receipt)
    return found
