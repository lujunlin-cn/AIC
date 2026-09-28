"""Anonymous listing/download of a public Box shared folder.

Box embeds the folder listing as ``Box.postStreamData = {...}`` in the shared
page HTML; pages are ``?page=N`` with 20 items each.  Files download through
``index.php?rm=box_download_shared_file`` which redirects to boxcloud with
Range support.
"""
from __future__ import annotations

import json
import re
import time

from .download import proxies_for, session

POST_RE = re.compile(r"Box\.postStreamData\s*=\s*(\{.*?\});\s*</script>", re.S)


def _page(host: str, shared: str, folder_id: int, page: int, sess) -> dict:
    url = f"https://{host}/s/{shared}/folder/{folder_id}?page={page}"
    for attempt in range(5):
        try:
            r = sess.get(url, timeout=60, proxies=proxies_for(url))
            r.raise_for_status()
            m = POST_RE.search(r.text)
            if not m:
                raise ValueError("no postStreamData in Box page")
            return json.loads(m.group(1))["/app-api/enduserapp/shared-folder"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def list_folder(host: str, shared: str, folder_id: int, recursive: bool = True, path: str = "",
                sess=None) -> list[dict]:
    sess = sess or session()
    first = _page(host, shared, folder_id, 1, sess)
    pages = int(first.get("pageCount") or 1)
    items = list(first["items"])
    for p in range(2, pages + 1):
        items += _page(host, shared, folder_id, p, sess)["items"]
    out = []
    for it in items:
        rel = f"{path}{it['name']}"
        if it["type"] == "folder":
            out.append({"type": "folder", "id": it["id"], "path": rel, "size": it.get("itemSize"),
                        "files_count": it.get("filesCount")})
            if recursive:
                out += list_folder(host, shared, it["id"], True, rel + "/", sess)
        else:
            out.append({"type": "file", "id": it["id"], "path": rel, "size": it.get("itemSize"),
                        "sha1_hint": it.get("fileIDHash"), "modified": it.get("contentUpdated")})
    return out


def file_url(host: str, shared: str, file_id: int) -> str:
    return (f"https://{host}/index.php?rm=box_download_shared_file&shared_name={shared}"
            f"&file_id=f_{file_id}")
