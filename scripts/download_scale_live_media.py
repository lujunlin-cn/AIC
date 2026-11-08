"""Recover original LIVE public videos directly from the author's Box release.

The original source SHA256 inventory is authoritative. Cached PNGs are never
used as substitutes for missing original videos or PTS evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

SHARE = "hylumfu8akjhdgdd4teynsyc6ickwv1j"
FOLDER = "407721469541"
ORIGIN = "https://utexas.app.box.com"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def page(number):
    response = requests.get(f"{ORIGIN}/s/{SHARE}/folder/{FOLDER}",
                            params={"page": number}, timeout=90)
    response.raise_for_status()
    text = response.text
    marker = "Box.postStreamData = "
    if marker not in text:
        raise ValueError("public Box folder metadata is unavailable")
    payload = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
    folder = payload["/app-api/enduserapp/shared-folder"]
    if folder["pageNumber"] != number or str(folder["currentFolderID"]) != FOLDER:
        raise ValueError("Box pagination changed; refusing incomplete media inventory")
    return folder


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--original-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    expected = json.loads(a.original_manifest.read_text())["video_sha256"]
    first = page(1)
    items = list(first["items"])
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(page, range(2, first["pageCount"] + 1)):
            items.extend(result["items"])
    files = {Path(item["name"]).stem: item for item in items
             if item["type"] == "file" and item["name"].endswith(".mp4")}
    if set(files) != set(expected):
        raise ValueError(f"original/public source inventories differ: missing={len(set(expected)-set(files))}, extra={len(set(files)-set(expected))}")
    total = sum(int(item["itemSize"]) for item in files.values())
    required = sum(int(item["itemSize"]) for source,item in files.items()
                   if not (a.output / item["name"]).exists())
    if shutil.disk_usage(a.output).free < required + 5_000_000_000:
        raise ValueError("insufficient disk space including 5 GB shutdown/checkpoint reserve")
    inventory = {"status": "downloading", "expected_sources": len(expected),
                 "public_release_bytes": total, "original_manifest_sha256": sha(a.original_manifest),
                 "public_folder": f"{ORIGIN}/s/{SHARE}/folder/{FOLDER}", "sources": {},
                 "dataset_use_review_managed_by_user": True}
    started = time.monotonic()

    def download(source):
        item = files[source]
        target = a.output / item["name"]
        if target.exists():
            digest = sha(target)
            if digest != expected[source]:
                raise ValueError("existing original source hash differs: " + source)
            return source, {"path": str(target), "sha256": digest, "bytes": target.stat().st_size,
                            "reused": True, "box_file_id": str(item["id"])}
        url = f"{ORIGIN}/index.php?rm=box_download_shared_file&shared_name={SHARE}&file_id=f_{item['id']}"
        partial = target.with_suffix(".mp4.part")
        for attempt in range(6):
            try:
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if "video" not in response.headers.get("Content-Type", ""):
                        raise ValueError("download did not return original video bytes")
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                digest = sha(partial)
                if digest != expected[source]:
                    raise ValueError("downloaded source SHA256 differs from original: " + source)
                partial.replace(target)
                return source, {"path": str(target), "sha256": digest, "bytes": target.stat().st_size,
                                "reused": False, "box_file_id": str(item["id"]), "public_url": url}
            except Exception as exc:
                if attempt == 5:
                    return source, {"error": type(exc).__name__ + ": " + str(exc)}
                time.sleep(min(2 ** attempt, 16))

    def save():
        tmp = a.report.with_suffix(".tmp")
        tmp.write_text(json.dumps(inventory, indent=2) + "\n")
        tmp.replace(a.report)

    save()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(download, source) for source in sorted(expected)]
        for n, future in enumerate(as_completed(futures), 1):
            source, result = future.result()
            inventory["sources"][source] = result
            if n % 10 == 0 or n == len(futures):
                inventory.update(completed_sources=n, elapsed_seconds=time.monotonic()-started,
                                 errors=sum("error" in value for value in inventory["sources"].values()))
                save()
                print(json.dumps({k:inventory[k] for k in ("completed_sources","elapsed_seconds","errors")}), flush=True)
    inventory["status"] = "verified" if not inventory["errors"] else "incomplete_explicit_errors"
    save()
    return 0 if not inventory["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
