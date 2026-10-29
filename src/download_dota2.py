"""Download and verify a pinned Betty Dota 2 snapshot inside this project.

Author: Md Forhadul Islam. Source data: wolframko/betty-dota2.
No Hugging Face cache or authentication token is required.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import threading
import re
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/betty-dota2"
RESULT = ROOT / "outputs/results/D_download_manifest.json"
REPO = "wolframko/betty-dota2"
LOCAL = threading.local()
RATE_LOCK = threading.Lock()
NEXT_REQUEST = 0.0
COOLDOWN_UNTIL = 0.0


def wait_for_resolver():
    global NEXT_REQUEST
    while True:
        with RATE_LOCK:
            now = time.monotonic()
            delay = max(NEXT_REQUEST, COOLDOWN_UNTIL) - now
            if delay <= 0:
                NEXT_REQUEST = now + 0.14
                return
        time.sleep(min(delay, 1))


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def get_json(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json(), response.links.get("next", {}).get("url")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(entry, revision):
    relative = Path(entry["path"])
    target = DEST / relative
    if not target.resolve().is_relative_to(DEST.resolve()):
        raise ValueError("Unsafe source path")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get("lfs", {}).get("oid")
    if target.exists() and target.stat().st_size == entry["size"]:
        checksum = digest(target)
        if not expected or checksum == expected:
            return dict(path=entry["path"], bytes=entry["size"], sha256=checksum,
                        publisher_sha256=expected, verified=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(entry['path'])}?download=true"
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(10):
        try:
            h = hashlib.sha256()
            wait_for_resolver()
            with session().get(url, stream=True, timeout=(30, 120)) as response:
                if response.status_code == 429:
                    retry = response.headers.get("Retry-After", "")
                    if retry.isdigit():
                        delay = float(retry) + 2
                    else:
                        match = re.search(r"t=(\d+)", response.headers.get("RateLimit", ""))
                        delay = float(match.group(1)) + 2 if match else 60
                    global COOLDOWN_UNTIL
                    with RATE_LOCK:
                        COOLDOWN_UNTIL = max(COOLDOWN_UNTIL, time.monotonic() + delay)
                    print(f"Host rate limit: respecting {delay:.0f}s cooldown", flush=True)
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            h.update(chunk)
            checksum = h.hexdigest()
            if partial.stat().st_size != entry["size"]:
                raise ValueError("Download size mismatch")
            if expected and checksum != expected:
                raise ValueError("Publisher SHA256 mismatch")
            partial.replace(target)
            return dict(path=entry["path"], bytes=entry["size"], sha256=checksum,
                        publisher_sha256=expected, verified=True)
        except (requests.RequestException, ValueError):
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 16))


def main():
    parser = argparse.ArgumentParser()  # Expose bounded parallel downloading as a terminal command.
    parser.add_argument("--workers", type=int, default=16)  # Limit simultaneous transfers to avoid uncontrolled request load.
    parser.add_argument("--revision", default="b20e01577af2f4f5dffcdc4f1d5bad5c3807abca")  # Reproduce the studied snapshot even if the publisher later updates it.
    args = parser.parse_args()  # Validate options before contacting the source.
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    if RESULT.exists():
        previous = json.loads(RESULT.read_text(encoding="utf-8"))
        revision = previous["revision"]  # Resume the same historical acquisition.
        assert revision == args.revision, "Use a separate acquisition folder for a different revision."  # Never mix snapshots under one provenance record.
    else:
        revision = args.revision  # A fresh checkout uses the same pinned source as this study.
    entries = []
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}?recursive=true&limit=1000"
    while url:
        page, url = get_json(url)
        entries.extend(row for row in page if row["type"] == "file")
    manifest = dict(source=f"https://huggingface.co/datasets/{REPO}", revision=revision,
                    author="Md Forhadul Islam", dataset_publisher="wolframko",
                    declared_licence="MIT", download_date=datetime.now(timezone.utc).date().isoformat(),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    expected_files=len(entries), expected_bytes=sum(e["size"] for e in entries),
                    status="downloading", files=[], failures=[])
    def save():
        temporary = RESULT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(RESULT)
    save()
    print(f"Pinned revision {revision}: {len(entries)} files, {manifest['expected_bytes'] / 1e9:.2f} GB", flush=True)
    last_report = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, e, revision): e["path"] for e in entries}
        for future in as_completed(futures):
            try:
                manifest["files"].append(future.result())
            except Exception as exc:
                manifest["failures"].append(dict(path=futures[future], error=str(exc)))
            if time.monotonic() - last_report > 15:
                save()
                total = sum(f["bytes"] for f in manifest["files"])
                print(f"Verified {len(manifest['files'])}/{len(entries)} files, {total / 1e9:.2f} GB; failures {len(manifest['failures'])}", flush=True)
                last_report = time.monotonic()
    manifest["files"].sort(key=lambda f: f["path"])
    manifest["status"] = "complete" if not manifest["failures"] else "incomplete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["verified_bytes"] = sum(f["bytes"] for f in manifest["files"])
    save()
    print(f"Download {manifest['status']}: {len(manifest['files'])} verified files", flush=True)
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
