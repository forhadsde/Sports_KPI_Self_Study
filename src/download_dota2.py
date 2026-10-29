"""Download and verify a pinned Betty Dota 2 snapshot inside this project.

Author: Md Forhadul Islam. Source data: wolframko/betty-dota2.
No Hugging Face cache or authentication token is required.
"""
import argparse  # Read the requested revision and bounded worker count.
from concurrent.futures import ThreadPoolExecutor, as_completed  # Transfer independent files concurrently and collect finished work.
from datetime import datetime, timezone  # Record unambiguous UTC acquisition times.
import hashlib  # Check file integrity using SHA256 fingerprints.
import json  # Read source inventories and save acquisition evidence.
from pathlib import Path  # Keep destinations safely inside the project.
import time  # Pace requests and apply retry delays with a monotonic clock.
import threading  # Coordinate rate limiting across download workers.
import re  # Parse the source host's optional rate limit reset field.
from urllib.parse import quote  # Encode source paths safely in request URLs.

import requests  # Stream public files over HTTPS without a global dataset cache.

ROOT = Path(__file__).resolve().parents[1]  # Find the project containing this script.
DEST = ROOT / "data/raw/betty-dota2"  # Preserve the publisher's directory layout locally.
RESULT = ROOT / "outputs/results/D_download_manifest.json"  # Keep acquisition provenance separate from raw data.
REPO = "wolframko/betty-dota2"  # Identify the credited source dataset.
LOCAL = threading.local()  # Give each worker its own reusable HTTP session.
RATE_LOCK = threading.Lock()  # Prevent workers from racing when reserving request times.
NEXT_REQUEST = 0.0  # Track the next permitted resolver request in monotonic seconds.
COOLDOWN_UNTIL = 0.0  # Share any host requested cooldown across workers.


def wait_for_resolver():
    global NEXT_REQUEST  # Update the shared reservation rather than a local copy.
    while True:  # Recheck the shared cooldown after each bounded pause.
        with RATE_LOCK:  # Reserve one request time atomically.
            now = time.monotonic()  # Avoid changes caused by wall clock adjustments.
            delay = max(NEXT_REQUEST, COOLDOWN_UNTIL) - now  # Respect both ordinary pacing and host cooldowns.
            if delay <= 0:  # A nonpositive delay means a request can start now.
                NEXT_REQUEST = now + 0.14  # Space resolver requests even when workers finish together.
                return  # Release this worker to perform its transfer.
        time.sleep(min(delay, 1))  # Wait outside the lock so other workers can update shared state.


def session():
    if not hasattr(LOCAL, "session"):  # Initialise each worker's session only once.
        LOCAL.session = requests.Session()  # Reuse connections without sharing mutable session state across workers.
    return LOCAL.session  # Supply this worker's established HTTP connection pool.


def get_json(url):
    response = requests.get(url, timeout=60)  # Read a bounded source metadata page.
    response.raise_for_status()  # Stop rather than treating an error response as an inventory.
    return response.json(), response.links.get("next", {}).get("url")  # Return both entries and the optional next page.


def digest(path):
    h = hashlib.sha256()  # Start a reproducible content fingerprint.
    with path.open("rb") as stream:  # Read binary data without text decoding.
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):  # Hash bounded chunks until end of file.
            h.update(chunk)  # Include every observed byte in the fingerprint.
    return h.hexdigest()  # Store a portable hexadecimal representation.


def download(entry, revision):
    relative = Path(entry["path"])  # Preserve the source file's relative name.
    target = DEST / relative  # Resolve that name within the approved raw folder.
    if not target.resolve().is_relative_to(DEST.resolve()):  # Reject source names that escape the destination.
        raise ValueError("Unsafe source path")  # Never write a remote supplied path outside the dataset folder.
    target.parent.mkdir(parents=True, exist_ok=True)  # Create any source subfolder without deleting existing work.
    expected = entry.get("lfs", {}).get("oid")  # Read the publisher's SHA256 when available for an LFS object.
    if target.exists() and target.stat().st_size == entry["size"]:  # Reuse a completed file only after size and hash checks.
        checksum = digest(target)  # Verify content rather than trusting a filename alone.
        if not expected or checksum == expected:  # Some small repository files have no publisher SHA256 field.
            return dict(path=entry["path"], bytes=entry["size"], sha256=checksum,
                        publisher_sha256=expected, verified=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(entry['path'])}?download=true"
    partial = target.with_suffix(target.suffix + ".part")  # Keep incomplete transfers distinct from verified files.
    for attempt in range(10):  # Bound retries so a persistent failure eventually reaches the manifest.
        try:  # Retry network and integrity failures using the same source revision.
            h = hashlib.sha256()  # Hash bytes while streaming to avoid a second full read.
            wait_for_resolver()  # Respect shared pacing before starting the request.
            with session().get(url, stream=True, timeout=(30, 120)) as response:  # Bound connection and idle read times.
                if response.status_code == 429:  # The host is asking clients to slow down.
                    retry = response.headers.get("Retry-After", "")  # Prefer an explicit retry interval when present.
                    if retry.isdigit():  # Interpret a seconds value without assuming every header is numeric.
                        delay = float(retry) + 2  # Add a small margin beyond the requested interval.
                    else:  # Some responses use a reset field instead.
                        match = re.search(r"t=(\d+)", response.headers.get("RateLimit", ""))  # Extract its stated delay when available.
                        delay = float(match.group(1)) + 2 if match else 60  # Fall back to a conservative cooldown.
                    global COOLDOWN_UNTIL  # Share the host request with every worker.
                    with RATE_LOCK:  # Prevent concurrent cooldown updates from overwriting a longer wait.
                        COOLDOWN_UNTIL = max(COOLDOWN_UNTIL, time.monotonic() + delay)  # Preserve the latest required resume time.
                    print(f"Host rate limit: respecting {delay:.0f}s cooldown", flush=True)
                response.raise_for_status()  # Reject HTTP errors before interpreting their bodies as data.
                with partial.open("wb") as stream:  # Restart a failed partial transfer rather than appending unverified bytes.
                    for chunk in response.iter_content(1024 * 1024):  # Stream one MiB at a time to bound memory.
                        if chunk:  # Ignore empty keep alive chunks.
                            stream.write(chunk)  # Preserve the downloaded bytes unchanged.
                            h.update(chunk)  # Fingerprint exactly the bytes written to disk.
            checksum = h.hexdigest()  # Finalise the downloaded content hash.
            if partial.stat().st_size != entry["size"]:  # A truncated transfer cannot count as complete.
                raise ValueError("Download size mismatch")  # Retry a size failure within the same bounded loop.
            if expected and checksum != expected:  # A publisher hash gives stronger integrity evidence than size alone.
                raise ValueError("Publisher SHA256 mismatch")  # Reject altered or corrupt content.
            partial.replace(target)  # Promote the file only after all available checks pass.
            return dict(path=entry["path"], bytes=entry["size"], sha256=checksum,
                        publisher_sha256=expected, verified=True)
        except (requests.RequestException, ValueError):  # Handle recoverable transfer or integrity errors.
            if attempt == 9:  # The tenth unsuccessful attempt exhausts the retry budget.
                raise  # Let the caller record the actual failed file.
            time.sleep(min(2 ** attempt, 16))  # Increase retry spacing without an unbounded per attempt backoff.


def main():
    parser = argparse.ArgumentParser()  # Expose bounded parallel downloading as a terminal command.
    parser.add_argument("--workers", type=int, default=16)  # Limit simultaneous transfers to avoid uncontrolled request load.
    parser.add_argument("--revision", default="b20e01577af2f4f5dffcdc4f1d5bad5c3807abca")  # Reproduce the studied snapshot even if the publisher later updates it.
    args = parser.parse_args()  # Validate options before contacting the source.
    RESULT.parent.mkdir(parents=True, exist_ok=True)  # Ensure provenance can be saved before transfers begin.
    DEST.mkdir(parents=True, exist_ok=True)  # Keep every raw file inside this project.
    if RESULT.exists():  # A resumed acquisition already has a pinned revision.
        previous = json.loads(RESULT.read_text(encoding="utf-8"))  # Read that historical identity before any new request.
        revision = previous["revision"]  # Resume the same historical acquisition.
        assert revision == args.revision, "Use a separate acquisition folder for a different revision."  # Never mix snapshots under one provenance record.
    else:
        revision = args.revision  # A fresh checkout uses the same pinned source as this study.
    entries = []  # Collect the complete remote inventory before counting expected bytes.
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}?recursive=true&limit=1000"
    while url:  # Follow all source inventory pages rather than silently stopping at one page.
        page, url = get_json(url)  # Read entries and the next page link together.
        entries.extend(row for row in page if row["type"] == "file")  # Ignore directory nodes in file counts.
    manifest = dict(source=f"https://huggingface.co/datasets/{REPO}", revision=revision,
                    author="Md Forhadul Islam", dataset_publisher="wolframko",
                    declared_licence="MIT", download_date=datetime.now(timezone.utc).date().isoformat(),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    expected_files=len(entries), expected_bytes=sum(e["size"] for e in entries),
                    status="downloading", files=[], failures=[])
    def save():  # Checkpoint acquisition progress without leaving a half written JSON file.
        temporary = RESULT.with_suffix(".json.tmp")  # Write to a sibling temporary file first.
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # Preserve complete structured progress.
        temporary.replace(RESULT)  # Atomically replace the previous manifest after writing succeeds.
    save()  # Record the source and expected inventory before downloading.
    print(f"Pinned revision {revision}: {len(entries)} files, {manifest['expected_bytes'] / 1e9:.2f} GB", flush=True)
    last_report = 0  # Track the most recent concise progress update.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:  # Bound simultaneous transfer workers.
        futures = {executor.submit(download, e, revision): e["path"] for e in entries}  # Associate each submitted transfer with its source identity.
        for future in as_completed(futures):  # Process finished files without waiting for submission order.
            try:  # Preserve successful files even if a different transfer fails.
                manifest["files"].append(future.result())  # Add only returned verified transfer records.
            except Exception as exc:  # Keep the failed source and reason inspectable.
                manifest["failures"].append(dict(path=futures[future], error=str(exc)))  # Never hide an incomplete download behind a success message.
            if time.monotonic() - last_report > 15:  # Avoid printing one message per source file.
                save()  # Preserve a resumable progress record during a long acquisition.
                total = sum(f["bytes"] for f in manifest["files"])  # Count only verified completed bytes.
                print(f"Verified {len(manifest['files'])}/{len(entries)} files, {total / 1e9:.2f} GB; failures {len(manifest['failures'])}", flush=True)
                last_report = time.monotonic()  # Start the next reporting interval.
    manifest["files"].sort(key=lambda f: f["path"])  # Make manifest ordering independent of thread completion order.
    manifest["status"] = "complete" if not manifest["failures"] else "incomplete"  # Require every transfer to succeed before declaring completion.
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()  # Record when this acquisition attempt ended.
    manifest["verified_bytes"] = sum(f["bytes"] for f in manifest["files"])  # Preserve the exact completed volume.
    save()  # Write the final inventory and any failures.
    print(f"Download {manifest['status']}: {len(manifest['files'])} verified files", flush=True)
    if manifest["failures"]:  # Allow terminal workflows to detect an incomplete acquisition.
        raise SystemExit(1)  # Return a nonzero exit status rather than a false success.


if __name__ == "__main__":  # Importing helpers must not start network transfers.
    main()  # Download only when deliberately executed.
