"""Fetch the attributed, pinned hero dictionary needed for correct team joins."""
import hashlib  # Verify that future downloads reproduce the same dictionary bytes.
import json  # Record the exact source and fingerprint in the result ledger.
from pathlib import Path  # Keep every downloaded file inside the project.
import requests  # Fetch the public OpenDota source files over HTTPS.

ROOT = Path(__file__).resolve().parents[1]  # Find the study from this module's location.


def download():
    """Use a saved source revision when available; never guess hero team membership."""
    evidence = ROOT / "outputs/results/D_hero_constants_source.json"  # Preserve the dictionary's provenance separately from match data.
    previous = json.loads(evidence.read_text()) if evidence.exists() else {}  # Reuse recorded metadata when reproducing the study.
    revision = previous.get("revision")  # Prefer the already pinned repository version.
    if not revision:  # The first acquisition must resolve a version rather than relying permanently on master.
        response = requests.get("https://api.github.com/repos/odota/dotaconstants/commits?path=build/heroes.json&per_page=1", timeout=30)  # Find the latest published dictionary commit.
        response.raise_for_status()  # Do not write an error page as a dictionary.
        revision = response.json()[0]["sha"]  # Pin the commit that supplied the file.
    url = f"https://raw.githubusercontent.com/odota/dotaconstants/{revision}/build/heroes.json"  # Construct a version specific source URL.
    response = requests.get(url, timeout=30)  # Download the named dictionary version.
    response.raise_for_status()  # Stop clearly if the source becomes unavailable.
    fingerprint = hashlib.sha256(response.content).hexdigest()  # Calculate a content fingerprint before using the file.
    if previous.get("sha256"):  # A recorded study must not silently adopt changed identifiers or names.
        assert fingerprint == previous["sha256"], "Hero dictionary changed; review explicitly before updating."  # Require exact content reproduction.
    folder = ROOT / "data/raw/dota_constants"  # Keep third party inputs excluded from Git with other raw data.
    folder.mkdir(parents=True, exist_ok=True)  # Create only the authorised local destination.
    (folder / "heroes.json").write_bytes(response.content)  # Save original bytes so their fingerprint remains valid.
    licence_url = f"https://raw.githubusercontent.com/odota/dotaconstants/{revision}/LICENSE"  # Attribute the same source revision's licence.
    licence = requests.get(licence_url, timeout=30)  # Preserve its actual permission text.
    licence.raise_for_status()  # Do not claim a licence was checked if retrieval failed.
    (folder / "LICENSE").write_text(licence.text, encoding="utf-8")  # Keep the upstream attribution with the local dictionary.
    record = dict(source=url, revision=revision, publisher="The OpenDota Project and contributors", download_date="2026-10-29", sha256=fingerprint, licence="MIT", licence_url=licence_url)  # Record verified provenance rather than an invented citation.
    evidence.parent.mkdir(parents=True, exist_ok=True)  # Ensure the evidence folder is available on a fresh checkout.
    evidence.write_text(json.dumps(record, indent=2), encoding="utf-8")  # Save the reproducible source record.
    print("Pinned hero dictionary verified", flush=True)  # Report success without printing unnecessary data.
    return record  # Allow notebook code to display the provenance.


if __name__ == "__main__":  # Fetch only when deliberately run, not when imported.
    download()  # Acquire the identity dictionary and its licence.
