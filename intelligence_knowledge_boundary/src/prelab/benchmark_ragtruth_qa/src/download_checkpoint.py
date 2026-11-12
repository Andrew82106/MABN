"""Anonymous, resumable download of the explicitly approved pinned checkpoint.

Downloads only the allowlist below; never loads a model or reads benchmark data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import requests


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "prelab/models/Llama-2-7b-chat-hf"
MANIFEST = ROOT / "prelab/benchmark_ragtruth_qa/model_download_manifest.json"
REPO = "NousResearch/Llama-2-7b-chat-hf"
REVISION = "351844e75ed0bcbbe3f10671b3c808d2b83894ee"
FILES = [
    "LICENSE.txt", "USE_POLICY.md", "config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.model", "tokenizer.json",
    "model.safetensors.index.json", "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    state["updated_at_utc"] = now()
    temp = MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, MANIFEST)


def verify(path, rec):
    size = path.stat().st_size
    if size != rec["expected_bytes"]:
        raise ValueError(f"Size mismatch for {path.name}: {size}")
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    actual = sha256.hexdigest()
    expected_lfs = rec["expected_lfs_sha256"]
    if expected_lfs:
        matched = actual == expected_lfs
    else:
        matched = blob.hexdigest() == rec["expected_git_blob_sha1"]
    if not matched:
        raise ValueError(f"Hash mismatch for {path.name}")
    rec.update(actual_bytes=size, actual_sha256=actual,
               actual_git_blob_sha1=None if expected_lfs else blob.hexdigest(),
               source_hash_match=True, status="verified", verified_at_utc=now())


def download(session, rec, state):
    name = rec["filename"]
    dest = TARGET / name
    partial = TARGET / ("." + name + ".partial")
    if dest.exists():
        print(f"VERIFY_EXISTING {name}", flush=True)
        verify(dest, rec)
        save(state)
        return
    expected = rec["expected_bytes"]
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError(f"Oversized partial file: {name}")
        if offset == expected:
            break
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        rec.update(status="downloading", downloaded_bytes=offset, attempt=attempt)
        state.update(current_file=name, status="downloading")
        save(state)
        print(f"DOWNLOAD {name} attempt={attempt} resume={offset}/{expected}", flush=True)
        last_report = time.monotonic()
        try:
            with session.get(url, headers=headers, stream=True, timeout=(20, 45)) as response:
                response.raise_for_status()
                if offset:
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Server did not honor resume range; partial preserved")
                elif response.status_code != 200:
                    raise RuntimeError(f"Unexpected initial status {response.status_code}")
                with partial.open("ab" if offset else "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        if offset + len(chunk) > expected:
                            raise ValueError("Download exceeded source metadata size")
                        f.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report >= 10:
                            f.flush()
                            rec["downloaded_bytes"] = offset
                            save(state)
                            print(f"PROGRESS {name} {offset}/{expected} ({100*offset/expected:.2f}%)", flush=True)
                            last_report = time.monotonic()
                    f.flush()
            if offset != expected:
                raise IOError(f"Early EOF at {offset}/{expected}")
            break
        except (requests.RequestException, IOError) as error:
            # Store no redirected/signed URL or credential-bearing exception text.
            rec.update(status="retry_pending", last_error_type=type(error).__name__,
                       downloaded_bytes=partial.stat().st_size if partial.exists() else 0)
            save(state)
            print(f"RETRY {name} {type(error).__name__}; partial retained", flush=True)
            if attempt == 6:
                raise RuntimeError(f"Download retry budget exhausted for {name}") from None
            time.sleep(min(5 * attempt, 20))
    rec.update(status="verifying", downloaded_bytes=expected)
    save(state)
    print(f"VERIFY {name}", flush=True)
    verify(partial, rec)
    os.replace(partial, dest)
    save(state)
    print(f"VERIFIED {name} sha256={rec['actual_sha256']}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    TARGET.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False  # No token, .netrc, account login, or automatic credentials.
    session.headers["User-Agent"] = "RAGTruth-checkpoint-metadata-and-download/1.0"
    info_response = session.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=(20, 45))
    info_response.raise_for_status()
    info = info_response.json()
    if info.get("sha") != REVISION or info.get("gated") is not False:
        raise RuntimeError("Pinned public repository metadata mismatch")
    siblings = {f["rfilename"]: f for f in info["siblings"]}
    records = []
    for name in FILES:
        f = siblings[name]
        records.append({"filename": name, "expected_bytes": f["size"],
                        "expected_git_blob_sha1": f["blobId"],
                        "expected_lfs_sha256": (f.get("lfs") or {}).get("sha256"),
                        "status": "pending", "downloaded_bytes": 0})
    state = {"schema_version": 1, "repo_id": REPO, "revision": REVISION,
             "target_directory": str(TARGET), "started_at_utc": now(),
             "status": "starting", "anonymous_access": True,
             "license": "Llama 2 Community License; USE_POLICY.md applies",
             "account_authorization_modified": False, "model_loaded": False,
             "download_format": "safetensors", "total_expected_bytes": sum(r["expected_bytes"] for r in records),
             "source_metadata_url": f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",
             "files": records}
    save(state)
    print(f"START total_bytes={state['total_expected_bytes']} revision={REVISION}", flush=True)
    try:
        for rec in records:
            download(session, rec, state)
        state.update(status="complete", current_file=None, completed_at_utc=now(),
                     all_files_source_hash_matched=all(r.get("source_hash_match") for r in records))
        save(state)
        print(f"COMPLETE {MANIFEST}", flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        save(state)
        print(f"FAILED {type(error).__name__}; verified files and partial download retained", flush=True)
        raise


if __name__ == "__main__":
    main()
