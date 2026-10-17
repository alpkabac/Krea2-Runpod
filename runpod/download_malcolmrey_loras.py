"""Download selected malcolmrey/krea2 LoRAs from Hugging Face, newest versions only.

Subjects come from data/malcolmrey_subjects.txt (or --subject).  For each subject
only the newest --keep versions are downloaded (all variants of a kept version,
e.g. _onetrainer / _prodigy, are kept).  Existing files of the right size are
skipped, so re-running is cheap.  Standard library only; works on Linux and Windows.

Examples:
  python runpod/download_malcolmrey_loras.py --dry-run
  python runpod/download_malcolmrey_loras.py --out /workspace/models/loras
  python runpod/download_malcolmrey_loras.py --out D:/loras --keep 2 --subject taylorswift
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "malcolmrey/krea2"
PREFIX = "krea2_"
DEFAULT_SUBJECTS = Path(__file__).resolve().parent.parent / "data" / "malcolmrey_subjects.txt"
HEADERS = {"User-Agent": "krea2-lora-downloader/1.0"}
if os.environ.get("HF_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120)


def list_repo_files():
    """Return [{'path', 'size'}] for every .safetensors in the repo (follows pagination)."""
    url = f"https://huggingface.co/api/models/{REPO}/tree/main?limit=1000"
    files = []
    while url:
        with request(url) as response:
            files += json.load(response)
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return [f for f in files if f["type"] == "file" and f["path"].lower().endswith(".safetensors")]


def subject_and_version(path):
    stem = path[len(PREFIX):-len(".safetensors")]
    subject = stem.split("_")[0]
    match = re.search(r"_v(\d+)", stem)
    return subject, int(match.group(1)) if match else 0


def select(files, subjects, keep):
    by_subject = defaultdict(lambda: defaultdict(list))
    for f in files:
        if not f["path"].startswith(PREFIX):
            continue
        subject, version = subject_and_version(f["path"])
        by_subject[subject][version].append(f)
    chosen, missing = [], []
    for subject in subjects:
        versions = by_subject.get(subject)
        if not versions:
            missing.append(subject)
            continue
        for version in sorted(versions, reverse=True)[:keep]:
            chosen += sorted(versions[version], key=lambda f: f["path"])
    return chosen, missing


def download(item, out_dir):
    name, size = item["path"], item["size"]
    target = out_dir / name
    if target.is_file() and target.stat().st_size == size:
        return name, "skipped"
    url = f"https://huggingface.co/{REPO}/resolve/main/{urllib.parse.quote(name)}"
    part = target.with_suffix(".safetensors.part")
    for attempt in range(1, 4):
        try:
            with request(url) as response, open(part, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if part.stat().st_size != size:
                raise IOError(f"size mismatch: {part.stat().st_size} != {size}")
            os.replace(part, target)
            return name, "downloaded"
        except Exception as error:  # retry transient network errors
            if attempt == 3:
                return name, f"FAILED: {error}"
            time.sleep(2 * attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="./krea2_loras", help="download folder (on a RunPod Pod with a network volume: /workspace/models/loras)")
    parser.add_argument("--subjects-file", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subject", action="append", default=[], help="extra subject name, repeatable")
    parser.add_argument("--keep", type=int, default=3, help="newest versions to keep per subject (default 3)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    parser.add_argument("--names-only", action="store_true", help="print just the filenames, one per line (implies --dry-run)")
    args = parser.parse_args()

    subjects = []
    path = Path(args.subjects_file)
    if path.is_file():
        subjects += [l.strip().lower() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    subjects += [s.strip().lower() for s in args.subject]
    subjects = list(dict.fromkeys(subjects))
    if not subjects:
        sys.exit("No subjects: fill data/malcolmrey_subjects.txt or pass --subject NAME")

    chosen, missing = select(list_repo_files(), subjects, args.keep)
    if args.names_only:
        for f in chosen:
            print(f["path"])
        return
    total = sum(f["size"] for f in chosen)
    print(f"{len(subjects)} subjects -> {len(chosen)} files, {total / 1e9:.2f} GB (keeping newest {args.keep} versions each)")
    if missing:
        print("Not found in repo:", ", ".join(missing))
    if args.dry_run:
        for f in chosen:
            print(f"  {f['path']}  {f['size'] / 1e6:.0f} MB")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, status in pool.map(lambda item: download(item, out_dir), chosen):
            print(f"{status:10} {name}")
            failed += status.startswith("FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
