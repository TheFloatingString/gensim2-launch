"""
Pull files from a Modal volume and parse chat logs into structured JSON.

Usage:
    # Download all files to disk
    uv run python pull_volume.py

    # Parse chat logs from longhorizon_bottomup folders, print JSON
    uv run python pull_volume.py --parse-chats

    # Parse and write to a file
    uv run python pull_volume.py --parse-chats --out results.json

    # Narrow to a specific folder prefix
    uv run python pull_volume.py --parse-chats --prefix keypoint_pipeline_longhorizon_bottomup
"""

import argparse
import ast
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import modal


VOLUME_NAME = "gensim2-outputs"
DEFAULT_OUTPUT_DIR = "./gensim2-outputs-local"
CHAT_LOG_SUFFIX = "_chat_log.txt"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def walk_and_download(vol: modal.Volume, remote_path: str, local_base: Path, after: float | None = None) -> int:
    entries = list(vol.listdir(remote_path))
    downloaded = 0
    for entry in entries:
        rel = entry.path.lstrip("/")
        local_path = local_base / rel
        if entry.type.name == "DIRECTORY":
            local_path.mkdir(parents=True, exist_ok=True)
            downloaded += walk_and_download(vol, entry.path, local_base, after)
        else:
            if after is not None and entry.mtime < after:
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  {entry.path} -> {local_path}")
            with open(local_path, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            downloaded += 1
    return downloaded


# ---------------------------------------------------------------------------
# Chat log parsing
# ---------------------------------------------------------------------------

def list_chat_logs(vol: modal.Volume, prefix: str, after: float | None = None) -> list[str]:
    paths = []
    for entry in vol.listdir("/"):
        folder = entry.path.lstrip("/")
        if not folder.startswith(prefix):
            continue
        try:
            for child in vol.listdir(entry.path):
                if child.path.endswith(CHAT_LOG_SUFFIX):
                    if after is not None and child.mtime < after:
                        continue
                    paths.append(child.path)
            time.sleep(0.3)  # stay under VolumeListFiles rate limit
        except Exception as exc:
            print(f"[warn] could not list {entry.path}: {exc}", file=sys.stderr)
    return sorted(paths)


def read_file(vol: modal.Volume, path: str) -> str:
    return b"".join(vol.read_file(path)).decode("utf-8", errors="replace")


def parse_chat_log(text: str) -> dict | None:
    """
    Extract the long-horizon task dict and sub-task dicts from the last
    '>>> Answer:' section in the chat log.
    """
    matches = list(re.finditer(r">>> Answer:", text))
    if not matches:
        return None

    section = text[matches[-1].start():]
    raw_blocks = re.findall(r"```python\s*(\{.*?\})\s*```", section, re.DOTALL)
    if not raw_blocks:
        return None

    dicts = []
    for raw in raw_blocks:
        try:
            dicts.append(ast.literal_eval(raw.strip()))
        except Exception as exc:
            print(f"[warn] could not parse block: {exc}", file=sys.stderr)

    if not dicts:
        return None

    return {"task": dicts[0], "subtasks": dicts[1:]}


def parse_chats(vol: modal.Volume, prefix: str, out_path: str | None, after: float | None = None):
    print(f"[*] Listing chat logs with prefix '{prefix}' ...", file=sys.stderr)
    paths = list_chat_logs(vol, prefix, after)
    print(f"[*] Found {len(paths)} chat log(s)", file=sys.stderr)

    results = []
    for path in paths:
        print(f"    reading {path}", file=sys.stderr)
        try:
            text = read_file(vol, path)
        except Exception as exc:
            print(f"[warn] could not read {path}: {exc}", file=sys.stderr)
            continue

        parsed = parse_chat_log(text)
        if parsed is None:
            print(f"[warn] no task section found in {path}", file=sys.stderr)
            continue

        results.append({"source": path, **parsed})

    output = json.dumps(results, indent=2)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[*] Wrote {len(results)} record(s) to {out_path}", file=sys.stderr)
    else:
        print(output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pull files from a Modal Volume and/or parse chat logs.")
    parser.add_argument("--volume", default=VOLUME_NAME, help="Modal volume name")
    parser.add_argument("--parse-chats", action="store_true", help="Parse chat logs into JSON instead of downloading files")
    parser.add_argument("--prefix", default="keypoint_pipeline_longhorizon_bottomup", help="Folder prefix filter for --parse-chats")
    parser.add_argument("--out", default=None, help="Write JSON output to this file (default: stdout)")
    parser.add_argument("--remote-path", default="/", help="Remote path to start from when downloading (default: /)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Local directory for downloaded files")
    parser.add_argument("--after", default=None, help="Only include files modified after this date (e.g. 2026-07-01)")
    args = parser.parse_args()

    after_ts = None
    if args.after:
        after_ts = datetime.strptime(args.after, "%Y-%m-%d").timestamp()

    vol = modal.Volume.from_name(args.volume)

    if args.parse_chats:
        parse_chats(vol, args.prefix, args.out, after_ts)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Volume   : {args.volume}")
        print(f"Remote   : {args.remote_path}")
        print(f"Local    : {output_dir.resolve()}")
        print()
        try:
            total = walk_and_download(vol, args.remote_path, output_dir, after_ts)
        except modal.exception.NotFoundError:
            print(f"Error: volume '{args.volume}' not found.", file=sys.stderr)
            sys.exit(1)
        print(f"\nDone. Downloaded {total} file(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
