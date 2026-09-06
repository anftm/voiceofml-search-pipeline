#!/usr/bin/env python3
"""Merge PDF shard bundles and publish them in one atomic commit."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import sync_bucket

try:
    from . import pdf_assets
except ImportError:
    import pdf_assets


def merge_bundles(bundle_paths: list[Path], output: Path) -> list[dict]:
    grouped = {}
    output.mkdir(parents=True, exist_ok=True)
    for bundle in sorted(bundle_paths, key=lambda path: path.as_posix()):
        data = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("results"), list):
            raise ValueError(f"invalid PDF asset bundle: {bundle}")
        for result in data["results"]:
            key = result.get("key")
            if not key:
                raise ValueError(f"duplicate PDF asset result: {key}")
            grouped.setdefault(key, []).append(result)
        for source in sorted(bundle.rglob("*")):
            if not source.is_file() or source.name in {"bundle.json", pdf_assets.MANIFEST_NAME}:
                continue
            relative = source.relative_to(bundle)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise ValueError(f"conflicting PDF asset artifact: {relative}")
            else:
                shutil.copyfile(source, destination)
    results = []
    for key, entries in sorted(grouped.items()):
        ranged = [entry for entry in entries if "page_start" in entry or "page_end" in entry]
        if not ranged:
            if len(entries) != 1:
                raise ValueError(f"duplicate PDF asset result: {key}")
            results.append(entries[0])
            continue
        if len(ranged) != len(entries):
            raise ValueError(f"incomplete PDF page ranges: {key}")
        identity = ("source_sha256", "source_revision", "source_extension", "profile", "strategy",
                    "render_profile", "decision_profile")
        if any(tuple(entry.get(field) for field in identity) != tuple(entries[0].get(field) for field in identity)
               or entry.get("page_count") != entries[0].get("page_count")
               or entry.get("outline", []) != entries[0].get("outline", []) for entry in entries):
            raise ValueError(f"conflicting PDF page ranges: {key}")
        statuses = {entry.get("status") for entry in entries}
        if statuses == {"skipped"}:
            results.append({k: v for k, v in entries[0].items()
                            if k not in {"task_key", "page_start", "page_end", "range_page_count", "pages"}})
            continue
        if statuses != {"ready"}:
            raise ValueError(f"incomplete PDF page ranges: {key}")
        ordered = sorted(entries, key=lambda entry: int(entry.get("page_start", 0)))
        total = int(ordered[0].get("page_count") or ordered[0].get("pdf", {}).get("pages") or 0)
        expected = 1
        pages = []
        for entry in ordered:
            start, end = int(entry.get("page_start", 0)), int(entry.get("page_end", 0))
            if start != expected or end < start or end > total:
                raise ValueError(f"incomplete or overlapping PDF page ranges: {key}")
            entry_pages = entry.get("pages")
            if (not isinstance(entry_pages, list)
                    or [page.get("page") for page in entry_pages] != list(range(start, end + 1))):
                raise ValueError(f"incomplete PDF page ranges: {key}")
            pages.extend(entry_pages)
            expected = end + 1
        if expected != total + 1:
            raise ValueError(f"incomplete PDF page ranges: {key}")
        page_paths = [page.get("path") for page in pages]
        if None in page_paths or len(set(page_paths)) != len(page_paths):
            raise ValueError(f"conflicting PDF page artifact paths: {key}")
        result = {k: v for k, v in ordered[0].items()
                  if k not in {"task_key", "page_start", "page_end", "range_page_count", "pages", "page_manifest"}}
        result["pages"] = pages
        object_dir = pdf_assets.object_root(result["source_sha256"], key)
        manifest = {"version": 1, "kind": "pdf-pages", "source_sha256": result["source_sha256"],
                    "profile": pdf_assets.PDF_PROFILE, "pages": pages}
        if ordered[0].get("outline"):
            manifest["toc"] = ordered[0]["outline"]
        manifest_path = output / object_dir / "page-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
        manifest_sha, manifest_bytes = pdf_assets.digest(manifest_path)
        result["page_manifest"] = {"path": (object_dir / "page-manifest.json").as_posix(),
                                    "sha256": manifest_sha, "bytes": manifest_bytes}
        results.append(result)
    return results


def result_chunks(results: list[dict], size: int) -> list[list[dict]]:
    if size < 1:
        raise ValueError("publication chunk size must be positive")
    return [results[index:index + size] for index in range(0, len(results), size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", pdf_assets.READER_ASSETS_REPO))
    parser.add_argument("--chunk-results", type=int, default=1)
    parser.add_argument("--skipped", type=Path, help="JSON file with skipped entries from plan step")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as root:
        merged = Path(root)
        results = merge_bundles(args.bundles, merged)
        if args.skipped and args.skipped.is_file():
            skipped_data = json.loads(args.skipped.read_text(encoding="utf-8"))
            results.extend(skipped_data)
        manifest, _ = pdf_assets.build_publish(pdf_assets.empty_manifest(), results, merged)
        (merged / pdf_assets.MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if not args.dry_run:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required")
            api = HfApi(token=token)
            if (merged / "objects").is_dir():
                sync_bucket(str(merged), "hf://buckets/vomebook/pdf-pages",
                            include=["objects/**"], token=token, quiet=False)
            pdf_assets.publish(api, args.assets_repo, manifest, results, merged,
                               include_artifacts=False)
        print(f"published {len(results)} PDF asset(s) to {args.assets_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
