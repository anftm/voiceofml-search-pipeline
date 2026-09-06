#!/usr/bin/env python3
"""Plan a deterministic, page-weighted PDF asset build queue."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from . import pdf_assets
except ImportError:
    import pdf_assets


MAX_GITHUB_MATRIX_SHARDS = 240


def source_path(item: dict, source_dir: Path | None, assets_repo: str) -> Path:
    if item.get("source_kind") == "generated":
        return Path(hf_hub_download(item["reader_assets_repo"], item["reader_assets_path"],
                                    repo_type="dataset", revision=item["reader_assets_revision"],
                                    token=os.environ.get("HF_TOKEN")))
    if source_dir:
        return source_dir / item["repo"] / item["path"]
    return Path(hf_hub_download(item["repo"], item["path"], repo_type="dataset",
                                revision=item["source_revision"], token=os.environ.get("HF_TOKEN")))


def plan(records: list[dict], source_dir: Path | None, assets_repo: str, shard_count: int,
         workers: int = 8) -> dict:
    if shard_count != 18:
        raise ValueError("ordinary PDF shard count must be 18")

    def inspect(item: dict) -> dict:
        source = None
        try:
            source = source_path(item, source_dir, assets_repo)
            pages = pdf_assets._pages(source, str(item.get("extension") or "pdf"))
            if pages < 1:
                raise ValueError(f"invalid page count for {item['key']}")
            source_sha, source_bytes = pdf_assets.digest(source)
            classification = pdf_assets.classify_pdf(source, pages)
            return {**item, "page_count": pages, "source_sha256": source_sha,
                    "source_bytes": source_bytes, "classification": classification,
                    "decision_profile": pdf_assets.PDF_DECISION_PROFILE}
        except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError):
            source_sha = ""
            source_bytes = int(item.get("source_bytes") or 0)
            if source is not None:
                try:
                    source_sha, source_bytes = pdf_assets.digest(source)
                except OSError:
                    pass
            return {**item, "page_count": 0, "source_sha256": source_sha,
                    "source_bytes": source_bytes, "classification": "failed",
                    "decision_profile": pdf_assets.PDF_DECISION_PROFILE,
                    "status": "failed", "reason": "tool-error", "strategy": "none"}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        selected = list(executor.map(inspect, records))
    ordinary = []
    range_tasks = []
    skipped = []
    for item in selected:
        if item["classification"] == "failed":
            skipped.append(item)
            continue
        if item["classification"] == "native-text":
            skipped.append({**item, "status": "skipped", "reason": "native-text-pdf",
                            "strategy": "native-text"})
            continue
        page_limit = pdf_assets.max_pages_per_task(item)
        if item["page_count"] <= page_limit:
            ordinary.append(item)
            continue
        for start in range(1, item["page_count"] + 1, page_limit):
            end = min(item["page_count"], start + page_limit - 1)
            range_tasks.append({**item, "task_key": f"{item['key']}#pages-{start:06d}-{end:06d}",
                                "page_start": start, "page_end": end,
                                "range_page_count": end - start + 1})
    shards = pdf_assets.weighted_shards(ordinary, shard_count)
    shards.extend([[task] for task in range_tasks])
    dynamic_shard_count = len(shards)
    if dynamic_shard_count > MAX_GITHUB_MATRIX_SHARDS:
        raise ValueError(f"PDF shard count {dynamic_shard_count} exceeds GitHub Actions matrix limit {MAX_GITHUB_MATRIX_SHARDS}")
    return {"version": 1, "kind": "pdf-assets-queue", "shard_count": dynamic_shard_count,
            "shard_ids": list(range(dynamic_shard_count)),
            "ordinary_shard_count": shard_count,
            "total_records": len(selected), "total_tasks": len(ordinary) + len(range_tasks),
            "total_pages": sum(item["page_count"] for item in selected if item["page_count"] > 0),
            "skipped": skipped,
            "shards": [{"index": index, "page_count": sum(item.get("range_page_count", item["page_count"])
                                                                for item in shard),
                        "records": shard} for index, shard in enumerate(shards)]}


def pending_records(records: list[dict], manifest: dict) -> list[dict]:
    done = {}
    for key, entry in manifest.get("files", {}).items():
        if isinstance(entry, dict) and entry.get("status") in {"ready", "skipped"}:
            done[key] = entry
    pending = []
    for item in records:
        current = done.get(item["key"])
        complete = current and (
            (current.get("status") == "ready" and current.get("strategy") == "sampled-webp"
             and current.get("render_profile") == pdf_assets.PDF_PROFILE
             and current.get("decision_profile") == pdf_assets.PDF_DECISION_PROFILE)
            or (current.get("status") == "skipped" and current.get("reason") == "native-text-pdf"
                and current.get("decision_profile") == pdf_assets.PDF_DECISION_PROFILE)
        )
        if item.get("source_kind") == "generated":
            if not complete:
                pending.append(item)
            continue
        if (int(item.get("source_bytes") or 0) >= pdf_assets.LARGE_BYTES
                and not complete):
            pending.append(item)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--repo", default="")
    parser.add_argument("--source", choices=("upstream", "generated", "all"), default="all")
    parser.add_argument("--reader-assets-manifest", type=Path)
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", pdf_assets.READER_ASSETS_REPO))
    parser.add_argument("--limit", type=int, required=True, help="Total PDFs in this checkpoint")
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=18)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf-assets/queue.json"))
    args = parser.parse_args()
    records = []
    if args.source in {"upstream", "all"}:
        records.extend(pdf_assets.load_records(args.search_data, args.revisions, args.repo, "pdf"))
    if args.source in {"generated", "all"}:
        reader_assets_revision = HfApi(token=os.environ.get("HF_TOKEN")).repo_info(
            repo_id=args.assets_repo, repo_type="dataset").sha
        if not args.reader_assets_manifest:
            manifest = hf_hub_download(args.assets_repo, "manifest.json", repo_type="dataset",
                                       revision=reader_assets_revision,
                                       token=os.environ.get("HF_TOKEN"))
            args.reader_assets_manifest = Path(manifest)
        records.extend(pdf_assets.load_generated_records(
            args.reader_assets_manifest, args.assets_repo, args.repo, reader_assets_revision))
    records.sort(key=lambda item: (0 if item.get("source_extension") in {"caj", "kdh"} else 1,
                                   item["repo"], item["path"], item["source_kind"]))
    try:
        manifest_path = hf_hub_download(args.assets_repo, "pdf_manifest.json", repo_type="dataset",
                                        token=os.environ.get("HF_TOKEN"))
        pdf_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        pdf_manifest = {"files": {}}
    records = pending_records(records, pdf_manifest)
    selected = pdf_assets.queue(records, args.limit, args.checkpoint)
    planned = plan(selected, args.source_dir, args.assets_repo, args.shard_count, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(planned, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"planned {planned['total_records']} PDF asset(s) across {planned['shard_count']} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
