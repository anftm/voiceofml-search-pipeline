#!/usr/bin/env python3
"""Build and atomically publish the independent large-PDF asset collection."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

try:
    from .reader_assets import READER_ASSETS_REPO, decode_search_payload, relative_path, source_url
except ImportError:
    from reader_assets import READER_ASSETS_REPO, decode_search_payload, relative_path, source_url

MANIFEST_NAME = "pdf_manifest.json"
MANIFEST_VERSION = 1
MI = 1024 * 1024
MIN_BYTES = 50 * MI
LARGE_BYTES = 100 * MI
WEBP_QUALITY = int(os.environ.get("PDF_WEBP_QUALITY", "85"))
WEBP_MAX_DIMENSION = int(os.environ.get("PDF_WEBP_MAX_DIMENSION", "1800"))
SAMPLE_PAGES = int(os.environ.get("PDF_SAMPLE_PAGES", "3"))
WEBP_MAX_RATIO = float(os.environ.get("PDF_WEBP_MAX_RATIO", "0.9"))
MAX_PAGES_PER_TASK = 500
VERY_LARGE_MAX_PAGES_PER_TASK = 250
VERY_LARGE_BYTES = 500 * MI
MAX_PDF_OUTLINE_ENTRIES = 2000
MAX_PDF_OUTLINE_TITLE_CHARS = 500
MAX_PDF_OUTLINE_DEPTH = 32
PDF_PROFILE = f"pdf-pages-v3-{WEBP_QUALITY}-{WEBP_MAX_DIMENSION}-no-upscale-toc"
PDF_DECISION_PROFILE = f"pdf-large-v2-{LARGE_BYTES}-whole-book-{SAMPLE_PAGES}"
SOURCE_PROFILES = {
    "upstream": "pdf-assets-upstream-v1",
    "generated": "pdf-assets-reader-generated-v1",
}


def download_hf_source(repo: str, path: str, revision: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download
    for attempt in range(5):
        try:
            return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision, token=token))
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            delay = min(60, 2 ** attempt)
            print(f"transient PDF source download error ({status}); retrying in {delay}s", flush=True)
            time.sleep(delay)


def object_root(source_sha: str, key: str) -> Path:
    key_sha = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return Path("objects") / source_sha[:2] / source_sha / key_sha


def digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def extract_pdf_outline(path: Path, page_count: int) -> list[dict]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path, strict=False)
        outline = reader.outline
    except Exception:
        return []

    entries: list[dict] = []

    def append(items, depth: int = 0) -> None:
        for item in items or []:
            if len(entries) >= MAX_PDF_OUTLINE_ENTRIES:
                return
            if isinstance(item, list):
                append(item, min(depth + 1, MAX_PDF_OUTLINE_DEPTH))
                continue
            title = str(getattr(item, "title", "") or "").strip()
            if not title:
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                continue
            if not 1 <= page <= page_count:
                continue
            entries.append({
                "title": title[:MAX_PDF_OUTLINE_TITLE_CHARS],
                "page": page,
                "depth": min(depth, MAX_PDF_OUTLINE_DEPTH),
            })

    append(outline)
    return entries


def load_records(search_data: Path, revisions: Path, repo: str = "", extension: str = "pdf") -> list[dict]:
    records = decode_search_payload(json.loads(search_data.read_text(encoding="utf-8")))
    revision_map = json.loads(revisions.read_text(encoding="utf-8"))
    selected = []
    for record in records:
        source_repo = str(record.get("Repo") or "")
        source_extension = str(record.get("Extension") or "").lower().lstrip(".")
        revision = str(revision_map.get(source_repo) or "")
        if source_extension != extension or (repo and source_repo != repo) or not revision:
            continue
        path = relative_path(record)
        selected.append({
            "key": f"{source_repo}\0{path}", "repo": source_repo, "path": path,
            "extension": source_extension, "source_revision": revision, "source_bytes": int(record.get("Size") or 0),
            "source_url": source_url(source_repo, revision, path),
            "source_kind": "upstream", "profile": SOURCE_PROFILES["upstream"],
            "source_extension": source_extension,
        })
    selected.sort(key=lambda item: (item["repo"], item["path"]))
    return selected


def load_generated_records(manifest: Path | dict, assets_repo: str = READER_ASSETS_REPO,
                           repo: str = "", assets_revision: str = "main") -> list[dict]:
    data = manifest if isinstance(manifest, dict) else json.loads(manifest.read_text(encoding="utf-8"))
    selected = []
    for key, entry in data.get("files", {}).items():
        source_repo, separator, source_path = str(key).partition("\0")
        artifact = str(entry.get("path") or "") if isinstance(entry, dict) else ""
        artifact_bytes = entry.get("bytes") if isinstance(entry, dict) else None
        if (not separator or (repo and source_repo != repo) or entry.get("status", "ready") != "ready"
                or entry.get("reader_mode") != "pdf" or not artifact.endswith("/document.pdf")
                or not isinstance(artifact_bytes, int) or artifact_bytes < LARGE_BYTES):
            continue
        selected.append({
            "key": key, "repo": source_repo, "path": source_path,
            "source_revision": str(entry.get("source_revision") or ""),
            "source_bytes": artifact_bytes, "source_url": "", "extension": "pdf",
            "source_extension": Path(source_path).suffix.lower().lstrip("."),
            "source_kind": "generated", "profile": SOURCE_PROFILES["generated"],
            "reader_assets_repo": assets_repo, "reader_assets_path": artifact,
            "reader_assets_revision": assets_revision,
        })
    selected.sort(key=lambda item: (0 if item.get("source_extension") in {"caj", "kdh"} else 1,
                                    item["repo"], item["path"]))
    return selected


def queue(records: list[dict], limit: int = 0, checkpoint: int = 0) -> list[dict]:
    if limit < 0 or checkpoint < 0:
        raise ValueError("limit and checkpoint must be non-negative")
    if not limit:
        return records
    start = checkpoint * limit
    return records[start:start + limit]


def max_pages_per_task(item: dict) -> int:
    """Use smaller ranges for files whose size makes rendering expensive."""
    source_bytes = int(item.get("source_bytes") or 0)
    if source_bytes >= VERY_LARGE_BYTES:
        return VERY_LARGE_MAX_PAGES_PER_TASK
    return MAX_PAGES_PER_TASK


def shard_records(records: list[dict], shard_count: int, shard_index: int) -> list[dict]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid PDF asset shard")
    return [item for item in records
             if int.from_bytes(hashlib.sha256(item["key"].encode()).digest()[:8], "big") % shard_count == shard_index]


def weighted_shards(records: list[dict], shard_count: int = 10) -> list[list[dict]]:
    """Assign records to the least-loaded shard, using page counts as weight."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for item in sorted(records, key=lambda value: (-int(value.get("range_page_count", value["page_count"])),
                                                   value.get("task_key", value["key"]))):
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[shard].append(item)
        loads[shard] += int(item.get("range_page_count", item["page_count"]))
    return shards


def load_planned_shard(queue_file: Path, shard_count: int, shard_index: int) -> list[dict]:
    data = json.loads(queue_file.read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("shard_count") != shard_count:
        raise ValueError("invalid PDF asset queue")
    shards = data.get("shards")
    if not isinstance(shards, list) or len(shards) != shard_count:
        raise ValueError("invalid PDF asset shard queue")
    shard = shards[shard_index]
    if not isinstance(shard, dict) or shard.get("index") != shard_index or not isinstance(shard.get("records"), list):
        raise ValueError("invalid PDF asset shard queue")
    return shard["records"]


def _run(args: list[str], *, text: bool = False) -> str:
    result = subprocess.run(args, check=True, stdout=subprocess.PIPE if text else subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, text=text)
    return result.stdout.strip() if text else ""


def _pages(pdf: Path, extension: str = "pdf") -> int:
    if extension == "djvu":
        output = _run(["djvused", str(pdf), "-e", "n"], text=True)
        return int(output.split()[-1])
    output = _run(["pdfinfo", str(pdf)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("missing page count")


def classify_pdf(source: Path, pages: int) -> str:
    sample = sorted(set([1, max(1, (pages + 1) // 2), pages]))[:max(1, SAMPLE_PAGES)]
    text = "".join(
        _run(["pdftotext", "-f", str(page), "-l", str(page), str(source), "-"], text=True)
        for page in sample
    )
    return "native-text" if len("".join(text.split())) >= max(100, len(sample) * 40) else "scan"


def _image_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) == 24:
            return struct.unpack(">II", header[16:24])
        if not header.startswith((b"P6", b"P5")):
            raise ValueError("unsupported rendered image")
        stream.seek(0)
        tokens = []
        while len(tokens) < 4:
            line = stream.readline()
            if not line:
                break
            tokens.extend(line.split(b"#", 1)[0].split())
        if len(tokens) < 4:
            raise ValueError("invalid rendered image")
        return int(tokens[1]), int(tokens[2])


def _render(pdf: Path, page: int, directory: Path, extension: str = "pdf") -> Path:
    stem = directory / f"page-{page:06d}"
    if extension == "djvu":
        _run(["ddjvu", "-format=ppm", f"-page={page}", str(pdf), str(stem.with_suffix(".ppm"))])
    else:
        _run(["pdftocairo", "-png", "-singlefile", "-f", str(page), "-l", str(page), str(pdf), str(stem)])
    png = stem.with_suffix(".png") if extension != "djvu" else stem.with_suffix(".ppm")
    webp = stem.with_suffix(".webp")
    args = ["cwebp", "-quiet", "-q", str(WEBP_QUALITY)]
    width, height = _image_dimensions(png)
    if WEBP_MAX_DIMENSION > 0 and max(width, height) > WEBP_MAX_DIMENSION:
        args += (["-resize", str(WEBP_MAX_DIMENSION), "0"] if width >= height
                 else ["-resize", "0", str(WEBP_MAX_DIMENSION)])
    _run([*args, str(png), "-o", str(webp)])
    png.unlink(missing_ok=True)
    return webp


def build_item(item: dict, source: Path, bundle: Path) -> dict:
    source_sha, actual_bytes = digest(source)
    base = {**item, "source_sha256": source_sha, "source_bytes": actual_bytes}
    if actual_bytes < LARGE_BYTES:
        return {**base, "status": "skipped", "reason": "below-minimum-100-mib", "strategy": "none",
                "decision_profile": PDF_DECISION_PROFILE}
    object_dir = object_root(source_sha, str(item.get("key") or ""))
    ranged = "page_start" in item or "page_end" in item
    pages = _pages(source, str(item.get("extension") or "pdf"))
    page_start = int(item.get("page_start", 1))
    page_end = int(item.get("page_end", pages))
    if not 1 <= page_start <= page_end <= pages:
        raise ValueError("invalid PDF page range")
    outline = item.get("outline")
    if not isinstance(outline, list):
        outline = extract_pdf_outline(source, pages)
    with tempfile.TemporaryDirectory(dir=bundle) as temp:
        sample = sorted(set([page_start, (page_start + page_end) // 2, page_end]))[:max(1, SAMPLE_PAGES)]
        sample_sizes = []
        sample_end = max(sample)
        classification = str(item.get("classification") or classify_pdf(source, pages))
        metadata = {"pages": pages, "classification": classification, "sample_pages": sample}
        if classification == "native-text":
            return {**base, "status": "skipped", "reason": "native-text-pdf",
                    "strategy": "native-text", "decision_profile": PDF_DECISION_PROFILE, "pdf": metadata}
        for page in sample:
            sample_sizes.append(_render(source, page, Path(temp), str(item.get("extension") or "pdf")).stat().st_size)
        range_pages = page_end - page_start + 1
        estimated = int(round(sum(sample_sizes) / len(sample_sizes) * range_pages))
        metadata.update({"sample_webp_bytes": sample_sizes,
                    "estimated_webp_bytes": estimated})
        page_entries = []
        for page in range(page_start, page_end + 1):
            rendered = _render(source, page, Path(temp), str(item.get("extension") or "pdf"))
            page_sha, page_bytes = digest(rendered)
            destination = bundle / object_dir / "pages" / f"page-{page:06d}.webp"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rendered, destination)
            page_entries.append({"page": page, "path": (object_dir / "pages" / destination.name).as_posix(),
                                 "sha256": page_sha, "bytes": page_bytes})
    page_manifest = {
        "version": 1, "kind": "pdf-pages", "source_sha256": source_sha,
        "profile": PDF_PROFILE, "pages": page_entries,
    }
    if outline:
        page_manifest["toc"] = outline
    if ranged:
        return {**base, "path": "", "status": "ready", "strategy": "sampled-webp", "pdf": metadata,
                "render_profile": PDF_PROFILE, "decision_profile": PDF_DECISION_PROFILE,
                "pages": page_entries, "outline": outline}
    manifest_path = bundle / object_dir / "page-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(page_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_sha, manifest_bytes = digest(manifest_path)
    return {**base, "path": "", "status": "ready", "strategy": "sampled-webp", "pdf": metadata,
            "render_profile": PDF_PROFILE, "decision_profile": PDF_DECISION_PROFILE,
            "pages": page_entries, "outline": outline, "page_manifest": {
                "path": (object_dir / "page-manifest.json").as_posix(),
                "sha256": manifest_sha, "bytes": manifest_bytes,
            }}


def empty_manifest() -> dict:
    return {"version": MANIFEST_VERSION, "files": {}}


def build_publish(manifest: dict, results: list[dict], bundle: Path) -> tuple[dict, list[CommitOperationAdd]]:
    files = dict(manifest.get("files", {}))
    artifacts = {}
    for result in results:
        entry = {k: v for k, v in result.items() if k not in {"key", "task_key", "page_start", "page_end", "range_page_count"}}
        if result["status"] == "ready":
            paths = [result["path"]] if result.get("path") else [page["path"] for page in result["pages"]]
            if result.get("page_manifest"):
                paths.append(result["page_manifest"]["path"])
            for path in paths:
                artifact = bundle / path
                if not artifact.is_file():
                    raise ValueError("missing PDF asset artifact")
                artifacts[path] = str(artifact)
        files[result["key"]] = entry
    updated = {"version": MANIFEST_VERSION, "files": dict(sorted(files.items()))}
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=artifacts[path])
                  for path in sorted(artifacts)]
    operations.append(CommitOperationAdd(
        path_in_repo=MANIFEST_NAME,
        path_or_fileobj=json.dumps(updated, ensure_ascii=False, sort_keys=True, indent=2).encode(),
    ))
    return updated, operations


def remote_manifest(api: HfApi, repo: str) -> dict:
    try:
        path = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=MANIFEST_NAME)
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) != 404:
            raise
        return empty_manifest()
    return json.loads(Path(path).read_text(encoding="utf-8"))


def failed_source_keys(api: HfApi, repo: str, extension: str) -> set[str]:
    try:
        path = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename="manifest.json")
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return set()
        raise
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        key for key, entry in manifest.get("files", {}).items()
        if entry.get("status") == "failed" and entry.get("source_extension") == extension
    }


def remote_sidecar(api: HfApi, repo: str) -> dict:
    try:
        path = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename="reader_assets.json.gz")
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) != 404:
            raise
        return {"v": 1, "f": {}}
    return json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))


def update_sidecar(sidecar: dict, results: list[dict]) -> bytes:
    updated = {"v": 1, "f": dict(sidecar.get("f", {}))}
    for result in results:
        updated["f"].pop(result.get("key"), None)
        if result.get("status") != "ready":
            continue
        path = result.get("path") or result.get("page_manifest", {}).get("path")
        if not path:
            continue
        updated["f"][result["key"]] = {"s": 2, "m": "p", "p": path,
                                        "b": "vomebook/pdf-pages"}
    payload = json.dumps(updated, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return gzip.compress(payload, compresslevel=9, mtime=0)


def bundle_is_published(manifest: dict, results: list[dict]) -> bool:
    entries = manifest.get("files", {})
    for result in results:
        current = entries.get(result.get("key"))
        if not current or current.get("status") != result.get("status"):
            return False
        for field in ("source_revision", "source_sha256", "source_extension", "profile", "strategy",
                      "render_profile", "decision_profile"):
            if current.get(field) != result.get(field):
                return False
        if result.get("status") == "ready":
            if current.get("path") != result.get("path", ""):
                return False
            if result.get("page_manifest") and current.get("page_manifest") != result["page_manifest"]:
                return False
    return bool(results)


def publish(api: HfApi, repo: str, manifest: dict, results: list[dict], bundle: Path,
            max_attempts: int = 20, include_artifacts: bool = True) -> None:
    if not results:
        return
    baseline = None
    for attempt in range(max_attempts):
        info = api.repo_info(repo_id=repo, repo_type="dataset")
        current = remote_manifest(api, repo)
        sidecar = remote_sidecar(api, repo)
        sidecar_keys = sidecar.get("f", {})
        sidecar_matches = all(
            result.get("status") != "ready"
            or sidecar_keys.get(result["key"], {}).get("p") == (
                result.get("path") or result.get("page_manifest", {}).get("path")
            )
            for result in results
        )
        if bundle_is_published(current, results) and sidecar_matches:
            return
        keys = {result["key"] for result in results}
        current_entries = {key: current.get("files", {}).get(key) for key in keys}
        if baseline is None:
            baseline = current_entries
        elif current_entries != baseline:
            raise RuntimeError("PDF asset key changed during publication retry")
        updated, operations = build_publish(current, results, bundle)
        if not include_artifacts:
            operations = [operation for operation in operations
                          if operation.path_in_repo in {MANIFEST_NAME, "reader_assets.json.gz"}]
        operations.append(CommitOperationAdd(
            path_in_repo="reader_assets.json.gz",
            path_or_fileobj=update_sidecar(sidecar, results),
        ))
        try:
            api.create_commit(
                repo_id=repo, repo_type="dataset", operations=operations,
                commit_message="Publish independent PDF asset", parent_commit=info.sha,
            )
            return
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {409, 412} and not (status == 429 or 500 <= (status or 0) < 600):
                raise
            if attempt + 1 == max_attempts:
                raise
            time.sleep(min(60, 2 ** min(attempt, 5)))
    raise RuntimeError("PDF asset publication retry limit reached")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--repo", default="")
    parser.add_argument("--extension", choices=("pdf", "djvu"), default="pdf")
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--source", choices=("upstream", "generated", "all"), default="all")
    parser.add_argument("--reader-assets-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--queue-file", type=Path, help="Weighted queue produced by plan_pdf_assets.py")
    parser.add_argument("--source-dir", type=Path, help="Local source mirror, keyed by dataset/path")
    parser.add_argument("--bundle", type=Path, default=Path("output/pdf-assets/bundle"))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.queue_file:
        records = load_planned_shard(args.queue_file, args.shard_count, args.shard_index)
    else:
        records = []
        if args.source in {"upstream", "all"}:
            records.extend(load_records(args.search_data, args.revisions, args.repo, args.extension))
        if args.source in {"generated", "all"}:
            token = os.environ.get("HF_TOKEN")
            reader_assets_revision = HfApi(token=token).repo_info(
                repo_id=args.assets_repo, repo_type="dataset").sha
            if args.reader_assets_manifest:
                manifest_path = args.reader_assets_manifest
            else:
                from huggingface_hub import hf_hub_download
                manifest_path = Path(hf_hub_download(args.assets_repo, "manifest.json", repo_type="dataset",
                                                     revision=reader_assets_revision, token=token))
            records.extend(load_generated_records(
                manifest_path, args.assets_repo, args.repo, reader_assets_revision))
        records.sort(key=lambda item: (0 if item.get("source_extension") in {"caj", "kdh"} else 1,
                                       item["repo"], item["path"], item["source_kind"]))
        if args.failed_only:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required for --failed-only")
            failed = failed_source_keys(HfApi(token=token), args.assets_repo, args.extension)
            records = [item for item in records if item["key"] in failed]
        records = shard_records(records, args.shard_count, args.shard_index)
        records = queue(records, args.limit, args.checkpoint)
    args.bundle.mkdir(parents=True, exist_ok=True)
    if not records:
        (args.bundle / "bundle.json").write_text(
            json.dumps({"version": 1, "results": []}, sort_keys=True) + "\n", encoding="utf-8")
        print("processed 0 PDF asset(s)")
        return 0
    results = []
    built_by_sha = {}
    for item in records:
        source = None
        try:
            if item.get("source_kind") == "generated":
                source = download_hf_source(
                    item["reader_assets_repo"], item["reader_assets_path"],
                    item.get("reader_assets_revision", "main"), os.environ.get("HF_TOKEN"))
            elif args.source_dir:
                source = args.source_dir / item["repo"] / item["path"]
            elif not args.source_dir:
                source = download_hf_source(item["repo"], item["path"], item["source_revision"], os.environ.get("HF_TOKEN"))
            source_sha, source_bytes = digest(source)
            if source_sha in built_by_sha and "page_start" not in item:
                result = {**built_by_sha[source_sha], **item,
                          "source_sha256": source_sha, "source_bytes": source_bytes}
            else:
                result = build_item(item, source, args.bundle)
                if "page_start" not in item:
                    built_by_sha[source_sha] = {k: v for k, v in result.items()
                                                if k not in {"key", "repo", "path", "source_revision", "source_url"}}
            results.append(result)
        except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError):
            source_sha = ""
            try:
                if source is None:
                    raise OSError
                source_sha, source_bytes = digest(source)
            except OSError:
                source_bytes = item.get("source_bytes", 0)
            results.append({**item, "source_sha256": source_sha, "source_bytes": source_bytes,
                            "status": "failed", "reason": "tool-error", "strategy": "none"})
    for result in results:
        if result.get("status") == "ready":
            paths = [result.get("path")] if result.get("path") else [page["path"] for page in result.get("pages", [])]
            if result.get("page_manifest"):
                paths.append(result["page_manifest"]["path"])
            for path in paths:
                if not (args.bundle / path).is_file():
                    raise RuntimeError(f"missing built artifact: {path}")
    manifest, _ = build_publish(empty_manifest(), results, args.bundle)
    (args.bundle / MANIFEST_NAME).write_bytes(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode())
    (args.bundle / "bundle.json").write_text(
        json.dumps({"version": 1, "results": results}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.dry_run and not args.build_only:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required")
        publish(HfApi(token=token), args.assets_repo, manifest, results, args.bundle)
    print(f"processed {len(results)} PDF asset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
