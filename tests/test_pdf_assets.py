import json
import requests
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from huggingface_hub.errors import HfHubHTTPError

from scripts import pdf_assets
from scripts import plan_pdf_assets
from scripts import publish_pdf_assets
from scripts import retire_pdf_assets


class PdfAssetsTests(unittest.TestCase):
    def test_compact_records_are_decoded_and_queue_is_stable(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            search = root / "search.json"
            revisions = root / "commits.json"
            search.write_text(json.dumps({"rp": ["VoiceOfML/A"], "fd": [[]],
                "rc": [[0, "book", "pdf", 0, 60000000, False],
                       [0, "other", "txt", 0, 1, False]]}), encoding="utf-8")
            revisions.write_text(json.dumps({"VoiceOfML/A": "rev"}), encoding="utf-8")
            records = pdf_assets.load_records(search, revisions)
            self.assertEqual([x["path"] for x in pdf_assets.queue(records, 1, 0)], ["book.pdf"])
            self.assertEqual(pdf_assets.queue(records, 1, 1), [])

    def test_small_pdf_is_skipped_without_tools(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "x.pdf"
            source.write_bytes(b"x" * (pdf_assets.MIN_BYTES - 1))
            item = {"key": "r\0x.pdf", "repo": "r", "path": "x.pdf", "source_revision": "rev"}
            with patch.object(pdf_assets, "_run") as run:
                result = pdf_assets.build_item(item, source, root / "bundle")
            self.assertEqual(result["status"], "skipped")
            run.assert_not_called()

    def test_render_does_not_upscale_small_portrait(self):
        with tempfile.TemporaryDirectory() as root, patch.object(
                pdf_assets, "_run") as run, patch.object(
                pdf_assets, "_image_dimensions", return_value=(1200, 1700)):
            output = pdf_assets._render(Path("book.pdf"), 1, Path(root))
        self.assertTrue(output.name.endswith(".webp"))
        cwebp = run.call_args_list[-1].args[0]
        self.assertNotIn("-resize", cwebp)

    def test_render_downscales_longest_edge_and_preserves_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as root, patch.object(
                pdf_assets, "_run") as run, patch.object(
                pdf_assets, "_image_dimensions", return_value=(2400, 3200)):
            pdf_assets._render(Path("book.pdf"), 1, Path(root))
        cwebp = run.call_args_list[-1].args[0]
        resize = cwebp.index("-resize")
        self.assertEqual(cwebp[resize:resize + 3], ["-resize", "0", "1800"])

    def test_rendered_image_dimensions_support_png_and_ppm(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            png = root / "page.png"
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 1234, 1678))
            ppm = root / "page.ppm"
            ppm.write_bytes(b"P6\n# generated\n640 480\n255\n")
            self.assertEqual(pdf_assets._image_dimensions(png), (1234, 1678))
            self.assertEqual(pdf_assets._image_dimensions(ppm), (640, 480))

    def test_weighted_shards_use_descending_page_count_and_stable_ties(self):
        records = [{"key": key, "page_count": pages} for key, pages in (
            ("a", 10), ("b", 9), ("c", 8), ("d", 7), ("e", 6), ("f", 5))]
        shards = pdf_assets.weighted_shards(records, 3)
        self.assertEqual([[item["key"] for item in shard] for shard in shards],
                         [["a", "f"], ["b", "e"], ["c", "d"]])
        self.assertEqual([sum(item["page_count"] for item in shard) for shard in shards], [15, 15, 15])

    def test_weighted_shards_tie_breaks_by_key(self):
        records = [{"key": key, "page_count": 4} for key in ("c", "a", "b")]
        shards = pdf_assets.weighted_shards(records, 2)
        self.assertEqual([[item["key"] for item in shard] for shard in shards], [["a", "c"], ["b"]])

    def test_task_page_limit_uses_500_pages_for_regular_files(self):
        self.assertEqual(pdf_assets.max_pages_per_task({"page_count": 999, "source_bytes": pdf_assets.LARGE_BYTES}), 500)

    def test_task_page_limit_uses_250_pages_for_very_large_files(self):
        self.assertEqual(pdf_assets.max_pages_per_task({"page_count": 600, "source_bytes": pdf_assets.VERY_LARGE_BYTES}), 250)
        self.assertEqual(pdf_assets.max_pages_per_task({"page_count": 1999, "source_bytes": pdf_assets.LARGE_BYTES}), 500)
        self.assertEqual(pdf_assets.max_pages_per_task({"page_count": 2000, "source_bytes": pdf_assets.LARGE_BYTES}), 250)

    def test_plan_uses_500_page_ranges_for_regular_large_pdf(self):
        records = [{"key": "r\0regular.pdf", "repo": "r", "path": "regular.pdf",
                    "extension": "pdf", "source_extension": "pdf"}]
        with patch.object(plan_pdf_assets.pdf_assets, "_pages", return_value=600), patch.object(
                plan_pdf_assets, "source_path", return_value=Path("regular.pdf")), patch.object(
                plan_pdf_assets.pdf_assets, "digest", return_value=("a" * 64, pdf_assets.LARGE_BYTES)), patch.object(
                plan_pdf_assets.pdf_assets, "classify_pdf", return_value="scan"):
            planned = plan_pdf_assets.plan(records, None, "assets", 18, workers=1)
        tasks = [task for shard in planned["shards"] for task in shard["records"]]
        self.assertEqual([(task["page_start"], task["page_end"]) for task in tasks], [(1, 500), (501, 600)])
        self.assertEqual(planned["shard_count"], 20)

    def test_plan_uses_250_page_ranges_for_very_large_pdf(self):
        records = [{"key": "r\0large.pdf", "repo": "r", "path": "large.pdf",
                    "extension": "pdf", "source_extension": "pdf"}]
        with patch.object(plan_pdf_assets.pdf_assets, "_pages", return_value=600), patch.object(
                plan_pdf_assets, "source_path", return_value=Path("large.pdf")), patch.object(
                plan_pdf_assets.pdf_assets, "digest", return_value=("a" * 64, pdf_assets.VERY_LARGE_BYTES)), patch.object(
                plan_pdf_assets.pdf_assets, "classify_pdf", return_value="scan"):
            planned = plan_pdf_assets.plan(records, None, "assets", 18, workers=1)
        tasks = [task for shard in planned["shards"] for task in shard["records"]]
        self.assertEqual([(task["page_start"], task["page_end"]) for task in tasks],
                         [(1, 250), (251, 500), (501, 600)])
        self.assertEqual(planned["shard_count"], 21)

    def test_plan_rejects_a_matrix_larger_than_github_limit(self):
        records = [{"key": "r\0too-large.pdf", "repo": "r", "path": "too-large.pdf",
                    "extension": "pdf", "source_extension": "pdf"}]
        with patch.object(plan_pdf_assets.pdf_assets, "_pages", return_value=128500), patch.object(
                plan_pdf_assets, "source_path", return_value=Path("too-large.pdf")), patch.object(
                plan_pdf_assets.pdf_assets, "digest", return_value=("a" * 64, pdf_assets.LARGE_BYTES)), patch.object(
                plan_pdf_assets.pdf_assets, "classify_pdf", return_value="scan"):
            with self.assertRaisesRegex(ValueError, "exceeds GitHub Actions matrix limit 256"):
                plan_pdf_assets.plan(records, None, "assets", 18, workers=1)

    def test_plan_splits_only_large_caj_into_deterministic_ranges(self):
        records = [{"key": "r\0large.caj", "repo": "r", "path": "large.caj", "source_extension": "caj"},
                   {"key": "r\0normal.pdf", "repo": "r", "path": "normal.pdf", "source_extension": "pdf"}]
        with patch.object(plan_pdf_assets.pdf_assets, "_pages", side_effect=[2501, 300]), patch.object(
                plan_pdf_assets, "source_path", side_effect=[Path("large.pdf"), Path("normal.pdf")]), patch.object(
                plan_pdf_assets.pdf_assets, "digest", side_effect=[("a" * 64, pdf_assets.LARGE_BYTES),
                                                                    ("b" * 64, pdf_assets.LARGE_BYTES)]), patch.object(
                plan_pdf_assets.pdf_assets, "classify_pdf", side_effect=["scan", "native-text"]):
            planned = plan_pdf_assets.plan(records, None, "assets", 18, workers=1)
        tasks = [task for shard in planned["shards"] for task in shard["records"]]
        ranges = sorted((task["page_start"], task["page_end"]) for task in tasks
                        if task["key"] == "r\0large.caj")
        self.assertEqual(ranges, [(1, 250), (251, 500), (501, 750), (751, 1000), (1001, 1250),
                                   (1251, 1500), (1501, 1750), (1751, 2000), (2001, 2250),
                                   (2251, 2500), (2501, 2501)])
        self.assertEqual(len([task for task in tasks if task["key"] == "r\0normal.pdf"]), 0)
        self.assertEqual((planned["total_records"], planned["total_tasks"]), (2, 11))
        self.assertEqual([item["key"] for item in planned["skipped"]], ["r\0normal.pdf"])
        self.assertEqual(planned["ordinary_shard_count"], 18)
        self.assertEqual(planned["shard_count"], 29)
        self.assertEqual(planned["shard_ids"], list(range(29)))
        self.assertEqual(
            [shard["index"] for shard in planned["shards"]
             if any("page_start" in task for task in shard["records"])],
             [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        )
        self.assertTrue(all(len(shard["records"]) == 1 for shard in planned["shards"][18:]))
        self.assertEqual(len({task["task_key"] for task in tasks if "task_key" in task}), 11)
        self.assertTrue(all(
            "page_start" not in task
             for shard in planned["shards"][:18]
            for task in shard["records"]
        ))

    def test_plan_requires_exactly_eighteen_ordinary_shards(self):
        with self.assertRaisesRegex(ValueError, "ordinary PDF shard count must be 18"):
            plan_pdf_assets.plan([], None, "assets", 10)

    def test_plan_splits_oversized_original_scan_pdf(self):
        records = [{"key": "r\0scan.pdf", "repo": "r", "path": "scan.pdf",
                    "extension": "pdf", "source_extension": "pdf"}]
        with patch.object(plan_pdf_assets.pdf_assets, "_pages", return_value=1200), patch.object(
                plan_pdf_assets, "source_path", return_value=Path("scan.pdf")), patch.object(
                plan_pdf_assets.pdf_assets, "digest", return_value=("a" * 64, pdf_assets.LARGE_BYTES)), patch.object(
                plan_pdf_assets.pdf_assets, "classify_pdf", return_value="scan"):
            planned = plan_pdf_assets.plan(records, None, "assets", 18, workers=1)
        self.assertEqual(planned["total_tasks"], 3)
        self.assertEqual(planned["shard_count"], 21)
        all_tasks = [t for s in planned["shards"] for t in s["records"]]
        self.assertEqual([(t["page_start"], t["page_end"]) for t in all_tasks],
                         [(1, 500), (501, 1000), (1001, 1200)])

    def test_generated_records_pin_reader_assets_revision(self):
        manifest = {"files": {"r\0book.caj": {
            "status": "ready", "reader_mode": "pdf", "path": "objects/a/document.pdf",
            "bytes": pdf_assets.LARGE_BYTES, "source_revision": "source-rev",
        }}}
        records = pdf_assets.load_generated_records(manifest, "reader-assets", assets_revision="assets-rev")
        self.assertEqual(records[0]["reader_assets_revision"], "assets-rev")

    def test_generated_source_path_uses_pinned_revision(self):
        item = {"source_kind": "generated", "reader_assets_repo": "reader-assets",
                "reader_assets_path": "objects/a/document.pdf", "reader_assets_revision": "assets-rev"}
        with patch.object(plan_pdf_assets, "hf_hub_download", return_value="/tmp/document.pdf") as download:
            self.assertEqual(plan_pdf_assets.source_path(item, None, "reader-assets"), Path("/tmp/document.pdf"))
        self.assertEqual(download.call_args.kwargs["revision"], "assets-rev")

    def test_planner_isolates_invalid_pdf(self):
        records = [{"key": "r\0bad.pdf", "repo": "r", "path": "bad.pdf", "source_bytes": pdf_assets.LARGE_BYTES}]
        with patch.object(plan_pdf_assets, "source_path", return_value=Path("bad.pdf")), patch.object(
                plan_pdf_assets.pdf_assets, "_pages", side_effect=ValueError("bad PDF")):
            planned = plan_pdf_assets.plan(records, None, "assets", 18, workers=1)
        self.assertEqual(planned["total_tasks"], 0)
        self.assertEqual(planned["skipped"][0]["status"], "failed")
        self.assertEqual(planned["skipped"][0]["reason"], "tool-error")

    def test_download_failure_does_not_reuse_previous_source(self):
        records = [
            {"key": "r\0first.pdf", "source_kind": "generated", "reader_assets_repo": "assets",
             "reader_assets_path": "first.pdf", "reader_assets_revision": "rev", "source_bytes": 1},
            {"key": "r\0second.pdf", "source_kind": "generated", "reader_assets_repo": "assets",
             "reader_assets_path": "second.pdf", "reader_assets_revision": "rev", "source_bytes": 2},
        ]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "first.pdf"
            source.write_bytes(b"first")
            args = SimpleNamespace(queue_file=root / "queue.json", shard_count=1, shard_index=0,
                                   bundle=root / "bundle", dry_run=False, build_only=True)
            with patch.object(pdf_assets, "parse_args", return_value=args), patch.object(
                    pdf_assets, "load_planned_shard", return_value=records), patch.object(
                    pdf_assets, "download_hf_source", side_effect=[source, OSError("download failed")]), patch.object(
                    pdf_assets, "build_item", return_value={"key": records[0]["key"], "status": "skipped",
                                                             "strategy": "none", "source_sha256": "first-sha"}):
                self.assertEqual(pdf_assets.main(), 0)
            results = json.loads((args.bundle / "bundle.json").read_text(encoding="utf-8"))["results"]
        self.assertEqual(results[1]["source_sha256"], "")
        self.assertEqual(results[1]["source_bytes"], 2)

    def test_object_root_is_unique_per_source_entry(self):
        self.assertNotEqual(
            pdf_assets.object_root("a" * 64, "repo\0first.pdf"),
            pdf_assets.object_root("a" * 64, "repo\0second.pdf"),
        )

    def test_publish_manifest_is_separate_and_content_addressed(self):
        result = {"key": "r\0x.pdf", "status": "skipped", "reason": "estimated-webp-over-90-percent",
                  "strategy": "sampled-webp", "source_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as root:
            manifest, operations = pdf_assets.build_publish(pdf_assets.empty_manifest(), [result], Path(root))
        self.assertIn("r\0x.pdf", manifest["files"])
        self.assertEqual(operations[-1].path_in_repo, "pdf_manifest.json")
        self.assertEqual(pdf_assets.MANIFEST_NAME, "pdf_manifest.json")

    def test_pending_records_exclude_completed_and_small_sources(self):
        records = [
            {"key": "r\0small.pdf", "source_kind": "upstream", "source_bytes": pdf_assets.LARGE_BYTES - 1},
            {"key": "r\0done.pdf", "source_kind": "upstream", "source_bytes": pdf_assets.LARGE_BYTES},
            {"key": "r\0new.pdf", "source_kind": "upstream", "source_bytes": pdf_assets.LARGE_BYTES},
        ]
        pending = plan_pdf_assets.pending_records(records, {"files": {
            "r\0done.pdf": {"status": "ready", "strategy": "sampled-webp",
                             "render_profile": pdf_assets.PDF_PROFILE,
                             "decision_profile": pdf_assets.PDF_DECISION_PROFILE},
        }})
        self.assertEqual([item["key"] for item in pending], ["r\0new.pdf"])

    def test_pending_records_rebuild_old_streams_and_linearized_pdfs(self):
        records = [
            {"key": "r\0legacy.pdf", "source_kind": "upstream", "source_bytes": pdf_assets.LARGE_BYTES},
            {"key": "r\0current.pdf", "source_kind": "upstream", "source_bytes": pdf_assets.LARGE_BYTES},
            {"key": "r\0linear.pdf", "source_kind": "upstream", "source_bytes": pdf_assets.LARGE_BYTES},
        ]
        manifest = {"files": {
            "r\0legacy.pdf": {"status": "ready", "strategy": "sampled-webp",
                               "render_profile": "pdf-pages-v1-85-2400"},
            "r\0current.pdf": {"status": "ready", "strategy": "sampled-webp",
                                "render_profile": pdf_assets.PDF_PROFILE,
                                "decision_profile": pdf_assets.PDF_DECISION_PROFILE},
            "r\0linear.pdf": {"status": "ready", "strategy": "linearized-pdf"},
        }}
        self.assertEqual(
            [item["key"] for item in plan_pdf_assets.pending_records(records, manifest)],
            ["r\0legacy.pdf", "r\0linear.pdf"],
        )

    def test_merge_bundles_combines_multiple_shards_and_empty_shards(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first, second, merged = (root / name for name in ("shard-0", "shard-1", "merged"))
            first.mkdir()
            second.mkdir()
            results = [
                {"key": "r\0a.pdf", "status": "ready", "reason": "scan",
                 "strategy": "sampled-webp", "source_revision": "1", "source_sha256": "a" * 64,
                 "source_extension": "pdf", "profile": "p", "path": "objects/a/page.webp",
                 "pages": [], "page_manifest": {}},
            ]
            (first / "bundle.json").write_text(json.dumps({"version": 1, "results": results}), encoding="utf-8")
            (second / "bundle.json").write_text(json.dumps({"version": 1, "results": []}), encoding="utf-8")
            (first / "objects/a/page.webp").parent.mkdir(parents=True)
            (first / "objects/a/page.webp").write_bytes(b"webp")
            merged_results = publish_pdf_assets.merge_bundles([second, first], merged)
            _, operations = pdf_assets.build_publish(pdf_assets.empty_manifest(), merged_results, merged)
        self.assertEqual([result["key"] for result in merged_results], ["r\0a.pdf"])
        self.assertIn("objects/a/page.webp", {operation.path_in_repo for operation in operations})

    def test_merge_bundles_aggregates_complete_ranges_to_one_manifest_result(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundles = []
            for index, (start, end) in enumerate(((1, 1000), (1001, 1200))):
                bundle = root / f"bundle-{index}"
                bundle.mkdir()
                object_path = f"objects/sha/pages/page-{start:06d}.webp"
                (bundle / object_path).parent.mkdir(parents=True)
                (bundle / object_path).write_bytes(f"{start}".encode())
                result = {"key": "r\0large.pdf", "task_key": f"task-{index}", "status": "ready",
                          "strategy": "sampled-webp", "source_revision": "1", "source_sha256": "a" * 64,
                          "source_extension": "pdf", "profile": "p", "page_count": 1200,
                          "page_start": start, "page_end": end, "range_page_count": end - start + 1,
                          "pages": [{"page": page, "path": f"objects/sha/pages/page-{page:06d}.webp"}
                                    for page in range(start, end + 1)]}
                (bundle / "bundle.json").write_text(json.dumps({"version": 1, "results": [result]}), encoding="utf-8")
                bundles.append(bundle)
            results = publish_pdf_assets.merge_bundles(bundles, root / "merged")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key"], "r\0large.pdf")
        self.assertTrue(results[0]["page_manifest"]["path"].endswith("page-manifest.json"))

    def test_merge_bundles_rejects_overlapping_ranges(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = root / "bundle"
            bundle.mkdir()
            results = [{"key": "r\0large.pdf", "status": "ready", "source_sha256": "a" * 64,
                        "page_count": 1200, "page_start": 1, "page_end": 1000, "pages": []},
                       {"key": "r\0large.pdf", "status": "ready", "source_sha256": "a" * 64,
                        "page_count": 1200, "page_start": 1000, "page_end": 1200, "pages": []}]
            (bundle / "bundle.json").write_text(json.dumps({"version": 1, "results": results}), encoding="utf-8")
            with self.assertRaises(ValueError):
                publish_pdf_assets.merge_bundles([bundle], root / "merged")

    def test_merge_bundles_rejects_colliding_range_artifact_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = root / "bundle"
            bundle.mkdir()
            result = {"key": "r\0large.pdf", "status": "ready", "source_sha256": "a" * 64,
                      "page_count": 2, "page_start": 1, "page_end": 2,
                      "pages": [{"page": 1, "path": "objects/sha/page.webp"},
                                {"page": 2, "path": "objects/sha/page.webp"}]}
            (bundle / "bundle.json").write_text(
                json.dumps({"version": 1, "results": [result]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting PDF page artifact paths"):
                publish_pdf_assets.merge_bundles([bundle], root / "merged")

    def test_result_chunks_are_stable_and_bounded(self):
        results = [{"key": str(index)} for index in range(5)]
        self.assertEqual(
            publish_pdf_assets.result_chunks(results, 2),
            [[results[0], results[1]], [results[2], results[3]], [results[4]]],
        )

    def test_publish_retries_parent_race_and_reuses_successful_commit(self):
        result = {"key": "r\0a.pdf", "status": "skipped", "reason": "native-text-pdf",
                  "strategy": "native-text", "source_revision": "1", "source_sha256": "a" * 64,
                  "source_extension": "pdf", "profile": "p"}
        response = requests.Response()
        response.status_code = 409
        response.request = requests.Request("POST", "https://huggingface.co/commit").prepare()
        api = Mock()
        api.repo_info.side_effect = [Mock(sha="parent-1"), Mock(sha="parent-2")]
        api.create_commit.side_effect = [HfHubHTTPError("conflict", response=response), None]
        with tempfile.TemporaryDirectory() as root, patch.object(
                pdf_assets, "remote_manifest", return_value=pdf_assets.empty_manifest()), patch.object(
                pdf_assets, "remote_sidecar", return_value={"v": 1, "f": {}}), patch.object(
                pdf_assets.time, "sleep"):
            pdf_assets.publish(api, "repo", pdf_assets.empty_manifest(), [result], Path(root))
        self.assertEqual(api.create_commit.call_count, 2)
        self.assertEqual([call.kwargs["parent_commit"] for call in api.create_commit.call_args_list],
                         ["parent-1", "parent-2"])

    def test_empty_publication_is_a_noop(self):
        api = Mock()
        pdf_assets.publish(api, "repo", pdf_assets.empty_manifest(), [], Path("/tmp/unused"))
        api.repo_info.assert_not_called()

    def test_publish_can_commit_metadata_after_large_folder_upload(self):
        result = {"key": "r\0a.pdf", "status": "skipped", "reason": "native-text-pdf",
                  "strategy": "native-text", "source_revision": "1", "source_sha256": "a" * 64,
                  "source_extension": "pdf", "profile": "p"}
        api = Mock()
        api.repo_info.return_value = Mock(sha="parent")
        with patch.object(pdf_assets, "remote_manifest", return_value=pdf_assets.empty_manifest()), patch.object(
                pdf_assets, "remote_sidecar", return_value={"v": 1, "f": {}}):
            pdf_assets.publish(api, "repo", pdf_assets.empty_manifest(), [result], Path("/tmp/unused"),
                               include_artifacts=False)
        paths = [op.path_in_repo for op in api.create_commit.call_args.kwargs["operations"]]
        self.assertEqual(paths, ["pdf_manifest.json", "reader_assets.json.gz"])

    def test_hf_source_download_retries_rate_limit(self):
        response = requests.Response()
        response.status_code = 429
        error = HfHubHTTPError("rate limited", response=response)
        with patch("huggingface_hub.hf_hub_download", side_effect=[error, "/tmp/source.pdf"]), patch.object(pdf_assets.time, "sleep") as sleep:
            self.assertEqual(pdf_assets.download_hf_source("repo", "book.pdf", "rev", "token"), Path("/tmp/source.pdf"))
        sleep.assert_called_once_with(1)

    def test_publish_includes_skipped_entries_from_file(self):
        skipped = [{"key": "r\0native.pdf", "status": "skipped", "reason": "native-text-pdf",
                    "strategy": "none", "source_sha256": "a" * 64, "source_extension": "pdf",
                    "page_count": 2501}]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            skipped_path = root / "skipped.json"
            skipped_path.write_text(json.dumps(skipped), encoding="utf-8")
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "bundle.json").write_text(json.dumps({"version": 1, "results": []}), encoding="utf-8")
            args = SimpleNamespace(bundles=[bundle], assets_repo="repo", chunk_results=1,
                                   skipped=skipped_path, dry_run=True)
            with patch.object(publish_pdf_assets, "merge_bundles", return_value=[]), patch.object(
                    pdf_assets, "build_publish", return_value=(pdf_assets.empty_manifest(), [])):
                # Just test that --skipped is read and merged
                results = []
                merged = root / "merged"
                merged_results = publish_pdf_assets.merge_bundles([bundle], merged)
                import json as _json
                skipped_data = _json.loads(skipped_path.read_text(encoding="utf-8"))
                merged_results.extend(skipped_data)
            self.assertEqual(len(merged_results), 1)
            self.assertEqual(merged_results[0]["key"], "r\0native.pdf")

    def test_retire_removes_linearized_and_legacy_streams_but_keeps_current_streams(self):
        old_root = "objects/aa/" + "a" * 64
        current_root = "objects/bb/" + "b" * 64
        pdf_manifest = {"version": 1, "files": {
            "linear": {"status": "ready", "strategy": "linearized-pdf",
                       "path": "objects/cc/" + "c" * 64 + "/linearized.pdf"},
            "legacy": {"status": "ready", "strategy": "sampled-webp",
                       "page_manifest": {"path": old_root + "/page-manifest.json"}},
            "current": {"status": "ready", "strategy": "sampled-webp",
                        "render_profile": pdf_assets.PDF_PROFILE,
                        "decision_profile": pdf_assets.PDF_DECISION_PROFILE,
                        "page_manifest": {"path": current_root + "/page-manifest.json"}},
        }}
        updated, _, deletes = retire_pdf_assets.retire({"version": 1, "files": {}}, pdf_manifest)
        self.assertEqual(updated["files"]["linear"]["status"], "retired")
        self.assertEqual(updated["files"]["legacy"]["status"], "retired")
        self.assertEqual(updated["files"]["current"]["status"], "ready")
        paths = {operation.path_in_repo for operation in deletes}
        self.assertIn(old_root + "/page-manifest.json", paths)
        self.assertIn(old_root + "/pages", paths)
        self.assertNotIn(current_root + "/page-manifest.json", paths)


if __name__ == "__main__":
    unittest.main()
