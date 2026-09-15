from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "pack_run.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("care_pack_run", SCRIPT)
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


class PackRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.source = self.root / "experiments" / "sample" / "run-01"
        (self.source / "nested").mkdir(parents=True)
        self.contents = {
            "metrics.json": b'{"auc": 12.5}\n',
            "nested/trace.jsonl": '{"note": "保留原始测量"}\n'.encode("utf-8"),
            "nested/raw.bin": bytes(range(256)),
        }
        for name, data in self.contents.items():
            (self.source / name).write_bytes(data)
        self.catalog = self.root / "artifacts" / "catalog.json"

    def run_pack(self, archive_id: str = "sample-01", **kwargs):
        return pack.pack_run(self.source, archive_id, catalog_path=self.catalog, **kwargs)

    def assert_sources_unchanged(self) -> None:
        for name, data in self.contents.items():
            self.assertEqual((self.source / name).read_bytes(), data)

    def assert_no_temporary_files(self) -> None:
        if self.catalog.parent.exists():
            self.assertFalse(list(self.catalog.parent.rglob("*.tmp")))
            self.assertFalse((self.catalog.parent / ".catalog.pack.lock").exists())

    def test_roundtrip_catalog_and_source_preservation(self) -> None:
        old_entry = self.run_pack("existing-run")
        original = {"schema_version": 1, "base_commit": "frozen-base", "archives": [old_entry], "note": "keep"}
        self.catalog.write_text(json.dumps(original), encoding="utf-8")
        entry = self.run_pack(category="validation")
        archived = self.root / entry["path"]
        updated = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual(updated["base_commit"], "frozen-base")
        self.assertEqual(updated["note"], "keep")
        self.assertEqual(updated["archives"], [old_entry, entry])
        self.assertEqual(entry["file_count"], len(self.contents))
        self.assertEqual(entry["unpacked_bytes"], sum(map(len, self.contents.values())))
        self.assertEqual(entry["size_bytes"], archived.stat().st_size)
        self.assertEqual(entry["sha256"], hashlib.sha256(archived.read_bytes()).hexdigest())
        self.assertEqual(entry["source_ref"], "working-tree")
        self.assertEqual(entry["source_commit"], "working-tree")
        prefix = self.source.relative_to(self.root).as_posix()
        self.assertEqual(entry["source_prefixes"], [prefix])
        with tarfile.open(archived, "r:gz") as archive:
            manifest = json.load(archive.extractfile(pack.MANIFEST_NAME))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["source_ref"], "working-tree")
            self.assertEqual(manifest["source_commit"], "working-tree")
            self.assertEqual(set(archive.getnames()), {pack.MANIFEST_NAME} | {f"{prefix}/{name}" for name in self.contents})
            expected_records = []
            for name, data in sorted(self.contents.items()):
                member_name = f"{prefix}/{name}"
                self.assertEqual(archive.extractfile(member_name).read(), data)
                expected_records.append({"path": member_name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": None})
            self.assertEqual(manifest["files"], expected_records)
        loaded_root, loaded_entries = pack.artifacts.load_catalog(self.catalog)
        self.assertEqual(loaded_root, self.root)
        self.assertEqual(loaded_entries[-1], entry)
        restored = self.root.parent / "restored"
        pack.artifacts.restore_archives(self.root, [entry], destination=restored)
        for name, data in self.contents.items():
            self.assertEqual((restored / prefix / name).read_bytes(), data)
        self.assert_sources_unchanged()
        self.assert_no_temporary_files()

    def test_deterministic_contents_despite_name_and_mtime_changes(self) -> None:
        first = self.run_pack("first")
        for name in self.contents:
            os.utime(self.source / name, (123456789, 123456789))
        second = self.run_pack("second")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual((self.root / first["path"]).read_bytes(), (self.root / second["path"]).read_bytes())
        self.assert_sources_unchanged()

    def test_rejects_existing_id_and_existing_destination(self) -> None:
        first = self.run_pack()
        catalog_before = self.catalog.read_bytes()
        archive_before = (self.root / first["path"]).read_bytes()
        with self.assertRaisesRegex(pack.PackError, "already exists"):
            self.run_pack(category="another-category")
        occupied = self.catalog.parent / "new-runs" / "occupied.tar.gz"
        occupied.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(pack.PackError, "already exists"):
            self.run_pack("occupied")
        self.assertEqual(self.catalog.read_bytes(), catalog_before)
        self.assertEqual((self.root / first["path"]).read_bytes(), archive_before)
        self.assertEqual(occupied.read_bytes(), b"do not overwrite")
        self.assert_sources_unchanged()
        self.assert_no_temporary_files()

    def test_rejects_outside_root_and_reserved_directories(self) -> None:
        outside = self.root.parent / "outside"
        outside.mkdir()
        paths = [outside, self.root, self.root / "artifacts", self.root / ".git", self.root / ".venv"]
        for path in paths[2:]:
            path.mkdir(exist_ok=True)
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(pack.PackError):
                    pack.pack_run(path, "no", catalog_path=self.catalog)
        with self.assertRaises(pack.PackError):
            pack.pack_run(self.root / "experiments" / ".." / ".." / "outside", "no", catalog_path=self.catalog)
        for archive_id, category in [("../escape", "runs"), ("ok", "../escape"), ("", "runs"), ("bad/id", "runs")]:
            with self.subTest(archive_id=archive_id, category=category):
                with self.assertRaises(pack.PackError):
                    pack.pack_run(self.source, archive_id, category, self.catalog)
        self.assertFalse(self.catalog.exists())
        self.assert_sources_unchanged()

    def test_rejects_excluded_directory_nested_inside_run(self) -> None:
        (self.source / ".venv").mkdir()
        (self.source / ".venv" / "cache").write_text("not an experiment", encoding="utf-8")
        with self.assertRaisesRegex(pack.PackError, "Excluded"):
            self.run_pack()
        self.assertFalse(self.catalog.exists())

    def test_rejects_symlink_inside_run_and_linked_source(self) -> None:
        target = self.root.parent / "external-data"
        target.mkdir()
        (target / "private.txt").write_text("outside", encoding="utf-8")
        link = self.source / "linked"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Creating symlinks is unavailable: {error}")
        with self.assertRaisesRegex(pack.PackError, "Symlinks and junctions"):
            self.run_pack()
        with self.assertRaisesRegex(pack.PackError, "Symlinks and junctions"):
            pack.pack_run(link, "linked", catalog_path=self.catalog)
        self.assertEqual((target / "private.txt").read_text(encoding="utf-8"), "outside")
        self.assertFalse(self.catalog.exists())

    def test_total_size_limit_rejects_before_writing(self) -> None:
        with mock.patch.object(pack, "MAX_UNPACKED_BYTES", 5):
            with self.assertRaisesRegex(pack.PackError, "smaller subdirectory"):
                self.run_pack()
        self.assertFalse(self.catalog.exists())
        self.assert_sources_unchanged()

    @unittest.skipUnless(sys.platform == "win32", "Windows junction test")
    def test_rejects_windows_junction_without_reading_external_files(self) -> None:
        import _winapi

        target = self.root.parent / "external-junction-data"
        target.mkdir()
        (target / "private.txt").write_text("outside", encoding="utf-8")
        junction = self.source / "junction"
        _winapi.CreateJunction(str(target), str(junction))
        with self.assertRaisesRegex(pack.PackError, "Symlinks and junctions"):
            self.run_pack()
        with self.assertRaisesRegex(pack.PackError, "Symlinks and junctions"):
            pack.pack_run(junction, "junction", catalog_path=self.catalog)
        self.assertEqual((target / "private.txt").read_text(encoding="utf-8"), "outside")
        self.assertFalse(self.catalog.exists())

    def test_verification_failure_keeps_previous_catalog(self) -> None:
        self.run_pack("previous")
        before = self.catalog.read_bytes()
        with mock.patch.object(pack, "_verify_archive", side_effect=pack.PackError("test corruption")):
            with self.assertRaisesRegex(pack.PackError, "corruption"):
                self.run_pack("broken")
        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertFalse((self.catalog.parent / "new-runs" / "broken.tar.gz").exists())
        self.assert_sources_unchanged()
        self.assert_no_temporary_files()

    def test_catalog_publication_failure_rolls_back_new_archive(self) -> None:
        self.run_pack("previous")
        before = self.catalog.read_bytes()
        with mock.patch.object(pack.os, "replace", side_effect=OSError("catalog unavailable")):
            with self.assertRaisesRegex(OSError, "catalog unavailable"):
                self.run_pack("failed-transaction")
        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertFalse((self.catalog.parent / "new-runs" / "failed-transaction.tar.gz").exists())
        self.assert_sources_unchanged()
        self.assert_no_temporary_files()

    def test_corrupt_member_is_detected(self) -> None:
        entry = self.run_pack()
        path = self.root / entry["path"]
        with tarfile.open(path, "r:gz") as archive:
            manifest = json.load(archive.extractfile(pack.MANIFEST_NAME))
        manifest["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(pack.PackError, "source snapshot"):
            pack._verify_archive(self.root, entry, manifest)

    def test_cli_supports_custom_catalog_and_rejects_duplicates(self) -> None:
        command = [sys.executable, str(SCRIPT), str(self.source), "--id", "cli-run", "--category", "new-runs", "--catalog", str(self.catalog)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["id"], "cli-run")
        duplicate = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("already exists", duplicate.stderr)
        self.assert_sources_unchanged()


if __name__ == "__main__":
    unittest.main()
