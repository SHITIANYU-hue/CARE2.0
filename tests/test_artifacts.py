"""Portable archive integrity and restore safety tests; no real artifacts needed."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


_SPEC = importlib.util.spec_from_file_location("care_artifacts", Path(__file__).resolve().parents[1] / "tools" / "artifacts.py")
artifacts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = artifacts
_SPEC.loader.exec_module(artifacts)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "artifacts" / "catalog.json"
        self.catalog.parent.mkdir()
        self.entries = []

    def make_archive(self, archive_id="sample", files=None, mutate_manifest=None, extra_members=()):
        if files is None:
            files = {"experiments/run/results.json": b'{"score": 0.75}\n', "docs/note.txt": b"hello\n"}
        manifest = {
            "schema_version": 1,
            "source_ref": "origin/main",
            "source_commit": "a" * 40,
            "files": [
                {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
                for name, data in files.items()
            ],
        }
        if mutate_manifest:
            mutate_manifest(manifest)
        path = self.catalog.parent / f"{archive_id}.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            for info, data in extra_members:
                tar.addfile(info, io.BytesIO(data))
            data = json.dumps(manifest).encode()
            info = tarfile.TarInfo(artifacts.MANIFEST_NAME)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        entry = {
            "id": archive_id, "category": "results", "path": path.relative_to(self.root).as_posix(),
            "source_ref": "origin/main", "source_commit": "a" * 40,
            "file_count": len(files), "unpacked_bytes": sum(map(len, files.values())),
            "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_prefixes": ["experiments/"],
        }
        self.entries.append(entry)
        self.save_catalog()
        return entry

    def save_catalog(self):
        self.catalog.write_text(json.dumps({"schema_version": 1, "base_commit": "a" * 40, "archives": self.entries}), encoding="utf-8")

    def cli(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = artifacts.main(["--catalog", str(self.catalog), *args])
        return code, output.getvalue(), error.getvalue()

    def test_roundtrip_and_identical_existing_files_are_skipped(self):
        entry = self.make_archive(files={"nested/中文.txt": "你好\n".encode(), "empty": b""})
        root, entries = artifacts.load_catalog(self.catalog)
        self.assertEqual(artifacts.verify_archive(root, entries[0]).files["empty"].size, 0)
        destination = self.root / "restored"
        self.assertEqual(artifacts.restore_archives(root, entries, destination), (2, 0))
        self.assertEqual((destination / "nested/中文.txt").read_bytes(), "你好\n".encode())
        before = (destination / "empty").stat().st_mtime_ns
        self.assertEqual(artifacts.restore_archives(root, [entry], destination), (0, 2))
        self.assertEqual((destination / "empty").stat().st_mtime_ns, before)
        self.assertFalse((destination / artifacts.MANIFEST_NAME).exists())

    def test_cli_selection_and_category_filter(self):
        self.make_archive()
        self.assertEqual(self.cli("verify", "--all")[0], 0)
        self.assertIn("Verified sample", self.cli("verify", "sample")[1])
        self.assertEqual(len(self.cli("list", "--category", "missing")[1].splitlines()), 1)
        self.assertIn("sample\tresults", self.cli("list", "--category", "results")[1])
        for selection in ((), ("unknown",), ("sample", "--all")):
            with self.subTest(selection=selection):
                self.assertEqual(self.cli("restore", *selection)[0], 1)
        destination = self.root / "output"
        self.assertEqual(self.cli("restore", "sample", "--destination", str(destination))[0], 0)
        self.assertTrue((destination / "docs/note.txt").exists())

    def test_corrupt_archive_checksum_rejected(self):
        entry = self.make_archive()
        path = self.root / entry["path"]
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(artifacts.ArtifactError, "SHA-256"):
            artifacts.verify_archive(self.root, entry)

    def test_internal_hash_mismatch_rejected_with_valid_archive_checksum(self):
        entry = self.make_archive(mutate_manifest=lambda value: value["files"][0].update(sha256="0" * 64))
        with self.assertRaisesRegex(artifacts.ArtifactError, "SHA-256 mismatch"):
            artifacts.verify_archive(self.root, entry)

    def test_working_tree_manifest_can_use_null_git_blob(self):
        entry = self.make_archive(mutate_manifest=lambda value: [record.update(git_blob=None) for record in value["files"]])
        result = artifacts.verify_archive(self.root, entry)
        self.assertEqual(len(result.files), 2)

    def test_count_and_unpacked_size_rejected(self):
        entry = self.make_archive()
        for key in ("file_count", "unpacked_bytes", "size_bytes"):
            with self.subTest(key=key):
                invalid = {**entry, key: entry[key] + 1}
                with self.assertRaises(artifacts.ArtifactError):
                    artifacts.verify_archive(self.root, invalid)

    def test_unsafe_member_names_rejected(self):
        for index, name in enumerate(("../escape", "/absolute", "C:/drive", "safe/../../escape", "safe\\escape", "x:stream", "aux.txt", "trailing./x", ".git/config")):
            with self.subTest(name=name):
                entry = self.make_archive(f"unsafe-{index}", files={name: b"bad"})
                with self.assertRaises(artifacts.ArtifactError):
                    artifacts.verify_archive(self.root, entry)

    def test_links_directories_and_other_nonregular_members_rejected(self):
        for index, member_type in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE)):
            with self.subTest(member_type=member_type):
                info = tarfile.TarInfo("extra")
                info.type = member_type
                info.linkname = "../outside"
                entry = self.make_archive(f"type-{index}", extra_members=[(info, b"")])
                with self.assertRaisesRegex(artifacts.ArtifactError, "regular files"):
                    artifacts.verify_archive(self.root, entry)

    def test_duplicate_and_case_colliding_members_rejected(self):
        info = tarfile.TarInfo("docs/note.txt")
        info.size = 6
        entry = self.make_archive(extra_members=[(info, b"hello\n")])
        with self.assertRaisesRegex(artifacts.ArtifactError, "Duplicate"):
            artifacts.verify_archive(self.root, entry)
        entry = self.make_archive("case", files={"A.txt": b"a", "a.txt": b"a"})
        with self.assertRaisesRegex(artifacts.ArtifactError, "case-colliding"):
            artifacts.verify_archive(self.root, entry)

    def test_manifest_missing_file_and_source_mismatch_rejected(self):
        for index, mutation in enumerate((lambda value: value["files"].pop(), lambda value: value.update(source_commit="b" * 40))):
            with self.subTest(index=index):
                entry = self.make_archive(f"manifest-{index}", mutate_manifest=mutation)
                with self.assertRaises(artifacts.ArtifactError):
                    artifacts.verify_archive(self.root, entry)

    def test_all_archives_verified_before_any_destination_write(self):
        first = self.make_archive("good", files={"a/new": b"first"})
        second = self.make_archive("bad", files={"b/new": b"second"})
        second["sha256"] = "0" * 64
        destination = self.root / "output"
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.restore_archives(self.root, [first, second], destination)
        self.assertFalse(destination.exists())

    def test_all_destinations_checked_before_any_write(self):
        entry = self.make_archive(files={"new/file": b"new", "existing.txt": b"archive"})
        destination = self.root / "output"
        destination.mkdir()
        (destination / "existing.txt").write_bytes(b"user content")
        with self.assertRaisesRegex(artifacts.ArtifactError, "overwrite"):
            artifacts.restore_archives(self.root, [entry], destination)
        self.assertFalse((destination / "new").exists())
        self.assertEqual((destination / "existing.txt").read_bytes(), b"user content")

    def test_different_versions_across_selected_archives_rejected_before_writes(self):
        first = self.make_archive("one", files={"same": b"first"})
        second = self.make_archive("two", files={"same": b"second"})
        destination = self.root / "output"
        with self.assertRaisesRegex(artifacts.ArtifactError, "different versions"):
            artifacts.restore_archives(self.root, [first, second], destination)
        self.assertFalse(destination.exists())

    def test_identical_files_across_archives_written_once(self):
        first = self.make_archive("one", files={"same": b"same"})
        second = self.make_archive("two", files={"same": b"same"})
        self.assertEqual(artifacts.restore_archives(self.root, [first, second], self.root / "output"), (1, 0))

    def test_file_directory_collisions_across_archives_prevent_writes(self):
        first = self.make_archive("one", files={"parent": b"file"})
        second = self.make_archive("two", files={"parent/child": b"child"})
        destination = self.root / "output"
        with self.assertRaisesRegex(artifacts.ArtifactError, "File/directory"):
            artifacts.restore_archives(self.root, [first, second], destination)
        self.assertFalse(destination.exists())

    def test_symlink_destination_escape_rejected(self):
        entry = self.make_archive(files={"linked/new": b"data"})
        destination, outside = self.root / "output", self.root / "outside"
        destination.mkdir()
        outside.mkdir()
        try:
            (destination / "linked").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(artifacts.ArtifactError, "symlink or reparse"):
            artifacts.restore_archives(self.root, [entry], destination)
        self.assertFalse((outside / "new").exists())

    def test_reparse_point_detection_without_privileges(self):
        fake_stat = mock.Mock(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        with mock.patch.object(Path, "lstat", return_value=fake_stat):
            with self.assertRaisesRegex(artifacts.ArtifactError, "reparse point"):
                artifacts._lstat(self.root)

    def test_catalog_duplicate_ids_and_unsafe_archive_path_rejected(self):
        entry = self.make_archive()
        self.entries.append(dict(entry))
        self.save_catalog()
        with self.assertRaisesRegex(artifacts.ArtifactError, "Duplicate archive ID"):
            artifacts.load_catalog(self.catalog)
        self.entries.pop()
        entry["path"] = "../outside.tar.gz"
        self.save_catalog()
        with self.assertRaisesRegex(artifacts.ArtifactError, "Unsafe"):
            artifacts.load_catalog(self.catalog)

    def test_long_paths_roundtrip(self):
        name = "/".join(["segment" + "x" * 45] * 6) + "/result.txt"
        entry = self.make_archive(files={name: b"long path"})
        destination = self.root / "output"
        self.assertEqual(artifacts.restore_archives(self.root, [entry], destination), (1, 0))
        self.assertEqual(artifacts._fs(destination / name).read_bytes(), b"long path")


if __name__ == "__main__":
    unittest.main()
