#!/usr/bin/env python3
"""List, verify and restore the repository's explicitly selected artifact archives.

Only the Python standard library is required. Archive paths in a catalog are
relative to the repository root (the parent of the catalog's directory). Files
inside each archive retain their original repository-relative paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import zlib
from dataclasses import dataclass
from typing import BinaryIO


MANIFEST_NAME = "CARE_ARCHIVE_MANIFEST.json"
CHUNK_SIZE = 1024 * 1024
DEFAULT_CATALOG = Path(__file__).absolute().parent.parent / "artifacts" / "catalog.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_BLOB = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_WINDOWS_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)


class ArtifactError(Exception):
    """An archive, catalog, or destination failed validation."""


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class VerifiedArchive:
    entry: dict
    path: Path
    files: dict[str, FileRecord]
    signature: tuple[int, int]


def _absolute(path: str | Path) -> Path:
    # Do not resolve symlinks: their presence must remain visible to lstat.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _fs(path: Path) -> Path:
    """Use extended Windows paths without requiring a machine-wide setting."""
    if os.name != "nt":
        return path
    value = str(_absolute(path))
    if value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def _safe_relative(value: object, label: str = "path") -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ArtifactError(f"Unsafe {label}: {value!r}")
    parts = value.split("/")
    for part in parts:
        if (
            part in ("", ".", "..")
            or part[-1:] in (" ", ".")
            or any(ord(char) < 32 or char in '<>\"|?*' for char in part)
            or _WINDOWS_RESERVED.fullmatch(part)
        ):
            raise ArtifactError(f"Unsafe {label}: {value!r}")
    if parts[0].casefold() == ".git":
        raise ArtifactError(f"Refusing a path inside Git metadata: {value!r}")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactError(f"{label} must be a nonnegative integer")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(stream: BinaryIO, label: str) -> dict:
    try:
        # Metadata is held in memory; archive file contents are always streamed.
        # json.load accepts binary streams; TextIOWrapper cannot wrap tarfile's
        # non-seekable streaming reader on all supported Python versions.
        result = json.load(stream, object_pairs_hook=_object_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Invalid JSON in {label}: {exc}") from exc
    if not isinstance(result, dict) or type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ArtifactError(f"Unsupported schema in {label}; expected schema_version 1")
    return result


def _lstat(path: Path):
    try:
        info = _fs(path).lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ArtifactError(f"Refusing symlink or reparse point: {path}")
    return info


def _check_ancestors(path: Path, leaf_directory: bool = False) -> None:
    """Reject links and non-directory ancestors, including above destination."""
    chain = list(reversed(path.parents)) + [path]
    for index, component in enumerate(chain):
        info = _lstat(component)
        if info is not None and (index < len(chain) - 1 or leaf_directory):
            if not stat.S_ISDIR(info.st_mode):
                raise ArtifactError(f"Expected a directory: {component}")


def _hash_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _hash_file(path: Path) -> tuple[str, int]:
    with _fs(path).open("rb") as stream:
        return _hash_stream(stream)


def _signature(path: Path) -> tuple[int, int]:
    info = _lstat(path)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise ArtifactError(f"Expected a regular file: {path}")
    return info.st_size, info.st_mtime_ns


def _check_path_collisions(paths) -> None:
    # Be portable: archives must be unambiguous on case-insensitive Windows too.
    names = {}
    for name in paths:
        key = name.casefold()
        if key in names:
            raise ArtifactError(f"Duplicate or case-colliding paths: {names[key]!r}, {name!r}")
        names[key] = name
    for key, name in names.items():
        parts = key.split("/")
        if any("/".join(parts[:index]) in names for index in range(1, len(parts))):
            raise ArtifactError(f"File/directory path collision: {name!r}")


def load_catalog(path: str | Path) -> tuple[Path, list[dict]]:
    catalog_path = _absolute(path)
    root = catalog_path.parent.parent
    _check_ancestors(catalog_path)
    with _fs(catalog_path).open("rb") as stream:
        catalog = _read_json(stream, str(catalog_path))
    entries = catalog.get("archives")
    if not isinstance(entries, list):
        raise ArtifactError("Catalog archives must be a list")
    if not isinstance(catalog.get("base_commit"), str) or not catalog["base_commit"]:
        raise ArtifactError("Catalog base_commit is required")
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ArtifactError("Each catalog archive must be an object")
        for key in ("id", "category", "source_ref", "source_commit"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ArtifactError(f"Archive {key} must be a nonempty string")
        if entry["id"] in ids:
            raise ArtifactError(f"Duplicate archive ID: {entry['id']}")
        ids.add(entry["id"])
        _safe_relative(entry.get("path"), "archive path")
        for key in ("file_count", "unpacked_bytes", "size_bytes"):
            _nonnegative_integer(entry.get(key), key)
        _digest(entry.get("sha256"), "Archive sha256")
        prefixes = entry.get("source_prefixes")
        if not isinstance(prefixes, list) or not all(isinstance(item, str) for item in prefixes):
            raise ArtifactError("Archive source_prefixes must be a list of strings")
    return root, entries


def select_archives(entries: list[dict], ids: list[str], all_archives: bool) -> list[dict]:
    if bool(ids) == all_archives:
        raise ArtifactError("Specify one or more archive IDs, or --all (not both)")
    by_id = {entry["id"]: entry for entry in entries}
    unknown = sorted(set(ids) - by_id.keys())
    if unknown:
        raise ArtifactError("Unknown archive ID(s): " + ", ".join(unknown))
    return entries if all_archives else [by_id[item] for item in dict.fromkeys(ids)]


def verify_archive(root: Path, entry: dict) -> VerifiedArchive:
    path = root.joinpath(*_safe_relative(entry["path"]).split("/"))
    _check_ancestors(path)
    before = _signature(path)
    if before[0] != entry["size_bytes"]:
        raise ArtifactError(f"{entry['id']}: archive size does not match catalog")
    digest, _ = _hash_file(path)
    if digest != entry["sha256"]:
        raise ArtifactError(f"{entry['id']}: archive SHA-256 does not match catalog")
    actual: dict[str, FileRecord] = {}
    manifest = None
    names = []
    with tarfile.open(str(_fs(path)), mode="r|gz") as archive:
        for member in archive:
            name = _safe_relative(member.name, "tar member path")
            names.append(name)
            if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.sparse is not None:
                raise ArtifactError(f"{entry['id']}: only regular files are allowed: {name!r}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ArtifactError(f"Cannot read archive member: {name!r}")
            if name == MANIFEST_NAME:
                if manifest is not None:
                    raise ArtifactError("Duplicate archive manifest")
                with stream:
                    manifest = _read_json(stream, MANIFEST_NAME)
            else:
                if name in actual:
                    raise ArtifactError(f"Duplicate tar member: {name!r}")
                with stream:
                    digest, size = _hash_stream(stream)
                if size != member.size:
                    raise ArtifactError(f"Truncated tar member: {name!r}")
                actual[name] = FileRecord(name, size, digest)
    _check_path_collisions(names)
    if manifest is None:
        raise ArtifactError(f"{entry['id']}: missing {MANIFEST_NAME}")
    for key in ("source_ref", "source_commit"):
        if manifest.get(key) != entry[key]:
            raise ArtifactError(f"{entry['id']}: manifest {key} does not match catalog")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise ArtifactError("Manifest files must be a list")
    expected = {}
    for record in manifest_files:
        if not isinstance(record, dict):
            raise ArtifactError("Manifest file record must be an object")
        name = _safe_relative(record.get("path"), "manifest path")
        if name == MANIFEST_NAME or name in expected:
            raise ArtifactError(f"Invalid or duplicate manifest file path: {name!r}")
        size = _nonnegative_integer(record.get("size"), f"{name} size")
        digest = _digest(record.get("sha256"), f"{name} sha256")
        blob = record.get("git_blob")
        # A newly packed working-tree file may not have a Git object yet.
        if "git_blob" not in record or (blob is not None and (not isinstance(blob, str) or not _GIT_BLOB.fullmatch(blob))):
            raise ArtifactError(f"{name}: git_blob must be a Git object ID or null")
        expected[name] = FileRecord(name, size, digest)
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ArtifactError(f"{entry['id']}: manifest/member mismatch; missing={missing!r}, extra={extra!r}")
    for name, record in actual.items():
        if record != expected[name]:
            raise ArtifactError(f"{entry['id']}: size or SHA-256 mismatch for {name!r}")
    if len(actual) != entry["file_count"] or sum(item.size for item in actual.values()) != entry["unpacked_bytes"]:
        raise ArtifactError(f"{entry['id']}: file count or unpacked size does not match catalog")
    if _signature(path) != before:
        raise ArtifactError(f"Archive changed during verification: {path}")
    return VerifiedArchive(entry, path, actual, before)


def _target_status(target: Path, record: FileRecord) -> bool:
    """Return True for an identical existing file, otherwise require absence."""
    _check_ancestors(target)
    info = _lstat(target)
    if info is None:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise ArtifactError(f"Destination is not a regular file: {target}")
    if info.st_size == record.size and _hash_file(target)[0] == record.sha256:
        return True
    raise ArtifactError(f"Refusing to overwrite a different existing file: {target}")


def _make_directories(path: Path) -> None:
    for component in list(reversed(path.parents)) + [path]:
        info = _lstat(component)
        if info is None:
            try:
                _fs(component).mkdir()
            except FileExistsError:
                pass
            info = _lstat(component)
        if info is None or not stat.S_ISDIR(info.st_mode):
            raise ArtifactError(f"Expected a directory: {component}")


def restore_archives(root: Path, entries: list[dict], destination: str | Path | None = None) -> tuple[int, int]:
    # Complete verification and destination preflight before any mkdir or write.
    verified = [verify_archive(root, entry) for entry in entries]
    destination = _absolute(destination if destination is not None else root)
    _check_ancestors(destination, leaf_directory=True)
    files: dict[str, FileRecord] = {}
    for archive in verified:
        for name, record in archive.files.items():
            previous = files.get(name)
            if previous is not None and previous != record:
                raise ArtifactError(f"Selected archives contain different versions of {name!r}; restore separately")
            files[name] = record
    _check_path_collisions(files)
    targets = {name: destination.joinpath(*name.split("/")) for name in files}
    already_present = {name for name, record in files.items() if _target_status(targets[name], record)}
    written = set()
    for archive in verified:
        _check_ancestors(archive.path)
        if _signature(archive.path) != archive.signature:
            raise ArtifactError(f"Archive changed after verification: {archive.path}")
        with tarfile.open(str(_fs(archive.path)), mode="r|gz") as tar:
            for member in tar:
                name = member.name
                if name == MANIFEST_NAME or name in already_present or name in written:
                    continue
                record = archive.files.get(name)
                if record is None or member.size != record.size or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    raise ArtifactError(f"Archive changed after verification: {archive.path}")
                target = targets[name]
                if _target_status(target, record):
                    already_present.add(name)
                    continue
                _make_directories(target.parent)
                _check_ancestors(target)
                source = tar.extractfile(member)
                if source is None:
                    raise ArtifactError(f"Cannot read archive member: {name!r}")
                digest = hashlib.sha256()
                size = 0
                # Exclusive creation prevents overwrites, including a new symlink.
                # An interrupted write may leave a partial file; nothing is deleted.
                with source, _fs(target).open("xb") as output:
                    while chunk := source.read(CHUNK_SIZE):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != record.size or digest.hexdigest() != record.sha256:
                    raise ArtifactError(f"Archive changed during restore; partial file retained: {target}")
                written.add(name)
    if len(written | already_present) != len(files):
        raise ArtifactError("Archive changed during restore; some selected files were not restored")
    return len(written), len(already_present)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="catalog JSON (default: repository/artifacts/catalog.json)")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list archived artifacts")
    listing.add_argument("--category", help="show only this category")
    for command in ("verify", "restore"):
        child = commands.add_parser(command, help=f"{command} explicitly selected archives")
        child.add_argument("ids", metavar="ID", nargs="*")
        child.add_argument("--all", action="store_true", help="select every archive")
        if command == "restore":
            child.add_argument("--destination", type=Path, help="output directory (default: repository root)")
    args = parser.parse_args(argv)
    try:
        root, entries = load_catalog(args.catalog)
        if args.command == "list":
            print("ID\tCATEGORY\tFILES\tUNPACKED_BYTES\tARCHIVE_BYTES\tPATH")
            for entry in entries:
                if args.category is None or entry["category"] == args.category:
                    print("\t".join(str(entry[key]) for key in ("id", "category", "file_count", "unpacked_bytes", "size_bytes", "path")))
            return 0
        selected = select_archives(entries, args.ids, args.all)
        if args.command == "verify":
            for entry in selected:
                result = verify_archive(root, entry)
                print(f"Verified {entry['id']}: {len(result.files)} files")
        else:
            written, skipped = restore_archives(root, selected, args.destination)
            print(f"Restored {written} files; skipped {skipped} identical existing files")
        return 0
    except (ArtifactError, OSError, tarfile.TarError, EOFError, zlib.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
