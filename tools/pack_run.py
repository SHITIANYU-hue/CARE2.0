#!/usr/bin/env python3
"""Preserve one completed run in a verified archive without removing its files.

Usage: python tools/pack_run.py PATH --id UNIQUE_ID [--category new-runs]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO
import zlib

if __package__:
    from . import artifacts
else:
    import artifacts


MANIFEST_NAME = "CARE_ARCHIVE_MANIFEST.json"
MAX_UNPACKED_BYTES = 90 * 1024 * 1024
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
FORBIDDEN_PARTS = frozenset({"artifacts", ".git", ".venv"})
SLUG = re.compile(r"[A-Za-z0-9_-]+\Z")
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "artifacts" / "catalog.json"


class PackError(ValueError):
    """The run cannot safely be archived using this request."""


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_link(info: os.stat_result) -> bool:
    # st_file_attributes covers Windows junctions as well as symbolic links.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _check_path_links(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if _is_link(info):
            raise PackError(f"Symlinks and junctions are not allowed: {component}")


def _member_name(path: str) -> str:
    parts = PurePosixPath(path).parts
    if (
        not path
        or "\\" in path
        or ":" in path
        or path.startswith("/")
        or any(part in ("", ".", "..") for part in path.split("/"))
        or not parts
    ):
        raise PackError(f"Unsafe archive member path: {path!r}")
    return path


def _open_regular(path: Path) -> BinaryIO:
    _check_path_links(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise PackError(f"Only regular files may be archived: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    handle = os.fdopen(descriptor, "rb")
    try:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise PackError(f"Source changed while opening: {path}")
        return handle
    except BaseException:
        handle.close()
        raise


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with _open_regular(path) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _collect_files(source: Path, root: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    pending = [source]
    total = 0
    while pending:
        directory = pending.pop()
        _check_path_links(directory)
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            info = path.lstat()
            if _is_link(info):
                raise PackError(f"Symlinks and junctions are not allowed: {path}")
            if path.name.casefold() in FORBIDDEN_PARTS:
                raise PackError(f"Excluded directory or file inside run: {path}")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
                if total > MAX_UNPACKED_BYTES:
                    raise PackError("Run exceeds 90 MiB unpacked; choose a smaller subdirectory.")
                paths.append(path)
            else:
                raise PackError(f"Only directories and regular files may be archived: {path}")
    if not paths:
        raise PackError("The source directory contains no files.")
    records = []
    total = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        size, digest = _hash_file(path)
        total += size
        if total > MAX_UNPACKED_BYTES:
            raise PackError("Run exceeds 90 MiB unpacked; choose a smaller subdirectory.")
        records.append({
            "path": _member_name(path.relative_to(root).as_posix()),
            "size": size,
            "sha256": digest,
            "git_blob": None,
        })
    return records


def _source_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "working-tree"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", commit) else "working-tree"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


class _HashingReader:
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size)
        self.size += len(chunk)
        self.digest.update(chunk)
        return chunk


def _write_archive(path: Path, root: Path, manifest: dict[str, Any]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, mtime=0, fileobj=raw) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                data = _json_bytes(manifest)
                archive.addfile(_tar_info(MANIFEST_NAME, len(data)), io.BytesIO(data))
                for record in manifest["files"]:
                    source_file = root / record["path"]
                    with _open_regular(source_file) as handle:
                        reader = _HashingReader(handle)
                        archive.addfile(_tar_info(record["path"], record["size"]), reader)
                        if (
                            reader.size != record["size"]
                            or reader.digest.hexdigest() != record["sha256"]
                            or handle.read(1)
                        ):
                            raise PackError(f"Source changed during packing: {source_file}")
        raw.flush()
        os.fsync(raw.fileno())


def _verify_archive(root: Path, entry: dict[str, Any], expected_manifest: dict[str, Any]) -> None:
    """Use the restore tool's verifier and compare with the source snapshot."""
    verified = artifacts.verify_archive(root, entry)
    expected = {
        record["path"]: artifacts.FileRecord(record["path"], record["size"], record["sha256"])
        for record in expected_manifest["files"]
    }
    if verified.files != expected:
        raise PackError("Archive does not match the original source snapshot.")


def _load_catalog(path: Path, commit: str) -> tuple[dict[str, Any], bytes | None]:
    if path.exists():
        artifacts.load_catalog(path)
    original = path.read_bytes() if path.exists() else None
    if original is None:
        return {"schema_version": 1, "base_commit": commit, "archives": []}, None
    try:
        catalog = json.loads(original)
    except (ValueError, UnicodeError) as error:
        raise PackError(f"Invalid catalog JSON: {path}") from error
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1 or not isinstance(catalog.get("archives"), list):
        raise PackError("Expected catalog schema_version 1 with an archives list.")
    ids: set[str] = set()
    for entry in catalog["archives"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or entry["id"] in ids:
            raise PackError("Catalog has invalid or duplicate archive IDs.")
        ids.add(entry["id"])
    return catalog, original


def pack_run(
    source: Path | str,
    archive_id: str,
    category: str = "new-runs",
    catalog_path: Path | str | None = None,
) -> dict[str, Any]:
    """Pack a directory inside the catalog's repository and return its new entry."""
    if not SLUG.fullmatch(archive_id) or not SLUG.fullmatch(category):
        raise PackError("ID and category must contain only letters, digits, hyphens, or underscores.")
    catalog_path = _absolute(DEFAULT_CATALOG if catalog_path is None else catalog_path)
    if catalog_path.name != "catalog.json" or catalog_path.parent.name != "artifacts":
        raise PackError("--catalog must name <repository>/artifacts/catalog.json.")
    root = catalog_path.parent.parent
    source = _absolute(source)
    _check_path_links(catalog_path)
    _check_path_links(source)
    try:
        relative_source = source.relative_to(root)
    except ValueError as error:
        raise PackError("The source directory must be inside the catalog's repository.") from error
    if not relative_source.parts or any(part.casefold() in FORBIDDEN_PARTS for part in relative_source.parts):
        raise PackError("Cannot pack the repository root, artifacts, .git, or .venv.")
    if not source.is_dir():
        raise PackError(f"Source must be an existing directory: {source}")
    destination = catalog_path.parent / category / f"{archive_id}.tar.gz"
    _check_path_links(destination)
    if destination.exists():
        raise PackError(f"Archive already exists: {destination}")

    records = _collect_files(source, root)
    commit = _source_commit(root)
    manifest = {
        "schema_version": 1,
        "source_ref": "working-tree",
        "source_commit": commit,
        "files": records,
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = catalog_path.parent / ".catalog.pack.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise PackError(f"Another pack operation may be active; lock exists: {lock_path}") from error
    temporary_archive: Path | None = None
    temporary_catalog: Path | None = None
    published = False
    committed = False
    try:
        os.close(lock_fd)
        catalog, original_catalog = _load_catalog(catalog_path, commit)
        if any(entry["id"] == archive_id for entry in catalog["archives"]):
            raise PackError(f"Archive ID already exists in catalog: {archive_id}")
        _check_path_links(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{archive_id}-", suffix=".tar.gz.tmp", dir=destination.parent)
        os.close(descriptor)
        temporary_archive = Path(name)
        _write_archive(temporary_archive, root, manifest)
        if temporary_archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise PackError("Compressed archive exceeds 100 MiB; choose a smaller subdirectory.")
        archive_size, archive_hash = _hash_file(temporary_archive)
        entry = {
            "id": archive_id,
            "category": category,
            "path": destination.relative_to(root).as_posix(),
            "source_ref": "working-tree",
            "source_commit": commit,
            "file_count": len(records),
            "unpacked_bytes": sum(record["size"] for record in records),
            "size_bytes": archive_size,
            "sha256": archive_hash,
            "source_prefixes": [relative_source.as_posix()],
        }
        temporary_entry = dict(entry, path=temporary_archive.relative_to(root).as_posix())
        _verify_archive(root, temporary_entry, manifest)
        catalog["archives"].append(entry)
        descriptor, name = tempfile.mkstemp(prefix=".catalog-", suffix=".json.tmp", dir=catalog_path.parent)
        temporary_catalog = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(catalog))
            handle.flush()
            os.fsync(handle.fileno())

        _check_path_links(destination)
        _check_path_links(catalog_path)
        current_catalog = catalog_path.read_bytes() if catalog_path.exists() else None
        if current_catalog != original_catalog:
            raise PackError("Catalog changed during packing; retry with the updated catalog.")
        # Hard-linking a completed temporary file publishes it atomically and
        # fails if the destination exists, unlike os.replace or POSIX rename.
        os.link(temporary_archive, destination)
        published = True
        os.replace(temporary_catalog, catalog_path)
        committed = True
        return entry
    finally:
        if published and not committed:
            destination.unlink()
        for temporary in (temporary_archive, temporary_catalog):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Completed run directory inside the repository")
    parser.add_argument("--id", required=True, help="Unique archive ID")
    parser.add_argument("--category", default="new-runs")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    try:
        entry = pack_run(args.path, args.id, args.category, args.catalog)
    except (PackError, artifacts.ArtifactError, OSError, tarfile.TarError, EOFError, zlib.error) as error:
        parser.exit(2, f"pack_run: {error}\n")
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
