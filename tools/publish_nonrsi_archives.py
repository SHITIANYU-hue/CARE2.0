#!/usr/bin/env python3
"""Publish explicit, size-limited file groups without changing run directories.

This publication wrapper reuses pack_run's original-path manifest, streaming
writer and archive verifier. A plan records every source byte before packaging;
the executing research code remains identified by its frozen protocol hashes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import artifacts
import pack_run


ROOT = Path(__file__).resolve().parents[1]
OUT = Path("experiments/care_replay/outputs")
AUDIT = Path("outputs/submission_handoff_20261109")
V10 = OUT / "nonrsi_cross_model_resume_v10"
FENCE = OUT / "nonrsi_fence_supplement_v1_20261108"
PREFIX = "nonrsi-publication-20261109"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pack_run._json_bytes(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def files_under(path):
    path = ROOT / path
    pack_run._check_path_links(path)
    if path.is_file():
        return [path.relative_to(ROOT).as_posix()]
    if not path.is_dir():
        raise pack_run.PackError(f"Missing source: {path}")
    paths = []
    for child in sorted(path.rglob("*")):
        pack_run._check_path_links(child)
        if child.is_file():
            paths.append(child.relative_to(ROOT).as_posix())
    return paths


def immediate_files(path):
    return [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / path).iterdir()) if p.is_file()]


def source_groups():
    groups = []

    def add(suffix, description, paths):
        paths = sorted(set(paths))
        if not paths:
            raise pack_run.PackError(f"Empty source group: {suffix}")
        groups.append({"id": f"{PREFIX}-{suffix}", "description": description, "paths": paths})

    earlier = [p for p in sorted((ROOT / OUT).glob("nonrsi_*"))
               if p.is_dir() and p.name not in {V10.name, FENCE.name}
               and not re.fullmatch(r"nonrsi_cross_model_resume_v[4-9]", p.name)]
    add("history-part01", "Design, preparation, original execution, freeze, integration, offline continuation, resume v1-v3",
        [f for p in earlier for f in files_under(p.relative_to(ROOT))])
    for suffix, versions in [("history-part02", range(4, 7)), ("history-part03", range(7, 10))]:
        add(suffix, "Stopped generation history and immutable journals: " + ", ".join(f"v{x}" for x in versions),
            [f for x in versions for f in files_under(OUT / f"nonrsi_cross_model_resume_v{x}")])
    small = ["real_matbench_dielectric", "real_matbench_expt_gap", "real_moleculenet_freesolv", "real_moleculenet_lipophilicity"]
    add("strict-control", "Strict generation journal, recovery/preflight records, evaluation definitions, locks and reports",
        [f for p in sorted((ROOT / V10).iterdir()) if p.name != "evaluation" for f in files_under(p.relative_to(ROOT))]
        + immediate_files(V10 / "evaluation"))
    add("strict-molecular-materials", "Strict calibration and held-out metrics/traces for four molecular/material targets",
        [f for name in small for f in files_under(V10 / "evaluation" / name)])
    for short, target in [("bh", "real_buchwald_hartwig"), ("suzuki", "real_suzuki_miyaura")]:
        add("strict-" + short, "Strict complete calibration and held-out target replay: " + target,
            files_under(V10 / "evaluation" / target))
    add("fence-control", "Post-hoc supplement protocol, unchanged source bindings, normalized contents, definitions, locks and reports",
        [f for p in sorted((ROOT / FENCE).iterdir()) if p.name != "evaluation" for f in files_under(p.relative_to(ROOT))]
        + immediate_files(FENCE / "evaluation"))
    add("fence-molecular-materials", "Supplement calibration and held-out metrics/traces for four molecular/material targets",
        [f for name in small for f in files_under(FENCE / "evaluation" / name)])
    for short, target in [("bh", "real_buchwald_hartwig"), ("suzuki", "real_suzuki_miyaura")]:
        folder = FENCE / "evaluation" / target
        add("fence-" + short + "-calibration", "Supplement calibration target replay: " + target,
            files_under(folder / "calibration"))
        add("fence-" + short + "-heldout", "Supplement held-out target replay, fixed selection and aggregate metrics: " + target,
            files_under(folder / "heldout") + immediate_files(folder))
    base = json.loads((ROOT / OUT / "nonrsi_cross_model_freeze_v1/FROZEN_PROTOCOL.json").read_text())
    dependencies = {p for p in base["files"] if not p.startswith(str(OUT) + "/nonrsi_")}
    for folder in sorted((ROOT / "experiments/care_replay/protocols").glob("nonrsi_*")):
        dependencies.update(p.relative_to(ROOT).as_posix() for p in folder.glob("*.py"))
    dependencies.update({".python-version", "pyproject.toml", "uv.lock", "tools/publish_nonrsi_archives.py"})
    add("dependencies", "Frozen input/code closure, revision sources, environment, and the four referenced pre-existing audit files",
        dependencies)
    return groups, base


def secret_scan(records):
    # Report paths and rule names only. Never serialize matching bytes or secrets.
    known = []
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(errors="replace").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if re.search(r"key|secret|token|password|credential", key, re.I) and len(value) >= 12:
                known.append(value.encode())
    patterns = {
        "private_key": re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
        "token_shape": re.compile(rb"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{24,}"),
        "bearer_header": re.compile(rb"(?i)authorization[\"'\s]*[:=][\"'\s]*bearer\s+[A-Za-z0-9_.-]{16,}"),
    }
    findings = []
    for record in records:
        name = record["path"]
        parts = Path(name).parts
        if any(p in {".env", ".git", ".venv"} for p in parts) or Path(name).suffix.lower() in {".pem", ".key", ".p12"}:
            findings.append({"path": name, "rule": "forbidden_filename"})
        content = (ROOT / name).read_bytes()
        if known and any(value in content for value in known):
            findings.append({"path": name, "rule": "known_environment_secret"})
        for rule, pattern in patterns.items():
            if pattern.search(content):
                findings.append({"path": name, "rule": rule})
    return {"scope": "All source payload bytes, forbidden filenames, known local environment secret values, private-key and credential-header/token patterns; matching values never logged",
            "files_scanned": len(records), "known_secret_values_checked": len(known), "findings": findings}


def make_plan(path):
    groups, base = source_groups()
    all_names = [p for group in groups for p in group["paths"]]
    if len(all_names) != len(set(all_names)):
        raise pack_run.PackError("Source groups overlap")
    artifacts._check_path_collisions(all_names)
    contents = defaultdict(list)
    all_records = []
    for group in groups:
        records = []
        for name in group.pop("paths"):
            artifacts._safe_relative(name)
            size, sha = pack_run._hash_file(ROOT / name)
            record = {"path": name, "size": size, "sha256": sha, "git_blob": None}
            records.append(record)
            contents[(size, sha)].append(name)
        size = sum(r["size"] for r in records)
        if size > pack_run.MAX_UNPACKED_BYTES:
            raise pack_run.PackError(f"Oversize group: {group['id']}")
        group.update(files=records, file_count=len(records), unpacked_bytes=size)
        all_records.extend(records)
    by_name = {r["path"]: r for r in all_records}
    mismatches = [p for p, sha in base["files"].items() if p not in by_name or by_name[p]["sha256"] != sha]
    if mismatches:
        raise pack_run.PackError("Frozen dependency mismatch: " + ", ".join(mismatches))
    expected_outputs = {f for p in sorted((ROOT / OUT).glob("nonrsi_*")) if p.is_dir() for f in files_under(p.relative_to(ROOT))}
    archived_outputs = {p for p in by_name if p.startswith(str(OUT) + "/nonrsi_")}
    if expected_outputs != archived_outputs:
        raise pack_run.PackError("Output inventory is incomplete")
    scan = secret_scan(all_records)
    plan = {"schema_version": 1, "kind": "nonrsi_complete_publication_plan", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_provenance": "Publication snapshot only; actual execution identity is bound by the preserved original freeze/config/code SHA-256 records, not publication HEAD",
            "limits": {"max_source_bytes": pack_run.MAX_UNPACKED_BYTES, "max_archive_bytes": pack_run.MAX_ARCHIVE_BYTES},
            "output_scope": sorted({p.name for p in (ROOT / OUT).glob("nonrsi_*") if p.is_dir()}),
            "groups": groups, "total_files": len(all_records), "total_bytes": sum(r["size"] for r in all_records),
            "duplicate_content": {"different_content_blobs": len(contents), "duplicate_paths_retained": sum(len(v)-1 for v in contents.values()),
                                  "duplicate_bytes_retained": sum(size*(len(names)-1) for (size, _), names in contents.items())},
            "base_freeze_dependencies_verified": len(base["files"]), "secret_scan": scan,
            "exclusions": ["Repository .env and all credentials", "Unrelated root outputs, WSL migration outputs and TRACE_SKILL designs", "Protocol __pycache__ caches; original nonrsi output directories are fully preserved without filtering"]}
    save(path, plan)
    if scan["findings"]:
        raise pack_run.PackError("Sensitive scan findings; review paths in archive_plan.json (values suppressed)")
    print(json.dumps({"plan": str(path), "groups": [{k: v for k, v in g.items() if k != "files"} for g in groups],
                      "files": plan["total_files"], "bytes": plan["total_bytes"], "sensitive_findings": 0}, indent=2), flush=True)


def verify_sources(plan):
    records = [r for g in plan["groups"] for r in g["files"]]
    for record in records:
        if pack_run._hash_file(ROOT / record["path"]) != (record["size"], record["sha256"]):
            raise pack_run.PackError("Source changed: " + record["path"])
    return records


def publish(plan_path, source_commit, destination):
    plan = json.loads(plan_path.read_text())
    if plan["secret_scan"]["findings"]:
        raise pack_run.PackError("Sensitive scan findings remain")
    records = verify_sources(plan)
    # A new file added since planning must not silently escape a complete archive.
    actual, _ = source_groups()
    if {p for g in actual for p in g["paths"]} != {r["path"] for r in records}:
        raise pack_run.PackError("Source inventory changed since planning")
    result = subprocess.run(["git", "cat-file", "-t", source_commit], cwd=ROOT, capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != "commit":
        raise pack_run.PackError("Publication source commit must be a local Git commit")
    catalog_path = ROOT / "artifacts/catalog.json"
    lock = catalog_path.parent / ".catalog.pack.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    entries = []
    try:
        catalog, _ = pack_run._load_catalog(catalog_path, source_commit)
        for group in plan["groups"]:
            archive_id = group["id"]
            archive = ROOT / "artifacts/new-runs" / (archive_id + ".tar.gz")
            existing = next((e for e in catalog["archives"] if e["id"] == archive_id), None)
            manifest = {"schema_version": 1, "source_ref": "working-tree", "source_commit": source_commit, "files": group["files"]}
            if existing:
                pack_run._verify_archive(ROOT, existing, manifest)
                if existing["source_commit"] != source_commit:
                    raise pack_run.PackError("Publication commit differs from existing archive")
                entries.append(existing)
                continue
            if archive.exists():
                raise pack_run.PackError("Uncatalogued archive exists: " + str(archive))
            if group["unpacked_bytes"] > pack_run.MAX_UNPACKED_BYTES:
                raise pack_run.PackError("Oversize source group")
            descriptor, temporary_name = tempfile.mkstemp(prefix=".nonrsi-", suffix=".tar.gz.tmp", dir=archive.parent)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                pack_run._write_archive(temporary, ROOT, manifest)
                size, sha = pack_run._hash_file(temporary)
                if size >= pack_run.MAX_ARCHIVE_BYTES:
                    raise pack_run.PackError("Archive is not below 100 MiB")
                entry = {"id": archive_id, "category": "new-runs", "path": archive.relative_to(ROOT).as_posix(),
                         "source_ref": "working-tree", "source_commit": source_commit, "file_count": group["file_count"],
                         "unpacked_bytes": group["unpacked_bytes"], "size_bytes": size, "sha256": sha,
                         "source_prefixes": sorted({str(Path(r["path"]).parent) for r in group["files"]}),
                         "description": group["description"]}
                pack_run._verify_archive(ROOT, dict(entry, path=temporary.relative_to(ROOT).as_posix()), manifest)
                os.link(temporary, archive)
                catalog["archives"].append(entry)
                save(catalog_path, catalog)
                entries.append(entry)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Published {archive_id}: {group['file_count']} files, {size} archive bytes", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    verify_sources(plan)
    if destination.exists():
        raise pack_run.PackError("Clean restore destination already exists")
    restored, skipped = artifacts.restore_archives(ROOT, entries, destination)
    if skipped or restored != len(records):
        raise pack_run.PackError("Clean restore count mismatch")
    for record in records:
        if pack_run._hash_file(destination / record["path"]) != (record["size"], record["sha256"]):
            raise pack_run.PackError("Clean restore mismatch: " + record["path"])
    verify_sources(plan)
    audit = {"kind": "nonrsi_publication_verified", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "publication_source_commit": source_commit,
             "execution_identity": "Preserved frozen SHA-256 records; publication commit does not identify the original execution commit",
             "archives": entries, "source_file_count": len(records), "all_source_hashes_unchanged": True,
             "all_new_archives_fully_verified": True, "clean_restore_destination": str(destination),
             "restored_files": restored, "identical_existing_skips": skipped, "restored_file_hashes_all_match": True,
             "new_model_requests": 0, "secret_scan_findings": 0}
    save(ROOT / AUDIT / "archive_verification.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "archives"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["plan", "publish"])
    parser.add_argument("--plan", type=Path, default=ROOT / AUDIT / "archive_plan.json")
    parser.add_argument("--source-commit")
    parser.add_argument("--restore-destination", type=Path)
    args = parser.parse_args()
    if args.operation == "plan":
        make_plan(args.plan)
    else:
        if not args.source_commit or not args.restore_destination:
            parser.error("publish requires --source-commit and a new --restore-destination")
        publish(args.plan, args.source_commit, args.restore_destination)


if __name__ == "__main__":
    main()
