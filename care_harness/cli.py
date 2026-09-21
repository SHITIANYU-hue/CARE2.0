"""Run paired CARE, Codex and frozen EvE policies through one experiment loop."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from .environment import ReplayEnvironment, load_task
from .legacy import REPO_ROOT
from .runner import run_episode

BASE_COMMIT = "a0a8c309252ce66e979a6ae421ec3278362203a4"


def method_config(task_config: dict, method: dict) -> dict:
    from .agents.care import load_pair_config
    pair_id = task_config.get("care_pair") or method.get("pair_id")
    pair = load_pair_config(pair_id) if pair_id else {}
    config = {**pair, **method}
    selection_path = config.get("selection_file")
    if selection_path:
        selection = json.loads(Path(selection_path).read_text())
        if selection["source_dataset"] != task_config.get("source_dataset", pair.get("source_dataset")) or selection["target_dataset"] != task_config.get("target_dataset", pair.get("target_dataset")):
            raise ValueError("CARE selection artifact belongs to a different source-target pair")
        execution = dict(selection["execution"])
        execution["source_observations"] = execution.pop("source_observation_count")
        execution["target_llm_mode"] = execution.pop("target_anchor_mode")
        config.update(execution)
        for field, expected in selection["record_sha256"].items():
            actual = hashlib.sha256(Path(config[field]).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Frozen CARE record mismatch: {field}")
        config.update(deployment_mode=selection["deployment_mode"], calibration_status="frozen_offline_selection")
        config["calibration_provenance"] = selection
    if config["type"] == "eve" and config.get("solver"):
        provenance_path = Path(config["solver"]).parent / "provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            config["evolution_provenance"] = provenance
            evolution = provenance.get("evolution") or {}
            development = evolution.get("development", {})
            known_datasets = {row[key] for row in development.get("tasks", []) if isinstance(row, dict)
                              for key in ("source_dataset", "target_dataset") if key in row}
            target_id = task_config.get("target_dataset", pair.get("target_dataset"))
            if target_id in known_datasets:
                raise ValueError("Target test outcomes were available in the declared EvE development tasks")
    return config


def make_agent(config: dict, seed: int, workspace: Path | None = None):
    method = config["type"]
    if method == "random":
        from .agents.random import RandomAgent
        return RandomAgent(seed)
    if method == "target_gp":
        from .agents.care import TargetGPAgent
        return TargetGPAgent(config)
    if method == "care":
        from .agents.care import CareAgent
        return CareAgent(config)
    if method == "codex":
        from .agents.codex import CodexAgent
        return CodexAgent(config, workspace or Path(tempfile.mkdtemp(prefix="care-codex-")))
    if method == "eve":
        from .agents.eve import EvEAgent
        return EvEAgent(config["solver"], python_executable=config.get("python"),
                        timeout=config.get("timeout_seconds", 60))
    raise ValueError(f"Unknown method: {method}")


def summarize(results: list[dict], reference: str) -> dict:
    """Pair only the same task/seed and report failures separately from scores."""
    references = {(r["case_id"], r["seed"]): r for r in results
                  if r["method"] == reference and r["status"] == "complete"}
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["case_id"], row["method"])].append(row)
    summaries = []
    for (case, method), rows in sorted(grouped.items()):
        completed = [r for r in rows if r["status"] == "complete"]
        pairs = [(r, references[(case, r["seed"])]) for r in completed if (case, r["seed"]) in references]
        deltas = [r["metrics"]["best_so_far_auc"] - ref["metrics"]["best_so_far_auc"] for r, ref in pairs]
        normalized = [r["metrics"]["normalized_best_so_far_auc"] - ref["metrics"]["normalized_best_so_far_auc"] for r, ref in pairs]
        summaries.append({
            "case_id": case, "method": method, "requested": len(rows),
            "completed": len(completed), "failed": len(rows) - len(completed),
            "mean_auc": statistics.mean(r["metrics"]["best_so_far_auc"] for r in completed) if completed else None,
            "mean_final_best": statistics.mean(r["metrics"]["final_best"] for r in completed) if completed else None,
            "paired_reference": reference, "paired_count": len(pairs),
            "paired_auc_delta": statistics.mean(deltas) if deltas else None,
            "paired_normalized_auc_delta": statistics.mean(normalized) if normalized else None,
            "wins_ties_losses": [sum(x > 0 for x in deltas), sum(x == 0 for x in deltas), sum(x < 0 for x in deltas)],
            "total_usage": {k: sum(r.get("usage", {}).get(k, 0) for r in rows)
                            for k in sorted({k for r in rows for k in r.get("usage", {})})},
        })
    return {"reference": reference, "results": summaries,
            "interpretation": "Paired seed results conditional on these tasks and frozen policies; not independent task generalization. Failed campaigns are reported, not imputed as baseline performance."}


def run_config(config: dict, output: Path, methods: list[str] | None = None,
               seeds: list[int] | None = None, dry_run: bool = False) -> dict:
    from .agents.care import CareAgent, load_pair_config
    selected = methods or config.get("default_methods") or list(config["methods"])
    if set(selected) - set(config["methods"]):
        raise ValueError("Requested method is absent from the configuration")
    protocol = config.get("protocol", {})
    seeds = seeds if seeds is not None else protocol.get("seeds", [0])
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds would overwrite campaign outputs")
    plans = []
    for spec in config["tasks"]:
        pair = load_pair_config(spec["care_pair"]) if spec.get("care_pair") else {}
        task_config = {**pair, **spec}
        initial = protocol.get("initial_observations", pair.get("initial_observations", 5))
        rounds = protocol.get("reveal_rounds", pair.get("reveal_rounds", 10))
        plans.append((spec.get("id") or task_config.get("target_dataset") or Path(task_config["task_file"]).stem,
                      task_config, initial, rounds))
    if len({name for name, *_ in plans}) != len(plans):
        raise ValueError("Duplicate case IDs: give each source-target task its own id")
    plan = {"methods": selected, "seeds": seeds,
            "cases": [{"case_id": name, "initial_observations": initial, "reveal_rounds": rounds}
                      for name, _, initial, rounds in plans],
            "campaign_count": len(plans) * len(seeds) * len(selected)}
    if dry_run:
        return plan
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = {"schema": "care.baseline_run/v1", "base_commit": BASE_COMMIT,
                "checkout_commit": commit, "python": sys.version, "config": config,
                "plan": plan, "resolved_methods": {},
                "initialization": protocol.get("initialization", "random"),
                "external_agent_boundary": "public JSON staging; local CLI/solver execution is not host filesystem isolation"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for case_id, task_config, initial, rounds in plans:
        resolved = {name: method_config(task_config, config["methods"][name]) for name in selected}
        # Common data and initialization must not change when method jobs are run
        # separately (e.g. CARE today, Codex after API credit is restored).
        frozen = [resolved[name] if name in resolved else method_config(task_config, options)
                  for name, options in config["methods"].items()
                  if options["type"] == "care" and options.get("selection_file")]
        if config.get("initializer", {}).get("selection_file"):
            frozen.insert(0, method_config(task_config, {"type": "care", **config["initializer"]}))
        if frozen:
            selected_care = frozen[0]
            if protocol.get("initialization") != "care":
                raise ValueError("Frozen CARE deployment requires initialization='care' to preserve its calibrated design")
            if initial != selected_care["initial_observations"] or rounds != selected_care["reveal_rounds"]:
                raise ValueError("Experiment budget differs from the frozen CARE calibration")
            for options in frozen:
                artifact = options["calibration_provenance"]
                calibration = range(artifact["calibration_seed_start"], artifact["calibration_seed_start"] + artifact["calibration_seed_count"])
                if any(seed in calibration for seed in seeds):
                    raise ValueError("Evaluation seeds overlap CARE calibration seeds")
            task_config = {**task_config, "source_seed": selected_care["source_seed"],
                           "source_observations": selected_care["source_observations"]}
        task = load_task(task_config)
        for options in resolved.values():
            evolution = (options.get("evolution_provenance", {}).get("evolution") or {})
            if task.task_id in evolution.get("development_task_ids", []):
                raise ValueError("Target test outcomes were available in the declared EvE development tasks")
        for value in resolved.values():
            for key in ("llm_record", "target_llm_record", "solver"):
                if value.get(key) and Path(value[key]).is_file():
                    value[key + "_sha256"] = hashlib.sha256(Path(value[key]).read_bytes()).hexdigest()
        manifest["resolved_methods"][case_id] = resolved
        manifest.setdefault("shared_protocol", {})[case_id] = {
            "frozen_care": frozen[0].get("calibration_provenance") if frozen else None,
            "initial_observations": initial, "reveal_rounds": rounds,
        }
        manifest.setdefault("task_data", {})[case_id] = {
            "public_candidates_sha256": hashlib.sha256(json.dumps(task.candidates, sort_keys=True).encode()).hexdigest(),
            "outcomes_sha256": hashlib.sha256(json.dumps(task.outcomes, sort_keys=True).encode()).hexdigest(),
            "source_sha256": hashlib.sha256(json.dumps(task.source, sort_keys=True).encode()).hexdigest(),
            "candidate_count": len(task.candidates), "source_count": len(task.source["observations"]),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for seed in seeds:
            initial_ids = None
            if protocol.get("initialization", "random") == "care":
                initializer_config = frozen[0] if frozen else method_config(task_config, {"type": "care", **config.get("initializer", {})})
                public = ReplayEnvironment(task, seed, initial, rounds).view()
                public["observations"] = []
                initial_ids = CareAgent(initializer_config).initial_candidate_ids(public)
            elif protocol.get("initialization", "random") != "random":
                raise ValueError("initialization must be random or care")
            for name, options in resolved.items():
                run_dir = output / case_id / name / f"seed-{seed}"
                workspace = Path(tempfile.mkdtemp(prefix="care-codex-")) if options["type"] == "codex" else None
                try:
                    agent = make_agent(options, seed, workspace)
                    result = run_episode(task, agent, seed, initial, rounds, initial_ids, run_dir)
                except Exception as exc:
                    failure_path = run_dir / "result.json"
                    result = json.loads(failure_path.read_text()) if failure_path.exists() else {
                        "status": "failed", "task_id": task.task_id, "seed": seed,
                        "error": str(exc), "error_type": type(exc).__name__}
                    print(f"{case_id}/{name}/{seed}: FAILED: {exc}", file=sys.stderr)
                else:
                    print(f"{case_id}/{name}/{seed}: AUC={result['metrics']['best_so_far_auc']:.4f}")
                result.update(case_id=case_id, method=name)
                if workspace:
                    result["agent_workspace"] = str(workspace)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "result.json").write_text(json.dumps(result, indent=2))
                results.append(result)
    reference = config.get("reference", selected[0])
    summary = summarize(results, reference)
    summary["status"] = "complete" if all(r["status"] == "complete" for r in results) else "partial_failure"
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    fields = ["case_id", "method", "seed", "status", "final_best", "best_so_far_auc",
              "normalized_best_so_far_auc", "wall_seconds"]
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({**row, **row.get("metrics", {})})
    return summary


def freeze_selection(summary_path: Path, output: Path) -> dict:
    """Export a deployment choice, without importing held-out target outcomes."""
    payload = json.loads(summary_path.read_text())
    skill = payload["transfer_skill"]
    artifact = {
        "source_dataset": payload["source_dataset"], "target_dataset": payload["target_dataset"],
        "deployment_mode": payload["selection"]["selected_mode"],
        "calibration_seed_start": payload["calibration_seed_start"],
        "calibration_seed_count": payload["calibration_seed_count"],
        "offline_selection_cost": payload.get("offline_selection_cost", {}),
        "execution": skill["execution"],
        "record_sha256": {
            "llm_record": skill["provenance"]["source_patch_record_sha256"],
            "target_llm_record": skill["provenance"]["target_anchor_record_sha256"],
        },
        "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2))
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run methods against identical task/reveal protocols")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--methods", help="Comma-separated method names")
    run.add_argument("--seeds", help="Comma-separated integer campaign seeds")
    run.add_argument("--output", type=Path, default=REPO_ROOT / "runs/baselines")
    run.add_argument("--dry-run", action="store_true")
    freeze = sub.add_parser("freeze-care", help="Export deployment choice from a canonical calibration summary")
    freeze.add_argument("--summary", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-care":
        print(json.dumps(freeze_selection(args.summary, args.output), indent=2))
        return 0
    config = json.loads(args.config.read_text())
    result = run_config(config, args.output,
                        args.methods.split(",") if args.methods else None,
                        [int(x) for x in args.seeds.split(",")] if args.seeds else None,
                        args.dry_run)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "complete") == "complete" else 1
