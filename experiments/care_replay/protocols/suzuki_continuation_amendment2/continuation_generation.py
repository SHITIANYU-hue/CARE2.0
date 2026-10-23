"""Amendment 2 executor: original ten attempts plus exactly eleven authorized new POSTs."""
from bootstrap import *
from generation import *
from generation import _check_material
import continuation_integrity as integrity
def execute(config_path, lock_path, freeze_dir, output, *, api=None, sleep=time.sleep):
    config, lock, amendment = integrity.verify()
    if (Path(config_path),Path(lock_path),Path(freeze_dir),Path(output)) != (integrity.CONFIG,integrity.LOCK,integrity.FREEZE,integrity.NEW):
        raise Halt("Wrong continuation paths")
    output,freeze_dir=Path(output),Path(freeze_dir)
    if output.exists(): raise Halt("Continuation output exists; no automatic POST resume")
    output.mkdir(parents=True, exist_ok=False)  # Single durable claim before any request.
    for folder in ("receipts", "libraries", "outcomes"):
        (output / folder).mkdir()
    attempts = deepcopy(load(integrity.OLD/"ledger.json")["attempts"])
    outcomes = {p.stem:file_hash(p) for p in (integrity.OLD/"outcomes").glob("*.json")}
    libraries = {p.stem:file_hash(p) for p in (integrity.OLD/"libraries").glob("*.json")}
    known_new = integrity.KNOWN

    def write_ledger():
        save(output / "ledger.json", {"attempts": attempts,
             "actual_post_attempts": len(attempts), "prior_cost_usd": str(PRIOR_COST),
             "unknown_reservation_usd": str(sum((RESERVE for a in attempts if "cost_usd" not in a), Decimal(0))),
             "known_new_cost_usd": str(known_new),
             "cumulative_known_cost_usd": str(PRIOR_COST + known_new)})

    try:
        shutil.copytree(integrity.OLD,output/"original_evidence")
        for attempt in attempts[:9]:
            identifier=attempt["identifier"]
            shutil.copytree(integrity.OLD/"receipts"/identifier,output/"receipts"/identifier)
            shutil.copyfile(integrity.OLD/"outcomes"/(identifier+".json"),output/"outcomes"/(identifier+".json"))
            if identifier in libraries: shutil.copyfile(integrity.OLD/"libraries"/(identifier+".json"),output/"libraries"/(identifier+".json"))
        shutil.copyfile(integrity.AMENDMENT_LOCK,output/"amendment_lock.json")
        shutil.copyfile(config_path, output / "config.json")
        shutil.copyfile(lock_path, output / "protocol_lock.json")
        for group, folder, source_folder in (("messages_sha256", "messages", "messages"),
                                            ("pair_material_sha256", "pairs", "private_pairs")):
            destination = output / "input_snapshot" / folder
            destination.mkdir(parents=True)
            for identifier, expected in lock[group].items():
                path = destination / (identifier + ".json")
                shutil.copyfile(freeze_dir / source_folder / path.name, path)
                if file_hash(path) != expected:
                    raise Halt("Frozen input changed while copying snapshot")
        for relative, expected in lock["source_sha256"].items():
            path = output / "source_snapshot" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, path)
            if file_hash(path) != expected:
                raise Halt("Frozen source changed while copying snapshot")
        for group in ("data_sha256", "input_sha256"):
            for relative, expected in lock[group].items():
                path = output / "input_snapshot" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, path)
                if file_hash(path) != expected:
                    raise Halt("Frozen data/dependency changed while copying snapshot")
        if file_hash(output / "config.json") != lock["config_sha256"]:
            raise Halt("Config changed while copying snapshot")
        if api is None:
            api = OpenRouter(legacy.load_key(ROOT / ".env"), config["api"]["timeout_seconds"])
        model = next(m for m in api.request("/models")["data"] if m["id"] == config["model"]["id"])
        key_metadata = api.request("/key")["data"]
        budget_check = legacy.validate_budget(config, model, key_metadata)
        save(output / "preflight.json", {"checked_at": utc(), "budget": budget_check,
             "prior_cost_usd": str(PRIOR_COST), "cumulative_reserved_usd": str(integrity.KNOWN + integrity.UNKNOWN + 11 * RESERVE),
             "catalog_model": model, "key_authenticated": True, "secret_values_recorded": False}, exclusive=True)
        write_ledger()
        seen_response_ids = {a["response_id"] for a in attempts[:9]}
        for slot in config["generation_order"][9:]:
            integrity.check_budget(attempts,known_new)
            identifier = f"pair_{slot['pair_index']:02d}_{slot['arm']}"
            pair_id = f"pair_{slot['pair_index']:02d}"
            message_path = output / "input_snapshot/messages" / (identifier + ".json")
            material_path = output / "input_snapshot/pairs" / (pair_id + ".json")
            if (file_hash(message_path) != lock["messages_sha256"][identifier] or
                    file_hash(material_path) != lock["pair_material_sha256"][pair_id]):
                raise Halt("Frozen message or pair snapshot changed before POST")
            messages, material = load(message_path), load(material_path)
            if (material.get("pair_index") != slot["pair_index"] or
                    material.get("alias_seed") != config["paired_design"]["alias_seed_by_pair"][pair_id]):
                raise Halt("Pair material identity differs from frozen slot")
            _check_material(material, object_sha256(material))
            request = {"model": config["model"]["id"], "messages": messages,
                       "temperature": config["model"]["temperature"], "max_tokens": config["model"]["max_tokens"],
                       "response_format": {"type": "json_object"}, "stream": False,
                       "provider": config["api"]["provider"]}
            if len(json.dumps(messages, ensure_ascii=False).encode()) + 4096 + config["model"]["max_tokens"] > config["api"]["context_token_bound"]:
                raise Halt("Input byte bound exceeds frozen model context")
            receipt = output / "receipts" / identifier
            receipt.mkdir()
            save(receipt / "request.json", request, exclusive=True)
            attempt = {"slot": len(attempts), "identifier": identifier, **slot,
                       "started_at": utc(), "reservation_usd": str(RESERVE),
                       "status": "RESERVED_BEFORE_POST", "request_sha256": file_hash(receipt / "request.json")}
            attempts.append(attempt)
            write_ledger()  # Durable request/reservation precedes the only POST.
            try:
                response = api.request("/chat/completions", request)
            except TransportError as exc:
                save(receipt / "transport_error.json", {"http_status": exc.status,
                     "response": legacy.redact(exc.body, api)}, exclusive=True)
                raise
            save(receipt / "response.json", response, exclusive=True)
            attempt.update(response_sha256=file_hash(receipt / "response.json"),
                           response_id=response.get("id") if isinstance(response, dict) else None,
                           usage=response.get("usage") if isinstance(response, dict) else None)
            write_ledger()
            provenance(response, config)
            if response["id"] in seen_response_ids:
                raise Halt("Duplicate generation identity; fresh calls required")
            seen_response_ids.add(response["id"])
            cost = response_cost(response, api, receipt, sleep=sleep)
            known_new += cost
            attempt.update(cost_usd=str(cost), status="ACCOUNTED_RESPONSE")
            write_ledger()
            if cost > RESERVE or PRIOR_COST + known_new + integrity.UNKNOWN > BUDGET:
                raise Halt("Settled response cost exceeds reservation or cumulative budget")
            # Recheck on-disk snapshots and receipts immediately before interpreting content.
            if (file_hash(material_path) != lock["pair_material_sha256"][pair_id] or
                    file_hash(message_path) != lock["messages_sha256"][identifier] or
                    file_hash(receipt / "request.json") != attempt["request_sha256"] or
                    file_hash(receipt / "response.json") != attempt["response_sha256"]):
                raise Halt("Pinned generation evidence changed after POST")
            decision = classify_generation(response, slot["arm"], material,
                        response_sha256=object_sha256(response), material_sha256=object_sha256(material),
                        settled_cost=cost, expected_model=config["model"]["id"])
            library_sha = None
            if decision.status == VALID:
                save(output / "libraries" / (identifier + ".json"), decision.library, exclusive=True)
                library_sha = file_hash(output / "libraries" / (identifier + ".json"))
                libraries[identifier] = library_sha
            outcome = {"identifier": identifier, **slot, "status": decision.status,
                       "reason_codes": list(decision.reasons), "errors": list(decision.reasons[1:]),
                       "request_sha256": attempt["request_sha256"], "response_sha256": attempt["response_sha256"],
                       "response_id": response["id"], "cost_usd": str(cost), "library_sha256": library_sha,
                       "billing_sha256": file_hash(receipt / "billing.json")}
            save(output / "outcomes" / (identifier + ".json"), outcome, exclusive=True)
            outcomes[identifier] = file_hash(output / "outcomes" / (identifier + ".json"))
            attempt.update(status=decision.status, outcome_sha256=outcomes[identifier],
                           library_sha256=library_sha, completed_at=utc())
            write_ledger()
            print(json.dumps({"identifier": identifier, "status": decision.status,
                              "actual_post_attempts": len(attempts), "cumulative_known_cost_usd": str(PRIOR_COST + known_new)}), flush=True)
        if set(outcomes) != expected_slots(config) or len(attempts) != 21:
            raise Halt("Missing terminal slots; no completed generation run")
        integrity.verify()
        if (file_hash(output / "config.json") != file_hash(config_path) or
                file_hash(output / "protocol_lock.json") != file_hash(lock_path)):
            raise Halt("Frozen protocol changed before completion")
        result = {"status": "complete", "completed_at": utc(), "config_sha256": file_hash(config_path),
                  "protocol_lock_sha256": file_hash(lock_path), "messages_sha256": lock["messages_sha256"],
                  "pair_material_sha256": lock["pair_material_sha256"], "outcomes_sha256": outcomes,
                  "libraries_sha256": libraries, "actual_post_attempts": len(attempts),
                  "ledger_sha256": file_hash(output / "ledger.json"),
                  "unknown_reservation_usd": str(integrity.UNKNOWN),
                  "budget_occupancy_usd":str(known_new+integrity.UNKNOWN),
                  "amendment_lock_sha256":file_hash(integrity.AMENDMENT_LOCK),
                  "valid_library_count": len(libraries), "invalid_library_count": 20 - len(libraries),
                  "known_new_cost_usd": str(known_new), "cumulative_known_cost_usd": str(PRIOR_COST + known_new),
                  "prior_cost_usd": str(PRIOR_COST)}
        save(output / "completion.json", result, exclusive=True)
        return result
    except BaseException as exc:
        unresolved = sum((RESERVE for a in attempts if "cost_usd" not in a),Decimal(0))
        if len(attempts)>10 and attempts[-1]["status"] not in (VALID, INVALID):
            attempts[-1]["status"] = "HALT"
        write_ledger()
        error = legacy.redact(exc, api)
        save(output / "failure.json", {"status": "HALT", "stopped_at": utc(),
             "actual_post_attempts": len(attempts), "terminal_outcome_count": len(outcomes),
             "valid_library_count": len(libraries), "invalid_library_count": len(outcomes) - len(libraries),
             "prior_cost_usd": str(PRIOR_COST), "known_new_cost_usd": str(known_new),
             "cumulative_known_cost_usd": str(PRIOR_COST + known_new),
             "unresolved_reservation_usd": str(unresolved), "error_type": type(exc).__name__, "error": error,
             "retry_performed": False, "automatic_repair_performed": False,
             "incomplete_run_not_evaluated": True}, exclusive=True)
        raise Halt("Suzuki V1 generation halted: " + error) from None


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--execute-approved",action="store_true");args=parser.parse_args()
    integrity.verify()
    if args.execute_approved:
        execute(integrity.CONFIG,integrity.LOCK,integrity.FREEZE,integrity.NEW)
    else: print(json.dumps({"status":"verified_offline","new_post_limit":11,"total_post_limit":21,"unknown_reservation_usd":"0.045"}))
