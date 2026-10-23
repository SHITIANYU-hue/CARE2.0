from bootstrap import *
import continuation_integrity as integrity
"""Offline re-verification of all 20 paid outcomes, inherited receipt checks."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import common
ARMS=("labeled","anonymous")
VALID="VALID"
INVALID="INVALID_ANCHOR_FALLBACK"
def money(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Malformed cost") from None
    if not result.is_finite() or result < 0:
        raise ValueError("Cost must be finite and nonnegative")
    return result

def names(config):
    return [f"pair_{pair:02d}_{arm}" for pair in range(config["pair_count"]) for arm in ARMS]

def checked(path, expected_hash=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Required ordinary file missing: {path.name}")
    digest = common.file_hash(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"File hash mismatch: {path.name}")
    return digest

def replay_classification(response, arm, material, cost):
    """Use the production classifier only as a pure function, never its API runner."""
    import generation
    return generation.classify_generation(
        response, arm, material, response_sha256=generation.object_sha256(response),
        material_sha256=generation.object_sha256(material), settled_cost=cost)

def verify_billing_evidence(response, billing, receipt, cost):
    """Reproduce the generator's billing decision using retained receipts only."""
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("is_byok") is True:
        raise ValueError("Original response usage is missing/malformed or reports BYOK")
    inline = money(usage["cost"]) if usage.get("cost") is not None else None
    if inline is not None and usage.get("is_byok") is False:
        if billing["source"] != "response_usage" or cost != inline:
            raise ValueError("Billing differs from the settled original response usage")
        return
    if billing["source"] != "generation_lookup":
        raise ValueError("Missing original non-BYOK evidence requires a generation lookup")
    for attempt in range(3):
        path = receipt / f"cost_lookup_{attempt}.json"
        error_path = receipt / f"cost_lookup_{attempt}_error.json"
        if path.exists() == error_path.exists():
            raise ValueError("Lookup attempts must have exactly one retained response or error")
        if error_path.exists():
            error = common.load(error_path)
            if error.get("http_status") != 404:
                raise ValueError("Non-retryable billing lookup error cannot have a completed outcome")
            continue
        info = common.load(path)
        data = info.get("data") if isinstance(info, dict) else None
        if (not isinstance(data, dict) or data.get("id") != response["id"]
                or data.get("model") != response.get("model")
                or data.get("provider_name") != response.get("provider")
                or data.get("is_byok") is True):
            raise ValueError("Original billing lookup provenance/BYOK mismatch")
        if data.get("total_cost") is not None and data.get("is_byok") is False:
            lookup = money(data["total_cost"])
            expected = max(lookup, inline) if inline is not None else lookup
            if cost != expected:
                raise ValueError("Settled billing differs from original lookup and inline cost")
            return
    raise ValueError("Retained billing lookups never settle the response cost and non-BYOK status")

def verify_generation(config_path, lock_path, generation_output, config, lock):
    """Require twenty settled, traceable terminal outcomes before any calibration."""
    integrity.verify()
    base = Path(generation_output)
    if inventory_original := integrity.inventory(base/"original_evidence"):
        if inventory_original != integrity.inventory(integrity.OLD): raise ValueError("Original evidence copy differs")
    else: raise ValueError("Original evidence missing")
    checked(base/"amendment_lock.json",checked(integrity.AMENDMENT_LOCK))
    if (base / "failure.json").exists():
        raise ValueError("A halted generation directory cannot be evaluated")
    completion = common.load(base / "completion.json")
    if completion.get("status") != "complete" or completion.get("actual_post_attempts") != 21:
        raise ValueError("Twenty terminal slots and twenty-one actual attempts are required")
    slots = names(config)
    expected = set(slots)
    if expected != common.expected_slots(config):
        raise ValueError("Unexpected slot inventory")
    hashes = {"completion.json": checked(base / "completion.json")}
    if (completion.get("config_sha256") != checked(config_path)
            or completion.get("protocol_lock_sha256") != checked(lock_path)):
        raise ValueError("Completion config/protocol mismatch")
    hashes["config.json"] = checked(base / "config.json", checked(config_path))
    hashes["protocol_lock.json"] = checked(base / "protocol_lock.json", checked(lock_path))
    if set(completion.get("outcomes_sha256", {})) != expected:
        raise ValueError("All twenty outcome hashes are required")
    for directory in ("outcomes", "libraries", "receipts"):
        if not (base / directory).is_dir() or (base / directory).is_symlink():
            raise ValueError(f"Required directory missing: {directory}")
    if {p.name for p in (base / "outcomes").iterdir()} != {slot + ".json" for slot in slots}:
        raise ValueError("Outcome files must match all twenty slots exactly")
    if {p.name for p in (base / "receipts").iterdir()} != expected:
        raise ValueError("Receipt directories must match all twenty slots exactly")
    for section, folder, identifiers in (
        ("messages_sha256", "messages", expected),
        ("pair_material_sha256", "pairs", {f"pair_{i:02d}" for i in range(config["pair_count"])}),
    ):
        if set(lock.get(section, {})) != identifiers or completion.get(section) != lock[section]:
            raise ValueError(f"Completion differs from frozen {section}")
        for identifier, digest in lock[section].items():
            relative = f"input_snapshot/{folder}/{identifier}.json"
            hashes[relative] = checked(base / relative, digest)
    for relative, digest in lock["source_sha256"].items():
        destination = "source_snapshot/" + relative
        hashes[destination] = checked(base / destination, digest)
    for group in ("data_sha256", "input_sha256"):
        for relative, digest in lock[group].items():
            destination = "input_snapshot/" + relative
            hashes[destination] = checked(base / destination, digest)

    outcomes, payloads, response_ids = {}, {}, set()
    cost_total = Decimal("0")
    for pair in range(config["pair_count"]):
        material = common.load(base / f"input_snapshot/pairs/pair_{pair:02d}.json")
        for arm in ARMS:
            identifier = f"pair_{pair:02d}_{arm}"
            relative = f"outcomes/{identifier}.json"
            hashes[relative] = checked(base / relative, completion["outcomes_sha256"][identifier])
            outcome = common.load(base / relative)
            if (outcome.get("identifier") != identifier or outcome.get("pair_index") != pair
                    or outcome.get("arm") != arm or outcome.get("status") not in {VALID, INVALID}):
                raise ValueError("Wrong slot identity or nonterminal/HALT outcome")
            receipt = base / "receipts" / identifier
            if receipt.is_symlink() or not receipt.is_dir():
                raise ValueError("Receipt must be an ordinary directory")
            for filename, digest in (("request.json", outcome.get("request_sha256")),
                                     ("response.json", outcome.get("response_sha256"))):
                if not isinstance(digest, str):
                    raise ValueError("Request/response provenance hash missing")
                key = f"receipts/{identifier}/{filename}"
                hashes[key] = checked(base / key, digest)
            request = common.load(receipt / "request.json")
            response = common.load(receipt / "response.json")
            billing = common.load(receipt / "billing.json")
            if not isinstance(outcome.get("billing_sha256"), str):
                raise ValueError("Billing provenance hash missing")
            hashes[f"receipts/{identifier}/billing.json"] = checked(
                receipt / "billing.json", outcome["billing_sha256"])
            expected_request = {
                "model": config["model"]["id"],
                "messages": common.load(base / f"input_snapshot/messages/{identifier}.json"),
                "temperature": config["model"]["temperature"], "max_tokens": config["model"]["max_tokens"],
                "response_format": {"type": "json_object"}, "stream": False,
                "provider": config["api"]["provider"],
            }
            if request != expected_request:
                raise ValueError("Request differs from its frozen message/parameters")
            response_id = response.get("id") if isinstance(response, dict) else None
            if (not isinstance(response_id, str) or not response_id or response_id in response_ids
                    or outcome.get("response_id") != response_id or billing.get("response_id") != response_id):
                raise ValueError("Missing, duplicate or inconsistent response identity")
            response_ids.add(response_id)
            if (billing.get("is_byok") is not False
                    or billing.get("source") not in {"response_usage", "generation_lookup"}):
                raise ValueError("Every outcome requires explicitly settled non-BYOK billing")
            cost = money(billing.get("cost_usd"))
            if cost != money(outcome.get("cost_usd")) or cost > money(config["api"]["reservation_per_attempt_usd"]):
                raise ValueError("Outcome billing mismatch or reservation exceeded")
            verify_billing_evidence(response, billing, receipt, cost)
            decision = replay_classification(response, arm, material, cost)
            if (decision.status != outcome["status"]
                    or list(decision.reasons) != outcome.get("reason_codes")
                    or list(decision.reasons[1:]) != outcome.get("errors")):
                raise ValueError("Recorded terminal state/reasons differ from deterministic classification")
            if money(decision.cost_usd) != cost:
                raise ValueError("Classifier billing mismatch")
            if outcome["status"] == VALID:
                if not isinstance(outcome.get("library_sha256"), str):
                    raise ValueError("Valid library hash missing")
                key = f"libraries/{identifier}.json"
                hashes[key] = checked(base / key, outcome["library_sha256"])
                payloads[identifier] = common.load(base / key)
                if payloads[identifier] != decision.library:
                    raise ValueError("Valid library differs from the decoded complete response")
            elif (outcome.get("library_sha256") is not None or decision.library is not None
                  or not outcome.get("reason_codes")):
                raise ValueError("Invalid slots must retain reasons and no partial library")
            outcomes[identifier] = outcome
            cost_total += cost
            # Preserve extra provider reconciliation receipts as immutable audit inputs.
            for path in receipt.rglob("*"):
                if path.is_file():
                    hashes[path.relative_to(base).as_posix()] = checked(path)
                elif path.is_symlink():
                    raise ValueError("Receipt links are forbidden")
    valid_ids = set(payloads)
    if set(completion.get("libraries_sha256", {})) != valid_ids:
        raise ValueError("Completion libraries must equal the VALID subset")
    if {p.name for p in (base / "libraries").iterdir()} != {name + ".json" for name in valid_ids}:
        raise ValueError("Invalid slots cannot retain partial libraries; valid files cannot be missing")
    if any(completion["libraries_sha256"][name] != outcomes[name]["library_sha256"] for name in valid_ids):
        raise ValueError("Outcome/completion library hashes differ")
    if (completion.get("valid_library_count") != len(valid_ids)
            or completion.get("invalid_library_count") != len(slots) - len(valid_ids)):
        raise ValueError("Completion validity counts differ from terminal outcomes")
    ledger = common.load(base / "ledger.json")
    if not isinstance(completion.get("ledger_sha256"), str):
        raise ValueError("Completed ledger hash missing")
    hashes["ledger.json"] = checked(base / "ledger.json", completion["ledger_sha256"])
    attempts = ledger.get("attempts")
    if (not isinstance(attempts, list) or len(attempts) != 21
            or ledger.get("actual_post_attempts") != 21
            or {a.get("identifier") for a in attempts} != expected):
        raise ValueError("Ledger must contain twenty-one attempts covering twenty logical slots")
    original=common.load(integrity.OLD/"ledger.json")["attempts"]
    if attempts[:10]!=original: raise ValueError("Original ten attempts changed")
    terminal_attempts=attempts[:9]+attempts[10:]
    for slot_number, (attempt, scheduled) in enumerate(zip(terminal_attempts, config["generation_order"], strict=True)):
        identifier = f"pair_{scheduled['pair_index']:02d}_{scheduled['arm']}"
        outcome = outcomes[identifier]
        if money(attempt.get("reservation_usd"))!=integrity.UNKNOWN: raise ValueError("Attempt reservation differs")
        if (attempt.get("slot") != (slot_number if slot_number<9 else slot_number+1) or attempt.get("identifier") != identifier
                or attempt.get("pair_index") != scheduled["pair_index"]
                or attempt.get("arm") != scheduled["arm"]):
            raise ValueError("Ledger slot order differs from frozen generation order")
        for key in ("request_sha256", "response_sha256", "response_id", "library_sha256"):
            if attempt.get(key) != outcome[key]:
                raise ValueError(f"Ledger/outcome provenance mismatch: {key}")
        if attempt.get("outcome_sha256") != completion["outcomes_sha256"][identifier]:
            raise ValueError("Ledger outcome hash differs from completion")
        if attempt.get("status") != outcome["status"] or money(attempt.get("cost_usd")) != money(outcome["cost_usd"]):
            raise ValueError("Ledger/outcome mismatch")
    prior = money(config["resource_authorization"]["prior_known_cost_usd"])
    cumulative = prior + cost_total
    if cumulative + integrity.UNKNOWN > money(config["api"]["max_cost_usd"]):
        raise ValueError("Cumulative experimental spend exceeds the frozen cap")
    for record in (ledger, completion):
        for key, expected_cost in (("prior_cost_usd", prior), ("known_new_cost_usd", cost_total),
                                    ("cumulative_known_cost_usd", cumulative)):
            if money(record.get(key)) != expected_cost:
                raise ValueError(f"Cost ledger mismatch: {key}")
    for record in (ledger,completion):
        if money(record.get("unknown_reservation_usd"))!=integrity.UNKNOWN: raise ValueError("Unknown reservation lost")
    if money(completion.get("budget_occupancy_usd"))!=cumulative+integrity.UNKNOWN: raise ValueError("Budget occupancy mismatch")
    if completion.get("amendment_lock_sha256")!=checked(integrity.AMENDMENT_LOCK): raise ValueError("Wrong amendment")
    hashes["amendment_lock.json"]=checked(base/"amendment_lock.json")
    for rel,sha in integrity.inventory(base/"original_evidence").items(): hashes["original_evidence/"+rel]=sha
    return outcomes, payloads, hashes
