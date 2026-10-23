"""Dedicated Suzuki integrity gates; legacy files remain immutable dependencies."""
from pathlib import Path
import os,json,math,tempfile,hashlib,random
ROOT=Path(__file__).resolve().parents[4]
PROTOCOL_REL="experiments/care_replay/protocols/suzuki_second_dataset_v1"
CONFIG_REL="experiments/care_replay/configs/suzuki_second_dataset_v1.json"
ARMS=("labeled","anonymous")
APPROVED_CONFIG_OBJECT_SHA="0e58fae14398ef6eaae43d95d307e2cbc78eb721403db9059381f97ffa0eea1b"
def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

def save(path, value, exclusive=False):
    """Fsync before exclusive creation/atomic replace; never partially update JSON."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        if exclusive:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        else:
            fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary is not None:
            temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def load(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def finite(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Nonfinite JSON number")
        return parsed
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_float=finite, parse_constant=reject)

def repo_path(relative, *, must_exist=True):
    """Resolve one repository-relative path without traversal or symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative path required")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        raise ValueError("Unsafe repository-relative path")
    root = ROOT.resolve()
    path = root / rel
    walk = root
    for part in rel.parts:
        walk /= part
        if walk.is_symlink():
            raise ValueError("Symlink is not a frozen input/output path")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Path escapes repository")
    if must_exist and not path.is_file():
        raise ValueError("Frozen input is missing or not an ordinary file: " + relative)
    return path

def relative_path(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Path traversal is not permitted")
    if not path.is_absolute():
        return str(path.as_posix())
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Path is outside repository") from exc

def object_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def expected_slots(config):
    from run_paired_anonymization_generation import expected_slots as old_slots
    return old_slots(config)

def validate_config(config):
    if object_hash(config)!=APPROVED_CONFIG_OBJECT_SHA: raise ValueError("Approved Suzuki configuration differs")
    expected_slots(config)
    calibration=set(range(config["calibration_seed_start"],config["calibration_seed_start"]+50))
    heldout=set(range(config["heldout_seed_start"],config["heldout_seed_start"]+100))
    aux=set(config["paired_design"]["alias_seed_by_pair"].values())|{config["source"]["seed"],config["random_controls"]["seed"],config["paired_design"]["order_seed"],config["bootstrap_seed"]}
    if len(calibration|heldout|aux)!=164: raise ValueError("Seed namespaces overlap")
    return calibration,heldout

def source_inventory():
    files=list((ROOT/"experiments/care_replay/scripts").glob("*.py"))+list((ROOT/PROTOCOL_REL).rglob("*.py"))
    return sorted(relative_path(p) for p in files)

def input_inventory(config): return sorted(config["input_files"])

def verify_freeze(config_path,lock_path,freeze_dir):
    if relative_path(config_path)!=CONFIG_REL: raise ValueError("Wrong production config")
    config_path=repo_path(CONFIG_REL);lock_path=repo_path(relative_path(lock_path));freeze_dir=repo_path(relative_path(freeze_dir),must_exist=False)
    config,lock=load(config_path),load(lock_path);validate_config(config)
    if lock.get("protocol_id")!=config["protocol_id"] or lock.get("status")!="frozen_before_generation" or lock.get("model_api_calls_at_freeze")!=0: raise ValueError("Not a pregeneration freeze")
    if lock.get("freeze_directory")!=relative_path(freeze_dir): raise ValueError("Wrong freeze directory")
    if lock.get("config_sha256")!=file_hash(config_path) or file_hash(freeze_dir/"config.json")!=file_hash(config_path): raise ValueError("Config changed")
    if file_hash(freeze_dir/"protocol_lock.json")!=file_hash(lock_path): raise ValueError("Lock snapshot changed")
    for group,names,folder in (("source_sha256",source_inventory(),"source_snapshot"),("data_sha256",config["data_files"],"input_snapshot"),("input_sha256",input_inventory(config),"input_snapshot")):
        if set(lock[group])!=set(names): raise ValueError("Inventory changed: "+group)
        for rel,sha in lock[group].items():
            if file_hash(repo_path(rel))!=sha or file_hash(freeze_dir/folder/rel)!=sha: raise ValueError("Input/source changed: "+rel)
    for group,folder,names in (("messages_sha256","messages",expected_slots(config)),("pair_material_sha256","private_pairs",{f"pair_{i:02d}" for i in range(10)})):
        if set(lock[group])!=names: raise ValueError("Incomplete generation material")
        if {p.name for p in (freeze_dir/folder).iterdir()}!={n+".json" for n in names}: raise ValueError("Unexpected material")
        for name,sha in lock[group].items():
            if file_hash(freeze_dir/folder/(name+".json"))!=sha: raise ValueError("Material changed")
    if file_hash(freeze_dir/"offline_validation.log")!=lock["offline_validation_log_sha256"]: raise ValueError("Validation log changed")
    if file_hash(freeze_dir/"preflight_audit.json")!=lock["preflight_audit_sha256"]: raise ValueError("Audit changed")
    return config,lock
