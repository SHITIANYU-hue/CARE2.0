#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.client
import json
import math
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Literal
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
RAW_DATA = ROOT / "data" / "raw"
REACTION_DESCRIPTOR_PATH = ROOT / "data" / "descriptors" / "reaction_component_descriptors.csv"
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELS_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
PUBLIC_DATA_URLS = {
    "dreher_doyle_buchwald_hartwig.xlsx": (
        "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
        "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
    ),
    "perera_suzuki_miyaura.xlsx": (
        "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
        "data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx"
    ),
    "moleculenet_esol_delaney.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "moleculenet_freesolv_sampl.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
    "moleculenet_lipophilicity.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "chemlex_acidamine_wetlab_v3.xlsx": "https://zenodo.org/records/17596563/files/Chemlex_Acidamine_Wetlab_Data.xlsx?download=1",
    "matbench_expt_gap.json.gz": "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz",
    "matbench_dielectric.json.gz": "https://ml.materialsproject.org/projects/matbench_dielectric.json.gz",
    "matbench_phonons.json.gz": "https://ml.materialsproject.org/projects/matbench_phonons.json.gz",
    "matbench_log_kvrh.json.gz": "https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz",
}

REACTION_DESCRIPTOR_METADATA_FIELDS = {
    "rdkit_parse_ok": "rdkit_parse_ok",
    "descriptor_mw_bin": "mw_bin",
    "descriptor_logp_bin": "logp_bin",
    "descriptor_tpsa_bin": "tpsa_bin",
    "descriptor_hbd_bin": "hbd_bin",
    "descriptor_hba_bin": "hba_bin",
    "descriptor_rotatable_bin": "rotatable_bin",
    "descriptor_aromatic_ring_bin": "aromatic_ring_bin",
    "descriptor_fraction_csp3_bin": "fraction_csp3_bin",
    "descriptor_complexity_bin": "complexity_bin",
    "formal_charge_class": "formal_charge_class",
    "ring_system_class": "ring_system_class",
    "acid_functional_class": "acid_functional_class",
    "amine_functional_class": "amine_functional_class",
    "amide_count_bin": "amide_count_bin",
    "has_phosphorus": "has_phosphorus",
    "has_phosphine": "has_phosphine",
    "has_boron": "has_boron",
    "has_aryl_halide": "has_aryl_halide",
    "has_heteroaromatic": "has_heteroaromatic",
    "halide_type": "halide_type",
    "boron_species": "boron_species",
    "ligand_family": "ligand_family",
    "reagent_base_family": "reagent_base_family",
    "solvent_family": "solvent_family",
    "solvent_is_protic": "solvent_is_protic",
    "functional_class": "functional_class",
}
REACTION_DESCRIPTOR_CACHE: dict[tuple[str, str, str], dict[str, str]] | None = None

SkillFamily = Literal["ranker", "constraint", "exploration", "data_analysis", "fallback"]
Mode = Literal[
    "no_care_random",
    "incumbent",
    "no_gate",
    "gate_v1",
    "gate_v2",
    "llm_no_gate",
    "llm_gate_v1",
    "llm_explore_no_gate",
    "llm_explore_gate_v1",
]
DEFAULT_MODES: tuple[Mode, ...] = ("no_care_random", "incumbent", "no_gate", "gate_v1", "gate_v2")
ALL_MODES: tuple[Mode, ...] = (
    *DEFAULT_MODES,
    "llm_no_gate",
    "llm_gate_v1",
    "llm_explore_no_gate",
    "llm_explore_gate_v1",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group: str
    x1: float
    x2: float
    x3: float
    objective_value: float
    metadata: dict[str, Any] = field(default_factory=dict)
    numeric_features: tuple[float, ...] = ()


@dataclass(frozen=True)
class DatasetAdapter:
    dataset_id: str
    title: str
    objective: str
    decision_columns: tuple[str, ...]
    hidden_target: str
    group_column: str
    preferred_groups: tuple[str, ...]
    failure_note: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class TaskSpec:
    dataset_id: str
    objective: str
    decision_columns: tuple[str, ...]
    hidden_target: str
    initial_observations: int
    reveal_budget: int
    oracle_value: float


@dataclass(frozen=True)
class SkillCard:
    skill_id: str
    version: str
    family: SkillFamily
    scope: str
    trigger_rules: tuple[dict[str, Any], ...]
    bounded_parameters: dict[str, Any]
    certificate_schema: dict[str, Any]
    required_checks: tuple[str, ...]
    prohibited_behaviors: tuple[str, ...]
    provenance: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    api_mode: Literal["chat", "completion"] = "chat"
    structured_mode: Literal["tool", "json"] = "tool"


@dataclass
class HypothesisEntry:
    hypothesis_id: str
    status: Literal["active", "inactive", "falsified", "pending"]
    scope: Literal["group_preference", "similarity_region", "mechanism"]
    trigger: dict[str, Any]
    target_spec: dict[str, Any]
    claim: str
    confidence: float
    support_count: int
    alpha: float
    beta: float
    evidence_summary: str
    known_failure_modes: list[str]
    created_round: int
    last_updated_round: int

    def update(self, supports: bool, round_index: int, evidence_summary: str) -> None:
        if supports:
            self.alpha += 1.0
            self.support_count += 1
        else:
            self.beta += 1.0
        self.confidence = self.alpha / (self.alpha + self.beta)
        self.last_updated_round = round_index
        self.evidence_summary = evidence_summary


@dataclass(frozen=True)
class GateCertificate:
    gate_version: str
    incumbent_candidate: str
    challenger_candidate: str
    selected_candidate: str
    authorized: bool
    gate_margin: float
    acquisition_loss: float
    row_order_stable: bool
    applied_skill_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class AuditEntry:
    dataset_id: str
    seed: int
    round_index: int
    public_observed_count: int
    incumbent_candidate: str
    challenger_candidate: str
    selected_candidate: str
    selected_by: str
    gate: GateCertificate
    revealed_value: float
    best_so_far: float
    hypothesis_snapshot: dict[str, Any]


def stable_noise(*parts: object, scale: float = 2.0) -> float:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 2.0 * scale


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 4)


def stable_fraction(*parts: object) -> float:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def ensure_public_data_file(filename: str) -> Path:
    RAW_DATA.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA / filename
    if path.exists():
        return path
    url = PUBLIC_DATA_URLS[filename]
    print(f"downloading {filename} from {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_component_name(value: Any) -> str:
    return str(value).strip()


def load_reaction_component_descriptors() -> dict[tuple[str, str, str], dict[str, str]]:
    global REACTION_DESCRIPTOR_CACHE
    if REACTION_DESCRIPTOR_CACHE is not None:
        return REACTION_DESCRIPTOR_CACHE
    if not REACTION_DESCRIPTOR_PATH.exists():
        REACTION_DESCRIPTOR_CACHE = {}
        return REACTION_DESCRIPTOR_CACHE
    rows = read_csv_dicts(REACTION_DESCRIPTOR_PATH)
    REACTION_DESCRIPTOR_CACHE = {
        (row["dataset_id"], row["role"], clean_component_name(row["raw_name"])): row
        for row in rows
    }
    return REACTION_DESCRIPTOR_CACHE


def reaction_component_metadata(dataset_id: str, role: str, raw_name: Any, prefix: str) -> dict[str, str]:
    descriptors = load_reaction_component_descriptors()
    row = descriptors.get((dataset_id, role, clean_component_name(raw_name)))
    if row is None:
        return {}
    out: dict[str, str] = {
        f"{prefix}_raw_name": clean_component_name(raw_name),
        f"{prefix}_canonical_smiles": row.get("canonical_smiles", ""),
    }
    for source_field, target_suffix in REACTION_DESCRIPTOR_METADATA_FIELDS.items():
        out[f"{prefix}_{target_suffix}"] = row.get(source_field, "")
    return out


def normalized_descriptor(value: str, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (number - lower) / (upper - lower)))


def reaction_component_numeric_features(dataset_id: str, role: str, raw_name: Any) -> tuple[float, ...]:
    row = load_reaction_component_descriptors().get(
        (dataset_id, role, clean_component_name(raw_name))
    )
    if row is None or row.get("rdkit_parse_ok") != "yes":
        return (0.0,) * 8
    return (
        normalized_descriptor(row.get("mol_weight", ""), 0.0, 1000.0),
        normalized_descriptor(row.get("logp", ""), -5.0, 10.0),
        normalized_descriptor(row.get("tpsa", ""), 0.0, 250.0),
        normalized_descriptor(row.get("hbd", ""), 0.0, 10.0),
        normalized_descriptor(row.get("hba", ""), 0.0, 20.0),
        normalized_descriptor(row.get("rotatable_bonds", ""), 0.0, 25.0),
        normalized_descriptor(row.get("aromatic_ring_count", ""), 0.0, 8.0),
        normalized_descriptor(row.get("fraction_csp3", ""), 0.0, 1.0),
    )


def excel_col_index(cell_ref: str) -> int:
    col = 0
    for ch in cell_ref:
        if not ch.isalpha():
            break
        col = col * 26 + ord(ch.upper()) - ord("A") + 1
    return col - 1


def parse_xlsx_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text_node = cell.find(f"{XLSX_NS}is/{XLSX_NS}t")
        return "" if text_node is None or text_node.text is None else text_node.text
    value_node = cell.find(f"{XLSX_NS}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    try:
        value = float(raw)
    except ValueError:
        return raw
    return int(value) if value.is_integer() else value


def read_xlsx_rows(path: Path, sheet_name: str) -> list[list[Any]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{XLSX_NS}si"):
                texts = [node.text or "" for node in item.iter(f"{XLSX_NS}t")]
                shared_strings.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_by_id = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall(f"{RELS_NS}Relationship")}
        sheet_target = None
        for sheet in workbook.findall(f"{XLSX_NS}sheets/{XLSX_NS}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                sheet_target = rel_by_id[rel_id]
                break
        if sheet_target is None:
            raise ValueError(f"Sheet not found: {sheet_name} in {path}")

        sheet_path = "xl/" + sheet_target.lstrip("/")
        root = ET.fromstring(zf.read(sheet_path))
        rows: list[list[Any]] = []
        for row_node in root.findall(f".//{XLSX_NS}row"):
            values: list[Any] = []
            for cell in row_node.findall(f"{XLSX_NS}c"):
                idx = excel_col_index(cell.attrib["r"])
                while len(values) <= idx:
                    values.append(None)
                values[idx] = parse_xlsx_value(cell, shared_strings)
            rows.append(values)
        return rows


def rows_to_dicts(rows: list[list[Any]]) -> list[dict[str, Any]]:
    header = [str(value).strip() if value is not None else "" for value in rows[0]]
    out = []
    for row in rows[1:]:
        item = {name: row[idx] if idx < len(row) else None for idx, name in enumerate(header) if name}
        out.append(item)
    return out


def read_matbench_json_gz(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    columns = raw["columns"]
    return [dict(zip(columns, row)) for row in raw["data"]]


def label_map(values: list[Any], prefix: str) -> dict[str, str]:
    ordered = sorted({str(value) for value in values if value is not None})
    return {value: f"{prefix}{idx:02d}" for idx, value in enumerate(ordered)}


def numeric_bin(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if len(labels) != len(edges) + 1:
        raise ValueError("numeric_bin expects one more label than edge")
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def smiles_public_descriptors(smiles: str) -> tuple[dict[str, str], tuple[float, float, float]]:
    length = float(len(smiles))
    hetero_count = float(sum(smiles.count(token) for token in ("N", "O", "S", "P", "n", "o", "s")))
    halogen_count = float(smiles.count("Cl") + smiles.count("Br") + smiles.count("F") + smiles.count("I"))
    aromatic_count = float(sum(1 for ch in smiles if ch in {"c", "n", "o", "s"}))
    ring_token_count = float(sum(1 for ch in smiles if ch.isdigit()))
    branch_count = float(smiles.count("(") + smiles.count(")"))
    double_bond_count = float(smiles.count("="))
    descriptor_bins = {
        "smiles_length_bin": numeric_bin(length, (20.0, 45.0, 80.0), ("smiles_short", "smiles_mid", "smiles_long", "smiles_very_long")),
        "hetero_atom_bin": numeric_bin(hetero_count, (0.0, 2.0, 5.0), ("hetero_none", "hetero_low", "hetero_mid", "hetero_high")),
        "halogen_bin": numeric_bin(halogen_count, (0.0, 1.0, 3.0), ("halogen_none", "halogen_low", "halogen_mid", "halogen_high")),
        "aromatic_bin": numeric_bin(aromatic_count, (0.0, 6.0, 12.0), ("aromatic_none", "aromatic_low", "aromatic_mid", "aromatic_high")),
        "ring_token_bin": numeric_bin(ring_token_count, (0.0, 2.0, 6.0), ("ring_token_none", "ring_token_low", "ring_token_mid", "ring_token_high")),
        "branch_bin": numeric_bin(branch_count, (0.0, 4.0, 10.0), ("branch_none", "branch_low", "branch_mid", "branch_high")),
        "double_bond_bin": numeric_bin(double_bond_count, (0.0, 1.0, 4.0), ("double_bond_none", "double_bond_low", "double_bond_mid", "double_bond_high")),
    }
    x1 = max(0.0, min(1.0, length / 160.0))
    x2 = max(0.0, min(1.0, (hetero_count + halogen_count) / 16.0))
    x3 = max(0.0, min(1.0, (aromatic_count + ring_token_count + branch_count) / 36.0))
    return descriptor_bins, (x1, x2, x3)


def prefixed_smiles_public_descriptors(prefix: str, smiles: str) -> dict[str, str]:
    descriptors, _numeric = smiles_public_descriptors(smiles)
    return {f"{prefix}_{field}": value for field, value in descriptors.items()}


def prefixed_smiles_motif_descriptors(prefix: str, smiles: str) -> dict[str, str]:
    nitrogen_count = float(smiles.count("N") + smiles.count("n"))
    oxygen_count = float(smiles.count("O") + smiles.count("o"))
    carbonyl_count = float(
        smiles.count("C(=O)")
        + smiles.count("O=C(")
        + smiles.count("C(=S)")
        + smiles.count("S=C(")
    )
    return {
        f"{prefix}_nitrogen_bin": numeric_bin(
            nitrogen_count,
            (0.0, 1.0, 3.0),
            ("nitrogen_none", "nitrogen_one", "nitrogen_few", "nitrogen_many"),
        ),
        f"{prefix}_oxygen_bin": numeric_bin(
            oxygen_count,
            (0.0, 2.0, 5.0),
            ("oxygen_none", "oxygen_low", "oxygen_mid", "oxygen_high"),
        ),
        f"{prefix}_carbonyl_bin": numeric_bin(
            carbonyl_count,
            (0.0, 1.0, 2.0),
            ("carbonyl_none", "carbonyl_one", "carbonyl_two", "carbonyl_many"),
        ),
        f"{prefix}_amide_flag": (
            "has_amide"
            if "C(=O)N" in smiles or "O=C(N" in smiles or "NC(=O)" in smiles
            else "no_amide"
        ),
        f"{prefix}_nitrile_flag": (
            "has_nitrile" if "C#N" in smiles or "N#C" in smiles else "no_nitrile"
        ),
        f"{prefix}_sulfur_flag": (
            "has_sulfur" if "S" in smiles or "s" in smiles else "no_sulfur"
        ),
        f"{prefix}_phosphorus_flag": (
            "has_phosphorus" if "P" in smiles else "no_phosphorus"
        ),
        f"{prefix}_formal_charge_flag": (
            "has_formal_charge" if "+" in smiles or "-" in smiles else "no_formal_charge"
        ),
        f"{prefix}_aromatic_hetero_flag": (
            "has_aromatic_hetero"
            if any(token in smiles for token in ("n", "o", "s"))
            else "no_aromatic_hetero"
        ),
        f"{prefix}_multi_component_flag": (
            "multi_component" if "." in smiles else "single_component"
        ),
    }


def chemlex_reagent_family(smiles: str) -> str:
    if "N=C=N" in smiles:
        return "carbodiimide_coupling"
    if "[P+]" in smiles:
        return "phosphonium_coupling"
    if "On1nnc2cccnc21" in smiles:
        return "aza_benzotriazole_uronium"
    if "n1n[n+]([O-])c2ncccc21" in smiles:
        return "n_oxide_uronium"
    if "CN(C)C(Cl)=[N+]" in smiles:
        return "chloroformamidinium"
    return "other_uronium"


ELEMENT_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
TRANSITION_METALS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}
LANTHANIDES = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
POST_TRANSITION = {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po", "Nh", "Fl", "Mc", "Lv"}
METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te"}
HALOGENS = {"F", "Cl", "Br", "I", "At", "Ts"}
CHALCOGENS = {"O", "S", "Se", "Te", "Po"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}


def read_formula_number(formula: str, start: int) -> tuple[float, int]:
    end = start
    while end < len(formula) and (formula[end].isdigit() or formula[end] == "."):
        end += 1
    if end == start:
        return 1.0, start
    return float(formula[start:end]), end


def parse_composition(formula: str) -> dict[str, float]:
    stack: list[dict[str, float]] = [{}]
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch == "(":
            stack.append({})
            i += 1
        elif ch == ")":
            group = stack.pop()
            multiplier, i = read_formula_number(formula, i + 1)
            for element, count in group.items():
                stack[-1][element] = stack[-1].get(element, 0.0) + count * multiplier
        elif ch.isupper():
            j = i + 1
            if j < len(formula) and formula[j].islower():
                j += 1
            element = formula[i:j]
            count, i = read_formula_number(formula, j)
            stack[-1][element] = stack[-1].get(element, 0.0) + count
        else:
            i += 1
    if len(stack) != 1:
        raise ValueError(f"Unbalanced formula: {formula}")
    return stack[0]


def element_family(element: str) -> str:
    if element in ALKALI:
        return "alkali"
    if element in ALKALINE_EARTH:
        return "alkaline_earth"
    if element in TRANSITION_METALS:
        return "transition_metal"
    if element in LANTHANIDES:
        return "lanthanide"
    if element in POST_TRANSITION:
        return "post_transition"
    if element in METALLOIDS:
        return "metalloid"
    if element in HALOGENS:
        return "halogen"
    if element in CHALCOGENS:
        return "chalcogen"
    if element in PNICTOGENS:
        return "pnictogen"
    if element in {"C", "H"}:
        return "light_nonmetal"
    return "other"


def anion_family(elements: set[str]) -> str:
    if "O" in elements:
        return "oxide"
    if elements & {"S", "Se", "Te"}:
        return "chalcogenide"
    if elements & {"F", "Cl", "Br", "I"}:
        return "halide"
    if elements & {"N", "P", "As", "Sb", "Bi"}:
        return "pnictide"
    if "C" in elements:
        return "carbide_or_carbon"
    return "other"


def synthetic_suzuki_adapter() -> DatasetAdapter:
    ligands = [f"L{i}" for i in range(7)]
    residence_times = [30.0, 60.0, 90.0, 120.0]
    temperatures = [50.0, 70.0, 90.0, 110.0]
    loadings = [0.5, 1.0, 2.0, 4.0]
    ligand_base = {"L0": 42.0, "L1": 48.0, "L2": 76.0, "L3": 62.0, "L4": 58.0, "L5": 72.0, "L6": 55.0}
    ligand_temp_opt = {"L0": 70.0, "L1": 70.0, "L2": 90.0, "L3": 90.0, "L4": 70.0, "L5": 110.0, "L6": 90.0}
    pool: list[Candidate] = []
    for ligand in ligands:
        for time in residence_times:
            for temp in temperatures:
                for loading in loadings:
                    temp_effect = -0.018 * (temp - ligand_temp_opt[ligand]) ** 2
                    time_effect = -0.0018 * (time - 90.0) ** 2 + 5.0
                    loading_effect = 4.0 * math.log1p(loading) - 1.1 * loading
                    interaction = 4.0 if ligand in {"L2", "L5"} and temp >= 90.0 and time >= 90.0 else 0.0
                    penalty = -10.0 if temp <= 50.0 and loading <= 0.5 else 0.0
                    y = clamp_score(
                        ligand_base[ligand]
                        + temp_effect
                        + time_effect
                        + loading_effect
                        + interaction
                        + penalty
                        + stable_noise(ligand, time, temp, loading, scale=2.3)
                    )
                    cid = f"{ligand}_T{int(temp)}_R{int(time)}_C{str(loading).replace('.', 'p')}"
                    pool.append(
                        Candidate(
                            candidate_id=cid,
                            group=ligand,
                            x1=temp / 110.0,
                            x2=time / 120.0,
                            x3=math.log1p(loading) / math.log1p(4.0),
                            objective_value=y,
                            metadata={
                                "ligand_identity": ligand,
                                "temperature": temp,
                                "residence_time": time,
                                "catalyst_loading": loading,
                                "yield_value": y,
                            },
                        )
                    )
    return DatasetAdapter(
        dataset_id="synthetic_suzuki_i",
        title="Synthetic Suzuki finite-pool replay",
        objective="maximize_yield",
        decision_columns=("ligand_identity", "residence_time", "temperature", "catalyst_loading"),
        hidden_target="yield_value",
        group_column="ligand_identity",
        preferred_groups=("L2", "L5"),
        failure_note="Low temperature and low catalyst loading can erase the ligand advantage.",
        candidates=tuple(pool),
    )


def synthetic_chemlex_adapter() -> DatasetAdapter:
    acids = [f"A{i}" for i in range(6)]
    amines = [f"N{i}" for i in range(6)]
    solvent_polarities = [0.2, 0.5, 0.8]
    base_equivalents = [0.5, 1.0, 1.5, 2.0]
    temperatures = [25.0, 50.0, 75.0, 100.0]
    acid_base = {"A0": 40.0, "A1": 47.0, "A2": 68.0, "A3": 55.0, "A4": 72.0, "A5": 63.0}
    amine_base = {"N0": 0.0, "N1": 8.0, "N2": -4.0, "N3": 11.0, "N4": 5.0, "N5": -2.0}
    pair_bonus = {("A2", "N3"): 10.0, ("A4", "N1"): 8.0, ("A5", "N4"): 6.0}
    pool: list[Candidate] = []
    for acid in acids:
        for amine in amines:
            group = f"{acid}-{amine}"
            for polarity in solvent_polarities:
                for base_eq in base_equivalents:
                    for temp in temperatures:
                        polarity_opt = 0.8 if acid in {"A2", "A4"} else 0.5
                        temp_opt = 75.0 if amine in {"N1", "N3", "N4"} else 50.0
                        solvent_effect = -22.0 * (polarity - polarity_opt) ** 2 + 4.0
                        base_effect = -4.0 * (base_eq - 1.5) ** 2 + 5.0
                        temp_effect = -0.006 * (temp - temp_opt) ** 2 + 4.0
                        y = clamp_score(
                            acid_base[acid]
                            + amine_base[amine]
                            + pair_bonus.get((acid, amine), 0.0)
                            + solvent_effect
                            + base_effect
                            + temp_effect
                            + stable_noise(acid, amine, polarity, base_eq, temp, scale=2.8)
                        )
                        cid = f"{acid}_{amine}_S{int(polarity * 10)}_B{str(base_eq).replace('.', 'p')}_T{int(temp)}"
                        pool.append(
                            Candidate(
                                candidate_id=cid,
                                group=group,
                                x1=polarity,
                                x2=base_eq / 2.0,
                                x3=temp / 100.0,
                                objective_value=y,
                                metadata={
                                    "acid": acid,
                                    "amine": amine,
                                    "solvent_polarity": polarity,
                                    "base_equivalents": base_eq,
                                    "temperature": temp,
                                    "yield_value": y,
                                },
                            )
                        )
    return DatasetAdapter(
        dataset_id="synthetic_chemlex_i",
        title="Synthetic ChemLex-style acid-amine replay",
        objective="maximize_yield",
        decision_columns=("acid", "amine", "solvent_polarity", "base_equivalents", "temperature"),
        hidden_target="yield_value",
        group_column="acid_amine_pair",
        preferred_groups=("A2-N3", "A4-N1", "A5-N4"),
        failure_note="Matched acid-amine pairs still depend on solvent polarity and base equivalents.",
        candidates=tuple(pool),
    )


def synthetic_materials_adapter() -> DatasetAdapter:
    dopants = [f"D{i}" for i in range(7)]
    ratios = [0.10, 0.20, 0.35, 0.50]
    anneal_temps = [300.0, 450.0, 600.0, 750.0]
    dwell_times = [10.0, 30.0, 60.0]
    dopant_base = {"D0": 38.0, "D1": 54.0, "D2": 76.0, "D3": 58.0, "D4": 73.0, "D5": 49.0, "D6": 61.0}
    temp_opt = {"D0": 450.0, "D1": 450.0, "D2": 600.0, "D3": 600.0, "D4": 750.0, "D5": 450.0, "D6": 600.0}
    pool: list[Candidate] = []
    for dopant in dopants:
        for ratio in ratios:
            for temp in anneal_temps:
                for dwell in dwell_times:
                    ratio_effect = -90.0 * (ratio - 0.35) ** 2 + 5.0
                    temp_effect = -0.00016 * (temp - temp_opt[dopant]) ** 2 + 6.0
                    dwell_effect = 4.5 * math.log1p(dwell / 10.0) - 0.045 * dwell
                    interaction = 5.0 if dopant in {"D2", "D4"} and ratio >= 0.20 and temp >= 600.0 else 0.0
                    overcook_penalty = -8.0 if temp >= 750.0 and dwell >= 60.0 else 0.0
                    score = clamp_score(
                        dopant_base[dopant]
                        + ratio_effect
                        + temp_effect
                        + dwell_effect
                        + interaction
                        + overcook_penalty
                        + stable_noise(dopant, ratio, temp, dwell, scale=2.5)
                    )
                    cid = f"{dopant}_R{int(ratio * 100)}_T{int(temp)}_D{int(dwell)}"
                    pool.append(
                        Candidate(
                            candidate_id=cid,
                            group=dopant,
                            x1=ratio / 0.50,
                            x2=temp / 750.0,
                            x3=dwell / 60.0,
                            objective_value=score,
                            metadata={
                                "dopant": dopant,
                                "dopant_ratio": ratio,
                                "anneal_temperature": temp,
                                "dwell_time": dwell,
                                "stability_score": score,
                            },
                        )
                    )
    return DatasetAdapter(
        dataset_id="synthetic_materials_i",
        title="Synthetic materials formulation replay",
        objective="maximize_stability_score",
        decision_columns=("dopant", "dopant_ratio", "anneal_temperature", "dwell_time"),
        hidden_target="stability_score",
        group_column="dopant",
        preferred_groups=("D2", "D4"),
        failure_note="Preferred dopants are sensitive to annealing temperature and long dwell overcooking.",
        candidates=tuple(pool),
    )


def real_buchwald_hartwig_adapter() -> DatasetAdapter:
    path = ensure_public_data_file("dreher_doyle_buchwald_hartwig.xlsx")
    records = rows_to_dicts(read_xlsx_rows(path, "FullCV_01"))
    ligand_labels = label_map([r["Ligand"] for r in records], "L")
    additive_labels = label_map([r["Additive"] for r in records], "A")
    base_labels = label_map([r["Base"] for r in records], "B")
    aryl_labels = label_map([r["Aryl halide"] for r in records], "H")
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        output = row.get("Output")
        if output is None:
            continue
        ligand_raw = clean_component_name(row["Ligand"])
        additive_raw = clean_component_name(row["Additive"])
        base_raw = clean_component_name(row["Base"])
        aryl_raw = clean_component_name(row["Aryl halide"])
        ligand = ligand_labels[str(row["Ligand"])]
        additive = additive_labels[str(row["Additive"])]
        base = base_labels[str(row["Base"])]
        aryl = aryl_labels[str(row["Aryl halide"])]
        pool.append(
            Candidate(
                candidate_id=f"bh_{idx:04d}_{ligand}_{additive}_{base}_{aryl}",
                group=ligand,
                x1=stable_fraction(additive),
                x2=stable_fraction(base),
                x3=stable_fraction(aryl),
                objective_value=clamp_score(float(output)),
                metadata={
                    "ligand": ligand,
                    "additive": additive,
                    "base": base,
                    "aryl_halide": aryl,
                    "ligand_raw_name": ligand_raw,
                    "additive_raw_name": additive_raw,
                    "base_raw_name": base_raw,
                    "aryl_halide_raw_name": aryl_raw,
                    **reaction_component_metadata("real_buchwald_hartwig", "ligand", ligand_raw, "ligand"),
                    **reaction_component_metadata("real_buchwald_hartwig", "additive", additive_raw, "additive"),
                    **reaction_component_metadata("real_buchwald_hartwig", "base", base_raw, "base"),
                    **reaction_component_metadata("real_buchwald_hartwig", "aryl_halide", aryl_raw, "aryl_halide"),
                    "yield_value": clamp_score(float(output)),
                    "source_row": idx + 2,
                },
            )
        )
    return DatasetAdapter(
        dataset_id="real_buchwald_hartwig",
        title="Dreher-Doyle Buchwald-Hartwig HTE replay",
        objective="maximize_yield",
        decision_columns=("ligand", "additive", "base", "aryl_halide"),
        hidden_target="yield_value",
        group_column="ligand",
        preferred_groups=(),
        failure_note="No fixed preferred ligand prior is encoded; the policy may only use revealed observations.",
        candidates=tuple(pool),
    )


def real_suzuki_miyaura_adapter() -> DatasetAdapter:
    path = ensure_public_data_file("perera_suzuki_miyaura.xlsx")
    records = rows_to_dicts(read_xlsx_rows(path, "Sheet1"))
    ligand_labels = label_map([r["Ligand_Short_Hand"] for r in records], "L")
    catalyst_labels = label_map([r["Catalyst_1_Short_Hand"] for r in records], "C")
    reagent_labels = label_map([r["Reagent_1_Short_Hand"] for r in records], "R")
    solvent_labels = label_map([r["Solvent_1_Short_Hand"] for r in records], "S")
    reactant_labels = label_map([r["Reactant_1_Short_Hand"] for r in records], "Q")
    boronic_labels = label_map([r["Reactant_2_Name"] for r in records], "BA")
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        output = row.get("Product_Yield_PCT_Area_UV")
        if output is None:
            continue
        reactant_raw = clean_component_name(row["Reactant_1_Name"])
        boronic_raw = clean_component_name(row["Reactant_2_Name"])
        catalyst_raw = clean_component_name(row["Catalyst_1_Short_Hand"])
        ligand_raw = clean_component_name(row["Ligand_Short_Hand"])
        reagent_raw = clean_component_name(row["Reagent_1_Short_Hand"])
        solvent_raw = clean_component_name(row["Solvent_1_Short_Hand"])
        ligand = ligand_labels[str(row["Ligand_Short_Hand"])]
        catalyst = catalyst_labels[str(row["Catalyst_1_Short_Hand"])]
        reagent = reagent_labels[str(row["Reagent_1_Short_Hand"])]
        solvent = solvent_labels[str(row["Solvent_1_Short_Hand"])]
        reactant = reactant_labels[str(row["Reactant_1_Short_Hand"])]
        boronic = boronic_labels[str(row["Reactant_2_Name"])]
        pool.append(
            Candidate(
                candidate_id=f"sm_{idx:04d}_{reactant}_{boronic}_{catalyst}_{ligand}_{reagent}_{solvent}",
                group=ligand,
                x1=stable_fraction(catalyst),
                x2=stable_fraction(reagent),
                x3=stable_fraction(solvent),
                objective_value=clamp_score(float(output)),
                metadata={
                    "reactant_1": reactant,
                    "reactant_2": boronic,
                    "catalyst": catalyst,
                    "ligand": ligand,
                    "reagent": reagent,
                    "solvent": solvent,
                    "reactant_1_raw_name": reactant_raw,
                    "reactant_2_raw_name": boronic_raw,
                    "catalyst_raw_name": catalyst_raw,
                    "ligand_raw_name": ligand_raw,
                    "reagent_raw_name": reagent_raw,
                    "solvent_raw_name": solvent_raw,
                    **reaction_component_metadata("real_suzuki_miyaura", "reactant_1", reactant_raw, "reactant_1"),
                    **reaction_component_metadata("real_suzuki_miyaura", "reactant_2", boronic_raw, "reactant_2"),
                    **reaction_component_metadata("real_suzuki_miyaura", "catalyst", catalyst_raw, "catalyst"),
                    **reaction_component_metadata("real_suzuki_miyaura", "ligand", ligand_raw, "ligand"),
                    **reaction_component_metadata("real_suzuki_miyaura", "reagent", reagent_raw, "reagent"),
                    **reaction_component_metadata("real_suzuki_miyaura", "solvent", solvent_raw, "solvent"),
                    "yield_value": clamp_score(float(output)),
                    "source_row": idx + 2,
                },
            )
        )
    return DatasetAdapter(
        dataset_id="real_suzuki_miyaura",
        title="Perera Suzuki-Miyaura HTE replay",
        objective="maximize_yield",
        decision_columns=("reactant_1", "reactant_2", "catalyst", "ligand", "reagent", "solvent"),
        hidden_target="yield_value",
        group_column="ligand",
        preferred_groups=(),
        failure_note="No fixed preferred ligand prior is encoded; the policy may only use revealed observations.",
        candidates=tuple(pool),
    )


def real_chemlex_acidamine_adapter() -> DatasetAdapter:
    path = ensure_public_data_file("chemlex_acidamine_wetlab_v3.xlsx")
    records = rows_to_dicts(read_xlsx_rows(path, "Sheet1"))
    acid_labels = label_map([r["Acid"] for r in records], "A")
    amine_labels = label_map([r["Amine"] for r in records], "N")
    reagent_labels = label_map([r["Reagents"] for r in records], "R")
    solvent_labels = label_map([r["Solvent"] for r in records], "S")
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        conversion = row.get("Conversion")
        if conversion is None:
            continue
        acid_smiles = str(row["Acid"])
        amine_smiles = str(row["Amine"])
        reagent_smiles = str(row["Reagents"])
        solvent_smiles = str(row["Solvent"])
        acid = acid_labels[acid_smiles]
        amine = amine_labels[amine_smiles]
        reagent = reagent_labels[reagent_smiles]
        solvent = solvent_labels[solvent_smiles]
        acid_numeric = reaction_component_numeric_features(
            "real_chemlex_acidamine", "acid", acid_smiles
        )
        amine_numeric = reaction_component_numeric_features(
            "real_chemlex_acidamine", "amine", amine_smiles
        )
        reagent_numeric = reaction_component_numeric_features(
            "real_chemlex_acidamine", "reagent", reagent_smiles
        )
        reaction_numeric = (*acid_numeric, *amine_numeric, *reagent_numeric)
        pool.append(
            Candidate(
                candidate_id=f"chemlex_{idx:05d}_{acid}_{amine}_{reagent}_{solvent}",
                group=f"{acid}_{amine}",
                x1=sum(acid_numeric) / len(acid_numeric),
                x2=sum(amine_numeric) / len(amine_numeric),
                x3=sum(reagent_numeric) / len(reagent_numeric),
                objective_value=clamp_score(float(conversion)),
                metadata={
                    "acid": acid,
                    "amine": amine,
                    "reagent": reagent,
                    "solvent": solvent,
                    "acid_smiles": acid_smiles,
                    "amine_smiles": amine_smiles,
                    "reagent_smiles": reagent_smiles,
                    "solvent_smiles": solvent_smiles,
                    **prefixed_smiles_public_descriptors("acid", acid_smiles),
                    **prefixed_smiles_public_descriptors("amine", amine_smiles),
                    **prefixed_smiles_public_descriptors("reagent", reagent_smiles),
                    **prefixed_smiles_public_descriptors("solvent", solvent_smiles),
                    **prefixed_smiles_motif_descriptors("acid", acid_smiles),
                    **prefixed_smiles_motif_descriptors("amine", amine_smiles),
                    **prefixed_smiles_motif_descriptors("reagent", reagent_smiles),
                    **reaction_component_metadata(
                        "real_chemlex_acidamine", "acid", acid_smiles, "acid_rdkit"
                    ),
                    **reaction_component_metadata(
                        "real_chemlex_acidamine", "amine", amine_smiles, "amine_rdkit"
                    ),
                    **reaction_component_metadata(
                        "real_chemlex_acidamine", "reagent", reagent_smiles, "reagent_rdkit"
                    ),
                    "reagent_coupling_family": chemlex_reagent_family(reagent_smiles),
                    "random_split": str(row.get("Random_Split", "")),
                    "stratified_split_one_unseen": str(row.get("Stratified_Split_One_Unseen", "")),
                    "stratified_split_both_unseen": str(row.get("Stratified_Split_Both_Unseen", "")),
                    "conversion_value": clamp_score(float(conversion)),
                    "source_row": idx + 2,
                },
                numeric_features=reaction_numeric,
            )
        )
    return DatasetAdapter(
        dataset_id="real_chemlex_acidamine",
        title="ChemLex acid-amine wetlab conversion replay",
        objective="maximize_conversion",
        decision_columns=("acid", "amine", "reagent", "solvent"),
        hidden_target="conversion_value",
        group_column="acid_amine_pair",
        preferred_groups=(),
        failure_note="This real wetlab replay uses public acid, amine, reagent, and solvent labels; no fixed preferred acid-amine pair prior is encoded.",
        candidates=tuple(pool),
    )


def real_moleculenet_esol_adapter() -> DatasetAdapter:
    path = ensure_public_data_file("moleculenet_esol_delaney.csv")
    records = read_csv_dicts(path)
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        measured_log_s = float(row["measured log solubility in mols per litre"])
        predicted_log_s = float(row["ESOL predicted log solubility in mols per litre"])
        molecular_weight = float(row["Molecular Weight"])
        hbond_donors = int(float(row["Number of H-Bond Donors"]))
        rings = int(float(row["Number of Rings"]))
        rotatable_bonds = int(float(row["Number of Rotatable Bonds"]))
        polar_surface_area = float(row["Polar Surface Area"])
        smiles = row["smiles"].strip()

        mw_bin = numeric_bin(molecular_weight, (150.0, 300.0, 450.0), ("mw_low", "mw_mid", "mw_high", "mw_very_high"))
        donor_bin = numeric_bin(float(hbond_donors), (0.0, 2.0, 5.0), ("donor_none", "donor_low", "donor_mid", "donor_high"))
        ring_bin = numeric_bin(float(rings), (0.0, 2.0, 4.0), ("rings_none", "rings_low", "rings_mid", "rings_high"))
        rotatable_bin = numeric_bin(float(rotatable_bonds), (1.0, 4.0, 8.0), ("rot_low", "rot_mid", "rot_high", "rot_very_high"))
        psa_bin = numeric_bin(polar_surface_area, (25.0, 75.0, 125.0), ("psa_low", "psa_mid", "psa_high", "psa_very_high"))
        smiles_len_bin = numeric_bin(float(len(smiles)), (20.0, 45.0, 80.0), ("smiles_short", "smiles_mid", "smiles_long", "smiles_very_long"))

        # Fixed logS range avoids using dataset min/max as hidden target information.
        normalized_solubility = clamp_score((measured_log_s + 12.0) / 14.0 * 100.0)
        pool.append(
            Candidate(
                candidate_id=f"esol_{idx:04d}",
                group=mw_bin,
                x1=max(0.0, min(1.0, molecular_weight / 700.0)),
                x2=max(0.0, min(1.0, polar_surface_area / 250.0)),
                x3=max(0.0, min(1.0, len(smiles) / 120.0)),
                objective_value=normalized_solubility,
                metadata={
                    "compound_id": row["Compound ID"],
                    "smiles": smiles,
                    "molecular_weight_bin": mw_bin,
                    "hbond_donor_bin": donor_bin,
                    "ring_bin": ring_bin,
                    "rotatable_bond_bin": rotatable_bin,
                    "polar_surface_area_bin": psa_bin,
                    "smiles_length_bin": smiles_len_bin,
                    "esol_predicted_log_solubility": round(predicted_log_s, 4),
                    "measured_log_solubility": round(measured_log_s, 4),
                    "normalized_solubility_score": normalized_solubility,
                    "source_row": idx + 2,
                },
            )
        )
    return DatasetAdapter(
        dataset_id="real_moleculenet_esol",
        title="MoleculeNet ESOL Delaney solubility replay",
        objective="maximize_normalized_solubility",
        decision_columns=(
            "molecular_weight_bin",
            "hbond_donor_bin",
            "ring_bin",
            "rotatable_bond_bin",
            "polar_surface_area_bin",
            "smiles_length_bin",
        ),
        hidden_target="normalized_solubility_score",
        group_column="molecular_weight_bin",
        preferred_groups=(),
        failure_note="This is molecular property replay, not wet-lab reaction optimization; no fixed preferred molecular bin prior is encoded.",
        candidates=tuple(pool),
    )


def real_moleculenet_freesolv_adapter() -> DatasetAdapter:
    path = ensure_public_data_file("moleculenet_freesolv_sampl.csv")
    records = read_csv_dicts(path)
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        smiles = row["smiles"].strip()
        experimental_delta_g = float(row["expt"])
        calculated_delta_g = float(row["calc"])
        descriptor_bins, (x1, x2, x3) = smiles_public_descriptors(smiles)

        # Fixed [-25, 5] kcal/mol scale; more negative hydration free energy is better.
        hydration_affinity_score = clamp_score((5.0 - experimental_delta_g) / 30.0 * 100.0)
        pool.append(
            Candidate(
                candidate_id=f"freesolv_{idx:04d}",
                group=descriptor_bins["hetero_atom_bin"],
                x1=x1,
                x2=x2,
                x3=x3,
                objective_value=hydration_affinity_score,
                metadata={
                    "iupac": row["iupac"],
                    "smiles": smiles,
                    **descriptor_bins,
                    "experimental_hydration_free_energy": round(experimental_delta_g, 4),
                    "calculated_hydration_free_energy": round(calculated_delta_g, 4),
                    "hydration_affinity_score": hydration_affinity_score,
                    "source_row": idx + 2,
                },
            )
        )
    return DatasetAdapter(
        dataset_id="real_moleculenet_freesolv",
        title="MoleculeNet FreeSolv hydration free-energy replay",
        objective="maximize_hydration_affinity_score",
        decision_columns=(
            "smiles_length_bin",
            "hetero_atom_bin",
            "halogen_bin",
            "aromatic_bin",
            "ring_token_bin",
            "branch_bin",
            "double_bond_bin",
        ),
        hidden_target="hydration_affinity_score",
        group_column="hetero_atom_bin",
        preferred_groups=(),
        failure_note="This is molecular property replay over SMILES-derived public descriptors; no fixed preferred chemistry prior is encoded.",
        candidates=tuple(pool),
    )


def real_moleculenet_freesolv_continuous_adapter() -> DatasetAdapter:
    """FreeSolv replay with a fixed monotonic score that avoids clipping ties."""
    base = real_moleculenet_freesolv_adapter()
    candidates: list[Candidate] = []
    for candidate in base.candidates:
        delta_g = float(candidate.metadata["experimental_hydration_free_energy"])
        score = 100.0 / (1.0 + math.exp((delta_g + 5.0) / 5.0))
        metadata = dict(candidate.metadata)
        metadata["hydration_affinity_continuous_score"] = score
        candidates.append(replace(
            candidate,
            objective_value=score,
            metadata=metadata,
        ))
    return replace(
        base,
        dataset_id="real_moleculenet_freesolv_continuous",
        title="MoleculeNet FreeSolv continuous hydration free-energy replay",
        objective="maximize_hydration_affinity_continuous_score",
        hidden_target="hydration_affinity_continuous_score",
        failure_note=(
            "This versioned replay uses a fixed monotonic logistic transform of hydration "
            "free energy so strong molecules remain ordered instead of clipping at 100."
        ),
        candidates=tuple(candidates),
    )


def real_moleculenet_lipophilicity_adapter() -> DatasetAdapter:
    path = ensure_public_data_file("moleculenet_lipophilicity.csv")
    records = read_csv_dicts(path)
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        smiles = row["smiles"].strip()
        experimental_logd = float(row["exp"])
        descriptor_bins, (x1, x2, x3) = smiles_public_descriptors(smiles)

        # Fixed [-3, 5] logD-like scale avoids using dataset min/max as hidden target information.
        normalized_lipophilicity_score = clamp_score((experimental_logd + 3.0) / 8.0 * 100.0)
        pool.append(
            Candidate(
                candidate_id=f"lipo_{idx:04d}",
                group=descriptor_bins["smiles_length_bin"],
                x1=x1,
                x2=x2,
                x3=x3,
                objective_value=normalized_lipophilicity_score,
                metadata={
                    "chembl_id": row["CMPD_CHEMBLID"],
                    "smiles": smiles,
                    **descriptor_bins,
                    "experimental_lipophilicity": round(experimental_logd, 4),
                    "normalized_lipophilicity_score": normalized_lipophilicity_score,
                    "source_row": idx + 2,
                },
            )
        )
    return DatasetAdapter(
        dataset_id="real_moleculenet_lipophilicity",
        title="MoleculeNet Lipophilicity replay",
        objective="maximize_normalized_lipophilicity",
        decision_columns=(
            "smiles_length_bin",
            "hetero_atom_bin",
            "halogen_bin",
            "aromatic_bin",
            "ring_token_bin",
            "branch_bin",
            "double_bond_bin",
        ),
        hidden_target="normalized_lipophilicity_score",
        group_column="smiles_length_bin",
        preferred_groups=(),
        failure_note="This is molecular property replay over SMILES-derived public descriptors; no fixed preferred lipophilicity prior is encoded.",
        candidates=tuple(pool),
    )


def real_matbench_expt_gap_adapter() -> DatasetAdapter:
    path = ensure_public_data_file("matbench_expt_gap.json.gz")
    records = read_matbench_json_gz(path)
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        formula = str(row["composition"]).strip()
        gap_ev = float(row["gap expt"])
        composition = parse_composition(formula)
        total_atoms = sum(composition.values())
        if total_atoms <= 0:
            continue
        elements = set(composition)
        dominant_element = max(composition.items(), key=lambda item: (item[1], item[0]))[0]
        dominant_family = element_family(dominant_element)
        family = anion_family(elements)
        element_count_bin = numeric_bin(
            float(len(elements)),
            (2.0, 4.0, 6.0),
            ("binary", "ternary_quaternary", "quinary_senary", "complex"),
        )
        mean_atomic_number = sum(ELEMENT_Z.get(element, 0) * count for element, count in composition.items()) / total_atoms
        max_fraction = max(composition.values()) / total_atoms
        transition_flag = "has_transition_metal" if elements & TRANSITION_METALS else "no_transition_metal"
        lanthanide_flag = "has_lanthanide" if elements & LANTHANIDES else "no_lanthanide"

        pool.append(
            Candidate(
                candidate_id=f"matbench_expt_gap_{idx:04d}",
                group=family,
                x1=max(0.0, min(1.0, len(elements) / 8.0)),
                x2=max(0.0, min(1.0, mean_atomic_number / 90.0)),
                x3=max(0.0, min(1.0, max_fraction)),
                objective_value=clamp_score(gap_ev / 8.0 * 100.0),
                metadata={
                    "composition": formula,
                    "experimental_band_gap_ev": round(gap_ev, 6),
                    "normalized_band_gap_score": clamp_score(gap_ev / 8.0 * 100.0),
                    "anion_family": family,
                    "element_count_bin": element_count_bin,
                    "dominant_element": dominant_element,
                    "dominant_family": dominant_family,
                    "transition_metal_flag": transition_flag,
                    "lanthanide_flag": lanthanide_flag,
                    "mean_atomic_number_bin": numeric_bin(
                        mean_atomic_number,
                        (20.0, 40.0, 60.0),
                        ("mean_z_low", "mean_z_mid", "mean_z_high", "mean_z_very_high"),
                    ),
                    "max_element_fraction_bin": numeric_bin(
                        max_fraction,
                        (0.34, 0.50, 0.75),
                        ("balanced", "moderately_concentrated", "concentrated", "dominant_element_heavy"),
                    ),
                    "source_row": idx,
                },
            )
        )
    return DatasetAdapter(
        dataset_id="real_matbench_expt_gap",
        title="Matbench experimental band gap replay",
        objective="maximize_normalized_experimental_band_gap",
        decision_columns=(
            "anion_family",
            "element_count_bin",
            "dominant_family",
            "transition_metal_flag",
            "lanthanide_flag",
            "mean_atomic_number_bin",
            "max_element_fraction_bin",
        ),
        hidden_target="normalized_band_gap_score",
        group_column="anion_family",
        preferred_groups=(),
        failure_note="This real materials replay uses composition-only public features and revealed experimental band gaps; no fixed material-family prior is encoded.",
        candidates=tuple(pool),
    )


MATERIAL_DECISION_COLUMNS = (
    "anion_family",
    "element_count_bin",
    "dominant_family",
    "transition_metal_flag",
    "lanthanide_flag",
    "mean_atomic_number_bin",
    "max_element_fraction_bin",
)


def structure_composition(structure: dict[str, Any]) -> dict[str, float]:
    composition: dict[str, float] = {}
    for site in structure.get("sites", []):
        for species in site.get("species", []):
            element = str(species.get("element", "")).strip()
            if not element:
                continue
            composition[element] = composition.get(element, 0.0) + float(species.get("occu", 1.0))
    return composition


def material_public_features(composition: dict[str, float]) -> tuple[dict[str, Any], tuple[float, float, float]]:
    total_atoms = sum(composition.values())
    if total_atoms <= 0:
        raise ValueError("Material composition must contain at least one atom")
    elements = set(composition)
    dominant_element = max(composition.items(), key=lambda item: (item[1], item[0]))[0]
    mean_atomic_number = sum(ELEMENT_Z.get(element, 0) * count for element, count in composition.items()) / total_atoms
    max_fraction = max(composition.values()) / total_atoms
    metadata = {
        "anion_family": anion_family(elements),
        "element_count_bin": numeric_bin(
            float(len(elements)),
            (2.0, 4.0, 6.0),
            ("binary", "ternary_quaternary", "quinary_senary", "complex"),
        ),
        "dominant_element": dominant_element,
        "dominant_family": element_family(dominant_element),
        "transition_metal_flag": "has_transition_metal" if elements & TRANSITION_METALS else "no_transition_metal",
        "lanthanide_flag": "has_lanthanide" if elements & LANTHANIDES else "no_lanthanide",
        "mean_atomic_number_bin": numeric_bin(
            mean_atomic_number,
            (20.0, 40.0, 60.0),
            ("mean_z_low", "mean_z_mid", "mean_z_high", "mean_z_very_high"),
        ),
        "max_element_fraction_bin": numeric_bin(
            max_fraction,
            (0.34, 0.50, 0.75),
            ("balanced", "moderately_concentrated", "concentrated", "dominant_element_heavy"),
        ),
    }
    numeric = (
        max(0.0, min(1.0, len(elements) / 8.0)),
        max(0.0, min(1.0, mean_atomic_number / 90.0)),
        max(0.0, min(1.0, max_fraction)),
    )
    return metadata, numeric


def real_matbench_structure_property_adapter(
    *,
    filename: str,
    dataset_id: str,
    title: str,
    objective: str,
    target_column: str,
    target_metadata_field: str,
    normalize_target: Callable[[float], float],
) -> DatasetAdapter:
    path = ensure_public_data_file(filename)
    records = read_matbench_json_gz(path)
    pool: list[Candidate] = []
    for idx, row in enumerate(records):
        structure = row["structure"]
        composition = structure_composition(structure)
        if not composition:
            continue
        metadata, (x1, x2, x3) = material_public_features(composition)
        raw_target = float(row[target_column])
        normalized_target = clamp_score(normalize_target(raw_target))
        formula = "".join(
            f"{element}{count:g}"
            for element, count in sorted(composition.items())
        )
        pool.append(
            Candidate(
                candidate_id=f"{dataset_id}_{idx:05d}",
                group=str(metadata["anion_family"]),
                x1=x1,
                x2=x2,
                x3=x3,
                objective_value=normalized_target,
                metadata={
                    "composition": formula,
                    **metadata,
                    target_metadata_field: round(raw_target, 6),
                    "normalized_property_score": normalized_target,
                    "source_row": idx,
                },
            )
        )
    return DatasetAdapter(
        dataset_id=dataset_id,
        title=title,
        objective=objective,
        decision_columns=MATERIAL_DECISION_COLUMNS,
        hidden_target="normalized_property_score",
        group_column="anion_family",
        preferred_groups=(),
        failure_note=(
            "This real Matbench replay uses composition-derived public features and sequentially revealed "
            "property values; no target-family preference is encoded."
        ),
        candidates=tuple(pool),
    )


def real_matbench_dielectric_adapter() -> DatasetAdapter:
    return real_matbench_structure_property_adapter(
        filename="matbench_dielectric.json.gz",
        dataset_id="real_matbench_dielectric",
        title="Matbench refractive-index replay",
        objective="maximize_refractive_index",
        target_column="n",
        target_metadata_field="refractive_index",
        normalize_target=lambda value: math.log1p(max(value, 0.0)) / math.log(64.0) * 100.0,
    )


def real_matbench_phonons_adapter() -> DatasetAdapter:
    return real_matbench_structure_property_adapter(
        filename="matbench_phonons.json.gz",
        dataset_id="real_matbench_phonons",
        title="Matbench phonon peak replay",
        objective="maximize_last_phonon_dos_peak",
        target_column="last phdos peak",
        target_metadata_field="last_phonon_dos_peak_cm-1",
        normalize_target=lambda value: value / 4000.0 * 100.0,
    )


def real_matbench_log_kvrh_adapter() -> DatasetAdapter:
    return real_matbench_structure_property_adapter(
        filename="matbench_log_kvrh.json.gz",
        dataset_id="real_matbench_log_kvrh",
        title="Matbench bulk-modulus replay",
        objective="maximize_log10_bulk_modulus",
        target_column="log10(K_VRH)",
        target_metadata_field="log10_bulk_modulus_gpa",
        normalize_target=lambda value: value / 3.0 * 100.0,
    )


DATASET_BUILDERS: dict[str, Callable[[], DatasetAdapter]] = {
    "synthetic_suzuki_i": synthetic_suzuki_adapter,
    "synthetic_chemlex_i": synthetic_chemlex_adapter,
    "synthetic_materials_i": synthetic_materials_adapter,
    "real_buchwald_hartwig": real_buchwald_hartwig_adapter,
    "real_suzuki_miyaura": real_suzuki_miyaura_adapter,
    "real_chemlex_acidamine": real_chemlex_acidamine_adapter,
    "real_moleculenet_esol": real_moleculenet_esol_adapter,
    "real_moleculenet_freesolv": real_moleculenet_freesolv_adapter,
    "real_moleculenet_freesolv_continuous": real_moleculenet_freesolv_continuous_adapter,
    "real_moleculenet_lipophilicity": real_moleculenet_lipophilicity_adapter,
    "real_matbench_expt_gap": real_matbench_expt_gap_adapter,
    "real_matbench_dielectric": real_matbench_dielectric_adapter,
    "real_matbench_phonons": real_matbench_phonons_adapter,
    "real_matbench_log_kvrh": real_matbench_log_kvrh_adapter,
}


def make_task(adapter: DatasetAdapter, initial_observations: int, reveal_budget: int) -> TaskSpec:
    return TaskSpec(
        dataset_id=adapter.dataset_id,
        objective=adapter.objective,
        decision_columns=adapter.decision_columns,
        hidden_target=adapter.hidden_target,
        initial_observations=initial_observations,
        reveal_budget=reveal_budget,
        oracle_value=max(c.objective_value for c in adapter.candidates),
    )


def make_skills(adapter: DatasetAdapter) -> list[SkillCard]:
    skills: list[SkillCard] = [
        SkillCard(
            skill_id="factor_evidence_ranker",
            version="1.0.0",
            family="data_analysis",
            scope="Add bounded adjustments from public factor-level outcome evidence.",
            trigger_rules=({"field": "observed_count", "operator": ">=", "value": 8},),
            bounded_parameters={
                "bonus_cap": {"value": 0.06},
                "penalty_cap": {"value": -0.04},
                "min_support": {"value": 2},
                "effect_threshold": {"value": 6.0},
            },
            certificate_schema={"required": ["scored_candidates", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("access_hidden_outcomes", "directly_select_candidate_id", "modify_observed_data"),
            provenance={"source": "CARE 2.0 public observation model", "dataset": adapter.dataset_id},
            rationale="Use only revealed outcomes to reward condition factors that repeatedly overperform public baseline.",
        ),
        SkillCard(
            skill_id="group_risk_penalty",
            version="1.0.0",
            family="constraint",
            scope=f"Penalize {adapter.group_column} groups that repeatedly underperform in public observations.",
            trigger_rules=({"field": "observed_count", "operator": ">=", "value": 5},),
            bounded_parameters={"penalty_cap": {"value": -0.12}, "min_support": {"value": 2}},
            certificate_schema={"required": ["penalized_groups", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("block_candidate_permanently", "access_hidden_outcomes"),
            provenance={"source": "CARE 2.0 skill specification", "dataset": adapter.dataset_id},
            rationale="Use public repeated failures as bounded risk evidence without permanently blocking candidates.",
        ),
    ]
    if adapter.preferred_groups:
        skills.insert(
            0,
            SkillCard(
                skill_id="group_prior",
                version="1.0.0",
                family="ranker",
                scope=f"Add bounded prior bonus to preferred {adapter.group_column} groups.",
                trigger_rules=({"field": "round_index", "operator": ">=", "value": 0},),
                bounded_parameters={"preferred_groups": {"value": list(adapter.preferred_groups)}, "prior_bonus_cap": {"value": 0.10}},
                certificate_schema={"required": ["applied_bonuses", "max_abs_adjustment"]},
                required_checks=("static", "sandbox", "full_pool", "row_order"),
                prohibited_behaviors=("access_hidden_outcomes", "directly_select_candidate_id", "modify_observed_data"),
                provenance={"source": "CARE 2.0 skill specification", "dataset": adapter.dataset_id},
                rationale="Convert a dataset-level scientific prior into bounded full-pool score adjustments.",
            ),
        )
        skills.append(
            SkillCard(
                skill_id="group_diversity_explorer",
                version="1.0.0",
                family="exploration",
                scope=f"Add a small bounded bonus to unseen {adapter.group_column} groups.",
                trigger_rules=({"field": "observed_count", "operator": ">=", "value": 5},),
                bounded_parameters={"unseen_group_bonus_cap": {"value": 0.05}},
                certificate_schema={"required": ["unseen_groups", "max_abs_adjustment"]},
                required_checks=("static", "sandbox", "full_pool", "row_order"),
                prohibited_behaviors=("access_hidden_outcomes", "directly_select_candidate_id"),
                provenance={"source": "CARE 2.0 skill specification", "dataset": adapter.dataset_id},
                rationale="Encourage controlled exploration of public feature groups not yet covered by observations.",
            )
        )
    return skills


def make_hypothesis(adapter: DatasetAdapter) -> HypothesisEntry:
    preferred = ", ".join(adapter.preferred_groups)
    if preferred:
        claim = f"Groups {preferred} tend to produce higher objective values under suitable conditions."
        evidence_summary = "Initialized from dataset prior; no reveal evidence yet."
    else:
        claim = f"No fixed preferred {adapter.group_column} prior is encoded for this dataset."
        evidence_summary = "Initialized without a target-specific prior; updates only use revealed observations."
    return HypothesisEntry(
        hypothesis_id=f"{adapter.dataset_id}_preferred_group_hypothesis",
        status="active",
        scope="group_preference",
        trigger={"dataset": adapter.dataset_id, "condition": "observed_count >= 4"},
        target_spec={"group_column": adapter.group_column, "group_values": list(adapter.preferred_groups)},
        claim=claim,
        confidence=0.5,
        support_count=0,
        alpha=1.0,
        beta=1.0,
        evidence_summary=evidence_summary,
        known_failure_modes=[adapter.failure_note],
        created_round=0,
        last_updated_round=0,
    )


def observed_mean(observed: list[Candidate]) -> float:
    return mean(c.objective_value for c in observed) if observed else 50.0


def group_stats(observed: list[Candidate]) -> dict[str, tuple[int, float]]:
    by_group: dict[str, list[float]] = {}
    for c in observed:
        by_group.setdefault(c.group, []).append(c.objective_value)
    return {group: (len(vals), mean(vals)) for group, vals in by_group.items()}


def factor_values(candidate: Candidate, decision_columns: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for col in decision_columns:
        if col in candidate.metadata:
            values.append((col, str(candidate.metadata[col])))
    return tuple(values)


def factor_stats(observed: list[Candidate], decision_columns: tuple[str, ...]) -> dict[tuple[str, str], tuple[int, float]]:
    by_factor: dict[tuple[str, str], list[float]] = {}
    for c in observed:
        for key in factor_values(c, decision_columns):
            by_factor.setdefault(key, []).append(c.objective_value)
    return {key: (len(vals), mean(vals)) for key, vals in by_factor.items()}


def smoothed_mean(count: int, value_mean: float, global_mean: float, prior_weight: float = 2.0) -> float:
    return (count * value_mean + prior_weight * global_mean) / (count + prior_weight)


def public_incumbent_scores(
    adapter: DatasetAdapter,
    observed_ids: set[str],
    observed: list[Candidate],
) -> dict[str, float]:
    group_summary = group_stats(observed)
    factor_summary = factor_stats(observed, adapter.decision_columns)
    global_mean = observed_mean(observed)
    scores: dict[str, float] = {}
    for c in adapter.candidates:
        if c.candidate_id in observed_ids:
            continue
        group_count, group_mean = group_summary.get(c.group, (0, global_mean))
        estimates = [smoothed_mean(group_count, group_mean, global_mean, prior_weight=3.0)]
        support_counts = [group_count]
        for key in factor_values(c, adapter.decision_columns):
            count, value_mean = factor_summary.get(key, (0, global_mean))
            estimates.append(smoothed_mean(count, value_mean, global_mean, prior_weight=2.0))
            support_counts.append(count)
        public_mean_estimate = mean(estimates)
        uncertainty = 8.0 / math.sqrt(max(support_counts) + 1.0)
        public_condition_prior = 0.035 * c.x1 + 0.020 * c.x2 + 0.015 * c.x3
        estimated = (public_mean_estimate + uncertainty) / 100.0 + public_condition_prior
        scores[c.candidate_id] = estimated
    return scores


def trigger_satisfied(rule: dict[str, Any], round_index: int, observed: list[Candidate]) -> bool:
    field_name = rule["field"]
    value = rule["value"]
    actual = round_index if field_name == "round_index" else len(observed)
    op = rule["operator"]
    if op == ">=":
        return actual >= value
    if op == "==":
        return actual == value
    raise ValueError(f"Unsupported operator: {op}")


def skill_adjustments(
    adapter: DatasetAdapter,
    pool: tuple[Candidate, ...],
    observed_ids: set[str],
    observed: list[Candidate],
    skills: list[SkillCard],
    round_index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    cert: dict[str, Any] = {"skills": {}, "max_abs_adjustment": 0.0}
    stats = group_stats(observed)
    factor_summary = factor_stats(observed, adapter.decision_columns)
    global_mean = observed_mean(observed)
    seen_groups = {c.group for c in observed}
    for skill in skills:
        if not all(trigger_satisfied(rule, round_index, observed) for rule in skill.trigger_rules):
            cert["skills"][skill.skill_id] = {"active": False, "reason": "trigger_not_satisfied"}
            continue
        if skill.skill_id == "group_prior":
            preferred = set(skill.bounded_parameters["preferred_groups"]["value"])
            cap = float(skill.bounded_parameters["prior_bonus_cap"]["value"])
            applied = {}
            for c in pool:
                if c.candidate_id not in adjustments or c.group not in preferred:
                    continue
                adjustments[c.candidate_id] += cap
                applied[c.candidate_id] = cap
            cert["skills"][skill.skill_id] = {"active": True, "applied_count": len(applied), "cap": cap}
        elif skill.skill_id == "factor_evidence_ranker":
            bonus_cap = float(skill.bounded_parameters["bonus_cap"]["value"])
            penalty_cap = float(skill.bounded_parameters["penalty_cap"]["value"])
            min_support = int(skill.bounded_parameters["min_support"]["value"])
            threshold = float(skill.bounded_parameters["effect_threshold"]["value"])
            scored = 0
            positive = 0
            negative = 0
            for c in pool:
                if c.candidate_id not in adjustments:
                    continue
                signals: list[float] = []
                for key in factor_values(c, adapter.decision_columns):
                    count, value_mean = factor_summary.get(key, (0, global_mean))
                    if count < min_support:
                        continue
                    effect = smoothed_mean(count, value_mean, global_mean) - global_mean
                    if abs(effect) < threshold:
                        continue
                    signals.append(effect / 100.0)
                if not signals:
                    continue
                delta = mean(signals)
                bounded = max(penalty_cap, min(bonus_cap, delta))
                adjustments[c.candidate_id] += bounded
                scored += 1
                positive += int(bounded > 0)
                negative += int(bounded < 0)
            cert["skills"][skill.skill_id] = {
                "active": True,
                "scored_candidates": scored,
                "positive_adjustments": positive,
                "negative_adjustments": negative,
                "bonus_cap": bonus_cap,
                "penalty_cap": penalty_cap,
            }
        elif skill.skill_id == "group_risk_penalty":
            cap = float(skill.bounded_parameters["penalty_cap"]["value"])
            min_support = int(skill.bounded_parameters["min_support"]["value"])
            risky = {group for group, (count, group_mean) in stats.items() if count >= min_support and group_mean < global_mean - 8.0}
            for c in pool:
                if c.candidate_id in adjustments and c.group in risky:
                    adjustments[c.candidate_id] += cap
            cert["skills"][skill.skill_id] = {"active": True, "penalized_groups": sorted(risky), "cap": cap}
        elif skill.skill_id == "group_diversity_explorer":
            cap = float(skill.bounded_parameters["unseen_group_bonus_cap"]["value"])
            unseen = sorted({c.group for c in pool} - seen_groups)
            for c in pool:
                if c.candidate_id in adjustments and c.group in unseen:
                    adjustments[c.candidate_id] += cap
            cert["skills"][skill.skill_id] = {"active": True, "unseen_groups": unseen, "cap": cap}
    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert["max_abs_adjustment"] = round(max_abs, 6)
    return adjustments, cert


def top_candidate(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]


def row_order_stability_check(
    adapter: DatasetAdapter,
    pool: tuple[Candidate, ...],
    observed_ids: set[str],
    observed: list[Candidate],
    skills: list[SkillCard],
    round_index: int,
    reference_adjustments: dict[str, float],
) -> bool:
    shuffled = list(pool)
    random.Random(1000 + round_index + len(observed)).shuffle(shuffled)
    shuffled_adjustments, _ = skill_adjustments(adapter, tuple(shuffled), observed_ids, observed, skills, round_index)
    return all(abs(reference_adjustments[k] - shuffled_adjustments[k]) < 1e-12 for k in reference_adjustments)


def gate_decision(
    gate_version: str,
    base_scores: dict[str, float],
    adjusted_scores: dict[str, float],
    adjustments: dict[str, float],
    row_order_stable: bool,
    active_skill_ids: tuple[str, ...],
) -> GateCertificate:
    incumbent = top_candidate(base_scores)
    challenger = top_candidate(adjusted_scores)
    if challenger == incumbent:
        return GateCertificate(
            gate_version=gate_version,
            incumbent_candidate=incumbent,
            challenger_candidate=challenger,
            selected_candidate=incumbent,
            authorized=False,
            gate_margin=0.0,
            acquisition_loss=0.0,
            row_order_stable=row_order_stable,
            applied_skill_ids=active_skill_ids,
            reason="challenger_matches_incumbent",
        )
    epsilon = 0.05 if gate_version in {"gate_v1", "llm_gate_v1"} else 0.12
    min_margin = 0.025 if gate_version in {"gate_v1", "llm_gate_v1"} else 0.010
    gate_margin = adjusted_scores[challenger] - base_scores[incumbent]
    acquisition_loss = max(0.0, base_scores[incumbent] - base_scores[challenger])
    max_adjustment = max(abs(v) for v in adjustments.values()) if adjustments else 0.0
    authorized = row_order_stable and max_adjustment <= 0.20 and gate_margin >= min_margin and acquisition_loss <= epsilon
    reason = "authorized_bounded_skill_adjustment" if authorized else "rejected_by_gate_bounds"
    return GateCertificate(
        gate_version=gate_version,
        incumbent_candidate=incumbent,
        challenger_candidate=challenger,
        selected_candidate=challenger if authorized else incumbent,
        authorized=authorized,
        gate_margin=round(gate_margin, 6),
        acquisition_loss=round(acquisition_loss, 6),
        row_order_stable=row_order_stable,
        applied_skill_ids=active_skill_ids,
        reason=reason,
    )


def exploration_gate_decision(
    gate_version: str,
    base_scores: dict[str, float],
    adjusted_scores: dict[str, float],
    adjustments: dict[str, float],
    novelty_scores: dict[str, float],
    row_order_stable: bool,
    active_skill_ids: tuple[str, ...],
    seed: int,
    round_index: int,
    proposal_confidence: float = 0.5,
    prior_exploration_interventions: int = 0,
    public_best_value: float = 0.0,
) -> GateCertificate:
    incumbent = top_candidate(base_scores)
    challenger = top_candidate(adjusted_scores)
    if challenger == incumbent:
        return GateCertificate(
            gate_version=gate_version,
            incumbent_candidate=incumbent,
            challenger_candidate=challenger,
            selected_candidate=incumbent,
            authorized=False,
            gate_margin=0.0,
            acquisition_loss=0.0,
            row_order_stable=row_order_stable,
            applied_skill_ids=active_skill_ids,
            reason="challenger_matches_incumbent",
        )

    gate_margin = adjusted_scores[challenger] - base_scores[incumbent]
    acquisition_loss = max(0.0, base_scores[incumbent] - base_scores[challenger])
    max_adjustment = max((abs(value) for value in adjustments.values()), default=0.0)
    novelty = max(0.0, min(1.0, novelty_scores.get(challenger, 0.0)))

    # Exploration is most valuable early and should become increasingly expensive.
    risk_budget = max(0.015, 0.060 * math.exp(-0.22 * round_index))
    temperature = max(0.012, 0.040 * math.exp(-0.18 * round_index))
    calibrated_confidence = max(0.0, min(1.0, proposal_confidence))
    has_exploration_headroom = public_best_value < 90.0
    acceptance_probability = min(
        0.70,
        (0.15 + 0.55 * novelty)
        * (0.45 + 0.55 * calibrated_confidence)
        * math.exp(-acquisition_loss / temperature),
    )
    deterministic_draw = random.Random(
        (seed + 1) * 1_000_003 + (round_index + 1) * 9_176
    ).random()
    risk_authorized = (
        acquisition_loss <= risk_budget
        and calibrated_confidence >= 0.55
        and public_best_value < 95.0
        and deterministic_draw <= acceptance_probability
    )
    quota_authorized = (
        round_index <= 2
        and prior_exploration_interventions == 0
        and acquisition_loss <= risk_budget
        and calibrated_confidence >= 0.62
        and has_exploration_headroom
    )
    authorized = (
        row_order_stable
        and max_adjustment <= 0.20
        and (quota_authorized or risk_authorized)
    )
    if authorized and quota_authorized:
        reason = "authorized_exploration_quota"
    elif authorized:
        reason = "authorized_exploration_risk_budget"
    elif not has_exploration_headroom:
        reason = "rejected_exploration_insufficient_headroom"
    elif calibrated_confidence < 0.55:
        reason = "rejected_exploration_low_confidence"
    elif acquisition_loss > risk_budget:
        reason = "rejected_exploration_loss_above_budget"
    else:
        reason = "rejected_exploration_probability"
    return GateCertificate(
        gate_version=gate_version,
        incumbent_candidate=incumbent,
        challenger_candidate=challenger,
        selected_candidate=challenger if authorized else incumbent,
        authorized=authorized,
        gate_margin=round(gate_margin, 6),
        acquisition_loss=round(acquisition_loss, 6),
        row_order_stable=row_order_stable,
        applied_skill_ids=active_skill_ids,
        reason=reason,
    )


def no_gate_decision(
    base_scores: dict[str, float],
    adjusted_scores: dict[str, float],
    row_order_stable: bool,
    active_skill_ids: tuple[str, ...],
) -> GateCertificate:
    incumbent = top_candidate(base_scores)
    challenger = top_candidate(adjusted_scores)
    gate_margin = adjusted_scores[challenger] - base_scores[incumbent]
    acquisition_loss = max(0.0, base_scores[incumbent] - base_scores[challenger])
    return GateCertificate(
        gate_version="no_gate",
        incumbent_candidate=incumbent,
        challenger_candidate=challenger,
        selected_candidate=challenger,
        authorized=challenger != incumbent,
        gate_margin=round(gate_margin, 6),
        acquisition_loss=round(acquisition_loss, 6),
        row_order_stable=row_order_stable,
        applied_skill_ids=active_skill_ids,
        reason="ablation_gate_disabled_challenger_selected"
        if challenger != incumbent
        else "challenger_matches_incumbent",
    )


def is_llm_mode(mode: str) -> bool:
    return mode in {
        "llm_no_gate",
        "llm_gate_v1",
        "llm_explore_no_gate",
        "llm_explore_gate_v1",
    }


def is_exploration_llm_mode(mode: str) -> bool:
    return mode in {"llm_explore_no_gate", "llm_explore_gate_v1"}


def compact_candidate(
    c: Candidate,
    public_fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    public_metadata = {
        key: value
        for key, value in c.metadata.items()
        if (public_fields is None or key in public_fields)
        and key not in {
            "yield_value",
            "conversion_value",
            "stability_score",
            "normalized_solubility_score",
            "hydration_affinity_score",
            "normalized_lipophilicity_score",
            "normalized_band_gap_score",
        }
    }
    return {
        "candidate_id": c.candidate_id,
        "group": c.group,
        "x1": round(c.x1, 4),
        "x2": round(c.x2, 4),
        "x3": round(c.x3, 4),
        "revealed_objective": round(c.objective_value, 4),
        "metadata": public_metadata,
    }


def observed_evidence_payload(adapter: DatasetAdapter, observed: list[Candidate]) -> dict[str, Any]:
    global_mean = observed_mean(observed)
    factor_rows = []
    for (field_name, value), (count, value_mean) in factor_stats(observed, adapter.decision_columns).items():
        if count < 2:
            continue
        factor_rows.append(
            {
                "field": field_name,
                "value": value,
                "count": count,
                "mean": round(value_mean, 4),
                "delta_vs_global": round(value_mean - global_mean, 4),
            }
        )
    factor_rows = sorted(factor_rows, key=lambda item: (abs(item["delta_vs_global"]), item["count"]), reverse=True)[:24]
    top_observed = sorted(observed, key=lambda c: c.objective_value, reverse=True)[:6]
    bottom_observed = sorted(observed, key=lambda c: c.objective_value)[:6]
    return {
        "dataset": adapter.dataset_id,
        "objective": adapter.objective,
        "decision_columns": list(adapter.decision_columns),
        "hidden_target": adapter.hidden_target,
        "group_column": adapter.group_column,
        "observed_count": len(observed),
        "global_revealed_mean": round(global_mean, 4),
        "factor_evidence": factor_rows,
        "top_revealed": [compact_candidate(c, adapter.decision_columns) for c in top_observed],
        "bottom_revealed": [compact_candidate(c, adapter.decision_columns) for c in bottom_observed],
        "output_contract": {
            "adjustments": [
                {
                    "field": "one decision column",
                    "value": "one observed factor value",
                    "direction": "prefer or penalize",
                    "weight": "number between 0.0 and 0.08",
                    "reason": "short evidence-based reason",
                }
            ],
            "confidence": "number between 0 and 1",
        },
    }


def factor_coverage_payload(
    adapter: DatasetAdapter,
    observed: list[Candidate],
) -> list[dict[str, Any]]:
    observed_summary = factor_stats(observed, adapter.decision_columns)
    pool_counts: dict[tuple[str, str], int] = {}
    for candidate in adapter.candidates:
        for key in factor_values(candidate, adapter.decision_columns):
            pool_counts[key] = pool_counts.get(key, 0) + 1
    rows = []
    for (field_name, value), pool_count in pool_counts.items():
        observed_count = observed_summary.get((field_name, value), (0, 0.0))[0]
        rows.append(
            {
                "field": field_name,
                "value": value,
                "observed_count": observed_count,
                "pool_count": pool_count,
                "coverage_ratio": round(observed_count / max(1, pool_count), 4),
                "uncertainty_score": round(1.0 / math.sqrt(observed_count + 1.0), 4),
                "coverage_state": (
                    "unseen"
                    if observed_count == 0
                    else "low_support"
                    if observed_count == 1
                    else "covered"
                ),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["observed_count"],
            -item["pool_count"],
            item["field"],
            item["value"],
        ),
    )[:80]


def exploration_evidence_payload(
    adapter: DatasetAdapter,
    observed: list[Candidate],
    round_index: int,
) -> dict[str, Any]:
    payload = observed_evidence_payload(adapter, observed)
    exploration_budget = max(
        0.015,
        min(0.055, 0.055 * math.exp(-0.12 * max(0, len(observed) - 8))),
    )
    payload["round_index"] = round_index
    payload["factor_coverage"] = factor_coverage_payload(adapter, observed)
    payload["exploration_budget"] = round(exploration_budget, 4)
    payload["decision_goal"] = (
        "Balance exploitation, uncertainty reduction, and failure avoidance. "
        "A useful proposal should be capable of changing the next-candidate ranking."
    )
    payload["output_contract"] = {
        "decision_summary": {
            "hypothesis": "short testable hypothesis",
            "counter_hypothesis": "short plausible alternative",
            "uncertainty_target": "which low-support factor should be tested and why",
            "evidence_for": "concise evidence summary",
            "evidence_against": "concise contradictory evidence or risk",
        },
        "adjustments": [
            {
                "field": "one decision column",
                "value": "one public factor value",
                "direction": "prefer or penalize",
                "intent": "explore, exploit, or avoid",
                "weight": "number between 0.01 and 0.08",
                "reason": "short auditable decision reason",
            }
        ],
        "confidence": "evidence-calibrated number between 0 and 1",
    }
    return payload


def candidate_novelty_scores(
    adapter: DatasetAdapter,
    observed_ids: set[str],
    observed: list[Candidate],
) -> dict[str, float]:
    observed_summary = factor_stats(observed, adapter.decision_columns)
    scores: dict[str, float] = {}
    for candidate in adapter.candidates:
        if candidate.candidate_id in observed_ids:
            continue
        signals = [
            1.0 / math.sqrt(observed_summary.get(key, (0, 0.0))[0] + 1.0)
            for key in factor_values(candidate, adapter.decision_columns)
        ]
        scores[candidate.candidate_id] = mean(signals) if signals else 0.0
    return scores


def calibrate_exploration_confidence(
    raw_confidence: float,
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    bounded_raw = max(0.0, min(1.0, raw_confidence))
    if not specs:
        return {
            "raw": round(bounded_raw, 4),
            "calibrated": 0.0,
            "intent_coverage": 0.0,
        }
    evidence_strength = mean(float(spec["evidence_strength"]) for spec in specs)
    intents = {str(spec["intent"]) for spec in specs}
    intent_coverage = len(intents) / 3.0
    calibrated = max(
        0.05,
        min(0.95, 0.25 * bounded_raw + 0.65 * evidence_strength + 0.10 * intent_coverage),
    )
    return {
        "raw": round(bounded_raw, 4),
        "calibrated": round(calibrated, 4),
        "intent_coverage": round(intent_coverage, 4),
        "evidence_strength": round(evidence_strength, 4),
    }


def normalize_llm_adjustments(
    adapter: DatasetAdapter,
    pool: tuple[Candidate, ...],
    observed_ids: set[str],
    observed: list[Candidate],
    parsed: dict[str, Any],
    exploration_mode: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    adjustments = {
        candidate.candidate_id: 0.0
        for candidate in pool
        if candidate.candidate_id not in observed_ids
    }
    allowed_fields = set(adapter.decision_columns)
    public_values = {
        key
        for candidate in pool
        for key in factor_values(candidate, adapter.decision_columns)
    }
    factor_summary = factor_stats(observed, adapter.decision_columns)
    global_mean = observed_mean(observed)
    exploration_budget = max(
        0.015,
        min(0.055, 0.055 * math.exp(-0.12 * max(0, len(observed) - 8))),
    )
    proposed_specs: list[dict[str, Any]] = []
    max_specs = 6 if exploration_mode else 4
    for item in list(parsed.get("adjustments", []))[:max_specs]:
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field", ""))
        value = str(item.get("value", ""))
        direction = str(item.get("direction", "")).lower()
        if field_name not in allowed_fields or direction not in {"prefer", "penalize"}:
            continue
        if exploration_mode and (field_name, value) not in public_values:
            continue
        try:
            requested_weight = max(0.0, abs(float(item.get("weight", 0.0))))
        except (TypeError, ValueError):
            continue
        if requested_weight <= 0.0:
            continue

        count, value_mean = factor_summary.get((field_name, value), (0, global_mean))
        delta_vs_global = value_mean - global_mean if count else 0.0
        intent = str(item.get("intent", "")).strip().lower()
        if intent not in {"explore", "exploit", "avoid"}:
            intent = "avoid" if direction == "penalize" else "exploit"

        evidence_strength = 0.5
        max_weight = 0.08
        if exploration_mode:
            if intent == "explore":
                if direction != "prefer" or count > 1:
                    continue
                evidence_strength = 1.0 / math.sqrt(count + 1.0)
                max_weight = exploration_budget
            elif intent == "avoid":
                if direction != "penalize" or count < 2 or delta_vs_global >= 0.0:
                    continue
                evidence_strength = min(1.0, count / 6.0) * min(1.0, abs(delta_vs_global) / 15.0)
            else:
                if direction != "prefer" or count < 2 or delta_vs_global <= 0.0:
                    continue
                evidence_strength = min(1.0, count / 6.0) * min(1.0, delta_vs_global / 15.0)
        proposed_specs.append(
            {
                "field": field_name,
                "value": value,
                "direction": direction,
                "intent": intent,
                "requested_weight": min(max_weight, requested_weight),
                "support_count": count,
                "delta_vs_global": round(delta_vs_global, 4),
                "evidence_strength": round(evidence_strength, 4),
                "reason": str(item.get("reason", ""))[:240],
            }
        )

    try:
        raw_confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        raw_confidence = 0.0
    confidence = (
        calibrate_exploration_confidence(raw_confidence, proposed_specs)
        if exploration_mode
        else {
            "raw": round(max(0.0, min(1.0, raw_confidence)), 4),
            "calibrated": round(max(0.0, min(1.0, raw_confidence)), 4),
            "intent_coverage": 0.0,
        }
    )
    confidence_scale = (
        0.55 + 0.45 * float(confidence["calibrated"])
        if exploration_mode
        else 1.0
    )
    applied_specs: list[dict[str, Any]] = []
    for spec in proposed_specs:
        evidence_scale = (
            0.65 + 0.35 * float(spec["evidence_strength"])
            if exploration_mode
            else 1.0
        )
        magnitude = min(0.08, float(spec["requested_weight"])) * confidence_scale * evidence_scale
        delta = magnitude if spec["direction"] == "prefer" else -magnitude
        matched = 0
        for candidate in pool:
            if candidate.candidate_id not in adjustments:
                continue
            if str(candidate.metadata.get(spec["field"], "")) != spec["value"]:
                continue
            adjustments[candidate.candidate_id] += delta
            matched += 1
        applied_specs.append(
            {
                **spec,
                "weight": round(delta, 6),
                "matched_candidates": matched,
            }
        )

    for candidate_id, value in list(adjustments.items()):
        adjustments[candidate_id] = max(-0.12, min(0.12, value))
    return adjustments, applied_specs, confidence


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in LLM response.")


def repair_llm_response_shape(parsed: dict[str, Any]) -> tuple[dict[str, Any], str]:
    decision_summary = parsed.get("decision_summary")
    if (
        "adjustments" not in parsed
        and isinstance(decision_summary, dict)
        and isinstance(decision_summary.get("adjustments"), list)
    ):
        repaired = dict(parsed)
        repaired["adjustments"] = decision_summary["adjustments"]
        repaired["confidence"] = decision_summary.get(
            "confidence",
            parsed.get("confidence", 0.0),
        )
        repaired["decision_summary"] = {
            key: value
            for key, value in decision_summary.items()
            if key not in {"adjustments", "confidence"}
        }
        return repaired, "hoisted_adjustments_from_decision_summary"
    return parsed, ""


def trace_llm_event(event: dict[str, Any]) -> None:
    trace_path = os.environ.get("CARE_LLM_TRACE_LOG")
    if not trace_path:
        return
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **event,
    }
    try:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        return


def chat_completion_text(config: LLMConfig, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    if config.api_mode not in {"chat", "completion"}:
        raise ValueError(f"Unknown LLM API mode: {config.api_mode}")
    if config.api_mode == "chat":
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if config.structured_mode == "tool":
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "return_json",
                        "description": "Return exactly the JSON object requested by the prompt.",
                        "parameters": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                }
            ]
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "return_json"},
            }
        elif config.structured_mode != "json":
            raise ValueError(f"Unknown structured output mode: {config.structured_mode}")
        endpoint = "/chat/completions"
    else:
        prompt = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in messages
        )
        payload = {
            "model": config.model,
            "prompt": prompt + "\n\nASSISTANT:\n",
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        endpoint = "/completions"
    retryable_http = {408, 409, 425, 429, 500, 502, 503, 504}
    retryable_errors = (
        TimeoutError,
        socket.timeout,
        ConnectionError,
        urllib.error.URLError,
        http.client.RemoteDisconnected,
    )
    last_error: BaseException | None = None
    call_id = f"{os.getpid()}-{time.time_ns()}"
    call_started = time.monotonic()
    trace_llm_event(
        {
            "event": "call_start",
            "call_id": call_id,
            "model": config.model,
            "base_url": config.base_url.rstrip("/"),
            "api_mode": config.api_mode,
            "structured_mode": config.structured_mode,
            "message_count": len(messages),
            "max_tokens": config.max_tokens,
            "response_format": payload.get("response_format"),
            "tool_choice": payload.get("tool_choice"),
        }
    )
    for attempt in range(4):
        req = urllib.request.Request(
            config.base_url.rstrip("/") + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + config.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            attempt_started = time.monotonic()
            trace_llm_event({"event": "attempt_start", "call_id": call_id, "attempt": attempt + 1})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            trace_llm_event(
                {
                    "event": "attempt_ok",
                    "call_id": call_id,
                    "attempt": attempt + 1,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                    "response_model": data.get("model", config.model),
                    "usage": data.get("usage", {}),
                }
            )
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code not in retryable_http or attempt == 3:
                trace_llm_event(
                    {
                        "event": "call_error",
                        "call_id": call_id,
                        "attempt": attempt + 1,
                        "elapsed_seconds": round(time.monotonic() - call_started, 3),
                        "error_type": type(exc).__name__,
                        "http_code": exc.code,
                        "retryable": False,
                    }
                )
                raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {body[:500]}") from exc
            last_error = RuntimeError(f"LLM endpoint returned retryable HTTP {exc.code}: {body[:500]}")
            trace_llm_event(
                {
                    "event": "attempt_retry",
                    "call_id": call_id,
                    "attempt": attempt + 1,
                    "elapsed_seconds": round(time.monotonic() - call_started, 3),
                    "error_type": type(exc).__name__,
                    "http_code": exc.code,
                    "retryable": True,
                }
            )
        except retryable_errors as exc:
            if attempt == 3:
                trace_llm_event(
                    {
                        "event": "call_error",
                        "call_id": call_id,
                        "attempt": attempt + 1,
                        "elapsed_seconds": round(time.monotonic() - call_started, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:200],
                        "retryable": False,
                    }
                )
                raise RuntimeError(f"LLM endpoint request failed after retries: {exc}") from exc
            last_error = exc
            trace_llm_event(
                {
                    "event": "attempt_retry",
                    "call_id": call_id,
                    "attempt": attempt + 1,
                    "elapsed_seconds": round(time.monotonic() - call_started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:200],
                    "retryable": True,
                }
            )
        time.sleep(2.0 * (attempt + 1))
    else:
        trace_llm_event(
            {
                "event": "call_error",
                "call_id": call_id,
                "elapsed_seconds": round(time.monotonic() - call_started, 3),
                "error_type": type(last_error).__name__ if last_error else "UnknownError",
                "error": str(last_error)[:200],
                "retryable": False,
            }
        )
        raise RuntimeError(f"LLM endpoint request failed after retries: {last_error}")
    if config.api_mode == "chat":
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            content = tool_calls[0].get("function", {}).get("arguments", "") or content
    else:
        content = data["choices"][0].get("text", "")
        tool_calls = []
    usage = data.get("usage", {})
    trace_llm_event(
        {
            "event": "call_ok",
            "call_id": call_id,
            "elapsed_seconds": round(time.monotonic() - call_started, 3),
            "response_model": data.get("model", config.model),
            "tool_call_count": len(tool_calls),
            "usage": usage,
        }
    )
    return content, {"model": data.get("model", config.model), "usage": usage}


def llm_skill_adjustments(
    adapter: DatasetAdapter,
    pool: tuple[Candidate, ...],
    observed_ids: set[str],
    observed: list[Candidate],
    seed: int,
    round_index: int,
    mode: Mode,
    config: LLMConfig,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    exploration_mode = is_exploration_llm_mode(mode)
    skill_id = "llm_exploration_policy" if exploration_mode else "llm_factor_policy"
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    if len(observed) < 8:
        cert = {
            "skills": {skill_id: {"active": False, "reason": "observed_count_below_8"}},
            "max_abs_adjustment": 0.0,
        }
        record = {
            "called": False,
            "policy_variant": "exploration_aware" if exploration_mode else "legacy",
            "reason": "observed_count_below_8",
        }
        return adjustments, cert, record

    if exploration_mode:
        prompt_payload = exploration_evidence_payload(adapter, observed, round_index)
        system = (
            "You are the exploration-aware policy proposer for CARE 2.0 finite-pool scientific optimization. "
            "Use only revealed outcomes and the public factor coverage supplied in the user JSON. "
            "Never infer or claim hidden outcomes for unrevealed candidates. Analyze the tradeoffs privately, "
            "then return only a concise auditable JSON decision; do not return chain-of-thought or markdown."
        )
        user = (
            "Propose a diversified factor-level policy for the next selection. The adjustments must be capable "
            "of changing the ranking, but should respect the exploration_budget. Include one explore/prefer "
            "adjustment for an unseen or low-support value when available, one avoid/penalize adjustment for a "
            "supported negative factor when available, and no more than two exploit/prefer adjustments. "
            "Do not give every adjustment the same weight. Use roughly 0.01-0.04 for uncertain exploration, "
            "0.01-0.03 for weak evidence, and 0.06-0.08 only for strong repeated evidence. Calibrate confidence "
            "from support, effect size, and contradictory evidence instead of using a default value. Return only "
            "the JSON shape specified by output_contract. JSON input:\n"
            + json.dumps(prompt_payload, ensure_ascii=False)
        )
    else:
        prompt_payload = observed_evidence_payload(adapter, observed)
        system = (
            "You are a CARE policy proposer for scientific finite-pool replay. "
            "Use only the revealed observations in the user JSON. "
            "Do not assume hidden outcomes for unrevealed candidates. "
            "Return only one JSON object with an adjustments array and confidence. "
            "No markdown. No prose. No chain-of-thought."
        )
        user = (
            "Propose bounded factor-level score adjustments for the next candidate selection. "
            "Use fields only from decision_columns. Use values that are supported by factor_evidence "
            "or shown in top_revealed/bottom_revealed. Prefer high-evidence factors and penalize "
            "low-evidence factors. Max 4 adjustments. Return exactly this shape: "
            "{\"adjustments\":[{\"field\":\"dopant\",\"value\":\"D4\",\"direction\":\"prefer\",\"weight\":0.05,\"reason\":\"short evidence reason\"}],\"confidence\":0.7}. "
            "JSON input:\n"
            + json.dumps(prompt_payload, ensure_ascii=False)
        )
    content, response_meta = chat_completion_text(config, [{"role": "system", "content": system}, {"role": "user", "content": user}])

    parse_error = ""
    try:
        parsed = extract_json_object(content)
    except ValueError as exc:
        parsed = {"adjustments": [], "confidence": 0.0}
        parse_error = str(exc)

    parsed, schema_repair = repair_llm_response_shape(parsed)
    if "adjustments" not in parsed and {"field", "value", "direction"} <= set(parsed):
        parsed = {"adjustments": [parsed], "confidence": parsed.get("confidence", 0.5)}

    adjustments, applied_specs, confidence_calibration = normalize_llm_adjustments(
        adapter,
        pool,
        observed_ids,
        observed,
        parsed,
        exploration_mode,
    )
    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert = {
        "skills": {
            skill_id: {
                "active": bool(applied_specs),
                "model": response_meta["model"],
                "policy_variant": "exploration_aware" if exploration_mode else "legacy",
                "applied_specs": applied_specs,
                "confidence_calibration": confidence_calibration,
                "schema_repair": schema_repair,
                "parse_error": parse_error,
            }
        },
        "max_abs_adjustment": round(max_abs, 6),
    }
    record = {
        "called": True,
        "mode": mode,
        "policy_variant": "exploration_aware" if exploration_mode else "legacy",
        "seed": seed,
        "round_index": round_index,
        "model": response_meta["model"],
        "usage": response_meta["usage"],
        "prompt_payload": prompt_payload,
        "raw_response": content,
        "parsed_response": parsed,
        "decision_summary": parsed.get("decision_summary", {}),
        "applied_specs": applied_specs,
        "confidence_calibration": confidence_calibration,
        "schema_repair": schema_repair,
        "parse_error": parse_error,
    }
    return adjustments, cert, record


def update_hypothesis_from_reveal(
    h: HypothesisEntry,
    selected: Candidate,
    observed: list[Candidate],
    round_index: int,
    preferred_groups: tuple[str, ...],
) -> None:
    if selected.group not in preferred_groups:
        return
    public_mean_before = observed_mean(observed)
    supports = selected.objective_value >= public_mean_before
    h.update(
        supports=supports,
        round_index=round_index,
        evidence_summary=(
            f"Round {round_index}: {selected.candidate_id} revealed {selected.objective_value:.2f}; "
            f"public mean before reveal was {public_mean_before:.2f}; supports={supports}."
        ),
    )


def run_policy(
    adapter: DatasetAdapter,
    task: TaskSpec,
    seed: int,
    mode: Mode,
    llm_config: LLMConfig | None = None,
) -> tuple[dict[str, Any], list[AuditEntry], HypothesisEntry]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {c.candidate_id: c for c in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {c.candidate_id for c in observed}
    skills = make_skills(adapter)
    hypothesis = make_hypothesis(adapter)
    if mode == "no_care_random":
        hypothesis.status = "inactive"
        hypothesis.evidence_summary = "No CARE hypothesis or skill update is used in this random-search baseline."
    audit: list[AuditEntry] = []
    top10 = {c.candidate_id for c in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    best_trace: list[float] = []
    intervention_count = 0
    bad_interventions = 0
    rejected_good_challengers = 0
    llm_call_count = 0
    challenger_change_count = 0
    exploration_adjustment_count = 0
    penalize_adjustment_count = 0
    exploration_risk_accept_count = 0
    exploration_quota_accept_count = 0
    llm_calibrated_confidence_total = 0.0
    selected_top10 = False

    for round_index in range(task.reveal_budget):
        llm_record: dict[str, Any] | None = None
        if mode == "no_care_random":
            selected_candidate = rng.choice([c for c in pool if c.candidate_id not in observed_ids])
            gate = GateCertificate(
                gate_version="no_care",
                incumbent_candidate=selected_candidate.candidate_id,
                challenger_candidate=selected_candidate.candidate_id,
                selected_candidate=selected_candidate.candidate_id,
                authorized=False,
                gate_margin=0.0,
                acquisition_loss=0.0,
                row_order_stable=True,
                applied_skill_ids=(),
                reason="baseline_random_search_no_care",
            )
        else:
            base_scores = public_incumbent_scores(adapter, observed_ids, observed)
            if mode == "incumbent":
                incumbent = top_candidate(base_scores)
                gate = GateCertificate(
                    gate_version="none",
                    incumbent_candidate=incumbent,
                    challenger_candidate=incumbent,
                    selected_candidate=incumbent,
                    authorized=False,
                    gate_margin=0.0,
                    acquisition_loss=0.0,
                    row_order_stable=True,
                    applied_skill_ids=(),
                    reason="baseline_incumbent_only",
                )
            else:
                if is_llm_mode(mode):
                    if llm_config is None:
                        raise RuntimeError("LLM mode requested but no LLM config was provided.")
                    adjustments, skill_cert, llm_record = llm_skill_adjustments(
                        adapter,
                        pool,
                        observed_ids,
                        observed,
                        seed,
                        round_index,
                        mode,
                        llm_config,
                    )
                    llm_call_count += int(bool(llm_record.get("called")))
                    if llm_record.get("called"):
                        specs = llm_record.get("applied_specs", [])
                        exploration_adjustment_count += sum(
                            1 for spec in specs if spec.get("intent") == "explore"
                        )
                        penalize_adjustment_count += sum(
                            1 for spec in specs if spec.get("direction") == "penalize"
                        )
                        llm_calibrated_confidence_total += float(
                            llm_record.get("confidence_calibration", {}).get("calibrated", 0.0)
                        )
                    row_order_stable = True
                else:
                    adjustments, skill_cert = skill_adjustments(adapter, pool, observed_ids, observed, skills, round_index)
                    row_order_stable = row_order_stability_check(adapter, pool, observed_ids, observed, skills, round_index, adjustments)
                adjusted_scores = {cid: base_scores[cid] + adjustments.get(cid, 0.0) for cid in base_scores}
                active_skill_ids = tuple(k for k, v in skill_cert["skills"].items() if v.get("active"))
                if mode in {"no_gate", "llm_no_gate", "llm_explore_no_gate"}:
                    gate = no_gate_decision(base_scores, adjusted_scores, row_order_stable, active_skill_ids)
                elif mode == "llm_explore_gate_v1":
                    gate = exploration_gate_decision(
                        mode,
                        base_scores,
                        adjusted_scores,
                        adjustments,
                        candidate_novelty_scores(adapter, observed_ids, observed),
                        row_order_stable,
                        active_skill_ids,
                        seed,
                        round_index,
                        float(
                            (llm_record or {}).get("confidence_calibration", {}).get(
                                "calibrated",
                                0.0,
                            )
                        ),
                        intervention_count,
                        max(candidate.objective_value for candidate in observed),
                    )
                else:
                    gate = gate_decision(mode, base_scores, adjusted_scores, adjustments, row_order_stable, active_skill_ids)
                challenger_change_count += int(gate.challenger_candidate != gate.incumbent_candidate)
                exploration_risk_accept_count += int(
                    gate.reason == "authorized_exploration_risk_budget"
                )
                exploration_quota_accept_count += int(
                    gate.reason == "authorized_exploration_quota"
                )
                if gate.authorized:
                    intervention_count += 1
                    if by_id[gate.challenger_candidate].objective_value < by_id[gate.incumbent_candidate].objective_value:
                        bad_interventions += 1
                elif by_id[gate.challenger_candidate].objective_value > by_id[gate.incumbent_candidate].objective_value:
                    rejected_good_challengers += 1

        selected = by_id[gate.selected_candidate]
        if mode != "no_care_random":
            update_hypothesis_from_reveal(hypothesis, selected, observed, round_index, adapter.preferred_groups)
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(c.objective_value for c in observed)
        best_trace.append(best_so_far)
        hypothesis_snapshot = asdict(hypothesis)
        if llm_record is not None:
            hypothesis_snapshot["llm_policy"] = llm_record
        audit.append(
            AuditEntry(
                dataset_id=adapter.dataset_id,
                seed=seed,
                round_index=round_index,
                public_observed_count=len(observed) - 1,
                incumbent_candidate=gate.incumbent_candidate,
                challenger_candidate=gate.challenger_candidate,
                selected_candidate=selected.candidate_id,
                selected_by="no_care_random"
                if mode == "no_care_random"
                else "llm_no_gate_challenger"
                if mode in {"llm_no_gate", "llm_explore_no_gate"}
                else "llm_gate_authorized_challenger"
                if is_llm_mode(mode) and gate.authorized
                else "llm_gate_rejected_incumbent"
                if is_llm_mode(mode)
                else "no_gate_challenger"
                if mode == "no_gate"
                else "gate_authorized_challenger"
                if gate.authorized
                else "incumbent",
                gate=gate,
                revealed_value=selected.objective_value,
                best_so_far=best_so_far,
                hypothesis_snapshot=hypothesis_snapshot,
            )
        )
    final_best = max(c.objective_value for c in observed)
    metrics = {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10 or any(c.candidate_id in top10 for c in observed)),
        "intervention_count": intervention_count,
        "bad_intervention_count": bad_interventions,
        "rejected_good_challenger_count": rejected_good_challengers,
        "llm_call_count": llm_call_count,
        "challenger_change_count": challenger_change_count,
        "challenger_change_rate": round(challenger_change_count / max(1, task.reveal_budget), 4),
        "exploration_adjustment_count": exploration_adjustment_count,
        "penalize_adjustment_count": penalize_adjustment_count,
        "exploration_risk_accept_count": exploration_risk_accept_count,
        "exploration_quota_accept_count": exploration_quota_accept_count,
        "llm_mean_calibrated_confidence": round(
            llm_calibrated_confidence_total / max(1, llm_call_count),
            4,
        ),
        "hypothesis_confidence": round(hypothesis.confidence, 4),
        "hypothesis_support_count": hypothesis.support_count,
    }
    return metrics, audit, hypothesis


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    out: dict[str, Any] = {}
    numeric_fields = [
        "final_best",
        "best_so_far_auc",
        "simple_regret",
        "top10_hit",
        "intervention_count",
        "bad_intervention_count",
        "rejected_good_challenger_count",
        "llm_call_count",
        "challenger_change_count",
        "challenger_change_rate",
        "exploration_adjustment_count",
        "penalize_adjustment_count",
        "exploration_risk_accept_count",
        "exploration_quota_accept_count",
        "llm_mean_calibrated_confidence",
        "hypothesis_confidence",
        "hypothesis_support_count",
    ]
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in numeric_fields:
            vals = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(vals), 4),
                "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            }
    return out


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audits: dict[tuple[str, int], list[AuditEntry]],
    hypotheses: dict[tuple[str, int], HypothesisEntry],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_TABLES / f"{output_id}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for (mode, seed), audit in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as f:
            for entry in audit:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    for (mode, seed), hypothesis in sorted(hypotheses.items()):
        (OUTPUT_RUNS / f"{output_id}_knowledge_{mode}_seed{seed}.json").write_text(
            json.dumps(asdict(hypothesis), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    # Keep the original seed-0 filenames as a short compatibility handle.
    if ("gate_v2", 0) in audits:
        with (OUTPUT_RUNS / f"{output_id}_audit_seed0.jsonl").open("w", encoding="utf-8") as f:
            for entry in audits[("gate_v2", 0)]:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    if ("gate_v2", 0) in hypotheses:
        (OUTPUT_RUNS / f"{output_id}_knowledge_seed0.json").write_text(
            json.dumps(asdict(hypotheses[("gate_v2", 0)]), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_modes(raw: str) -> tuple[Mode, ...]:
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in ALL_MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Available modes: {', '.join(ALL_MODES)}")
    return modes  # type: ignore[return-value]


def llm_config_from_args(args: argparse.Namespace, modes: tuple[Mode, ...]) -> LLMConfig | None:
    if not any(is_llm_mode(mode) for mode in modes):
        return None
    api_key = (
        os.environ.get(args.llm_api_key_env)
        or os.environ.get("COMMONSTACK_API_KEY")
        or os.environ.get("CARE_LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            f"Set {args.llm_api_key_env} or COMMONSTACK_API_KEY before running LLM modes."
        )
    return LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        api_mode=args.llm_api_mode,
        structured_mode=args.llm_structured_mode,
    )


def run_dataset(
    adapter: DatasetAdapter,
    seeds: int,
    rounds: int,
    initial: int,
    modes: tuple[Mode, ...] = DEFAULT_MODES,
    llm_config: LLMConfig | None = None,
    output_tag: str = "",
    seed_start: int = 0,
) -> dict[str, Any]:
    task = make_task(adapter, initial, rounds)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[AuditEntry]] = {}
    hypotheses: dict[tuple[str, int], HypothesisEntry] = {}
    for mode in modes:
        for seed in range(seed_start, seed_start + seeds):
            metrics, audit, hypothesis = run_policy(adapter, task, seed, mode, llm_config)
            rows.append(metrics)
            audits[(mode, seed)] = audit
            hypotheses[(mode, seed)] = hypothesis
    output_id = adapter.dataset_id if not output_tag else f"{adapter.dataset_id}_{output_tag}"
    summary = {
        "experiment": "care_multi_dataset_skill_knowledge_replay",
        "disclaimer": "Finite-pool replay harness; not a CARE 1.0 paper reproduction.",
        "output_id": output_id,
        "dataset": {
            "dataset_id": adapter.dataset_id,
            "title": adapter.title,
            "group_column": adapter.group_column,
            "preferred_groups": list(adapter.preferred_groups),
        },
        "task": asdict(task),
        "candidate_count": len(adapter.candidates),
        "seeds": seeds,
        "seed_start": seed_start,
        "rounds": rounds,
        "initial_observations": initial,
        "modes": list(modes),
        "llm": None
        if llm_config is None
        else {
            "base_url": llm_config.base_url,
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "api_mode": llm_config.api_mode,
            "structured_mode": llm_config.structured_mode,
        },
        "aggregate": aggregate(rows),
    }
    write_outputs(output_id, rows, summary, audits, hypotheses)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE 2.0 finite-pool replay experiments.")
    parser.add_argument("--dataset", default="synthetic_suzuki_i", choices=[*DATASET_BUILDERS.keys(), "all"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"Comma-separated modes from: {', '.join(ALL_MODES)}")
    parser.add_argument("--llm-base-url", default=os.environ.get("CARE_LLM_BASE_URL", "https://api.commonstack.ai/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("CARE_LLM_MODEL", "moonshotai/kimi-k2.7-code"))
    parser.add_argument("--llm-api-key-env", default="CARE_LLM_API_KEY")
    parser.add_argument(
        "--llm-api-mode",
        choices=("chat", "completion"),
        default=os.environ.get("CARE_LLM_API_MODE", "chat"),
    )
    parser.add_argument(
        "--llm-structured-mode",
        choices=("tool", "json"),
        default=os.environ.get("CARE_LLM_STRUCTURED_MODE", "tool"),
    )
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=500)
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames, e.g. llm_commonstack.")
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    llm_config = llm_config_from_args(args, modes)
    dataset_ids = list(DATASET_BUILDERS) if args.dataset == "all" else [args.dataset]
    summaries = [
        run_dataset(
            DATASET_BUILDERS[dataset_id](),
            args.seeds,
            args.rounds,
            args.initial,
            modes,
            llm_config,
            args.output_tag,
            args.seed_start,
        )
        for dataset_id in dataset_ids
    ]
    if len(summaries) == 1:
        print(json.dumps(summaries[0], ensure_ascii=False, indent=2))
    else:
        combined = {
            "experiment": "care_multi_dataset_skill_knowledge_replay",
            "disclaimer": "Finite-pool replay harness; not a CARE 1.0 paper reproduction.",
            "datasets": [summary["dataset"]["dataset_id"] for summary in summaries],
            "modes": list(modes),
            "summaries": summaries,
        }
        (OUTPUT_RUNS / "all_datasets_summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
