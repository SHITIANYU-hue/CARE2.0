#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    GraphDescriptors,
    Lipinski,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "descriptors" / "reaction_component_descriptors.csv"
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)

DESCRIPTOR_FIELDS = (
    "dataset_id",
    "role",
    "local_label",
    "raw_name",
    "canonical_smiles",
    "rdkit_parse_ok",
    "mol_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_ring_count",
    "hetero_atom_count",
    "heavy_atom_count",
    "fraction_csp3",
    "formal_charge",
    "chiral_center_count",
    "bertz_complexity",
    "descriptor_mw_bin",
    "descriptor_logp_bin",
    "descriptor_tpsa_bin",
    "descriptor_hbd_bin",
    "descriptor_hba_bin",
    "descriptor_rotatable_bin",
    "descriptor_aromatic_ring_bin",
    "descriptor_fraction_csp3_bin",
    "descriptor_complexity_bin",
    "formal_charge_class",
    "ring_system_class",
    "acid_functional_class",
    "amine_functional_class",
    "amide_count_bin",
    "has_phosphorus",
    "has_phosphine",
    "has_boron",
    "has_aryl_halide",
    "has_heteroaromatic",
    "halide_type",
    "boron_species",
    "ligand_family",
    "reagent_base_family",
    "solvent_family",
    "solvent_is_protic",
    "functional_class",
    "morgan_fp_128",
    "maccs_fp",
)


SUZUKI_SMILES: dict[tuple[str, str], str] = {
    ("reactant_1", "6-chloroquinoline"): "Clc1ccc2ncccc2c1",
    ("reactant_1", "6-Bromoquinoline"): "Brc1ccc2ncccc2c1",
    ("reactant_1", "6-Iodoquinoline"): "Ic1ccc2ncccc2c1",
    ("reactant_1", "6-triflatequinoline"): "FC(F)(F)S(=O)(=O)Oc1ccc2ncccc2c1",
    ("reactant_1", "6-quinoline-boronic acid hydrochloride"): "OB(O)c1ccc2ncccc2c1",
    ("reactant_1", "6-Quinolineboronic acid pinacol ester"): "CC1(C)OB(c2ccc3ncccc3c2)OC1(C)C",
    ("reactant_1", "Potassium quinoline-6-trifluoroborate"): "[K+].F[B-](F)(F)c1ccc2ncccc2c1",
    ("reactant_2", "2a, Boronic Acid"): "OB(O)c1ccccc1",
    ("reactant_2", "2b, Boronic Ester"): "CC1(C)OB(c2ccccc2)OC1(C)C",
    ("reactant_2", "2c, Trifluoroborate"): "[K+].F[B-](F)(F)c1ccccc1",
    ("reactant_2", "2d, Bromide"): "Brc1ccccc1",
    ("catalyst", "Pd(OAc)2"): "CC(=O)[O-].CC(=O)[O-].[Pd+2]",
    ("ligand", "AmPhos"): "P(c1ccccc1)(c1ccccc1)c1ccc(N(C)C)cc1",
    ("ligand", "CataCXium A"): "P(C(C)(C)C)(C(C)(C)C)c1ccccc1",
    ("ligand", "P(Cy)3"): "P(C1CCCCC1)(C1CCCCC1)C1CCCCC1",
    ("ligand", "P(Ph)3"): "P(c1ccccc1)(c1ccccc1)c1ccccc1",
    ("ligand", "P(o-Tol)3"): "P(c1ccccc1C)(c1ccccc1C)c1ccccc1C",
    ("ligand", "P(tBu)3"): "P(C(C)(C)C)(C(C)(C)C)C(C)(C)C",
    ("ligand", "SPhos"): "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
    ("ligand", "XPhos"): "CC(C)c1cccc(C(C)C)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
    ("reagent", "CsF"): "[Cs+].[F-]",
    ("reagent", "Et3N"): "CCN(CC)CC",
    ("reagent", "K3PO4"): "[K+].[K+].[K+].[O-]P([O-])([O-])=O",
    ("reagent", "KOH"): "[K+].[OH-]",
    ("reagent", "LiOtBu"): "[Li+].CC(C)(C)[O-]",
    ("reagent", "NaHCO3"): "[Na+].O=C([O-])O",
    ("reagent", "NaOH"): "[Na+].[OH-]",
    ("solvent", "DMF"): "CN(C)C=O",
    ("solvent", "MeCN"): "CC#N",
    ("solvent", "MeOH"): "CO",
    ("solvent", "MeOH/H2O_V2 9:1"): "CO.O",
    ("solvent", "THF"): "C1CCOC1",
    ("solvent", "THF_V2"): "C1CCOC1",
}

LIGAND_FAMILIES = {
    "AmPhos": "monophosphine",
    "CataCXium A": "bulky_monophosphine",
    "P(Cy)3": "trialkylphosphine",
    "P(Ph)3": "triarylphosphine",
    "P(o-Tol)3": "triarylphosphine",
    "P(tBu)3": "trialkylphosphine",
    "SPhos": "biaryl_phosphine",
    "XPhos": "biaryl_phosphine",
    "Xantphos": "bisphosphine",
    "dppf": "ferrocenyl_bisphosphine",
    "dtbpf": "ferrocenyl_bisphosphine",
    "None": "none",
}

REAGENT_FAMILIES = {
    "CsF": "fluoride_base",
    "Et3N": "amine_base",
    "K3PO4": "phosphate_base",
    "KOH": "hydroxide_base",
    "LiOtBu": "alkoxide_base",
    "NaHCO3": "carbonate_base",
    "NaOH": "hydroxide_base",
    "None": "none",
}

SOLVENT_FAMILIES = {
    "DMF": "polar_aprotic_amide",
    "MeCN": "polar_aprotic_nitrile",
    "MeOH": "protic_alcohol",
    "MeOH/H2O_V2 9:1": "aqueous_alcohol",
    "THF": "ether",
    "THF_V2": "ether",
}

BH_BASE_FAMILIES = {
    "CC(C)(C)/N=C(N(C)C)/N(C)C": "guanidine_base",
    "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "phosphazene_base",
    "CN1CCCN2C1=NCCC2": "amidine_base",
}


def bitvect_to_text(bitvect: DataStructs.ExplicitBitVect) -> str:
    return "".join("1" if bitvect.GetBit(idx) else "0" for idx in range(bitvect.GetNumBits()))


def clean_text(value: Any) -> str:
    return str(value).strip()


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def numeric_bin(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    return replay.numeric_bin(value, edges, labels)


def smiles_for(dataset_id: str, role: str, raw_name: str) -> str:
    if raw_name.lower() == "none":
        return ""
    if dataset_id == "real_buchwald_hartwig":
        return raw_name
    if dataset_id == "real_chemlex_acidamine":
        return raw_name
    return SUZUKI_SMILES.get((role, raw_name), "")


def smarts_count(mol: Chem.Mol | None, pattern: str) -> int:
    if mol is None:
        return 0
    query = Chem.MolFromSmarts(pattern)
    return 0 if query is None else len(mol.GetSubstructMatches(query))


def ring_system_class(mol: Chem.Mol | None) -> str:
    if mol is None:
        return "unknown"
    ring_count = int(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    has_aromatic_hetero = has_heteroaromatic(mol)
    if ring_count == 0:
        return "acyclic"
    if aromatic_rings == 0:
        return "aliphatic_ring"
    if has_aromatic_hetero and aromatic_rings >= 2:
        return "fused_or_multi_heteroaromatic"
    if has_aromatic_hetero:
        return "heteroaromatic"
    if aromatic_rings >= 2:
        return "fused_or_multi_carbocyclic_aromatic"
    return "single_carbocyclic_aromatic"


def acid_functional_class(mol: Chem.Mol | None) -> str:
    acid_count = smarts_count(mol, "[CX3](=O)[OX2H1,O-]")
    if acid_count == 0:
        return "no_carboxylic_acid"
    if acid_count > 1:
        return "polycarboxylic_acid"
    if smarts_count(mol, "[NX3;!$(N[C,S,P]=O)]"):
        return "amino_acid_like"
    if smarts_count(mol, "[c][CX3](=O)[OX2H1,O-]"):
        return "aromatic_carboxylic_acid"
    if smarts_count(mol, "[n][CX3](=O)[OX2H1,O-]"):
        return "heteroaromatic_carboxylic_acid"
    return "aliphatic_carboxylic_acid"


def amine_functional_class(mol: Chem.Mol | None) -> str:
    if mol is None:
        return "unknown"
    free_nitrogens = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N"
        and not atom.GetIsAromatic()
        and not any(
            neighbor.GetSymbol() in {"C", "S", "P"}
            and any(bond.GetBondTypeAsDouble() == 2.0 for bond in neighbor.GetBonds())
            for neighbor in atom.GetNeighbors()
        )
    ]
    if not free_nitrogens:
        return "no_free_amine"
    atom = max(free_nitrogens, key=lambda item: item.GetTotalNumHs())
    aromatic_neighbor = any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors())
    ring_member = atom.IsInRing()
    hydrogens = atom.GetTotalNumHs()
    if aromatic_neighbor and hydrogens >= 1:
        return "aniline_like_amine"
    if ring_member:
        return "cyclic_amine"
    if hydrogens >= 2:
        return "primary_amine"
    if hydrogens == 1:
        return "secondary_amine"
    return "tertiary_amine"


def halide_type(raw_name: str, mol: Chem.Mol | None) -> str:
    text = raw_name.lower()
    if "triflate" in text or "otf" in text:
        return "triflate"
    if "chloro" in text or "cl-" in text:
        return "chloride"
    if "bromo" in text or "br-" in text or "bromide" in text:
        return "bromide"
    if "iodo" in text or "i-" in text:
        return "iodide"
    if mol is not None:
        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if "I" in symbols:
            return "iodide"
        if "Br" in symbols:
            return "bromide"
        if "Cl" in symbols:
            return "chloride"
    return "none"


def boron_species(raw_name: str, mol: Chem.Mol | None) -> str:
    text = raw_name.lower()
    if "trifluoroborate" in text or "bf3k" in text:
        return "trifluoroborate"
    if "pinacol" in text or "bpin" in text or "boronic ester" in text:
        return "boronic_ester"
    if "boronic acid" in text or "boh2" in text:
        return "boronic_acid"
    if mol is not None and any(atom.GetSymbol() == "B" for atom in mol.GetAtoms()):
        return "boron_species"
    return "none"


def solvent_is_protic(raw_name: str) -> str:
    return yes_no(raw_name in {"MeOH", "MeOH/H2O_V2 9:1"})


def role_functional_class(dataset_id: str, role: str, raw_name: str, mol: Chem.Mol | None) -> str:
    if clean_text(raw_name).lower() == "none":
        return "none"
    if role == "ligand":
        return ligand_family(raw_name, mol)
    if dataset_id == "real_chemlex_acidamine" and role == "acid":
        return acid_functional_class(mol)
    if dataset_id == "real_chemlex_acidamine" and role == "amine":
        return amine_functional_class(mol)
    if dataset_id == "real_chemlex_acidamine" and role == "reagent":
        return replay.chemlex_reagent_family(raw_name)
    if role in {"base", "reagent"}:
        return reagent_family(dataset_id, raw_name)
    if role in {"solvent"}:
        return SOLVENT_FAMILIES.get(raw_name, "solvent_other")
    if role == "additive":
        return "heteroaromatic_additive"
    if role in {"aryl_halide", "reactant_1", "reactant_2"}:
        boron = boron_species(raw_name, mol)
        halide = halide_type(raw_name, mol)
        if boron != "none":
            return boron
        if halide != "none":
            return f"aryl_{halide}"
        return "reactant_other"
    if role == "catalyst":
        return "palladium_catalyst"
    return "other"


def ligand_family(raw_name: str, mol: Chem.Mol | None) -> str:
    if raw_name in LIGAND_FAMILIES:
        return LIGAND_FAMILIES[raw_name]
    if mol is not None and any(atom.GetSymbol() == "P" for atom in mol.GetAtoms()):
        return "phosphine_ligand"
    return "non_phosphine_ligand"


def reagent_family(dataset_id: str, raw_name: str) -> str:
    if dataset_id == "real_buchwald_hartwig":
        return BH_BASE_FAMILIES.get(raw_name, "organic_base")
    return REAGENT_FAMILIES.get(raw_name, "reagent_other")


def has_heteroaromatic(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    return any(atom.GetIsAromatic() and atom.GetSymbol() not in {"C", "H"} for atom in mol.GetAtoms())


def descriptor_row(dataset_id: str, role: str, local_label: str, raw_name: str) -> dict[str, str]:
    smiles = smiles_for(dataset_id, role, raw_name)
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    parsed = mol is not None
    canonical = Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""
    if mol is None:
        mw = logp = tpsa = hbd = hba = rotatable = aromatic_rings = hetero_atoms = 0.0
        heavy_atoms = fraction_csp3 = formal_charge = chiral_centers = bertz = 0.0
        morgan_fp = ""
        maccs_fp = ""
    else:
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        hbd = float(Lipinski.NumHDonors(mol))
        hba = float(Lipinski.NumHAcceptors(mol))
        rotatable = float(Lipinski.NumRotatableBonds(mol))
        aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        hetero_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"}))
        heavy_atoms = float(mol.GetNumHeavyAtoms())
        fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        formal_charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        chiral_centers = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))
        bertz = float(GraphDescriptors.BertzCT(mol))
        morgan_fp = bitvect_to_text(MORGAN_GENERATOR.GetFingerprint(mol))
        maccs_fp = bitvect_to_text(MACCSkeys.GenMACCSKeys(mol))
    has_phosphorus = mol is not None and any(atom.GetSymbol() == "P" for atom in mol.GetAtoms())
    has_boron_flag = mol is not None and any(atom.GetSymbol() == "B" for atom in mol.GetAtoms())
    h_type = halide_type(raw_name, mol)
    b_species = boron_species(raw_name, mol)
    functional_class = role_functional_class(dataset_id, role, raw_name, mol)
    ligand = ligand_family(raw_name, mol) if role == "ligand" else "not_ligand"
    reagent = reagent_family(dataset_id, raw_name) if role in {"base", "reagent"} else "not_reagent"
    solvent = SOLVENT_FAMILIES.get(raw_name, "not_solvent" if role != "solvent" else "solvent_other")

    return {
        "dataset_id": dataset_id,
        "role": role,
        "local_label": local_label,
        "raw_name": raw_name,
        "canonical_smiles": canonical,
        "rdkit_parse_ok": yes_no(parsed),
        "mol_weight": f"{mw:.4f}",
        "logp": f"{logp:.4f}",
        "tpsa": f"{tpsa:.4f}",
        "hbd": f"{hbd:.0f}",
        "hba": f"{hba:.0f}",
        "rotatable_bonds": f"{rotatable:.0f}",
        "aromatic_ring_count": f"{aromatic_rings:.0f}",
        "hetero_atom_count": f"{hetero_atoms:.0f}",
        "heavy_atom_count": f"{heavy_atoms:.0f}",
        "fraction_csp3": f"{fraction_csp3:.6f}",
        "formal_charge": f"{formal_charge:.0f}",
        "chiral_center_count": f"{chiral_centers:.0f}",
        "bertz_complexity": f"{bertz:.4f}",
        "descriptor_mw_bin": numeric_bin(mw, (120.0, 250.0, 450.0), ("mw_none_or_low", "mw_mid", "mw_high", "mw_very_high")),
        "descriptor_logp_bin": numeric_bin(logp, (0.0, 2.0, 5.0), ("logp_low", "logp_mid", "logp_high", "logp_very_high")),
        "descriptor_tpsa_bin": numeric_bin(tpsa, (20.0, 60.0, 120.0), ("tpsa_low", "tpsa_mid", "tpsa_high", "tpsa_very_high")),
        "descriptor_hbd_bin": numeric_bin(hbd, (0.0, 1.0, 3.0), ("hbd_none", "hbd_one", "hbd_few", "hbd_many")),
        "descriptor_hba_bin": numeric_bin(hba, (1.0, 3.0, 7.0), ("hba_low", "hba_mid", "hba_high", "hba_very_high")),
        "descriptor_rotatable_bin": numeric_bin(rotatable, (0.0, 3.0, 8.0), ("rot_none", "rot_low", "rot_mid", "rot_high")),
        "descriptor_aromatic_ring_bin": numeric_bin(aromatic_rings, (0.0, 1.0, 3.0), ("aromatic_none", "aromatic_low", "aromatic_mid", "aromatic_high")),
        "descriptor_fraction_csp3_bin": numeric_bin(fraction_csp3, (0.1, 0.35, 0.7), ("fsp3_low", "fsp3_mid", "fsp3_high", "fsp3_very_high")),
        "descriptor_complexity_bin": numeric_bin(bertz, (150.0, 400.0, 800.0), ("complexity_low", "complexity_mid", "complexity_high", "complexity_very_high")),
        "formal_charge_class": "negative" if formal_charge < 0 else "positive" if formal_charge > 0 else "neutral",
        "ring_system_class": ring_system_class(mol),
        "acid_functional_class": acid_functional_class(mol) if role == "acid" else "not_acid",
        "amine_functional_class": amine_functional_class(mol) if role == "amine" else "not_amine",
        "amide_count_bin": numeric_bin(float(smarts_count(mol, "[NX3][CX3](=[OX1])")), (0.0, 1.0, 3.0), ("amide_none", "amide_one", "amide_few", "amide_many")),
        "has_phosphorus": yes_no(has_phosphorus),
        "has_phosphine": yes_no(ligand != "none" and "phosphine" in ligand),
        "has_boron": yes_no(has_boron_flag or b_species != "none"),
        "has_aryl_halide": yes_no(h_type in {"chloride", "bromide", "iodide", "triflate"}),
        "has_heteroaromatic": yes_no(has_heteroaromatic(mol)),
        "halide_type": h_type,
        "boron_species": b_species,
        "ligand_family": ligand,
        "reagent_base_family": reagent,
        "solvent_family": solvent,
        "solvent_is_protic": solvent_is_protic(raw_name) if role == "solvent" else "not_solvent",
        "functional_class": functional_class,
        "morgan_fp_128": morgan_fp,
        "maccs_fp": maccs_fp,
    }


def label_for(labels: dict[str, str], value: Any) -> str:
    return labels[str(value)]


def collect_components() -> list[dict[str, str]]:
    components: dict[tuple[str, str, str], dict[str, str]] = {}

    bh_records = replay.rows_to_dicts(
        replay.read_xlsx_rows(replay.ensure_public_data_file("dreher_doyle_buchwald_hartwig.xlsx"), "FullCV_01")
    )
    bh_labels = {
        "ligand": replay.label_map([r["Ligand"] for r in bh_records], "L"),
        "additive": replay.label_map([r["Additive"] for r in bh_records], "A"),
        "base": replay.label_map([r["Base"] for r in bh_records], "B"),
        "aryl_halide": replay.label_map([r["Aryl halide"] for r in bh_records], "H"),
    }
    bh_columns = {
        "ligand": "Ligand",
        "additive": "Additive",
        "base": "Base",
        "aryl_halide": "Aryl halide",
    }
    for row in bh_records:
        for role, column in bh_columns.items():
            raw_name = clean_text(row[column])
            key = ("real_buchwald_hartwig", role, raw_name)
            components[key] = descriptor_row(
                "real_buchwald_hartwig",
                role,
                label_for(bh_labels[role], row[column]),
                raw_name,
            )

    suzuki_records = replay.rows_to_dicts(
        replay.read_xlsx_rows(replay.ensure_public_data_file("perera_suzuki_miyaura.xlsx"), "Sheet1")
    )
    suzuki_label_sources = {
        "reactant_1": "Reactant_1_Short_Hand",
        "reactant_2": "Reactant_2_Name",
        "catalyst": "Catalyst_1_Short_Hand",
        "ligand": "Ligand_Short_Hand",
        "reagent": "Reagent_1_Short_Hand",
        "solvent": "Solvent_1_Short_Hand",
    }
    suzuki_raw_sources = {
        "reactant_1": "Reactant_1_Name",
        "reactant_2": "Reactant_2_Name",
        "catalyst": "Catalyst_1_Short_Hand",
        "ligand": "Ligand_Short_Hand",
        "reagent": "Reagent_1_Short_Hand",
        "solvent": "Solvent_1_Short_Hand",
    }
    suzuki_prefixes = {
        "reactant_1": "Q",
        "reactant_2": "BA",
        "catalyst": "C",
        "ligand": "L",
        "reagent": "R",
        "solvent": "S",
    }
    suzuki_labels = {
        role: replay.label_map([r[column] for r in suzuki_records], suzuki_prefixes[role])
        for role, column in suzuki_label_sources.items()
    }
    for row in suzuki_records:
        for role, raw_column in suzuki_raw_sources.items():
            raw_name = clean_text(row[raw_column])
            key = ("real_suzuki_miyaura", role, raw_name)
            components[key] = descriptor_row(
                "real_suzuki_miyaura",
                role,
                label_for(suzuki_labels[role], row[suzuki_label_sources[role]]),
                raw_name,
            )

    chemlex_records = replay.rows_to_dicts(
        replay.read_xlsx_rows(
            replay.ensure_public_data_file("chemlex_acidamine_wetlab_v3.xlsx"),
            "Sheet1",
        )
    )
    chemlex_columns = {
        "acid": ("Acid", "A"),
        "amine": ("Amine", "N"),
        "reagent": ("Reagents", "R"),
        "solvent": ("Solvent", "S"),
    }
    chemlex_labels = {
        role: replay.label_map([row[column] for row in chemlex_records], prefix)
        for role, (column, prefix) in chemlex_columns.items()
    }
    for row in chemlex_records:
        for role, (column, _prefix) in chemlex_columns.items():
            raw_name = clean_text(row[column])
            key = ("real_chemlex_acidamine", role, raw_name)
            components[key] = descriptor_row(
                "real_chemlex_acidamine",
                role,
                label_for(chemlex_labels[role], row[column]),
                raw_name,
            )

    return sorted(components.values(), key=lambda item: (item["dataset_id"], item["role"], item["local_label"], item["raw_name"]))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_components()
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DESCRIPTOR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    parsed = sum(1 for row in rows if row["rdkit_parse_ok"] == "yes")
    print(
        f"wrote {len(rows)} reaction component descriptors to {OUTPUT_PATH} "
        f"({parsed} parsed by RDKit, {len(rows) - parsed} semantic-only)"
    )


if __name__ == "__main__":
    main()
