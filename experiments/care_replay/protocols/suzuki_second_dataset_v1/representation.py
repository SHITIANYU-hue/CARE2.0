"""Approved isolated Suzuki semantic view; no mutation of legacy CSV or GP features."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
from functools import lru_cache
import sys, json, hashlib
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/care_replay/scripts"))
import run_synthetic_suzuki as replay
import run_surrogate_baselines as surrogate
import build_reaction_descriptors as descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
TARGET = "real_suzuki_miyaura"
SOURCE = "real_buchwald_hartwig"
FIELDS = ("ligand_ligand_family", "ligand_has_phosphine", "ligand_mw_bin", "ligand_logp_bin",
          "reagent_reagent_base_family", "electrophile_halide_type", "boron_partner_boron_species",
          "solvent_solvent_family", "solvent_solvent_is_protic")
ROLE_MAP = {"ligand_ligand_family":"ligand_ligand_family", "ligand_has_phosphine":"ligand_has_phosphine",
            "ligand_mw_bin":"ligand_mw_bin", "ligand_logp_bin":"ligand_logp_bin",
            "base_reagent_base_family":"reagent_reagent_base_family", "aryl_halide_halide_type":"electrophile_halide_type"}
ORIGINAL_BUILDER = replay.DATASET_BUILDERS[TARGET]
REFERENCES = Path(__file__).with_name("structure_references.json")

def structure_values(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("Invalid audited structure")
    mw, lp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    return {"canonical_smiles":Chem.MolToSmiles(mol), "mol_weight":mw, "logp":lp,
            "mw_bin":replay.numeric_bin(mw,(120.,250.,450.),("mw_none_or_low","mw_mid","mw_high","mw_very_high")),
            "logp_bin":replay.numeric_bin(lp,(0.,2.,5.),("logp_low","logp_mid","logp_high","logp_very_high"))}

@lru_cache(maxsize=1)
def corrections():
    refs = json.loads(REFERENCES.read_text())
    return {row["ligand"]:structure_values(row["supplier_smiles"]) for row in refs["approved_corrections"]}

def view(candidate):
    m = candidate.metadata
    fields = {k:m[k] for k in FIELDS if k in m}
    name = m["ligand_raw_name"]
    if name == "None":
        fields["ligand_mw_bin"] = fields["ligand_logp_bin"] = "none"
    elif name in corrections():
        fields["ligand_mw_bin"] = corrections()[name]["mw_bin"]
        fields["ligand_logp_bin"] = corrections()[name]["logp_bin"]
    elif m["ligand_rdkit_parse_ok"] != "yes":
        if name not in {"Xantphos","dppf","dtbpf"}: raise ValueError("Unapproved missing ligand")
        fields["ligand_mw_bin"] = fields["ligand_logp_bin"] = "unknown"
    flags = [m[p+"_has_boron"] == "yes" for p in ("reactant_1","reactant_2")]
    if sum(flags)!=1: raise ValueError("Exactly one boron partner required")
    boron, electrophile = ("reactant_1","reactant_2") if flags[0] else ("reactant_2","reactant_1")
    fields["electrophile_halide_type"] = m[electrophile+"_halide_type"]
    fields["boron_partner_boron_species"] = m[boron+"_boron_species"]
    if fields["electrophile_halide_type"] not in {"bromide","chloride","iodide","triflate"}: raise ValueError("Invalid electrophile")
    if fields["boron_partner_boron_species"] not in {"boronic_acid","boronic_ester","trifluoroborate"}: raise ValueError("Invalid boron species")
    if set(fields)!=set(FIELDS): raise ValueError("Unexpected field inventory")
    return {k:fields[k] for k in FIELDS}

def build_adapter(original=None):
    original = ORIGINAL_BUILDER() if original is None else original
    candidates=[]
    for c in original.candidates:
        md=dict(c.metadata)
        # Only the approved semantic projection is exposed to rules. Legacy raw decision
        # columns remain unchanged for the GP. Unused old descriptors are not exported.
        md.update(view(c))
        md.pop("reagent_functional_class",None)
        if md["ligand_raw_name"] in corrections():
            for key,value in corrections()[md["ligand_raw_name"]].items():
                if key not in ("mw_bin","logp_bin"): md["ligand_"+key]=value
        new=replace(c,metadata=md)
        if surrogate.candidate_features(original,c)!=surrogate.candidate_features(original,new): raise ValueError("GP feature changed")
        candidates.append(new)
    return replace(original,candidates=tuple(candidates))

def catalog(adapter):
    return {f:dict(sorted(Counter(view(c)[f] for c in adapter.candidates).items())) for f in FIELDS}

def field_types(cat):
    return {f:("boolean_" if f in ("ligand_has_phosphine","solvent_solvent_is_protic") else "numeric_bin_" if f in ("ligand_mw_bin","ligand_logp_bin") else "categorical_")+str(len(values)) for f,values in cat.items()}

def install():
    replay.DATASET_BUILDERS[TARGET] = build_adapter


def audit():
    original=ORIGINAL_BUILDER(); target=build_adapter(original)
    refs=json.loads(REFERENCES.read_text()); verified=[]
    for ref in refs["additional_verified"]:
        mol=Chem.MolFromSmiles(descriptors.SUZUKI_SMILES[("ligand",ref["name"])])
        if "smiles" in ref:
            assert Chem.MolToSmiles(mol,isomericSmiles=False)==Chem.MolToSmiles(Chem.MolFromSmiles(ref["smiles"]),isomericSmiles=False)
        else:
            assert rdMolDescriptors.CalcMolFormula(mol)==ref["formula"]
            assert abs(Descriptors.MolWt(mol)-ref["mw"])<0.03
        verified.append(ref["name"])
    cat=catalog(target)
    return {"candidate_count":len(target.candidates),"catalog":cat,"field_types":field_types(cat),
            "corrections":corrections(),"additional_verified_ligands":verified,
            "gp_features_unchanged":True,"model_api_calls":0,"scientific_replay_trajectories":0}
