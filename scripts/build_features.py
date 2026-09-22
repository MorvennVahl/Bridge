"""Build ingredient-level and target-level feature tables for the Bridge drug side.

Grain: one row per OMOP RxNorm Ingredient concept (4,280 rows = 4,276 CEM ingredients
plus the 4 indication-roster exposures absent from the CEM list).

Every feature column is registered in data_dictionary.csv with its source, the exact
endpoint it came from, the source release and the retrieval timestamp. Columns whose
values are label-adjacent (derived from the same FAERS lineage as the CEM labels) are
written to a SEPARATE file so they cannot be used as predictors by accident.

Outputs: target_features.csv, ingredient_target_long.csv, ingredient_features.csv,
         ingredient_features_label_adjacent.csv, data_dictionary.csv, provenance.json
"""
import json, re, os
import numpy as np
import pandas as pd

H = "handoff"
PROP = ["full_mwt", "alogp", "hba", "hbd", "psa", "rtb", "aromatic_rings", "heavy_atoms",
        "num_ro5_violations", "cx_logp", "cx_logd", "cx_most_apka", "cx_most_bpka",
        "qed_weighted", "molecular_species", "hbd_lipinski", "hba_lipinski", "np_likeness_score"]
AVAIL = {-1: "unknown", 0: "discontinued", 1: "prescription_only", 2: "otc", 3: "withdrawn"}
L1 = ["enzyme", "membrane receptor", "ion channel", "transporter", "transcription factor",
      "secreted protein", "surface antigen", "structural protein", "other cytosolic protein",
      "other nuclear protein", "unclassified protein"]

# ------------------------------------------------------------------ inputs ----
ing = pd.read_csv(f"{H}/ingredient_chembl_map.csv")
ing["omop_concept_id"] = ing.omop_concept_id.astype("Int64")
# surrogate join key: the two non-drug exposures have no OMOP ingredient concept, and a
# NaN join key would cross-product against every other NaN row.
ing["_jk"] = ing.omop_concept_id.astype("float")
_m = ing._jk.isna()
ing.loc[_m, "_jk"] = -np.arange(1, int(_m.sum()) + 1, dtype=float)
mols = {m["molecule_chembl_id"]: m for m in
        json.load(open(f"{H}/chembl_molecule_phase.json")) + json.load(open(f"{H}/chembl_molecule_therap.json"))}
mech = pd.read_csv(f"{H}/ingredient_mechanism.csv")
warn = pd.read_csv(f"{H}/ingredient_warning.csv")
metab = pd.read_csv(f"{H}/ingredient_metabolism.csv")
trecs = json.load(open(f"{H}/targets_chembl.json"))
comps = {c["component_id"]: c for c in json.load(open(f"{H}/target_components.json"))}
pcs = {p["protein_class_id"]: p for p in json.load(open(f"{H}/protein_classification.json"))}
sym2ensg = json.load(open(f"{H}/targets_sym2ensg.json"))
ot = json.load(open(f"{H}/targets_opentargets.json"))
kegg_match = pd.read_csv(f"{H}/kegg_name_match.csv")
kegg_entries = json.load(open(f"{H}/kegg_entries.json"))
degree = pd.read_csv(f"{H}/ingredient_cem_degree.csv")
degree["omop_concept_id"] = degree.omop_concept_id.astype("Int64")

up = pd.read_csv(f"{H}/targets_uniprot.tsv", sep="\t")
up.columns = ["accession", "uniprot_entry", "protein_name", "gene_symbol", "aa_length",
              "uniprot_keywords", "uniprot_subcellular_location", "uniprot_function"]

prov = {"generated_by": "build_features.py", "sources": {}}
for f in ["provenance_chembl", "provenance_targets", "provenance_protein_class",
          "provenance_kegg", "provenance_openfda"]:
    p = f"{H}/{f}.json"
    if os.path.exists(p):
        prov["sources"][f.replace("provenance_", "")] = json.load(open(p))

# ------------------------------------------------------- target feature table --
def class_path(cid):
    paths = []
    for pc in (comps.get(cid, {}).get("protein_classifications") or []):
        rec = pcs.get(pc["protein_classification_id"], {})
        if rec.get("protein_class_desc"):
            paths.append(rec["protein_class_desc"])
    return max(paths, key=len) if paths else None


trows = []
for t in trecs:
    for c in (t.get("target_components") or []):
        cp = class_path(c["component_id"])
        trows.append(dict(target_chembl_id=t["target_chembl_id"], target_pref_name=t.get("pref_name"),
                          target_type=t.get("target_type"), target_organism=t.get("organism"),
                          component_id=c["component_id"], accession=c.get("accession"),
                          component_description=c.get("component_description"),
                          component_relationship=c.get("relationship"),
                          chembl_protein_class_path=cp,
                          chembl_protein_class_L1=cp.split("  ")[0] if cp else None,
                          chembl_protein_class_leaf=cp.split("  ")[-1] if cp else None))
tgt = pd.DataFrame(trows).merge(up, on="accession", how="left")
tgt["ensembl_gene_id"] = tgt.gene_symbol.map(sym2ensg)


def ot_feats(g):
    t = ot.get(g)
    if not t:
        return {}
    d = {"ot_approved_symbol": t.get("approvedSymbol"), "ot_biotype": t.get("biotype"),
         "ot_is_essential": t.get("isEssential")}
    for x in (t.get("tractability") or []):
        d[f"tract_{x['modality']}_{re.sub('[^a-z0-9]+','_',x['label'].lower())}"] = bool(x["value"])
    for c in (t.get("geneticConstraint") or []):
        d[f"constraint_{c['constraintType']}_score"] = c.get("score")
        d[f"constraint_{c['constraintType']}_upperBin"] = c.get("upperBin")
    sl = t.get("safetyLiabilities") or []
    d["n_safety_liabilities"] = len(sl)
    d["safety_events"] = "|".join(sorted({s["event"] for s in sl if s.get("event")}))
    d["safety_datasources"] = "|".join(sorted({s["datasource"] for s in sl if s.get("datasource")}))
    pw = t.get("pathways") or []
    d["n_reactome_pathways"] = len(pw)
    d["reactome_top_level_terms"] = "|".join(sorted({p["topLevelTerm"] for p in pw if p.get("topLevelTerm")}))
    return d


tgt = pd.concat([tgt, pd.DataFrame([ot_feats(g) for g in tgt.ensembl_gene_id], index=tgt.index)], axis=1)
tgt.to_csv("target_features.csv", index=False)

# --------------------------------------------------- ingredient x target long --
gene_cols = ["target_chembl_id", "target_pref_name", "target_type", "accession", "gene_symbol",
             "ensembl_gene_id", "component_relationship", "chembl_protein_class_L1",
             "chembl_protein_class_leaf"]
long = (mech[["omop_concept_id", "ingredient_name", "chembl_id", "target_chembl_id", "action_type",
              "mechanism_of_action", "direct_interaction", "molecular_mechanism", "disease_efficacy"]]
        .merge(tgt[gene_cols], on="target_chembl_id", how="left"))
long["mechanism_source"] = "ChEMBL drug_mechanism"
long.to_csv("ingredient_target_long.csv", index=False)

# ChEMBL records a mechanism string for some drugs without assigning a target (e.g. "Unknown").
# Keep that distinction: it separates "no mechanism record" from "mechanism known, target not".
untarg = mech.groupby("omop_concept_id").agg(
    n_mechanism_rows_any=("mechanism_of_action", "size"),
    n_mechanism_rows_without_target=("target_chembl_id", lambda s: int(s.isna().sum()))).reset_index()
untarg["mechanism_recorded_but_target_unassigned"] = (
    untarg.n_mechanism_rows_without_target == untarg.n_mechanism_rows_any)
untarg = untarg.rename(columns={"omop_concept_id": "_jk"})
untarg["_jk"] = untarg._jk.astype(float)

# --------------------------------------------------------- KEGG entry parsing --
GENE = re.compile(r"([A-Z0-9][A-Z0-9\-]{1,12})\s*\[HSA:")


def kegg_feats(did):
    e = kegg_entries.get(did)
    if not e:
        return {}
    d = {}
    met = " ; ".join(e.get("METABOLISM", []))
    inter = " ; ".join(e.get("INTERACTION", []))
    enz = re.findall(r"Enzyme:(.*?)(?:Transporter:|$)", met, re.S)
    trn = re.findall(r"Transporter:(.*?)$", met, re.S)
    d["kegg_metab_enzymes"] = "|".join(sorted(set(GENE.findall(enz[0])))) if enz else ""
    d["kegg_transporters"] = "|".join(sorted(set(GENE.findall(trn[0])))) if trn else ""
    d["kegg_interaction_genes"] = "|".join(sorted(set(GENE.findall(inter))))
    d["kegg_n_metab_enzymes"] = len([x for x in d["kegg_metab_enzymes"].split("|") if x])
    d["kegg_n_transporters"] = len([x for x in d["kegg_transporters"].split("|") if x])
    d["kegg_n_interaction_genes"] = len([x for x in d["kegg_interaction_genes"].split("|") if x])
    d["kegg_cyp_inhibition"] = bool(re.search(r"CYP inhibition", inter))
    d["kegg_cyp_induction"] = bool(re.search(r"CYP induction|Enzyme induction", inter))
    d["kegg_transporter_inhibition"] = bool(re.search(r"Transporter inhibition", inter))
    d["kegg_efficacy"] = " ".join(e.get("EFFICACY", []))[:300]
    for cyp in ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2B6", "CYP2E1"]:
        d[f"kegg_substrate_{cyp}"] = cyp in d["kegg_metab_enzymes"].split("|")
    return d


kegg_match["kegg_drug_id"] = kegg_match.kegg_drug_id.astype("object")
kf = pd.DataFrame([kegg_feats(d) if isinstance(d, str) else {} for d in kegg_match.kegg_drug_id],
                  index=kegg_match.index)
kegg_blk = pd.concat([kegg_match[["ingredient_name", "kegg_drug_id", "kegg_n_candidates"]], kf], axis=1)

# ------------------------------------------------------- molecule-level block --
FLAT = ["max_phase", "first_approval", "therapeutic_flag", "availability_type", "dosed_ingredient",
        "withdrawn_flag", "black_box_warning", "prodrug", "natural_product", "molecule_type",
        "chirality", "inorganic_flag", "polymer_flag", "oral", "parenteral", "topical",
        "indication_class", "usan_stem_definition"]
mrows = []
for _, r in ing.iterrows():
    d = {"_jk": r._jk}
    if not isinstance(r.chembl_id, str):
        mrows.append(d); continue
    m, p = mols.get(r.chembl_id, {}), mols.get(r.chembl_parent_id, {})
    g = lambda k: (p.get(k) if (m.get(k) in (None, "")) and p else m.get(k))
    for k in FLAT:
        d[k] = g(k)
    d["chembl_pref_name"] = g("pref_name")
    d["availability_label"] = AVAIL.get(d["availability_type"])
    atc = m.get("atc_classifications") or p.get("atc_classifications") or []
    d["atc_codes"] = "|".join(atc); d["n_atc_codes"] = len(atc)
    d["atc_l1"] = "|".join(sorted({a[0] for a in atc})); d["n_atc_l1_groups"] = len({a[0] for a in atc})
    d["atc_l3"] = "|".join(sorted({a[:4] for a in atc}))
    props = m.get("molecule_properties") or p.get("molecule_properties") or {}
    for k in PROP:
        d[k] = props.get(k)
    mrows.append(d)
molblk = pd.DataFrame(mrows)
for c in ["max_phase"] + [k for k in PROP if k != "molecular_species"]:
    molblk[c] = pd.to_numeric(molblk[c], errors="coerce")
molblk["ro5_like"] = ((molblk.full_mwt <= 500) & (molblk.alogp <= 5) &
                      (molblk.hbd <= 5) & (molblk.hba <= 10))
molblk["n_routes_labelled"] = molblk[["oral", "parenteral", "topical"]].sum(axis=1, min_count=1)

# ------------------------------------------------------------ warnings block --
wag = warn.groupby("omop_concept_id").agg(
    n_drug_warnings=("warning_id", "nunique"),
    warning_types=("warning_type", lambda s: "|".join(sorted(set(s.dropna())))),
    warning_classes=("warning_class", lambda s: "|".join(sorted(set(s.dropna())))),
    warning_year_min=("warning_year", "min")).reset_index().rename(columns={"omop_concept_id": "_jk"})
wag["_jk"] = wag._jk.astype(float)

mtag = metab.groupby("omop_concept_id").agg(
    chembl_n_metabolism_rows=("met_id", "nunique"),
    chembl_metab_enzymes=("enzyme_name", lambda s: "|".join(sorted(set(s.dropna())))),
    chembl_metabolite_names=("metabolite_name", lambda s: "|".join(sorted(set(s.dropna()))[:12]))
    ).reset_index().rename(columns={"omop_concept_id": "_jk"})
mtag["_jk"] = mtag._jk.astype(float)

# --------------------------------------------- mechanism / target aggregation --
gf = tgt.dropna(subset=["ensembl_gene_id"]).drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")
# use the level-1 class labels ChEMBL actually emits, not a hard-coded guess
L1 = sorted(tgt.chembl_protein_class_L1.dropna().unique())
TR = [c for c in tgt.columns if c.startswith("tract_")]
agg_rows = []
for cid, d in long.groupby("omop_concept_id"):
    genes = d.ensembl_gene_id.dropna().unique().tolist()
    sub = gf.reindex(genes)
    r = {"omop_concept_id": cid,
         "n_mechanism_rows": len(d), "n_targets_chembl": d.target_chembl_id.nunique(),
         "n_target_genes": len(genes),
         "target_chembl_ids": "|".join(sorted(d.target_chembl_id.dropna().unique())),
         "target_genes": "|".join(sorted(genes)),
         "target_gene_symbols": "|".join(sorted(d.gene_symbol.dropna().unique())),
         "action_types": "|".join(sorted(d.action_type.dropna().unique())),
         "mechanisms_of_action": "|".join(sorted(d.mechanism_of_action.dropna().unique()))[:500],
         "has_family_or_complex_target": bool((d.target_type.dropna() != "SINGLE PROTEIN").any()),
         "frac_direct_interaction": d.direct_interaction.mean(),
         "dominant_target_class": (sub.chembl_protein_class_L1.mode().iat[0]
                                   if len(sub) and sub.chembl_protein_class_L1.notna().any() else None)}
    for c in L1:
        r[f"n_genes_{re.sub('[^a-z]+','_',c)}"] = int((sub.chembl_protein_class_L1 == c).sum())
    if len(sub):
        r.update({
            "safety_liabilities_sum": sub.n_safety_liabilities.sum(),
            "safety_liabilities_max": sub.n_safety_liabilities.max(),
            "n_genes_with_safety_liability": int((sub.n_safety_liabilities > 0).sum()),
            "target_safety_events": "|".join(sorted({e for s in sub.safety_events.dropna()
                                                     for e in s.split("|") if e}))[:500],
            "constraint_lof_upperBin_min": sub.get("constraint_lof_upperBin", pd.Series(dtype=float)).min(),
            "constraint_lof_upperBin_mean": sub.get("constraint_lof_upperBin", pd.Series(dtype=float)).mean(),
            "constraint_mis_upperBin_mean": sub.get("constraint_mis_upperBin", pd.Series(dtype=float)).mean(),
            "n_essential_genes": int(sub.ot_is_essential.fillna(False).sum()),
            "n_reactome_pathways_union": int(sub.n_reactome_pathways.fillna(0).sum()),
            "reactome_top_level_terms": "|".join(sorted({t for s in sub.reactome_top_level_terms.dropna()
                                                         for t in s.split("|") if t}))[:500]})
        for c in TR:
            if c in sub:
                r[f"n_genes_{c}"] = int(sub[c].fillna(False).sum())
    agg_rows.append(r)
mechblk = pd.DataFrame(agg_rows).rename(columns={"omop_concept_id": "_jk"})
mechblk["_jk"] = mechblk._jk.astype(float)

# ------------------------------------------------------------- openFDA block --
ofda = {}
for nm in ["label_any", "label_boxed_warning", "label_pregnancy", "label_drug_interactions"]:
    p = f"{H}/openfda_{nm}.json"
    if os.path.exists(p):
        ofda[nm] = {d["term"].lower(): d["count"] for d in json.load(open(p))}
lab = ing[["_jk", "ingredient_name"]].copy()
lab["_k"] = lab.ingredient_name.str.lower()
for nm, v in ofda.items():
    lab[f"openfda_{nm}_n_labels"] = lab._k.map(v)
lab["openfda_has_boxed_warning_label"] = lab.openfda_label_boxed_warning_n_labels.notna()
lab = lab.drop(columns=["_k", "ingredient_name"])

# --------------------------------------------------------------- final merge --
feat = (ing[["_jk", "omop_concept_id", "ingredient_name", "roster_name", "indication", "source_class",
             "in_indication_roster", "in_cem_list", "exposure_type", "chembl_id", "chembl_parent_id",
             "chembl_match_method", "chembl_match_n_candidates", "chembl_fuzzy_candidate_id",
             "chembl_fuzzy_candidate_name", "chembl_fuzzy_ratio"]]
        .merge(molblk, on="_jk", how="left")
        .merge(wag, on="_jk", how="left")
        .merge(mtag, on="_jk", how="left")
        .merge(mechblk, on="_jk", how="left")
        .merge(untarg, on="_jk", how="left")
        .merge(kegg_blk, on="ingredient_name", how="left")
        .merge(lab, on="_jk", how="left")
        .drop(columns=["_jk"]))
assert len(feat) == len(ing), f"row count changed: {len(feat)} vs {len(ing)}"
for c in ["n_drug_warnings", "chembl_n_metabolism_rows", "n_mechanism_rows", "n_targets_chembl",
          "n_target_genes"] + [c for c in feat.columns if c.startswith("n_genes_")]:
    if c in feat:
        feat[c] = feat[c].fillna(0).astype("Int64")
feat["has_chembl_mechanism"] = feat.n_targets_chembl.fillna(0) > 0
feat.to_csv("ingredient_features.csv", index=False)

lab_adj = ing[["omop_concept_id", "ingredient_name"]].merge(degree, on="omop_concept_id", how="left")
lab_adj.to_csv("ingredient_features_label_adjacent.csv", index=False)

# --------------------------------------------------------- data dictionary ----
CH_REL = prov["sources"].get("chembl", {}).get("release", "ChEMBL_37")
CH_DATE = prov["sources"].get("chembl", {}).get("release_date", "2026-05-01")
SRC = {
 "OMOP":    ("OMOP/RxNorm via CEM", "cem_ingredients.csv (local Postgres cem, cem_output.cem_unified)", ""),
 "ROSTER":  ("User-supplied indication roster", "conversation, 2026-09-22", ""),
 "CHEMBL_MOL": ("ChEMBL", f"{CH_REL} ({CH_DATE})", "https://www.ebi.ac.uk/chembl/api/data/molecule.json"),
 "CHEMBL_MECH": ("ChEMBL", f"{CH_REL} ({CH_DATE})", "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"),
 "CHEMBL_WARN": ("ChEMBL", f"{CH_REL} ({CH_DATE})", "https://www.ebi.ac.uk/chembl/api/data/drug_warning.json"),
 "CHEMBL_METAB": ("ChEMBL", f"{CH_REL} ({CH_DATE})", "https://www.ebi.ac.uk/chembl/api/data/metabolism.json"),
 "CHEMBL_TGT": ("ChEMBL", f"{CH_REL} ({CH_DATE})", "https://www.ebi.ac.uk/chembl/api/data/target.json"),
 "CHEMBL_PC": ("ChEMBL", f"{CH_REL} ({CH_DATE})", "https://www.ebi.ac.uk/chembl/api/data/target_component.json + protein_classification.json"),
 "UNIPROT": ("UniProtKB", "current release at retrieval", "https://rest.uniprot.org/uniprotkb/stream"),
 "ENSEMBL": ("Ensembl", "current release at retrieval", "https://rest.ensembl.org/lookup/symbol/homo_sapiens"),
 "OT":      ("Open Targets Platform", "current release at retrieval", "https://api.platform.opentargets.org/api/v4/graphql"),
 "KEGG":    ("KEGG DRUG", "current release at retrieval", "https://rest.kegg.jp/list/drug + /get/dr:<id>"),
 "OPENFDA": ("openFDA", "see provenance.json last_updated", "https://api.fda.gov/drug/label.json (count aggregation)"),
 "DERIVED": ("Derived in build_features.py", "-", "-"),
 "CEM":     ("OHDSI Common Evidence Model", "local Postgres cem", "cem_ingredient_condition_associations.csv"),
}
DESC = {
 "omop_concept_id": ("identity", "OMOP RxNorm Ingredient concept id", "OMOP"),
 "ingredient_name": ("identity", "OMOP concept name for the ingredient", "OMOP"),
 "roster_name": ("identity", "Name as given in the user's indication roster, where applicable", "ROSTER"),
 "indication": ("identity", "Indication block the drug was listed under in the user's roster", "ROSTER"),
 "source_class": ("identity", "Pharmacologic class as given in the user's roster CSV", "ROSTER"),
 "in_indication_roster": ("identity", "Present in the user's 97-exposure indication roster", "DERIVED"),
 "in_cem_list": ("identity", "Present in cem_ingredients.csv", "DERIVED"),
 "exposure_type": ("identity", "drug vs non-drug exposure (ECT, psychotherapy)", "ROSTER"),
 "chembl_id": ("identity", "ChEMBL molecule matched by exact normalized name or qualifier-stripped name", "CHEMBL_MOL"),
 "chembl_parent_id": ("identity", "Parent molecule from ChEMBL molecule_hierarchy (salts resolved to parent)", "CHEMBL_MOL"),
 "chembl_match_method": ("identity", "How the name was resolved; no fuzzy matches are used here", "DERIVED"),
 "chembl_match_n_candidates": ("identity", "Number of distinct ChEMBL molecules sharing the matched name key", "DERIVED"),
 "chembl_fuzzy_candidate_id": ("identity", "REVIEW ONLY. Nearest fuzzy name match; NOT used by any feature (high false-positive rate)", "DERIVED"),
 "chembl_fuzzy_candidate_name": ("identity", "REVIEW ONLY. Preferred name of the fuzzy candidate", "DERIVED"),
 "chembl_fuzzy_ratio": ("identity", "REVIEW ONLY. difflib ratio of the fuzzy match", "DERIVED"),
 "chembl_pref_name": ("identity", "ChEMBL preferred name of the matched molecule", "CHEMBL_MOL"),
 "kegg_drug_id": ("identity", "KEGG DRUG entry matched by exact normalized name", "KEGG"),
 "kegg_n_candidates": ("identity", "Number of KEGG entries sharing the matched name key", "DERIVED"),
 "max_phase": ("development", "Highest development phase reached (4 = approved; -1 = unknown in ChEMBL)", "CHEMBL_MOL"),
 "first_approval": ("development", "Year of first regulatory approval", "CHEMBL_MOL"),
 "therapeutic_flag": ("development", "Flagged by ChEMBL as a therapeutic (vs diagnostic/reagent)", "CHEMBL_MOL"),
 "availability_type": ("development", "Raw ChEMBL availability code", "CHEMBL_MOL"),
 "availability_label": ("development", "Availability decoded: prescription_only / otc / discontinued / withdrawn / unknown", "DERIVED"),
 "dosed_ingredient": ("development", "Appears as a dosed ingredient in a marketed product", "CHEMBL_MOL"),
 "withdrawn_flag": ("safety", "Withdrawn from market in at least one jurisdiction", "CHEMBL_MOL"),
 "black_box_warning": ("safety", "ChEMBL black-box warning flag on the molecule", "CHEMBL_MOL"),
 "n_drug_warnings": ("safety", "Count of ChEMBL drug_warning records (boxed warnings / withdrawals)", "CHEMBL_WARN"),
 "warning_types": ("safety", "Warning types, pipe-delimited", "CHEMBL_WARN"),
 "warning_classes": ("safety", "Toxicity classes (cardiotoxicity, teratogenicity, ...), pipe-delimited", "CHEMBL_WARN"),
 "warning_year_min": ("safety", "Earliest warning year on record", "CHEMBL_WARN"),
 "prodrug": ("pharmacology", "ChEMBL prodrug flag; requires metabolic activation", "CHEMBL_MOL"),
 "natural_product": ("chemistry", "ChEMBL natural-product flag", "CHEMBL_MOL"),
 "molecule_type": ("chemistry", "Small molecule / Protein / Oligosaccharide / ...", "CHEMBL_MOL"),
 "chirality": ("chemistry", "ChEMBL chirality code", "CHEMBL_MOL"),
 "inorganic_flag": ("chemistry", "ChEMBL inorganic flag", "CHEMBL_MOL"),
 "polymer_flag": ("chemistry", "ChEMBL polymer flag", "CHEMBL_MOL"),
 "oral": ("exposure", "Labelled for oral administration", "CHEMBL_MOL"),
 "parenteral": ("exposure", "Labelled for parenteral administration", "CHEMBL_MOL"),
 "topical": ("exposure", "Labelled for topical administration", "CHEMBL_MOL"),
 "n_routes_labelled": ("exposure", "Count of oral/parenteral/topical route flags set", "DERIVED"),
 "indication_class": ("pharmacology", "ChEMBL free-text indication class", "CHEMBL_MOL"),
 "usan_stem_definition": ("pharmacology", "USAN stem definition (encodes mechanism class from the name)", "CHEMBL_MOL"),
 "atc_codes": ("pharmacology", "All WHO ATC codes, pipe-delimited", "CHEMBL_MOL"),
 "n_atc_codes": ("pharmacology", "Number of ATC codes", "DERIVED"),
 "atc_l1": ("pharmacology", "ATC level-1 anatomical main groups present", "DERIVED"),
 "n_atc_l1_groups": ("pharmacology", "Number of distinct ATC level-1 groups (breadth of clinical use)", "DERIVED"),
 "atc_l3": ("pharmacology", "ATC level-4 code prefixes (pharmacological subgroups)", "DERIVED"),
 "ro5_like": ("chemistry", "Derived Lipinski rule-of-five compliance (MW<=500, alogp<=5, HBD<=5, HBA<=10)", "DERIVED"),
 "n_mechanism_rows": ("mechanism", "ChEMBL mechanism rows with an assigned target", "CHEMBL_MECH"),
 "n_mechanism_rows_any": ("mechanism", "ChEMBL mechanism rows including those with no target assigned", "CHEMBL_MECH"),
 "n_mechanism_rows_without_target": ("mechanism", "Mechanism rows where ChEMBL records no target", "CHEMBL_MECH"),
 "mechanism_recorded_but_target_unassigned": ("mechanism", "Mechanism is documented but no target is assigned (e.g. hydralazine)", "DERIVED"),
 "n_targets_chembl": ("mechanism", "Distinct ChEMBL targets (may be complexes or families)", "CHEMBL_MECH"),
 "n_target_genes": ("mechanism", "Distinct human genes behind those targets", "DERIVED"),
 "target_chembl_ids": ("mechanism", "ChEMBL target ids, pipe-delimited", "CHEMBL_MECH"),
 "target_genes": ("mechanism", "Ensembl gene ids of targets, pipe-delimited", "DERIVED"),
 "target_gene_symbols": ("mechanism", "HGNC symbols of targets, pipe-delimited", "UNIPROT"),
 "action_types": ("mechanism", "Action types (INHIBITOR/AGONIST/BLOCKER/...), pipe-delimited", "CHEMBL_MECH"),
 "mechanisms_of_action": ("mechanism", "ChEMBL mechanism-of-action strings, pipe-delimited (truncated at 500 chars)", "CHEMBL_MECH"),
 "has_family_or_complex_target": ("mechanism", "At least one target is a protein family or complex, not a single protein", "DERIVED"),
 "frac_direct_interaction": ("mechanism", "Fraction of mechanism rows flagged as direct target interaction", "CHEMBL_MECH"),
 "dominant_target_class": ("mechanism", "Modal ChEMBL level-1 protein class across target genes", "DERIVED"),
 "has_chembl_mechanism": ("mechanism", "At least one mechanism row with an assigned target", "DERIVED"),
 "safety_liabilities_sum": ("target biology", "Sum of Open Targets curated safety liabilities across target genes", "OT"),
 "safety_liabilities_max": ("target biology", "Max safety liabilities on any single target gene", "OT"),
 "n_genes_with_safety_liability": ("target biology", "Count of target genes with >=1 curated safety liability", "OT"),
 "target_safety_events": ("target biology", "Union of safety-liability event terms across target genes (truncated)", "OT"),
 "constraint_lof_upperBin_min": ("target biology", "Most constrained LoF decile bin across target genes (lower = more constrained)", "OT"),
 "constraint_lof_upperBin_mean": ("target biology", "Mean LoF constraint decile bin across target genes", "OT"),
 "constraint_mis_upperBin_mean": ("target biology", "Mean missense constraint decile bin across target genes", "OT"),
 "n_essential_genes": ("target biology", "Count of target genes flagged essential in DepMap via Open Targets", "OT"),
 "n_reactome_pathways_union": ("target biology", "Summed Reactome pathway memberships across target genes", "OT"),
 "reactome_top_level_terms": ("target biology", "Union of Reactome top-level pathway terms for target genes (truncated)", "OT"),
 "chembl_n_metabolism_rows": ("metabolism", "ChEMBL metabolism records for the drug or its parent", "CHEMBL_METAB"),
 "chembl_metab_enzymes": ("metabolism", "Enzymes named in ChEMBL metabolism records, pipe-delimited", "CHEMBL_METAB"),
 "chembl_metabolite_names": ("metabolism", "Named metabolites (first 12), pipe-delimited", "CHEMBL_METAB"),
 "kegg_metab_enzymes": ("metabolism", "Metabolising enzymes from the KEGG DRUG METABOLISM field", "KEGG"),
 "kegg_transporters": ("metabolism", "Transporters from the KEGG DRUG METABOLISM field", "KEGG"),
 "kegg_interaction_genes": ("interactions", "Genes named in the KEGG DRUG INTERACTION field (DDI partners)", "KEGG"),
 "kegg_n_metab_enzymes": ("metabolism", "Count of KEGG metabolising enzymes", "DERIVED"),
 "kegg_n_transporters": ("metabolism", "Count of KEGG transporters", "DERIVED"),
 "kegg_n_interaction_genes": ("interactions", "Count of KEGG interaction genes", "DERIVED"),
 "kegg_cyp_inhibition": ("interactions", "KEGG INTERACTION field records CYP inhibition", "KEGG"),
 "kegg_cyp_induction": ("interactions", "KEGG INTERACTION field records CYP or enzyme induction", "KEGG"),
 "kegg_transporter_inhibition": ("interactions", "KEGG INTERACTION field records transporter inhibition", "KEGG"),
 "kegg_efficacy": ("pharmacology", "KEGG EFFICACY string (truncated at 300 chars)", "KEGG"),
 "openfda_has_boxed_warning_label": ("safety", "Ingredient appears among generic names with a boxed_warning SPL section", "OPENFDA"),
}
PATTERNS = [
 (re.compile(r"^n_genes_tract_"), ("target biology", "Count of target genes with this Open Targets tractability bucket true", "DERIVED")),
 (re.compile(r"^n_genes_"), ("mechanism", "Count of target genes in this ChEMBL level-1 protein class", "DERIVED")),
 (re.compile(r"^kegg_substrate_CYP"), ("metabolism", "KEGG METABOLISM names this CYP as a metabolising enzyme", "KEGG")),
 (re.compile(r"^openfda_label_.*_n_labels$"), ("safety", "Number of SPL label documents for this generic name with the section present (top-1000 aggregation only)", "OPENFDA")),
 (re.compile(r"^tract_"), ("target biology", "Open Targets tractability bucket", "OT")),
 (re.compile(r"^constraint_"), ("target biology", "Open Targets gnomAD genetic-constraint statistic", "OT")),
 (re.compile(r"^ot_"), ("target biology", "Open Targets target attribute", "OT")),
 (re.compile(r"^uniprot_|^protein_name$|^gene_symbol$|^aa_length$|^accession$"), ("target biology", "UniProtKB annotation", "UNIPROT")),
 (re.compile(r"^ensembl_gene_id$"), ("target biology", "Ensembl gene id resolved from HGNC symbol", "ENSEMBL")),
 (re.compile(r"^target_|^component_|^chembl_protein_class"), ("target biology", "ChEMBL target / component definition", "CHEMBL_TGT")),
 (re.compile(r"^n_safety_liabilities$|^safety_events$|^safety_datasources$|^n_reactome_pathways$"), ("target biology", "Open Targets target attribute", "OT")),
 (re.compile(r"^cem_"), ("LABEL-ADJACENT", "Derived from the CEM association file itself (same FAERS lineage as the labels) - nuisance/degree term, not a biology predictor", "CEM")),
 (re.compile("|".join(f"^{re.escape(p)}$" for p in PROP)), ("chemistry", "ChEMBL computed physicochemical property", "CHEMBL_MOL")),
 (re.compile(r"^(action_type|mechanism_of_action|direct_interaction|molecular_mechanism|disease_efficacy|mechanism_source)$"), ("mechanism", "ChEMBL mechanism record field", "CHEMBL_MECH")),
]
rows = []
for table, df in [("ingredient_features.csv", feat), ("target_features.csv", tgt),
                  ("ingredient_target_long.csv", long),
                  ("ingredient_features_label_adjacent.csv", lab_adj)]:
    for c in df.columns:
        spec = DESC.get(c)
        if spec is None:
            for pat, s in PATTERNS:
                if pat.match(c):
                    spec = s
                    break
        if spec is None:
            raise SystemExit(f"data dictionary: no source registered for column {c!r} in {table}")
        block, desc, src = spec
        source, release, endpoint = SRC[src]
        nn = df[c].notna().mean()
        rows.append(dict(table=table, column=c, block=block, description=desc, source=source,
                         source_release=release, endpoint=endpoint,
                         dtype=str(df[c].dtype), non_null_fraction=round(float(nn), 4)))
dd = pd.DataFrame(rows)
dd.to_csv("data_dictionary.csv", index=False)

json.dump(prov, open("provenance.json", "w"), indent=2)
print("data_dictionary", dd.shape)
print("target_features", tgt.shape, "| long", long.shape, "| ingredient_features", feat.shape)
print("with chembl id", feat.chembl_id.notna().sum(), "| with mechanism", feat.has_chembl_mechanism.sum(),
      "| with kegg", feat.kegg_drug_id.notna().sum())
