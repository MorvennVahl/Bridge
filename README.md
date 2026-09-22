# Bridge

Predict, for a drug–condition pair with no real-world data, whether the drug is likely to
**treat** the condition or **cause** it, using biology (targets, genes, pathways, disease
hierarchy) as the bridge between drugs that have real-world evidence and drugs that do not.

```
Ingredient --has_mechanism--> Target --encoded_by--> Gene --member_of--> Pathway
                                                       |                    :
                                                       +--associated_with-->+
                                                                            v
Ingredient ==================== RWD label: treats / causes ===========> Condition --is_a--> Condition
```

Single-line edges are inputs. The double-line edge is the label, learned from pairs that have
real-world evidence and predicted for pairs that do not.

## Data

All data files live in [data/](data/README.md), stored with Git LFS. That README lists each file,
its source, and how to regenerate the CEM extracts.

## Current task

1. Build condition-side features for the 5,631 conditions: implicated genes and pathways
   (Open Targets, Reactome), OMOP/SNOMED hierarchy position, phenotype profile.
2. Build drug-side features: ChEMBL mechanisms and targets.
3. Fit an explicit path-feature baseline (gradient boosting) before any GNN.
4. Evaluate with drugs held out entirely, never by random pair split.

See [DESIGN.md](DESIGN.md) for the full rationale and known data limitations.
