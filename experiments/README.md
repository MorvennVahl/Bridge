# Experiments — Round 1, Modal-sized

Five experiments. Each one is a **single Modal function invocation that must finish inside
ten minutes** (`timeout=600`). They are independent: any order, all five in parallel.

These five are a re-cut of the eight lanes in `lab/round_01.md` to fit that budget. The
mapping is recorded in each file under *Lineage*, and nothing from the round-1 plan is
dropped — lanes 1 and 8 merge into exp01, lanes 2/3/4 into exp02:

| file | question | round-1 lane | label | est. wall clock |
|---|---|---|---|---|
| `exp01_degree_floor.md` | what does observation frequency alone explain, and does that depend on the label definition? | L1 + L8 | 5 harm variants | ~4 min |
| `exp02_intrinsic_blocks.md` | does either side's intrinsic biology add anything on top of degree? | L2 + L3 + L4 | `y_faers_signal` | ~8 min |
| `exp03_target_neighbour_transport.md` | do drugs sharing a target behave alike across the group boundary? | L5 | `y_faers_signal` | ~6 min |
| `exp04_disproportionality_residual.md` | is the degree-free residual a better-behaved target than the raw flag? | L6 | `log(faers_prr)` residual | ~7 min |
| `exp05_efficacy_label_learnability.md` | is the efficacy half of the project viable at all? | L7 | `y_semmeddb_treats` | ~7 min |

Read `AGENT.md` before running any of them. It holds the rules and the traps, and the
rules there override anything here.

---

## Read this first: what these five are for

exp02 is the **project's null hypothesis**. If concatenating both sides without any
pair-level term matches what Round 2's pathway overlap achieves, the bridge hypothesis is
not supported and we have to say so.

exp03 is the **project's core claim in its cheapest testable form** — biology transports
across drugs — and it needs no condition-side genes, so it can run now while the Open
Targets disease arm is still missing.

exp01 and exp04 are about the label rather than the features. exp05 decides whether the
efficacy half of the project is worth investing in before we go buy an indication layer.

---

## Runtime contract

Every experiment obeys all of this. Deviating is fine if you say so in `findings`.

**One invocation, ten minutes.** `timeout=600` on the function. No hyperparameter grids —
each spec fixes its hyperparameters. If a step is going to blow the budget, take the
fallback the spec names rather than extending the timeout, and record in `findings` that
you took it.

**CPU only.** `cpu=8.0, memory=16384`. Nothing here is GPU work; requesting a GPU is a
review-blocker per `CLAUDE.md`.

**Code lives next to the spec.** `experiments/exp01_degree_floor.py` implements
`experiments/exp01_degree_floor.md`. Run it with:

```bash
python3 -m modal run experiments/exp01_degree_floor.py
```

The `modal` CLI is not on PATH — always `python3 -m modal` (see `skills/modal-compute`).

**Skeleton.** Pinned versions, stable app name, explicit timeout, `logging` inside the
function and `print` only in the local entrypoint:

```python
import logging
import pathlib
import modal

app = modal.App("bridge-exp01")  # bridge-exp0N, stable, no random suffix

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pandas==2.2.3",
    "numpy==2.1.3",
    "scikit-learn==1.5.2",
    "lightgbm==4.5.0",
    "pyarrow==17.0.0",
    "scipy==1.14.1",
    "matplotlib==3.9.2",
)

data = modal.Volume.from_name("bridge-data")
results = modal.Volume.from_name("bridge-results", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)
    ...
    results.commit()
    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp01",
        title=...,
        hypothesis=...,
        approach=...,
        label="y_faers_signal",
        features=[...],
        split="train/validate",
    )
    metrics = run.remote(exp)
    print(metrics)
    ln.complete(exp, metrics=metrics, findings=..., artifacts=[...], next_steps=...)
```

Registration happens in the **local entrypoint**, before `.remote()` — a hypothesis
registered after seeing the result is not a hypothesis. Completion happens locally too,
so the notebook write stays on one machine and the append-only file is never touched from
inside a container.

**The volume enforces the test seal.** Stage `train.csv` and `validate.csv` only. Do not
put `test.csv` on any volume; then no container can read it even by mistake.

```bash
python3 -m modal volume create bridge-data
python3 -m modal volume put bridge-data data/splits/train.csv           /splits/train.csv
python3 -m modal volume put bridge-data data/splits/validate.csv        /splits/validate.csv
python3 -m modal volume put bridge-data data/splits/split_assignment.csv /splits/split_assignment.csv
python3 -m modal volume put bridge-data data/input/drug/ingredient_features.csv    /drug/ingredient_features.csv
python3 -m modal volume put bridge-data data/input/drug/ingredient_target_long.csv /drug/ingredient_target_long.csv
python3 -m modal volume put bridge-data data/input/drug/target_features.csv        /drug/target_features.csv
python3 -m modal volume put bridge-data data/input/condition/condition_features_basic.csv /condition/condition_features_basic.csv
python3 -m modal volume put bridge-data data/input/condition/condition_group_long.csv     /condition/condition_group_long.csv
python3 -m modal volume put bridge-data data/data_dictionary.csv        /data_dictionary.csv
```

That is 125 MB and nothing in it is a label-adjacent or test file.
`condition_gene_hpo_long.csv` (116 MB) is deliberately **not** staged — no experiment in
this round needs per-condition gene lists, only the counts already aggregated into
`condition_features_basic.csv`.

**Results come back through a volume.** Write figures and tables to
`/results/<exp_id>/`, `results.commit()`, then pull them into the repo:

```bash
python3 -m modal volume get bridge-results <exp_id> results/
```

Every file keeps the experiment id in its name, as `AGENT.md` §8 requires.

**Preconditions, checked in the first seconds.** The repo is worked by several sessions at
once and `data/` gets rebuilt underneath you — at the time of writing,
`split_assignment.csv`, `condition_features_basic.csv` and `condition_group_long.csv` were
mid-regeneration and briefly absent. Assert every input path exists and has the expected
row count before fitting anything. If an input is missing, do not improvise a substitute:
`ln.complete(..., failed=True)` with `findings` naming the missing file, and stop. A null
from a missing input is a result; a quietly swapped input is a corrupted record.

---

## Shared method rules

**Degree from train only.** Drug degree (conditions per ingredient) and condition degree
(ingredients per condition) are counts over observed pairs. Computing them over the full
table leaks validate into the training features. Fit on train, apply the same mapping to
validate, and give conditions unseen in train the train **median**, not zero.

**Selection inside train, grouped on `group_key`.** 3-fold `GroupKFold` — not 5, to fit
the budget. Never group by row. One validate evaluation per experiment, after every choice
is fixed. Check `labnotebook.validate_evaluations()` first; when it is large, stop trusting
differences of a few thousandths.

**Average precision first.** Prevalence is 11.0% for `y_faers_signal`, 0.54% for
`y_semmeddb_treats`, 0.10% for `y_semmeddb_causes`. ROC-AUC flatters every model at these
rates. Report AP, then ROC-AUC, then a calibration curve, and always the prevalence
alongside so AP is interpretable.

**Report the subgroup breakdown**, not just the headline: mapped vs unmapped conditions,
disease arm vs phenotype arm, and by `best_match_tier`. A feature that only works on
well-mapped conditions is still useful, but only if you know that.

**Never train on `ingredient_features_label_adjacent.csv`.** Also drop from
`ingredient_features.csv` the whole `identity` block plus `in_cem_list`,
`in_indication_roster`, `roster_name`, `indication`, and every `chembl_fuzzy_*` column.
Select by the `block` column of `data_dictionary.csv` rather than by hand.

**Absence is not a negative.** Every row you fit on is an observed pair. None of these
five experiments invents unobserved negatives; if you change that, say what it assumes.

---

## Fold facts, verified from the written split files

| | train | validate |
|---|---|---|
| pairs | 723,586 | 434,151 |
| ingredients | 2,126 | 1,284 |
| groups (`group_key`) | 1,553 | 928 |
| `y_faers_signal` positives | 79,554 (11.0%) | 50,944 (11.7%) |
| `y_any_harm` positives | 80,261 | 51,309 |
| `y_semmeddb_treats` positives | 3,926 (0.54%) | 2,092 (0.48%) |
| `y_semmeddb_causes` positives | 715 (0.10%) | 370 (0.09%) |
| rows with `faers_case_count >= 3` | 344,653 (47.6%) | 206,586 (47.6%) |

**Measured timing, for sizing.** A `HistGradientBoostingClassifier` (200 iterations, 63
leaves) on 723,586 × 160 float32 takes **17 s** to fit and 2.3 s to score 434 k rows on 11
cores. Reading `train.csv` with pandas takes a few seconds. So a ten-minute budget buys
roughly 20 fits at full width — which is why the specs cap CV at 3 folds and fix
hyperparameters instead of searching them.

---

## What Round 2 is waiting for

The Open Targets disease-arm pull is **not** usable yet: `data/ref/ot_associations.jsonl`
holds 1,914 diseases and the entries are truncated (`"truncated": true`, e.g. 250 of 5,731
targets fetched for metabolic syndrome). Until that completes there are no per-datatype
genetic evidence scores, so no pathway-overlap and no graph experiment — which is the
right order anyway: these five establish the number that biology has to beat.
