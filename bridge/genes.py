"""Gene identifier bridging between the condition side and the drug side.

The two halves of this repo do not speak the same gene language. HPO's annotation files —
the only disease-to-gene and phenotype-to-gene evidence we hold — key on **NCBI gene ids**.
The drug side (`ingredient_target_long.csv`, `target_features.csv`) keys on **gene symbol**
and **Ensembl gene id**. Without a bridge the biology never joins, and the whole premise of
the project is that it does.

Open Targets' target index has no NCBI cross-reference (its `dbXrefs` carry HGNC, PDB,
Reactome and friends, not Entrez), so the usable pivot is the gene symbol, which both sides
carry. Symbols drift over time, so `symbol_to_ensembl` also indexes Open Targets'
`symbolSynonyms` and `obsoleteSymbols` to catch renamed genes.
"""

import logging
from collections.abc import Sequence

import polars as pl

from bridge import paths

logger = logging.getLogger(__name__)


def load_gene_symbols() -> dict[int, str]:
    """NCBI gene id -> gene symbol, from the two HPO annotation files.

    Both files pair the id and the symbol directly, which makes them the authoritative
    source for this mapping; no external lookup is needed.
    """
    symbols: dict[int, str] = {}
    for path, id_column in (
        (paths.HPO_PHENOTYPE_TO_GENES, "ncbi_gene_id"),
        (paths.HPO_GENES_TO_DISEASE, "ncbi_gene_id"),
    ):
        frame = pl.read_csv(
            path, separator="\t", columns=[id_column, "gene_symbol"], infer_schema_length=0
        )
        for raw_id, symbol in frame.iter_rows():
            if not raw_id or not symbol:
                continue
            # genes_to_disease prefixes the id ("NCBIGene:64170"); phenotype_to_genes does not.
            gene_id = int(str(raw_id).removeprefix("NCBIGene:"))
            symbols.setdefault(gene_id, symbol)
    logger.info("%d NCBI gene ids carry a symbol", len(symbols))
    return symbols


def symbol_to_ensembl() -> dict[str, str]:
    """Gene symbol -> Ensembl gene id, from the Open Targets target index.

    Approved symbols win over synonyms and obsolete symbols, so a current symbol is never
    shadowed by another gene's former name.
    """
    files = sorted(paths.OT_TARGET_GLOB_DIR.glob("target__part-*.parquet"))
    if not files:
        logger.warning("no Open Targets target parquet files found; Ensembl ids unavailable")
        return {}

    frame = pl.read_parquet(
        files, columns=["id", "approvedSymbol", "symbolSynonyms", "obsoleteSymbols"]
    )

    def labels(values: Sequence[dict] | None) -> list[str]:
        return [v["label"] for v in values or () if v and v.get("label")]

    mapping: dict[str, str] = {}
    for ensembl, _, synonyms, obsolete in frame.iter_rows():
        for label in labels(synonyms) + labels(obsolete):
            mapping.setdefault(label, ensembl)
    # Second pass so approved symbols overwrite anything a synonym claimed.
    for ensembl, approved, _, _ in frame.iter_rows():
        if approved:
            mapping[approved] = ensembl

    logger.info("%d gene symbols resolve to an Ensembl id", len(mapping))
    return mapping
