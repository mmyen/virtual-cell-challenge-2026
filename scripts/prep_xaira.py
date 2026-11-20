"""Turn a raw X-Atlas/Orion subset (from xaira_subset.py) into VCC-shaped validation files.

Outputs, both raw integer counts on the 18,533 VCC genes in gene_names.csv order:
    <prefix>_controls.h5ad   non-targeting cells, laid out like data/controls/context_*.h5ad
    <prefix>_perturbed.h5ad  perturbed cells (the held-out "truth"), one row per cell

Genes: Xaira uses current HGNC symbols, the VCC list uses some older ones (HIST1H1B -> H1-5,
SARS -> SARS1). Each VCC gene is matched exactly, else through an HGNC previous symbol, else
through an alias, taking only unambiguous matches to genes not already used. VCC genes that
stay unmatched are all-zero columns, flagged var["in_source"] = False; leave them out of scoring.

Example:
    python scripts/prep_xaira.py data/xaira/HCT116_vcc_test_subset.h5ad \
        --context HCT116 --out-prefix data/xaira/HCT116
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = __file__.rsplit("/scripts/", 1)[0]


def hgnc_lookup(path):
    """symbol lists keyed by previous symbol and by alias, from the HGNC complete set."""
    h = pd.read_csv(path, sep="\t", usecols=["symbol", "prev_symbol", "alias_symbol"], dtype=str)
    maps = {}
    for col in ("prev_symbol", "alias_symbol"):
        m = {}
        for sym, olds in zip(h.symbol, h[col].fillna("")):
            for o in filter(None, olds.split("|")):
                m.setdefault(o, []).append(sym)
        maps[col] = m
    return maps


def match_genes(vcc, source, maps):
    """For each VCC gene, the source column index or -1, plus how it was matched."""
    pos = {g: i for i, g in enumerate(source)}
    idx = np.full(len(vcc), -1)
    how = np.array(["missing"] * len(vcc), dtype=object)
    used = set()
    for k, g in enumerate(vcc):
        if g in pos:
            idx[k], how[k] = pos[g], "exact"
            used.add(g)
    for col, label in (("prev_symbol", "hgnc_previous"), ("alias_symbol", "hgnc_alias")):
        for k in np.flatnonzero(idx < 0):
            c = [s for s in dict.fromkeys(maps[col].get(vcc[k], [])) if s in pos and s not in used]
            if len(c) == 1:
                idx[k], how[k] = pos[c[0]], label
                used.add(c[0])
    return idx, how


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subset")
    p.add_argument("--context", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--genes", default=f"{ROOT}/data/controls/gene_names.csv")
    p.add_argument("--hgnc", default=f"{ROOT}/data/reference/hgnc_complete_set.txt")
    a = p.parse_args()

    x = ad.read_h5ad(a.subset)
    vcc = pd.read_csv(a.genes)["gene_name"].to_numpy()
    idx, how = match_genes(vcc, np.asarray(x.var_names, dtype=str), hgnc_lookup(a.hgnc))

    # source columns -> VCC order; unmatched VCC genes become empty columns
    X = sp.csr_matrix(x.X, dtype=np.float32)
    have = idx >= 0
    sel = sp.csr_matrix((np.ones(have.sum(), np.float32), (idx[have], np.flatnonzero(have))),
                        shape=(x.n_vars, len(vcc)))
    Y = (X @ sel).tocsr()
    Y.sort_indices()
    assert np.array_equal(Y.data, np.round(Y.data)), "counts are not integers"

    var = pd.DataFrame({"in_source": have, "match": how}, index=pd.Index(vcc, name=None))
    tgt = x.obs["target_gene"].astype(str).to_numpy()
    is_ctrl = tgt == "non-targeting"
    obs = pd.DataFrame({"target_gene": tgt, "context": a.context,
                        "sample": x.obs["sample"].astype(str).to_numpy()}, index=x.obs_names)
    for name, rows in (("controls", is_ctrl), ("perturbed", ~is_ctrl)):
        o = obs[rows].copy()
        for c in o:
            o[c] = o[c].astype("category")
        out = ad.AnnData(X=Y[np.flatnonzero(rows)], obs=o, var=var.copy())
        out.uns["source"] = x.uns.get("source", "")
        out.write_h5ad(f"{a.out_prefix}_{name}.h5ad", compression="gzip")

    # report
    lib = np.asarray(X.sum(1)).ravel()
    kept = np.asarray(Y.sum(1)).ravel() / lib
    n_per = pd.Series(tgt[~is_ctrl]).value_counts()
    print(f"genes: {pd.Series(how).value_counts().to_dict()}")
    print(f"share of each cell's UMIs on VCC genes: median {np.median(kept):.3f}")
    print(f"controls: {is_ctrl.sum():,} cells; perturbed: {(~is_ctrl).sum():,} cells, {len(n_per)} targets, "
          f"cells/target median {n_per.median():.0f} (min {n_per.min()}, max {n_per.max()})")
    print(f"median UMI/cell (all source genes): {np.median(lib):.0f}")
    print(f"wrote {a.out_prefix}_controls.h5ad and {a.out_prefix}_perturbed.h5ad")


if __name__ == "__main__":
    main()
