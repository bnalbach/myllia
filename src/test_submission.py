import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import issparse


VCC_PATH        = "/home/data/single_cell/vcc_2025/adata_Training.h5ad"
REPLOGLE_PATH   = "/home/data/single_cell/Replogle_et_al/K562_essential_raw_singlecell_01.h5ad"
MEANS_PATH      = "/home/data/kaggle_data/myllia/training_data_means.csv"
SUBMISSION_PATH = "/home/data/kaggle_data/myllia/sample_submission.csv"
PERT_PATH       = "/home/data/kaggle_data/myllia/pert_ids_all.csv"
OUT_PATH        = "external_deltas.csv"

# reference sets 
pert_df = pd.read_csv(PERT_PATH, index_col="pert")
p0 = set(pert_df.index)

submission_df = pd.read_csv(SUBMISSION_PATH, index_col="pert_id")
s0 = list(submission_df.columns)
fallback = submission_df.iloc[0]


def compute_deltas_sparse(adata, pert_col, chunk_size=5000):
    """
    Computes mean expression per perturbation entirely in sparse format.
    Never loads the full dense matrix into RAM.
    """
    genes = adata.var_names.tolist()
    pert_labels = adata.obs[pert_col].values
    unique_perts = np.unique(pert_labels)

    # accumulate sum and count per perturbation
    n_genes = len(genes)
    sums   = {p: np.zeros(n_genes, dtype=np.float64) for p in unique_perts}
    counts = {p: 0 for p in unique_perts}

    n_cells = adata.n_obs
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)

        # load chunck-wise
        chunk = adata.X[start:end]
        if issparse(chunk):
            chunk = chunk.toarray()

        chunk_perts = pert_labels[start:end]
        for p in np.unique(chunk_perts):
            mask = chunk_perts == p
            sums[p]   += chunk[mask].sum(axis=0)
            counts[p] += mask.sum()

        if start % 50000 == 0:
            print(f"    processed {start}/{n_cells} cells...")

    means = pd.DataFrame(
        {p: sums[p] / counts[p] for p in unique_perts},
        index=genes
    ).T  # (n_perts, n_genes)

    # delta = mean(pert) - mean(non-targeting)
    baseline = means.loc["non-targeting"]
    deltas = means.drop(index="non-targeting") - baseline

    return deltas


# load & process VCC 
print("Loading VCC...")
vcc = sc.read_h5ad(VCC_PATH, backed='r')
vcc_pert_col = "target_gene"

cell_mask = vcc.obs[vcc_pert_col].isin(p0 | {"non-targeting"}).values
vcc_genes = [g for g in s0 if g in vcc.var_names]

cell_idx = np.where(cell_mask)[0]
gene_idx = [vcc.var_names.tolist().index(g) for g in vcc_genes]

# single slice (no chained views)
X = vcc.X[cell_idx][:, gene_idx]
if issparse(X):
    X = X.toarray()
vcc.file.close()

# rebuild a small in-memory adata from the slice
import anndata as ad
vcc = ad.AnnData(
    X=X,
    obs=vcc.obs.iloc[cell_idx][[vcc_pert_col]].copy(),
    var=pd.DataFrame(index=vcc_genes)
)

print(f"  VCC: {vcc.n_obs} cells, {vcc.n_vars} genes, "
      f"{vcc.obs[vcc_pert_col].nunique()} perturbations")

sc.pp.normalize_total(vcc, target_sum=1e4)
sc.pp.log1p(vcc, base=2)

vcc_deltas = compute_deltas_sparse(vcc, vcc_pert_col)
del vcc, X
print(f"  VCC deltas: {vcc_deltas.shape}")


# load & process Replogle
print("Loading Replogle...")
rep = sc.read_h5ad(REPLOGLE_PATH, backed='r')
rep_pert_col = "gene"

mask = rep.obs[rep_pert_col].isin(p0 | {"non-targeting"}).values
rep_genes = [g for g in s0 if g in rep.var_names]

# don't call to_memory() on Replogle (too large)
# keep backed and process in chunks directly
rep_filtered_obs = rep.obs[mask].copy()
rep_var_names = rep.var_names.tolist()
gene_idx = [rep_var_names.index(g) for g in rep_genes]

print(f"  Replogle: {mask.sum()} cells after filter, {len(rep_genes)} genes")

# process Replogle in chunks directly from disk
chunk_size = 5000
pert_labels = rep_filtered_obs[rep_pert_col].values
unique_perts = np.unique(pert_labels)
n_genes = len(rep_genes)
sums   = {p: np.zeros(n_genes, dtype=np.float64) for p in unique_perts}
counts = {p: 0 for p in unique_perts}

filtered_indices = np.where(mask)[0]
n_filtered = len(filtered_indices)

for i in range(0, n_filtered, chunk_size):
    idx_chunk = filtered_indices[i:i + chunk_size]
    
    chunk = rep.X[idx_chunk][:, gene_idx]
    if issparse(chunk):
        chunk = chunk.toarray()

    row_sums = chunk.sum(axis=1, keepdims=True)
    chunk = chunk / (row_sums + 1e-9) * 1e4
    chunk = np.log2(chunk + 1)

    chunk_perts = pert_labels[i:i + chunk_size]
    for p in np.unique(chunk_perts):
        pmask = chunk_perts == p
        sums[p]   += chunk[pmask].sum(axis=0)
        counts[p] += pmask.sum()

    if i % 50000 == 0:
        print(f"    processed {i}/{n_filtered} cells...")

rep.file.close()

means = pd.DataFrame(
    {p: sums[p] / counts[p] for p in unique_perts},
    index=rep_genes
).T
baseline = means.loc["non-targeting"]
rep_deltas = means.drop(index="non-targeting") - baseline
print(f"  Replogle deltas: {rep_deltas.shape}")


# combine
print("Combining...")
all_perts = sorted(p0)

vcc_full = vcc_deltas.reindex(index=all_perts, columns=s0)
rep_full = rep_deltas.reindex(index=all_perts, columns=s0)

result = (vcc_full + rep_full) / 2
result = result.fillna(vcc_full)
result = result.fillna(rep_full)
result = result.fillna(fallback)

pert_df2 = pd.read_csv(PERT_PATH)
result = result.reindex(pert_df2["pert"].values)
result.index = pert_df["pert_id"].values
result.index.name = "pert_id"
result = result.astype(np.float32)
result.to_csv(OUT_PATH)
print(f"Saved: {OUT_PATH} — shape: {result.shape}")