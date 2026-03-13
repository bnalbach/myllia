import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import scanpy as sc
from scipy.sparse import issparse


DATASET_COLUMN_MAP = {
    "myllia":   {"sgrna_symbol": "sgrna_symbol", "nCount_RNA": "nCount_RNA", "nFeature_RNA": "nFeature_RNA", "percent.mt": "percent.mt"},
    "vcc":      {"target_gene": "sgrna_symbol"},
    "replogle": {"gene": "sgrna_symbol", "UMI_count": "nCount_RNA", "mitopercent": "percent.mt"},
}

KEEP_COLS = ["sgrna_symbol", "nCount_RNA", "nFeature_RNA", "percent.mt"]


def normalize_adata(adata):
    """Log-normalize raw UMI counts (same as Myllia pipeline: /total * 1e4, log2(x+1))"""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata, base=2)
    return adata


def standardize_obs(adata, dataset_name):
    """Rename and keep only the standard obs columns"""
    col_map = DATASET_COLUMN_MAP[dataset_name]

    rename = {k: v for k, v in col_map.items() if k in adata.obs.columns}
    adata.obs.rename(columns=rename, inplace=True)

    drop_cols = [c for c in adata.obs.columns if c not in KEEP_COLS]
    adata.obs.drop(columns=drop_cols, inplace=True)
    return adata


class ChunkedPerturbationDataset(Dataset):
    def __init__(self, h5ad_path, pert_col, baseline, gene_names, dataset_name, chunk_size=20000):
        self.path = h5ad_path
        self.pert_col = pert_col
        self.baseline = torch.tensor(baseline, dtype=torch.float32)
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size

        # Load ONLY obs metadata (no X) in RAM
        adata = sc.read_h5ad(h5ad_path, backed='r')
        adata = standardize_obs(adata, dataset_name)
        self.cell_indices = np.arange(adata.n_obs)  # all cells
        self.pert_labels  = adata.obs[pert_col].values
        self.gene_idx     = [i for i, g in enumerate(adata.var_names)
                             if g in set(gene_names)]
        self.avail_genes  = [g for g in adata.var_names if g in set(gene_names)]
        adata.file.close()

        # Chunk cache — only one chunk in RAM at a time
        self._cache_chunk_id = -1
        self._cache_X = None

    def _load_chunk(self, chunk_id):
        if chunk_id == self._cache_chunk_id:
            return
        start = chunk_id * self.chunk_size
        end   = min(start + self.chunk_size, len(self.cell_indices))
        idx   = self.cell_indices[start:end]

        adata = sc.read_h5ad(self.path, backed='r')
        X = adata.X[idx][:, self.gene_idx]
        if issparse(X):
            X = X.toarray()
        adata.file.close()

        # Normalize
        row_sums = X.sum(axis=1, keepdims=True)
        X = X / (row_sums + 1e-9) * 1e4
        X = np.log2(X + 1).astype(np.float32)

        self._cache_X = X
        self._cache_chunk_id = chunk_id

    def __len__(self):
        return len(self.cell_indices)

    def __getitem__(self, idx):
        chunk_id  = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        self._load_chunk(chunk_id)

        expr = torch.tensor(self._cache_X[local_idx], dtype=torch.float32)
        pert_gene     = self.pert_labels[idx]
        pert_gene_idx = self.gene_to_idx.get(pert_gene, -1)

        return {
            "baseline":      self.baseline,
            "expr":          expr,
            "pert_gene_idx": torch.tensor(pert_gene_idx, dtype=torch.long),
            "target":        expr - self.baseline,
        }


class Preprocessing:
    def __init__(self, config):
        self.config       = config
        self.means_path   = config["datasets"]["means"]
        self.myllia_path  = config["datasets"]["myllia"]
        self.vcc_path     = config["datasets"]["vcc"]
        self.replogle_path= config["datasets"]["replogle"]
        self.batch_size   = config["ml"]["batch_size"]

    def load_baseline(self):
        """Load myllia baseline expressions."""
        df = pd.read_csv(self.means_path, index_col="pert_symbol")
        baseline   = df.loc["non-targeting"].values.astype(np.float32)
        gene_names = df.columns.tolist()
        return baseline, gene_names

    def build_datasets(self):
        baseline, gene_names = self.load_baseline()

        # only loads metadata into RAM
        datasets = [
            ChunkedPerturbationDataset(self.myllia_path, "sgrna_symbol", baseline, gene_names, "myllia"),
            ChunkedPerturbationDataset(self.vcc_path, "target_gene", baseline, gene_names, "vcc"),
            ChunkedPerturbationDataset(self.replogle_path, "gene", baseline, gene_names, "replogle"),
        ]
        print("Skipping Jiang datasets")
        return datasets, gene_names, baseline

    def run_preprocessing(self):
        print("Building datasets...")
        datasets, gene_names, baseline = self.build_datasets()
        combined = ConcatDataset(datasets)
        print(f"Total cells: {len(combined)}")

        dataloader = DataLoader(
            combined,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,   # must be 0 — chunk cache doesn't work with multiprocessing
            pin_memory=True
        )
        return dataloader, len(gene_names), gene_names, baseline
































