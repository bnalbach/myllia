import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import scanpy as sc
from scipy.sparse import issparse
import h5py
import os


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
    adata.obs = adata.obs.rename(columns=rename)

    keep = [c for c in KEEP_COLS if c in adata.obs.columns]
    adata.obs = adata.obs[keep]
    return adata


class HDF5PerturbationDataset(Dataset):
    def __init__(self, cache_path, baseline, gene_names):
        self.cache_path = cache_path
        self.baseline = torch.tensor(baseline, dtype=torch.float32)
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Read only length — don't keep file open
        with h5py.File(cache_path, 'r') as f:
            self.n_cells = f['X'].shape[0]

        self._file  = None  # opened lazily per worker

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        # Open file lazily — each DataLoader worker gets its own handle
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r')

        expr = torch.tensor(self._file['X'][idx], dtype=torch.float32)
        pert_gene = self._file['perts'][idx].decode('utf-8')
        pert_gene_idx = self.gene_to_idx.get(pert_gene, -1)

        return {
            "baseline": self.baseline,
            "expr": expr,
            "pert_gene_idx": torch.tensor(pert_gene_idx, dtype=torch.long),
            "target": expr,
        }


class Preprocessing:
    def __init__(self, config):
        self.config = config
        self.means_path = config["datasets"]["means"]
        self.myllia_path = config["datasets"]["myllia"]
        self.vcc_path = config["datasets"]["vcc"]
        self.replogle_path = config["datasets"]["replogle"]
        self.batch_size = config["ml"]["batch_size"]

    def load_baseline(self):
        """Load the unperturbed reference expression from training_data_means.csv"""
        df = pd.read_csv(self.means_path, index_col="pert_symbol")
        baseline = df.loc["non-targeting"].values.astype(np.float32)
        gene_names = df.columns.tolist()
        return baseline, gene_names

    def load_dataset_chunked(self, baseline, path, dataset_name, target_genes, chunk_size=20000):
        """
        For large datasets that don't fit in RAM — read in chunks using h5ad backed mode.
        Yields (X_chunk, pert_symbols_chunk) without loading the full file.
        """
        print(f"  Loading {dataset_name} (chunked)...")
        adata = sc.read_h5ad(path, backed='r')
        adata = standardize_obs(adata, dataset_name)

        common_genes = [g for g in target_genes if g in adata.var_names]
        gene_idx = [adata.var_names.tolist().index(g) for g in common_genes]
        n_cells = adata.n_obs

        nt_mask = adata.obs["sgrna_symbol"] == "non-targeting"
        nt_cells = adata[nt_mask.values, common_genes].to_memory()
        normalize_adata(nt_cells)
        X_nt = nt_cells.X.toarray() if issparse(nt_cells.X) else nt_cells.X
        dataset_baseline = X_nt.mean(axis=0)  # (n_genes,) mean per gene
        del nt_cells, X_nt
        print(f"  Baseline computed from {nt_mask.sum()} non-targeting cells")

        common_gene_to_idx = {g: i for i, g in enumerate(common_genes)}
        target_gene_to_idx = {g: i for i, g in enumerate(target_genes)}
        
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = adata[start:end].to_memory()
            chunk = chunk[:, common_genes].copy()
            normalize_adata(chunk)
            X = chunk.X.toarray() if issparse(chunk.X) else chunk.X
            X = (X - dataset_baseline) # * (baseline / dataset_baseline) does only make sense mathematically
            pert_syms = chunk.obs["sgrna_symbol"].tolist()

            # fill missing genes with baseline (delta = 0 for unobserved genes) --> BETTER (TODO): infer activity based on pathway (reactome)
            X_full = np.zeros((X.shape[0], len(target_genes)), dtype=np.float32)
            for g in target_genes:
                if g in common_gene_to_idx:
                    X_full[:, target_gene_to_idx[g]] = X[:, common_gene_to_idx[g]]
                # else: stays 0 (delta = 0 = no change observed)

            del chunk
            yield X_full, pert_syms, common_genes


        adata.file.close()

    def build_datasets(self):
        baseline, gene_names = self.load_baseline()
        cache_path = "/home/data/kaggle_data/myllia/combined_cache.h5"

        # build cache file if it doesn't exist
        if not os.path.exists(cache_path):
            with h5py.File(cache_path, 'w') as f:
                f.create_dataset('gene_names', data=np.array(gene_names, dtype='S'))
                # pre-allocate X and perts as resizable datasets
                f.create_dataset('X', shape=(0, len(gene_names)), maxshape=(None, len(gene_names)), dtype=np.float32, chunks=(1000, len(gene_names)))
                f.create_dataset('perts', shape=(0,), maxshape=(None,), dtype='S50')

            for path, name in [
                (self.myllia_path, "myllia"),
                (self.vcc_path, "vcc"),
                (self.replogle_path, "replogle"),
            ]:
                print(f"Writing {name} to cache...")
                for X_chunk, perts_chunk, _ in self.load_dataset_chunked(baseline, path, name, gene_names):
                    with h5py.File(cache_path, 'a') as f:
                        n_existing = f['X'].shape[0]
                        n_new = X_chunk.shape[0]
                        f['X'].resize(n_existing + n_new, axis=0)
                        f['perts'].resize(n_existing + n_new, axis=0)
                        f['X'][n_existing:] = X_chunk.astype(np.float32)
                        f['perts'][n_existing:] = np.array(perts_chunk, dtype='S50')
                    del X_chunk
                    print(f"  appended chunk, total cells: {n_existing + n_new}")

        # lazy dataset that reads from disk during training
        dataset = HDF5PerturbationDataset(cache_path, baseline, gene_names)
        return dataset, gene_names, baseline

    def run_preprocessing(self):
        print("Building datasets...")
        dataset, gene_names, baseline = self.build_datasets()
        print(f"Total cells: {len(dataset)}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,       
            pin_memory=True      # faster GPU transfer
        )

        return dataloader, len(gene_names), gene_names, baseline



































