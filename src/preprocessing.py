import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from scipy.sparse import issparse

DATASET_COLUMN_MAP = {
    "myllia":   {"sgrna_symbol": "sgrna_symbol", "nCount_RNA": "nCount_RNA", "nFeature_RNA": "nFeature_RNA", "percent.mt": "percent.mt"},
    "vcc":      {"target_gene": "sgrna_symbol",  "": "nCount_RNA", "": "nFeature_RNA", "": "percent.mt"},
    "replogle": {"gene": "sgrna_symbol",          "UMI_count":  "nCount_RNA", "": "nFeature_RNA",     "mitopercent": "percent.mt"},
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
    # Rename columns that exist
    rename = {k: v for k, v in col_map.items() if k in adata.obs.columns}
    adata.obs = adata.obs.rename(columns=rename)
    # Keep only standard cols that exist
    keep = [c for c in KEEP_COLS if c in adata.obs.columns]
    adata.obs = adata.obs[keep]
    return adata


class PerturbationDataset(Dataset):
    def __init__(self, expr_matrix, pert_symbols, baseline, gene_names):
        """
        expr_matrix:  (n_cells, n_genes) np.ndarray — log-normalized expression
        pert_symbols: list of str — which gene was perturbed per cell
        baseline:     (n_genes,) np.ndarray — mean unperturbed expression
        gene_names:   list of str
        """
        self.expr = torch.tensor(expr_matrix, dtype=torch.float32)
        self.baseline = torch.tensor(baseline, dtype=torch.float32)
        self.pert_symbols = pert_symbols
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, idx):
        expr = self.expr[idx]                           # (n_genes,) actual expression of this cell
        pert_gene = self.pert_symbols[idx]

        pert_gene_idx = self.gene_to_idx.get(pert_gene, -1)

        # Delta = this cell's expression - unperturbed baseline
        delta = expr - self.baseline                    # (n_genes,) target

        return {
            "baseline": self.baseline,                  # (n_genes,) unperturbed reference
            "expr": expr,                               # (n_genes,) actual perturbed expression
            "pert_gene_idx": torch.tensor(pert_gene_idx, dtype=torch.long),
            "target": delta,                            # (n_genes,) true delta
        }


class Preprocessing:
    def __init__(self, config):
        self.means_path = config["datasets"]["means"]
        self.myllia = config["datasets"]["myllia"]
        self.vcc = config["datasets"]["vcc"]
        self.replogle = config["datasets"]["replogle"]
        self.jiang_TGFB = config["datasets"]["jiang_TGFB"]
        self.jiang_IFNB = config["datasets"]["jiang_IFNB"]
        self.jiang_IFNG = config["datasets"]["jiang_IFNG"]
        self.batch_size = config["ml"]["batch_size"]

    def concat_datasets(self):
        ### columns i want to keep
        # myllia: sgrna_symbol, nCount_RNA, nFeature_RNA, percent.mt
        # VCC: target_gene, , , 
        # replogle: gene, UMI_count, , mitopercent
        # jiang: gene, nCount_RNA, nFeature_RNA, percent.mito
        ### i want to make it so adata objects only have these obs columns remaining, and they are all named, like in "myllia" (sgrna_symbol, nCount_RNA, nFeature_RNA, percent.mt).
        ### Problem: the jiang dataset is too big so it doesnt fit in my RAM (only use the othes for now)
        ### also, do each dataset at a time, turn it in this format and find a way do the training (i cant fit it all into one adata object (too big))

        return

    def load(self):
        df = pd.read_csv(self.means_path, index_col="pert_symbol")

        # Baseline = non-targeting row
        baseline = df.loc["non-targeting"].values.astype(np.float32)

        adata = sc.read_h5ad(self.myllia)
        vcc = sc.read_h5ad(self.vcc)

        adata = adata.concat([adata, vcc], label="batch", keys=[""], join="inner")

        # Normalization of UMIs
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata, base=2)

        train_df = adata.to_df()
        train_obs = adata.obs.copy()

        # Delta = perturbed - baseline
        deltas = (train_df.values - baseline).astype(np.float32)
        pert_symbols = train_obs["sgrna_symbol"].to_list()
        gene_names = train_df.columns.to_list()

        return baseline, deltas, pert_symbols, gene_names

    def run_preprocessing(self):
        baseline, deltas, pert_symbols, gene_names = self.load()
        dataset = PerturbationDataset(deltas, baseline, gene_names)
        dataloader = DataLoader(dataset, batch_size=self.config["ml"]["batch_size"], shuffle=True)
        return dataloader, len(gene_names), pert_symbols, gene_names
