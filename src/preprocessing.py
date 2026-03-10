import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc


class PerturbationDataset(Dataset):
    def __init__(self, deltas, baseline, gene_names, pert_symbols):
        """
        deltas:     (n_cells, n_genes) — target delta expressions per cell
        baseline:   (n_genes,)         — mean unperturbed reference expression 
        gene_names: list of gene names (length n_genes)
        """
        self.baseline = torch.tensor(baseline, dtype=torch.float32)
        self.deltas = torch.tensor(deltas, dtype=torch.float32)
        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.pert_symbols = pert_symbols

    def __len__(self):
        return len(self.deltas)

    def __getitem__(self, idx):
        pert_gene = self.pert_symbols[idx]  # e.g. "BRCA1"

        # Zero out the perturbed gene in the baseline input
        baseline_perturbed = self.baseline.clone()
        if pert_gene in self.gene_to_idx:
            baseline_perturbed[self.gene_to_idx[pert_gene]] = 0.0

        return {
            "baseline": baseline_perturbed,        # (n_genes,) — with knocked-out gene zeroed
            "pert_gene_idx": torch.tensor(
                self.gene_to_idx.get(pert_gene, -1), dtype=torch.long
            ),                                      # index of perturbed gene, -1 if not found
            "target": self.deltas[idx],             # (n_genes,) — true delta
        }


class Preprocessing:
    def __init__(self, config):
        self.means_path = config["datasets"]["means"]
        self.myllia = config["datasets"]["myllia"]
        self.batch_size = config["ml"]["batch_size"]

    def load(self):
        df = pd.read_csv(self.means_path, index_col="pert_symbol")

        # Baseline = non-targeting row
        baseline = df.loc["non-targeting"].values.astype(np.float32)

        adata = sc.read_h5ad(self.myllia)

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
