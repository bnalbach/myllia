import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.sparse import issparse

class Preprocessing:
    def __init__(self, config):
        self.baseline = config["datasets"]["baseline"]
        self.myllia = config["datasets"]["myllia"]
        self.vcc = config["datasets"]["vcc"]
        self.reactome = config["datasets"]["reactome"]
        
        self.max_seq_len = config["ml"]["max_seq_len"]
        self.batch_size = config["ml"]["batch_size"]
        return
    
    def combine_and_preprocess(self):
        sc_data = sc.read_h5ad(self.myllia)

        ### ADD BETTER FILTERS
        sc.pp.filter_cells(sc_data, min_genes=200)
        sc.pp.filter_genes(sc_data, min_cells=3)
        
        ### should already be applied
        sc.pp.normalize_total(sc_data, target_sum=1e4)
        sc.pp.log1p(sc_data)

        return sc_data
    
    def process_vcc_2025(self):
        vcc = sc.read_h5ad(self.vcc)

        return
    
    # def create_embedding(self, sc_data):
    #     ### custom embedding to incooperate pathway info (seperate function)!
    #     X = torch.tensor(sc_data.X.toarray(), dtype=torch.float32) # (n_cells, n_genes)
    #     return X
    
    def run_preprocessing(self):
        sc_data = self.combine_and_preprocess()
        n_genes = sc_data.n_var
        dataset = SingleCellDataset(sc_data, max_seq_len=self.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader, n_genes


### ADD HERE: add pathway information in the var df! --> in which pathways is each gene? close interactors/which other proteins does it interact with?
class SingleCellDataset(Dataset):
    def __init__(self, sc_data, max_seq_len=2000):
        self.max_seq_len = max_seq_len

        X = sc_data.X.toarray() if issparse(sc_data.X) else sc_data.X
        self.expressions = X

        self.perturbations = sc_data.obs["sgrna_symbol"].values  # RENAME when using multiple datasets!

    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        expr = self.expressions[idx] # (n_genes,)

        # drop non-expressed genes --> LATER: use normalized counts (drop no change genes)
        expressed_mask =  expr > 0
        gene_ids = np.where(expressed_mask)[0]
        expr_values = expr[expressed_mask]

        # if more too many expressed genes in a cell (too many tokens), sample random subset
        if len(gene_ids) > self.max_seq_len:
            # CAN BE IMPROVED
            chosen = np.random.choice(len(gene_ids), self.max_seq_len, replace=False)
            gene_ids = gene_ids[chosen]
            expr_values = expr_values[chosen]

        seq_len = len(gene_ids)

        # pad to max_seq_len
        padded_gene_ids = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_expr = np.zeros(self.max_seq_len, dtype=np.float32)
        padding_mask = np.ones(self.max_seq_len, dtype=bool) # if true, ignore padding

        padded_gene_ids[:seq_len] = gene_ids
        padded_expr[:seq_len] = expr_values
        padding_mask[:seq_len] = False

        sample = {
            "gene_ids": torch.tensor(padded_gene_ids),
            "expr_values": torch.tensor(padded_expr),
            "padding_mask": torch.tensor(padding_mask),
            "perturbation": self.perturbations[idx]
        }

        return sample