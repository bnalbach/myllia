import torch
import torch.nn as nn


class GeneTokenizer(nn.Module):
    def __init__(self, n_genes, d_model):
        super().__init__()
        # Each gene gets a learnable identity vector
        self.gene_embedding = nn.Embedding(n_genes, d_model)
        # Projects scalar expression value → d_model
        self.expr_projection = nn.Linear(1, d_model)

    def forward(self, gene_ids, expr_values):
        gene_tok = self.gene_embedding(gene_ids)                    # (batch, seq, d_model)
        expr_tok = self.expr_projection(expr_values.unsqueeze(-1))  # (batch, seq, d_model)
        return gene_tok + expr_tok


class Model(nn.Module):
    def __init__(self, config, n_genes):
        super().__init__()
        d_model = config["ml"]["d_model"]
        self.n_genes = n_genes

        self.tokenizer = GeneTokenizer(n_genes, d_model)

        # Learnable [PERT] token embedding — one per gene
        # This encodes "this gene was knocked down"
        self.pert_embedding = nn.Embedding(n_genes + 1, d_model, padding_idx=n_genes)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config["ml"]["nhead"],
                dim_feedforward=config["ml"]["dim_feedforward"],
                dropout=config["ml"]["dropout"],
                batch_first=True
            ),
            num_layers=config["ml"]["num_layers"]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc_out = nn.Linear(d_model, n_genes)

    def forward(self, baseline, pert_gene_idx):
        """
        baseline:      (batch, n_genes) — expression with perturbed gene zeroed out
        pert_gene_idx: (batch,)         — index of knocked-down gene
        """
        batch_size = baseline.shape[0]
        gene_ids = torch.arange(self.n_genes, device=baseline.device)
        gene_ids = gene_ids.unsqueeze(0).expand(batch_size, -1)     # (batch, n_genes)

        # Tokenize baseline expression
        x = self.tokenizer(gene_ids, baseline)                       # (batch, n_genes, d_model)

        # Add perturbation signal to the knocked-out gene's token
        pert_tok = self.pert_embedding(pert_gene_idx)                # (batch, d_model)
        pert_idx = pert_gene_idx.clamp(0)                            # avoid -1 index
        x[torch.arange(batch_size), pert_idx] += pert_tok            # inject into that gene's position

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)              # (batch, 1, d_model)
        x = torch.cat([cls, x], dim=1)                               # (batch, n_genes+1, d_model)

        x = self.transformer(x)                                      # (batch, n_genes+1, d_model)

        cls_repr = x[:, 0, :]                                        # (batch, d_model)
        return self.fc_out(cls_repr)                                  # (batch, n_genes)