import torch
import torch.nn as nn

class GeneTokenizer(nn.Module):
    def __init__(self, n_genes, d_model):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes+1, d_model, padding_idx=0) # +1 for padding token
        self.expr_projection = nn.Linear(1, d_model)

    def forward(self, gene_ids, expr_values):
        gene_tok = self.gene_embedding(gene_ids) # (batch, seq_len, d_model)
        expr_tok = self.expr_projection(expr_values.unsqueeze(-1)) # (batch, seq_len, d_model)

        return gene_tok + expr_tok


class Model(nn.Module):
    def __init__(self, config, n_genes):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config["ml"]["d_model"],
                nhead=config["ml"]["nhead"],
                dim_feedforward=config["ml"]["dim_feedforward"],
                dropout=config["ml"]["dropout"],
                batch_first=True
            ),
            num_layers=config["ml"]["num_layers"]
        )
        self.fc_out = nn.Linear(config["ml"]["d_model"], config["ml"]["output_dim"])
        self.input_proj = nn.Linear(n_genes, config["ml"]["d_model"])

    def forward(self, x): # x: (batch, n_genes)
        x = self.input_proj(x).unsqueeze(1) # (batch, 1, d_model)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x

