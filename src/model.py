import torch
import torch.nn as nn

class GeneTokenizer(nn.Module):
    def __init__(self, n_genes, d_model):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes+1, d_model, padding_idx=0) # +1 for padding token
        self.expr_projection = nn.Linear(1, d_model)

    def forward(self, gene_ids, expr_values):
        gene_tok = self.gene_embedding(gene_ids)                        # (batch, seq_len, d_model)
        expr_tok = self.expr_projection(expr_values.unsqueeze(-1))      # (batch, seq_len, d_model)

        return gene_tok + expr_tok


class Model(nn.Module):
    def __init__(self, config, n_genes):
        super().__init__()
        d_model = config["ml"]["d_model"]
        self.n_genes = n_genes

        self.tokenizer = GeneTokenizer(n_genes, d_model)
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc_out = nn.Linear(config["ml"]["d_model"], config["ml"]["output_dim"])

    def forward(self, gene_ids, expr_values, padding_mask):
        batch_size = gene_ids.shape[0]

        x = self.tokenizer(gene_ids, expr_values)                       # (batch, seq_len, d_model)

        # prepend CLS token (like BERT)
        cls = self.cls_token.expand(batch_size, -1, -1)                 # (batch, 1, d_model)
        x = torch.cat([cls, x], dim=1)                                  # (batch, seq_len+1, d_model)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)      # (batch, seq_len+1, d_model)
        
        cls_repr = x[:, 0, :]                                           # (batch, d_model) — CLS token output
        out = self.fc_out(cls_repr)                                     # (batch, n_genes) — predicted post-perturbation expression
        return out

