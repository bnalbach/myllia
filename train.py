import torch
import torch.nn as nn
from src.config import load_config
from src.preprocessing import Preprocessing
from src.model import Model


def train():
    config = load_config("./configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    preprocess = Preprocessing(config)
    dataloader, n_genes, pert_symbols, gene_names = preprocess.run_preprocessing()
    print(f"Genes: {n_genes} | Perturbations: {len(pert_symbols)}")

    # Model
    model = Model(config, n_genes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["ml"]["lr"])
    criterion = nn.MSELoss()

    for epoch in range(config["ml"]["epochs"]):
        model.train()
        total_loss = 0

        for batch in dataloader:
            baseline = batch["baseline"].to(device)  # (batch, n_genes)
            pert_gene_idx = batch["pert_gene_idx"].to(device)  # add this
            target   = batch["target"].to(device)    # (batch, n_genes) — true delta

            optimizer.zero_grad()
            pred = model(baseline, pert_gene_idx)                   # (batch, n_genes) — predicted delta
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['ml']['epochs']} | Loss: {total_loss/len(dataloader):.6f}")

    return model, gene_names


if __name__ == "__main__":
    train()