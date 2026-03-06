import torch 
import torch.nn as nn
from src.config import load_config
from src.preprocessing import Preprocessing
from src.model import Model

def train():
    config = load_config("./configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = Preprocessing(config)
    dataloader, n_genes = preprocess.run_preprocessing()

    model = Model(config, n_genes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["ml"]["lr"])
    criterion = nn.MSELoss()

    for epoch in range(config["ml"]["epochs"]):
        model.train()
        total_loss = 0

        for batch in dataloader:
            gene_ids = batch["gene_ids"].to(device)
            expr_values = batch["expr_values"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            targets = # placeholder

            optimizer.zero_grad()
            preds = model(gene_ids, expr_values, padding_mask)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train()