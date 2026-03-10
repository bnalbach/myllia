import torch
import pandas as pd
import numpy as np
from src.config import load_config
from src.preprocessing import Preprocessing
from src.model import Model


def predict(model, config, gene_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preprocess = Preprocessing(config)
    baseline, _, pert_symbols, _ = preprocess.load()

    baseline_tensor = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(device)

    # Load val perturbation IDs
    val_df = pd.read_csv(config["datasets"]["pert_ids_val"])

    submission_rows = []
    with torch.no_grad():
        pred_delta = model(baseline_tensor).squeeze(0).cpu().numpy()  # (n_genes,)

        for _, row in val_df.iterrows():
            submission_rows.append({
                "pert_id": row["pert_id"],
                "pert_symbol": row["pert"],
                **dict(zip(gene_names, pred_delta))
            })

    # Pad remaining 60 test rows with zeros (placeholder until test set released)
    existing_ids = {r["pert_id"] for r in submission_rows}
    for i in range(61, 121):
        pid = f"pert_{i}"
        if pid not in existing_ids:
            submission_rows.append({
                "pert_id": pid,
                "pert_symbol": "unknown",
                **dict(zip(gene_names, np.zeros(len(gene_names))))
            })

    submission = pd.DataFrame(submission_rows)
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")