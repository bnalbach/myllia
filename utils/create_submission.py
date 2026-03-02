import pandas as pd


"""
Example baseline submission script

Idea:
- Use the mean expression across all *training* perturbations (excluding "non-targeting")
  as the prediction for each of the 60 leaderboard perturbations.
- Convert predicted expression to deltas (log fold-changes) by subtracting the
  "non-targeting" control vector.
- Pad to 120 rows by adding 60 zero rows for the hidden test perturbations

Notes:
- The training dataset already contains exactly the 5127 genes to predict.
- This is a deliberately simple baseline / sanity-check submission.
- This submission should be scored at exactly 0.0 points
"""

import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
TRAIN_PATH = (
    "/kaggle/input/training-data/"
    "training_data_means.csv"
)
OUT_PATH = (
    "/kaggle/working/"
    "sample_submission.csv"
)

N_LEADERBOARD_PERTS = 60
N_TOTAL_PERTS = 120
CONTROL_LABEL = "non-targeting"


# -----------------------------
# Load data
# -----------------------------
train_data = pd.read_csv(TRAIN_PATH)

# Separate control vector ("non-targeting") and perturbation rows
control_rows = train_data.loc[train_data["pert_symbol"] == CONTROL_LABEL]
pert_rows = train_data.loc[train_data["pert_symbol"] != CONTROL_LABEL]

if control_rows.empty:
    raise ValueError(f'No control rows found for pert_symbol == "{CONTROL_LABEL}"')

# Control vector (single row expected)
nt_vec = control_rows.drop(columns=["pert_symbol"]).iloc[0]

# Mean predicted expression across perturbations (excluding control)
avg_vec = pert_rows.drop(columns=["pert_symbol"]).mean(axis=0)

# Convert to deltas (log fold-changes): predicted - control
delta_vec = avg_vec - nt_vec


# -----------------------------
# Build submission
# -----------------------------
# Leaderboard predictions: repeat the same delta vector for each of the 60 perts
predictions = pd.DataFrame(
    [delta_vec.to_numpy()] * N_LEADERBOARD_PERTS,
    columns=delta_vec.index,
)
predictions.insert(
    0,
    "pert_id",
    [f"pert_{i}" for i in range(1, N_LEADERBOARD_PERTS + 1)],
)

# Pad remaining rows (hidden test set) with zeros
n_pad = N_TOTAL_PERTS - N_LEADERBOARD_PERTS
pad = pd.DataFrame(0.0, index=range(n_pad), columns=delta_vec.index)
pad.insert(
    0,
    "pert_id",
    [f"pert_{i}" for i in range(N_LEADERBOARD_PERTS + 1, N_TOTAL_PERTS + 1)],
)

# Combine and write
submission = pd.concat([predictions, pad], ignore_index=True)
submission.to_csv(OUT_PATH, index=False)

print(f"Wrote submission with shape {submission.shape} -> {OUT_PATH}")