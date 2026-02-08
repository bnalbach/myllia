

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

# Present in Kaggle's metric runtime, or not, so that the code can also run locally
try:
    import kaggle_metric_utilities
except ImportError:  # pragma: no cover
    kaggle_metric_utilities = None


class ParticipantVisibleError(Exception):
    """Raise this for errors that should be shown to participants."""
    pass


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smoothstep easing: t^2 * (3 - 2t) for t in [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


def _gate_smoothstep(x: np.ndarray, a: float = 0.0, b: float = 0.2) -> np.ndarray:
    """
    Gate weights in [0, 1] using a smoothstep ramp from [a, b].

    x <= a -> 0
    x >= b -> 1
    else   -> smoothstep((x-a)/(b-a))
    """
    if b <= a:
        raise ValueError("gate_smoothstep requires b > a")

    t = (x - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return _smoothstep(t)


def _weighted_cosine(a: np.ndarray, b: np.ndarray, left: float, right: float, eps: float) -> float:
    """
    Weighted cosine similarity with smoothstep gating

    - x_i = max(|a_i|, |b_i|)
    - w_i = gate_smoothstep(x_i, left, right)
    - cosine uses w_i^2 as weights
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("weighted_cosine: a and b must have the same length")

    x = np.maximum(np.abs(a), np.abs(b))
    w = _gate_smoothstep(x, left, right)
    w2 = w * w

    num = np.sum(w2 * a * b)
    den_a = np.sqrt(np.sum(w2 * a * a))
    den_b = np.sqrt(np.sum(w2 * b * b))
    den = den_a * den_b

    if den < eps:
        return 0.0
    return float(num / den)


def _score_impl(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w: np.ndarray,
    baseline_wmae: np.ndarray,
    eps: float,
    max_log2: float,
    cos_left: float,
    cos_right: float,
) -> float:
    """
    Core numeric implementation. All inputs are numpy arrays aligned by row and column.
    """
    # Per-perturbation weighted MAE (weights assumed per-gene, typically sum to 1 per row)
    abs_err = np.abs(y_true - y_pred)
    pred_wmae = np.mean(abs_err * w, axis=1)

    # Safeguards: prevent division by zero / log blowups
    pred_wmae = np.maximum(pred_wmae, eps)
    baseline = np.maximum(baseline_wmae, eps)

    # Relative term per perturbation: log2(baseline / pred), capped above by max_log2
    terms = np.log2(baseline / pred_wmae)
    terms = np.minimum(terms, max_log2)
    sum_wmae = float(np.sum(terms))

    # Weighted cosine over concatenated vectors
    wcos = _weighted_cosine(y_pred.ravel(), y_true.ravel(), left=cos_left, right=cos_right, eps=eps)

    # Final: (sum_wmae) * max(0, wcos)
    final_score = float(sum_wmae * max(0.0, wcos))
    return  round(final_score, 5)


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    baseline_column_name: str = "baseline_wmae",
    weight_prefix: str = "w_",
    eps: float = 1e-12,
    max_log2: float = 5.0,
    cos_left: float = 0.0,
    cos_right: float = 0.2,
) -> float:
    """
    Weighted MAE relative to baseline multiplied by weighted cosine similarity

    Participant submission format (public):
      pert_id, TP53, GATA3, ...   (genes only)

    Host solution format (private):
      pert_id, TP53, GATA3, ..., w_TP53, w_GATA3, ..., baseline_wmae
      
    Weights must sum to n_genes for every perturbation

    Scoring:
      - For each perturbation row:
          pred_wmae = mean( |true_i - pred_i| * w_i )
          term      = log2(baseline_wmae / pred_wmae)
          cap: term <= max_log2
        sum_wmae = sum(term over rows)

      - Weighted cosine over the concatenated vectors (all rows * all genes),
        clipped at 0:
          wcos = max(0, weighted_cosine(pred_concat, true_concat))

      - Final score:
          sum_wmae * wcos

    Examples
    --------
    >>> sol = pd.DataFrame({
    ...     "pert_id": [0, 1],
    ...     "TP53":    [1.0, 0.0],
    ...     "GATA3":   [0.0, 1.0],
    ...     "w_TP53":  [1.0, 1.0],
    ...     "w_GATA3": [1.0, 1.0],
    ...     "baseline_wmae": [2.0, 2.0],
    ... })
    >>> sub = pd.DataFrame({
    ...     "pert_id": [0, 1],
    ...     "TP53":    [1.0, 0.8],
    ...     "GATA3":   [1.0, 0.8],
    ... })
    >>> score(sol.copy(), sub.copy(), "pert_id")
    2.81113
    """
    # --- Basic presence checks ---
    if row_id_column_name not in solution.columns:
        raise ValueError(f"Host error: solution missing id column '{row_id_column_name}'")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission missing id column '{row_id_column_name}'")
    if baseline_column_name not in solution.columns:
        raise ValueError(f"Host error: solution missing '{baseline_column_name}'")

    # --- Align rows by id (robust to row order) ---
    sol = solution.copy()
    sub = submission.copy()

    if sol[row_id_column_name].duplicated().any():
        raise ValueError("Host error: duplicate ids in solution")
    if sub[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError("Submission has duplicate ids")

    sol = sol.set_index(row_id_column_name)
    sub = sub.set_index(row_id_column_name)

    # Check id sets match exactly
    sol_ids = sol.index
    sub_ids = sub.index
    missing = sol_ids.difference(sub_ids)
    extra = sub_ids.difference(sol_ids)
    if len(missing) > 0:
        raise ParticipantVisibleError(f"Submission is missing {len(missing)} ids (example: {missing[0]})")
    if len(extra) > 0:
        raise ParticipantVisibleError(f"Submission has {len(extra)} unexpected ids (example: {extra[0]})")

    # Reorder submission to match solution order
    sub = sub.loc[sol_ids]

    # --- Determine the expected gene columns from solution ---
    # A column is treated as a gene if it has a matching weight column w_<gene>.
    expected_genes = []
    for c in sol.columns:
        if c == baseline_column_name:
            continue
        if c.startswith(weight_prefix):
            continue
        if f"{weight_prefix}{c}" in sol.columns:
            expected_genes.append(c)

    if len(expected_genes) == 0:
        raise ValueError("Host error: could not infer gene columns (need gene + matching w_<gene> columns)")

    # --- Validate submission columns: must be exactly the gene set (no weights/baseline) ---
    submitted_cols = list(sub.columns)

    if baseline_column_name in submitted_cols:
        raise ParticipantVisibleError(f"Submission must not include '{baseline_column_name}'")
    bad_weight_cols = [c for c in submitted_cols if c.startswith(weight_prefix)]
    if bad_weight_cols:
        raise ParticipantVisibleError("Submission must not include weight columns")

    # Must match expected genes exactly
    submitted_genes = set(submitted_cols)
    expected_set = set(expected_genes)

    missing_genes = sorted(expected_set - submitted_genes)
    extra_genes = sorted(submitted_genes - expected_set)

    if missing_genes or extra_genes:
        msg = []
        if missing_genes:
            msg.append(f"missing {len(missing_genes)} gene columns (example: {missing_genes[0]})")
        if extra_genes:
            msg.append(f"unexpected {len(extra_genes)} columns (example: {extra_genes[0]})")
        raise ParticipantVisibleError("Submission columns mismatch: " + "; ".join(msg))

    # Reorder submission gene columns to match solution’s expected order
    sub = sub[expected_genes]

    # Build weight column list in matching order
    weight_cols = [f"{weight_prefix}{g}" for g in expected_genes]

    # --- Dtype / finiteness checks ---
    bad_dtypes = {c: sub[c].dtype for c in expected_genes if not ptypes.is_numeric_dtype(sub[c])}
    if bad_dtypes:
        raise ParticipantVisibleError(f"Non-numeric submission columns found: {bad_dtypes}")

    for c in expected_genes:
        if not ptypes.is_numeric_dtype(sol[c]):
            raise ValueError(f"Host error: non-numeric solution gene column '{c}'")
    for c in weight_cols:
        if not ptypes.is_numeric_dtype(sol[c]):
            raise ValueError(f"Host error: non-numeric weight column '{c}'")
    if not ptypes.is_numeric_dtype(sol[baseline_column_name]):
        raise ValueError(f"Host error: non-numeric baseline column '{baseline_column_name}'")

    # --- Extract arrays ---
    y_true = sol[expected_genes].to_numpy(dtype=np.float64)
    y_pred = sub[expected_genes].to_numpy(dtype=np.float64)
    w = sol[weight_cols].to_numpy(dtype=np.float64)
    baseline = sol[baseline_column_name].to_numpy(dtype=np.float64)

    if not np.isfinite(y_pred).all():
        raise ParticipantVisibleError("Submission contains NaN or infinite values")
    if not np.isfinite(y_true).all():
        raise ValueError("Host error: solution contains NaN/inf in gene columns")
    if not np.isfinite(w).all():
        raise ValueError("Host error: solution contains NaN/inf in weight columns")
    if not np.isfinite(baseline).all():
        raise ValueError("Host error: solution contains NaN/inf in baseline column")

    if (w < 0).any():
        raise ValueError("Host error: weights must be non-negative")
    
    # Host sanity check
    # Expect weights to sum to the number of genes per row (so average weight ~ 1),
    # because pred_wmae is computed as mean(abs_err * w) over genes.
    n_genes = w.shape[1]
    if not np.allclose(np.sum(w, axis=1), n_genes, atol=1e-6, rtol=0.0):
        raise ValueError("Host error: weights must sum to n_genes per row")
    
    # --- Compute score (wrapped for Kaggle safety if available) ---
    
    if kaggle_metric_utilities is not None:
        return kaggle_metric_utilities.safe_call_score(
            _score_impl, y_true, y_pred, w, baseline, eps, max_log2, cos_left, cos_right
        )

    # Local fallback (e.g., doctest outside Kaggle)
    return _score_impl(y_true, y_pred, w, baseline, eps, max_log2, cos_left, cos_right)

