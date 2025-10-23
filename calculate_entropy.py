import numpy as np
import pandas as pd

# --- entropy function (log2 → bits) ---
def distribution_entropy(counts, base="bits", epsilon=1e-12):
    counts = np.asarray(counts, dtype=float)
    probs = counts / counts.sum()
    probs = np.where(probs == 0, epsilon, probs)
    log_fn = np.log2 if base == "bits" else np.log
    return -float(np.sum(probs * log_fn(probs)))

# ===== counts replaced by qwen2.5-3B =====
# Keep the label order noted in comments for each dataset.
datasets = {
    # order: [invalid, mixed, negative, neutral, positive]
    "imdb_sentiment": {
        "base": [0, 0, 5309, 0, 4691],
        "F":    [0, 0, 5149, 0, 4851],
        "T":    [0, 0, 5435, 0, 4565],
    },
    # order: [depression, invalid, normal]
    "mental_sentiment": {
        "base": [15621, 0, 16126],
        "F":    [17428, 0, 14319],
        "T":    [14451, 0, 17296],
    },
    # order: [bearish, bullish, invalid, mixed, neutral]
    "news_sentiment": {
        "base": [1943, 3116, 0, 0, 6872],
        "F":    [1546, 3991, 0, 0, 6394],
        "T":    [1985, 2617, 0, 0, 7329],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    "fiqasa_sentiment": {
        "base": [0, 0, 467, 149, 557],
        "F":    [0, 0, 447, 86, 640],
        "T":    [0, 0, 448, 244, 481],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    "imdb_sklearn": {
        "base": [1, 0, 5582, 0, 4417],
        "F":    [1, 0, 5481, 0, 4518],
        "T":    [8, 0, 5689, 0, 4303],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    "sst2": {
        "base": [9, 0, 6813, 0, 3178],
        "F":    [2, 0, 6251, 0, 3747],
        "T":    [19, 0, 7339, 0, 2642],
    }
}

# --- compute entropies (bits) ---
rows = []
for name, data in datasets.items():
    base_ent = distribution_entropy(data["base"], base="bits")
    f_ent = distribution_entropy(data["F"], base="bits")
    t_ent = distribution_entropy(data["T"], base="bits")
    rows.append({
        "数据集": name,
        "Base分布熵 (bits)": round(base_ent, 6),
        "F分布熵 (bits)": round(f_ent, 6),
        "T分布熵 (bits)": round(t_ent, 6),
    })

df_entropy = pd.DataFrame(rows).sort_values("数据集").reset_index(drop=True)
print(df_entropy.to_string(index=False))
