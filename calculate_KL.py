import numpy as np
import pandas as pd

# KL divergence in bits (log2)
def kl_divergence_bits(q_counts, p_counts, epsilon=1e-12):
    p_probs = np.array(p_counts, dtype=float)
    q_probs = np.array(q_counts, dtype=float)
    p_probs = p_probs / p_probs.sum()
    q_probs = q_probs / q_probs.sum()
    p_probs = np.where(p_probs == 0, epsilon, p_probs)
    q_probs = np.where(q_probs == 0, epsilon, q_probs)
    return np.sum(q_probs * np.log2(q_probs / p_probs))

# ===== counts replaced by qwen2.5-3B =====
# Each dataset uses the label order shown in the comment.
datasets = {
    # order: [invalid, mixed, negative, neutral, positive]
    'imdb_sentiment': {
        'base': [0, 0, 5309, 0, 4691],
        'F':    [0, 0, 5149, 0, 4851],
        'T':    [0, 0, 5435, 0, 4565],
    },
    # order: [depression, invalid, normal]
    'mental_sentiment': {
        'base': [15621, 0, 16126],
        'F':    [17428, 0, 14319],
        'T':    [14451, 0, 17296],
    },
    # order: [bearish, bullish, invalid, mixed, neutral]
    'news_sentiment': {
        'base': [1943, 3116, 0, 0, 6872],
        'F':    [1546, 3991, 0, 0, 6394],
        'T':    [1985, 2617, 0, 0, 7329],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    'fiqasa_sentiment': {
        'base': [0, 0, 467, 149, 557],
        'F':    [0, 0, 447, 86, 640],
        'T':    [0, 0, 448, 244, 481],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    'imdb_sklearn': {
        'base': [1, 0, 5582, 0, 4417],
        'F':    [1, 0, 5481, 0, 4518],
        'T':    [8, 0, 5689, 0, 4303],
    },
    # order: [invalid, mixed, negative, neutral, positive]
    'sst2': {
        'base': [9, 0, 6813, 0, 3178],
        'F':    [2, 0, 6251, 0, 3747],
        'T':    [19, 0, 7339, 0, 2642],
    }
}

# Compute KL(F||Base) and KL(T||Base)
rows = []
for name, d in datasets.items():
    f_kl = kl_divergence_bits(d['F'], d['base'])
    t_kl = kl_divergence_bits(d['T'], d['base'])
    rows.append({
        "数据集": name,
        "F模型 KL散度 (bits)": round(f_kl, 6),
        "T模型 KL散度 (bits)": round(t_kl, 6),
    })

df = pd.DataFrame(rows).sort_values("数据集").reset_index(drop=True)
print(df.to_string(index=False, float_format='%.6f'))
