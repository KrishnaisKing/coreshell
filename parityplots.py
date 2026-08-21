"""
Parity plots for the three trained targets (hysteresis, log ON/OFF ratio,
log retention). Same style as the friend's "Model Parity" plot, so you
can compare directly -- yours should show real scatter around the 1:1
line, not a flat-ceiling collapse like the leaked version.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

targets = {
    "hysteresis_window_V": "Hysteresis Window (V)",
    "log10_on_off_ratio": "log10(ON/OFF Ratio)",
    "log10_retention_tau_s": "log10(Retention Time, s)",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (target, label) in zip(axes, targets.items()):
    df = pd.read_csv(f"predictions_{target}.csv")
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    r2 = r2_score(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolor="k", linewidth=0.3)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Ideal 1:1 Parity")

    ax.set_xlabel(f"Ground Truth ({label})")
    ax.set_ylabel(f"ML Prediction ({label})")
    ax.set_title(f"{label}\nR² = {r2:.3f}  (n={len(df)})")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("parity_comparison.png", dpi=150)
print("Saved parity_comparison.png")
plt.show()