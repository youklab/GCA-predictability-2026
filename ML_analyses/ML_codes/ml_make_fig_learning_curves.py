import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Where your step2 script wrote outputs
IN_DIR = "ml_results_tabular"
AGG_CSV = os.path.join(IN_DIR, "learning_curves_tabular_agg.csv")
RAW_CSV = os.path.join(IN_DIR, "learning_curves_tabular.csv")

OUT_DIR = "ml_figs"
os.makedirs(OUT_DIR, exist_ok=True)

# Model display names (optional)
MODEL_ORDER = ["logreg_onehot", "extratrees", "hgb"]
MODEL_LABEL = {
    "logreg_onehot": "LogReg (one-hot)",
    "extratrees": "ExtraTrees",
    "hgb": "HistGB",
}

def plot_metric(agg_df, metric_mean, metric_sem, ylabel, outname, ylim=None):
    plt.figure()
    for m in MODEL_ORDER:
        d = agg_df[agg_df["model"] == m].sort_values("N_train")
        if len(d) == 0:
            continue
        x = d["N_train"].to_numpy()
        y = d[metric_mean].to_numpy()
        e = d[metric_sem].to_numpy()

        plt.errorbar(
            np.log10(x),
            y,
            yerr=e,
            marker="o",
            linestyle="-",
            capsize=3,
            label=MODEL_LABEL.get(m, m),
        )

    plt.xlabel(r"$\log_{10}(N_{\mathrm{train}})$")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, outname + ".png")
    pdf_path = os.path.join(OUT_DIR, outname + ".pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def main():
    agg = pd.read_csv(AGG_CSV)
    raw = pd.read_csv(RAW_CSV)

    # Sanity print
    print("Models:", sorted(agg["model"].unique()))
    print("Train sizes:", sorted(agg["N_train"].unique()))
    print("Test metrics columns:", [c for c in agg.columns if "mean" in c])

    # 1) Balanced accuracy learning curve
    plot_metric(
        agg,
        metric_mean="mean_bal_acc",
        metric_sem="sem_bal_acc",
        ylabel="Balanced accuracy (test)",
        outname="learning_curve_balanced_accuracy",
        ylim=(0.45, 0.55),
    )

    # 2) ROC-AUC learning curve
    plot_metric(
        agg,
        metric_mean="mean_auc",
        metric_sem="sem_auc",
        ylabel="ROC-AUC (test)",
        outname="learning_curve_auc",
        ylim=(0.45, 0.55),
    )

    # Optional: accuracy learning curve (shows class-imbalance artifact)
    plot_metric(
        agg,
        metric_mean="mean_acc",
        metric_sem="sem_acc",
        ylabel="Accuracy (test)",
        outname="learning_curve_accuracy",
        ylim=(0.0, 1.0),
    )

    # Also save a compact table of the agg results you can paste into SI
    agg_out = os.path.join(OUT_DIR, "learning_curves_table_for_SI.csv")
    agg.sort_values(["model", "N_train"]).to_csv(agg_out, index=False)
    print(f"Saved: {agg_out}")

if __name__ == "__main__":
    main()