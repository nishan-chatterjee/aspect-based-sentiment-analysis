import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent blocking
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set premium visualization styling
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.figsize': (12, 7),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})
palette = {"slovenian": "#3498db", "serbian": "#e74c3c"}
LANGUAGE_LABELS = {
    "slovenian": "Slovenian",
    "serbian": "Serbo-Croatian",
}
DISPLAY_PALETTE = {
    "Slovenian": palette["slovenian"],
    "Serbo-Croatian": palette["serbian"],
}


def language_label(language):
    return LANGUAGE_LABELS.get(language, str(language).capitalize())

# Set up paths relative to this script directory
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
REVIEWS_DIR = BASE_DIR / "reviews" / "slavic_specific" / "masked"
UNCERTAINTY_DIR = BASE_DIR / "uncertainty" / "slavic_specific_masked"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Base Directory: {BASE_DIR}")
print(f"Loading data from: {REVIEWS_DIR} and {UNCERTAINTY_DIR}")

data = []

for lang in ["slovenian", "serbian"]:
    lang_unc_dir = UNCERTAINTY_DIR / lang
    if not lang_unc_dir.exists():
        print(f"Directory not found: {lang_unc_dir}")
        continue
        
    for pct_dir in sorted(lang_unc_dir.glob("pct_*")):
        pct_str = pct_dir.name.replace("pct_", "").replace("p", ".")
        pct = float(pct_str)
        
        # Load MC metrics
        metrics_file = pct_dir / f"{lang}_test_complete_metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
            
        # Load training metadata for train/val sizes and fold performance
        train_dir = REVIEWS_DIR / lang / pct_dir.name
        summary_file = train_dir / "test_metrics_summary.json"
        
        train_size = 0
        val_size = 0
        
        f1_macro_folds = []
        qwk_folds = []
        acc_folds = []
        
        f1_neg_folds = []
        f1_neu_folds = []
        f1_pos_folds = []
        
        if summary_file.exists():
            with open(summary_file) as f:
                train_summary = json.load(f)
                
            subs = train_summary.get("subset_summaries", {}).get("0", {})
            train_size = subs.get("train", {}).get("selected_count", 0)
            val_size = subs.get("val", {}).get("selected_count", 0)
            
            # Extract metrics for individual folds
            for m_key in ["model_0", "model_1", "model_2"]:
                if m_key in train_summary:
                    m_data = train_summary[m_key]
                    if isinstance(m_data, dict) and "error" not in m_data:
                        f1_macro_folds.append(m_data.get("f1_macro", np.nan))
                        qwk_folds.append(m_data.get("qwk", np.nan))
                        acc_folds.append(m_data.get("accuracy", np.nan))
                        
                        class_rep = m_data.get("per_class_report", {})
                        if class_rep:
                            f1_neg_folds.append(class_rep.get("Negative (0)", {}).get("f1-score", np.nan))
                            f1_neu_folds.append(class_rep.get("Neutral (1)", {}).get("f1-score", np.nan))
                            f1_pos_folds.append(class_rep.get("Positive (2)", {}).get("f1-score", np.nan))
                            
        # Load predictions to compute calibration error and uncertainty metrics
        preds_file = pct_dir / f"{lang}_test_complete.json"
        ece = np.nan
        brier = np.nan
        mean_predictive_entropy = np.nan
        mean_mutual_information = np.nan
        mean_confidence = np.nan
        
        if preds_file.exists():
            with open(preds_file) as f:
                preds_data = json.load(f)
                
            y_true = []
            y_prob = []
            records_list = []
            if isinstance(preds_data, dict):
                for val in preds_data.values():
                    if isinstance(val, list):
                        records_list.extend(val)
            else:
                records_list = preds_data
                
            pred_entropies = []
            mutual_infos = []
            confidences = []
            
            for item in records_list:
                gold = item.get("sentiment")
                if gold not in [-1, 0, 1]:
                    continue
                
                # convert to 0, 1, 2
                gold_idx = {-1: 0, 0: 1, 1: 2}[gold]
                probs = item.get("slavic_specific/masked/probabilities", {})
                prob_vals = [probs.get(k, 0) for k in ["Negative", "Neutral", "Positive"]]
                
                y_true.append(gold_idx)
                y_prob.append(prob_vals)
                
                unc_data = item.get("slavic_specific/masked/uncertainty", {})
                if unc_data:
                    if "predictive_entropy" in unc_data:
                        pred_entropies.append(unc_data["predictive_entropy"])
                    if "mutual_information" in unc_data:
                        mutual_infos.append(unc_data["mutual_information"])
                    if "confidence_score" in unc_data:
                        confidences.append(unc_data["confidence_score"])
                        
            if len(y_true) > 0:
                y_true = np.array(y_true)
                y_prob = np.array(y_prob)
                
                # Brier score
                y_true_onehot = np.zeros_like(y_prob)
                y_true_onehot[np.arange(len(y_true)), y_true] = 1
                brier = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
                
                # ECE
                confidences_arr = np.max(y_prob, axis=1)
                predictions = np.argmax(y_prob, axis=1)
                accuracies = (predictions == y_true)
                
                n_bins = 10
                bins = np.linspace(0, 1, n_bins + 1)
                bin_indices = np.digitize(confidences_arr, bins) - 1
                
                ece_val = 0
                for i in range(n_bins):
                    mask = bin_indices == i
                    if np.sum(mask) > 0:
                        bin_acc = np.mean(accuracies[mask])
                        bin_conf = np.mean(confidences_arr[mask])
                        ece_val += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(confidences_arr)
                ece = ece_val
                
            mean_predictive_entropy = np.mean(pred_entropies) if pred_entropies else np.nan
            mean_mutual_information = np.mean(mutual_infos) if mutual_infos else np.nan
            mean_confidence = np.mean(confidences) if confidences else np.nan
            
        data.append({
            "language": lang,
            "percentage": pct,
            "train_size": train_size,
            "val_size": val_size,
            "f1_macro": metrics.get("f1_macro"),
            "qwk": metrics.get("qwk"),
            "accuracy": metrics.get("accuracy"),
            "ece": ece,
            "brier": brier,
            "f1_macro_mean": np.nanmean(f1_macro_folds) if f1_macro_folds else np.nan,
            "f1_macro_std": np.nanstd(f1_macro_folds, ddof=1) if len(f1_macro_folds) > 1 else 0.0,
            "qwk_mean": np.nanmean(qwk_folds) if qwk_folds else np.nan,
            "qwk_std": np.nanstd(qwk_folds, ddof=1) if len(qwk_folds) > 1 else 0.0,
            "accuracy_mean": np.nanmean(acc_folds) if acc_folds else np.nan,
            "accuracy_std": np.nanstd(acc_folds, ddof=1) if len(acc_folds) > 1 else 0.0,
            "f1_neg_mean": np.nanmean(f1_neg_folds) if f1_neg_folds else np.nan,
            "f1_neu_mean": np.nanmean(f1_neu_folds) if f1_neu_folds else np.nan,
            "f1_pos_mean": np.nanmean(f1_pos_folds) if f1_pos_folds else np.nan,
            "mean_predictive_entropy": mean_predictive_entropy,
            "mean_mutual_information": mean_mutual_information,
            "mean_confidence": mean_confidence
        })

df = pd.DataFrame(data)
df = df.sort_values(["language", "percentage"])

# --- Visualizations ---
print("\nGenerating and saving plots to minimum-viable-set/plots/...")

# 1. Performance Scaling
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for i, (metric, label) in enumerate([("f1_macro_mean", "Macro F1"), ("qwk_mean", "Quadratic Weighted Kappa")]):
    ax = axes[i]
    for lang, color in [("slovenian", "#3498db"), ("serbian", "#e74c3c")]:
        sub_df = df[df["language"] == lang]
        x = sub_df["percentage"]
        y = sub_df[metric]
        std = sub_df[metric.replace("_mean", "_std")]
        
        ax.plot(x, y, marker="o", label=language_label(lang), color=color)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)
        
    ax.set_title(f"{label} vs. Training Data Percentage")
    ax.set_xlabel("Percentage of Data (%)")
    ax.set_ylabel(label)
    ax.set_ylim(0.4, 0.9)
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "performance_scaling.png", dpi=150)
plt.close()

# 2. Per-Class F1 Scaling
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for i, lang in enumerate(["slovenian", "serbian"]):
    ax = axes[i]
    sub_df = df[df["language"] == lang]
    x = sub_df["percentage"]
    
    ax.plot(x, sub_df["f1_neg_mean"], marker="s", label="Negative F1 (Minority)", color="#e74c3c")
    ax.plot(x, sub_df["f1_neu_mean"], marker="^", label="Neutral F1", color="#f1c40f")
    ax.plot(x, sub_df["f1_pos_mean"], marker="o", label="Positive F1", color="#2ecc71")
    
    ax.set_title(f"{language_label(lang)} Per-Class F1 Scaling")
    ax.set_xlabel("Percentage of Data (%)")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0.2, 1.0)
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "per_class_scaling.png", dpi=150)
plt.close()

# 3. Calibration
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for lang, color in [("slovenian", "#3498db"), ("serbian", "#e74c3c")]:
    sub_df = df[df["language"] == lang]
    axes[0].plot(sub_df["percentage"], sub_df["ece"], marker="o", label=language_label(lang), color=color)
    axes[1].plot(sub_df["percentage"], sub_df["brier"], marker="o", label=language_label(lang), color=color)
    
axes[0].set_title("Expected Calibration Error (ECE) vs. Percentage")
axes[0].set_xlabel("Percentage of Data (%)")
axes[0].set_ylabel("ECE (lower is better)")
axes[0].legend()

axes[1].set_title("Brier Score vs. Percentage")
axes[1].set_xlabel("Percentage of Data (%)")
axes[1].set_ylabel("Brier Score (lower is better)")
axes[1].legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "uncertainty_calibration.png", dpi=150)
plt.close()

# 4. Uncertainty Scaling
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for lang, color in [("slovenian", "#3498db"), ("serbian", "#e74c3c")]:
    sub_df = df[df["language"] == lang]
    axes[0].plot(sub_df["percentage"], sub_df["mean_predictive_entropy"], marker="o", label=language_label(lang), color=color)
    axes[1].plot(sub_df["percentage"], sub_df["mean_mutual_information"], marker="o", label=language_label(lang), color=color)
    
axes[0].set_title("Total Uncertainty (Predictive Entropy) vs. Percentage")
axes[0].set_xlabel("Percentage of Data (%)")
axes[0].set_ylabel("Predictive Entropy (nats)")
axes[0].legend()

axes[1].set_title("Parameter Uncertainty (Mutual Information) vs. Percentage")
axes[1].set_xlabel("Percentage of Data (%)")
axes[1].set_ylabel("Mutual Information (nats)")
axes[1].legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "uncertainty_scaling.png", dpi=150)
plt.close()

# 5. Combined Performance Scaling By Language
PERFORMANCE_COLORS = {
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "positive": "#2ca02c",
    "macro_f1": "#f1c40f",
    "qwk": "#ff7f0e",
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
for ax, lang in zip(axes, ["slovenian", "serbian"]):
    sub_df = df[df["language"] == lang]
    x = sub_df["percentage"]

    for metric, std_metric, label, color in [
        ("f1_macro_mean", "f1_macro_std", "Macro F1", PERFORMANCE_COLORS["macro_f1"]),
        ("qwk_mean", "qwk_std", "QWK", PERFORMANCE_COLORS["qwk"]),
    ]:
        y = sub_df[metric]
        std = sub_df[std_metric]
        ax.plot(x, y, marker="o", label=label, color=color, linewidth=2.8)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16)

    ax.plot(x, sub_df["f1_neg_mean"], marker="s", label="Negative F1", color=PERFORMANCE_COLORS["negative"], linewidth=2.2)
    ax.plot(x, sub_df["f1_neu_mean"], marker="^", label="Neutral F1", color=PERFORMANCE_COLORS["neutral"], linewidth=2.2)
    ax.plot(x, sub_df["f1_pos_mean"], marker="D", label="Positive F1", color=PERFORMANCE_COLORS["positive"], linewidth=2.2)
    ax.set_title(f"{language_label(lang)} Performance Scaling")
    ax.set_xlabel("Percentage of Data (%)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig(PLOTS_DIR / "combined_performance_by_language.png", dpi=150)
plt.close()

# 6. Combined Calibration And Uncertainty Diagnostics
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
diagnostic_panels = [
    ("ece", "Expected Calibration Error (ECE)", "ECE (lower is better)"),
    ("mean_predictive_entropy", "Total Uncertainty", "Predictive Entropy (nats)"),
    ("brier", "Brier Score", "Brier Score (lower is better)"),
    ("mean_mutual_information", "Parameter Uncertainty", "Mutual Information (nats)"),
]
for ax, (metric, title, ylabel) in zip(axes.flat, diagnostic_panels):
    for lang, color in [("slovenian", "#3498db"), ("serbian", "#e74c3c")]:
        sub_df = df[df["language"] == lang]
        ax.plot(sub_df["percentage"], sub_df[metric], marker="o", label=language_label(lang), color=color)
    ax.set_title(f"{title} vs. Training Data Percentage")
    ax.set_xlabel("Percentage of Data (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "combined_calibration_uncertainty_grid.png", dpi=150)
plt.close()


# --- MVS Analytical Methods ---

def find_knee_point(x, y, scale_x='log'):
    x = np.array(x)
    y = np.array(y)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    if scale_x == 'log':
        x_scaled = np.log10(x)
    else:
        x_scaled = x
        
    x_norm = (x_scaled - x_scaled.min()) / (x_scaled.max() - x_scaled.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    
    numerator = np.abs((p2[1] - p1[1]) * x_norm - (p2[0] - p1[0]) * y_norm + p2[0] * p1[1] - p2[1] * p1[0])
    denominator = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    distances = numerator / (denominator + 1e-8)
    
    if len(distances) <= 2:
        return x[0], y[0], distances
        
    knee_idx = np.argmax(distances[1:-1]) + 1
    return x[knee_idx], y[knee_idx], distances

def find_mvs_threshold(df_lang, metric="f1_macro", threshold=0.02):
    full_perf = df_lang[df_lang["percentage"] == 100][metric].values
    if len(full_perf) == 0: return None
    full_perf = full_perf[0]
    target = full_perf * (1 - threshold)
    valid = df_lang[df_lang[metric] >= target]
    return valid.iloc[0] if len(valid) > 0 else None

def find_mvs_1se(df_lang, metric_mean="f1_macro_mean", metric_std="f1_macro_std", n_folds=3):
    full_row = df_lang[df_lang["percentage"] == 100]
    if len(full_row) == 0: return None
    full_mean = full_row[metric_mean].values[0]
    full_std = full_row[metric_std].values[0]
    full_se = full_std / np.sqrt(n_folds)
    target = full_mean - full_se
    valid = df_lang[df_lang[metric_mean] >= target]
    return (valid.iloc[0], target, full_se) if len(valid) > 0 else None

def find_uncertainty_plateau(df_lang, metric="mean_mutual_information", relative_threshold=1.10):
    full_row = df_lang[df_lang["percentage"] == 100]
    if len(full_row) == 0: return None
    full_val = full_row[metric].values[0]
    target = full_val * relative_threshold
    valid = df_lang[df_lang[metric] <= target]
    return valid.iloc[0] if len(valid) > 0 else None


results_list = []

print("\n" + "="*50)
print("MINIMUM VIABLE SET (MVS) REPORT")
print("="*50)
print(
    "\nRule definitions:\n"
    "- Geometric Knee: the percentage at the largest bend in the scaling curve after normalizing the axes; it marks the point where additional data starts giving diminishing returns.\n"
    "- Statistical 1-SE Rule: the smallest percentage whose mean score is within one standard error of the 100% data score; it favors the simpler/smaller training set when its performance is statistically close to full data.\n"
    "- Epistemic Plateau: the smallest percentage whose MC-dropout mutual information is within 10% of the full-data value; it estimates where parameter uncertainty has mostly stabilized.\n"
)

for lang in ["slovenian", "serbian"]:
    df_lang = df[df["language"] == lang]
    print(f"\n--- {language_label(lang).upper()} ---")
    
    # 1. Static Relative Thresholds
    for th in [0.02, 0.05, 0.10]:
        mvs = find_mvs_threshold(df_lang, "f1_macro", th)
        if mvs is not None:
            print(f"Threshold F1-Macro within {th*100:g}% of full: {mvs['percentage']}% ({int(mvs['train_size'])} examples)")
            results_list.append({"Language": language_label(lang), "Method": f"F1-Macro Within {th*100:g}%", "MVS (%)": mvs["percentage"], "Examples": mvs["train_size"]})
            
    # 2. Geometric Knee Point
    knee_pct, knee_f1, _ = find_knee_point(df_lang["percentage"], df_lang["f1_macro_mean"], 'log')
    knee_row = df_lang[df_lang["percentage"] == knee_pct].iloc[0]
    print(f"Geometric Knee Point (F1-Macro):       {knee_pct}% ({int(knee_row['train_size'])} examples) | Val: {knee_f1:.4f}")
    results_list.append({"Language": language_label(lang), "Method": "Geometric Knee (F1-Macro)", "MVS (%)": knee_pct, "Examples": knee_row["train_size"]})
    
    # 3. Statistical 1-SE Rule
    mvs_1se = find_mvs_1se(df_lang, "f1_macro_mean", "f1_macro_std")
    if mvs_1se is not None:
        best_row, target, se = mvs_1se
        print(f"Statistical 1-SE Rule MVS:             {best_row['percentage']}% ({int(best_row['train_size'])} examples) | Target F1: >= {target:.4f} (SE={se:.4f})")
        results_list.append({"Language": language_label(lang), "Method": "Statistical 1-SE Rule", "MVS (%)": best_row["percentage"], "Examples": best_row["train_size"]})
        
    # 4. Epistemic Uncertainty Plateau
    mvs_up = find_uncertainty_plateau(df_lang, "mean_mutual_information", 1.10)
    if mvs_up is not None:
        print(f"Epistemic Plateau MVS (10% Tol):       {mvs_up['percentage']}% ({int(mvs_up['train_size'])} examples) | MI: {mvs_up['mean_mutual_information']:.5f}")
        results_list.append({"Language": language_label(lang), "Method": "Epistemic Plateau (10% Tol)", "MVS (%)": mvs_up["percentage"], "Examples": mvs_up["train_size"]})

df_res = pd.DataFrame(results_list)

# 5. Save comparison bar plot
plt.figure(figsize=(14, 8))
sns.barplot(data=df_res, x="MVS (%)", y="Method", hue="Language", palette=DISPLAY_PALETTE)
plt.title("Comparison of Minimum Viable Set (MVS) Detection Methods")
plt.xlabel("Recommended MVS Data Percentage (%)")
plt.ylabel("Method")
plt.xlim(0, 110)
plt.legend(title="Language")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "mvs_comparison.png", dpi=150)
plt.close()

print("\n" + "="*50)
print("SUMMARY MVS RECOMMENDATIONS TABLE")
print("="*50)
df_res_pivot = df_res.pivot(index="Method", columns="Language", values=["MVS (%)", "Examples"])
print(df_res_pivot)
print("="*50)
print("All plots saved in minimum-viable-set/plots/ directory.")
print("Analysis complete.")
