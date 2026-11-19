"""netflix-ml-tasks TSK2: task2_content_type_prediction.py""" 

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import netflix_core as nc


path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.csv"
raw = nc.read_data(path)
print("Shape:", raw.shape)
print("\nMissing values per column (blank or 'Not Given'):")
print(raw.astype(str).apply(lambda c: c.str.strip()).isin(["Not Given", "nan", ""]).sum().to_string())
print("\nTarget distribution:\n", raw["type"].value_counts(normalize=True).round(3))

try:
    df, y = nc.prep_t2(path, fill_years=False)
except ValueError as err:
    sys.exit(f"Cannot continue: {err}")
print(f"\nRows in the file: {len(raw)}   rows used: {len(df)}   (left out: {len(raw) - len(df)})")
for note in df.attrs.get("issues", []):
    print("  NOTE:", note)# to avoid leakage
df["duration_value"] = df["duration"].str.extract(r"(\d+)")[0].astype(float)
df["is_seasons"] = df["duration"].fillna("").str.contains("Season").astype(int)
NUM_CLEAN = nc.T2_NUM
NUM_LEAKY = NUM_CLEAN + ["duration_value", "is_seasons"]
DATE_COLS = ["year_added", "month_added", "years_to_netflix"]
NUM_NO_DATES = [c for c in NUM_CLEAN if c not in DATE_COLS]
SHOW = ["cv_macro_f1", "cv_macro_f1_std", "accuracy", "balanced_accuracy", "macro_f1", "f1_TV",
        "recall_Movie", "recall_TV", "roc_auc"]
experiments = {
    "A: with leakage (duration)": NUM_LEAKY,
    "B: leak-free": NUM_CLEAN,
    "C: no date-added features": NUM_NO_DATES,
}
studies, frames = {}, []
for name, cols in experiments.items():
    study = nc.t2_study(df, y, cols)
    studies[name] = study
    table = study["table"]
    print(f"\n===== {name} =====")
    print(f"5-fold CV: {study['n_train']} training rows; test: {study['n_test']} rows")
    print(table[SHOW].round(4).to_string())
    if name.startswith("A"):
        print("Duration leaks the label; these scores are for comparison only.")
    else:
        print(f"-> selected by training CV: {study['selected']}")
        print("   " + study["selection_note"])
    frame = table.reset_index()
    frame.insert(0, "experiment", name)
    frame.insert(2, "selected", frame["model"] == study["selected"] if not name.startswith("A") else False)
    frames.append(frame)
all_res = pd.concat(frames, ignore_index=True)
all_res.round(4).to_csv("task2_model_comparison.csv", index=False)
b = studies["B: leak-free"]
best = b["selected"]
nc.t2_save_selection(best, b["table"].loc[best], b["selection_note"])
print(f"\nSaved model selection to {nc.T2_SELECTION_FILE}.")
report = nc.t2_class_report(b["y_test"], b["preds"][best])
bt = b["table"].loc[best]
print(f"\n===== {best} on the test part: both classes =====")
print(report.round(3).to_string())
print(f"Balanced accuracy {bt['balanced_accuracy']:.4f}   macro-F1 {bt['macro_f1']:.4f}   "
      f"accuracy {bt['accuracy']:.4f}   (majority baseline accuracy {b['table'].loc['Baseline (majority)', 'accuracy']:.4f})")
with open("task2_classification_report.txt", "w") as f:
    f.write(f"Model chosen by cross-validation on the training part (experiment B): {best}\n"
            f"{b['selection_note']}\n\n{report.round(3).to_string()}\n\n"
            f"balanced accuracy {bt['balanced_accuracy']:.4f}  macro-F1 {bt['macro_f1']:.4f}  accuracy {bt['accuracy']:.4f}\n")
print(f"\n===== Feature ablation: {best} (training CV) =====")
ablation = nc.t2_ablation(df, y, best)
print(ablation.round(4).to_string())
print("Compare each macro-F1 drop with the variation between CV folds.")
for variant, row in ablation.iloc[1:].iterrows():
    wobble = ablation.iloc[0]["macro_f1 std"]
    verdict = "no clear loss" if row["macro_f1 change vs full"] > -wobble else "a clear loss: the model relies on this"
    print(f"  {variant:<30} {row['macro_f1 change vs full']:+.4f} macro-F1  -> {verdict}")
ablation.round(4).to_csv("task2_ablation.csv")
real_table = all_res[all_res["model"] != "Baseline (majority)"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title, base_col in [(axes[0], "accuracy", "Accuracy", "accuracy"),
                                 (axes[1], "balanced_accuracy", "Balanced accuracy (average recall of both types)", None)]:
    real_table.pivot(index="model", columns="experiment", values=col).plot(kind="bar", ax=ax)
    base = all_res[(all_res["model"] == "Baseline (majority)") & (all_res["experiment"] == "B: leak-free")][col].iloc[0]
    ax.axhline(base, color="grey", linestyle="--", linewidth=1)
    ax.text(0.01, base + 0.01, f"majority baseline {base:.2f}", color="grey", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_ylim(0, 1.05); ax.set_title(f"Task 2 - {title}", fontsize=10); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig("task2_accuracy_comparison.png", dpi=150); plt.close()
cm = confusion_matrix(b["y_test"], b["preds"][best])
ConfusionMatrixDisplay(cm, display_labels=["Movie", "TV Show"]).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best} (leak-free)\nchosen by cross-validation, not by the test set", fontsize=10)
plt.tight_layout(); plt.savefig("task2_confusion_matrix.png", dpi=150); plt.close()
print(f"\nChosen model (experiment B): {best}")
print("Saved: task2_model_comparison.csv, task2_classification_report.txt, task2_ablation.csv, task2_selected_model.json,\n"
      "       task2_accuracy_comparison.png, task2_confusion_matrix.png")
print("Note: earlier experiments used this test set for model selection, so reported test scores may be optimistic.")


