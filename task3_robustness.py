
"""netflix-ml-tasks TSK3 task3_robustness""" 


import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
import netflix_core as nc

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
args = [a for a in sys.argv[1:] if a != "--quick"]
path = args[0] if args else "Dataset.csv"
FOLDS = 3 if QUICK else 5
REPEATS_STABILITY = 1 if QUICK else 2
REPEATS_OTHER = 1
MIN_GAIN = 0.01
try:
    df = nc.prep_t3(path).dropna(subset=["rating_group"]).reset_index(drop=True)
except ValueError as err:
    sys.exit(f"Cannot continue: {err}")
X = df[nc.T3_FEATURES + ["title"]]
y = df["rating_group"]
X_tr, _test_part, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=nc.RANDOM_STATE)
del _test_part
print(f"Training rows used: {len(X_tr)}   (cross-validation: {FOLDS} folds"
      f"{', quick mode' if QUICK else ''})")

best_name, best_params, source = nc.t3_config()
all_tuned = nc.t3_all_tuned() or ({best_name: best_params} if source == "tuned" else {})
if not all_tuned:
    print("NOTE: no tuned settings found (run task3_rating_classification.py first), so the defaults are used.")

def small(params, model="Random Forest"):
    """Quick mode uses smaller forests (only the Random Forest has trees)."""
    return {**params, "n_estimators": 40} if (QUICK and model == "Random Forest") else dict(params)

def one_group_f1(group):
    return make_scorer(lambda t, p: f1_score(t, p, labels=[group], average="macro", zero_division=0))

def severe_lenient(t, p):
    return nc.t3_safety(t, p)["severe_lenient_rate"]

SCORERS = {"macro_f1": "f1_macro", "accuracy": "accuracy",
           "family_f1": one_group_f1("Family"), "teens_f1": one_group_f1("Teens"),
           "mature_as_kids_family": make_scorer(severe_lenient)}

def evaluate(label, pipe, repeats):
    folds = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=repeats, random_state=nc.RANDOM_STATE)
    res = cross_validate(pipe, X_tr, y_tr, cv=folds, scoring=SCORERS, n_jobs=1)
    row = {"variant": label}
    for key in SCORERS:
        scores = res[f"test_{key}"]
        row[f"{key} mean"], row[f"{key} std"] = float(scores.mean()), float(scores.std())
    return row

def show(table, columns):
    shown = table[["variant"] + columns].copy()
    print(shown.round(3).to_string(index=False))

# comparison default and tuned stability
print(f"\n===== A. Is tuning really better? ({FOLDS} folds x {REPEATS_STABILITY} repeat(s), training part only) =====")
rows = []
for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
    for version, params in [("default", {}), ("tuned", all_tuned.get(name))]:
        if params is None:
            continue
        print(f"  {name} ({version}) ...", flush=True)
        row = evaluate(f"{name} ({version})", nc.t3_pipeline(nc.t3_model(name, small(params, name))), REPEATS_STABILITY)
        row.update({"model": name, "version": version})
        rows.append(row)
stability = pd.DataFrame(rows)
show(stability, ["macro_f1 mean", "macro_f1 std", "accuracy mean", "accuracy std"])
print("\nStability across folds:")
for name in stability["model"].unique():
    pair = stability[stability["model"] == name].set_index("version")
    if {"default", "tuned"} <= set(pair.index):
        diff = pair.loc["tuned", "macro_f1 mean"] - pair.loc["default", "macro_f1 mean"]
        spread = max(pair.loc["tuned", "macro_f1 std"], pair.loc["default", "macro_f1 std"])
        verdict = ("tuned scores higher beyond fold-to-fold variation" if diff > spread else
                   "tuned scores lower beyond fold-to-fold variation" if diff < -spread else
                   "the difference is within fold-to-fold variation")
        print(f"  {name}: tuned minus default = {diff:+.3f} macro-F1 (fold std about {spread:.3f}): {verdict}.")
stability.round(4).to_csv("task3_repeated_cv.csv", index=False)
# Family and Teens alternatives
rf = small(all_tuned.get("Random Forest", {}))
counts = y_tr.value_counts()
sqrt_weights = {g: float((len(y_tr) / (len(counts) * c)) ** 0.5) for g, c in counts.items()}
variants = {
    "current (balanced_subsample)": nc.t3_pipeline(nc.t3_model("Random Forest", rf)),
    "no class weights": nc.t3_pipeline(nc.t3_model("Random Forest", {**rf, "class_weight": None})),
    "balanced (whole sample)": nc.t3_pipeline(nc.t3_model("Random Forest", {**rf, "class_weight": "balanced"})),
    "square-root weights": nc.t3_pipeline(nc.t3_model("Random Forest", {**rf, "class_weight": sqrt_weights})),
    "ordinal (3 binary forests)": nc.t3_pipeline(nc.T3Ordinal(nc.t3_model("Random Forest", rf))),
    "+ title words": nc.t3_pipeline(nc.t3_model("Random Forest", rf), title_words=True),
}
print(f"\n===== B. Family and Teens: alternative designs ({FOLDS} folds, training part only) =====")
alt_rows = []
for label, pipe in variants.items():
    print(f"  {label} ...", flush=True)
    alt_rows.append(evaluate(label, pipe, REPEATS_OTHER))
alternatives = pd.DataFrame(alt_rows)
show(alternatives, ["macro_f1 mean", "macro_f1 std", "family_f1 mean", "teens_f1 mean", "mature_as_kids_family mean"])
current = alternatives.iloc[0]
others = alternatives.iloc[1:]
best_alt = others.loc[others["macro_f1 mean"].idxmax()]
gain = best_alt["macro_f1 mean"] - current["macro_f1 mean"]
print(f"\nBest alternative: {best_alt['variant']} ({gain:+.3f} macro-F1 vs the current setup).")
if gain >= max(MIN_GAIN, current["macro_f1 std"]):
    print("-> worth a closer look: it beats the current setup by more than one point and more than the variation "
          "between folds. Confirm it with repeated cross-validation before changing the pipeline.")
else:
    print("-> no alternative clearly beats the current setup (less than one point, or inside the variation between "
          "folds). Family and Teens stay hard with this data: report that as a limit, not as something tuning will fix.")
alternatives.round(4).to_csv("task3_alternatives.csv", index=False)
# Feature ablation
print(f"\n===== C. Ablation: what if obvious audience clues are taken away? ({FOLDS} folds, training part only) =====")
abl_rows = []
for variant in nc.T3_ABLATIONS:
    print(f"  {variant} ...", flush=True)
    abl_rows.append(evaluate(variant, nc.t3_ablation_pipeline(nc.t3_model("Random Forest", rf), variant), REPEATS_OTHER))
ablation = pd.DataFrame(abl_rows)
full = ablation.iloc[0]
ablation["macro_f1 change vs full"] = ablation["macro_f1 mean"] - full["macro_f1 mean"]
show(ablation, ["macro_f1 mean", "macro_f1 std", "macro_f1 change vs full", "accuracy mean"])
print("\nAblation results:")
for _, r in ablation.iloc[1:].iterrows():
    change = r["macro_f1 change vs full"]
    meaning = ("no clear drop" if change > -full["macro_f1 std"] else
               "clear drop")
    print(f"  {r['variant']:<26} {change:+.3f} macro-F1  -> {meaning}")
print(f"Audience genres removed here: {', '.join(nc.T3_AUDIENCE_GENRES)}. They are legitimate content labels known "
      "before a rating exists, not the answer itself; the question is only how much of the score rests on them.")
ablation.round(4).to_csv("task3_ablation.csv", index=False)
print("\nSaved: task3_repeated_cv.csv, task3_alternatives.csv, task3_ablation.csv")

