"""netflix-ml-tasks TSK3 task3_raw_vs_grouped"""


import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import netflix_core as nc


MIN_TITLES = 10
path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.csv"
try:
    df = nc.prep_t3(path).dropna(subset=["rating_group"])
except ValueError as err:
    sys.exit(f"Cannot continue: {err}")
counts = df["rating"].value_counts()
keep = counts[counts >= MIN_TITLES].index
if len(keep) < 2:
    sys.exit(f"Cannot continue: fewer than 2 raw ratings have at least {MIN_TITLES} titles.")
dropped = len(df) - df["rating"].isin(keep).sum()
df = df[df["rating"].isin(keep)].reset_index(drop=True)
try:                                                    # two raw ratings can still map to the same group
    nc.check_classes(df["rating_group"], minimum=1, what="audience group")
except ValueError as err:
    sys.exit(f"Cannot continue: after dropping rare ratings, {err}")
print(f"Raw rating classes used: {len(keep)}  ({dropped} titles in very rare ratings left out)")
print(counts.loc[keep].to_string())
X = df[nc.T3_FEATURES]
y_raw, y_grp = df["rating"], df["rating_group"]
X_tr, X_te, yr_tr, yr_te, yg_tr, yg_te = train_test_split(
    X, y_raw, y_grp, test_size=0.2, stratify=y_raw, random_state=nc.SEED)
def scores(true, pred):
    return {"accuracy": accuracy_score(true, pred),
            "macro_f1": f1_score(true, pred, average="macro", zero_division=0)}

raw_model = nc.t3_pipeline(nc.t3_model("Random Forest")).fit(X_tr, yr_tr)
grp_model = nc.t3_pipeline(nc.t3_model("Random Forest")).fit(X_tr, yg_tr)
base_raw = DummyClassifier(strategy="most_frequent").fit(X_tr, yr_tr)
base_grp = DummyClassifier(strategy="most_frequent").fit(X_tr, yg_tr)
raw_pred = raw_model.predict(X_te)
rows = [
    {"level": "raw ratings", "model": "Baseline (majority)", **scores(yr_te, base_raw.predict(X_te))},
    {"level": "raw ratings", "model": "RAW Random Forest", **scores(yr_te, raw_pred)},
    {"level": "audience groups", "model": "Baseline (majority)", **scores(yg_te, base_grp.predict(X_te))},
    {"level": "audience groups", "model": "RAW Random Forest, mapped to groups",
     **scores(yg_te, pd.Series(raw_pred).map(nc.RATING_TO_GROUP).values)},
    {"level": "audience groups", "model": "GROUPED Random Forest", **scores(yg_te, grp_model.predict(X_te))},
]
out = pd.DataFrame(rows).round(3)
print("\n", out.to_string(index=False))
out.to_csv("task3_raw_vs_grouped.csv", index=False)
gap, low, high = nc.paired_bootstrap(yg_te.values, grp_model.predict(X_te),
                                     pd.Series(raw_pred).map(nc.RATING_TO_GROUP).values,
                                     metric=lambda t, p: float(np.mean(t == p)), n=5000)
print(f"\nGROUPED minus RAW-mapped accuracy at group level: {gap:+.3f} (95% range {low:+.3f} to {high:+.3f})")
print("Raw ratings are harder to predict. Compare both approaches at the group level "
      "to see whether grouping improves accuracy.")
r = out.set_index(["level", "model"])
raw_acc, raw_base = r.loc[("raw ratings", "RAW Random Forest"), "accuracy"], r.loc[("raw ratings", "Baseline (majority)"), "accuracy"]
grp, mapped = r.loc[("audience groups", "GROUPED Random Forest")], r.loc[("audience groups", "RAW Random Forest, mapped to groups")]
print("\nSummary: " + nc.T3_TARGET_NOTE)
print(f"Predicting the {len(keep)} raw ratings directly reached {raw_acc:.1%} accuracy (majority baseline {raw_base:.1%}). "
      f"Judged on the four groups, the grouped model scored {grp['accuracy']:.1%} accuracy / {grp['macro_f1']:.3f} macro-F1 and the "
      f"raw-rating model mapped to groups {mapped['accuracy']:.1%} / {mapped['macro_f1']:.3f} "
      f"(accuracy gap {gap:+.1%}, 95% range {low:+.1%} to {high:+.1%}).")
ax = out[out["level"] == "audience groups"].set_index("model")[["accuracy", "macro_f1"]].plot(kind="bar", figsize=(9, 4.5))
ax.set_ylim(0, 1); ax.set_title("Task 3 - grouped vs raw ratings (judged on the 4 groups)")
plt.xticks(rotation=15, ha="right"); plt.tight_layout()
plt.savefig("task3_raw_vs_grouped.png", dpi=150); plt.close()
print("\nSaved: task3_raw_vs_grouped.csv, task3_raw_vs_grouped.png")
