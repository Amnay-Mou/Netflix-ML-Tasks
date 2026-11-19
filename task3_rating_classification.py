"""netflix-ml-tasks TSK3 task3_rating_classification"""


import json 
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
import netflix_core as nc


warnings.filterwarnings("ignore", category=UserWarning)
BOOTSTRAP_REPS = 5000
def metrics(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    return pred, {
        "accuracy": accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_te, pred, average="weighted", zero_division=0),}

def clean(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.csv"
    try:
        df_all = nc.prep_t3(path)
    except ValueError as err:
        sys.exit(f"Cannot continue: {err}")
    print("Titles:", len(df_all))
    for note in df_all.attrs.get("issues", []):
        print("NOTE:", note)
    print("\nRaw rating counts:\n", df_all["rating"].value_counts(dropna=False).to_string())
    df = df_all.dropna(subset=["rating_group"]).reset_index(drop=True)
    rep = df_all.attrs["rating_report"]
    print(f"\nLeft out of Task 3: {len(df_all) - len(df)} titles")
    print(f"  expected - NR/UR (no audience meaning): {rep['not_rated']}")
    print(f"  expected - rating missing             : {rep['missing']}")
    print(f"  UNEXPECTED rating values              : {sum(rep['unexpected'].values())}"
          + (f"  {rep['unexpected']}" if rep["unexpected"] else ""))
    genre_counts = pd.Series([g.lower() for s in df["listed_in"] for g in nc.split_items(s)]).value_counts()
    print(f"\nGenres used as features: {len(genre_counts)} "
          f"({int((genre_counts < 5).sum())} of them appear in fewer than 5 titles; they are kept, not dropped)")
    group_counts = df["rating_group"].value_counts().reindex(nc.GROUP_ORDER).dropna().astype(int)
    print("\nAudience groups:")
    for g, c in group_counts.items():
        print(f"  {g:<7} {c:>5}  ({c / len(df):.1%})   <- {nc.GROUP_INFO[g]}")
    baseline = group_counts.max() / len(df)
    print(f"\nBaseline (always guess the biggest group): {baseline:.1%}")
    print("\nGroup definitions: " + nc.T3_TARGET_NOTE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df_all["rating"].value_counts().plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Raw ratings"); axes[0].set_ylabel("Titles")
    group_counts.plot(kind="bar", ax=axes[1], color="#DD8452")
    axes[1].set_title("Audience groups used as the target")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig("task3_class_distribution.png", dpi=150); plt.close()

    # Train/test split
    X, y = df[nc.T3_FEATURES], df["rating_group"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=nc.RANDOM_STATE)
    print(f"\nTrain: {len(X_tr)} rows   Test: {len(X_te)} rows")
    # Default models
    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": nc.t3_model("Logistic Regression"),
        "Decision Tree": nc.t3_model("Decision Tree"),
        "Random Forest": nc.t3_model("Random Forest"),
    }
    cv = StratifiedKFold(4, shuffle=True, random_state=nc.RANDOM_STATE)
    rows = []
    for name, model in models.items():
        pipe = nc.t3_pipeline(model).fit(X_tr, y_tr)
        _, m = metrics(pipe, X_te, y_te)
        cv_default = cross_val_score(nc.t3_pipeline(clone(model)), X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
        rows.append({"model": name, "version": "default", **m, "cv_macro_f1": cv_default, "best_params": ""})
    res_default = pd.DataFrame(rows).round(4)
    print("\n===== Step 3: default models (test set; cv_macro_f1 = cross-validation on the training set) =====")
    print(res_default[["model", "accuracy", "macro_f1", "weighted_f1", "cv_macro_f1"]].to_string(index=False))
    # Hyperparameter tuning
    searches = {
        "Logistic Regression": GridSearchCV(
            nc.t3_pipeline(nc.t3_model("Logistic Regression")),
            {"model__C": [0.1, 0.3, 1, 3, 10]},
            cv=cv, scoring="f1_macro"),
        "Decision Tree": GridSearchCV(
            nc.t3_pipeline(nc.t3_model("Decision Tree")),
            {"model__max_depth": [4, 6, 8, 12, 16, None],
             "model__min_samples_leaf": [1, 5, 10, 20],
             "model__criterion": ["gini", "entropy"]},
            cv=cv, scoring="f1_macro"),
        "Random Forest": RandomizedSearchCV(
            nc.t3_pipeline(nc.t3_model("Random Forest")),
            {"model__n_estimators": [200, 300, 500],
             "model__max_depth": [10, 20, 30, None],
             "model__min_samples_leaf": [1, 2, 4, 8],
             "model__max_features": ["sqrt", "log2", 0.3]},
            n_iter=12, cv=cv, scoring="f1_macro", random_state=nc.RANDOM_STATE),
    }
    rows, tuned = [], {}
    print("\n===== Step 4: tuning (cross-validated on the training set) =====")
    for name, search in searches.items():
        print(f"Tuning {name} ...", flush=True)
        search.fit(X_tr, y_tr)
        pred, m = metrics(search.best_estimator_, X_te, y_te)
        params = {k.replace("model__", ""): clean(v) for k, v in search.best_params_.items()}
        tuned[name] = (search, pred, params)
        rows.append({"model": name, "version": "tuned", **m,
                     "cv_macro_f1": search.best_score_, "best_params": json.dumps(params)})
        print(f"   best CV macro-F1 = {search.best_score_:.4f}   params = {params}")
    res_tuned = pd.DataFrame(rows).round(4)
    # Model comparison
    comparison = pd.concat([res_default, res_tuned], ignore_index=True)
    comparison.to_csv("task3_model_comparison.csv", index=False)
    print("\n===== Step 5: default vs tuned (test set) =====")
    print(comparison[["model", "version", "accuracy", "macro_f1", "weighted_f1", "cv_macro_f1"]]
          .to_string(index=False))
    tuning_lines, tuning_summary = nc.t3_tuning_text(comparison)
    print("\nTuning summary:")
    for line in tuning_lines:
        print("  " + line)
    print("  " + tuning_summary)
    # Pick the model by CV, not test scores
    best_name = res_tuned.sort_values("cv_macro_f1", ascending=False).iloc[0]["model"]
    best_search, best_pred, best_params = tuned[best_name]
    print(f"\nBest model by cross-validated macro-F1: {best_name}")
    report = classification_report(y_te, best_pred, labels=nc.GROUP_ORDER, zero_division=0)
    print("\nPer-class report on the test set:\n", report)
    with open("task3_classification_report.txt", "w") as f:
        f.write(f"Best model: {best_name}\nParams: {best_params}\n\n{report}")
    # Error analysis
    err = nc.t3_error_table(y_te, best_pred)
    print("\n===== Error analysis per audience group (share of each TRUE group, test set) =====")
    print(err.round(2).to_string())
    err.round(4).to_csv("task3_error_analysis.csv")
    safety = nc.t3_safety(y_te, best_pred)
    print("\n===== Higher-risk classification errors =====")
    print(f"  Teens/Adults titles predicted as Kids or Family: {safety['severe_lenient']} of {safety['older_titles']} "
          f"({safety['severe_lenient_rate']:.1%})   (Adults predicted as Kids: {safety['adults_as_kids']})")
    print(f"  All titles predicted as a LESS mature group than the real one: {safety['too_lenient']} "
          f"({safety['too_lenient_rate']:.1%}); as a MORE mature group: {safety['too_strict_rate']:.1%}")
    print(f"  Average cost per title: {safety['mean_cost']:.2f} (a step too lenient costs {nc.T3_UNDER_COST}, "
          f"a step too strict {nc.T3_OVER_COST}; matrix below, rows = true group, columns = predicted)")
    print(nc.t3_cost_matrix().to_string())
    pd.DataFrame([safety]).round(4).to_csv("task3_safety.csv", index=False)
    print("  This model should not be used alone to judge age suitability.")
    oof = cross_val_predict(clone(best_search.best_estimator_), X_tr, y_tr, cv=cv, method="predict_proba")
    curve = nc.t3_review_curve(y_tr.values, pd.DataFrame(oof, columns=np.unique(y_tr)))
    print("\n===== 'Review required' (scores from the training folds, never from the test set) =====")
    print(curve.round(3).to_string(index=False))
    print(f"  A title whose top model score is below the threshold would be sent to a person. "
          f"The dashboard starts at {nc.T3_REVIEW_THRESHOLD}. The scores are not calibrated probabilities.")
    curve.round(4).to_csv("task3_review_curve.csv", index=False)
    with open("task3_best_params.json", "w") as f:
        json.dump({"model": best_name, "params": best_params,
                   "cv_macro_f1": round(float(best_search.best_score_), 4),
                   "all_tuned": {n: tuned[n][2] for n in tuned}}, f, indent=2)
    # Paired bootstrap on the test set
    print("\n===== Model differences (paired bootstrap) =====")
    sig_rows = []
    names = list(tuned)
    n_comparisons = len(names) * (len(names) - 1) // 2 * 2
    level = 1 - 0.05 / n_comparisons                                 # stricter range, because we look at many gaps
    print(f"  ({n_comparisons} comparisons at once, so the range is the {level:.1%} one instead of 95%; {BOOTSTRAP_REPS} repetitions each)")
    # Encode labels for bootstrap comparisons
    code = {g: k for k, g in enumerate(nc.GROUP_ORDER)}
    truth = y_te.map(code).values
    as_codes = {name: pd.Series(tuned[name][1]).map(code).values for name in names}
    metric_fns = {"accuracy": lambda t_, p_: float(np.mean(t_ == p_)),
                  "macro_f1": lambda t_, p_: nc.macro_f1_codes(t_, p_, len(code))}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for label, metric in metric_fns.items():
                gap, low, high = nc.paired_bootstrap(truth, as_codes[a], as_codes[b], metric,
                                                     n=BOOTSTRAP_REPS, level=level)
                clear = low > 0 or high < 0
                sig_rows.append({"A": a, "B": b, "metric": label, "gap_A_minus_B": round(gap, 4),
                                 "low": round(low, 4), "high": round(high, 4),
                                 "clearly_different": clear, "range_level": round(level, 4)})
                verdict = ("clear gap even after adjusting" if clear else "not clearly different")
                print(f"  {a} vs {b} [{label}]: {gap:+.3f}  ({low:+.3f} to {high:+.3f})  -> {verdict}")
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv("task3_model_significance.csv", index=False)
    print(f"\nIn words (ranges are the {level:.1%} ones; gaps are shown from the better model's side):")
    for line in nc.t3_significance_text(sig_df)[1]:
        print("  " + line)
    print("Wording for the report: if the range includes 0, write that a model 'performed slightly better',\n"
          "not that it 'is better'. The final model was picked by cross-validation, not by these numbers.")
    # Default vs tuned comparison
    plot_df = comparison[comparison["model"] != "Baseline (majority)"]
    baseline_f1 = float(comparison.loc[comparison["model"] == "Baseline (majority)", "macro_f1"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro-F1")]:
        plot_df.pivot(index="model", columns="version", values=col)[["default", "tuned"]].plot(
            kind="bar", ax=ax)
        base_value = baseline if col == "accuracy" else baseline_f1
        ax.axhline(base_value, color="grey", linestyle="--", linewidth=1, label="majority baseline")
        ax.text(0.01, base_value + 0.01, f"baseline {base_value:.3f}", color="grey", fontsize=8,
                transform=ax.get_yaxis_transform())
        ax.set_ylim(0, 1); ax.set_title(f"{title}: default vs tuned"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig("task3_model_comparison.png", dpi=150); plt.close()
    # Confusion matrices
    cm = confusion_matrix(y_te, best_pred, labels=nc.GROUP_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm, display_labels=nc.GROUP_ORDER).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{best_name}: counts")
    ConfusionMatrixDisplay(cm / cm.sum(axis=1, keepdims=True), display_labels=nc.GROUP_ORDER).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".0%")
    axes[1].set_title("Share of each true group")
    plt.tight_layout(); plt.savefig("task3_confusion_matrix.png", dpi=150); plt.close()
    # Random Forest importance
    rf_pipe = tuned["Random Forest"][0].best_estimator_
    try:
        names = rf_pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        names = np.array([f"feature {i}" for i in range(len(rf_pipe.named_steps["model"].feature_importances_))])
    names = [str(n).replace("cat__", "").replace("genres__", "genre: ").replace("num__", "") for n in names]
    imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names).nlargest(15)
    imp.sort_values().plot(kind="barh", figsize=(8, 6), color="#55A868")
    plt.title("Top 15 features (tuned Random Forest, impurity-based)\nthis kind can favour columns with many values, "
              "see the permutation chart", fontsize=10)
    plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig("task3_feature_importance.png", dpi=150); plt.close()
    print("\nTop features (impurity-based):\n", imp.round(3).to_string())
    # Test-set permutation importance
    perm = permutation_importance(best_search.best_estimator_, X_te, y_te, scoring="f1_macro",
                                  n_repeats=10, random_state=nc.RANDOM_STATE, n_jobs=1)
    perm_df = pd.DataFrame({"drop_in_macro_f1": perm.importances_mean, "std": perm.importances_std},
                           index=X_te.columns).sort_values("drop_in_macro_f1")
    perm_df.round(4).to_csv("task3_permutation_importance.csv")
    ax = perm_df["drop_in_macro_f1"].plot(kind="barh", xerr=perm_df["std"], figsize=(8, 5), color="#4C72B0")
    ax.set_title(f"Permutation importance on the test set ({best_name})\n"
                 "drop in macro-F1 when the column is shuffled; 'listed_in' = all genres together", fontsize=10)
    ax.set_xlabel("Drop in macro-F1")
    plt.tight_layout(); plt.savefig("task3_permutation_importance.png", dpi=150); plt.close()
    print("\nPermutation importance (test set, biggest first):\n",
          perm_df["drop_in_macro_f1"].iloc[::-1].round(4).to_string())
    print("Columns that carry the same information (n_genres and the genres, year_added and years_to_netflix) "
          "share their importance, so each can look smaller than it is.")

    print("\nSaved: task3_class_distribution.png, task3_model_comparison.csv, task3_model_comparison.png,\n"
          "       task3_confusion_matrix.png, task3_feature_importance.png,\n"
          "       task3_classification_report.txt, task3_best_params.json, task3_model_significance.csv,\n"
          "       task3_error_analysis.csv, task3_safety.csv, task3_review_curve.csv,\n"
          "       task3_permutation_importance.png / .csv")



if __name__ == "__main__":
    main()

    