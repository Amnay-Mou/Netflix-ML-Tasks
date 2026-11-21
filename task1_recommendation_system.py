"""netflix-ml-tasks TSK1: task1_recommendation_system"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netflix_core as nc

path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.csv"
user_titles = sys.argv[2:]
try:
    df = nc.load_t1(path)
    size_note = nc.check_size(df)
except ValueError as err:
    sys.exit(f"Cannot continue: {err}")
print("Titles:", len(df))
for note in df.attrs.get("issues", []) + ([size_note] if size_note else []):
    print("  NOTE:", note)
weights = nc.load_weights()
print("Weights:", "tuned file found" if weights != nc.DEFAULT_WEIGHTS else "hand-picked defaults", weights)
rec = nc.Recommender(df, weights)
X_full = rec.X
X_nogenre = nc.make_features(df, use_genre=False, weights=weights)
print("Feature matrix:", X_full.shape)

def label(i):
    return f'{df.at[i, "title"]} ({df.at[i, "release_year"]}, {df.at[i, "type"]})'

def choose(rows):
    """Let the user pick from possible title matches."""
    for n, i in enumerate(rows, 1):
        print(f"    {n}. {label(i)}")
    if not sys.stdin.isatty():
        return None
    answer = input("  Type the number (or press Enter to skip): ").strip()
    return rows[int(answer) - 1] if answer.isdigit() and 1 <= int(answer) <= len(rows) else None

def find_row(text):
    """Find a title without guessing when there is no exact match."""
    kind, rows = nc.find_titles(df, text)
    if kind == "exact":
        return rows[0]
    if kind == "none":
        print(f'Title "{text}" not found.')
        return None
    if kind == "several":
        print(f'"{text}" matches {len(rows)} titles:')
        pick = choose(rows)
        if pick is None:
            print(f'  Not guessing. Run again with the year, e.g. "{df.at[rows[0], "title"]} ({df.at[rows[0], "release_year"]})".')
        return pick
    print(f'No exact match for "{text}". Did you mean:')
    pick = choose(rows)
    if pick is None:
        print("  Not guessing. Run again with the exact title.")
    return pick

demo_rng = np.random.default_rng(7)
start_rows = ([find_row(q) for q in user_titles] if user_titles
              else [int(i) for i in demo_rng.choice(len(df), 3, replace=False)])

examples, found = [], []
for i in start_rows:
    if i is None:
        continue
    found.append(i)
    res = rec.recommend(i, n=10)
    res.insert(0, "query", df.at[i, "title"])
    print(f'\n=== Because you watched: {label(i)} ===')
    print(res[["similarity", "title", "type", "listed_in"]].to_string(index=False))
    examples.append(res)
if len(found) > 1:
    names = " + ".join(df.loc[found, "title"])
    if df.loc[found, "type"].nunique() > 1:
        print("\n  (your picks mix movies and shows)")
    combo = rec.recommend_many(found, n=10)
    print(f"\n=== Average of: {names} (one taste profile) ===")
    print(combo[["similarity", "title", "type", "listed_in"]].to_string(index=False))
    combo.insert(0, "query", names + " [average]")
    examples.append(combo)
    each = rec.recommend_many(found, n=10, method="each")
    print(f"\n=== Each of: {names} in turn ===")
    print(each[["similarity", "title", "type", "because_of"]].to_string(index=False))
    each.insert(0, "query", names + " [each in turn]")
    examples.append(each)
    print("  Averaging very different picks can give generic suggestions; 'each in turn' keeps every pick represented. "
          "task1_weight_tuning.py tests both on simulated users.")
if examples:
    pd.concat(examples).to_csv("task1_example_recommendations.csv", index=False)
SHEET, RUBRIC, SUMMARY = "task1_human_review_sheet.csv", "task1_human_review_rubric.txt", "task1_human_review_summary.csv"
if os.path.exists(SHEET):
    review = nc.t1_review_summary(pd.read_csv(SHEET))
    print("\n===== Human review =====")
    for line in nc.t1_review_text(review):
        print(line)
    pd.DataFrame([{k: (str(v) if isinstance(v, dict) else v) for k, v in review.items()}]).to_csv(SUMMARY, index=False)
else:
    sheet = nc.t1_review_sheet(df, rec)
    sheet.to_csv(SHEET, index=False)
    with open(RUBRIC, "w") as f:
        f.write(nc.T1_REVIEW_RUBRIC + "\n" + nc.T1_REVIEW_LIMITS + "\n")
    print(f"\nSaved {SHEET} ({len(sheet)} suggestions for {sheet['query'].nunique()} query titles) and {RUBRIC}: "
          "fill the score column(s) by hand and run again for a human-judged score.")

N_QUERIES = min(500, len(df) // 2)
per_model = {
    "Random recommendations": nc.t1_evaluate(df, None, n_queries=N_QUERIES),
    "No-genre model (fair test)": nc.t1_evaluate(df, X_nogenre, n_queries=N_QUERIES),
    "Full model": nc.t1_evaluate(df, X_full, n_queries=N_QUERIES),
}
results, long_table = nc.t1_eval_table(per_model)
results = results.round(3)

print(f"\n===== Offline evaluation ({N_QUERIES} random titles, top-10 each, ranked like the dashboard) =====")
print("(proxy scores: they show that suggestions look alike on paper, not that viewers will enjoy them)")
print(results.to_string())
print("\nWhat each number can show (95% range of the mean over the query titles):")
for r in long_table.itertuples():
    print(f"  {r.model:<27} {r.metric:<18} {r.mean:.3f} ({r.low_95:.3f} to {r.high_95:.3f})   {r.kind}")
results.to_csv("task1_evaluation.csv")
long_table.round(4).to_csv("task1_evaluation_ci.csv", index=False)


def value(model, metric, col="mean"):
    return float(long_table[(long_table["model"] == model) & (long_table["metric"] == metric)][col].iloc[0])


NG, RND = "No-genre model (fair test)", "Random recommendations"
print("\nFor the report (copy this):")
print(f"  Independent proxy (genres hidden from the model): the suggestions share a genre with the starting title "
      f"{value(NG, 'genre_hit_rate@10'):.1%} of the time (95% range {value(NG, 'genre_hit_rate@10', 'low_95'):.1%} to "
      f"{value(NG, 'genre_hit_rate@10', 'high_95'):.1%}) against {value(RND, 'genre_hit_rate@10'):.1%} for random suggestions; "
      f"genre overlap {value(NG, 'genre_overlap@10'):.3f} against {value(RND, 'genre_overlap@10'):.3f}.")
print(f"  The full model's genre scores ({value('Full model', 'genre_overlap@10'):.3f} overlap, "
      f"{value('Full model', 'genre_hit_rate@10'):.1%} hit rate) and every 'same type' score are metadata consistency, "
      "not evidence of quality: genres and type are inputs of that model.")

tie_rng = np.random.default_rng(1)
ties = 0
sample = tie_rng.choice(len(df), min(300, len(df) // 2), replace=False)
for i in sample:
    s = np.sort(np.round(nc.row_similarity(X_full, int(i)), 6))[::-1]
    ties += s[10] == s[11]
print(f"Queries with a tie at the top-10 cut-off: {ties / len(sample):.0%} "
      "(ties are broken by closest release year)")
ax = results.plot(kind="bar", figsize=(10, 5.5))
kinds = {(r.model, r.metric): r.kind for r in long_table.itertuples()}
errors = {(r.model, r.metric): (r.high_95 - r.low_95) / 2 for r in long_table.itertuples()}
for container, metric in zip(ax.containers, results.columns):
    for bar, model in zip(container, results.index):
        if kinds[(model, metric)].startswith("metadata"):
            bar.set_hatch("//")
            bar.set_alpha(0.55)
        ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=errors[(model, metric)],
                    color="black", capsize=2, linewidth=1)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
# ax.set_title("Task 1 - Recommendation quality (proxy scores, higher is better)\\n"
#              "hatched = metadata consistency: the model uses this information, so it is NOT independent evidence", fontsize=10)
ax.set_title(
    "Task 1 - Recommendation Quality",
    fontsize=12,
    pad=35
)
ax.text(
    0.5, 1.015,
    "Hatched bars = metadata consistency, not independent quality evidence",
    transform=ax.transAxes,
    ha="center",
    fontsize=9
)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("task1_evaluation.png", dpi=150)
plt.close()
print("\nSaved: task1_example_recommendations.csv, task1_evaluation.csv, task1_evaluation_ci.csv, task1_evaluation.png,\n"
      f"       {SHEET}, {RUBRIC}" + (f", {SUMMARY}" if os.path.exists(SUMMARY) else ""))

