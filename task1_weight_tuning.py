"""netflix-ml-tasks TSK1: task1_weight_tuning"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netflix_core as nc


K = 10
path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.csv"
try:
    df = nc.load_t1(path)
    size_note = nc.check_size(df)
except ValueError as err:
    sys.exit(f"Cannot continue: {err}")
N = len(df)
for note in df.attrs.get("issues", []) + ([size_note] if size_note else []):
    print("NOTE:", note)
genre_list = nc.get_genres(df)
G = np.array([[g in set(nc.split_items(s)) for g in genre_list] for s in df["listed_in"]], dtype=np.float32)

blocks_full = nc.build_blocks(df, use_genre=True)
blocks_nogenre = {c: b for c, b in blocks_full.items() if c != "listed_in"}
YEARS = df["release_year"].values
def top_lists(X, queries, k=K):
    """Top-k neighbors with the same ranking and tie-break as the dashboard."""
    S = (X[queries] @ X.T).toarray()
    return np.array([nc.rank_like_production(S[r], YEARS, YEARS[q], k, skip=[q]) for r, q in enumerate(queries)])

def genre_overlap(top, queries):
    """Mean genre overlap between queries and recommendations."""
    inter = np.einsum("qkg,qg->qk", G[top], G[queries])
    union = G[top].sum(2) + G[queries].sum(1)[:, None] - inter
    return (inter / np.maximum(union, 1)).mean(axis=1)
def fair_score(weights, queries):
    X = nc.combine_blocks(blocks_nogenre, weights)
    return genre_overlap(top_lists(X, queries), queries)

rng = np.random.default_rng(42)
order = rng.permutation(N)
n_tune, n_check = min(300, N // 3), min(500, N // 2)
tune_q, check_q = order[:n_tune], order[n_tune:n_tune + n_check]

others = [c for c in nc.DEFAULT_WEIGHTS if c != "listed_in"]
candidates = [dict(nc.DEFAULT_WEIGHTS)]
for _ in range(40):
    w = dict(nc.DEFAULT_WEIGHTS)
    for c in others:
        w[c] = float(np.exp(rng.uniform(np.log(0.1), np.log(3.0))))
    candidates.append(w)

print(f"Trying {len(candidates)} weight settings on {len(tune_q)} titles ...")
tune_scores = [fair_score(w, tune_q).mean() for w in candidates]
best = candidates[int(np.argmax(tune_scores))]
print(f"Hand-picked weights score {tune_scores[0]:.4f} on the search titles; best found {max(tune_scores):.4f}")

old_check = fair_score(nc.DEFAULT_WEIGHTS, check_q)
new_check = fair_score(best, check_q)
gain = new_check - old_check
se = gain.std(ddof=1) / np.sqrt(len(gain))
low, high = gain.mean() - 1.96 * se, gain.mean() + 1.96 * se
adopted = bool(low > 0)
print(f"\nOn {len(check_q)} NEW titles (genre overlap@10, fair test):")
print(f"  hand-picked : {old_check.mean():.4f}")
print(f"  tuned       : {new_check.mean():.4f}")
print(f"  gain        : {gain.mean():+.4f}  (95% range {low:+.4f} to {high:+.4f})")
print("  -> " + ("clearly better, tuned weights will be used." if adopted else
                 "not clearly better. Keeping the hand-picked weights and saying so in the report."))
with open(nc.WEIGHTS_FILE, "w") as f:
    json.dump({"adopted": adopted,
               "weights": {c: round(best[c], 3) for c in others},
               "fair_score_hand_picked": round(float(old_check.mean()), 4),
               "fair_score_tuned": round(float(new_check.mean()), 4),
               "gain_95_range": [round(float(low), 4), round(float(high), 4)],
               "note": "genre weight is hand-picked (cannot be tuned fairly with a genre-based score)"},
              f, indent=2)

base_w = nc.load_weights()
reference = top_lists(nc.combine_blocks(blocks_full, base_w), check_q[:n_check])
rows = []
for gw in [0.0, 1.0, 2.0, 3.0, 5.0]:
    X = nc.combine_blocks(blocks_full, {**base_w, "listed_in": gw})
    top = top_lists(X, check_q[:n_check])
    same_list = np.mean([len(set(a) & set(b)) / K for a, b in zip(top, reference)])
    rows.append({"genre_weight": gw,
                 "same_titles_as_weight_3": round(same_list, 3),
                 "genre_overlap (circular!)": round(genre_overlap(top, check_q[:n_check]).mean(), 3)})
sens = pd.DataFrame(rows)
print("\nWhat the genre weight does (the last column is circular - genres are an input):")
print(sens.to_string(index=False))
sens.to_csv("task1_genre_weight_sensitivity.csv", index=False)
X_ng = nc.combine_blocks(blocks_nogenre, {c: base_w[c] for c in others})
by_genre = {g: np.where(G[:, j] > 0)[0] for j, g in enumerate(genre_list)}
usable = [g for g, rows_ in by_genre.items() if len(rows_) >= 8]
STRATEGIES = {"1 liked title": None, "average of 3": "average", "each of 3 in turn": "each"}
SCENARIOS = ["3 titles, same genre", "3 titles, 3 different genres"]
N_USERS = 300
profile_rows, results5 = [], {}

def ci(v):
    v = np.asarray(v, dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), v.mean() - 1.96 * se, v.mean() + 1.96 * se

if len(usable) < 3:
    print("\nSkipping the liked-titles test: fewer than 3 genres have at least 8 titles in this dataset.")
else:
    for scenario in SCENARIOS:
        hits = {k: [] for k in STRATEGIES}
        covs = {k: [] for k in STRATEGIES}
        for _ in range(N_USERS):
            if scenario.endswith("same genre"):
                wanted = [usable[rng.integers(len(usable))]]
                liked = [int(i) for i in rng.choice(by_genre[wanted[0]], 3, replace=False)]
            else:
                wanted = [str(g) for g in rng.choice(usable, 3, replace=False)]
                liked = [int(rng.choice(by_genre[g])) for g in wanted]
                if len(set(liked)) < 3:
                    continue
            cols = [genre_list.index(g) for g in wanted]
            for name, method in STRATEGIES.items():
                if method is None:
                    rows = nc.rank_like_production(nc.row_similarity(X_ng, liked[0]), YEARS, YEARS[liked[0]], K, skip=liked)
                else:
                    rows = nc.combine_picks(X_ng, YEARS, liked, K, method)[0]
                has = G[rows][:, cols]
                hits[name].append(has.max(axis=1).mean())
                covs[name].append(has.max(axis=0).mean())
        results5[scenario] = (hits, covs)
        for name in STRATEGIES:
            (h, hl, hh), (c, cl, ch) = ci(hits[name]), ci(covs[name])
            profile_rows.append({"scenario": scenario, "strategy": name, "users": len(hits[name]),
                                 "genre_hit_share": h, "hit_low_95": hl, "hit_high_95": hh,
                                 "genre_coverage": c, "coverage_low_95": cl, "coverage_high_95": ch})
    profile = pd.DataFrame(profile_rows)
    print(f"\nSimulated users ({N_USERS} per situation, genres hidden from the vectors). hit share = share of the 10 suggestions "
          "with a liked genre; coverage = share of the liked genres that appear at least once:")
    print(profile.round(3).to_string(index=False))
    summary5 = []
    for scenario in SCENARIOS:
        hits, covs = results5[scenario]
        for label, data in [("hit share", hits), ("genre coverage", covs)]:
            diff = np.array(data["each of 3 in turn"]) - np.array(data["average of 3"])
            m, lo, hi = ci(diff)
            verdict = ("'each in turn' is clearly better" if lo > 0 else
                       "'average' is clearly better" if hi < 0 else "no clear difference")
            line = (f"{scenario}: {label}, each in turn minus average = {m:+.3f} (95% range {lo:+.3f} to {hi:+.3f}): {verdict}.")
            print("  " + line)
            summary5.append(line)
    profile.round(4).to_csv("task1_profile_test.csv", index=False)
sens_lookup = sens.set_index("genre_weight")["same_titles_as_weight_3"]
lines = [
    f"Weights: {len(candidates)} weight settings were tried on {len(tune_q)} titles and the best one was checked on "
    f"{len(check_q)} different titles (genre overlap@10, genres hidden): hand-picked {old_check.mean():.4f} vs tuned "
    f"{new_check.mean():.4f}, gain {gain.mean():+.4f} (95% range {low:+.4f} to {high:+.4f}). "
    + ("The gain is clear, so the tuned weights are used." if adopted else
       "There is no clear gain, so the hand-picked weights are kept."),
    f"The genre weight (3.0) stays a judgment call. The brief asks for suggestions 'based on genres, categories and content "
    f"attributes', so genres lead; and it cannot be tuned against genre overlap because that would be circular. "
    f"Sensitivity: with genre weight 1 only {sens_lookup.get(1.0, float('nan')):.0%} of the suggested titles are the same as with 3, "
    f"with weight 5 it is {sens_lookup.get(5.0, float('nan')):.0%}. A non-circular way to justify it is the human review: "
    "have reviewers compare lists made with different genre weights.",
]
if results5:
    lines.append(f"Several liked titles ({N_USERS} simulated users per situation): " + " ".join(summary5))
    lines.append("If 'each in turn' is clearly better when the liked titles are from different genres, say so and offer it as the "
                 "better way to combine very different picks; if there is no clear difference, keep the simple average.")
print("\nFor the report (copy this):")
for line in lines:
    print("  - " + line)
with open("task1_tuning_summary.txt", "w") as f:
    f.write("\n\n".join(lines) + "\n")

n_panels = 2 if results5 else 1
fig, axes = plt.subplots(1, n_panels, figsize=(6 + 6 * (n_panels - 1), 4.4), squeeze=False)
axes = axes[0]
axes[0].bar(["hand-picked", "tuned"], [old_check.mean(), new_check.mean()], color=["#4C72B0", "#55A868"])
axes[0].set_ylim(0, max(old_check.mean(), new_check.mean()) * 1.3 or 1)
axes[0].set_title("Fair test on new titles (genre overlap@10)")
if results5:
    width = 0.25
    for k, (name, color) in enumerate(zip(STRATEGIES, ["#999999", "#DD8452", "#55A868"])):
        vals = [profile[(profile["scenario"] == sc) & (profile["strategy"] == name)]["genre_coverage"].iloc[0] for sc in SCENARIOS]
        lo = [profile[(profile["scenario"] == sc) & (profile["strategy"] == name)]["coverage_low_95"].iloc[0] for sc in SCENARIOS]
        hi = [profile[(profile["scenario"] == sc) & (profile["strategy"] == name)]["coverage_high_95"].iloc[0] for sc in SCENARIOS]
        axes[1].bar(np.arange(2) + (k - 1) * width, vals, width, color=color, label=name,
                    yerr=[np.array(vals) - lo, np.array(hi) - np.array(vals)], capsize=3)
    axes[1].set_xticks(range(2)); axes[1].set_xticklabels(["same genre", "3 different genres"])
    axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=8)
    axes[1].set_title("Share of the liked genres that appear in the suggestions", fontsize=10)
plt.tight_layout(); plt.savefig("task1_weight_tuning.png", dpi=150); plt.close()
print(f"\nSaved: {nc.WEIGHTS_FILE}, task1_genre_weight_sensitivity.csv, task1_profile_test.csv, "
      "task1_tuning_summary.txt, task1_weight_tuning.png")


