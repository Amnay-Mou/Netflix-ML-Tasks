"""Shared functions for the Netflix ML tasks.

Task 1: recommendations; Task 2: content type; Task 3: audience groups; Task 5: forecasting.
The task scripts and dashboard use this module."""
import difflib
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier

SEED = 42
RANDOM_STATE = SEED          # Alias used by Task 3


# helpers
def read_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def split_items(text):
    """'Dramas, International Movies' -> ['Dramas', 'International Movies']"""
    return [t.strip() for t in str(text).split(",") if t.strip()]


def count_items(col):
    """How many comma-separated items are in each row (0 if empty)."""
    return col.fillna("").apply(lambda s: len(s.split(",")) if s else 0)


def get_genres(df):
    """Sorted list of every genre in the listed_in column."""
    return sorted({g for s in df["listed_in"].fillna("") for g in split_items(s)})


# data checks
REQUIRED_T1 = ["show_id", "type", "title", "director", "country",
               "release_year", "rating", "listed_in"]
REQUIRED_ALL = REQUIRED_T1 + ["date_added", "duration"]     # Tasks 2 and 3 also need these
MIN_TITLES = 30        # Minimum usable sample
SMALL_DATASET = 300    # Warn for smaller samples


def check_columns(df, needed=REQUIRED_ALL):
    """Stop early, with a readable message, if the file does not look like the Netflix dataset."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("The dataset is missing required column(s): " + ", ".join(missing) +
                         ". Columns found: " + ", ".join(map(str, df.columns)) +
                         ". Rename the columns in the CSV if they only differ in spelling.")


def check_size(df):
    """Raise if there are too few titles to evaluate; return a warning text if it is merely small."""
    if len(df) < MIN_TITLES:
        raise ValueError(f"Only {len(df)} usable titles. At least {MIN_TITLES} are needed "
                         "for the evaluation to make any sense.")
    if len(df) < SMALL_DATASET:
        return (f"Only {len(df)} titles: the scores below are based on a small sample "
                "and can change a lot with a different sample.")
    return None


EXPECTED_TYPES = ["Movie", "TV Show"]
MIN_PER_CLASS = 10     # Minimum samples per class


def check_classes(labels, minimum=MIN_PER_CLASS, what="class"):
    """Refuse to train when a class is missing or too small for a stratified split and cross-validation."""
    counts = pd.Series(list(labels), dtype="object").value_counts()
    if counts.empty:
        raise ValueError(f"There are no labelled titles to learn from (no {what} values at all).")
    if len(counts) < 2:
        raise ValueError(f"Only one {what} is present ({counts.index[0]}). A classifier needs at least two.")
    small = counts[counts < minimum]
    if len(small):
        raise ValueError(f"Too few titles in the {what}: " + ", ".join(f"{k} ({v})" for k, v in small.items()) +
                         f". At least {minimum} per {what} are needed.")


GENRE_MIN_TITLES = 10    # Minimum frequency for genre validation


def check_genres(genre_column):
    """Task 3 uses genres as features. Stop with a clear message if no genre repeats enough to carry information."""
    genres = [g.lower() for text in genre_column for g in split_items(text)]
    counts = pd.Series(genres, dtype="object").value_counts()
    if counts.empty:
        raise ValueError("Task 3 needs the genres (listed_in), but every title has an empty genre list.")
    if counts.iloc[0] < GENRE_MIN_TITLES:
        raise ValueError(f"Task 3 needs genres that repeat: the most common genre ('{counts.index[0]}') appears in only "
                         f"{counts.iloc[0]} title(s); at least {GENRE_MIN_TITLES} are needed. "
                         "Check that listed_in holds genre names separated by commas.")


def clean_years(df, issues, fill=True):
    """Make release_year a whole number. Missing or unreadable years get the median year (and are reported).

    fill=False (used by Task 3) leaves them empty instead: the median would be computed from ALL the rows,
    including the ones later kept for testing, so the model pipeline fills them from its training rows only.
    """
    df = df.copy()
    years = pd.to_numeric(df["release_year"], errors="coerce")
    if years.notna().sum() == 0:
        raise ValueError("The release_year column has no valid numbers.")
    bad = years.isna()
    if bad.any() and fill:
        median = int(years.median())
        issues.append(f"{int(bad.sum())} row(s) had a missing or invalid release year; "
                      f"the median year ({median}) was used for them.")
        years = years.fillna(median)
    elif bad.any():
        issues.append(f"{int(bad.sum())} row(s) had a missing or invalid release year; the model fills them with "
                      "the typical year of its training rows only.")
    df["release_year"] = years.astype(int) if not years.isna().any() else years
    return df


# Task 1
# How much each column counts in the similarity. These started as hand-picked
# numbers. task1_weight_tuning.py tests the non-genre ones on unseen titles and
# only saves new values if they clearly beat these (see load_weights).
# The genre weight CANNOT be tuned fairly with a genre-based test (that would be
# circular), so it stays a judgment call.
DEFAULT_WEIGHTS = {"listed_in": 3.0, "director": 1.0, "country": 1.0,
                   "rating": 1.0, "type": 1.0, "decade": 0.5, "title": 0.7}
OPTIONAL_WEIGHTS = {"description": 1.0}     # used only if the file has that column
WEIGHTS_FILE = "task1_tuned_weights.json"


def load_weights(path=WEIGHTS_FILE):
    """Hand-picked weights, unless the tuning script found clearly better ones."""
    weights = dict(DEFAULT_WEIGHTS)
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg = json.load(f)
            if cfg.get("adopted"):
                weights.update(cfg["weights"])
        except Exception:
            pass
    return weights


def load_t1(path):
    """Read and clean the file for Task 1. What was repaired is listed in df.attrs['issues']."""
    df = read_data(path)
    check_columns(df, REQUIRED_T1)
    issues = []

    # Remove repeated IDs, but keep rows without IDs
    repeated = df.duplicated(subset="show_id") & df["show_id"].notna()
    df = df[~repeated].copy()

    text_cols = ["title", "type", "director", "country", "rating", "listed_in"]
    if "description" in df.columns:                  # Optional column
        text_cols.append("description")
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip().replace({"Not Given": ""})

    no_title = df["title"] == ""
    if no_title.any():
        issues.append(f"{int(no_title.sum())} row(s) without a title were removed.")
        df = df[~no_title]

    df = clean_years(df, issues)

    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows left after cleaning.")
    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"
    df.attrs["issues"] = issues
    return df


def find_titles(df, text, close_cutoff=0.6, max_close=3):
    """Look a title up by name without guessing silently.

    Returns (kind, rows):
      "exact"   one title has exactly this name
      "several" several titles share the name (write 'Name (2019)' to pick by year)
      "close"   no exact match; rows are the closest spellings (the caller should ask)
      "none"    nothing similar
    """
    text = text.strip()
    year = None
    m = re.match(r"^(.*?)\s*\((\d{4})\)$", text)
    if m:
        text, year = m.group(1).strip(), int(m.group(2))

    lower = df["title"].str.lower()
    exact = df.index[lower == text.lower()]
    if year is not None:
        exact = exact[df.loc[exact, "release_year"] == year]
    if len(exact) == 1:
        return "exact", [int(exact[0])]
    if len(exact) > 1:
        return "several", [int(i) for i in exact]

    close = difflib.get_close_matches(text.lower(), lower.unique().tolist(), n=max_close, cutoff=close_cutoff)
    rows = [int(df.index[lower == c][0]) for c in close]
    return ("close", rows) if rows else ("none", [])


def build_blocks(df, use_genre=True):
    """One TF-IDF block per column (no weights yet). Built once, re-weighted many times.

    A column with no usable words at all (for example every director is 'Not Given')
    is left out with a warning instead of crashing. 'description' is used only if the
    file has that column (the Netflix file we use does not).
    """
    list_kw = dict(tokenizer=split_items, token_pattern=None)
    title_kw = dict(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    kinds = {"listed_in": list_kw, "director": list_kw, "country": list_kw,
             "rating": list_kw, "type": list_kw, "decade": list_kw, "title": title_kw,
             "description": title_kw}

    blocks, skipped = {}, []
    for col, kw in kinds.items():
        if col not in df.columns or (col == "listed_in" and not use_genre):
            continue
        try:
            blocks[col] = TfidfVectorizer(**kw).fit_transform(df[col])
        except ValueError:                           # Empty vocabulary
            skipped.append(col)
    if skipped:
        warnings.warn("No usable text in column(s): " + ", ".join(skipped) + ". They were left out.")
    if not blocks:
        raise ValueError("None of the columns has usable text, so no recommendations are possible.")
    return blocks


def combine_blocks(blocks, weights):
    """Multiply each block by its weight, stack them side by side, normalise each row."""
    return normalize(hstack([b * weights.get(col, OPTIONAL_WEIGHTS.get(col, 1.0))
                             for col, b in blocks.items()]).tocsr())


def make_features(df, use_genre=True, weights=None):
    return combine_blocks(build_blocks(df, use_genre), weights or DEFAULT_WEIGHTS)


# Shared ranking rule
def row_similarity(X, idx):
    """Cosine similarity of row idx with every row (the rows of X are already length 1, so it is a dot product)."""
    return np.asarray((X @ X[idx].T).toarray()).ravel()


def rank_like_production(sims, years, ref_year, n, ok=None, skip=()):
    """Rank by similarity, breaking ties by the closest release year. Apply optional filters and exclude selected rows."""
    sims = np.round(np.asarray(sims, dtype=float), 6)
    order = np.lexsort((np.abs(np.asarray(years) - ref_year), -sims))        # Similarity is the primary sort key
    keep = np.ones(len(sims), dtype=bool) if ok is None else np.array(ok, dtype=bool)
    keep[list(skip)] = False
    return order[keep[order]][:n]


def top_neighbours(X, years, idx, n=10, ok=None):
    """The n suggestions for row idx under the production ranking, for ANY feature matrix X."""
    return rank_like_production(row_similarity(X, idx), years, years[idx], n, ok, skip=[idx])


COMBINE_METHODS = ("average", "each")


def combine_picks(X, years, idxs, n=10, method="average", ok=None):
    """Combine recommendations from several liked titles.

The average method builds one profile; the each method interleaves suggestions from each title. Returns rows, source titles and similarities."""
    if method not in COMBINE_METHODS:
        raise ValueError(f"method must be one of {COMBINE_METHODS}, not {method!r}")
    years = np.asarray(years)
    idxs = list(dict.fromkeys(int(i) for i in idxs))
    if method == "average":
        profile = normalize(np.asarray(X[idxs].mean(axis=0)))
        sims = np.asarray(X @ profile.T).ravel()
        rows = rank_like_production(sims, years, float(years[idxs].mean()), n, ok, skip=idxs)
        return rows, [None] * len(rows), np.round(sims, 6)[rows]
    lists, shown = [], []
    for i in idxs:
        s = row_similarity(X, i)
        shown.append(np.round(s, 6))
        lists.append(rank_like_production(s, years, years[i], len(s), ok, skip=idxs))
    pointer, rows, source, sims, seen = [0] * len(idxs), [], [], [], set()
    while len(rows) < n:
        moved = False
        for p, lst in enumerate(lists):
            while pointer[p] < len(lst) and int(lst[pointer[p]]) in seen:
                pointer[p] += 1
            if pointer[p] < len(lst) and len(rows) < n:
                r = int(lst[pointer[p]])
                seen.add(r)
                rows.append(r)
                source.append(idxs[p])
                sims.append(shown[p][r])
                pointer[p] += 1
                moved = True
        if not moved:
            break
    return np.array(rows, dtype=int), source, np.array(sims)


class Recommender:
    SHOW = ["title", "type", "listed_in", "country", "rating", "release_year"]

    def __init__(self, df, weights=None):
        self.df = df
        self.weights = weights or load_weights()
        self.X = make_features(df, use_genre=True, weights=self.weights)

    def _ok(self, only_type, min_year):
        ok = np.ones(len(self.df), dtype=bool)
        if only_type is not None:
            ok &= self.df["type"].values == only_type
        if min_year is not None:
            ok &= self.df["release_year"].values >= min_year
        return ok

    def _frame(self, rows, sims):
        out = self.df.loc[rows, self.SHOW].copy()
        out.insert(0, "similarity", np.round(np.asarray(sims, dtype=float), 3))
        return out.reset_index(drop=True)

    def _top(self, sims, skip, n, only_type, min_year, ref_year):
        """Best n rows by similarity. Ties are broken by release year closest to ref_year."""
        rows = rank_like_production(sims, self.df["release_year"].values, ref_year, n,
                                    self._ok(only_type, min_year), skip)
        return self._frame(rows, np.round(np.asarray(sims, dtype=float), 6)[rows])

    def neighbours(self, idx, n=10, same_type=False, min_year=None):
        """Row positions of the suggestions for row idx: the same list that recommend() shows."""
        only_type = self.df.at[idx, "type"] if same_type else None
        return rank_like_production(row_similarity(self.X, idx), self.df["release_year"].values,
                                    self.df.at[idx, "release_year"], n, self._ok(only_type, min_year), [idx])

    def recommend(self, idx, n=10, same_type=False, min_year=None):
        """Top n titles most similar to row idx, with optional filters.

        'similarity' is how alike two titles' details are (0 to 1). It is NOT the chance
        that someone will enjoy the title. If the filters leave fewer than n candidates,
        fewer than n rows come back.
        """
        sims = row_similarity(self.X, idx)
        only_type = self.df.at[idx, "type"] if same_type else None
        return self._top(sims, [idx], n, only_type, min_year, self.df.at[idx, "release_year"])

    def recommend_many(self, idxs, n=10, same_type=False, min_year=None, method="average"):
        """Suggestions from one or several liked titles (see combine_picks for the two methods).

        Nothing is remembered between calls (no viewing history, no saved user).
        If the picks mix movies and shows, same_type is ignored on purpose, because
        there is no single type to filter on. With method='each' a 'because_of' column says
        which liked title a suggestion came from.
        """
        idxs = list(dict.fromkeys(int(i) for i in idxs))
        types = set(self.df.loc[idxs, "type"])
        # "same type" only makes sense when everything they picked is the same type
        only_type = types.pop() if same_type and len(types) == 1 else None
        rows, source, sims = combine_picks(self.X, self.df["release_year"].values, idxs, n, method,
                                           self._ok(only_type, min_year))
        out = self._frame(rows, sims)
        if method == "each":
            out["because_of"] = [self.df.at[p, "title"] for p in source]
        return out


# Offline evaluation
def t1_evaluate(df, X=None, k=10, n_queries=500, seed=42):
    """Proxy scores for random query titles, as one value per query (X=None: random suggestions).

    The suggestions are ranked with the PRODUCTION rule (top_neighbours), so these numbers describe the lists the
    dashboard really shows. Metrics: genre overlap (Jaccard), genre hit rate (a shared genre exists), same type.
    """
    genre_sets = [{g.lower() for g in split_items(s)} for s in df["listed_in"]]
    types, years = df["type"].values, df["release_year"].values
    rng = np.random.default_rng(seed)
    overlap, hit, same_type = [], [], []
    for i in rng.choice(len(df), min(n_queries, len(df)), replace=False):
        if X is None:
            top = rng.choice(len(df), k + 1, replace=False)
            top = top[top != i][:k]
        else:
            top = top_neighbours(X, years, int(i), k)
        a = genre_sets[i]
        overlap.append(np.mean([len(a & genre_sets[j]) / max(len(a | genre_sets[j]), 1) for j in top]))
        hit.append(np.mean([len(a & genre_sets[j]) > 0 for j in top]))
        same_type.append(np.mean([types[j] == types[i] for j in top]))
    return {f"genre_overlap@{k}": np.array(overlap), f"genre_hit_rate@{k}": np.array(hit),
            f"same_type@{k}": np.array(same_type)}


def t1_metric_kind(model, metric):
    """What a score can show. Only genres-hidden genre scores are an independent proxy."""
    if model.startswith("Random"):
        return "baseline"
    if model.startswith("No-genre") and metric.startswith("genre"):
        return "independent proxy (genres hidden from the model)"
    if metric.startswith("same_type"):
        return "metadata consistency (the model uses the type)"
    return "metadata consistency (the model uses the genres)"


def t1_eval_table(per_model):
    """per_model = {model name: t1_evaluate result}. Returns (means, long table with 95% ranges and the kind)."""
    means = pd.DataFrame({m: {k: float(v.mean()) for k, v in r.items()} for m, r in per_model.items()}).T
    rows = []
    for model, res in per_model.items():
        for metric, v in res.items():
            se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            rows.append({"model": model, "metric": metric, "mean": float(v.mean()),
                         "low_95": float(v.mean() - 1.96 * se), "high_95": float(v.mean() + 1.96 * se),
                         "queries": len(v), "kind": t1_metric_kind(model, metric)})
    return means, pd.DataFrame(rows)


# Human review
T1_REVIEW_COLUMNS = ["reviewer_score_1to5", "reviewer2_score_1to5"]
T1_REVIEW_RUBRIC = """How to score each suggestion (1 to 5)

Question: how similar is the SUGGESTION to the title in the 'query' column?
  5 = very similar: it belongs in the same "if you liked this, try that" list
  4 = similar
  3 = somewhat related (same broad area, but a different feel)
  2 = weakly related
  1 = not similar at all

Judge how alike the two TITLES are (subject, mood, audience), not whether you personally like the suggestion.
If you do not know a title, look it up, or leave the score empty. Score every row you can.
Write whole numbers in the last column(s). A second person uses the column reviewer2_score_1to5 and must not
look at the first person's scores. Then run task1_recommendation_system.py again: it prints the summary.
"""
T1_REVIEW_LIMITS = ("Limits of this review: a small sample (a few queries with a handful of suggestions each), very few "
                    "reviewers, reviewers judged how alike two titles are and not whether viewers would enjoy them, and "
                    "reviewers may not know every title.")


def t1_review_queries(df, n_queries=15, seed=11):
    """Query rows for the review sheet: random, but with both types present in roughly the real proportion."""
    rng = np.random.default_rng(seed)
    n_queries = min(n_queries, len(df))
    movies, shows = np.where(df["type"] == "Movie")[0], np.where(df["type"] != "Movie")[0]
    n_shows = int(round(n_queries * len(shows) / len(df))) if len(shows) else 0
    n_shows = min(len(shows), max(n_shows, min(4, len(shows), n_queries // 3)))
    n_movies = min(len(movies), n_queries - n_shows)
    picks = np.concatenate([rng.choice(movies, n_movies, replace=False), rng.choice(shows, n_shows, replace=False)]) \
        if len(shows) else rng.choice(movies, n_movies, replace=False)
    return [int(i) for i in rng.permutation(picks)]


def t1_review_sheet(df, rec, n_queries=15, per_query=5, seed=11):
    """The blank sheet people score by hand (the second reviewer column is optional)."""
    rows = []
    for i in t1_review_queries(df, n_queries, seed):
        label = f'{df.at[i, "title"]} ({df.at[i, "release_year"]}, {df.at[i, "type"]})'
        for _, r in rec.recommend(i, n=per_query).iterrows():
            rows.append({"query": label, "suggestion": r["title"], "type": r["type"], "genres": r["listed_in"],
                         "reviewer_score_1to5": "", "reviewer2_score_1to5": ""})
    return pd.DataFrame(rows)


def t1_review_summary(sheet):
    """Numbers from the scored sheet. Scores outside 1-5 or not whole are counted as invalid and ignored."""
    scores, invalid = {}, 0
    for col in [c for c in T1_REVIEW_COLUMNS if c in sheet.columns]:
        s = pd.to_numeric(sheet[col], errors="coerce")
        good = s.between(1, 5) & (s % 1 == 0)
        typed = sheet[col].notna() & (sheet[col].astype(str).str.strip() != "")      # anything written in the cell
        invalid += int((typed & ~good).sum())
        if good.any():
            scores[col] = s.where(good)
    out = {"rows": len(sheet), "queries": int(sheet["query"].nunique()), "reviewers": len(scores),
           "invalid": invalid, "scored_rows": 0}
    if not scores:
        return out
    table = pd.DataFrame(scores)
    per_row = table.mean(axis=1)
    values = table.stack()
    out.update({"scored_rows": int(per_row.notna().sum()), "mean": float(values.mean()), "median": float(values.median()),
                "distribution": {int(k): int(v) for k, v in values.value_counts().sort_index().items()},
                "share_4_or_5": float((values >= 4).mean())})
    by_query = pd.concat([sheet["query"], per_row.rename("score")], axis=1).groupby("query")["score"].mean().dropna()
    if len(by_query):
        out["best_query"], out["worst_query"] = by_query.idxmax(), by_query.idxmin()
        out["best_query_mean"], out["worst_query_mean"] = float(by_query.max()), float(by_query.min())
    if len(scores) >= 2:
        a, b = list(scores.values())[:2]
        both = a.notna() & b.notna()
        out["agreement_rows"] = int(both.sum())
        if both.sum() >= 5:
            out["within_one_point"] = float(((a - b).abs() <= 1)[both].mean())
    return out


def t1_review_text(summary):
    """Plain sentences for the report and the dashboard."""
    if summary["scored_rows"] == 0:
        return [f"The human review has not been done yet: 0 of {summary['rows']} suggestions are scored. It is the "
                "strongest evidence available for Step 5, because the automatic numbers are only proxies. "
                "See task1_human_review_rubric.txt for how to score."]
    s = summary
    lines = [f"{s['scored_rows']} of {s['rows']} suggestions ({s['queries']} query titles) were scored by "
             f"{s['reviewers']} reviewer(s): mean {s['mean']:.2f} out of 5, median {s['median']:.0f}, "
             f"{s['share_4_or_5']:.0%} rated 4 or 5. Score counts: {s['distribution']}."]
    if "best_query" in s:
        lines.append(f"Best query: {s['best_query']} ({s['best_query_mean']:.2f}); weakest: {s['worst_query']} "
                     f"({s['worst_query_mean']:.2f}).")
    if "within_one_point" in s:
        lines.append(f"The two reviewers were within one point on {s['within_one_point']:.0%} of {s['agreement_rows']} "
                     "suggestions they both scored.")
    elif s["reviewers"] > 1:
        lines.append("Too few suggestions were scored by both reviewers to measure their agreement.")
    if s["invalid"]:
        lines.append(f"{s['invalid']} score(s) were not whole numbers from 1 to 5 and were ignored.")
    if s["scored_rows"] < s["rows"]:
        lines.append(f"Only {s['scored_rows']} of {s['rows']} rows are scored, so the mean is uncertain.")
    lines.append(T1_REVIEW_LIMITS)
    return lines


# Task 2
T2_CAT = ["rating", "main_country"]
T2_NUM = ["release_year", "year_added", "month_added", "years_to_netflix",
          "has_director", "n_directors", "n_countries", "n_genres"]


def prep_t2(path, strict=True, fill_years=True):
    """Prepare the Task 2 features and binary target (TV Show = 1).

With fill_years=False, missing years are imputed by the training pipeline rather than from the full dataset. strict=False relaxes dataset-size checks for the overview."""
    df = read_data(path)
    check_columns(df)
    issues = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    df = df.replace({"Not Given": np.nan, "nan": np.nan, "": np.nan})
    repeated = df.duplicated(subset="show_id") & df["show_id"].notna()
    if repeated.any():
        issues.append(f"{int(repeated.sum())} row(s) repeat an earlier show_id and were counted once.")
    df = df[~repeated].reset_index(drop=True)
    twins = int(df.duplicated(subset=["title", "type", "release_year"], keep=False).sum())
    if twins:
        issues.append(f"{twins} row(s) share title, type and release year with another row but have a different "
                      "show_id; they are kept (a different id is a different record).")

    bad_type = ~df["type"].isin(EXPECTED_TYPES)
    if bad_type.any():
        seen = sorted(df.loc[bad_type, "type"].fillna("(missing)").astype(str).unique())[:5]
        issues.append(f"{int(bad_type.sum())} row(s) with an unexpected type ({', '.join(seen)}) were removed; "
                      "only Movie and TV Show are accepted.")
        df = df[~bad_type].reset_index(drop=True)
    df = clean_years(df, issues, fill=fill_years)

    if strict:
        note = check_size(df)                  # raises if there are too few titles
        if note:
            issues.append(note)
        check_classes(df["type"], what="type")

    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    df["years_to_netflix"] = df["year_added"] - df["release_year"]
    df["has_director"] = df["director"].notna().astype(int)
    df["n_directors"] = count_items(df["director"])
    df["n_countries"] = count_items(df["country"])
    df["n_genres"] = count_items(df["listed_in"])
    df["main_country"] = df["country"].fillna("Unknown").str.split(",").str[0].str.strip()

    y = (df["type"] == "TV Show").astype(int)
    df.attrs["issues"] = issues
    return df, y


T2_SELECTION_FILE = "task2_selected_model.json"


def t2_models():
    """The three Task 2 models (the script and the dashboard use exactly these)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
    }


def t2_config(path=T2_SELECTION_FILE):
    """Model chosen by cross-validation in task2_content_type_prediction.py (Logistic Regression if not run yet)."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                name = json.load(f)["model"]
            if name in t2_models():
                return name, "selected by cross-validation"
        except Exception:
            pass
    return "Logistic Regression", "default"


class T2Imputer(BaseEstimator, TransformerMixin):
    """Fills an empty release_year from the TRAINING rows only, then recomputes years_to_netflix.

    The median of all rows would include the rows kept for testing. Used as the first step of t2_pipeline,
    so cross-validation refits it on every training fold. Columns that are not present are left alone.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.year_ = float(X["release_year"].median()) if "release_year" in X else np.nan
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if "release_year" in X:
            X["release_year"] = X["release_year"].fillna(self.year_)
            if "years_to_netflix" in X and "year_added" in X:
                gap = X["years_to_netflix"].isna() & X["year_added"].notna()
                X.loc[gap, "years_to_netflix"] = X.loc[gap, "year_added"] - X.loc[gap, "release_year"]
        return X


def t2_pipeline(num_cols=None, model=None, cat_cols=None):
    """Fill gaps (from training rows only), one-hot the categories, scale the numbers, then the model."""
    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
        ]), T2_CAT if cat_cols is None else cat_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols or T2_NUM),
    ])
    if model is None:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("fill", T2Imputer()), ("prep", prep), ("model", model)])


def t2_split(y):
    """Row positions (train, test) of the stratified 80/20 split used everywhere in Task 2."""
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)


# Per-class metrics
def t2_metrics(y_true, y_pred, proba=None):
    """Accuracy, balanced accuracy (average recall of both types), precision / recall / F1 for each type, macro-F1."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
           "precision_Movie": p[0], "recall_Movie": r[0], "f1_Movie": f[0],
           "precision_TV": p[1], "recall_TV": r[1], "f1_TV": f[1], "macro_f1": float(f.mean())}
    if proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    return {k: float(v) for k, v in out.items()}


def t2_class_report(y_true, y_pred):
    """The per-type table of precision, recall, F1 and support (Movie and TV Show)."""
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": n}, index=["Movie", "TV Show"])


# Select using training data only
T2_CV_SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_TV": "f1", "macro_f1": "f1_macro"}


def t2_choose(cv):
    """Pick the model from the cross-validation table (one row per real model; columns cv_macro_f1 and
    cv_macro_f1_std). The highest mean cross-validated macro-F1 wins. Only these training-part numbers are
    looked at: the test set plays no part. Returns (model name, explanation)."""
    ranked = cv["cv_macro_f1"].sort_values(ascending=False, kind="stable")
    best = ranked.index[0]
    note = f"{best} had the highest cross-validated macro-F1 ({ranked.iloc[0]:.3f}) on the training part."
    if len(ranked) > 1:
        gap, wobble = ranked.iloc[0] - ranked.iloc[1], float(cv.loc[best, "cv_macro_f1_std"])
        if gap <= wobble:
            note += (f" Close call: {ranked.index[1]} was only {gap:.3f} behind, less than the wobble between folds "
                     f"({wobble:.3f}), so the models are practically tied.")
    return best, note


def t2_study(df, y, num_cols, cat_cols=None, split=None, folds=5):
    """Run one Task 2 experiment. Select the model by cross-validation on training data, then evaluate it on the held-out set."""
    cat_cols = T2_CAT if cat_cols is None else cat_cols
    X = df[list(cat_cols) + list(num_cols)]
    tr, te = t2_split(y) if split is None else split
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows, preds = {}, {}
    for name, model in {"Baseline (majority)": DummyClassifier(strategy="most_frequent"), **t2_models()}.items():
        res = cross_validate(t2_pipeline(num_cols, clone(model), cat_cols), X_tr, y_tr, cv=cv,
                             scoring=T2_CV_SCORING, n_jobs=1)
        row = {}
        for key in T2_CV_SCORING:
            row[f"cv_{key}"], row[f"cv_{key}_std"] = float(res[f"test_{key}"].mean()), float(res[f"test_{key}"].std())
        pipe = t2_pipeline(num_cols, clone(model), cat_cols).fit(X_tr, y_tr)         # first use of the test part
        preds[name] = pipe.predict(X_te)
        row.update(t2_metrics(y_te, preds[name], pipe.predict_proba(X_te)[:, 1]))
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    real = table.drop(index="Baseline (majority)")
    selected, note = t2_choose(real[["cv_macro_f1", "cv_macro_f1_std"]])
    return {"table": table, "selected": selected, "selection_note": note, "preds": preds, "y_test": y_te,
            "split": (tr, te), "n_train": len(tr), "n_test": len(te)}


# Feature ablation
T2_ABLATIONS = {                     # name -> (numeric columns taken away, categorical columns taken away)
    "full (leak-free model)": ([], []),
    "without genre count": (["n_genres"], []),
    "without date-added features": (["year_added", "month_added", "years_to_netflix"], []),
    "without director features": (["has_director", "n_directors"], []),
    "without rating": ([], ["rating"]),
    "without country": ([], ["main_country"]),
}


def t2_ablation(df, y, model_name, split=None, folds=5):
    """Cross-validated scores (TRAINING part only) with one group of columns taken away at a time.
    Shows whether the accuracy depends on dataset-specific metadata such as the genre count or the dates."""
    tr, _ = t2_split(y) if split is None else split
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows = {}
    for name, (drop_num, drop_cat) in T2_ABLATIONS.items():
        num = [c for c in T2_NUM if c not in drop_num]
        cat = [c for c in T2_CAT if c not in drop_cat]
        X_tr, y_tr = df[cat + num].iloc[tr], y.iloc[tr]
        res = cross_validate(t2_pipeline(num, t2_models()[model_name], cat), X_tr, y_tr, cv=cv,
                             scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy",
                                      "accuracy": "accuracy"}, n_jobs=1)
        rows[name] = {f"{k} mean": float(res[f"test_{k}"].mean()) for k in ["macro_f1", "balanced_accuracy", "accuracy"]}
        rows[name]["macro_f1 std"] = float(res["test_macro_f1"].std())
    table = pd.DataFrame(rows).T
    table["macro_f1 change vs full"] = table["macro_f1 mean"] - table.iloc[0]["macro_f1 mean"]
    table.index.name = "variant"
    return table


def t2_save_selection(name, row, note, path=T2_SELECTION_FILE):
    """Write the cross-validation choice where the dashboard (train_t2) will find it."""
    with open(path, "w") as f:
        json.dump({"model": name, "cv_macro_f1": round(float(row["cv_macro_f1"]), 4),
                   "cv_macro_f1_std": round(float(row["cv_macro_f1_std"]), 4),
                   "selected_by": "5-fold cross-validation on the training part only", "note": note}, f, indent=2)


def t2_selection_info(path=T2_SELECTION_FILE):
    """The saved choice with its scores ({} if there is none)."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def train_t2(path):
    """Train the Task 2 model and return everything the dashboard needs."""
    df, y = prep_t2(path, fill_years=False)           # gaps are filled inside the pipeline, from training rows only
    X = df[T2_CAT + T2_NUM]
    tr, te = t2_split(y)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    name, source = t2_config()
    model = t2_pipeline(model=t2_models()[name]).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = t2_metrics(y_te, pred, model.predict_proba(X_te)[:, 1])
    counts = df["main_country"].value_counts()

    return {
        "model": model, "model_name": name, "model_source": source, "selection": t2_selection_info(),
        "df": df,
        "test_idx": set(X_te.index),
        "accuracy": accuracy_score(y_te, pred),
        "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
        "report": t2_class_report(y_te, pred),
        "f1": f1_score(y_te, pred),
        "baseline": max(y_te.mean(), 1 - y_te.mean()),
        "cm": confusion_matrix(y_te, pred),
        "n_test": len(y_te),
        "ratings": sorted(df["rating"].dropna().unique().tolist()),
        "countries": counts[counts >= 20].index.tolist() + ["Other"],
        "year_min": int(df["release_year"].min()),
        "year_max": int(df["release_year"].max()),
    }


def t2_row(rating, country, release_year, year_added, month_added,
           has_director, n_directors, n_countries, n_genres):
    """One-row DataFrame in the format the Task 2 model expects."""
    row = {
        "rating": rating,
        "main_country": country,
        "release_year": release_year,
        "year_added": year_added,
        "month_added": month_added,
        "years_to_netflix": year_added - release_year,
        "has_director": int(has_director),
        "n_directors": int(n_directors) if has_director else 0,
        "n_countries": 0 if country == "Unknown" else int(n_countries),
        "n_genres": int(n_genres),
    }
    return pd.DataFrame([row])[T2_CAT + T2_NUM]


def t2_predict(model, row):
    """Probability (0 to 1) that the title is a TV Show."""
    return float(model.predict_proba(row)[0, 1])


# Task 3
# The 12 real ratings are grouped into 4 audience groups (rare ratings like
# NC-17 are too small to learn from). NR / UR have no clear audience, so those
# titles are dropped.
GROUP_ORDER = ["Kids", "Family", "Teens", "Adults"]
RATING_TO_GROUP = {
    "G": "Kids", "TV-G": "Kids", "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids",
    "PG": "Family", "TV-PG": "Family",
    "PG-13": "Teens", "TV-14": "Teens",
    "R": "Adults", "TV-MA": "Adults", "NC-17": "Adults",
}
# Built from RATING_TO_GROUP, so the text shown to people can never drift away from the code.
GROUP_INFO = {g: ", ".join(r for r, grp in RATING_TO_GROUP.items() if grp == g) for g in GROUP_ORDER}
GROUP_INFO["Family"] += " (parental guidance)"
T3_TARGET_NOTE = ("The four audience groups are our own age-suitability groups, built from the content ratings. They are not "
                  "official Netflix labels and not observed audience demographics. G and TV-G mean 'general audiences' "
                  "(everyone), not only children; we put them in Kids because they are the least restricted ratings. "
                  "NR and UR titles are left out. Neighbouring ratings (TV-14 vs TV-MA, PG vs TV-PG) differ by content "
                  "(violence, language) that this data does not hold.")
T3_DISCLAIMER = ("A demonstration model only. It must not decide on its own whether a title is suitable for a child: "
                 "it can be wrong, and mature titles are sometimes predicted as Kids or Family.")

# The rating is the target, so it is NOT a feature. Genres, type and duration
# are fine here (in Task 2 they would give the answer away).
T3_CAT = ["type", "main_country"]
T3_GENRE = "listed_in"
T3_NUM = ["release_year", "year_added", "month_added", "years_to_netflix",
          "has_director", "n_directors", "n_countries", "n_genres",
          "movie_minutes", "seasons"]
T3_FEATURES = T3_CAT + [T3_GENRE] + T3_NUM


# Ratings that mean "no audience group" on purpose. Anything else that does not
# map to a group is unexpected data and is reported.
NOT_GROUPED = {"NR", "UR", "NOT RATED", "UNRATED"}


def prep_t3(path, strict=True):
    """Prepare audience-group features and report excluded ratings.

Missing duration and year values are handled by the training pipeline to avoid test-set leakage."""
    df, _ = prep_t2(path, strict, fill_years=False)
    issues = list(df.attrs.get("issues", []))

    dur = df["duration"].astype(str).str.strip()
    minutes = pd.to_numeric(dur.str.extract(r"^(\d+)\s*min", flags=re.I)[0], errors="coerce")
    seasons = pd.to_numeric(dur.str.extract(r"^(\d+)\s*season", flags=re.I)[0], errors="coerce")
    is_movie = df["type"] == "Movie"
    bad = (is_movie & minutes.isna()) | (~is_movie & seasons.isna())
    if bad.any():
        issues.append(f"{int(bad.sum())} title(s) have a missing or unreadable duration; the model fills it with the "
                      "typical value of the title's own type (median minutes / median seasons) from its training rows only.")
    df["movie_minutes"] = np.where(is_movie, minutes, 0.0)     # 0 for TV shows; empty for a movie with no readable minutes
    df["seasons"] = np.where(is_movie, 0.0, seasons)           # 0 for movies; empty for a show with no readable seasons

    df["listed_in"] = df["listed_in"].fillna("")
    df["rating_group"] = df["rating"].map(RATING_TO_GROUP)
    missing = df["rating"].isna()
    not_rated = df["rating"].fillna("").str.upper().isin(NOT_GROUPED)
    unexpected = df["rating_group"].isna() & ~missing & ~not_rated
    report = {"not_rated": int(not_rated.sum()), "missing": int(missing.sum()),
              "unexpected": df.loc[unexpected, "rating"].value_counts().to_dict()}
    if report["unexpected"]:
        issues.append(f"{int(unexpected.sum())} title(s) have rating values we do not recognise "
                      f"({', '.join(map(str, report['unexpected']))}); they were left out of Task 3.")
    if strict:
        rated = df["rating_group"].notna()
        check_classes(df.loc[rated, "rating_group"], what="audience group")
        check_genres(df.loc[rated, "listed_in"])

    df.attrs["issues"] = issues
    df.attrs["rating_report"] = report
    return df


def t3_model(name, params=None):
    """Build one of the 3 models. `params` overrides the defaults."""
    defaults = {
        "Logistic Regression": (LogisticRegression,
                                dict(max_iter=2000, class_weight="balanced")),
        "Decision Tree": (DecisionTreeClassifier,
                          dict(class_weight="balanced", random_state=SEED)),
        "Random Forest": (RandomForestClassifier,
                          dict(n_estimators=300, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=SEED)),
    }
    cls, base = defaults.get(name, defaults["Random Forest"])
    return cls(**{**base, **(params or {})})


class T3Imputer(BaseEstimator, TransformerMixin):
    """Fill missing Task 3 features using training data only. Movie lengths and TV seasons are handled separately."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        movie = X["type"] == "Movie"
        self.year_ = float(X["release_year"].median())
        self.minutes_ = float(X.loc[movie, "movie_minutes"].median())
        self.seasons_ = float(X.loc[~movie, "seasons"].median())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        movie = X["type"] == "Movie"
        X["release_year"] = X["release_year"].fillna(self.year_)
        gap = X["years_to_netflix"].isna() & X["year_added"].notna()
        X.loc[gap, "years_to_netflix"] = X.loc[gap, "year_added"] - X.loc[gap, "release_year"]
        X.loc[movie & X["movie_minutes"].isna(), "movie_minutes"] = self.minutes_
        X.loc[~movie & X["seasons"].isna(), "seasons"] = self.seasons_
        return X


def t3_pipeline(model, use_genres=True, title_words=False, ablate=None):
    """Imputer + preprocessing + model.

    use_genres=False leaves the genre columns out; title_words=True adds TF-IDF of the title words (needs a
    'title' column in X); ablate is a function applied to X first (used by the ablation experiment).
    """
    genres = [("genres", CountVectorizer(tokenizer=split_items, token_pattern=None, binary=True, min_df=1),
               T3_GENRE)] if use_genres else []
    title = [("title", TfidfVectorizer(stop_words="english", min_df=5, max_features=300,
                                       token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"), "title")] if title_words else []
    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
        ]), T3_CAT),
        # min_df=1 on purpose: with 5, a cross-validation fold could lose every genre and crash.
        # Datasets where all genres are common (like the Netflix file) keep exactly the same features.
        *genres,
        *title,
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), T3_NUM),
    ])
    first = [("ablate", FunctionTransformer(ablate))] if ablate is not None else []
    return Pipeline([*first, ("fill", T3Imputer()), ("prep", prep), ("model", model)])


def t3_config(params_file="task3_best_params.json"):
    """Best model + params saved by task3_rating_classification.py (if it was run)."""
    if os.path.exists(params_file):
        try:
            with open(params_file) as f:
                cfg = json.load(f)
            return cfg["model"], cfg.get("params", {}), "tuned"
        except Exception:
            pass
    return "Random Forest", {}, "default"


def train_t3(path, params_file="task3_best_params.json"):
    all_df = prep_t3(path)
    df = all_df.dropna(subset=["rating_group"]).reset_index(drop=True)
    df.attrs.update(all_df.attrs)
    X, y = df[T3_FEATURES], df["rating_group"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    name, params, source = t3_config(params_file)
    model = t3_pipeline(t3_model(name, params)).fit(X_tr, y_tr)
    pred = model.predict(X_te)

    labels = [g for g in GROUP_ORDER if g in set(y)]
    report = classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.loc[labels, ["precision", "recall", "f1-score", "support"]]
    counts = df["main_country"].value_counts()

    return {
        "model": model, "model_name": name, "params_source": source, "df": df,
        "test_idx": set(X_te.index),
        "accuracy": accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, average="macro"),
        "baseline": y_te.value_counts(normalize=True).max(),
        "cm": confusion_matrix(y_te, pred, labels=labels), "labels": labels,
        "per_class": per_class, "n_test": len(y_te), "n_dropped": len(all_df) - len(df),
        "rating_report": all_df.attrs["rating_report"],
        "error_table": t3_error_table(y_te, pred), "safety": t3_safety(y_te, pred), "cost_matrix": t3_cost_matrix(),
        "countries": counts[counts >= 20].index.tolist() + ["Other"],
        "genres": get_genres(df),
        "year_min": int(df["release_year"].min()), "year_max": int(df["release_year"].max()),
    }


def t3_row(type_, country, genres, release_year, year_added, month_added,
           has_director, n_directors, n_countries, movie_minutes, seasons):
    """One-row DataFrame in the format the Task 3 model expects."""
    is_movie = type_ == "Movie"
    row = {
        "type": type_,
        "main_country": country,
        "listed_in": ", ".join(genres),
        "release_year": release_year,
        "year_added": year_added,
        "month_added": month_added,
        "years_to_netflix": year_added - release_year,
        "has_director": int(has_director),
        "n_directors": int(n_directors) if has_director else 0,
        "n_countries": 0 if country == "Unknown" else int(n_countries),
        "n_genres": len(genres),
        "movie_minutes": float(movie_minutes) if is_movie else 0.0,
        "seasons": 0.0 if is_movie else float(seasons),
    }
    return pd.DataFrame([row])[T3_FEATURES]


def t3_predict(model, row):
    """{'Kids': p, 'Family': p, ...} for one row, in audience order."""
    probs = dict(zip(model.classes_, model.predict_proba(row)[0]))
    return {g: float(probs[g]) for g in GROUP_ORDER if g in probs}


# Age-group error costs
T3_RANK = {g: i for i, g in enumerate(GROUP_ORDER)}      # Kids 0 < Family 1 < Teens 2 < Adults 3
T3_UNDER_COST, T3_OVER_COST = 2, 1       # predicting a LESS mature group than the truth costs double, per step
T3_REVIEW_THRESHOLD = 0.5                # a top model score below this means 'Review required'
T3_REVIEW = "Review required"


def t3_cost_matrix():
    """Confusion-cost matrix (rows = true group, columns = predicted group). A step too lenient costs
    T3_UNDER_COST, a step too strict T3_OVER_COST, a correct answer 0."""
    def cost(true, pred):
        d = T3_RANK[pred] - T3_RANK[true]
        return T3_OVER_COST * d if d > 0 else T3_UNDER_COST * -d
    return pd.DataFrame({p: {t: cost(t, p) for t in GROUP_ORDER} for p in GROUP_ORDER})


def t3_safety(y_true, y_pred):
    """The errors that matter most: mature titles shown as suitable for younger groups.

    too_lenient_rate     share of ALL titles predicted as a less mature group than the real one
    severe_lenient_rate  share of Teens + Adults titles predicted as Kids or Family
    adults_as_kids       Adults titles predicted as Kids
    mean_cost            average cost per title with t3_cost_matrix()
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rt = np.array([T3_RANK[v] for v in yt])
    rp = np.array([T3_RANK[v] for v in yp])
    older = rt >= T3_RANK["Teens"]
    severe = older & (rp <= T3_RANK["Family"])
    cost = t3_cost_matrix()
    return {"too_lenient": int((rp < rt).sum()), "too_lenient_rate": float((rp < rt).mean()),
            "too_strict_rate": float((rp > rt).mean()),
            "severe_lenient": int(severe.sum()), "older_titles": int(older.sum()),
            "severe_lenient_rate": float(severe.sum() / older.sum()) if older.any() else 0.0,
            "adults_as_kids": int(((yt == "Adults") & (yp == "Kids")).sum()),
            "mean_cost": float(np.mean([cost.loc[a, b] for a, b in zip(yt, yp)]))}


def t3_error_table(y_true, y_pred):
    """Per-group error analysis: how many titles, how many are right, where the others go, and whether
    the mistakes are small (one step away) and in which direction (too lenient / too strict)."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rows = {}
    for g in GROUP_ORDER:
        m = yt == g
        if not m.any():
            continue
        pred = yp[m]
        steps = np.array([T3_RANK[v] - T3_RANK[g] for v in pred])
        row = {"titles": int(m.sum()), "correct": float((pred == g).mean())}
        for h in GROUP_ORDER:
            row[f"predicted {h}"] = float((pred == h).mean())
        wrong = steps != 0
        row["errors one step away"] = float((np.abs(steps[wrong]) == 1).mean()) if wrong.any() else np.nan
        row["too lenient"] = float((steps < 0).mean())
        row["too strict"] = float((steps > 0).mean())
        rows[g] = row
    table = pd.DataFrame(rows).T
    table["titles"] = table["titles"].astype(int)
    return table


def t3_review_label(probs, threshold=T3_REVIEW_THRESHOLD):
    """(label, top score). The label is 'Review required' when the model is not sure enough."""
    best = max(probs, key=probs.get)
    return (best if probs[best] >= threshold else T3_REVIEW), float(probs[best])


def t3_review_curve(y_true, proba, thresholds=(0.0, 0.4, 0.5, 0.6, 0.7)):
    """What 'Review required' costs and buys. proba = DataFrame with one column of scores per group
    (use out-of-fold scores from the training part). For each threshold: how many titles still get an
    automatic answer, and how good those answers are."""
    proba = proba[[g for g in GROUP_ORDER if g in proba.columns]]
    yt, best, top = np.asarray(y_true), proba.idxmax(axis=1).values, proba.max(axis=1).values
    rows = []
    for th in thresholds:
        keep = top >= th
        if not keep.any():
            continue
        safe = t3_safety(yt[keep], best[keep])
        rows.append({"threshold": th, "answered": float(keep.mean()), "answered titles": int(keep.sum()),
                     "sent to review": int((~keep).sum()),
                     "accuracy of answered": float((yt[keep] == best[keep]).mean()),
                     "macro-F1 of answered": float(f1_score(yt[keep], best[keep], average="macro", zero_division=0)),
                     "mature shown as Kids/Family (answered)": safe["severe_lenient_rate"]})
    return pd.DataFrame(rows)


# Ordinal alternative
class T3Ordinal(BaseEstimator, ClassifierMixin):
    """Ordinal classifier (Frank & Hall) for Kids < Family < Teens < Adults.

    One binary model per step of the ladder ('at least Family?', 'at least Teens?', 'at least Adults?').
    The group scores are the differences between the steps, so the order of the groups is used;
    the predicted group is the median of those scores (see predict).
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        y = np.asarray(y)
        if not set(y.tolist()) <= set(GROUP_ORDER):
            raise ValueError("T3Ordinal needs the group names (Kids, Family, Teens, Adults) to know their order. "
                             "cross_val_predict(method='predict_proba') replaces them by numbers: use cross_validate "
                             "or fit / predict_proba directly instead.")
        self.classes_ = np.array([g for g in GROUP_ORDER if g in set(y)])
        rank = np.array([int(np.where(self.classes_ == v)[0][0]) for v in y])
        base = self.base if self.base is not None else RandomForestClassifier(n_estimators=200, random_state=SEED)
        self.models_ = [clone(base).fit(X, (rank >= k).astype(int)) for k in range(1, len(self.classes_))]
        return self

    def _above(self, X):
        """P(group is at least step k) for k = 1..3, shape (steps, rows); never increases with k."""
        above = []
        for m in self.models_:
            p = m.predict_proba(X)
            above.append(p[:, list(m.classes_).index(1)] if 1 in m.classes_ else np.zeros(p.shape[0]))
        return np.minimum.accumulate(np.clip(np.array(above), 0, 1), axis=0)

    def predict_proba(self, X):
        above = self._above(X)
        n_rows = above.shape[1]
        steps = np.vstack([np.ones(n_rows), above, np.zeros(n_rows)])
        return (steps[:-1] - steps[1:]).T

    def predict(self, X):
        """The median group: the first one at which 'at least the next step' is no longer more likely than not.
        (Taking the highest of the four scores instead rarely picks the two middle groups.)"""
        return self.classes_[(self._above(X) > 0.5).sum(axis=0)]


# Audience-feature ablation
T3_AUDIENCE_GENRES = ("kids' tv", "children & family movies", "teen tv shows")
T3_ABLATIONS = ["full", "without audience genres", "without all genres", "without duration", "without date features"]


def t3_drop_audience_genres(X):
    X = X.copy()
    kept = [", ".join(g for g in split_items(s) if g.lower() not in T3_AUDIENCE_GENRES) for s in X["listed_in"]]
    X["listed_in"] = kept
    X["n_genres"] = [len(split_items(s)) for s in kept]
    return X


def t3_drop_all_genres(X):
    X = X.copy()
    X["listed_in"], X["n_genres"] = "", 0
    return X


def t3_drop_duration(X):
    X = X.copy()
    X["movie_minutes"], X["seasons"] = 0.0, 0.0
    return X


def t3_drop_dates(X):
    X = X.copy()
    X["year_added"], X["month_added"], X["years_to_netflix"] = 0.0, 0.0, 0.0
    return X


def t3_ablation_pipeline(model, variant):
    """The Task 3 pipeline with one group of features taken away."""
    if variant == "full":
        return t3_pipeline(model)
    if variant == "without audience genres":
        return t3_pipeline(model, ablate=t3_drop_audience_genres)
    if variant == "without all genres":
        return t3_pipeline(model, use_genres=False, ablate=t3_drop_all_genres)
    if variant == "without duration":
        return t3_pipeline(model, ablate=t3_drop_duration)
    if variant == "without date features":
        return t3_pipeline(model, ablate=t3_drop_dates)
    raise ValueError(f"Unknown ablation variant: {variant!r}")


# Summary text for the results
def t3_all_tuned(params_file="task3_best_params.json"):
    """Tuned settings of all three models (saved by task3_rating_classification.py); {} if the file has none."""
    if os.path.exists(params_file):
        try:
            with open(params_file) as f:
                return json.load(f).get("all_tuned", {})
        except Exception:
            pass
    return {}


def t3_significance_text(sig):
    """Plain sentences from task3_model_significance.csv. Returns (range level, sentences).

    The number of comparisons in the file sets the range level (the script widens each range because
    several gaps are checked at once): 6 comparisons -> 1 - 0.05/6 = 99.2%, not 95%.
    """
    level = 1 - 0.05 / max(len(sig), 1)
    names = {"accuracy": "accuracy", "macro_f1": "macro-F1"}
    sentences = []
    for r in sig.itertuples():
        a, b, gap, low, high = r.A, r.B, r.gap_A_minus_B, r.low, r.high
        if gap < 0:                                      # always say it from the better model's side
            a, b, gap, low, high = b, a, -gap, -high, -low
        metric = names.get(r.metric, r.metric)
        if bool(r.clearly_different):
            sentences.append(f"{a} beat {b} by {gap * 100:.1f} points in {metric} "
                             f"(range {low * 100:+.1f} to {high * 100:+.1f}): a clear gap.")
        else:
            sentences.append(f"{a} vs {b}: {gap * 100:.1f} points of {metric} in favour of {a} "
                             f"(range {low * 100:+.1f} to {high * 100:+.1f}): not clearly different.")
    return level, sentences


def t3_tuning_text(comp):
    """What tuning did to each model, in plain words. Tuning aimed at macro-F1, so accuracy can fall."""
    lines, signs = [], []
    for m in [x for x in comp["model"].unique() if x != "Baseline (majority)"]:
        d = comp[(comp["model"] == m) & (comp["version"] == "default")].iloc[0]
        t = comp[(comp["model"] == m) & (comp["version"] == "tuned")].iloc[0]
        da, df_ = (t["accuracy"] - d["accuracy"]) * 100, (t["macro_f1"] - d["macro_f1"]) * 100
        text = f"{m}: on the test set tuning changed accuracy by {da:+.1f} and macro-F1 by {df_:+.1f} points"
        if pd.notna(d.get("cv_macro_f1")) and pd.notna(t.get("cv_macro_f1")):
            text += f"; in cross-validation macro-F1 went from {d['cv_macro_f1']:.3f} to {t['cv_macro_f1']:.3f}"
        lines.append(text + ".")
        signs += [np.sign(round(da, 1)), np.sign(round(df_, 1))]
    mixed = len(set(s for s in signs if s != 0)) > 1
    summary = ("Tuning aimed at macro-F1, not accuracy, and the results are mixed: it did not improve every model "
               "on every measure." if mixed else
               "Tuning moved every measure in the same direction for every model.")
    return lines, summary


# Task 5
# Forecast how many titles are added to Netflix each month (from date_added).
# Only numpy + scikit-learn are used (no extra package to install).
# Rules that matter most for a time series:
#   - the test months are always the LAST ones, never a random sample
#   - nothing about the test months may influence a choice made before the test:
#     not the model, and not which early months are skipped
#   - every model has to beat the three simple rules (last month, same month last year,
#     12-month average) to count as useful
REQUIRED_T5 = ["type", "date_added"]     # show_id is optional (it only helps to spot repeated rows)
T5_HOLDOUT = 12          # months kept back for the final test
T5_CV_H = 12             # rolling tests look this many months ahead ...
T5_CV_ORIGINS = 5        # ... from this many start dates ...
T5_CV_STEP = 3           # ... spaced this many months apart
T5_MIN_TRAIN = 24        # every model needs at least this many months to learn from
T5_MIN_MONTHS = T5_MIN_TRAIN + T5_CV_H + T5_HOLDOUT      # 48 months in total
T5_SPARSE_SHARE = 0.15   # a month below this share of a typical month is 'sparse' (early years with a handful of titles)
T5_RANGE_NOTE = ("The shaded range is the forecast plus or minus the model's typical miss (RMSE) on the final test. "
                 "It is illustrative only: it is not a prediction interval and it comes with no coverage guarantee.")
T5_TARGET_NOTE = ("What is forecast: the number of titles ADDED to the Netflix catalogue each month (date_added). "
                  "It is not the number of films or series first released that month: a title released years ago "
                  "can be added much later.")


def _t5_parse_dates(text):
    """Read dates written like 9/25/2021 (month first) or 'September 25, 2021'. Unreadable ones become NaT."""
    parsed = pd.to_datetime(text, format="%m/%d/%Y", errors="coerce")
    rest = parsed.isna() & text.notna()
    if rest.any():
        with warnings.catch_warnings():              # pandas warns when it has to guess each date's style
            warnings.simplefilter("ignore")
            parsed[rest] = pd.to_datetime(text[rest], errors="coerce")
    return parsed


def prep_t5(path, strict=True):
    """Clean dated Netflix additions for Task 5.

Handle duplicates, missing dates and incomplete months separately from the classification tasks. Return the cleaned dates and a record of exclusions."""
    raw = read_data(path)
    check_columns(raw, REQUIRED_T5)
    issues = []

    if "show_id" in raw.columns:
        repeated = raw.duplicated(subset="show_id") & raw["show_id"].notna()
    else:
        repeated = pd.Series(False, index=raw.index)
        issues.append("There is no show_id column, so repeated rows cannot be detected: every row is counted.")
    n_repeated = int(repeated.sum())
    df = raw[~repeated]
    if n_repeated:
        issues.append(f"{n_repeated} row(s) repeat an earlier show_id and were counted once.")

    text = df["date_added"].astype(str).str.strip()
    text = text.where(~text.isin(["", "nan", "NaN", "None", "Not Given"]))
    dates = _t5_parse_dates(text)
    missing = text.isna()
    unreadable = text.notna() & dates.isna()
    n_missing, n_unreadable = int(missing.sum()), int(unreadable.sum())
    if n_missing:
        issues.append(f"{n_missing} title(s) have no date_added; they cannot be placed in a month "
                      "and are left out of the monthly counts.")
    if n_unreadable:
        example = text[unreadable].iloc[0]
        issues.append(f"{n_unreadable} title(s) have a date_added that could not be read (for example '{example}'); "
                      "they are left out of the monthly counts. Dates are read month first (9/25/2021 is 25 September 2021).")

    types = df["type"].astype(str).str.strip().where(df["type"].notna())
    unexpected = dates.notna() & ~types.isin(EXPECTED_TYPES)
    if unexpected.any():
        seen = sorted(types[unexpected].fillna("(missing)").unique())[:5]
        issues.append(f"{int(unexpected.sum())} dated title(s) have a type other than Movie / TV Show "
                      f"({', '.join(map(str, seen))}); they count in 'All titles' but not in the Movie or TV Show series.")

    out = pd.DataFrame({"date_added": dates, "type": types.where(types.isin(EXPECTED_TYPES))})
    out = out[dates.notna()].reset_index(drop=True)
    if out.empty:
        raise ValueError("The date_added column has no usable dates: every value is missing or cannot be read "
                         "(dates are expected like 9/25/2021 or 'September 25, 2021'). "
                         "Task 5 needs real dates to count titles per month.")
    if strict:
        note = check_size(out)               # raises if there are too few titles
        if note:
            issues.append(note)

    out.attrs["issues"] = issues
    out.attrs["quality"] = {"rows_read": len(raw), "repeated_removed": n_repeated,
                            "missing_date": n_missing, "unreadable_date": n_unreadable,
                            "dated": len(out), "unexpected_type": int(unexpected.sum())}
    return out


def _t5_auto_start(series, holdout=T5_HOLDOUT):
    """First month of the history to use: skips early years that have only a handful of titles.

    Only the months BEFORE the final test are looked at (both for the 'typical month' and for
    the rule itself), so the test months cannot influence which history is used.
    """
    before_test = series.iloc[:-holdout]
    if len(before_test) < T5_MIN_TRAIN:          # too short to judge: keep everything (the length check reports it)
        return series.index[0]
    floor = T5_SPARSE_SHARE * before_test.iloc[-24:].median()
    normal = (before_test >= floor).values
    for i in range(len(before_test)):
        if normal[i] and normal[i:].mean() >= 0.95:      # the first month must be a normal one too
            return before_test.index[i]
    return before_test.index[0]


def t5_series(path, kind=None, start=None, strict=True):
    """Build monthly counts of titles added to Netflix and return series metadata. The start month can be set explicitly."""
    if kind not in (None, *EXPECTED_TYPES):
        raise ValueError(f"kind must be None, 'Movie' or 'TV Show', not {kind!r}.")
    dated = prep_t5(path, strict)
    issues = list(dated.attrs["issues"])
    quality = dict(dated.attrs["quality"])

    first_month = dated["date_added"].min().to_period("M")
    last = dated["date_added"].max()
    partial = bool(last.day < last.days_in_month - 2)
    end = last.to_period("M") - 1 if partial else last.to_period("M")
    if end < first_month:
        raise ValueError("All the dates fall in one month that is not complete "
                         f"(the data stops on {last:%Y-%m-%d}), so there is no complete month to count.")

    full = pd.period_range(first_month, end, freq="M")
    periods = dated["date_added"].dt.to_period("M")

    def monthly(mask):
        counts = periods[mask].value_counts()
        s = counts.reindex(full, fill_value=0).sort_index().astype(float)
        s.index = s.index.to_timestamp()
        return s

    all_titles = monthly(pd.Series(True, index=dated.index))
    if start is None:
        start_ts = _t5_auto_start(all_titles)
    else:
        start_ts = pd.Timestamp(start).to_period("M").to_timestamp()
        start_ts = max(start_ts, all_titles.index[0])
        if start_ts > all_titles.index[-1]:
            raise ValueError(f"The start month {start_ts:%Y-%m} is after the last complete month "
                             f"({all_titles.index[-1]:%Y-%m}).")

    series = monthly(pd.Series(True, index=dated.index) if kind is None else dated["type"] == kind)
    series = series[series.index >= start_ts]
    if series.empty:
        raise ValueError("No months are left after choosing the start month.")
    series.name = kind or "All titles"

    n_partial = int((periods > end).sum())
    before_start = int(all_titles[all_titles.index < start_ts].sum())
    quality.update({"before_start": before_start, "partial_month": n_partial,
                    "in_series": int(all_titles[all_titles.index >= start_ts].sum())})
    quality["excluded_total"] = quality["rows_read"] - quality["in_series"]
    if before_start:
        n_early = int((all_titles.index < start_ts).sum())
        issues.append(f"{before_start} title(s) added in the {n_early} early month(s) before {start_ts:%Y-%m} were left out: "
                      "those months have only a handful of titles each and would distort the trend.")
    if partial:
        issues.append(f"The data stops on {last:%Y-%m-%d}, in the middle of a month, so that last month "
                      f"({n_partial} title(s)) was left out.")

    info = {"kind": kind, "first_date": dated["date_added"].min(), "last_date": last,
            "dropped_partial_month": partial, "start": series.index[0], "end": series.index[-1],
            "n_months": len(series), "n_titles": int(series.sum()), "n_dated": len(dated),
            "months_before_skipping": len(all_titles), "issues": issues, "quality": quality}
    if strict and len(series) < T5_MIN_MONTHS:
        raise ValueError(f"Not enough history: {len(series)} usable months ({series.index[0]:%Y-%m} to "
                         f"{series.index[-1]:%Y-%m}) but at least {T5_MIN_MONTHS} are needed "
                         f"({T5_MIN_TRAIN} to train, {T5_CV_H} for the rolling tests, {T5_HOLDOUT} for the final test). "
                         f"The file has {len(all_titles)} complete months before skipping early sparse ones. "
                         "Use a dataset with a longer date_added range.")
    return series, info


def t5_accounting(info):
    """Where every record of the file went, as a small table (for the report and the dashboard)."""
    q = info["quality"]
    rows = [
        ("Rows in the file", q["rows_read"]),
        ("  left out: same show_id again", -q["repeated_removed"]),
        ("  left out: date_added missing", -q["missing_date"]),
        ("  left out: date_added could not be read", -q["unreadable_date"]),
        ("Titles with a usable date", q["dated"]),
        ("  left out: added in the early months that have only a handful of titles", -q["before_start"]),
        ("  left out: added in the last, incomplete month", -q["partial_month"]),
        ("Titles in the monthly series (all titles)", q["in_series"]),
        ("TOTAL left out of the series", q["excluded_total"]),
        ("(of the series, type is not Movie / TV Show: in 'All titles' only)", q["unexpected_type"]),
    ]
    return pd.DataFrame(rows, columns=["Step", "Titles"])


def _t5_future_index(train, h):
    return pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def _t5_month_matrix(index):
    """One 0/1 column per calendar month (January is the reference, so 11 columns)."""
    return np.eye(12)[index.month.values - 1][:, 1:]


# Forecasting models
def t5_naive(train, h):
    """Next months = last month."""
    return np.repeat(float(train.iloc[-1]), h)


def t5_seasonal_naive(train, h):
    """Each month = the same month one year earlier."""
    if len(train) < 12:
        raise ValueError("need 12 months")
    last = train.iloc[-12:].values
    return np.array([last[i % 12] for i in range(h)], dtype=float)


def t5_mean12(train, h):
    """Next months = average of the last 12 months."""
    return np.repeat(float(train.iloc[-12:].mean()), h)


def t5_trend_season(train, h):
    """Straight-line trend plus a fixed effect for each calendar month."""
    if len(train) < T5_MIN_TRAIN:
        raise ValueError(f"need {T5_MIN_TRAIN} months")
    t = np.arange(len(train)).reshape(-1, 1)
    model = LinearRegression().fit(np.hstack([t, _t5_month_matrix(train.index)]), train.values)
    future = _t5_future_index(train, h)
    tf = np.arange(len(train), len(train) + h).reshape(-1, 1)
    return np.clip(model.predict(np.hstack([tf, _t5_month_matrix(future)])), 0, None)


def _hw_run(y, m, alpha, beta, gamma, phi):
    """One pass of additive Holt-Winters. Returns (sum of squared one-step errors, final states)."""
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    season = list(y[:m] - level)
    sse = 0.0
    for t in range(m, len(y)):
        forecast = level + phi * trend + season[t - m]
        sse += (y[t] - forecast) ** 2
        new_level = alpha * (y[t] - season[t - m]) + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        season.append(gamma * (y[t] - new_level) + (1 - gamma) * season[t - m])
        level = new_level
    return sse, level, trend, season


def t5_holt_winters(train, h):
    """Exponential smoothing with a (damped) trend and yearly seasonality.
    The smoothing settings are picked by the smallest one-step error on the training data only."""
    y, m = train.values.astype(float), 12
    if len(y) < 2 * m:
        raise ValueError("need 24 months")
    best = None
    for alpha in (0.1, 0.3, 0.5, 0.7):
        for beta in (0.01, 0.05, 0.1, 0.3):
            for gamma in (0.1, 0.3, 0.5):
                for phi in (0.8, 0.9, 0.98, 1.0):
                    sse, level, trend, season = _hw_run(y, m, alpha, beta, gamma, phi)
                    if best is None or sse < best[0]:
                        best = (sse, level, trend, season, phi)
    _, level, trend, season, phi = best
    n = len(y)
    out = []
    for k in range(1, h + 1):
        damp = sum(phi ** i for i in range(1, k + 1))
        out.append(level + damp * trend + season[n - m + ((k - 1) % m)])
    return np.clip(np.array(out), 0, None)


def _t5_lag_features(history, month):
    history = np.asarray(history)
    return [history[-1], history[-2], history[-3], history[-12],
            history[-3:].mean(), history[-12:].mean(),
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]


def t5_forest(train, h):
    """Random Forest that predicts next month from recent months (lags), one step at a time.
    Trees cannot continue a trend, so this model tends to forecast a flat level."""
    y = train.values.astype(float)
    if len(y) < T5_MIN_TRAIN:
        raise ValueError(f"need {T5_MIN_TRAIN} months")
    months = train.index.month.values
    X = [_t5_lag_features(y[:i], months[i]) for i in range(12, len(y))]
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED).fit(X, y[12:])
    history, out = list(y), []
    for stamp in _t5_future_index(train, h):
        p = float(model.predict([_t5_lag_features(history, stamp.month)])[0])
        out.append(p)
        history.append(p)
    return np.clip(np.array(out), 0, None)


T5_MODELS = {
    "Naive (last month)": t5_naive,
    "Seasonal naive (same month last year)": t5_seasonal_naive,
    "12-month average": t5_mean12,
    "Trend + season regression": t5_trend_season,
    "Holt-Winters": t5_holt_winters,
    "Random Forest (lags)": t5_forest,
}
T5_BASELINE = "Seasonal naive (same month last year)"
T5_SIMPLE_RULES = ["Naive (last month)", T5_BASELINE, "12-month average"]   # what a fancier model has to beat
T5_DEFAULT_MODEL = "12-month average"   # kept unless another model clearly beats it in the rolling tests
T5_MIN_GAIN = 0.10                      # "clearly" = at least 10% lower average miss in the rolling tests


# Rolling evaluation
def _t5_origin_ends(n, h, n_origins, step):
    """Where each rolling test stops training. Every model needs T5_MIN_TRAIN months, so a short series gets fewer start dates."""
    n_origins = min(n_origins, (n - h - T5_MIN_TRAIN) // step + 1)
    if n_origins < 1:
        raise ValueError("The series is too short for the rolling tests.")
    return [n - h - step * (n_origins - 1 - k) for k in range(n_origins)]


def t5_rolling_detail(series, h=T5_CV_H, n_origins=T5_CV_ORIGINS, step=T5_CV_STEP, models=None):
    """Evaluate forecasting models across rolling training windows. Keep the per-origin and per-horizon errors for inspection."""
    models = models or T5_MODELS
    ends = _t5_origin_ends(len(series), h, n_origins, step)
    errors = {name: np.zeros((len(ends), h)) for name in models}
    for i, end in enumerate(ends):
        train, test = series.iloc[:end], series.iloc[end:end + h]
        for name, fn in models.items():
            errors[name][i] = test.values - fn(train, h)
    summary = pd.DataFrame({"MAE": {n: np.abs(e).mean() for n, e in errors.items()},
                            "RMSE": {n: np.sqrt((e ** 2).mean()) for n, e in errors.items()},
                            "forecasts_scored": {n: e.size for n, e in errors.items()}})
    summary.index.name = "model"
    by_origin = pd.DataFrame({n: np.abs(e).mean(axis=1) for n, e in errors.items()},
                             index=[f"{series.index[e]:%Y-%m}" for e in ends])
    by_origin.index.name = "first forecast month"
    horizons = [k for k in (1, 3, 6, 12) if k <= h]
    by_horizon = pd.DataFrame({n: [np.abs(e[:, k - 1]).mean() for k in horizons] for n, e in errors.items()},
                              index=[f"{k} month{'s' if k > 1 else ''} ahead" for k in horizons])
    return {"summary": summary, "by_origin": by_origin, "by_horizon": by_horizon,
            "errors": errors, "n_origins": len(ends)}


def t5_rolling_origin(series, h=T5_CV_H, n_origins=T5_CV_ORIGINS, step=T5_CV_STEP, models=None):
    """The summary table of t5_rolling_detail (average miss per model)."""
    return t5_rolling_detail(series, h, n_origins, step, models)["summary"]


def t5_holdout(series, h=T5_HOLDOUT, models=None):
    """Train on all but the last h months, forecast those h months, compare with what happened."""
    models = models or T5_MODELS
    train, test = series.iloc[:-h], series.iloc[-h:]
    forecasts = pd.DataFrame({name: fn(train, h) for name, fn in models.items()}, index=test.index)
    err = forecasts.rsub(test.values, axis=0)
    table = pd.DataFrame({"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean())})
    return table.sort_values("MAE"), forecasts, test


def t5_future(series, model_name, h=12, rmse=0.0):
    """Refit on ALL the data and forecast the next h months.

    range_low / range_high = forecast minus / plus `rmse` (the model's typical miss on the final test).
    This is an illustrative error range, NOT a calibrated prediction interval (see T5_RANGE_NOTE).
    """
    pred = T5_MODELS[model_name](series, h)
    return pd.DataFrame({"forecast": pred, "range_low": np.clip(pred - rmse, 0, None), "range_high": pred + rmse},
                        index=_t5_future_index(series, h))


def t5_choose(cv):
    """Choose a forecasting model from training-only rolling MAE. Keep the 12-month average unless another model improves on it by at least 10%."""
    best = cv["MAE"].idxmin()
    default_miss, best_miss = cv.loc[T5_DEFAULT_MODEL, "MAE"], cv.loc[best, "MAE"]
    gain = float(1 - best_miss / default_miss) if default_miss > 0 else 0.0
    if best == T5_DEFAULT_MODEL:
        return best, f"The {T5_DEFAULT_MODEL} had the lowest error in the rolling tests.", 0.0
    # compared this way (not gain >= 0.10) so that exactly 10% better counts despite rounding in floating point
    if best_miss <= (1 - T5_MIN_GAIN) * default_miss * (1 + 1e-9):
        return best, (f"{best} beat the {T5_DEFAULT_MODEL} by {gain:.0%} in the rolling tests "
                      f"(at least {T5_MIN_GAIN:.0%} is needed), so it was chosen."), gain
    return T5_DEFAULT_MODEL, (f"{best} had the lowest rolling-test error but was only {gain:.0%} better than the "
                              f"{T5_DEFAULT_MODEL} (at least {T5_MIN_GAIN:.0%} is needed), so the "
                              f"{T5_DEFAULT_MODEL} was kept."), gain


def t5_study(series, holdout=T5_HOLDOUT, cv_h=T5_CV_H, cv_origins=T5_CV_ORIGINS, cv_step=T5_CV_STEP):
    """The full workflow for one monthly series.
    1) choose the model with rolling tests inside the TRAINING part only (see t5_choose for the 10% rule),
    2) score every model once on the last `holdout` months,
    3) refit the chosen model on everything and forecast the future."""
    if len(series) < T5_MIN_MONTHS:
        raise ValueError(f"Not enough history: {len(series)} months, at least {T5_MIN_MONTHS} are needed.")
    detail = t5_rolling_detail(series.iloc[:-holdout], h=cv_h, n_origins=cv_origins, step=cv_step)
    cv, n_orig = detail["summary"], detail["n_origins"]
    selected, selection_note, _ = t5_choose(cv)
    best = cv["MAE"].idxmin()
    if best != T5_DEFAULT_MODEL:          # is the best model's edge consistent, or one lucky start date?
        wins = int((detail["by_origin"][best] < detail["by_origin"][T5_DEFAULT_MODEL]).sum())
        selection_note += f" {best} was better than the {T5_DEFAULT_MODEL} at {wins} of {n_orig} start dates."
    hold, hold_fc, test = t5_holdout(series, holdout)
    rmse = hold.loc[selected, "RMSE"]
    future = t5_future(series, selected, 12, rmse=rmse)
    default_future = pd.Series(T5_MODELS[T5_DEFAULT_MODEL](series, 12), index=future.index)   # Stable reference
    rules = hold.loc[T5_SIMPLE_RULES, "MAE"]
    best_rule = rules.idxmin()            # Final test is not used to select models
    skill = 1 - hold.loc[selected, "MAE"] / rules.min()
    # How often would the shaded range have held? Checked on the rolling-test errors (other months than the
    # ones the range width came from), and how the typical miss changes with the months ahead.
    errs = detail["errors"][selected]
    coverage = float((np.abs(errs) <= rmse).mean())
    err_by_horizon = pd.Series({f"months {a + 1}-{min(a + 3, cv_h)}": float(np.sqrt((errs[:, a:a + 3] ** 2).mean()))
                                for a in range(0, cv_h, 3)})
    return {"series": series, "cv": cv.sort_values("MAE"), "holdout": hold,
            "holdout_fc": hold_fc, "holdout_actual": test, "selected": selected,
            "best_on_final_test": hold.index[0], "selection_note": selection_note,
            "best_in_rolling_tests": best,
            "future": future, "default_future": default_future, "skill_vs_baseline": float(skill),
            "best_simple_rule": best_rule, "holdout_months": holdout, "cv_h": cv_h,
            "cv_by_origin": detail["by_origin"], "cv_by_horizon": detail["by_horizon"], "cv_origins": n_orig,
            "range_coverage": coverage, "error_by_horizon": err_by_horizon}


def t5_verdict(res):
    """One honest sentence about whether the chosen model beats the simple rules.

    The yardstick is the best simple rule on the final test (the strictest fair comparison).
    It is only a yardstick: the model is never switched after seeing the test.
    skill_vs_baseline = 1 - (model's average miss / that rule's average miss).
    """
    sel, n, rule = res["selected"], res["holdout_months"], res["best_simple_rule"]
    miss = res["holdout"]["MAE"]
    if sel in T5_SIMPLE_RULES:
        if sel == T5_DEFAULT_MODEL:
            text = (f"No fancier model beat the {sel} by at least {T5_MIN_GAIN:.0%} in the rolling tests, "
                    "so no fancier model earned its place.")
        else:
            text = f"The model chosen by the rolling tests, {sel}, is itself a simple rule, so no fancier model earned its place."
        if rule != sel:
            text += (f" On the last {n} months {rule} did better ({miss[rule]:.0f} vs {miss[sel]:.0f} titles "
                     "average miss), but the choice is not changed after seeing the test.")
        return text
    skill = res["skill_vs_baseline"]
    if skill >= 0.10:
        return (f"{sel} cut the error on the last {n} months by {skill:.0%} compared with the best simple rule "
                f"({rule}): a real improvement.")
    if skill > -0.10:
        return (f"{sel} was chosen, but its error on the last {n} months is within 10% of the best simple rule "
                f"({rule}). Honest conclusion: no clear improvement.")
    return (f"{sel} was chosen by the rolling tests, but on the last {n} months it did worse than the best "
            f"simple rule, {rule} ({skill:.0%}). Report this plainly.")


def t5_snapshot_note(info, future_index, today=None):
    """Say plainly that the forecast comes from a data snapshot, and whether its months have already passed."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    text = (f"Forecast made from a data snapshot that ends {info['last_date']:%d %b %Y} "
            f"(last complete month: {info['end']:%b %Y}). It covers {future_index[0]:%b %Y} to {future_index[-1]:%b %Y}.")
    if future_index[-1] + pd.offsets.MonthEnd(0) < today:
        text += " Those months have already passed, so this is a demonstration on historical data, not a current forecast."
    return text


def t5_year_labels(series):
    """Tick labels for a yearly chart: a year with fewer than 12 months gets '(n mo)' so it is not misread."""
    months = series.groupby(series.index.year).size()
    return [str(y) if n == 12 else f"{y}\n({n} mo)" for y, n in months.items()]


def t5_reconcile(all_res, movie_res, tv_res, holdout=T5_HOLDOUT, h=T5_CV_H, n_origins=T5_CV_ORIGINS, step=T5_CV_STEP):
    """Compare and reconcile separately fitted forecasts so Movies + TV Shows equals All titles."""
    sa, sm, st_ = all_res["series"], movie_res["series"], tv_res["series"]
    if not (sa.index.equals(sm.index) and sa.index.equals(st_.index) and np.allclose(sa.values, sm.values + st_.values)):
        return {"ok": False, "note": "Movies + TV Shows do not add up to All titles in this file (some rows have "
                                     "another type), so the forecasts cannot be made to add up exactly."}
    names = [all_res["selected"], movie_res["selected"], tv_res["selected"]]
    methods = ("independent", "bottom-up", "top-down")

    def forecasts(train_a, train_m, train_t, months):
        fa, fm, ft = (T5_MODELS[n](t, months) for n, t in zip(names, (train_a, train_m, train_t)))
        m12, t12 = train_m.iloc[-12:].sum(), train_t.iloc[-12:].sum()
        share = float(m12 / (m12 + t12)) if (m12 + t12) > 0 else 0.5
        return {"independent": (fa, fm, ft), "bottom-up": (fm + ft, fm, ft),
                "top-down": (fa, fa * share, fa * (1 - share))}, share

    def score(cases):
        """cases: list of (forecasts by method, (actual All, actual Movies, actual TV))."""
        rows = {}
        for method in methods:
            miss = np.array([[np.abs(act[i] - fc[method][i]).mean() for i in range(3)] for fc, act in cases]).mean(axis=0)
            level = np.array([[act[i].mean() for i in range(3)] for fc, act in cases]).mean(axis=0)
            rows[method] = {"All titles MAE": miss[0], "Movies MAE": miss[1], "TV Shows MAE": miss[2],
                            "relative miss": float((miss / level).mean())}
        return pd.DataFrame(rows).T

    pre = [s.iloc[:-holdout] for s in (sa, sm, st_)]
    cases = []
    for end in _t5_origin_ends(len(pre[0]), h, n_origins, step):
        fc, _ = forecasts(pre[0].iloc[:end], pre[1].iloc[:end], pre[2].iloc[:end], h)
        cases.append((fc, tuple(p.iloc[end:end + h].values for p in pre)))
    rolling = score(cases)
    fc, _ = forecasts(pre[0], pre[1], pre[2], holdout)
    final = score([(fc, tuple(s.iloc[-holdout:].values for s in (sa, sm, st_)))])

    bu, td = rolling.loc["bottom-up", "relative miss"], rolling.loc["top-down", "relative miss"]
    method = "top-down" if td <= (1 - T5_MIN_GAIN) * bu * (1 + 1e-9) else "bottom-up"
    fcf, share = forecasts(sa, sm, st_, 12)
    a, m, t = fcf[method]
    future = pd.DataFrame({"All titles": a, "Movies": m, "TV Shows": t}, index=_t5_future_index(sa, 12))
    note = (f"{method} was used. In the rolling tests the average relative miss was {bu:.1%} for bottom-up and "
            f"{td:.1%} for top-down (top-down needs to be at least {T5_MIN_GAIN:.0%} better to replace bottom-up).")
    return {"ok": True, "method": method, "rolling": rolling, "final": final, "future": future,
            "share_movies": share, "note": note}


def t5_start_sensitivity(path, kind=None, shifts=(-12, -6, 0, 6, 12)):
    """Does the choice of the first month matter? Re-run the whole study from other start months.

    A start that leaves too little history (or lies outside the data) is skipped. Early months with only a
    handful of titles are included when the start moves earlier, so you can see what they would change.
    """
    base = t5_series(path, kind)[1]["start"]
    rows, seen = [], set()
    for shift in shifts:
        try:
            series, info = t5_series(path, kind, start=base + pd.DateOffset(months=shift))
            res = t5_study(series)
        except ValueError:
            continue
        if info["start"] in seen:
            continue
        seen.add(info["start"])
        rows.append({"start": f"{info['start']:%Y-%m}", "months": info["n_months"], "model chosen": res["selected"],
                     "12-month average: final-test miss": res["holdout"].loc[T5_DEFAULT_MODEL, "MAE"],
                     "chosen model: final-test miss": res["holdout"].loc[res["selected"], "MAE"],
                     "level next month (chosen)": res["future"]["forecast"].iloc[0],
                     "level next month (12-month average)": res["default_future"].iloc[0]})
    return pd.DataFrame(rows)


def train_t5(path, kind=None, start=None):
    """Everything the dashboard needs for one series: the study results plus the data info."""
    series, info = t5_series(path, kind, start)
    res = t5_study(series)
    res["info"] = info
    return res


# Extras
def paired_bootstrap(y_true, pred_a, pred_b, metric=accuracy_score, n=1000, seed=SEED, level=0.95):
    """Compare two models using paired bootstrap resampling of the same test examples."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n):
        pick = rng.integers(0, len(y_true), len(y_true))
        gaps.append(metric(y_true[pick], pred_a[pick]) - metric(y_true[pick], pred_b[pick]))
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(gaps, [tail, 100 - tail])
    return float(np.mean(gaps)), float(low), float(high)


def macro_f1_codes(y_true, y_pred, n_classes):
    """Macro-F1 for labels already turned into whole numbers 0..n_classes-1.

    Gives the same value as sklearn's f1_score(average='macro', zero_division=0) but is much faster,
    which makes thousands of bootstrap repetitions affordable.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=1) + cm.sum(axis=0)          # Per-class counts
    present = denom > 0                              # Observed classes
    return float(np.mean(np.where(present, 2 * tp / np.maximum(denom, 1), 0.0)[present]))


def eda_tables(path):
    """Small tables for the 'look at the data first' step (used by eda.py and the dashboard)."""
    raw = read_data(path)
    blank = raw.astype(str).apply(lambda c: c.str.strip()).isin(["Not Given", "nan", ""])
    df = prep_t3(path, strict=False)
    movies = df[df["type"] == "Movie"]
    genres = pd.Series([g for s in df["listed_in"] for g in split_items(s)])
    try:                                           # Keep the overview usable if forecasting data is insufficient
        monthly, t5_info = t5_series(path, strict=False)
        t5_note = ""
    except ValueError as err:
        monthly, t5_info, t5_note = None, None, str(err)
    return {
        "missing": blank.sum().sort_values(ascending=False),
        "types": df["type"].value_counts(),
        "release_decade": (df["release_year"] // 10 * 10).astype(int).value_counts().sort_index(),
        "added_by_type": pd.crosstab(df["year_added"], df["type"]),
        "top_genres": genres.value_counts().head(15),
        "ratings": df["rating"].value_counts(),
        "rating_groups": df["rating_group"].value_counts().reindex(GROUP_ORDER).dropna(),
        "movie_minutes": movies.loc[movies["movie_minutes"] > 0, ["movie_minutes", "rating_group"]],
        "n_rows": len(df),
        "monthly_added": monthly, "t5_info": t5_info, "t5_note": t5_note,
    }
