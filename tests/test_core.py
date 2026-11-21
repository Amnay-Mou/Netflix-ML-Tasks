

"""netflix-ml-tasks Test Core"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import netflix_core as nc
from make_demo_data import make_demo


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "demo.csv")
        make_demo().to_csv(cls.path, index=False)
        cls.df1 = nc.load_t1(cls.path)
        cls.rec = nc.Recommender(cls.df1, weights=dict(nc.DEFAULT_WEIGHTS))
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def test_not_given_becomes_missing(self):
        df, _ = nc.prep_t2(self.path)
        self.assertEqual((df["director"] == "Not Given").sum(), 0)
        self.assertGreater(df["director"].isna().sum(), 0)
        self.assertTrue((df["has_director"] == df["director"].notna().astype(int)).all())

    def test_target_matches_type(self):
        df, y = nc.prep_t2(self.path)
        self.assertTrue(((y == 1) == (df["type"] == "TV Show")).all())

    def test_task2_features_do_not_leak(self):
        used = set(nc.T2_CAT + nc.T2_NUM)
        for bad in ["duration", "movie_minutes", "seasons", "listed_in", "type"]:
            self.assertNotIn(bad, used)

    def test_task3_never_uses_the_rating(self):
        self.assertNotIn("rating", nc.T3_FEATURES)
        self.assertNotIn("rating_group", nc.T3_FEATURES)

    def test_every_grouped_rating_has_a_known_group(self):
        self.assertTrue(set(nc.RATING_TO_GROUP.values()) <= set(nc.GROUP_ORDER))
    # Tsk 1
    def test_recommendation_never_returns_the_query(self):
        res = self.rec.recommend(5, n=10)
        self.assertEqual(len(res), 10)
        self.assertNotIn(self.df1.at[5, "title"], res["title"].tolist())

    def test_same_type_filter(self):
        res = self.rec.recommend(3, n=15, same_type=True)
        self.assertEqual(set(res["type"]), {self.df1.at[3, "type"]})

    def test_min_year_filter(self):
        res = self.rec.recommend(3, n=15, min_year=2010)
        self.assertTrue((res["release_year"] >= 2010).all())

    def test_recommendations_are_repeatable(self):
        a = self.rec.recommend(7, n=10)
        b = self.rec.recommend(7, n=10)
        self.assertEqual(a["title"].tolist(), b["title"].tolist())

    def test_several_titles_exclude_all_picks(self):
        picks = [1, 2, 3]
        res = self.rec.recommend_many(picks, n=10)
        self.assertTrue(set(res["title"]).isdisjoint(set(self.df1.loc[picks, "title"])))
        self.assertTrue((res["similarity"].diff().dropna() <= 1e-9).all())

    def test_weights_file_is_ignored_unless_adopted(self):
        path = os.path.join(self.tmp.name, "w.json")
        with open(path, "w") as f:
            f.write('{"adopted": false, "weights": {"title": 9}}')
        self.assertEqual(nc.load_weights(path), nc.DEFAULT_WEIGHTS)
        with open(path, "w") as f:
            f.write('{"adopted": true, "weights": {"title": 9}}')
        self.assertEqual(nc.load_weights(path)["title"], 9)
        self.assertEqual(nc.load_weights(os.path.join(self.tmp.name, "missing.json")), nc.DEFAULT_WEIGHTS)

    # Tsk 2/Tsk 3
    def test_task2_prediction_is_a_probability(self):
        m = nc.train_t2(self.path)
        row = nc.t2_row(m["ratings"][0], "Unknown", 2015, 2018, 6, True, 1, 1, 2)
        p = nc.t2_predict(m["model"], row)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_task3_scores_add_up_to_one(self):
        m = nc.train_t3(self.path, params_file=os.path.join(self.tmp.name, "none.json"))
        self.assertEqual(m["params_source"], "default")
        row = nc.t3_row("Movie", "United States", ["Dramas"], 2015, 2018, 6, True, 1, 1, 100, 1)
        probs = nc.t3_predict(m["model"], row)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertEqual(list(probs), [g for g in nc.GROUP_ORDER if g in probs])

    def test_paired_bootstrap_sees_no_gap_between_identical_models(self):
        y = np.array([0, 1] * 100)
        pred = np.array([0, 1, 1, 0] * 50)
        gap, low, high = nc.paired_bootstrap(y, pred, pred, n=200)
        self.assertEqual((gap, low, high), (0.0, 0.0, 0.0))

    def test_paired_bootstrap_sees_a_big_gap(self):
        y = np.array([0, 1] * 200)
        good, bad = y.copy(), 1 - y
        gap, low, high = nc.paired_bootstrap(y, good, bad, n=200)
        self.assertGreater(low, 0.5)


    # data checks and corner cases (Tsk 1)
    def _write(self, df, name):
        path = os.path.join(self.tmp.name, name)
        df.to_csv(path, index=False)
        return path

    def test_missing_column_gives_a_readable_error(self):
        path = self._write(make_demo(200).drop(columns="rating"), "nocol.csv")
        with self.assertRaises(ValueError) as ctx:
            nc.load_t1(path)
        self.assertIn("rating", str(ctx.exception))

    def test_bad_release_years_are_repaired_and_reported(self):
        df = make_demo(200)
        df["release_year"] = df["release_year"].astype(object)
        df.loc[3, "release_year"] = "unknown"
        df.loc[4, "release_year"] = None
        clean = nc.load_t1(self._write(df, "years.csv"))
        self.assertTrue(pd.api.types.is_integer_dtype(clean["release_year"]))
        self.assertTrue(any("release year" in m for m in clean.attrs["issues"]))
        nc.Recommender(clean, dict(nc.DEFAULT_WEIGHTS)).recommend(0, n=5)   # still wrk

    def test_rows_without_a_title_are_removed(self):
        df = make_demo(200)
        df.loc[5, "title"] = "Not Given"
        clean = nc.load_t1(self._write(df, "titles.csv"))
        self.assertEqual(len(clean), 199)
        self.assertTrue(any("title" in m for m in clean.attrs["issues"]))

    def test_a_completely_empty_column_is_skipped_not_fatal(self):
        df = make_demo(200)
        df["director"] = "Not Given"
        clean = nc.load_t1(self._write(df, "nodir.csv"))
        with self.assertWarns(UserWarning):
            blocks = nc.build_blocks(clean)
        self.assertNotIn("director", blocks)
        self.assertEqual(len(nc.Recommender(clean, dict(nc.DEFAULT_WEIGHTS)).recommend(0, n=5)), 5)

    def test_description_column_is_used_only_when_present(self):
        self.assertNotIn("description", nc.build_blocks(self.df1))
        df = make_demo(200)
        df["description"] = "a story about family and friends " + df["title"]
        clean = nc.load_t1(self._write(df, "desc.csv"))
        self.assertIn("description", nc.build_blocks(clean))

    def test_too_few_titles_is_refused_and_small_is_warned(self):
        with self.assertRaises(ValueError):
            nc.check_size(self.df1.head(10))
        self.assertIsNotNone(nc.check_size(self.df1.head(100)))
        self.assertIsNone(nc.check_size(make_demo(400)))

    def test_title_lookup_exact_several_close_none(self):
        df = self.df1.copy()
        df.loc[1, "title"], df.loc[1, "release_year"] = df.at[0, "title"], 1999
        df.loc[0, "release_year"] = 2015
        name = df.at[0, "title"]
        self.assertEqual(nc.find_titles(df, df.at[7, "title"].upper())[0], "exact")
        kind, rows = nc.find_titles(df, name)
        self.assertEqual((kind, sorted(rows)), ("several", [0, 1]))
        self.assertEqual(nc.find_titles(df, f"{name} (1999)"), ("exact", [1]))
        self.assertEqual(nc.find_titles(df, name[:-1] + "x")[0], "close")
        self.assertEqual(nc.find_titles(df, "qqqqzzzz"), ("none", []))

    def test_asking_for_more_than_exists_returns_fewer_without_error(self):
        res = self.rec.recommend(0, n=5000, same_type=True, min_year=2019)
        self.assertLess(len(res), 5000)
        self.assertGreater(len(res), 0)

    def test_filters_that_remove_everything_return_an_empty_table(self):
        self.assertTrue(self.rec.recommend(0, n=10, min_year=2999).empty)

    def test_title_with_almost_no_details_still_gets_suggestions(self):
        df = make_demo(200)
        df.loc[0, ["director", "country", "rating", "listed_in"]] = "Not Given"
        clean = nc.load_t1(self._write(df, "sparse.csv"))
        self.assertEqual(len(nc.Recommender(clean, dict(nc.DEFAULT_WEIGHTS)).recommend(0, n=10)), 10)

    def test_mixed_types_ignore_the_same_type_filter(self):
        movie = int(self.df1.index[self.df1["type"] == "Movie"][0])
        show = int(self.df1.index[self.df1["type"] == "TV Show"][0])
        with_filter = self.rec.recommend_many([movie, show], n=10, same_type=True)
        without = self.rec.recommend_many([movie, show], n=10, same_type=False)
        self.assertEqual(with_filter["title"].tolist(), without["title"].tolist())

    def test_equal_scores_are_ordered_by_closest_year(self):
        n = len(self.df1)
        sims = np.zeros(n)
        sims[[10, 20, 30]] = 0.5
        res = self.rec._top(sims, [], 3, None, None, ref_year=2010)
        years = self.df1.loc[[10, 20, 30], "release_year"]
        expected = years.sub(2010).abs().sort_values(kind="stable").index.tolist()
        got = [int(self.df1.index[self.df1["title"] == t][0]) for t in res["title"]]
        self.assertEqual(got, expected)

    def test_tasks_2_and_3_also_check_columns(self):
        path = self._write(make_demo(200).drop(columns="duration"), "nodur.csv")
        with self.assertRaises(ValueError):
            nc.prep_t2(path)
    #(Tsks 2 and 3)
    def test_s1_unreadable_year_does_not_crash_task2(self):
        df = make_demo(300)
        df["release_year"] = df["release_year"].astype(object)
        df.loc[3, "release_year"] = "unknown"
        clean, y = nc.prep_t2(self._write(df, "s1.csv"))
        self.assertTrue(pd.api.types.is_integer_dtype(clean["release_year"]))
        self.assertFalse(clean["years_to_netflix"].isna().all())
        self.assertTrue(any("release year" in m for m in clean.attrs["issues"]))

    def test_s2_unexpected_type_values_are_removed_not_called_movies(self):
        df = make_demo(300)
        df.loc[[5, 6, 7], "type"] = "Documentary"
        clean, y = nc.prep_t2(self._write(df, "s2.csv"))
        self.assertEqual(set(clean["type"]), {"Movie", "TV Show"})
        self.assertEqual(len(clean), 297)
        self.assertTrue(any("Documentary" in m for m in clean.attrs["issues"]))
        self.assertTrue(((y == 1) == (clean["type"] == "TV Show")).all())

    def test_s3_a_class_that_is_too_small_is_refused(self):
        df = make_demo(300)
        df.loc[df.index[df["type"] == "TV Show"][5:], "type"] = "Movie"
        path = self._write(df, "s3.csv")
        with self.assertRaises(ValueError) as ctx:
            nc.prep_t2(path)
        self.assertIn("TV Show", str(ctx.exception))
        nc.prep_t2(path, strict=False)

    def test_s3_too_few_titles_for_training_is_refused(self):
        with self.assertRaises(ValueError):
            nc.prep_t2(self._write(make_demo(20), "s3b.csv"))

    def test_data_overview_still_works_on_a_problem_file(self):
        df = make_demo(300)
        df.loc[df.index[df["type"] == "TV Show"][5:], "type"] = "Movie"
        self.assertIn("types", nc.eda_tables(self._write(df, "s3c.csv")))
    # Tsk 2
    def test_t2_selection_file_is_respected_and_unknown_names_ignored(self):
        path = os.path.join(self.tmp.name, "sel.json")
        self.assertEqual(nc.t2_config(path)[0], "Logistic Regression")
        with open(path, "w") as f:
            f.write('{"model": "Random Forest"}')
        self.assertEqual(nc.t2_config(path), ("Random Forest", "selected by cross-validation"))
        with open(path, "w") as f:
            f.write('{"model": "Magic Model"}')
        self.assertEqual(nc.t2_config(path)[1], "default")
    # Tsk 3
    def test_t3_1_rating_exclusions_are_explained(self):
        df = make_demo(400)
        df.loc[[1, 2], "rating"] = "74 min"
        df.loc[3, "rating"] = "NR"
        df.loc[4, "rating"] = None
        out = nc.prep_t3(self._write(df, "t31.csv"))
        rep = out.attrs["rating_report"]
        self.assertEqual((rep["not_rated"], rep["missing"], rep["unexpected"]), (1, 1, {"74 min": 2}))
        self.assertTrue(any("74 min" in m for m in out.attrs["issues"]))
        self.assertEqual(int(out["rating_group"].isna().sum()), 4)
    def test_t3_3_missing_duration_stays_empty_and_is_filled_from_training_rows_of_its_own_type(self):
        df = make_demo(400)
        show = df.index[df["type"] == "TV Show"][0]
        movie = df.index[df["type"] == "Movie"][0]
        df.loc[show, "duration"] = "Not Given"
        df.loc[movie, "duration"] = "soon"
        out = nc.prep_t3(self._write(df, "t33.csv"))
        s, m = out.loc[out["title"] == df.at[show, "title"]].iloc[0], out.loc[out["title"] == df.at[movie, "title"]].iloc[0]
        self.assertEqual(s["movie_minutes"], 0.0)
        self.assertTrue(np.isnan(s["seasons"]))
        self.assertEqual(m["seasons"], 0.0)
        self.assertTrue(np.isnan(m["movie_minutes"]))
        self.assertTrue(any("duration" in n and "training rows" in n for n in out.attrs["issues"]))
        X = out[nc.T3_FEATURES]
        filled = nc.T3Imputer().fit(X).transform(X)
        self.assertGreaterEqual(filled.loc[s.name, "seasons"], 1)
        self.assertEqual(filled.loc[s.name, "movie_minutes"], 0.0)
        self.assertGreaterEqual(filled.loc[m.name, "movie_minutes"], 60)
        self.assertEqual(filled.loc[m.name, "seasons"], 0.0)
        self.assertFalse(filled[["movie_minutes", "seasons"]].isna().any().any())

    def test_t3_3_valid_durations_are_read_exactly(self):
        out = nc.prep_t3(self.path)
        raw = pd.read_csv(self.path).drop_duplicates(subset="show_id").reset_index(drop=True)
        movies = out["type"] == "Movie"
        self.assertEqual(out.loc[movies, "movie_minutes"].tolist(),
                         raw.loc[movies, "duration"].str.extract(r"(\d+)")[0].astype(float).tolist())

    def test_t3_a_group_that_is_too_small_is_refused(self):
        df = make_demo(400)
        df["rating"] = "TV-MA"
        df.loc[[0, 1, 2], "rating"] = "G"
        with self.assertRaises(ValueError) as ctx:
            nc.prep_t3(self._write(df, "t3small.csv"))
        self.assertIn("Kids", str(ctx.exception))

    def test_t3_4_a_stricter_level_gives_a_wider_range(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        a = np.where(rng.random(300) < 0.85, y, 1 - y)
        b = np.where(rng.random(300) < 0.80, y, 1 - y)
        _, lo95, hi95 = nc.paired_bootstrap(y, a, b, n=500, level=0.95)
        _, lo99, hi99 = nc.paired_bootstrap(y, a, b, n=500, level=0.99)
        self.assertLessEqual(lo99, lo95)
        self.assertGreaterEqual(hi99, hi95)
    # Tsk 3 remaining points
    def test_empty_label_list_gives_a_clear_error_not_an_indexerror(self):
        with self.assertRaises(ValueError) as ctx:
            nc.check_classes([], what="audience group")
        self.assertIn("no labelled titles", str(ctx.exception))

    def test_all_titles_with_unexpected_ratings_gives_a_clear_error(self):
        df = make_demo(300)
        df["rating"] = "74 min"
        with self.assertRaises(ValueError):
            nc.prep_t3(self._write(df, "allweird.csv"))
    def test_task3_refuses_data_without_usable_genres(self):
        empty = make_demo(300)
        empty["listed_in"] = "Not Given"
        with self.assertRaises(ValueError) as ctx:
            nc.train_t3(self._write(empty, "nogenre.csv"), params_file=os.path.join(self.tmp.name, "n.json"))
        self.assertIn("genre", str(ctx.exception).lower())
        unique = make_demo(300)
        unique["listed_in"] = [f"Genre {i}" for i in range(300)]
        with self.assertRaises(ValueError) as ctx:
            nc.prep_t3(self._write(unique, "uniquegenre.csv"))
        self.assertIn("repeat", str(ctx.exception))
        nc.prep_t3(self._write(unique, "uniquegenre2.csv"), strict=False)

    def test_borderline_genre_file_survives_every_cross_validation_fold(self):
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        d = make_demo(400)
        genres = [f"Genre {i}" for i in range(400)]
        for i in range(10):
            genres[i * 40] = "Common"
        d["listed_in"] = genres
        df = nc.prep_t3(self._write(d, "borderline.csv"))
        rated = df.dropna(subset=["rating_group"]).reset_index(drop=True)
        for seed in range(8):
            cv = StratifiedKFold(4, shuffle=True, random_state=seed)
            cross_val_score(nc.t3_pipeline(nc.t3_model("Logistic Regression")),
                            rated[nc.T3_FEATURES], rated["rating_group"], cv=cv,
                            scoring="f1_macro", error_score="raise")
    def test_genre_features_keep_every_genre(self):
        genre_step = dict((name, step) for name, step, _ in nc.t3_pipeline(nc.t3_model("Logistic Regression"))
                          .named_steps["prep"].transformers)["genres"]
        self.assertEqual(genre_step.min_df, 1)
    def test_fast_macro_f1_matches_sklearn(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(3)
        for n_classes, size in [(4, 500), (4, 12), (3, 40)]:
            t = rng.integers(0, n_classes, size)
            p = np.where(rng.random(size) < 0.6, t, rng.integers(0, n_classes, size))
            self.assertAlmostEqual(nc.macro_f1_codes(t, p, n_classes),
                                   f1_score(t, p, average="macro", zero_division=0), places=10)
        t = np.array([0, 0, 1, 1]); p = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(nc.macro_f1_codes(t, p, 4), f1_score(t, p, average="macro", zero_division=0), places=10)

    def test_raw_vs_grouped_stops_when_only_one_audience_group_is_left(self):
        import subprocess
        df = make_demo(400)
        df["rating"] = "TV-Y"
        df.loc[df.index[:150], "rating"] = "TV-G"
        work = tempfile.mkdtemp(dir=self.tmp.name)
        csv = self._write(df, "onegroup.csv")
        project = os.path.join(os.path.dirname(__file__), "..")
        out = subprocess.run([sys.executable, "-W", "ignore", os.path.join(project, "task3_raw_vs_grouped.py"), csv],
                             cwd=work, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": project})
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("audience group", out.stderr + out.stdout)
        self.assertNotIn("Traceback", out.stderr)

    # Eda
    def test_eda_tables_have_what_the_dashboard_needs(self):
        t = nc.eda_tables(self.path)
        for key in ["missing", "types", "release_decade", "added_by_type", "top_genres",
                    "ratings", "rating_groups", "movie_minutes", "n_rows"]:
            self.assertIn(key, t)
        self.assertGreater(t["missing"]["director"], 0)

# Tsk 5
def t5_frame(counts, start="2016-01", last_month_days=None):
    """Create a synthetic monthly dataset from the supplied counts. The last month may end early."""
    rows, first = [], pd.Period(start, "M")
    for k, n in enumerate(counts):
        p = first + k
        days = last_month_days if (last_month_days and k == len(counts) - 1) else p.days_in_month
        for j in range(n):
            day = days if j == n - 1 else 1 + j % days
            kind = "TV Show" if j % 3 == 0 else "Movie"
            rows.append({"show_id": f"s{len(rows)}", "type": kind, "title": f"Title {len(rows)}",
                         "director": "", "country": "United States",
                         "date_added": f"{p.month}/{day}/{p.year}", "release_year": 2015,
                         "rating": "TV-MA", "duration": "90 min" if kind == "Movie" else "1 Season",
                         "listed_in": "Dramas"})
    return pd.DataFrame(rows)
def t5_counts(months=60):
    """A steady upward trend with a yearly wave (about 50 titles a month)."""
    return [int(20 + k + 6 * np.sin(2 * np.pi * k / 12)) for k in range(months)]

def t5_series_from(values, start="2012-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)

class Task5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.counts = t5_counts(60)
        cls.df = t5_frame(cls.counts)
        cls.path = cls._write_to(cls.tmp.name, cls.df, "t5.csv")
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    @staticmethod
    def _write_to(folder, df, name):
        path = os.path.join(folder, name)
        df.to_csv(path, index=False)
        return path
    def _write(self, df, name):
        return self._write_to(self.tmp.name, df, name)
    def test_only_type_and_date_added_are_needed(self):
        path = self._write(self.df[["type", "date_added"]], "t5_two_columns.csv")
        series, info = nc.t5_series(path)
        self.assertEqual(info["n_months"], 60)
        self.assertTrue(any("show_id" in m for m in info["issues"]))
    def test_a_missing_date_column_gives_a_readable_error(self):
        path = self._write(self.df.drop(columns="date_added"), "t5_nodate.csv")
        with self.assertRaises(ValueError) as ctx:
            nc.t5_series(path)
        self.assertIn("date_added", str(ctx.exception))
    def test_no_usable_dates_is_a_clear_error_not_a_crash(self):
        for bad in ["Not Given", "not a date"]:
            df = self.df.copy()
            df["date_added"] = bad
            with self.assertRaises(ValueError) as ctx:
                nc.t5_series(self._write(df, "t5_nodates.csv"))
            self.assertIn("usable dates", str(ctx.exception))
    def test_no_complete_month_is_a_clear_error_not_an_indexerror(self):
        path = self._write(t5_frame([40], last_month_days=10), "t5_onemonth.csv")
        with self.assertRaises(ValueError) as ctx:
            nc.t5_series(path)
        self.assertIn("complete month", str(ctx.exception))
    def test_both_date_styles_are_read(self):
        got = nc._t5_parse_dates(pd.Series(["9/25/2021", "September 25, 2021", "not a date", np.nan]))
        self.assertEqual(got.iloc[0], pd.Timestamp("2021-09-25"))
        self.assertEqual(got.iloc[1], pd.Timestamp("2021-09-25"))
        self.assertTrue(got.iloc[2:].isna().all())
    def test_a_repeated_show_id_is_counted_once(self):
        df = pd.concat([self.df, self.df.head(100)], ignore_index=True)
        series, info = nc.t5_series(self._write(df, "t5_dups.csv"))
        self.assertEqual(info["quality"]["repeated_removed"], 100)
        self.assertEqual(info["quality"]["in_series"], len(self.df))
    def test_an_unexpected_type_counts_in_all_titles_but_not_in_the_split(self):
        df = self.df.copy()
        df.loc[df.index[:20], "type"] = "Documentary"
        path = self._write(df, "t5_type.csv")
        all_s, info = nc.t5_series(path)
        movie, _ = nc.t5_series(path, "Movie")
        show, _ = nc.t5_series(path, "TV Show")
        self.assertEqual(info["quality"]["unexpected_type"], 20)
        self.assertEqual(all_s.sum() - movie.sum() - show.sum(), 20)
        self.assertTrue(any("Documentary" in m for m in info["issues"]))
    def test_every_left_out_record_is_accounted_for(self):
        df = pd.concat([t5_frame([2, 1, 3, 0, 2, 1, 4, 2], start="2015-04"), t5_frame(t5_counts(60), start="2016-01")],
                       ignore_index=True)
        df["show_id"] = [f"s{i}" for i in range(len(df))]
        df.loc[df.index[200:205], "date_added"] = "Not Given"
        df.loc[df.index[210:213], "date_added"] = "31/31/2020"
        df = pd.concat([df, df.head(4)], ignore_index=True)
        df = pd.concat([df, t5_frame([30], start="2021-01", last_month_days=9).assign(
            show_id=lambda d: [f"x{i}" for i in range(len(d))])], ignore_index=True)
        series, info = nc.t5_series(self._write(df, "t5_audit.csv"))
        q = info["quality"]
        self.assertEqual(q["missing_date"], 5)
        self.assertEqual(q["unreadable_date"], 3)
        self.assertEqual(q["repeated_removed"], 4)
        self.assertEqual(q["rows_read"] - q["repeated_removed"] - q["missing_date"] - q["unreadable_date"], q["dated"])
        self.assertEqual(q["dated"] - q["before_start"] - q["partial_month"], q["in_series"])
        self.assertEqual(q["excluded_total"], q["rows_read"] - q["in_series"])
        table = nc.t5_accounting(info)
        self.assertEqual(int(table.loc[table["Step"].str.startswith("TOTAL"), "Titles"].iloc[0]), q["excluded_total"])
        self.assertGreater(q["before_start"], 0)
        self.assertGreater(q["partial_month"], 0)
    def test_an_incomplete_last_month_is_dropped_and_counted(self):
        path = self._write(t5_frame(self.counts, last_month_days=10), "t5_partial.csv")
        series, info = nc.t5_series(path)
        self.assertEqual(len(series), 59)
        self.assertTrue(info["dropped_partial_month"])
        self.assertEqual(info["quality"]["partial_month"], self.counts[-1])
    def test_early_sparse_months_are_skipped(self):
        path = self._write(t5_frame([2, 1, 3, 0, 2, 1, 4, 2] + self.counts, start="2015-05"), "t5_sparse.csv")
        series, info = nc.t5_series(path)
        self.assertEqual(info["start"], pd.Timestamp("2016-01-01"))
        self.assertEqual(info["quality"]["before_start"], 15)
    def test_the_start_month_ignores_the_final_test_months(self):
        early = [2, 1, 3, 1, 2, 2, 1, 3, 2, 1, 2, 3]
        normal = [40 + (k % 5) for k in range(48)]
        quiet_end = t5_series_from(early + normal + [40] * 12)
        huge_end = t5_series_from(early + normal + [4000] * 12)
        self.assertEqual(nc._t5_auto_start(quiet_end), nc._t5_auto_start(huge_end))
        self.assertEqual(nc._t5_auto_start(quiet_end), quiet_end.index[len(early)])

    def test_the_model_choice_ignores_the_final_test_months(self):
        rng = np.random.default_rng(3)
        calm = t5_series_from(np.array(t5_counts(60)) + rng.normal(0, 2, 60), "2016-01")
        wild = calm.copy()
        wild.iloc[-12:] = wild.iloc[-12:] * 30
        a, b = nc.t5_study(calm), nc.t5_study(wild)
        self.assertEqual(a["selected"], b["selected"])
        pd.testing.assert_frame_equal(a["cv"], b["cv"])
    def test_too_little_history_is_refused_but_the_overview_still_opens(self):
        path = self._write(t5_frame(t5_counts(30)), "t5_short.csv")
        with self.assertRaises(ValueError) as ctx:
            nc.t5_series(path)
        self.assertIn(str(nc.T5_MIN_MONTHS), str(ctx.exception))
        series, info = nc.t5_series(path, strict=False)
        self.assertEqual(len(series), 30)
    def test_every_model_returns_the_right_length_and_never_goes_negative(self):
        series = t5_series_from(t5_counts(60), "2016-01")
        for name, fn in nc.T5_MODELS.items():
            got = fn(series, 12)
            self.assertEqual(len(got), 12, name)
            self.assertTrue((got >= 0).all(), name)
    def test_seasonal_naive_repeats_last_year(self):
        series = t5_series_from(range(36), "2018-01")
        np.testing.assert_array_equal(nc.t5_seasonal_naive(series, 12), series.iloc[-12:].values)

    def test_the_error_range_is_plain_and_not_called_a_prediction_interval(self):
        series = t5_series_from(t5_counts(60), "2016-01")
        fut = nc.t5_future(series, "Naive (last month)", 6, rmse=10.0)
        self.assertEqual(list(fut.columns), ["forecast", "range_low", "range_high"])
        np.testing.assert_allclose(fut["range_high"] - fut["forecast"], 10.0)
        self.assertNotIn("80%", nc.T5_RANGE_NOTE)
        self.assertIn("not a prediction interval", nc.T5_RANGE_NOTE)
    def test_the_study_returns_what_the_dashboard_needs(self):
        res = nc.train_t5(self.path)
        for key in ["series", "cv", "holdout", "selected", "future", "info", "best_on_final_test", "skill_vs_baseline",
                    "best_simple_rule", "selection_note", "best_in_rolling_tests", "default_future"]:
            self.assertIn(key, res)
        self.assertEqual(len(res["future"]), 12)
        self.assertEqual(len(res["default_future"]), 12)
        self.assertEqual(list(res["default_future"].index), list(res["future"].index))
        self.assertIn(res["selected"], nc.T5_MODELS)
        self.assertIsInstance(nc.t5_verdict(res), str)
    @staticmethod
    def _verdict_input(selected, mae):
        """A minimal study result: `mae` maps model -> average miss on the final test."""
        hold = pd.DataFrame({"MAE": mae})
        rules = hold.loc[nc.T5_SIMPLE_RULES, "MAE"]
        return {"selected": selected, "holdout_months": 12, "holdout": hold,
                "best_simple_rule": rules.idxmin(),
                "skill_vs_baseline": float(1 - hold.loc[selected, "MAE"] / rules.min())}

    def test_a_simple_rule_is_never_called_a_real_improvement(self):
        mae = {"Naive (last month)": 41.5, nc.T5_BASELINE: 47.3, "12-month average": 30.0,
               "Holt-Winters": 39.8, "Random Forest (lags)": 34.3, "Trend + season regression": 53.3}
        text = nc.t5_verdict(self._verdict_input("Naive (last month)", mae))
        self.assertIn("itself a simple rule", text)
        self.assertIn("12-month average did better", text)
        self.assertNotIn("real improvement", text)

    def test_a_fancy_model_is_judged_against_the_best_simple_rule(self):
        base = {"Naive (last month)": 40.0, nc.T5_BASELINE: 50.0, "12-month average": 30.0, "Holt-Winters": 30.0}
        better = nc.t5_verdict(self._verdict_input("Holt-Winters", {**base, "Holt-Winters": 21.0}))
        self.assertIn("real improvement", better)
        self.assertIn("12-month average", better)
        same = nc.t5_verdict(self._verdict_input("Holt-Winters", {**base, "Holt-Winters": 31.0}))
        self.assertIn("no clear improvement", same)
        worse = nc.t5_verdict(self._verdict_input("Holt-Winters", {**base, "Holt-Winters": 45.0}))
        self.assertIn("worse than the best simple rule", worse)
    def test_the_study_measures_skill_against_the_best_simple_rule(self):
        res = nc.t5_study(t5_series_from(np.array(t5_counts(60)) + np.random.default_rng(5).normal(0, 3, 60), "2016-01"))
        rules = res["holdout"].loc[nc.T5_SIMPLE_RULES, "MAE"]
        self.assertEqual(res["best_simple_rule"], rules.idxmin())
        self.assertAlmostEqual(res["skill_vs_baseline"], 1 - res["holdout"].loc[res["selected"], "MAE"] / rules.min())

    @staticmethod
    def _cv(changes=None):
        """A rolling-test table with the 12-month average at 30; `changes` overrides other models' misses."""
        base = {"Naive (last month)": 40.0, nc.T5_BASELINE: 45.0, "12-month average": 30.0,
                "Holt-Winters": 35.0, "Random Forest (lags)": 36.0, "Trend + season regression": 50.0}
        base.update(changes or {})
        return pd.DataFrame({"MAE": base})

    def test_a_small_win_does_not_replace_the_12_month_average(self):
        cv = self._cv({"Holt-Winters": 28.5})
        chosen, note, gain = nc.t5_choose(cv)
        self.assertEqual(chosen, nc.T5_DEFAULT_MODEL)
        self.assertAlmostEqual(gain, 0.05)
        self.assertIn("was kept", note)

    def test_a_clear_win_replaces_the_12_month_average(self):
        chosen, note, gain = nc.t5_choose(self._cv({"Holt-Winters": 24.0}))
        self.assertEqual(chosen, "Holt-Winters")
        self.assertIn("so it was chosen", note)

    def test_exactly_ten_percent_is_enough_and_the_default_can_simply_win(self):
        chosen, _, _ = nc.t5_choose(self._cv({"Holt-Winters": 27.0}))
        self.assertEqual(chosen, "Holt-Winters")
        chosen, note, _ = nc.t5_choose(self._cv())
        self.assertEqual(chosen, nc.T5_DEFAULT_MODEL)
        self.assertIn("lowest error", note)
    def test_the_study_follows_the_rule(self):
        res = nc.t5_study(t5_series_from(np.array(t5_counts(60)) + np.random.default_rng(8).normal(0, 4, 60), "2016-01"))
        chosen, note, gain = nc.t5_choose(res["cv"])
        self.assertEqual(res["selected"], chosen)
        self.assertTrue(res["selection_note"].startswith(note))
        self.assertTrue(res["selected"] == nc.T5_DEFAULT_MODEL or gain >= nc.T5_MIN_GAIN)

    def test_the_verdict_explains_a_kept_default(self):
        mae = {"Naive (last month)": 41.5, nc.T5_BASELINE: 47.3, "12-month average": 30.0,
               "Holt-Winters": 39.8, "Random Forest (lags)": 34.3, "Trend + season regression": 53.3}
        text = nc.t5_verdict(self._verdict_input("12-month average", mae))
        self.assertIn("No fancier model beat the 12-month average by at least 10%", text)
        self.assertNotIn("real improvement", text)
    def test_the_rolling_detail_shows_each_start_date_and_horizon(self):
        series = t5_series_from(np.array(t5_counts(54)) + np.random.default_rng(1).normal(0, 3, 54), "2016-01")
        detail = nc.t5_rolling_detail(series)
        pd.testing.assert_frame_equal(detail["summary"], nc.t5_rolling_origin(series))
        self.assertEqual(len(detail["by_origin"]), detail["n_origins"])
        self.assertEqual(list(detail["by_horizon"].index),
                         ["1 month ahead", "3 months ahead", "6 months ahead", "12 months ahead"])
        for model, errs in detail["errors"].items():
            self.assertEqual(errs.shape, (detail["n_origins"], nc.T5_CV_H), model)

    def test_the_number_of_start_dates_follows_the_data(self):
        long_res = nc.t5_study(t5_series_from(t5_counts(66), "2016-01"))
        short_res = nc.t5_study(t5_series_from(t5_counts(nc.T5_MIN_MONTHS), "2016-01"))
        self.assertEqual(long_res["cv_origins"], nc.T5_CV_ORIGINS)
        self.assertEqual(short_res["cv_origins"], 1)
        self.assertEqual(len(short_res["cv_by_origin"]), 1)
    def test_the_overlapping_windows_cover_fewer_months_than_the_forecast_count_suggests(self):
        n_pre = 66 - nc.T5_HOLDOUT
        ends = nc._t5_origin_ends(n_pre, nc.T5_CV_H, nc.T5_CV_ORIGINS, nc.T5_CV_STEP)
        months = {m for e in ends for m in range(e, e + nc.T5_CV_H)}
        self.assertEqual(len(ends) * nc.T5_CV_H, 60)
        self.assertEqual(len(months), 24)
    def test_the_range_check_and_the_error_by_months_ahead_are_reported(self):
        res = nc.t5_study(t5_series_from(np.array(t5_counts(60)) + np.random.default_rng(4).normal(0, 4, 60), "2016-01"))
        self.assertTrue(0.0 <= res["range_coverage"] <= 1.0)
        self.assertEqual(list(res["error_by_horizon"].index), ["months 1-3", "months 4-6", "months 7-9", "months 10-12"])
        self.assertTrue((res["error_by_horizon"] >= 0).all())

    def test_the_snapshot_note_says_when_the_forecast_months_have_passed(self):
        info = {"last_date": pd.Timestamp("2021-09-25"), "end": pd.Timestamp("2021-08-01")}
        months = pd.date_range("2021-09-01", periods=12, freq="MS")
        past = nc.t5_snapshot_note(info, months, today="2026-11-04")
        self.assertIn("25 Sep 2021", past)
        self.assertIn("Sep 2021 to Aug 2022", past)
        self.assertIn("already passed", past)
        self.assertIn("not a current forecast", past)
        self.assertNotIn("already passed", nc.t5_snapshot_note(info, months, today="2021-10-01"))

    def test_the_target_is_defined_as_catalogue_additions_not_releases(self):
        self.assertIn("ADDED", nc.T5_TARGET_NOTE)
        self.assertIn("not the number of films or series first released", nc.T5_TARGET_NOTE)

    def test_incomplete_years_are_labelled_on_the_yearly_chart(self):
        series = t5_series_from(np.ones(62), "2016-07")
        labels = nc.t5_year_labels(series)
        self.assertEqual(labels[0], "2016\n(6 mo)")
        self.assertEqual(labels[-1], "2021\n(8 mo)")
        self.assertEqual(labels[1:-1], ["2017", "2018", "2019", "2020"])

    @staticmethod
    def _three_series(models):
        """Movies + TV Shows = All titles, with the model each series 'chose' set by hand (as on the real data)."""
        rng = np.random.default_rng(9)
        movies = t5_series_from(np.clip(100 + rng.normal(0, 12, 60), 20, None), "2016-01")
        shows = t5_series_from(np.clip(50 + rng.normal(0, 8, 60), 10, None), "2016-01")
        total = movies + shows
        return [{"series": s, "selected": m} for s, m in zip((total, movies, shows), models)]
    def test_forecasts_made_separately_do_not_add_up_but_reconciled_ones_do(self):
        a, m, t = self._three_series(["Naive (last month)", "12-month average", "Naive (last month)"])
        separate = [nc.T5_MODELS[r["selected"]](r["series"], 12) for r in (a, m, t)]
        self.assertGreater(np.abs(separate[0] - (separate[1] + separate[2])).max(), 1.0)
        rec = nc.t5_reconcile(a, m, t)
        self.assertTrue(rec["ok"])
        self.assertIn(rec["method"], ("bottom-up", "top-down"))
        fut = rec["future"]
        np.testing.assert_allclose(fut["Movies"] + fut["TV Shows"], fut["All titles"])
        self.assertEqual(list(rec["rolling"].index), ["independent", "bottom-up", "top-down"])
        self.assertIn(rec["method"], rec["note"])
    def test_bottom_up_uses_the_separate_movie_and_tv_forecasts(self):
        a, m, t = self._three_series(["Naive (last month)", "12-month average", "Naive (last month)"])
        rec = nc.t5_reconcile(a, m, t)
        if rec["method"] == "bottom-up":
            np.testing.assert_allclose(rec["future"]["Movies"], nc.t5_mean12(m["series"], 12))
            np.testing.assert_allclose(rec["future"]["TV Shows"], nc.t5_naive(t["series"], 12))

    def test_reconciliation_is_refused_when_the_types_do_not_add_up(self):
        a, m, t = self._three_series(["Naive (last month)", "12-month average", "Naive (last month)"])
        a["series"] = a["series"] + 5
        rec = nc.t5_reconcile(a, m, t)
        self.assertFalse(rec["ok"])
        self.assertIn("another type", rec["note"])
    def test_the_start_month_check_skips_starts_with_too_little_history(self):
        table = nc.t5_start_sensitivity(self.path, shifts=(-12, 0, 6, 12, 24))
        starts = list(table["start"])
        self.assertEqual(starts, ["2016-01", "2016-07", "2017-01"])
        self.assertTrue((table["months"] >= nc.T5_MIN_MONTHS).all())
        self.assertIn("12-month average: final-test miss", table.columns)

    def test_eda_tables_carry_the_monthly_series_and_survive_a_file_without_dates(self):
        t = nc.eda_tables(self.path)
        self.assertIsNotNone(t["monthly_added"])
        bad = self.df.copy()
        bad["date_added"] = "Not Given"
        t2 = nc.eda_tables(self._write(bad, "t5_eda_nodates.csv"))
        self.assertIsNone(t2["monthly_added"])
        self.assertIn("date_added", t2["t5_note"])

def t3_frame(n=600, seed=0):
    """A fake Netflix-style file whose ratings depend on the genre, so the models have something to learn."""
    rng = np.random.default_rng(seed)
    kids_g = ["Children & Family Movies", "Kids' TV"]
    other = ["Dramas", "Comedies", "Documentaries", "Horror Movies", "Thrillers", "Stand-Up Comedy",
             "International Movies", "Crime TV Shows"]
    rows = []
    for i in range(n):
        kind = "Movie" if rng.random() < .7 else "TV Show"
        g = list(rng.choice(kids_g + other, size=int(rng.integers(1, 4)), replace=False))
        if g[0] in kids_g:
            rating = rng.choice(["TV-Y", "TV-Y7", "TV-G", "G", "TV-PG", "PG", "TV-Y7-FV"], p=[.2, .2, .1, .1, .2, .15, .05])
        elif g[0] in ("Horror Movies", "Crime TV Shows"):
            rating = rng.choice(["TV-MA", "R", "TV-14"], p=[.6, .25, .15])
        else:
            rating = rng.choice(["TV-MA", "TV-14", "PG-13", "TV-PG", "R", "PG"])
        rows.append({"show_id": f"s{i}", "type": kind, "title": f"Title {i} {rng.choice(['Love', 'War', 'Night'])}",
                     "director": "Not Given" if rng.random() < .3 else f"D{rng.integers(0, 30)}",
                     "country": rng.choice(["United States", "India", "France", "Not Given"]),
                     "date_added": f"{rng.integers(1, 13)}/{rng.integers(1, 28)}/{rng.integers(2016, 2022)}",
                     "release_year": int(rng.integers(1990, 2022)), "rating": rating,
                     "duration": f"{rng.integers(70, 140)} min" if kind == "Movie" else f"{rng.integers(1, 6)} Seasons",
                     "listed_in": ", ".join(g)})
    return pd.DataFrame(rows)


GOOD_ORDER_SIG = pd.DataFrame([
    ("Logistic Regression", "Decision Tree", "accuracy", 0.0301, -0.0011, 0.0609, False),
    ("Logistic Regression", "Decision Tree", "macro_f1", 0.0216, -0.0094, 0.0542, False),
    ("Logistic Regression", "Random Forest", "accuracy", -0.02, -0.0437, 0.0041, False),
    ("Logistic Regression", "Random Forest", "macro_f1", -0.0147, -0.0399, 0.011, False),
    ("Decision Tree", "Random Forest", "accuracy", -0.0501, -0.0775, -0.023, True),
    ("Decision Tree", "Random Forest", "macro_f1", -0.0363, -0.063, -0.0091, True)],
    columns=["A", "B", "metric", "gap_A_minus_B", "low", "high", "clearly_different"])

REAL_COMPARISON = pd.DataFrame([
    ("Baseline (majority)", "default", 0.4601, 0.1576, np.nan),
    ("Logistic Regression", "default", 0.6025, 0.5927, np.nan), ("Decision Tree", "default", 0.5652, 0.5393, np.nan),
    ("Random Forest", "default", 0.6284, 0.5873, np.nan),
    ("Logistic Regression", "tuned", 0.5962, 0.5859, 0.6099), ("Decision Tree", "tuned", 0.5663, 0.5646, 0.5778),
    ("Random Forest", "tuned", 0.6163, 0.6007, 0.6348)],
    columns=["model", "version", "accuracy", "macro_f1", "cv_macro_f1"])
HAND_TRUE = ["Adults"] * 4 + ["Teens"] * 2 + ["Kids"] * 2 + ["Family"] * 2
HAND_PRED = ["Kids", "Family", "Adults", "Teens"] + ["Family", "Teens"] + ["Adults", "Kids"] + ["Kids", "Family"]

class Task3AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df = t3_frame()
        cls.path = os.path.join(cls.tmp.name, "t3.csv")
        cls.df.to_csv(cls.path, index=False)
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def _xy(self):
        d = nc.prep_t3(self.path)
        d = d.dropna(subset=["rating_group"]).reset_index(drop=True)
        return d[nc.T3_FEATURES + ["title"]], d["rating_group"]
    def test_the_displayed_mapping_covers_every_rating_in_the_code(self):
        shown = {r.strip() for text in nc.GROUP_INFO.values() for r in text.replace(" (parental guidance)", "").split(",")}
        self.assertEqual(shown, set(nc.RATING_TO_GROUP))
        self.assertIn("TV-Y7-FV", nc.GROUP_INFO["Kids"])
    def test_the_target_note_says_these_are_our_own_groups(self):
        for phrase in ["our own age-suitability groups", "not official Netflix labels", "G and TV-G",
                       "general audiences", "NR and UR"]:
            self.assertIn(phrase, nc.T3_TARGET_NOTE)
        self.assertIn("must not decide on its own", nc.T3_DISCLAIMER)
    # Check training-only preprocessing
    def test_prep_t3_leaves_gaps_empty_while_task_2_still_fills_them(self):
        df = self.df.copy()
        df.loc[df.index[:3], "release_year"] = np.nan
        path = os.path.join(self.tmp.name, "t3_gaps.csv")
        df.to_csv(path, index=False)
        out = nc.prep_t3(path)
        self.assertEqual(int(out["release_year"].isna().sum()), 3)
        self.assertTrue(any("release year" in n and "training rows" in n for n in out.attrs["issues"]))
        t2, _ = nc.prep_t2(path)
        self.assertFalse(t2["release_year"].isna().any())
    def test_the_imputer_uses_the_training_rows_only(self):
        X = pd.DataFrame({"type": ["Movie"] * 40 + ["TV Show"] * 10,
                          "release_year": [2000.0] * 20 + [2020.0] * 20 + [2010.0] * 10,
                          "year_added": 2021.0, "years_to_netflix": 21.0,
                          "movie_minutes": [100.0] * 20 + [200.0] * 20 + [0.0] * 10,
                          "seasons": [0.0] * 40 + [2.0] * 10})
        train = X.iloc[:20]
        imp = nc.T3Imputer().fit(pd.concat([train, X.iloc[40:42]]))
        self.assertEqual(imp.minutes_, 100.0)
        self.assertEqual(imp.year_, nc.T3Imputer().fit(pd.concat([train, X.iloc[40:42]])).year_)
        test = pd.DataFrame({"type": ["Movie", "TV Show"], "release_year": [np.nan, 2015.0], "year_added": [2021.0, 2021.0],
                             "years_to_netflix": [np.nan, 6.0], "movie_minutes": [np.nan, 0.0], "seasons": [0.0, np.nan]})
        out = imp.transform(test)
        self.assertEqual(out.at[0, "movie_minutes"], 100.0)
        self.assertEqual(out.at[1, "seasons"], 2.0)
        self.assertEqual(out.at[0, "release_year"], imp.year_)
        self.assertEqual(out.at[0, "years_to_netflix"], 2021.0 - imp.year_)
        self.assertEqual(nc.T3Imputer().fit(X).minutes_, 150.0)

    def test_the_pipeline_runs_on_rows_with_gaps_in_cross_validation(self):
        from sklearn.model_selection import cross_val_score
        X, y = self._xy()
        X = X.copy()
        X.loc[X.index[:15], "movie_minutes"] = np.nan
        X.loc[X.index[15:30], "release_year"] = np.nan
        scores = cross_val_score(nc.t3_pipeline(nc.t3_model("Decision Tree")), X, y, cv=3)
        self.assertEqual(len(scores), 3)
    # Age-grp
    def test_the_safety_numbers_match_a_hand_count(self):
        s = nc.t3_safety(HAND_TRUE, HAND_PRED)
        self.assertEqual(s["too_lenient"], 5)
        self.assertAlmostEqual(s["too_lenient_rate"], 0.5)
        self.assertEqual((s["severe_lenient"], s["older_titles"]), (3, 6))
        self.assertAlmostEqual(s["severe_lenient_rate"], 0.5)
        self.assertEqual(s["adults_as_kids"], 1)
        self.assertAlmostEqual(s["mean_cost"], 1.9)
    def test_the_real_confusion_matrix_gives_the_numbers_in_the_audit(self):
        cm = np.array([[133, 30, 11, 7], [31, 101, 56, 42], [4, 55, 264, 206], [3, 56, 167, 575]])
        yt = [g for g, row in zip(nc.GROUP_ORDER, cm) for _ in range(row.sum())]
        yp = [p for row in cm for p, k in zip(nc.GROUP_ORDER, row) for _ in range(k)]
        s = nc.t3_safety(yt, yp)
        self.assertEqual((s["severe_lenient"], s["older_titles"]), (118, 1330))
        self.assertEqual(s["adults_as_kids"], 3)
        self.assertAlmostEqual(s["severe_lenient_rate"], 118 / 1330)

    def test_the_cost_matrix_punishes_lenient_errors_double(self):
        c = nc.t3_cost_matrix()
        self.assertEqual(c.loc["Adults", "Kids"], 6)
        self.assertEqual(c.loc["Kids", "Adults"], 3)
        self.assertTrue((np.diag(c.values) == 0).all())
    def test_the_error_table_shows_where_each_group_goes(self):
        t = nc.t3_error_table(HAND_TRUE, HAND_PRED)
        pred_cols = [f"predicted {g}" for g in nc.GROUP_ORDER]
        np.testing.assert_allclose(t[pred_cols].sum(axis=1), 1.0)
        self.assertAlmostEqual(t.loc["Adults", "correct"], 0.25)
        self.assertAlmostEqual(t.loc["Adults", "errors one step away"], 1 / 3)
        self.assertAlmostEqual(t.loc["Adults", "too lenient"], 0.75)
        self.assertAlmostEqual(t.loc["Kids", "too strict"], 0.5)

    def test_review_required_when_the_model_is_not_sure(self):
        unsure = {"Kids": .3, "Family": .3, "Teens": .2, "Adults": .2}
        sure = {"Kids": .05, "Family": .05, "Teens": .1, "Adults": .8}
        self.assertEqual(nc.t3_review_label(unsure, 0.5)[0], nc.T3_REVIEW)
        self.assertEqual(nc.t3_review_label(sure, 0.5), ("Adults", 0.8))
        self.assertEqual(nc.t3_review_label(unsure, 0.25)[0], "Kids")

    def test_the_review_curve_trades_coverage_for_quality(self):
        rng = np.random.default_rng(2)
        scores = pd.DataFrame(rng.dirichlet([1, 1, 1, 1], 400), columns=nc.GROUP_ORDER)
        p_right = np.clip(scores.max(axis=1).values + 0.2, 0, 0.98)
        truth = scores.idxmax(axis=1).where(rng.random(400) < p_right, rng.choice(nc.GROUP_ORDER, 400))
        curve = nc.t3_review_curve(truth, scores)
        self.assertEqual(curve.iloc[0]["answered"], 1.0)
        self.assertTrue((np.diff(curve["answered"]) <= 0).all())
        self.assertGreater(curve.iloc[-1]["accuracy of answered"], curve.iloc[0]["accuracy of answered"])

    # Model comparisons
    def test_the_ordinal_model_gives_ordered_probabilities_that_add_up(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(4)
        X = rng.normal(size=(500, 3))
        latent = X[:, 0] + 0.3 * rng.normal(size=500)
        y = np.array(nc.GROUP_ORDER)[np.digitize(latent, np.quantile(latent, [.25, .5, .75]))]
        model = nc.T3Ordinal(RandomForestClassifier(n_estimators=30, random_state=0)).fit(X, y)
        self.assertEqual(list(model.classes_), nc.GROUP_ORDER)
        p = model.predict_proba(X)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertTrue((p >= 0).all())
        self.assertGreater((model.predict(X) == y).mean(), 0.5)
        noisy = np.array(nc.GROUP_ORDER)[np.digitize(latent + 0.9 * rng.normal(size=500), np.quantile(latent, [.2, .4, .75]))]
        mid = nc.T3Ordinal(RandomForestClassifier(n_estimators=40, min_samples_leaf=3, random_state=0)).fit(X, noisy)
        self.assertTrue({"Family", "Teens"} <= set(mid.predict(X)))
        with self.assertRaises(ValueError):
            nc.T3Ordinal(RandomForestClassifier(n_estimators=5)).fit(X, np.digitize(latent, [0.0]))
        from sklearn.base import clone
        self.assertEqual(clone(model).fit(X, y).classes_.tolist(), nc.GROUP_ORDER)

    def test_every_ablation_variant_runs_and_removes_what_it_says(self):
        X, y = self._xy()
        for variant in nc.T3_ABLATIONS:
            pipe = nc.t3_ablation_pipeline(nc.t3_model("Decision Tree"), variant).fit(X, y)
            self.assertEqual(len(pipe.predict(X)), len(X), variant)
        stripped = nc.t3_drop_audience_genres(X)
        for text in stripped["listed_in"]:
            self.assertFalse({g.lower() for g in nc.split_items(text)} & set(nc.T3_AUDIENCE_GENRES))
        self.assertEqual(list(stripped["n_genres"]), [len(nc.split_items(t)) for t in stripped["listed_in"]])
        self.assertTrue((nc.t3_drop_all_genres(X)["listed_in"] == "").all())
        with self.assertRaises(ValueError):
            nc.t3_ablation_pipeline(nc.t3_model("Decision Tree"), "nonsense")

    def test_title_words_can_be_added_as_a_feature(self):
        X, y = self._xy()
        pipe = nc.t3_pipeline(nc.t3_model("Decision Tree"), title_words=True).fit(X, y)
        names = pipe.named_steps["prep"].get_feature_names_out()
        self.assertTrue(any(str(n).startswith("title__") for n in names))

    # Result summaries
    def test_the_significance_text_matches_the_saved_results(self):
        level, lines = nc.t3_significance_text(GOOD_ORDER_SIG)
        self.assertAlmostEqual(level, 1 - 0.05 / 6) 
        text = "\n".join(lines)
        self.assertIn("Random Forest beat Decision Tree by 5.0 points in accuracy (range +2.3 to +7.8): a clear gap.", text)
        self.assertIn("Random Forest beat Decision Tree by 3.6 points in macro-F1", text)
        self.assertIn("Random Forest vs Logistic Regression", text)
        self.assertEqual(text.count("a clear gap"), 2)
        self.assertEqual(text.count("not clearly different"), 4)
        self.assertNotIn("within a point or two", text)

    def test_the_tuning_text_says_the_results_are_mixed(self):
        lines, summary = nc.t3_tuning_text(REAL_COMPARISON)
        self.assertEqual(len(lines), 3)
        self.assertIn("accuracy by -1.2 and macro-F1 by +1.3", " ".join(lines))
        self.assertIn("mixed", summary)
        both = REAL_COMPARISON.copy()
        both.loc[both["model"] != "Baseline (majority)", "cv_macro_f1"] = 0.6
        self.assertIn("cross-validation macro-F1 went from", " ".join(nc.t3_tuning_text(both)[0]))

    def test_the_tuned_settings_of_all_models_are_read_from_the_params_file(self):
        import json
        path = os.path.join(self.tmp.name, "params.json")
        with open(path, "w") as f:
            json.dump({"model": "Random Forest", "params": {"max_depth": 5},
                       "all_tuned": {"Decision Tree": {"max_depth": 4}}}, f)
        self.assertEqual(nc.t3_all_tuned(path), {"Decision Tree": {"max_depth": 4}})
        self.assertEqual(nc.t3_config(path)[:2], ("Random Forest", {"max_depth": 5})) 
        self.assertEqual(nc.t3_all_tuned(os.path.join(self.tmp.name, "missing.json")), {})

    # @staticmethod
    # def _source(name):
    #     with open(os.path.join(os.path.dirname(__file__), "..", name)) as f:
    #         return f.read()
    @staticmethod
    def _source(name):
        with open(
            os.path.join(os.path.dirname(__file__), "..", name),
            encoding="utf-8"
        ) as f:
            return f.read()

    def _run_script(self, name, *args):
        import subprocess
        project = os.path.join(os.path.dirname(__file__), "..")
        work = tempfile.mkdtemp(dir=self.tmp.name)
        out = subprocess.run([sys.executable, "-W", "ignore", os.path.join(project, name), *args],
                             cwd=work, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": project})
        return out, work

    def test_the_robustness_script_runs_and_never_touches_the_test_part(self):
        out, work = self._run_script("task3_robustness.py", self.path, "--quick")
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for name in ["task3_repeated_cv.csv", "task3_alternatives.csv", "task3_ablation.csv"]:
            self.assertTrue(os.path.exists(os.path.join(work, name)), name)
        ablation = pd.read_csv(os.path.join(work, "task3_ablation.csv"))
        self.assertEqual(list(ablation["variant"]), nc.T3_ABLATIONS)
        alternatives = pd.read_csv(os.path.join(work, "task3_alternatives.csv"))
        self.assertIn("ordinal (3 binary forests)", list(alternatives["variant"]))
        self.assertIn("del _test_part", self._source("task3_robustness.py"))

    def test_the_robustness_script_stops_with_a_message_on_a_one_group_file(self):
        df = t3_frame(300)
        df["rating"] = "TV-Y"
        path = os.path.join(self.tmp.name, "one_group_rob.csv")
        df.to_csv(path, index=False)
        out, _ = self._run_script("task3_robustness.py", path, "--quick")
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn("Traceback", out.stderr)

    def test_the_dashboard_cache_for_task_3_is_keyed_on_the_files(self):
        source = self._source("app.py")
        self.assertIn("def get_t3(path, modified, config_modified):", source)
        self.assertIn("get_t3(data_path, os.path.getmtime(data_path),", source)
        self.assertIn("task3_best_params.json", source)

    def test_the_dashboard_text_comes_from_the_saved_results_not_from_typed_claims(self):
        source = self._source("app.py")
        self.assertNotIn("within a point or two", source)
        self.assertNotIn("95% range of the gap", source)
        self.assertIn("nc.t3_significance_text(sig3)", source)
        self.assertIn("nc.t3_tuning_text(comp3)", source)

    def test_the_comparison_chart_draws_the_real_macro_f1_baseline(self):
        source = self._source("task3_rating_classification.py")
        self.assertIn("baseline_f1", source)
        self.assertNotIn('baseline if col == "accuracy" else 0', source)

    def test_training_reports_the_safety_numbers_and_the_error_analysis(self):
        res = nc.train_t3(self.path, params_file=os.path.join(self.tmp.name, "none.json"))
        for key in ["safety", "error_table", "cost_matrix"]:
            self.assertIn(key, res)
        self.assertTrue(0 <= res["safety"]["severe_lenient_rate"] <= 1)
        self.assertEqual(list(res["error_table"].index), [g for g in nc.GROUP_ORDER if g in res["error_table"].index])


# Tsk 2 regression chk
class Task2AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df = t3_frame(700, seed=3)
        cls.df["type"] = np.where(cls.df["duration"].str.contains("Season"), "TV Show", "Movie")
        cls.path = os.path.join(cls.tmp.name, "t2.csv")
        cls.df.to_csv(cls.path, index=False)
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    @staticmethod
    def _source(name):
        with open(os.path.join(os.path.dirname(__file__), "..", name), encoding="utf-8") as f:
            return f.read()
    def _data(self):
        return nc.prep_t2(self.path, fill_years=False)
    def _run_script(self, *args):
        import subprocess
        project = os.path.join(os.path.dirname(__file__), "..")
        work = tempfile.mkdtemp(dir=self.tmp.name)
        out = subprocess.run([sys.executable, "-W", "ignore", os.path.join(project, "task2_content_type_prediction.py"), *args],
                             cwd=work, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": project})
        return out, work

    # Training-only selection
    def test_cross_validation_and_the_choice_never_see_the_test_rows(self):
        df, y = self._data()
        tr, te = nc.t2_split(y)
        first = nc.t2_study(df, y, nc.T2_NUM, split=(tr, te), folds=3)
        df2, y2 = df.copy(), y.copy()
        y2.iloc[te] = 1 - y2.iloc[te].values
        df2.iloc[te, df2.columns.get_loc("year_added")] = 9999.0 
        second = nc.t2_study(df2, y2, nc.T2_NUM, split=(tr, te), folds=3)
        cv_cols = [c for c in first["table"].columns if c.startswith("cv_")]
        pd.testing.assert_frame_equal(first["table"][cv_cols], second["table"][cv_cols])
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["selection_note"], second["selection_note"])
        real = [m for m in first["table"].index if m != "Baseline (majority)"]
        self.assertFalse(np.allclose(first["table"].loc[real, "accuracy"], second["table"].loc[real, "accuracy"]))

    def test_the_choice_looks_only_at_the_cross_validation_columns(self):
        import inspect
        self.assertIn('t2_choose(real[["cv_macro_f1", "cv_macro_f1_std"]])', inspect.getsource(nc.t2_study))
        cv = pd.DataFrame({"cv_macro_f1": [0.90, 0.93, 0.80], "cv_macro_f1_std": [0.01, 0.01, 0.02]},
                          index=["Logistic Regression", "Decision Tree", "Random Forest"])
        name, note = nc.t2_choose(cv)
        self.assertEqual(name, "Decision Tree")
        self.assertNotIn("Close call", note)
        tie = cv.copy()
        tie.loc["Logistic Regression", "cv_macro_f1"] = 0.925
        name, note = nc.t2_choose(tie)
        self.assertEqual(name, "Decision Tree")
        self.assertIn("Close call", note)
        self.assertIn("Logistic Regression", note)
    def test_the_cross_validation_scores_come_from_the_training_part_only(self):
        df, y = self._data()
        study = nc.t2_study(df, y, nc.T2_NUM, folds=3)
        tr, te = study["split"]
        self.assertEqual((study["n_train"], study["n_test"]), (len(tr), len(te)))
        self.assertFalse(set(tr) & set(te))
        self.assertEqual(len(tr) + len(te), len(y))
        self.assertAlmostEqual(len(te) / len(y), 0.2, delta=0.01)
    def test_the_saved_selection_is_read_back_and_drives_the_dashboard_model(self):
        path = os.path.join(self.tmp.name, "sel.json")
        nc.t2_save_selection("Random Forest", {"cv_macro_f1": 0.93123, "cv_macro_f1_std": 0.0123}, "a note", path=path)
        self.assertEqual(nc.t2_config(path), ("Random Forest", "selected by cross-validation"))
        self.assertEqual(nc.t2_selection_info(path)["note"], "a note")
        self.assertEqual(nc.t2_config(os.path.join(self.tmp.name, "none.json"))[1], "default")
        work, here = tempfile.mkdtemp(dir=self.tmp.name), os.getcwd()
        nc.t2_save_selection("Decision Tree", {"cv_macro_f1": 0.9, "cv_macro_f1_std": 0.01}, "chosen by cv",
                             path=os.path.join(work, nc.T2_SELECTION_FILE))
        try:
            os.chdir(work)
            m = nc.train_t2(self.path)
        finally:
            os.chdir(here)
        self.assertEqual(m["model_name"], "Decision Tree")
        self.assertEqual(m["selection"]["note"], "chosen by cv")

    def test_the_dashboard_shows_the_selected_model_and_keys_its_cache_on_the_files(self):
        source = self._source("app.py")
        self.assertNotIn("A Logistic Regression model", source)
        self.assertIn("m['model_name']", source)
        self.assertIn("def get_t2(path, modified, config_modified):", source)
        self.assertIn("get_t2(data_path, os.path.getmtime(data_path),", source)
        self.assertIn("os.path.getmtime(nc.T2_SELECTION_FILE)", source)
        self.assertNotIn("does **not** use duration or genres", source)
        self.assertIn("how many genres are listed", source)

    #permetrics
    def test_balanced_metrics_match_a_hand_calculation(self):
        cm = np.array([[1189, 36], [56, 477]])
        yt = [0] * cm[0].sum() + [1] * cm[1].sum()
        yp = [0] * cm[0, 0] + [1] * cm[0, 1] + [0] * cm[1, 0] + [1] * cm[1, 1]
        m = nc.t2_metrics(yt, yp)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.9328, places=4)
        self.assertAlmostEqual(m["macro_f1"], 0.9374, places=4)
        self.assertAlmostEqual(m["recall_Movie"], 1189 / 1225)
        self.assertAlmostEqual(m["recall_TV"], 477 / 533)
        self.assertAlmostEqual(m["accuracy"], (1189 + 477) / 1758)
        rep = nc.t2_class_report(yt, yp)
        self.assertEqual(list(rep.index), ["Movie", "TV Show"])
        self.assertEqual(list(rep["support"]), [1225, 533])

    #trainingonly
    def test_the_release_year_imputer_uses_the_training_rows_only(self):
        X = pd.DataFrame({"release_year": [2000.0] * 20 + [2020.0] * 20, "year_added": 2021.0, "years_to_netflix": 21.0})
        imp = nc.T2Imputer().fit(X.iloc[:20])
        self.assertEqual(imp.year_, 2000.0) 
        test = pd.DataFrame({"release_year": [np.nan], "year_added": [2021.0], "years_to_netflix": [np.nan]})
        out = imp.transform(test)
        self.assertEqual(out.at[0, "release_year"], 2000.0)
        self.assertEqual(out.at[0, "years_to_netflix"], 21.0)
        self.assertEqual(nc.T2Imputer().fit(X).year_, 2010.0)

    def test_a_missing_release_year_stays_empty_and_the_model_still_trains(self):
        df = self.df.copy()
        df.loc[df.index[:4], "release_year"] = np.nan
        path = os.path.join(self.tmp.name, "t2_gaps.csv")
        df.to_csv(path, index=False)
        out, y = nc.prep_t2(path, fill_years=False)
        self.assertEqual(int(out["release_year"].isna().sum()), 4)
        self.assertTrue(any("training rows only" in n for n in out.attrs["issues"]))
        self.assertFalse(nc.prep_t2(path)[0]["release_year"].isna().any())
        m = nc.train_t2(path)
        self.assertIn("balanced_accuracy", m)

    def test_the_ablation_says_what_each_group_of_columns_is_worth(self):
        df, y = self._data()
        table = nc.t2_ablation(df, y, "Decision Tree", folds=3)
        self.assertEqual(list(table.index), list(nc.T2_ABLATIONS))
        self.assertEqual(table.iloc[0]["macro_f1 change vs full"], 0.0)
        self.assertIn("without genre count", table.index)

    def test_show_id_is_the_identity_and_look_alikes_are_kept_and_reported(self):
        df = self.df.copy()
        twins = df.head(3).copy()
        twins["show_id"] = ["twin1", "twin2", "twin3"]
        repeats = df.iloc[10:15].copy()
        path = os.path.join(self.tmp.name, "t2_ids.csv")
        pd.concat([df, twins, repeats], ignore_index=True).to_csv(path, index=False)
        out, _ = nc.prep_t2(path, fill_years=False)
        self.assertEqual(len(out), len(df) + 3)
        notes = " | ".join(out.attrs["issues"])
        self.assertIn("5 row(s) repeat an earlier show_id", notes)
        self.assertIn("share title, type and release year with another row but have a different show_id", notes)
        self.assertEqual(len(nc.load_t1(path)), len(out)) 

    def test_the_script_saves_the_choice_and_reports_both_classes(self):
        out, work = self._run_script(self.path)
        self.assertEqual(out.returncode, 0, out.stderr[-700:])
        comp = pd.read_csv(os.path.join(work, "task2_model_comparison.csv"))
        for col in ["balanced_accuracy", "macro_f1", "cv_macro_f1", "selected", "recall_Movie", "recall_TV"]:
            self.assertIn(col, comp.columns)
        self.assertFalse(comp[comp["experiment"].str.startswith("A")]["selected"].any())
        b = comp[comp["experiment"].str.startswith("B")]
        real = b[b["model"] != "Baseline (majority)"]
        chosen = real[real["selected"]]["model"].tolist()
        self.assertEqual(chosen, [real.loc[real["cv_macro_f1"].idxmax(), "model"]])
        import json
        with open(os.path.join(work, nc.T2_SELECTION_FILE)) as f:
            self.assertEqual(json.load(f)["model"], chosen[0])
        with open(os.path.join(work, "task2_classification_report.txt")) as f:
            report = f.read()
        self.assertIn("balanced accuracy", report)
        self.assertIn("TV Show", report)
        self.assertTrue(os.path.exists(os.path.join(work, "task2_ablation.csv")))

    def test_the_script_and_the_dashboard_fill_gaps_inside_the_model(self):
        source = self._source("task2_content_type_prediction.py")
        self.assertIn("nc.prep_t2(path, fill_years=False)", source)
        self.assertIn("nc.t2_study(", source)
        self.assertNotIn('sort_values("f1_TV"', source)
        self.assertNotIn("cross_val_score(", source)
        self.assertIn("prep_t2(path, fill_years=False)", self._source("netflix_core.py"))

def two_cluster_frame(n=40):
    """Two far-apart groups of titles: US dramas (movies) and Indian kids' shows (TV shows)."""
    rows = []
    for k in range(n):
        rows.append({"show_id": f"a{k}", "type": "Movie", "title": f"Alpha {k}", "director": f"Dir A{k % 5}",
                     "country": "United States", "date_added": "1/1/2020", "release_year": 2000 + k % 20, "rating": "R",
                     "duration": "100 min", "listed_in": "Dramas"})
        rows.append({"show_id": f"b{k}", "type": "TV Show", "title": f"Beta {k}", "director": f"Dir B{k % 5}",
                     "country": "India", "date_added": "1/1/2020", "release_year": 2000 + k % 20, "rating": "TV-Y",
                     "duration": "1 Season", "listed_in": "Kids' TV"})
    return pd.DataFrame(rows)
class Task1AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.frame = t3_frame(700, seed=5)
        cls.path = os.path.join(cls.tmp.name, "t1.csv")
        cls.frame.to_csv(cls.path, index=False)
        cls.df = nc.load_t1(cls.path)
        cls.rec = nc.Recommender(cls.df)
        cls.clusters = os.path.join(cls.tmp.name, "clusters.csv")
        two_cluster_frame().to_csv(cls.clusters, index=False)
        cls.crec = nc.Recommender(nc.load_t1(cls.clusters))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # @staticmethod
    # def _source(name):
    #     with open(os.path.join(os.path.dirname(__file__), "..", name)) as f:
    #         return f.read()
    @staticmethod
    def _source(name):
        with open(
            os.path.join(os.path.dirname(__file__), "..", name),
            encoding="utf-8"
        ) as f:
            return f.read()

    def _run(self, script, *args, path=None):
        import subprocess
        project = os.path.join(os.path.dirname(__file__), "..")
        work = tempfile.mkdtemp(dir=self.tmp.name)
        out = subprocess.run([sys.executable, "-W", "ignore", os.path.join(project, script), path or self.path, *args],
                             cwd=work, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": project})
        return out, work

    def test_evaluation_and_dashboard_use_the_same_ranking(self):
        years = self.df["release_year"].values
        for i in [0, 17, 250, 699]:
            rows = nc.top_neighbours(self.rec.X, years, i, 10)
            self.assertEqual(list(rows), list(self.rec.neighbours(i, 10)))
            shown = self.rec.recommend(i, n=10)
            self.assertEqual(list(shown["title"]), list(self.df.loc[rows, "title"]))

    def test_the_evaluation_scores_are_computed_on_the_lists_the_dashboard_shows(self):
        res = nc.t1_evaluate(self.df, self.rec.X, n_queries=40)
        queries = np.random.default_rng(42).choice(len(self.df), 40, replace=False)
        genre_sets = [{g.lower() for g in nc.split_items(t)} for t in self.df["listed_in"]]
        manual = []
        for i in queries:
            a = genre_sets[i]
            tops = self.rec.neighbours(int(i), 10)
            manual.append(np.mean([len(a & genre_sets[j]) > 0 for j in tops]))
        np.testing.assert_allclose(res["genre_hit_rate@10"], manual)

    def test_ties_follow_the_same_release_year_rule_in_every_path(self):
        df = two_cluster_frame(30)
        df = df[df["type"] == "Movie"].copy()
        df["director"], df["release_year"] = "Same", [1980 + 3 * k for k in range(len(df))]
        path = os.path.join(self.tmp.name, "ties.csv")
        df.to_csv(path, index=False)
        rec = nc.Recommender(nc.load_t1(path))
        q = 15
        rows = nc.top_neighbours(rec.X, rec.df["release_year"].values, q, 10)
        gaps = np.abs(rec.df["release_year"].values[rows] - rec.df.at[q, "release_year"])
        self.assertEqual(list(gaps), sorted(gaps))
        self.assertEqual(list(rows), list(rec.neighbours(q, 10)))

    def test_no_script_uses_the_unranked_shortcut(self):
        for name in ["task1_recommendation_system.py", "task1_weight_tuning.py"]:
            self.assertNotIn("argpartition", self._source(name), name)
        import inspect
        self.assertNotIn("argpartition", inspect.getsource(nc.t1_evaluate))
        self.assertIn("top_neighbours", inspect.getsource(nc.t1_evaluate))

    # eevaluation metrics
    def test_metrics_are_classified_by_what_they_can_show(self):
        k = nc.t1_metric_kind
        self.assertEqual(k("Random recommendations", "genre_overlap@10"), "baseline")
        self.assertTrue(k("No-genre model (fair test)", "genre_overlap@10").startswith("independent proxy"))
        self.assertTrue(k("No-genre model (fair test)", "genre_hit_rate@10").startswith("independent proxy"))
        self.assertTrue(k("No-genre model (fair test)", "same_type@10").startswith("metadata consistency"))
        for metric in ["genre_overlap@10", "genre_hit_rate@10", "same_type@10"]:
            self.assertTrue(k("Full model", metric).startswith("metadata consistency"), metric)

    def test_the_evaluation_table_has_ranges_and_kinds(self):
        per_model = {"Random recommendations": nc.t1_evaluate(self.df, None, n_queries=60),
                     "No-genre model (fair test)": nc.t1_evaluate(self.df, nc.make_features(self.df, use_genre=False), n_queries=60),
                     "Full model": nc.t1_evaluate(self.df, self.rec.X, n_queries=60)}
        means, long = nc.t1_eval_table(per_model)
        self.assertEqual(list(means.index), list(per_model))
        self.assertTrue(((long["low_95"] <= long["mean"]) & (long["mean"] <= long["high_95"])).all())
        self.assertTrue((long["queries"] == 60).all())
        self.assertEqual(set(long["kind"]), {"baseline", "independent proxy (genres hidden from the model)",
                                              "metadata consistency (the model uses the type)",
                                              "metadata consistency (the model uses the genres)"})
        full = means.loc["Full model"]
        self.assertGreater(full["genre_hit_rate@10"], means.loc["No-genre model (fair test)", "genre_hit_rate@10"])

    # multi-title recdtion
    def test_average_is_the_taste_profile_and_each_keeps_every_pick_represented(self):
        picks = [0, 1]
        years = self.crec.df["release_year"].values
        profile = nc.normalize(np.asarray(self.crec.X[picks].mean(axis=0)))
        sims = np.asarray(self.crec.X @ profile.T).ravel()
        expected = nc.rank_like_production(sims, years, float(years[picks].mean()), 10, skip=picks)
        avg = self.crec.recommend_many(picks, n=10)
        self.assertEqual(list(avg["title"]), list(self.crec.df.loc[expected, "title"]))
        self.assertNotIn("because_of", avg.columns)
        each = self.crec.recommend_many(picks, n=10, method="each")
        self.assertEqual(len(each), 10)
        self.assertEqual(len(set(each["title"])), 10)
        self.assertFalse(set(each["title"]) & set(self.crec.df.loc[picks, "title"]))
        self.assertEqual(set(each["because_of"]), set(self.crec.df.loc[picks, "title"]))
        self.assertEqual(each["type"].value_counts().to_dict(), {"Movie": 5, "TV Show": 5})
        self.assertEqual(list(each["because_of"][:2]), list(self.crec.df.loc[picks, "title"]))
    def test_unknown_combine_methods_are_refused(self):
        with self.assertRaises(ValueError):
            self.crec.recommend_many([0, 1], method="nonsense")

    def test_each_in_turn_respects_the_filters(self):
        out = self.crec.recommend_many([0, 2], n=6, method="each", same_type=True)
        self.assertTrue((out["type"] == "Movie").all())

    #humanrev
    def test_the_review_summary_reads_two_reviewers_and_flags_bad_scores(self):
        a = [5, 4, 3, 5, 2, 4, 4, 5, 1, 3, 7, 3.5, "x"]
        b = [4, 4, 3, 5, 3, 4, 5, 5, 2, 3, "", "", ""]
        sheet = pd.DataFrame({"query": ["Q1"] * 5 + ["Q2"] * 5 + ["Q3"] * 3, "suggestion": "s", "type": "Movie", "genres": "g",
                              "reviewer_score_1to5": a, "reviewer2_score_1to5": b})
        s = nc.t1_review_summary(sheet)
        self.assertEqual((s["rows"], s["reviewers"], s["scored_rows"], s["invalid"]), (13, 2, 10, 3))
        self.assertAlmostEqual(s["mean"], np.mean(a[:10] + b[:10]))
        self.assertEqual(sum(s["distribution"].values()), 20)
        self.assertAlmostEqual(s["within_one_point"], 1.0)
        self.assertEqual(s["agreement_rows"], 10)
        text = " ".join(nc.t1_review_text(s))
        self.assertIn("10 of 13 suggestions", text)
        self.assertIn("2 reviewer", text)
        self.assertIn("Limits of this review", text)
        self.assertIn("were not whole numbers from 1 to 5 and were ignored", text)

    def test_a_blank_sheet_says_the_review_is_not_done(self):
        sheet = pd.DataFrame({"query": ["Q"] * 5, "suggestion": "s", "type": "Movie", "genres": "g", "reviewer_score_1to5": ""})
        s = nc.t1_review_summary(sheet)
        self.assertEqual(s["scored_rows"], 0)
        self.assertIn("has not been done yet: 0 of 5", nc.t1_review_text(s)[0])

    def test_the_review_sheet_covers_both_types_and_comes_with_a_rubric(self):
        queries = nc.t1_review_queries(self.df, 15)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertEqual(len(queries), 15)
        self.assertTrue({"Movie", "TV Show"} <= set(self.df.loc[queries, "type"]))
        sheet = nc.t1_review_sheet(self.df, self.rec)
        self.assertEqual(len(sheet), 15 * 5)
        for col in nc.T1_REVIEW_COLUMNS:
            self.assertIn(col, sheet.columns)
        self.assertIn("1 = not similar at all", nc.T1_REVIEW_RUBRIC)
        self.assertIn("must not\nlook at the first person's scores", nc.T1_REVIEW_RUBRIC)

    def test_the_script_refuses_an_ambiguous_title_when_nobody_can_answer(self):
        frame = self.frame.copy()
        frame.loc[5, ["title", "release_year"]] = ["Twin Title", 2001]
        frame.loc[6, ["title", "release_year"]] = ["Twin Title", 2015]
        path = os.path.join(self.tmp.name, "twins.csv")
        frame.to_csv(path, index=False)
        out, _ = self._run("task1_recommendation_system.py", "Twin Title", path=path)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        self.assertIn('"Twin Title" matches 2 titles', out.stdout)
        self.assertIn("Not guessing", out.stdout)
        self.assertNotIn("Using the first one", out.stdout)
        self.assertNotIn("Because you watched: Twin Title", out.stdout)
        out2, _ = self._run("task1_recommendation_system.py", "Twin Title (2015)", path=path)
        self.assertIn("Because you watched: Twin Title (2015, ", out2.stdout)
    def test_the_script_writes_labelled_evaluation_files_and_the_review_sheet(self):
        out, work = self._run("task1_recommendation_system.py")
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        ci = pd.read_csv(os.path.join(work, "task1_evaluation_ci.csv"))
        self.assertTrue({"model", "metric", "mean", "low_95", "high_95", "kind"} <= set(ci.columns))
        main = pd.read_csv(os.path.join(work, "task1_evaluation.csv"), index_col=0)
        self.assertEqual(list(main.index), ["Random recommendations", "No-genre model (fair test)", "Full model"])
        sheet = pd.read_csv(os.path.join(work, "task1_human_review_sheet.csv"))
        self.assertIn("reviewer2_score_1to5", sheet.columns)
        self.assertTrue(os.path.exists(os.path.join(work, "task1_human_review_rubric.txt")))
        self.assertIn("Independent proxy (genres hidden from the model)", out.stdout)
        self.assertIn("metadata consistency, not evidence of quality", out.stdout)

    def test_the_tuning_script_tests_picks_from_very_different_genres(self):
        out, work = self._run("task1_weight_tuning.py")
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        profile = pd.read_csv(os.path.join(work, "task1_profile_test.csv"))
        self.assertEqual(set(profile["scenario"]), {"3 titles, same genre", "3 titles, 3 different genres"})
        self.assertEqual(set(profile["strategy"]), {"1 liked title", "average of 3", "each of 3 in turn"})
        with open(os.path.join(work, "task1_tuning_summary.txt")) as f:
            summary = f.read()
        self.assertIn("genre weight (3.0) stays a judgment call", summary)
        self.assertIn("circular", summary)

    def test_the_dashboard_cache_for_task_1_is_keyed_on_the_files(self):
        source = self._source("app.py")
        self.assertIn("def get_recommender(path, modified, weights_modified):", source)
        self.assertIn("get_recommender(data_path, os.path.getmtime(data_path),", source)
        self.assertIn("os.path.getmtime(nc.WEIGHTS_FILE)", source)
        self.assertIn("metadata-based content recommender", source)
        self.assertIn("Independent proxy", source)




if __name__ == "__main__":
    unittest.main()


