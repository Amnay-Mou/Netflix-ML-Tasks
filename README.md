# Netflix Machine Learning Projects

**Auspify Technologies — Machine Learning Internship**  
**Selected tasks:** 1, 2, 3 and 5 | **Language:** Python | **Interface:** Streamlit

## Project overview

This project implements four machine-learning workflows using a historical Netflix catalogue dataset and presents their results in a unified Streamlit dashboard.

| Task | Objective | Main approach |
|---|---|---|
| **1 — Content Recommendation System** | Suggest titles with similar content metadata | Weighted TF-IDF features, cosine similarity, multi-title preference profiles |
| **2 — Content Type Prediction** | Predict **Movie** versus **TV Show** | Logistic Regression, Decision Tree and Random Forest, with leak-aware evaluation |
| **3 — Audience Rating Classification** | Predict a derived age-suitability category | Multiclass classification, cross-validated tuning, error/safety analysis |
| **5 — Trend Forecasting** | Forecast monthly Netflix catalogue additions | Time-series baselines, regression, Holt–Winters and Random Forest lags |

The code is intended as a reproducible **educational demonstration**, not a production recommendation, child-safety or business-forecasting service. The underlying catalogue snapshot ends in **September 2021**; forecasting is explicitly historical.

## Setup

### Requirements

- Python **3.11** recommended (the scripts are written for Python 3 and use scikit-learn, pandas and related libraries).
- Python packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `streamlit`.
- The Netflix data file, normally named **`Dataset.csv`**, in the project directory.

Install the packages:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib streamlit
```

On Windows, `py -m pip install ...` may be used instead. If your submitted project includes a tested `requirements.txt`, you can alternatively run `python -m pip install -r requirements.txt`. **Verify that this file is actually included**; it was not among the files reviewed for this README.

### Organize the files

The project scripts import `netflix_core.py`, so keep these files in the **same working folder**. Use their normal names, without the numbered suffixes added to some uploaded copies:

```text
netflix-ml-project/
├── Dataset.csv                         # supply separately
├── README.md
├── netflix_core.py                    # shared features, models and evaluation helpers
├── app.py                             # Streamlit dashboard
├── task1_recommendation_system.py
├── task1_weight_tuning.py
├── task2_content_type_prediction.py
├── task3_rating_classification.py
├── task3_raw_vs_grouped.py
├── task3_robustness.py
├── task5_trend_forecasting.py
├── tests/                              # include if shipping the existing tests
│   ├── test_core.py
│   └── make_demo_data.py
└── [generated CSV, JSON, PNG and TXT outputs]
```

**Expected dataset columns** across the project: `show_id`, `type`, `title`, `director`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`. Task 1 needs only a subset; Task 5 primarily uses `type` and `date_added`. An optional `description` field can enrich Task 1, but the evaluated dataset does **not** contain plot descriptions or cast information.

## Reproduce the workflows

Run these commands **from the project directory**, in order:

```bash
# Task 1: tune metadata weights, then generate/evaluate recommendations
python task1_weight_tuning.py Dataset.csv
python task1_recommendation_system.py Dataset.csv

# Task 2: content-type classification
python task2_content_type_prediction.py Dataset.csv

# Task 3: classification, then two diagnostic studies
python task3_rating_classification.py Dataset.csv
python task3_raw_vs_grouped.py Dataset.csv
python task3_robustness.py Dataset.csv

# Task 5: historical trend analysis and forecasting
python task5_trend_forecasting.py Dataset.csv

# Dashboard
python -m streamlit run app.py
```

Use `py` instead of `python` on Windows if necessary. An alternative starting month for Task 5 can be supplied as, for example, `python task5_trend_forecasting.py Dataset.csv 2016-01`. Task 3's robustness script also supports `--quick` for a smaller, less comprehensive check.

The commands write results to the **current working directory**. Task 1's human-review CSV is generated on the first run and **preserved** on later runs, rather than being overwritten. After filling ratings, rerun `task1_recommendation_system.py` to update the review summary.

If the existing `tests/` directory is included, run:

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic data and are intended to check code behavior; their pass/fail results are **not** evidence of model predictive performance. For complete reproducibility, submit the actual dataset (if redistribution is permitted), environment/dependency information, chosen-model JSON files and generated outputs.

## Task 1 — Netflix Content Recommendation System

### Required workflow

1. **Prepare features:** clean the catalogue, deduplicate by `show_id`, and build metadata fields including genres, director, country, rating, content type, decade and title words.
2. **Vectorize text:** use separate **TF-IDF** blocks, combine them with feature weights and normalize the resulting vectors.
3. **Calculate similarity:** use **cosine similarity** and rank the nearest titles. Ties after rounding similarities to six decimals are resolved using proximity in release year.
4. **Recommend titles:** return top-*N* suggestions for one title or several liked titles. Multiple selections can be represented by an **average preference vector** or by **interleaving** each selected title's list. Optional content-type and minimum-year filters are supported.
5. **Evaluate quality:** compare random recommendations, a **no-genre** model and the full model; assess feature-weight sensitivity; simulate multiple-liked-title use; and collect relevance judgments.

A similarity score measures **metadata resemblance**, not the probability that a person will enjoy the suggestion. No viewing histories, preference labels or persistent user profiles are used.

### Offline results from the supplied exports

Evaluation used **500 query titles**, with the top **10** recommendations per query.

| Method | Mean genre Jaccard overlap@10 | Share with ≥1 common genre@10 | Same content type@10 |
|---|---:|---:|---:|
| Random recommendations | 0.091 | 23.6% | 57.8% |
| No-genre model | **0.338** | **64.9%** | 99.8% |
| Full metadata model | 0.964 | 100.0% | 100.0% |

**Interpretation matters:** The **no-genre** result is an independent *genre-overlap proxy* because genre labels are hidden from that model during recommendation. It is not an independent measure of user enjoyment. The full model receives genres as inputs, and models receiving content type are evaluated on content type; the resulting near-perfect scores describe **metadata consistency**, not independently established recommendation quality. The file `task1_evaluation_ci.csv` contains query-level approximate 95% intervals.

**Weight tuning.** `task1_weight_tuning.py` compares 41 candidate settings for the non-genre feature weights, using separate tuning and checking queries, and accepts a change only when the lower end of an approximate 95% gain interval exceeds zero. The genre weight remains a deliberately chosen design parameter (default **3.0**) because tuning it against genre overlap would be circular. Whether a candidate was actually adopted is recorded in `task1_tuned_weights.json`; verify this file before reporting that the tuned weights were used.

**Multi-title profiles.** In the supplied simulated evaluation, covering **300 simulated users per scenario**, the share of target genres represented among the top-10 recommendations was:

| Simulated preference scenario | Average of 3 titles | Each title in turn |
|---|---:|---:|
| Three liked titles sharing a genre | 71.3% | **79.0%** |
| Three liked titles associated with different genres | 41.4% | **63.4%** |

These figures concern **genre coverage**, not human preference or observed satisfaction. The different-genre selections can still overlap in other labels.

### Relevance review: one human and one AI-assisted assessment

The supplied review sheet contains **75 recommendations across 15 query titles**. Both rating columns are filled using a 1–5 similarity rubric. **These are not two independent human reviews:** `reviewer_score_1to5` contains the user's human ratings and `reviewer2_score_1to5` contains an **AI-assisted assessment** based on available title descriptions. The AI assessments were not blind to the original sheet and should not be described as a second human reviewer.

| Measure | Result |
|---|---:|
| Human ratings, mean | **2.67 / 5** |
| AI-assisted ratings, mean | **2.87 / 5** |
| Combined mean across 150 ratings | **2.77 / 5** |
| Combined median | **3 / 5** |
| Combined ratings scored 4 or 5 | **22%** |
| Ratings within one point of each other | **97.3% of 75 pairs** |

The results suggest that recommendations can share metadata while having only **modest perceived similarity** in storyline, mood or subject. The human assessment is small, and the additional AI ratings are **supporting analysis**, not independent human validation. If two human reviewers are required, obtain a genuinely independent second person's ratings and regenerate the summary.

**Task 1 outputs:** `task1_example_recommendations.csv`, `task1_evaluation.csv`, `task1_evaluation_ci.csv`, `task1_evaluation.png`, `task1_human_review_sheet.csv`, `task1_human_review_rubric.txt`, `task1_human_review_summary.csv`, `task1_tuned_weights.json`, `task1_genre_weight_sensitivity.csv`, `task1_profile_test.csv`, `task1_tuning_summary.txt`, `task1_weight_tuning.png`. The human summary appears after a run that reads the review sheet.

## Task 2 — Netflix Content Type Prediction

### Required workflow

1. Select catalogue metadata including rating, primary country, release year, addition-date features, director and country counts, and number of genres.
2. Impute numeric fields and encode categorical features **inside a scikit-learn pipeline**, using training data only.
3. Train **Logistic Regression**, **Decision Tree** and **Random Forest**, alongside a majority-class baseline.
4. Select the preferred model by **five-fold macro-F1 cross-validation on the training 80% only**.
5. Evaluate the selected model and alternatives on a separate stratified **20% test set** using accuracy, balanced accuracy, class-specific precision/recall/F1, macro-F1, ROC-AUC and confusion matrices.

Three experiments deliberately separate different feature-availability settings:

- **A — Duration leakage demonstration:** raw duration has units such as `90 min` versus `2 Seasons` and almost directly reveals the label. Its 100% results are **not valid evidence of generalizable classification performance**.
- **B — Leak-aware classification:** excludes duration-derived features; this is the selected dashboard model.
- **C — No addition-date features:** a harder scenario where `date_added` is not available.

Genre **names** are excluded from Task 2's classifier, but `n_genres` is included. Feature-ablation experiments measure the importance of selected groups of attributes.

### Main result (supplied experiment B output)

| Metric | Result |
|---|---:|
| Chosen model | **Logistic Regression** |
| Training CV macro-F1 | **0.9428** (SD 0.0033) |
| Held-out test accuracy | **94.88%** |
| Held-out test balanced accuracy | **93.41%** |
| Held-out test macro-F1 | **0.9388** |
| Movie F1 / TV Show F1 | **0.964 / 0.914** |
| ROC-AUC | **0.9740** |
| Majority-class test accuracy | **69.68%** |

The test contains **1,758 titles** (1,225 Movies; 533 TV Shows). The model chosen by training-only CV is written to `task2_selected_model.json`, which the dashboard reads.

**Limits:** Director-related fields strongly influence classification and may reflect dataset-specific metadata practices. Historical versions also inspected this same test set during development, so its final performance should not be claimed as a wholly untouched external validation. The additional no-date scenario is useful for understanding applicability to new titles.

**Task 2 outputs:** `task2_model_comparison.csv`, `task2_selected_model.json`, `task2_classification_report.txt`, `task2_ablation.csv`, `task2_accuracy_comparison.png`, `task2_confusion_matrix.png`.

## Task 3 — Netflix Audience Rating Classification

### Required workflow

1. Analyze the original rating labels and map supported values into **four derived age-suitability groups**.
2. Build encoded feature matrices with a stratified **80/20 train/test split** and training-only imputation.
3. Train a majority baseline, **Logistic Regression**, **Decision Tree** and **Random Forest**.
4. Tune model parameters with training-only cross-validation, using **macro-F1** for model selection.
5. Evaluate held-out accuracy, macro-F1, class-specific metrics, confusion matrices, error costs, model differences and feature sensitivity.

The target groups are **project-defined**, not official Netflix classes or measured audience demographics:

| Group | Original ratings mapped to it |
|---|---|
| **Kids** | G, TV-G, TV-Y, TV-Y7, TV-Y7-FV |
| **Family** | PG, TV-PG |
| **Teens** | PG-13, TV-14 |
| **Adults** | R, TV-MA, NC-17 |

`G`/`TV-G` mean general audiences, not children only; they are grouped with the least restrictive categories. NR/UR, missing and unrecognized ratings are excluded from the supervised target and accounted for. The original rating is **not** a model input. Inputs include content type, genres, country, release/addition dates, director indicators and type-specific duration.

### Main result (held-out evaluation)
The best model, selected using training-only cross-validated macro-F1, was a tuned **Random Forest** with 200 trees, maximum depth 20, minimum leaf size 2 and `max_features="log2"`.

| Metric | Result |
|---|---:|
| Training CV macro-F1 | **0.6265** |
| Test accuracy | **63.15%** |
| Test macro-F1 | **0.6079** |
| Majority-class test accuracy | **46.04%** |
| Test samples | **1,742** |

| Group | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Kids | 0.78 | 0.72 | 0.75 | 181 |
| Family | 0.43 | 0.41 | 0.42 | 230 |
| Teens | 0.54 | 0.53 | 0.54 | 529 |
| Adults | 0.71 | 0.74 | 0.72 | 802 |

Family and Teens remain the most difficult categories. The safety analysis identified **101 of 1,331 Teens/Adults titles (7.59%)** classified as Kids/Family, including five Adults titles classified as Kids. The model should therefore not independently determine age suitability.

Repeated training cross-validation showed that the tuned Random Forest achieved an average macro-F1 of **0.629**, compared with **0.610** for the default model. Alternative class-weighting strategies were also evaluated, but none demonstrated a sufficiently clear improvement to justify replacing the selected configuration.

### Additional diagnostics

- **Raw versus grouped target:** On the separate common-title comparison, a model trained directly on raw ratings achieved **55.7% raw-rating accuracy**. At the grouped level, the grouped Random Forest achieved **63.1% accuracy / 0.606 macro-F1**, versus **62.4% / 0.603** for the raw-rating model mapped back into groups. These are **not** the main held-out results above; the comparison uses its own restricted set of raw-rating classes.
- **Repeated training CV:** The tuned Random Forest averaged 0.6287 macro-F1, compared with 0.6100 for its default settings. The tuning and repeated validation use the training data; they are not a fresh external test.
- **Alternative designs:** Different class weights, ordinal classification and title words were compared; some change safety/accuracy trade-offs without clearly resolving all class weaknesses.
- **Ablation and permutation tests:** Genre metadata contributes strongly to performance, which may reflect dataset-specific category signals rather than all the information required to judge content suitability.

Model-comparison significance checks use paired bootstrap intervals adjusted for multiple comparisons. The saved parameter file `task3_best_params.json` is read by the dashboard.

**Task 3 outputs:** `task3_best_params.json`, `task3_model_comparison.csv`, `task3_model_significance.csv`, `task3_classification_report.txt`, `task3_error_analysis.csv`, `task3_safety.csv`, `task3_review_curve.csv`, `task3_permutation_importance.csv`, `task3_repeated_cv.csv`, `task3_alternatives.csv`, `task3_ablation.csv`, `task3_raw_vs_grouped.csv`; charts include class distribution, model comparison, confusion matrix, feature/permutation importance and raw-vs-grouped comparison.

## Task 5 — Netflix Trend Forecasting

### Forecast target and historical context

Task 5 forecasts the **number of titles added to the Netflix catalogue per month**, from `date_added`. It does **not** forecast the first-release dates of films or television series (`release_year`). This distinction is essential: an older film may enter the catalogue years after its original release.

The supplied snapshot's latest record is from **25 September 2021**. Because that month is incomplete, the evaluated series ends in **August 2021**; its 12 forecast months are **September 2021 through August 2022**. These dates are **in the past**: all forecasts in this project are **historical demonstrations**, not predictions for the current year. The internship provider confirmed that using the historical dataset is acceptable.

### Required workflow

1. Construct monthly counts and lag-based features, with accounting for unreadable dates, early sparse months and an incomplete final month.
2. Analyze year-by-year additions, content-type mix, seasonal averages and recent changes.
3. Compare **six methods**: last-month naive, same-month-last-year naive, 12-month average, trend-plus-season regression, damped Holt–Winters exponential smoothing and Random Forest with lag features.
4. Produce separate 12-month forecasts for all titles, Movies and TV Shows; also create an optional **coherent** forecast where Movies + TV Shows = All titles.
5. Select models using rolling-origin validation **within the training period**, then evaluate on the final **12 months**; inspect errors by origin and horizon, forecast reconciliation and start-date sensitivity.

Time is never randomly shuffled. Rolling windows can **overlap**, so their forecast errors are not statistically independent. The default 12-month-average method is retained unless another candidate improves rolling validation by at least **10%**.

### Results from supplied historical exports

The monthly forecasting series contains **8,341 titles**; the record-accounting file attributes **266** exclusions to early sparse months and **183** to the incomplete September 2021 month (449 total). Those exclusions and the start-date rule can influence the conclusions.

| Forecast series | Model chosen before the final test | Rolling MAE | Final-test MAE | 12-month-average final-test MAE |
|---|---|---:|---:|---:|
| All titles | Naive (last month) | **31.28** | **41.58** | **30.08** |
| Movies | 12-month average | **25.85** | **21.25** | **21.25** |
| TV Shows | Naive (last month) | **9.88** | **13.75** | **13.06** |

*MAE is the average absolute error, measured in titles added per month. Lower is better.* The all-titles model selected by rolling validation **underperformed** the 12-month-average baseline on the final test. The selected model is **not retroactively changed** after inspecting that test. This is a finding about limited forecasting stability, not an error to hide.

The independent monthly forecasts are about **178.0 All titles**, **109.2 Movies** and **61.0 TV Shows**. The latter two sum to **170.2**, not 178.0, because the series are modeled separately. The optional reconciliation export provides a coherent total; the supplied bottom-up version is approximately **170.17 titles/month**. Reconciliation choice is assessed on training-side validation, not selected using the final test. The independent all-titles rolling error was slightly lower than bottom-up in the supplied comparison.

**Uncertainty warning:** Shaded forecast bands show forecast ± final-test RMSE, which is an **illustrative error range**, **not a calibrated prediction interval** with guaranteed coverage. The snapshot also cannot account for titles removed before the dataset was captured, and changing the historical start month can change the preferred model.

**Task 5 outputs:** `task5_data_accounting.csv`, `task5_model_comparison.csv`, `task5_forecast.csv`, `task5_forecast_coherent.csv`, `task5_reconciliation.csv`, `task5_rolling_by_start_date.csv`, `task5_start_sensitivity.csv`, `task5_history.png`, `task5_model_comparison.png`, `task5_forecast_all_titles.png`, `task5_forecast_movies.png`, `task5_forecast_tv_shows.png`.

## Streamlit dashboard

Run `python -m streamlit run app.py` to open an interactive interface with a tab for each task plus a data overview. The dashboard provides title selection, recommendations, classification results, performance metrics, comparison charts and historical forecasts.

The code caches several expensive computations and uses data-file modification times and appropriate saved-model configuration timestamps as cache keys for the main task models. Generated CSV/PNG artifacts must nevertheless correspond to the **same dataset and code version** as the displayed predictions; rerun the scripts after changes rather than mixing old and new exports.

The dashboard's `Browse` dialog relies on a local Tkinter file picker and may not work in some hosted environments; entering a CSV path remains an alternative. Predicted classification scores are not calibrated probabilities, and content-suitability decisions require human review.

## Limitations and responsible interpretation

- **Recommendations:** Content similarity is not user preference. The primary dataset lacks plot, cast, viewing history and explicit relevance labels. Strong genre overlap may coexist with mediocre human relevance judgments.
- **Review provenance:** The second Task 1 rating column is AI-assisted, not a second independent human reviewer. The current script counts filled rating columns as “reviewers”; reporting must correct that wording.
- **Classification:** Test metrics describe this historical dataset. Task 2 may exploit director metadata conventions; Task 3's custom age groups and weak Family/Teens performance limit real-world applicability.
- **Forecasting:** The target is *catalogue additions* in an old snapshot, not current releases or current demand. Forecast selection was not consistently superior to simple baselines, and error bands are illustrative.
- **Reproducibility:** Set random seeds are used in the scripts, but results may vary across environments or changed datasets. Do not combine metrics from different script versions or reinterpret separate evaluation experiments as one test.

## Submission checklist

- [ ] Use the unnumbered filenames shown above and keep their imports together.
- [ ] Include `Dataset.csv` (if permitted) or documented instructions for obtaining the same snapshot.
- [ ] Include a tested dependency list / `requirements.txt` or a reproducible environment description.
- [ ] Rerun Task 1 tuning **before** Task 1 recommendation evaluation; preserve the adopted weights JSON and final outputs.
- [ ] Keep Task 1 human ratings and AI-assisted ratings clearly identified; include the rubric and review summary.
- [ ] Rerun Tasks 2 and 3 after any shared-data-cleaning changes; preserve the selected-model parameter files.
- [ ] Rerun Task 5 and verify that all exports describe the same historical window and selected models.
- [ ] Include relevant charts, sample predictions, dashboard screenshots and any available automated tests.
- [ ] Inspect the README's reported example metrics against the newly regenerated files before submission.

**Note about the values in this README:** Numerical results come from the provided project output files and evaluation summary, cross-checked against the supplied source scripts. The original `Dataset.csv` and a fully versioned runnable submission folder were **not** supplied in this review, so the real-data results have not been independently reproduced here. Regeneration from the final folder is the authoritative check.
