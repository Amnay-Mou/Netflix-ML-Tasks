"""netflix-ml-tasks APP"""
# py -m streamlit run app.py

import os
import subprocess
import sys
import pandas as pd
import streamlit as st
import netflix_core as nc


st.set_page_config(page_title="Netflix ML Dashboard", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root { --strip-width: 30px; --open-width: 320px; }

    section[data-testid="stSidebar"] {
        position: fixed !important; top: 0 !important; bottom: 0 !important; left: 0 !important;
        height: 100vh !important; height: 100dvh !important; z-index: 1000000 !important;
        width: var(--strip-width) !important; min-width: var(--strip-width) !important;
        max-width: var(--strip-width) !important;
        transform: none !important; margin-left: 0 !important;
        overflow: hidden; border-right: 3px solid #E50914;
        transition: width .3s ease, min-width .3s ease, max-width .3s ease, box-shadow .3s ease;
    }
    section[data-testid="stSidebar"]:hover {
        width: var(--open-width) !important; min-width: var(--open-width) !important;
        max-width: var(--open-width) !important; box-shadow: 6px 0 24px rgba(0, 0, 0, .55);
    }

    /* the settings are hidden while the strip is thin, and fade in when it opens */
    section[data-testid="stSidebar"] > div {
        width: var(--open-width); height: 100% !important; opacity: 0; transition: opacity .2s ease;
    }
    section[data-testid="stSidebar"]:hover > div { opacity: 1; transition-delay: .1s; }

    /* a small hint on the thin strip */
    section[data-testid="stSidebar"]::before {
        content: "DATASET ⚙️ "; position: absolute; top: 90px; left: 0; width: var(--strip-width);
        writing-mode: vertical-rl; text-align: center; font-size: 12px; letter-spacing: .15em;
        opacity: .75; transition: opacity .15s ease; pointer-events: none;
    }
    section[data-testid="stSidebar"]:hover::before { opacity: 0; }

    /* the normal collapse / expand buttons are not needed any more */
    [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* keep the page content clear of the strip */
    .block-container { padding-left: calc(var(--strip-width) + 2rem) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    @keyframes float-title {
        0%   { transform: translateY(0px);   text-shadow: 0 6px 12px rgba(229, 9, 20, 0.25); }
        50%  { transform: translateY(-12px); text-shadow: 0 18px 24px rgba(229, 9, 20, 0.45); }
        100% { transform: translateY(0px);   text-shadow: 0 6px 12px rgba(229, 9, 20, 0.25); }
    }
    .app-title { text-align: center; font-size: 3rem; font-weight: 800; letter-spacing: 0.08em;
                 margin: 0.5rem 0 0.2rem 0; animation: float-title 3.5s ease-in-out infinite; }
    .app-subtitle { text-align: center; opacity: 0.7; margin-bottom: 1.5rem; }
    @media (prefers-reduced-motion: reduce) { .app-title { animation: none; } }
    </style>
    <div class="app-title">NETFLIX ML DASHBOARD</div>
    <div class="app-subtitle">Auspify Machine Learning Internship · one tab per task</div>
    """,
    unsafe_allow_html=True,
)

# Dataset settings
def browse_for_csv():
    """Open a local file picker in a separate Python process. This is intended for local Streamlit use."""
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
        "print(filedialog.askopenfilename(title='Choose the dataset (CSV)',\n"
        "      filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]))\n"
    )
    st.session_state["browse_msg"] = ""
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        picked = done.stdout.strip()
        if picked:
            st.session_state["data_path"] = picked
        elif done.returncode != 0:
            st.session_state["browse_msg"] = "The file window could not open here. Type or paste the path instead."
    except Exception:
        st.session_state["browse_msg"] = "The file window could not open here. Type or paste the path instead."


st.session_state.setdefault("data_path", "Dataset.csv")
with st.sidebar:
    st.header("Dataset")
    st.text_input("CSV file path", key="data_path")
    st.button("📂 Browse...", on_click=browse_for_csv, use_container_width=True)
    if st.session_state.get("browse_msg"):
        st.warning(st.session_state["browse_msg"])
    st.caption("Keep the CSV in the same folder as app.py, pick it with Browse, or paste a full path.")

data_path = st.session_state["data_path"]

if not os.path.exists(data_path):
    st.error(f"Cannot find '{data_path}'. Put the dataset next to app.py or hover the left edge to open the Dataset menu and fix the path.")
    st.stop()

@st.cache_data(show_spinner=False)
def check_file(path, modified):
    """'' if the file looks right, otherwise the message to show. (`modified` refreshes the check after edits.)"""
    try:
        nc.check_columns(nc.read_data(path))
        return ""
    except Exception as err:
        return str(err)


problem = check_file(data_path, os.path.getmtime(data_path))
if problem:
    st.error(problem)
    st.stop()

st.markdown(f"<div class='app-subtitle' style='margin-top:-1rem; font-size:.85rem'>📁 Dataset in use: "
            f"{os.path.basename(data_path)}</div>", unsafe_allow_html=True)


# Cached models
@st.cache_resource(show_spinner="Building the recommender (first run only)...")
def get_recommender(path, modified, weights_modified):
    """Cache the recommender using the dataset and tuning-file timestamps."""
    return nc.Recommender(nc.load_t1(path))


@st.cache_resource(show_spinner="Training the Task 2 model (first run only)...")
def get_t2(path, modified, config_modified):
    """Cache Task 2 using the dataset and selected-model timestamps."""
    return nc.train_t2(path)


@st.cache_resource(show_spinner="Training the Task 3 model (first run only)...")
def get_t3(path, modified, config_modified):
    """Cache Task 3 using the dataset and tuned-parameter timestamps."""
    return nc.train_t3(path)


@st.cache_resource(show_spinner="Testing the forecasting models (about a minute, first run only)...")
def get_t5(path, kind, modified):
    """Cache Task 5 by dataset timestamp."""
    return nc.train_t5(path, kind)


@st.cache_data(show_spinner="Reading the data (first run only)...")
def get_eda(path):
    return nc.eda_tables(path)


# UI
def title_choices(df, show_type=False):
    """Row numbers sorted by title, plus the text shown for each row in the dropdown."""
    extra = ", " + df["type"] if show_type else ""
    labels = (df["title"] + " (" + df["release_year"].astype(str) + extra + ")").tolist()
    order = df["title"].str.lower().sort_values().index.tolist()
    return order, labels


def title_picker(df, label, key, show_type=False, default_title=None, multi=False):
    """Select one or several titles by name. Returns row indices."""
    order, labels = title_choices(df, show_type)
    if multi:
        return st.multiselect(label, options=order, format_func=lambda i: labels[i], key=key,
                              placeholder="Type a name or open the list, then click a title")
    start = None
    if default_title:
        start = next((p for p, i in enumerate(order) if df.at[i, "title"].lower() == default_title), None)
    return st.selectbox(label, options=order, index=start, format_func=lambda i: labels[i], key=key,
                        placeholder="Type a name or open the list, then click a title")


def have(i):
    """True if a title was picked; otherwise shows a hint."""
    if i is None:
        st.warning("Pick a title first: type a name or open the list.")
    return i is not None


def show_probs(probs, info=None):
    """Big prediction line plus one progress bar per class. Returns the winner."""
    best = max(probs, key=probs.get)
    st.markdown(f"### Prediction: **{best}** (model score {probs[best]:.0%})")
    if info:
        st.caption(f"{best} = {info[best]}")
    for name, p in probs.items():
        st.write(f"{name}: {p:.0%}")
        st.progress(float(p))
    st.caption("These are model scores, not calibrated probabilities: a 70% score does not mean "
               "the answer is right 70% of the time. Use them to compare the options, not as a guarantee.")
    return best


def show_split_note(i, test_idx):
    """Tell the user whether the model saw this title while training."""
    if i in test_idx:
        st.caption("This title was in the held-out test set: the model never saw it during training.")
    else:
        st.caption("This title was in the training set, so the model learned from it. "
                   "A correct answer here is less impressive than on unseen titles.")


def show_images(files, width):
    for f in files:
        if os.path.exists(f):
            st.image(f, width=width)


tab1, tab2, tab3, tab5, tab0 = st.tabs(["Task 1 🎯 · Recommendations",
                                        "Task 2 🔎 · Movie or TV Show?",
                                        "Task 3 👪 · Audience Rating",
                                        "Task 5 📈 · Trend Forecast",
                                        "Data overview 📊 "])



# Data Overview
with tab0:
    eda = get_eda(data_path)
    st.subheader("Look at the data first")
    st.write("Quick checks before any model: what is missing, and how the main columns are spread.")

    e1, e2, e3 = st.columns(3)
    e1.metric("Titles (after removing duplicates)", f"{eda['n_rows']:,}")
    e2.metric("Movies", f"{int(eda['types'].get('Movie', 0)):,}")
    e3.metric("TV Shows", f"{int(eda['types'].get('TV Show', 0)):,}")

    left, right = st.columns(2)
    left.write("**Missing values per column** (blank or 'Not Given')")
    left.bar_chart(eda["missing"])
    right.write("**Titles by release decade**")
    right.bar_chart(eda["release_decade"])

    left, right = st.columns(2)
    left.write("**15 most common genres** (feeds Task 1 and Task 3)")
    left.bar_chart(eda["top_genres"])
    right.write("**Titles added to Netflix per year** (feeds Task 2)")
    right.bar_chart(eda["added_by_type"])

    left, right = st.columns(2)
    left.write("**Raw ratings** (feeds Task 3)")
    left.bar_chart(eda["ratings"])
    mins = eda["movie_minutes"]["movie_minutes"]
    right.write("**Movie length in minutes** (feeds Task 3)")
    right.bar_chart(pd.cut(mins, bins=range(0, 361, 30)).value_counts().sort_index()
                    .rename(lambda b: f"{int(b.left)}-{int(b.right)}"))
    if eda.get("monthly_added") is not None:
        st.write("**Titles added to Netflix per month** (feeds Task 5; the incomplete last month and the early "
                 "months with only a handful of titles are left out)")
        st.line_chart(eda["monthly_added"])
        with st.expander("Which records are in the Task 5 monthly counts?"):
            st.dataframe(nc.t5_accounting(eda["t5_info"]), hide_index=True)
    elif eda.get("t5_note"):
        st.caption(f"Task 5 monthly counts are not shown: {eda['t5_note']}")
    st.caption("Director and country are the columns with many 'Not Given' values. "
               "We turn them into real missing values and also use 'has a director' as a feature.")

# TSK5

with tab5:
    st.subheader("How many titles are added to Netflix each month?")
    st.write(nc.T5_TARGET_NOTE + " Six models are tried, including simple rules like 'same month last year'. "
             "A model only counts as useful if it beats those simple rules on months it has never seen.")

    which5 = st.radio("Forecast for", ["All titles", "Movies only", "TV Shows only"],
                      horizontal=True, key="t5_which")
    kind5 = {"All titles": None, "Movies only": "Movie", "TV Shows only": "TV Show"}[which5]
    try:
        m5 = get_t5(data_path, kind5, os.path.getmtime(data_path))
    except ValueError as err:
        st.error(str(err))
        m5 = None

    if m5 is not None:
        info5, sel5, hold5, q5 = m5["info"], m5["selected"], m5["holdout"], m5["info"]["quality"]
        st.caption(f"Based on {info5['n_months']} months ({info5['start']:%b %Y} to {info5['end']:%b %Y}).")
        st.warning(nc.t5_snapshot_note(info5, m5["future"].index))
        lost5 = q5["repeated_removed"] + q5["missing_date"] + q5["unreadable_date"]
        if lost5:
            st.warning(f"{lost5} record(s) were left out because of repeated rows or missing / unreadable dates. "
                       "See 'Which records are in these counts?' below.")

        u1, u2, u3 = st.columns(3)
        u1.metric("Model chosen", sel5)
        u2.metric("Average miss, last 12 months", f"{hold5.loc[sel5, 'MAE']:.0f} titles")
        u3.metric("Best simple rule misses by", f"{hold5.loc[m5['best_simple_rule'], 'MAE']:.0f} titles")
        st.info(nc.t5_verdict(m5))
        if m5["best_on_final_test"] != sel5:
            st.caption(f"On the final test {m5['best_on_final_test']} scored best, but the model was chosen "
                       "before the final test and is not switched afterwards.")

        horizon5 = st.slider("Months to forecast", 3, 12, 12, key="t5_h")
        fut5 = m5["future"].iloc[:horizon5]
        hist5 = m5["series"].iloc[-36:]
        chart_cols5 = {"Titles added (history)": hist5, "Forecast": fut5["forecast"],
                       "Typical error range: low": fut5["range_low"],
                       "Typical error range: high": fut5["range_high"]}
        show_ref5 = sel5 != nc.T5_DEFAULT_MODEL
        if show_ref5:
            chart_cols5[f"{nc.T5_DEFAULT_MODEL} (stable reference)"] = m5["default_future"].iloc[:horizon5]
        chart5 = pd.DataFrame(chart_cols5)
        for col in chart5.columns[1:]:
            chart5.loc[hist5.index[-1], col] = hist5.iloc[-1]       # Connect history to the forecast
        st.line_chart(chart5)

        flat5 = float(fut5["forecast"].max() - fut5["forecast"].min()) < 0.5
        if flat5:
            st.info(f"This model gives the same number for every month: about {fut5['forecast'].iloc[0]:.0f} titles a "
                    f"month (typical range {fut5['range_low'].iloc[0]:.0f} to {fut5['range_high'].iloc[0]:.0f}). "
                    "That is not a bug: it has no trend and no yearly pattern, so a constant level is what it produces.")
        table5 = fut5.round(0).astype(int)
        table5.index = table5.index.strftime("%b %Y")
        table5.columns = ["Forecast", "Typical error range: low", "Typical error range: high"]
        if show_ref5:
            table5[f"{nc.T5_DEFAULT_MODEL} (reference)"] = m5["default_future"].iloc[:horizon5].round(0).astype(int).values
        if flat5:
            with st.expander("Month-by-month table (every month is the same)"):
                st.dataframe(table5)
        else:
            st.dataframe(table5)
        st.caption(nc.T5_RANGE_NOTE + " The data is a snapshot of the catalog, so titles that were removed "
                   "are missing and older months may be undercounted.")
        st.caption(f"Check of the range: in the rolling tests {m5['range_coverage']:.0%} of the errors stayed inside it. "
                   "Typical miss by months ahead: "
                   + ", ".join(f"{k} {v:.0f}" for k, v in m5["error_by_horizon"].items()) + ".")

        with st.expander("Which records are in these counts?"):
            for note in info5["issues"]:
                st.write("- " + note)
            st.dataframe(nc.t5_accounting(info5), hide_index=True)
            st.caption("Counted for all titles. The Movie and TV Show series use the same months. The last month is "
                       "treated as incomplete when the last date is more than 2 days before the end of that month: "
                       "a rule of thumb, because the file does not say when it was extracted.")

        with st.expander("How was this tested?"):
            st.write(f"**1. Choosing the model.** Inside the training part only, each model forecast "
                     f"{m5['cv_h']} months ahead from {m5['cv_origins']} start date{'s' if m5['cv_origins'] > 1 else ''}. The plain 12-month average is kept unless another model beats its average miss (MAE) by at least 10%. "
                     "The first month of the history was also picked from the months before the final test:")
            st.dataframe(m5["cv"].round(1))
            st.caption(m5["selection_note"])
            st.write("Average miss at each start date (is a model's advantage consistent, or one lucky date?):")
            st.dataframe(m5["cv_by_origin"].round(1))
            st.write(f"Average miss by months ahead (each cell averages only {m5['cv_origins']} errors, "
                     "so it is a rough guide):")
            st.dataframe(m5["cv_by_horizon"].round(1))
            st.caption("The test windows overlap, so together they cover far fewer distinct months than the number "
                       "of forecasts suggests. They are not independent evidence.")
            st.write(f"**2. One final test.** Every model forecast the last {m5['holdout_months']} months "
                     "using only earlier data, then we compared with what really happened:")
            st.dataframe(hold5.round(1))
            rule5 = m5["best_simple_rule"]
            test5 = pd.DataFrame({"Actual": m5["holdout_actual"], sel5: m5["holdout_fc"][sel5],
                                  f"{rule5} (best simple rule)": m5["holdout_fc"][rule5]})
            st.line_chart(test5)
            st.caption("The test months are always the most recent ones, never a random sample, and nothing "
                       "about them was used to choose the model or the history.")
            show_images(["task5_history.png", "task5_model_comparison.png"], 800)

        with st.expander("Do Movies + TV Shows add up to All titles?"):
            st.write("Each series is forecast on its own and may pick a different model, so the three numbers can "
                     "disagree. Two standard fixes are compared on the training part only: bottom-up (All = Movies + "
                     "TV Shows) and top-down (split the All forecast by the recent Movies / TV Shows share).")
            if st.checkbox("Show forecasts that add up (also tests the Movies and TV Shows series: about a minute)",
                           key="t5_rec"):
                stamp5 = os.path.getmtime(data_path)
                try:
                    rec5 = nc.t5_reconcile(get_t5(data_path, None, stamp5), get_t5(data_path, "Movie", stamp5),
                                           get_t5(data_path, "TV Show", stamp5))
                except ValueError as err:
                    rec5 = {"ok": False, "note": str(err)}
                if not rec5["ok"]:
                    st.warning(rec5["note"])
                else:
                    st.write(rec5["note"])
                    coh5 = rec5["future"].round(0).astype(int)
                    coh5.index = coh5.index.strftime("%b %Y")
                    st.dataframe(coh5)
                    st.write("Rolling tests used to choose (relative miss = average miss as a share of the level):")
                    st.dataframe(rec5["rolling"].round(3))
                    st.caption("The final test is not used to choose between the two ways.")

# TSK 1
with tab1:
    rec = get_recommender(data_path, os.path.getmtime(data_path),
                          os.path.getmtime(nc.WEIGHTS_FILE) if os.path.exists(nc.WEIGHTS_FILE) else 0)
    df1 = rec.df

    for note in df1.attrs.get("issues", []):
        st.warning(note)

    st.subheader("Find similar titles")
    st.write("Pick one title, or several you liked, then press the button. "
             "Nothing is saved between visits: several titles are combined for this search only.")
    st.caption("This is a metadata-based content recommender. It compares genres, director, country, rating, type, "
               "decade and title words. The Netflix file has no plot description, cast or themes, so two titles with "
               "the same genres can still be about very different subjects. Plot embeddings would be a natural "
               "future step if such data were available.")

    who = st.radio("Suggest based on", ["One title", "Several titles I liked"],
                   horizontal=True, key="t1_mode")
    if who == "One title":
        idx = title_picker(df1, "Movie / TV show", "t1_title", show_type=True, default_title="cobra kai")
        picked = [] if idx is None else [idx]
    else:
        picked = title_picker(df1, "Pick 2 or more titles you liked", "t1_multi",
                              show_type=True, multi=True)

    method = "average"
    if who != "One title":
        way = st.radio("How to combine your titles",
                       ["Average them into one taste profile", "Suggest for each title in turn"],
                       horizontal=True, key="t1_way")
        method = "each" if way.startswith("Suggest") else "average"
        st.caption("Averaging is simple but can drift towards a generic middle when your picks are very different. "
                   "'In turn' keeps every pick represented and says which title each suggestion came from. "
                   "task1_weight_tuning.py compares both on simulated users.")

    same_type = st.checkbox("Only suggest the same type (movies for movies, shows for shows)", key="t1_same")

    y_min, y_max = int(df1["release_year"].min()), int(df1["release_year"].max())
    col_c, col_d = st.columns(2)
    n_sugg = col_c.slider("Number of suggestions", 5, 20, 10, key="t1_n")
    min_year = col_d.slider("Released in or after", y_min, y_max, y_min, key="t1_year")

    if st.button("Show suggestions", type="primary", key="t1_go"):
        if not picked:
            st.warning("Pick at least one title first.")
        else:
            if who == "One title":
                chosen = df1.loc[picked[0]]
                st.markdown(
                    f"**You picked:** {chosen['title']} · {chosen['type']} · {chosen['listed_in']} · "
                    f"{chosen['country'] or 'country unknown'} · {chosen['rating']} · {chosen['release_year']}"
                )
                res = rec.recommend(picked[0], n=n_sugg, same_type=same_type, min_year=min_year)
                heading = f"Because you picked *{chosen['title']}*, you might like:"
            else:
                names = ", ".join(df1.loc[picked, "title"])
                st.markdown(f"**You liked:** {names}")
                res = rec.recommend_many(picked, n=n_sugg, same_type=same_type, min_year=min_year, method=method)
                heading = ("Based on the average of everything you picked, you might like:" if method == "average"
                           else "Taking your picks in turn, you might like:")
            if res.empty:
                st.warning("No titles match these filters. Try loosening them.")
            else:
                st.markdown(f"### {heading}")
                show = res.rename(columns={
                    "similarity": "Similarity", "title": "Title", "type": "Type",
                    "listed_in": "Genres", "country": "Country", "rating": "Rating",
                    "release_year": "Year", "because_of": "Because you liked"})
                show.index = show.index + 1
                st.dataframe(
                    show,
                    column_config={"Similarity": st.column_config.ProgressColumn(
                        "Similarity", min_value=0.0, max_value=1.0, format="%.2f",
                        help="How alike the titles' details are, from 0 to 1. Not the chance you will enjoy it.")},
                )
                if len(res) < n_sugg:
                    st.caption(f"Only {len(res)} title(s) match these filters, so fewer than {n_sugg} are shown.")
                if who != "One title" and same_type and df1.loc[picked, "type"].nunique() > 1:
                    st.caption("Your picks mix movies and shows, so the 'same type' filter was not applied.")
                st.caption("Similarity shows how alike two titles' details are (genres, country, director, "
                           "rating, ...). It is not the chance that you will enjoy a title. "
                           "When several titles have the same similarity, the one released closest in year comes first.")

    with st.expander("How good is this recommender? (offline evaluation)"):
        has_csv = os.path.exists("task1_evaluation.csv")
        has_png = os.path.exists("task1_evaluation.png")
        if has_csv:
            st.dataframe(pd.read_csv("task1_evaluation.csv", index_col=0))
        if has_png:
            st.image("task1_evaluation.png", width=650)
        if not (has_csv or has_png):
            st.info("Run task1_recommendation_system.py once in this folder to generate the evaluation files.")
        for f in ["task1_weight_tuning.png"]:
            if os.path.exists(f):
                st.image(f, width=650)
        if os.path.exists("task1_genre_weight_sensitivity.csv"):
            st.write("How the genre weight changes the lists:")
            st.dataframe(pd.read_csv("task1_genre_weight_sensitivity.csv"), hide_index=True)
        if os.path.exists("task1_evaluation_ci.csv"):
            st.write("Each score with its 95% range and what it can show:")
            st.dataframe(pd.read_csv("task1_evaluation_ci.csv").round(3), hide_index=True)
        st.caption("How to read these numbers. Independent proxy: the 'no-genre' model never sees genres but is judged on "
                   "them, so its genre scores against random are the stronger automatic evidence. Metadata consistency "
                   "(hatched in the chart): the full model's genre scores and every 'same type' score only show that the "
                   "model uses the information it was given, because genres and type are inputs. They are kept for "
                   "completeness, not as evidence of quality.")
        st.write("**Human review** (the strongest evidence for Step 5):")
        if os.path.exists("task1_human_review_sheet.csv"):
            for line in nc.t1_review_text(nc.t1_review_summary(pd.read_csv("task1_human_review_sheet.csv"))):
                st.write("- " + line)
        else:
            st.info("No review sheet yet. Run task1_recommendation_system.py to create task1_human_review_sheet.csv.")
        if os.path.exists("task1_profile_test.csv"):
            st.write("Several liked titles, on simulated users (genres hidden from the vectors):")
            st.dataframe(pd.read_csv("task1_profile_test.csv").round(3), hide_index=True)
        if os.path.exists("task1_tuning_summary.txt"):
            with open("task1_tuning_summary.txt") as f:
                st.write(f.read())
        tuned = rec.weights != nc.DEFAULT_WEIGHTS
        st.caption("Column weights in use: " + ", ".join(f"{k} {v:g}" for k, v in rec.weights.items()) + ". "
                   + ("The non-genre weights were tuned and checked on unseen titles. " if tuned else
                      "These weights were chosen by hand (run task1_weight_tuning.py to test them). ")
                   + "The genre weight is always a judgment call.")

# TSK 2
with tab2:
    m = get_t2(data_path, os.path.getmtime(data_path),
               os.path.getmtime(nc.T2_SELECTION_FILE) if os.path.exists(nc.T2_SELECTION_FILE) else 0)
    df2 = m["df"]

    st.subheader("Is it a Movie or a TV Show?")
    st.write(f"A **{m['model_name']}** model ({m['model_source']}) predicts the type from a title's details "
             "(rating, country, release year, when it was added, director and genre counts). "
             "It deliberately does **not** use duration or the genre names, because those give the answer away. "
             "It only knows how many genres are listed.")
    if m["selection"].get("note"):
        st.caption("How the model was chosen: " + m["selection"]["note"] + " The test set was not used for the choice.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy on unseen titles", f"{m['accuracy']:.1%}")
    k2.metric("Balanced accuracy", f"{m['balanced_accuracy']:.1%}")
    k3.metric("Macro-F1 (both types equal)", f"{m['macro_f1']:.2f}")
    k4.metric("Baseline (always guess 'Movie')", f"{m['baseline']:.1%}")
    st.caption("Movies are about 70% of the titles, so accuracy alone flatters the majority class. Balanced accuracy is "
               "the average of the share of Movies and of TV Shows found.")

    mode = st.radio("What would you like to do?",
                    ["Check a real title", "Predict from details"],
                    horizontal=True, key="t2_mode")

    if mode == "Predict from details":
        st.write("Fill in the details of a title and the model will guess its type.")
        with st.form("t2_form"):
            f1, f2, f3 = st.columns(3)
            rating = f1.selectbox("Rating", m["ratings"], key="t2_rating")
            country = f2.selectbox("Main country", m["countries"], key="t2_country")
            n_countries = f3.number_input("Number of countries", 0, 10, 1, key="t2_ncountries")

            g1, g2, g3 = st.columns(3)
            release_year = g1.number_input("Release year", m["year_min"], m["year_max"],
                                           min(2020, m["year_max"]), key="t2_ry")
            year_added = g2.number_input("Year added to Netflix", 2000, 2030, 2020, key="t2_ya")
            month_added = g3.selectbox("Month added", list(range(1, 13)), index=5, key="t2_ma")

            h1, h2, h3 = st.columns(3)
            has_director = h1.checkbox("A director is listed", value=True, key="t2_hasdir")
            n_directors = h2.number_input("Number of directors", 1, 10, 1, key="t2_ndir")
            n_genres = h3.number_input("Number of genres listed", 1, 3, 2, key="t2_ngen")

            go = st.form_submit_button("Predict")
        if go:
            row = nc.t2_row(rating, country, release_year, year_added, month_added,
                            has_director, n_directors, n_countries, n_genres)
            p_tv = nc.t2_predict(m["model"], row)
            show_probs({"Movie": 1 - p_tv, "TV Show": p_tv})

    else:
        st.write("Pick a title from the dataset. The model predicts its type, "
                 "and we compare with the real answer.")
        i2 = title_picker(df2, "Movie / TV show", "t2_title")
        if st.button("Predict this title", type="primary", key="t2_go") and have(i2):
            row = df2.loc[[i2], nc.T2_CAT + nc.T2_NUM]
            p_tv = nc.t2_predict(m["model"], row)
            label = show_probs({"Movie": 1 - p_tv, "TV Show": p_tv})
            actual = df2.at[i2, "type"]
            if label == actual:
                st.success(f"Correct. The real type is {actual}.")
            else:
                st.error(f"Wrong this time. The real type is {actual}.")
            show_split_note(i2, m["test_idx"])

    with st.expander("Model details and charts"):
        st.write(f"Test set size: {m['n_test']} titles.")
        cm = pd.DataFrame(m["cm"], index=["Actually Movie", "Actually TV Show"],
                          columns=["Predicted Movie", "Predicted TV Show"])
        st.dataframe(cm)
        st.write("Both types (test set):")
        st.dataframe(m["report"].round(3))
        if os.path.exists("task2_model_comparison.csv"):
            comp2 = pd.read_csv("task2_model_comparison.csv")
            honest2 = comp2[comp2["experiment"].str.startswith(("B", "C"))]
            st.write("All models, experiments B (leak-free) and C (without date-added features). "
                     "cv_* = cross-validation on the training part, used to choose; the rest = test part:")
            st.dataframe(honest2[["experiment", "model", "selected", "cv_macro_f1", "cv_macro_f1_std", "accuracy",
                                  "balanced_accuracy", "macro_f1", "f1_TV"]].round(3), hide_index=True)
        if os.path.exists("task2_ablation.csv"):
            st.write("What the model leans on (cross-validation on the training part, one group of columns taken away):")
            st.dataframe(pd.read_csv("task2_ablation.csv").round(3), hide_index=True)
        show_images(["task2_accuracy_comparison.png", "task2_confusion_matrix.png"], 600)
        st.caption("With duration included, every model scores about 100% (data leakage); that experiment is a demonstration "
                   "only. Limitation: earlier versions of this project chose the model by looking at this same test part, "
                   "so the test numbers are slightly optimistic. The choice is now made without it.")

    with st.expander("Extra: browse titles by genre"):
        picked = st.multiselect("Genres", nc.get_genres(df2), key="t2_genres")
        if picked:
            wanted = {g.lower() for g in picked}
            has_all = lambda s: wanted <= {g.lower() for g in nc.split_items(s)}
            hits = df2[df2["listed_in"].fillna("").apply(has_all)]
            st.write(f"{len(hits)} titles found.")
            st.dataframe(hits[["title", "type", "listed_in", "country", "rating", "release_year"]].head(200))
        else:
            st.caption("Choose one or more genres to list matching titles.")

# TSK 3
with tab3:
    t3_params_file = "task3_best_params.json"
    m3 = get_t3(data_path, os.path.getmtime(data_path),
                os.path.getmtime(t3_params_file) if os.path.exists(t3_params_file) else 0)
    df3, safety3 = m3["df"], m3["safety"]

    st.subheader("Who is this title for?")
    st.write(f"A **{m3['model_name']}** model predicts the audience group from a title's type, genres, "
             "country, dates, duration and director info. The rating itself is never used as an input. "
             "The ratings are grouped into 4 audience categories:")
    st.dataframe(pd.DataFrame({"Audience group": nc.GROUP_ORDER,
                               "Ratings included": [nc.GROUP_INFO[g] for g in nc.GROUP_ORDER]}),
                 hide_index=True)
    st.caption(nc.T3_TARGET_NOTE)
    st.warning(nc.T3_DISCLAIMER)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Accuracy on unseen titles", f"{m3['accuracy']:.1%}")
    q2.metric("Macro-F1 (fair across groups)", f"{m3['macro_f1']:.2f}")
    q3.metric("Baseline (always the biggest group)", f"{m3['baseline']:.1%}")
    q4.metric("Mature titles shown as Kids/Family", f"{safety3['severe_lenient_rate']:.1%}")
    st.caption("Last number: Teens and Adults titles that the model predicted as Kids or Family (test set). "
               "Lower is safer.")
    if m3["params_source"] == "tuned":
        st.caption("Using the tuned settings found by task3_rating_classification.py.")
    else:
        st.caption("Using default settings. Run task3_rating_classification.py in this folder "
                   "to tune the model and generate the charts.")

    th3 = st.slider("'Review required' when the top model score is below", 0.30, 0.90,
                    float(nc.T3_REVIEW_THRESHOLD), 0.05, key="t3_review")
    st.caption("Below this score the answer is shown as 'Review required': a person should check the title. "
               "The scores are not calibrated probabilities. task3_review_curve.csv shows what each threshold costs and buys.")

    def show_t3(probs):
        best = show_probs(probs, nc.GROUP_INFO)
        label, top = nc.t3_review_label(probs, th3)
        if label == nc.T3_REVIEW:
            st.warning(f"{nc.T3_REVIEW}: the top score ({top:.0%}) is below {th3:.0%}, so the model is not sure.")
        return best

    mode3 = st.radio("What would you like to do?",
                     ["Check a real title", "Predict from details"],
                     horizontal=True, key="t3_mode")

    if mode3 == "Predict from details":
        st.write("Describe a title and the model will guess its audience group.")
        with st.form("t3_form"):
            a1, a2, a3 = st.columns(3)
            type3 = a1.selectbox("Type", ["Movie", "TV Show"], key="t3_type")
            country3 = a2.selectbox("Main country", m3["countries"], key="t3_country")
            n_countries3 = a3.number_input("Number of countries", 0, 10, 1, key="t3_ncountries")

            genres3 = st.multiselect("Genres (pick one or more)", m3["genres"], key="t3_genres")

            b1, b2, b3 = st.columns(3)
            ry3 = b1.number_input("Release year", m3["year_min"], m3["year_max"],
                                  min(2020, m3["year_max"]), key="t3_ry")
            ya3 = b2.number_input("Year added to Netflix", 2000, 2030, 2020, key="t3_ya")
            ma3 = b3.selectbox("Month added", list(range(1, 13)), index=5, key="t3_ma")

            c1, c2, c3 = st.columns(3)
            minutes3 = c1.number_input("Length in minutes (movies)", 1, 400, 100, key="t3_min")
            seasons3 = c2.number_input("Number of seasons (TV shows)", 1, 40, 1, key="t3_seasons")
            dir3 = c3.checkbox("A director is listed", value=True, key="t3_hasdir")
            ndir3 = c3.number_input("Number of directors", 1, 10, 1, key="t3_ndir")

            go3 = st.form_submit_button("Predict audience")
        if go3:
            if not genres3:
                st.warning("Pick at least one genre. Genres are the strongest clue for the model.")
            else:
                row = nc.t3_row(type3, country3, genres3, ry3, ya3, ma3,
                                dir3, ndir3, n_countries3, minutes3, seasons3)
                show_t3(nc.t3_predict(m3["model"], row))
    else:
        st.write("Pick a title from the dataset. The model predicts its audience group, "
                 "and we compare it with the real rating.")
        i3 = title_picker(df3, "Movie / TV show", "t3_title")
        if st.button("Predict this title", type="primary", key="t3_go") and have(i3):
            row = df3.loc[[i3], nc.T3_FEATURES]
            best = show_t3(nc.t3_predict(m3["model"], row))
            actual = df3.at[i3, "rating_group"]
            raw = df3.at[i3, "rating"]
            if best == actual:
                st.success(f"Correct. The real rating is {raw}, which is in the {actual} group.")
            else:
                st.error(f"Wrong this time. The real rating is {raw}, which is in the {actual} group.")
            show_split_note(i3, m3["test_idx"])

    with st.expander("Model details and charts"):
        st.write(f"Test set size: {m3['n_test']} titles. "
                 f"{m3['n_dropped']} titles with no usable rating (e.g. NR, UR) were left out.")
        st.write("Results per audience group (test set):")
        st.dataframe(m3["per_class"].round(2))
        cm3 = pd.DataFrame(m3["cm"], index=[f"Actually {g}" for g in m3["labels"]],
                           columns=[f"Predicted {g}" for g in m3["labels"]])
        st.write("Confusion matrix:")
        st.dataframe(cm3)
        st.write("Where each group's titles go (share of the true group, test set):")
        st.dataframe(m3["error_table"].round(2))
        st.write(f"The errors that matter most: {safety3['severe_lenient']} of {safety3['older_titles']} Teens/Adults titles "
                 f"({safety3['severe_lenient_rate']:.1%}) were predicted as Kids or Family; "
                 f"{safety3['adults_as_kids']} Adults titles as Kids. Average cost per title: {safety3['mean_cost']:.2f} "
                 f"(a step too lenient costs {nc.T3_UNDER_COST}, a step too strict {nc.T3_OVER_COST}).")
        st.dataframe(m3["cost_matrix"])
        st.caption("Cost matrix: rows = true group, columns = predicted group.")
        if os.path.exists("task3_model_comparison.csv"):
            comp3 = pd.read_csv("task3_model_comparison.csv")
            st.write("Default vs tuned models (cv_macro_f1 = cross-validation on the training set):")
            st.dataframe(comp3[["model", "version", "accuracy", "macro_f1", "cv_macro_f1"]])
            tuning_lines3, tuning_summary3 = nc.t3_tuning_text(comp3)
            for line in tuning_lines3:
                st.write("- " + line)
            st.caption(tuning_summary3)
        if os.path.exists("task3_model_significance.csv"):
            sig3 = pd.read_csv("task3_model_significance.csv")
            level3, sentences3 = nc.t3_significance_text(sig3)
            st.write(f"Are the gaps between the tuned models real? Each range is the adjusted {level3:.1%} one "
                     f"({len(sig3)} comparisons are made at once); if it includes 0 there is no clear difference.")
            for line in sentences3:
                st.write("- " + line)
            st.dataframe(sig3, hide_index=True)
        for fname, title in [("task3_review_curve.csv", "What 'Review required' costs and buys (training folds):"),
                             ("task3_repeated_cv.csv", "Default vs tuned, repeated cross-validation (task3_robustness.py):"),
                             ("task3_alternatives.csv", "Alternative designs for Family and Teens (task3_robustness.py):"),
                             ("task3_ablation.csv", "Ablation: taking feature groups away (task3_robustness.py):")]:
            if os.path.exists(fname):
                st.write(title)
                st.dataframe(pd.read_csv(fname).round(3), hide_index=True)
        if os.path.exists("task3_raw_vs_grouped.csv"):
            st.write("Does grouping the ratings help? (raw ratings vs the 4 groups)")
            st.dataframe(pd.read_csv("task3_raw_vs_grouped.csv"), hide_index=True)
        show_images(["task3_class_distribution.png", "task3_model_comparison.png",
                     "task3_confusion_matrix.png", "task3_feature_importance.png",
                     "task3_permutation_importance.png"], 700)
        st.caption("The first importance chart is impurity-based, which can favour columns with many values. The "
                   "permutation chart shuffles one column of the TEST set at a time and shows how much the score "
                   "drops, which is the fairer reading. Genres such as Kids' TV are legitimate content labels, not the "
                   "rating itself, but the model leans on them (see the ablation).")
