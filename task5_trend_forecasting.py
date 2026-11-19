"""netflix-ml-tasks TSK5"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netflix_core as nc

warnings.filterwarnings("ignore")
SERIES = [("All titles", None), ("Movies", "Movie"), ("TV Shows", "TV Show")]
def slug(name):
    return name.lower().replace(" ", "_")
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.csv"
    start = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        all_s, info = nc.t5_series(path, start=start)
        series = {name: nc.t5_series(path, kind, start=info["start"])[0] for name, kind in SERIES}
    except ValueError as err:
        sys.exit(f"Cannot continue: {err}")
    print(nc.T5_TARGET_NOTE)
    print(f"\nRaw dates run from {info['first_date']:%Y-%m-%d} to {info['last_date']:%Y-%m-%d}")
    for note in info["issues"]:
        print("  NOTE:", note)
    print(f"\nSeries used: {info['start']:%Y-%m} to {info['end']:%Y-%m}  ({info['n_months']} months, "
          f"{info['n_titles']:,} titles)")
    print("The first month was chosen using only the months before the final test"
          + (" (you set it yourself)." if start else "."))
    print("The last month is treated as incomplete when the last date is more than 2 days before the end of that "
          "month. This is a rule of thumb: the file does not say when it was extracted.")
    if info["n_months"] < 60:
        print("Note: under 5 years of months. Forecasts will be rough; say so in your report.")
    print("\n===== Step 1: where every record went =====")
    print(nc.t5_accounting(info).to_string(index=False))
    print("\nTime-based features used by the Random Forest: last 1, 2, 3 and 12 months, 3- and 12-month "
          "averages, month of the year (as sine/cosine).")
    movies, shows = series["Movies"], series["TV Shows"]
    yearly = pd.DataFrame({"Movies": movies.groupby(movies.index.year).sum(),
                           "TV Shows": shows.groupby(shows.index.year).sum()})
    yearly["Total"] = all_s.groupby(all_s.index.year).sum()
    yearly["Months"] = all_s.groupby(all_s.index.year).size()
    growth, full_year = yearly["Total"].pct_change(), yearly["Months"] == 12
    yearly["Change vs previous year"] = [      # compare complete years only
        "" if (pd.isna(g) or not (full_year.iloc[i] and full_year.iloc[i - 1])) else f"{g:+.0%}"
        for i, g in enumerate(growth)]
    yearly["TV share"] = (yearly["TV Shows"] / yearly["Total"]).map("{:.0%}".format)
    print("\n===== Step 2: titles added per year =====")
    print(yearly.astype({"Movies": int, "TV Shows": int, "Total": int}).to_string())
    print("(years with fewer than 12 months are incomplete, so no year-on-year change is shown for them;"
          "\n Movies + TV Shows can be lower than Total if some titles have another type)")
    by_month = all_s.groupby(all_s.index.month).mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    top, low = by_month.idxmax(), by_month.idxmin()
    print(f"\nAverage per calendar month: busiest {month_names[top - 1]} ({by_month[top]:.0f}), "
          f"quietest {month_names[low - 1]} ({by_month[low]:.0f}).")
    spread = (by_month.max() - by_month.min()) / by_month.mean()
    print(f"The gap between busiest and quietest month is {spread:.0%} of the average, "
          "so seasonality is " + ("noticeable." if spread > 0.3 else "weak."))
    recent, before = all_s.iloc[-12:].mean(), all_s.iloc[-24:-12].mean()
    print(f"Average per month: last 12 months {recent:.0f} vs the 12 months before {before:.0f} "
          f"({recent / before - 1:+.0%}). (Descriptive only: no model choice uses these numbers.)")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    all_s.plot(ax=axes[0], color="#9bb7d4", label="monthly")
    all_s.rolling(12).mean().plot(ax=axes[0], color="#E50914", linewidth=2, label="12-month average")
    axes[0].set_title("Titles added per month"); axes[0].legend(); axes[0].set_xlabel("")
    axes[1].bar(month_names, by_month.values, color="#4C72B0")
    axes[1].set_title("Average per calendar month"); axes[1].tick_params(axis="x", rotation=45)
    yearly[["Movies", "TV Shows"]].plot(kind="bar", stacked=True, ax=axes[2], color=["#4C72B0", "#DD8452"])
    axes[2].set_xticklabels(nc.t5_year_labels(all_s), rotation=0)
    axes[2].set_title("Titles added per year (incomplete years show their months)"); axes[2].set_xlabel("")
    plt.tight_layout(); plt.savefig("task5_history.png", dpi=150); plt.close()
    comparison_rows, forecast_rows, detail_rows, studies = [], [], [], {}
    for name, _ in SERIES:
        print(f"\n\n################ {name} ################")
        try:
            res = nc.t5_study(series[name])
        except ValueError as err:
            sys.exit(f"Cannot continue: {err}")
        studies[name] = res
        n_orig = res["cv_origins"]
        print(f"\nStep 3/5a - rolling tests inside the training part ({res['cv_h']} months ahead, "
              f"{n_orig} start date{'s' if n_orig > 1 else ''}; MAE = average miss in titles):")
        print(res["cv"].round(1).to_string())
        print(f"-> model chosen BEFORE looking at the final test: {res['selected']}")
        print("   " + res["selection_note"])
        print("   (The rolling windows overlap, so together they cover far fewer distinct months than "
              f"{n_orig * res['cv_h']} forecasts suggest: they are not independent evidence.)")
        print("\n   Average miss at each start date (is the advantage consistent?):")
        print(res["cv_by_origin"].round(1).to_string())
        print("\n   Average miss by months ahead (each cell averages only "
              f"{n_orig} errors, so treat it as a rough guide):")
        print(res["cv_by_horizon"].round(1).to_string())

        print(f"\nStep 5b - final test on the last {res['holdout_months']} months "
              "(forecast made from earlier data only):")
        print(res["holdout"].round(1).to_string())
        if res["best_on_final_test"] != res["selected"]:
            print(f"(On this one test {res['best_on_final_test']} scored best, but the model is not switched "
                  "after seeing the test: that would make the test result too optimistic.)")
        print("\n" + nc.t5_verdict(res))

        fut = res["future"]
        if name == "All titles":
            print("\n" + nc.t5_snapshot_note(info, fut.index))
        print(f"\nStep 4 - forecast of {name.lower()} added per month (range = forecast +/- typical miss):")
        show = fut.round(0).astype(int)
        show.index = show.index.strftime("%Y-%m")
        print(show.to_string())
        if fut["forecast"].max() - fut["forecast"].min() < 0.5:
            print("(The forecast is the same every month: this model has no trend and no yearly pattern, so a "
                  "constant level is what it produces. That is not a bug.)")
        if res["selected"] != nc.T5_DEFAULT_MODEL:
            print(f"For comparison, the {nc.T5_DEFAULT_MODEL} would forecast {res['default_future'].iloc[0]:.0f} "
                  f"a month ({res['default_future'].iloc[0] - fut['forecast'].iloc[0]:+.0f} vs the chosen model).")
        print(f"\nHow good is the shaded range? In the rolling tests (other months than the ones its width came "
              f"from) {res['range_coverage']:.0%} of the errors stayed inside it. Typical miss by months ahead:")
        print(res["error_by_horizon"].round(1).to_string())
        print("(A range that grows with the months ahead would show up above; a flat one, as here, supports "
              "a constant width. The range stays illustrative.)")
        for model in res["holdout"].index:
            comparison_rows.append({
                "series": name, "model": model,
                "rolling_MAE": res["cv"].loc[model, "MAE"],
                "final_test_MAE": res["holdout"].loc[model, "MAE"],
                "final_test_RMSE": res["holdout"].loc[model, "RMSE"],
                "chosen": model == res["selected"]})
        out = fut.reset_index().rename(columns={"index": "month"})
        out.insert(0, "series", name)
        out.insert(2, "model", res["selected"])
        forecast_rows.append(out)
        d = res["cv_by_origin"].reset_index()
        d.insert(0, "series", name)
        detail_rows.append(d)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        s, hold_fc = res["series"], res["holdout_fc"]
        tail = s.iloc[-(res["holdout_months"] + 24):]
        axes[0].plot(tail.index, tail.values, color="#555", label="actual")
        axes[0].plot(hold_fc.index, hold_fc[res["selected"]], color="#E50914", linewidth=2, label=res["selected"])
        ref = res["best_simple_rule"]
        if ref != res["selected"]:
            axes[0].plot(hold_fc.index, hold_fc[ref], color="#4C72B0", linestyle="--", label=f"{ref} (best simple rule)")
        axes[0].axvline(hold_fc.index[0], color="grey", linestyle=":")
        axes[0].set_title(f"{name}: forecast vs what really happened"); axes[0].legend(fontsize=8)
        hist = s.iloc[-36:]
        axes[1].plot(hist.index, hist.values, color="#555", label="actual")
        axes[1].plot(fut.index, fut["forecast"], color="#E50914", linewidth=2, label="forecast")
        axes[1].fill_between(fut.index, fut["range_low"], fut["range_high"], color="#E50914", alpha=0.18,
                             label="typical error range (illustrative)")
        if res["selected"] != nc.T5_DEFAULT_MODEL:
            axes[1].plot(fut.index, res["default_future"], color="#4C72B0", linestyle="--",
                         label=f"{nc.T5_DEFAULT_MODEL} (stable reference)")
        axes[1].set_title(f"{name}: next 12 months ({res['selected']})\nforecast made from data up to "
                          f"{info['end']:%b %Y} (historical snapshot)")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); plt.savefig(f"task5_forecast_{slug(name)}.png", dpi=150); plt.close()

    comp = pd.DataFrame(comparison_rows)
    comp.round(3).to_csv("task5_model_comparison.csv", index=False)
    pd.concat(forecast_rows).round(1).to_csv("task5_forecast.csv", index=False)
    pd.concat(detail_rows).round(2).to_csv("task5_rolling_by_start_date.csv", index=False)
    nc.t5_accounting(info).to_csv("task5_data_accounting.csv", index=False)
    print("\n\n################ Do the forecasts add up? ################")
    first = {n: studies[n]["future"]["forecast"].iloc[0] for n, _ in SERIES}
    print(f"As forecast separately: Movies {first['Movies']:.1f} + TV Shows {first['TV Shows']:.1f} = "
          f"{first['Movies'] + first['TV Shows']:.1f}, but All titles {first['All titles']:.1f} "
          f"(gap {first['All titles'] - first['Movies'] - first['TV Shows']:+.1f} titles a month).")
    print("The gap appears when the three series pick different models; it is not an execution error.")
    rec = nc.t5_reconcile(studies["All titles"], studies["Movies"], studies["TV Shows"])
    if not rec["ok"]:
        print(rec["note"])
    else:
        print("\nRolling tests (used to choose; relative miss = average miss as a share of the level):")
        print(rec["rolling"].round(3).to_string())
        print("\nFinal test (information only, never used to choose):")
        print(rec["final"].round(3).to_string())
        print("\n" + rec["note"])
        coh = rec["future"]
        show = coh.round(1)
        show.index = show.index.strftime("%Y-%m")
        print("\nForecast where Movies + TV Shows = All titles:")
        print(show.to_string())
        coh.round(2).reset_index().rename(columns={"index": "month"}).to_csv("task5_forecast_coherent.csv", index=False)
        rec["rolling"].round(4).to_csv("task5_reconciliation.csv")

    print("\n\n################ Does the first month used matter? (all titles; takes about a minute) ################")
    sens = nc.t5_start_sensitivity(path)
    print(sens.round(1).to_string(index=False))
    print("Reading it: if the chosen model and the level barely move, the result does not depend on this choice. "
          "An earlier start includes months with only a handful of titles; a later start gives the models less "
          "history. The final test is always the same last 12 months.")
    sens.round(2).to_csv("task5_start_sensitivity.csv", index=False)
    piv = comp.pivot(index="model", columns="series", values="final_test_MAE")[[n for n, _ in SERIES]]
    ax = piv.plot(kind="bar", figsize=(11, 4.8))
    ax.set_ylabel("Average miss on the last 12 months (titles)")
    ax.set_title("Task 5 - forecast error by model (lower is better)")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig("task5_model_comparison.png", dpi=150); plt.close()
    print("\n\nSaved: task5_history.png, task5_forecast_all_titles.png, task5_forecast_movies.png,\n"
          "       task5_forecast_tv_shows.png, task5_model_comparison.png/.csv, task5_forecast.csv,\n"
          "       task5_forecast_coherent.csv, task5_reconciliation.csv, task5_rolling_by_start_date.csv,\n"
          "       task5_start_sensitivity.csv, task5_data_accounting.csv")
    print("\nReport tips: (1) " + nc.T5_TARGET_NOTE + " (2) " + nc.t5_snapshot_note(info, studies["All titles"]["future"].index)
          + " (3) the data is a snapshot of the catalogue, so removed titles are missing and older months may be "
          "undercounted; (4) " + nc.T5_RANGE_NOTE + " (5) if no model clearly beats the simple rules, say exactly that.")

if __name__ == "__main__":
    main()
