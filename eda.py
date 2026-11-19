"""Quick exploratory plots for Tasks 1, 2, 3 and 5.

Run: py eda.py Dataset.csv
Saves one eda_task*.png figure per task when the data is available."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import netflix_core as nc

path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.csv"
t = nc.eda_tables(path)
print(f"{t['n_rows']} titles after removing duplicates\n")
print("Missing values per column (blank or 'Not Given'):")
print(t["missing"].to_string())

# Task 1: genres and missing values
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
t["top_genres"].sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("15 most common genres")
t["missing"].plot(kind="bar", ax=axes[1], color="#C44E52")
axes[1].set_title("Missing values per column"); axes[1].tick_params(axis="x", rotation=45)
plt.suptitle("Task 1 - what the recommender works with"); plt.tight_layout()
plt.savefig("eda_task1.png", dpi=150); plt.close()

# Task 2: content types and additions by year
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
t["types"].plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Movie vs TV Show (the target)"); axes[0].tick_params(axis="x", rotation=0)
t["added_by_type"].plot(kind="bar", stacked=True, ax=axes[1], width=0.9)
axes[1].set_title("Titles added to Netflix per year")
plt.suptitle("Task 2 - the target and the date features"); plt.tight_layout()
plt.savefig("eda_task2.png", dpi=150); plt.close()

# Task 3: ratings and movie duration
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
t["ratings"].plot(kind="bar", ax=axes[0], color="#4C72B0"); axes[0].set_title("Raw ratings")
t["rating_groups"].plot(kind="bar", ax=axes[1], color="#DD8452"); axes[1].set_title("Audience groups (the target)")
m = t["movie_minutes"]
groups = [g for g in nc.GROUP_ORDER if g in set(m["rating_group"].dropna())]
axes[2].boxplot([m.loc[m["rating_group"] == g, "movie_minutes"] for g in groups])
axes[2].set_xticklabels(groups)
axes[2].set_title("Movie length by audience group (minutes)")
for ax in axes[:2]:
    ax.tick_params(axis="x", rotation=45)
plt.suptitle("Task 3 - the target and one useful feature"); plt.tight_layout()
plt.savefig("eda_task3.png", dpi=150); plt.close()
# Task 5: monthly additions
if t["monthly_added"] is not None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t["monthly_added"].plot(ax=ax, color="#9bb7d4", label="titles added per month")
    t["monthly_added"].rolling(12).mean().plot(ax=ax, color="#E50914", linewidth=2, label="12-month average")
    ax.set_title("Task 5 - titles added to Netflix per month (the series the forecast works on)")
    ax.set_xlabel(""); ax.legend()
    plt.tight_layout(); plt.savefig("eda_task5.png", dpi=150); plt.close()
    print("\nTask 5 - where every record went:")
    print(nc.t5_accounting(t["t5_info"]).to_string(index=False))
    print("\nSaved: eda_task1.png, eda_task2.png, eda_task3.png, eda_task5.png")
else:
    print(f"\nTask 5 monthly counts not made: {t['t5_note']}")
    print("Saved: eda_task1.png, eda_task2.png, eda_task3.png")
