import sys
import numpy as np
import pandas as pd

GENRES = {
    "Movie": ["Dramas", "Comedies", "Documentaries", "Action & Adventure",
              "Children & Family Movies", "Horror Movies", "International Movies"],
    "TV Show": ["TV Dramas", "TV Comedies", "Kids' TV", "Crime TV Shows",
                "Docuseries", "International TV Shows"],
}
RATING_BY_GENRE = {
    "Children & Family Movies": ["G", "PG", "TV-G"], "Kids' TV": ["TV-Y", "TV-Y7", "TV-G"],
    "Horror Movies": ["R", "TV-MA"], "Crime TV Shows": ["TV-MA", "TV-14"],
}
COUNTRIES = ["United States", "India", "United Kingdom", "Japan", "France", "Canada"]
FALLBACK_RATINGS = ["TV-14", "TV-MA", "PG-13", "R", "TV-PG", "PG"]
FIRST = ["Dark", "Last", "Silent", "Golden", "Wild", "Secret", "Broken", "Hidden", "Lost", "Blue"]
SECOND = ["River", "Kingdom", "Night", "Summer", "Road", "Garden", "Empire", "City", "Story", "Game"]

def make_demo(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = "Movie" if rng.random() < 0.7 else "TV Show"
        genres = list(rng.choice(GENRES[kind], size=rng.integers(1, 4), replace=False))
        rating = rng.choice(RATING_BY_GENRE.get(genres[0], FALLBACK_RATINGS))
        year = int(rng.integers(1995, 2021))
        added = pd.Timestamp(year=min(2021, year + int(rng.integers(0, 6))),
                             month=int(rng.integers(1, 13)), day=int(rng.integers(1, 28)))
        n_countries = int(rng.integers(1, 3))
        rows.append({
            "show_id": f"s{i + 1}", "type": kind,
            "title": f"{rng.choice(FIRST)} {rng.choice(SECOND)} {i}",
            "director": "Not Given" if rng.random() < 0.3 else f"Director {rng.integers(0, 80)}",
            "country": "Not Given" if rng.random() < 0.1
                       else ", ".join(rng.choice(COUNTRIES, n_countries, replace=False)),
            "date_added": f"{added.month}/{added.day}/{added.year}",
            "release_year": year, "rating": rating,
            "duration": f"{int(rng.integers(60, 180))} min" if kind == "Movie"
                        else f"{int(rng.integers(1, 6))} Seasons",
            "listed_in": ", ".join(genres),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_dataset.csv"
    make_demo().to_csv(out, index=False)
    print("Saved", out)



