#!/usr/bin/env python3
"""
Evaluate different weight configurations for user-to-city matching.

Loads pre-computed user and city feature vectors, runs the scoring algorithm
with multiple weight configs, and prints a formatted comparison table.

Usage:
    python3 evaluate_weights.py
    python3 evaluate_weights.py --top 10
"""

import argparse
import json
from collections import Counter

from match_user_to_cities import (
    compute_audio_stats,
    audio_similarity,
    cosine_similarity,
    top_shared,
    DATA_DIR,
    USER_FILE,
    CITIES_FILE,
)

WEIGHT_CONFIGS = [
    ("Balanced",     0.30, 0.40, 0.30),
    ("Audio-heavy",  0.50, 0.30, 0.20),
    ("Genre-heavy",  0.20, 0.60, 0.20),
    ("Artist-heavy", 0.20, 0.20, 0.60),
    ("Audio+Genre",  0.40, 0.40, 0.20),
    ("Discovery",    0.10, 0.50, 0.40),
]


def score_cities(user, cities, stats, aw, gw, rw):
    user_gv = user["genre_vector"]
    user_av = user.get("artist_vector", {})
    results = []
    for city in cities:
        a_sim = audio_similarity(user["audio_features"], city["audio_features"], stats)
        g_sim = cosine_similarity(user_gv, city["genre_vector"])
        r_sim = cosine_similarity(user_av, city.get("artist_vector", {}))
        score = aw * a_sim + gw * g_sim + rw * r_sim
        results.append({
            "city": city["city"],
            "country": city["country"],
            "score": score,
            "audio": a_sim,
            "genre": g_sim,
            "artist": r_sim,
            "shared_genres": top_shared(user_gv, city["genre_vector"], n=3),
            "shared_artists": top_shared(user_av, city.get("artist_vector", {}), n=2),
        })
    results.sort(key=lambda x: -x["score"])
    return results


def dominant_signal(result, aw, gw, rw):
    contributions = [
        ("audio",  aw * result["audio"]),
        ("genre",  gw * result["genre"]),
        ("artist", rw * result["artist"]),
    ]
    contributions.sort(key=lambda x: -x[1])
    name, val = contributions[0]
    return f"{name} ({val:.3f}/{result['score']:.3f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=5, help="Cities per config")
    args = parser.parse_args()

    user = json.loads(USER_FILE.read_text())
    cities = json.loads(CITIES_FILE.read_text())
    stats = compute_audio_stats(cities)

    print(f"User: {user['user']}  |  Scrobbles: {user['total_tracks']}  |  "
          f"Unique tracks: {user['unique_tracks']}  |  "
          f"Genres: {len(user['genre_vector'])}  |  Artists: {len(user.get('artist_vector', {}))}")
    print(f"Cities: {len(cities)}")
    print()

    # --- Per-config tables ---
    all_results = {}
    for label, aw, gw, rw in WEIGHT_CONFIGS:
        ranked = score_cities(user, cities, stats, aw, gw, rw)
        all_results[label] = (ranked, aw, gw, rw)

        print(f"{'=' * 95}")
        print(f"  {label}  (audio={aw:.2f}, genre={gw:.2f}, artist={rw:.2f})")
        print(f"{'=' * 95}")
        print(f" {'#':>2}  {'City':<22} {'CC':>2}  {'Score':>6}  {'Audio':>6}  "
              f"{'Genre':>6}  {'Artist':>6}  {'Shared Genres'}")
        print(f" {'--':>2}  {'----':<22} {'--':>2}  {'-----':>6}  {'-----':>6}  "
              f"{'-----':>6}  {'------':>6}  {'-------------'}")
        for i, r in enumerate(ranked[:args.top], 1):
            genres_str = ", ".join(r["shared_genres"][:3])
            artists_str = ", ".join(r["shared_artists"][:2])
            shared = genres_str
            if artists_str:
                shared += f"  | {artists_str}"
            print(f" {i:>2}  {r['city']:<22} {r['country']:>2}  {r['score']:.4f}  "
                  f"{r['audio']:.4f}  {r['genre']:.4f}  {r['artist']:.4f}  {shared}")
        print()

    # --- Cross-config comparison ---
    print(f"{'=' * 95}")
    print(f"  COMPARISON: Top City per Weight Config")
    print(f"{'=' * 95}")
    print(f" {'Config':<15} {'#1 City':<22} {'CC':>2}  {'Score':>6}  {'Dominant Signal'}")
    print(f" {'-' * 14:<15} {'-' * 21:<22} {'--':>2}  {'-----':>6}  {'---------------'}")
    for label, aw, gw, rw in WEIGHT_CONFIGS:
        ranked, _, _, _ = all_results[label]
        top = ranked[0]
        sig = dominant_signal(top, aw, gw, rw)
        print(f" {label:<15} {top['city']:<22} {top['country']:>2}  {top['score']:.4f}  {sig}")
    print()

    # --- Stability analysis ---
    print(f"{'=' * 95}")
    print(f"  STABILITY: Cities appearing in top {args.top} across configs")
    print(f"{'=' * 95}")
    city_appearances: Counter = Counter()
    for label, (ranked, _, _, _) in all_results.items():
        for r in ranked[:args.top]:
            city_appearances[f"{r['city']} ({r['country']})"] += 1

    stable = [(city, count) for city, count in city_appearances.most_common()
              if count >= 2]
    if stable:
        print(f" {'City':<30} {'Configs':>7}")
        print(f" {'-' * 29:<30} {'-------':>7}")
        for city, count in stable:
            ratio = f"{count}/{len(WEIGHT_CONFIGS)}"
            print(f" {city:<30} {ratio:>7}")
    else:
        print(" No cities appear in top results across multiple configs.")
    print()


if __name__ == "__main__":
    main()
