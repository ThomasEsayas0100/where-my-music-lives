#!/usr/bin/env python3
"""
Match a user's music profile to cities by combining three similarity signals:

  Audio:  z-scored Euclidean distance (9 continuous features, excluding key/mode)
  Genre:  cosine similarity on sparse genre vectors
  Artist: cosine similarity on sparse artist vectors

Combined: weighted sum (default 30% audio, 40% genre, 30% artist)

Usage:
    python3 match_user_to_cities.py
    python3 match_user_to_cities.py --audio-weight 0.3 --genre-weight 0.4 --artist-weight 0.3
    python3 match_user_to_cities.py --top 10
"""

import argparse
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
USER_FILE = DATA_DIR / "user_features.json"
CITIES_FILE = DATA_DIR / "city_features.json"
OUTPUT_FILE = DATA_DIR / "user_city_matches.json"

# Audio features used for distance — exclude key (categorical) and mode (binary)
AUDIO_KEYS = (
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
)


# ---------------------------------------------------------------------------
# Audio feature similarity
# ---------------------------------------------------------------------------

def compute_audio_stats(cities: list[dict]) -> dict[str, tuple[float, float]]:
    """Compute mean and std for each audio feature across all cities."""
    stats: dict[str, tuple[float, float]] = {}
    for key in AUDIO_KEYS:
        vals = [
            c["audio_features"][key]
            for c in cities
            if c["audio_features"].get(key) is not None
        ]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 1.0
        stats[key] = (mean, std)
    return stats


def zscore_vector(features: dict, stats: dict[str, tuple[float, float]]) -> list[float]:
    """Z-score a feature dict using precomputed stats. Returns list in AUDIO_KEYS order."""
    vec = []
    for key in AUDIO_KEYS:
        val = features.get(key)
        if val is None:
            vec.append(0.0)  # missing → neutral
        else:
            mean, std = stats[key]
            vec.append((val - mean) / std)
    return vec


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def audio_similarity(user_features: dict, city_features: dict,
                     stats: dict[str, tuple[float, float]]) -> float:
    """Compute audio similarity as 1 / (1 + euclidean_distance) on z-scored features."""
    u = zscore_vector(user_features, stats)
    c = zscore_vector(city_features, stats)
    dist = euclidean_distance(u, c)
    return 1.0 / (1.0 + dist)


# ---------------------------------------------------------------------------
# Cosine similarity (shared by genre and artist vectors)
# ---------------------------------------------------------------------------

def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    shared_keys = set(a.keys()) & set(b.keys())
    if not shared_keys:
        return 0.0

    dot = sum(a[k] * b[k] for k in shared_keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def top_shared(user_vec: dict[str, float], city_vec: dict[str, float],
               n: int = 5) -> list[str]:
    """Return the top N shared keys ranked by combined weight."""
    shared = set(user_vec.keys()) & set(city_vec.keys())
    ranked = sorted(shared, key=lambda k: user_vec[k] + city_vec[k], reverse=True)
    return ranked[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-weight", type=float, default=0.3)
    parser.add_argument("--genre-weight", type=float, default=0.4)
    parser.add_argument("--artist-weight", type=float, default=0.3)
    parser.add_argument("--top", type=int, default=20, help="Number of results to print")
    args = parser.parse_args()

    # Normalize weights
    total_w = args.audio_weight + args.genre_weight + args.artist_weight
    aw = args.audio_weight / total_w
    gw = args.genre_weight / total_w
    rw = args.artist_weight / total_w

    user = json.loads(USER_FILE.read_text())
    cities = json.loads(CITIES_FILE.read_text())

    # Precompute z-score stats from the city distribution
    stats = compute_audio_stats(cities)

    user_gv = user["genre_vector"]
    user_av = user.get("artist_vector", {})

    # Score each city
    results = []
    for city in cities:
        a_sim = audio_similarity(user["audio_features"], city["audio_features"], stats)
        g_sim = cosine_similarity(user_gv, city["genre_vector"])
        r_sim = cosine_similarity(user_av, city.get("artist_vector", {}))
        score = aw * a_sim + gw * g_sim + rw * r_sim

        shared_genres = top_shared(user_gv, city["genre_vector"])
        shared_artists = top_shared(user_av, city.get("artist_vector", {}), n=3)

        results.append({
            "city": city["city"],
            "country": city["country"],
            "score": round(score, 4),
            "audio_similarity": round(a_sim, 4),
            "genre_similarity": round(g_sim, 4),
            "artist_similarity": round(r_sim, 4),
            "top_shared_genres": shared_genres,
            "top_shared_artists": shared_artists,
        })

    results.sort(key=lambda x: -x["score"])

    # Add rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # Write full results
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Print top N
    print(f"Matching user '{user['user']}' to {len(cities)} cities")
    print(f"Weights: audio={aw:.0%}, genre={gw:.0%}, artist={rw:.0%}")
    print(f"{'='*90}")
    print(f"{'Rank':>4}  {'City':<25} {'CC':>2}  {'Score':>6}  {'Audio':>6}  {'Genre':>6}  {'Artist':>6}  Shared")
    print(f"{'-'*90}")

    for r in results[:args.top]:
        shared_g = ", ".join(r["top_shared_genres"][:3])
        shared_a = ", ".join(r["top_shared_artists"][:2])
        shared_str = shared_g
        if shared_a:
            shared_str += f" | {shared_a}"
        print(f"{r['rank']:>4}  {r['city']:<25} {r['country']:>2}  {r['score']:>6.4f}  "
              f"{r['audio_similarity']:>6.4f}  {r['genre_similarity']:>6.4f}  "
              f"{r['artist_similarity']:>6.4f}  {shared_str}")

    print(f"{'-'*90}")
    print(f"Full results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
