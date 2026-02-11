#!/usr/bin/env python3
"""
Single-call function that produces comprehensive music stats for a Last.fm user,
including top city matches based on pre-computed city feature vectors.

Usage:
    python3 get_user_stats.py                        # default user, 1000 scrobbles
    python3 get_user_stats.py --user someone          # different user
    python3 get_user_stats.py --limit 500 --top 10    # fewer scrobbles, top 10 cities
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from build_user_features import build_user_feature_vector
from match_user_to_cities import (
    compute_audio_stats,
    audio_similarity,
    cosine_similarity,
    top_shared,
    CITIES_FILE,
)


def get_user_stats(
    username: str,
    limit: int = 1000,
    top_n: int = 20,
    weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
) -> dict:
    """
    Build a user's music profile and match against pre-computed city vectors.

    Args:
        username: Last.fm username
        limit: Number of recent scrobbles to analyze
        top_n: Number of top city matches to include
        weights: (audio_weight, genre_weight, artist_weight) — auto-normalized

    Returns:
        Dict with user_profile, city_matches, and meta.
    """
    # 1. Build user feature vector
    user = build_user_feature_vector(username, limit)

    # 2. Load pre-computed city features
    cities = json.loads(CITIES_FILE.read_text())

    # 3. Normalize weights
    aw, gw, rw = weights
    total_w = aw + gw + rw
    aw, gw, rw = aw / total_w, gw / total_w, rw / total_w

    # 4. Score and rank cities
    stats = compute_audio_stats(cities)
    user_gv = user["genre_vector"]
    user_av = user.get("artist_vector", {})

    results = []
    for city in cities:
        a_sim = audio_similarity(user["audio_features"], city["audio_features"], stats)
        g_sim = cosine_similarity(user_gv, city["genre_vector"])
        r_sim = cosine_similarity(user_av, city.get("artist_vector", {}))
        score = aw * a_sim + gw * g_sim + rw * r_sim

        results.append({
            "rank": 0,
            "city": city["city"],
            "country": city["country"],
            "score": round(score, 4),
            "audio_similarity": round(a_sim, 4),
            "genre_similarity": round(g_sim, 4),
            "artist_similarity": round(r_sim, 4),
            "top_shared_genres": top_shared(user_gv, city["genre_vector"]),
            "top_shared_artists": top_shared(user_av, city.get("artist_vector", {}), n=3),
        })

    results.sort(key=lambda x: -x["score"])
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # 5. Assemble stats
    genre_items = sorted(user_gv.items(), key=lambda x: -x[1])
    artist_items = sorted(user_av.items(), key=lambda x: -x[1])

    return {
        "user_profile": {
            "username": username,
            "total_scrobbles": user["total_tracks"],
            "unique_tracks": user["unique_tracks"],
            "tracks_with_audio_features": user["tracks_with_features"],
            "audio_features": user["audio_features"],
            "top_genres": genre_items[:15],
            "top_artists": artist_items[:15],
            "total_unique_genres": len(user_gv),
            "total_unique_artists": len(user_av),
        },
        "city_matches": results[:top_n],
        "meta": {
            "scrobble_limit": limit,
            "cities_evaluated": len(cities),
            "weights": {"audio": round(aw, 2), "genre": round(gw, 2), "artist": round(rw, 2)},
        },
    }


def print_stats(stats: dict) -> None:
    """Pretty-print user stats to the terminal."""
    p = stats["user_profile"]
    matches = stats["city_matches"]
    meta = stats["meta"]
    W = 72

    # --- Header ---
    print()
    print(f"{'':=<{W}}")
    print(f"  WHERE YOUR MUSIC LIVES")
    print(f"{'':=<{W}}")
    print(f"  @{p['username']}")
    print(f"  {p['total_scrobbles']:,} scrobbles analyzed "
          f"({p['unique_tracks']:,} unique tracks)")
    print(f"{'':=<{W}}")

    # --- Audio DNA ---
    af = p["audio_features"]
    print(f"\n  AUDIO DNA")
    print(f"  {'':─<{W - 4}}")

    bars = [
        ("Danceability",     af.get("danceability", 0)),
        ("Energy",           af.get("energy", 0)),
        ("Valence",          af.get("valence", 0)),
        ("Acousticness",     af.get("acousticness", 0)),
        ("Instrumentalness", af.get("instrumentalness", 0)),
        ("Speechiness",      af.get("speechiness", 0)),
        ("Liveness",         af.get("liveness", 0)),
    ]
    for label, val in bars:
        val = val or 0
        filled = int(val * 20)
        bar = "\u2588" * filled + "\u2591" * (20 - filled)
        print(f"  {label:<18} {bar} {val:.0%}")

    print(f"  {'':·<{W - 4}}")
    tempo = af.get("tempo", 0) or 0
    loudness = af.get("loudness", 0) or 0
    print(f"  Tempo: {tempo:.0f} BPM    Loudness: {loudness:.1f} dB")

    # --- Top Genres ---
    genres = p["top_genres"]
    print(f"\n  TOP GENRES ({p['total_unique_genres']} unique)")
    print(f"  {'':─<{W - 4}}")
    max_g = genres[0][1] if genres else 1
    for i, (genre, weight) in enumerate(genres[:10], 1):
        filled = int((weight / max_g) * 16)
        bar = "\u2588" * filled + "\u2591" * (16 - filled)
        print(f"  {i:>2}. {genre:<22} {bar} {weight:.1%}")

    # --- Top Artists ---
    artists = p["top_artists"]
    print(f"\n  TOP ARTISTS ({p['total_unique_artists']} unique)")
    print(f"  {'':─<{W - 4}}")
    max_a = artists[0][1] if artists else 1
    for i, (artist, weight) in enumerate(artists[:10], 1):
        filled = int((weight / max_a) * 16)
        bar = "\u2588" * filled + "\u2591" * (16 - filled)
        print(f"  {i:>2}. {artist:<22} {bar} {weight:.1%}")

    # --- City Matches ---
    w = meta["weights"]
    print(f"\n{'':=<{W}}")
    print(f"  CITY MATCHES  "
          f"(audio {w['audio']:.0%} / genre {w['genre']:.0%} / artist {w['artist']:.0%})")
    print(f"  {meta['cities_evaluated']} cities evaluated")
    print(f"{'':=<{W}}")

    print(f"\n  {'#':>3}  {'City':<24} {'CC':>2}  {'Score':>6}  "
          f"{'Aud':>5} {'Gen':>5} {'Art':>5}  Shared Genres")
    print(f"  {'':─<{W - 4}}")

    for r in matches:
        shared = ", ".join(r["top_shared_genres"][:3])
        artists_shared = ", ".join(r["top_shared_artists"][:2])
        if artists_shared:
            shared += f" | {artists_shared}"

        # Score bar (scaled to max score in results)
        print(f"  {r['rank']:>3}  {r['city']:<24} {r['country']:>2}  "
              f"{r['score']:.4f}  "
              f"{r['audio_similarity']:.3f} {r['genre_similarity']:.3f} "
              f"{r['artist_similarity']:.3f}  {shared}")

    print(f"  {'':─<{W - 4}}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default="th_ma_", help="Last.fm username")
    parser.add_argument("--limit", type=int, default=1000, help="Number of recent scrobbles")
    parser.add_argument("--top", type=int, default=20, help="Number of top city matches")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted")
    args = parser.parse_args()

    stats = get_user_stats(args.user, limit=args.limit, top_n=args.top)
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print_stats(stats)
