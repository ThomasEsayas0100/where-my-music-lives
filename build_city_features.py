#!/usr/bin/env python3
"""
Extract playlist-level feature vectors from raw track data.

Reads data/all_cities.jsonl and produces data/city_features.json with:
  - audio_features: mean of each audio feature across tracks (skipping nulls)
  - genre_vector: normalized genre counts (count / total_tracks)

Usage:
    python3 build_city_features.py
"""

import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INPUT_FILE = DATA_DIR / "all_cities.jsonl"
OUTPUT_FILE = DATA_DIR / "city_features.json"

AUDIO_FEATURE_KEYS = (
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
)


def build_features(city: dict) -> dict:
    """Compute audio feature averages and normalized genre vector for one city."""
    tracks = city["tracks"]
    total_tracks = city["total_tracks"]

    # Audio feature averages (skip nulls)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for t in tracks:
        for key in AUDIO_FEATURE_KEYS:
            val = t.get(key)
            if val is not None:
                sums[key] = sums.get(key, 0.0) + val
                counts[key] = counts.get(key, 0) + 1

    audio_features = {}
    for key in AUDIO_FEATURE_KEYS:
        if counts.get(key, 0) > 0:
            audio_features[key] = round(sums[key] / counts[key], 4)
        else:
            audio_features[key] = None

    tracks_with_features = max(counts.values()) if counts else 0

    # Genre vector: count per genre / total_tracks
    genre_counts: Counter[str] = Counter()
    for t in tracks:
        for g in t.get("genres", []):
            genre_counts[g] += 1

    genre_vector = {
        g: round(c / total_tracks, 4)
        for g, c in sorted(genre_counts.items(), key=lambda x: -x[1])
    }

    # Artist vector: count per artist / total_tracks (lowercase, deduplicated per track)
    artist_counts: Counter[str] = Counter()
    for t in tracks:
        seen = set()
        for a in t.get("artists", []):
            a_lower = a.lower()
            if a_lower not in seen:
                seen.add(a_lower)
                artist_counts[a_lower] += 1

    artist_vector = {
        a: round(c / total_tracks, 4)
        for a, c in sorted(artist_counts.items(), key=lambda x: -x[1])
    }

    return {
        "city": city["city"],
        "country": city["country"],
        "playlist_id": city["playlist_id"],
        "total_tracks": total_tracks,
        "tracks_with_features": tracks_with_features,
        "audio_features": audio_features,
        "genre_vector": genre_vector,
        "artist_vector": artist_vector,
    }


def main() -> None:
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run collect_all.py first.")
        return

    results = []
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            city = json.loads(line)
            if city["total_tracks"] == 0:
                skipped += 1
                continue
            results.append(build_features(city))

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    # Stats
    all_genres: set[str] = set()
    all_artists: set[str] = set()
    feat_counts = []
    for r in results:
        all_genres.update(r["genre_vector"].keys())
        all_artists.update(r["artist_vector"].keys())
        feat_counts.append(r["tracks_with_features"])

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"Cities with data:    {len(results)}")
    print(f"Empty (skipped):     {skipped}")
    print(f"Unique genres:       {len(all_genres)}")
    print(f"Unique artists:      {len(all_artists)}")
    print(f"Audio features/city: min={min(feat_counts)}, max={max(feat_counts)}, "
          f"mean={sum(feat_counts)/len(feat_counts):.1f}")
    print(f"Output: {OUTPUT_FILE} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
