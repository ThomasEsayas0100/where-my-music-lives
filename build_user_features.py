#!/usr/bin/env python3
"""
Build a feature vector from a Last.fm user's recent listening history.

Produces the same format as build_city_features.py (audio_features + genre_vector)
so the user vector can be directly compared to city vectors.

Usage:
    python3 build_user_features.py                  # default: 1000 tracks
    python3 build_user_features.py --user someone    # different user
    python3 build_user_features.py --limit 500       # fewer scrobbles
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import requests

from fetch_audio_features import (
    AUDIO_FEATURES_BATCH_SIZE,
    SPOTIFY_API_BASE,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    CACHE_DIR,
    DATA_DIR,
    RateLimitedClient,
    SpotifyAuth,
    fetch_audio_features_batch,
)
from fetch_artist_genres import (
    LASTFM_API_KEY,
    fetch_lastfm_artist_tags,
    fetch_lastfm_track_tags,
    fetch_spotify_genres_for_artist_ids,
    merge_genres,
)

OUTPUT_FILE = DATA_DIR / "user_features.json"

AUDIO_FEATURE_KEYS = (
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
)

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"


# ---------------------------------------------------------------------------
# Last.fm scrobble fetching
# ---------------------------------------------------------------------------

def fetch_recent_tracks(user: str, limit: int) -> list[dict]:
    """
    Fetch recent scrobbles from Last.fm.
    Returns list of {artist, track, album, play_count} dicts (deduplicated).
    """
    per_page = min(limit, 200)
    pages_needed = (limit + per_page - 1) // per_page
    raw_scrobbles: list[tuple[str, str, str]] = []

    for page in range(1, pages_needed + 1):
        params = {
            "method": "user.getRecentTracks",
            "user": user,
            "api_key": LASTFM_API_KEY,
            "format": "json",
            "limit": per_page,
            "page": page,
        }
        resp = requests.get(LASTFM_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        tracks = data.get("recenttracks", {}).get("track", [])
        for t in tracks:
            # Skip "now playing" entries (they have @attr.nowplaying)
            if t.get("@attr", {}).get("nowplaying"):
                continue
            artist = t.get("artist", {}).get("#text", "")
            name = t.get("name", "")
            album = t.get("album", {}).get("#text", "")
            if artist and name:
                raw_scrobbles.append((artist, name, album))

        if len(raw_scrobbles) >= limit:
            break
        time.sleep(0.25)

    raw_scrobbles = raw_scrobbles[:limit]

    # Count plays per unique track
    play_counts: Counter[tuple[str, str]] = Counter()
    album_map: dict[tuple[str, str], str] = {}
    for artist, track, album in raw_scrobbles:
        key = (artist.lower(), track.lower())
        play_counts[key] += 1
        if key not in album_map:
            album_map[key] = album

    # Build deduplicated list
    unique_tracks = []
    for (artist_lower, track_lower), count in play_counts.items():
        # Use original casing from first occurrence
        orig_artist = artist_lower
        orig_track = track_lower
        for a, t, _ in raw_scrobbles:
            if a.lower() == artist_lower and t.lower() == track_lower:
                orig_artist = a
                orig_track = t
                break
        unique_tracks.append({
            "artist": orig_artist,
            "track": orig_track,
            "album": album_map.get((artist_lower, track_lower), ""),
            "play_count": count,
        })

    return unique_tracks


# ---------------------------------------------------------------------------
# Spotify search (to resolve track IDs)
# ---------------------------------------------------------------------------

def search_spotify_track(client: RateLimitedClient, artist: str, track: str) -> dict | None:
    """Search Spotify for a track, return {id, artist_ids} or None."""
    url = f"{SPOTIFY_API_BASE}/search"
    params = {"q": f"track:{track} artist:{artist}", "type": "track", "limit": 3}
    data = client.get(url, params=params, use_cache=True)
    items = data.get("tracks", {}).get("items", [])

    if not items:
        # Fallback: broad search
        params["q"] = f"{artist} {track}"
        data = client.get(url, params=params, use_cache=True)
        items = data.get("tracks", {}).get("items", [])

    if not items:
        return None

    best = items[0]
    artist_ids = [a["id"] for a in best.get("artists", []) if a.get("id")]
    return {"id": best["id"], "artist_ids": artist_ids}


# ---------------------------------------------------------------------------
# Core pipeline (importable)
# ---------------------------------------------------------------------------

def build_user_feature_vector(user: str, limit: int = 1000) -> dict:
    """
    Build a complete user feature vector from Last.fm scrobbles.

    Returns dict with keys: user, total_tracks, unique_tracks,
    tracks_with_features, audio_features, genre_vector, artist_vector.
    """
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # 1. Fetch scrobbles
    print(f"Fetching last {limit} scrobbles for '{user}'...")
    unique_tracks = fetch_recent_tracks(user, limit)
    total_scrobbles = sum(t["play_count"] for t in unique_tracks)
    print(f"  {total_scrobbles} scrobbles -> {len(unique_tracks)} unique tracks")

    # 2. Authenticate with Spotify
    print("Authenticating with Spotify...")
    auth = SpotifyAuth(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    client = RateLimitedClient(auth)
    auth.get_token()
    print("OK\n")

    # 3. Resolve Spotify IDs
    print("Resolving Spotify track IDs...")
    track_spotify: dict[int, dict] = {}  # index -> {id, artist_ids}
    for i, t in enumerate(unique_tracks):
        result = search_spotify_track(client, t["artist"], t["track"])
        if result:
            track_spotify[i] = result
        if (i + 1) % 50 == 0:
            print(f"  searched {i + 1}/{len(unique_tracks)}")

    print(f"  Matched {len(track_spotify)}/{len(unique_tracks)} tracks on Spotify")

    # 4. Fetch audio features (batch via ReccoBeats)
    print("Fetching audio features...")
    spotify_ids = [info["id"] for info in track_spotify.values()]
    all_features: dict[str, dict | None] = {}

    for i in range(0, len(spotify_ids), AUDIO_FEATURES_BATCH_SIZE):
        batch = spotify_ids[i: i + AUDIO_FEATURES_BATCH_SIZE]
        batch_features = fetch_audio_features_batch(client, batch)
        all_features.update(batch_features)

    features_found = sum(1 for v in all_features.values() if v)
    print(f"  Got audio features for {features_found}/{len(spotify_ids)} tracks")

    # 5. Fetch genres
    print("Fetching genres...")
    # 5a. Batch-fetch Spotify artist genres
    all_artist_ids: set[str] = set()
    for info in track_spotify.values():
        all_artist_ids.update(info["artist_ids"])

    artist_genre_map: dict[str, list[str]] = {}
    if all_artist_ids:
        artist_genre_map = fetch_spotify_genres_for_artist_ids(client, list(all_artist_ids))

    # 5b. Build per-track genre lists (Spotify + Last.fm)
    track_genres: dict[int, list[str]] = {}
    for i, t in enumerate(unique_tracks):
        spotify_genres: list[str] = []
        if i in track_spotify:
            for aid in track_spotify[i]["artist_ids"]:
                spotify_genres.extend(artist_genre_map.get(aid, []))

        # Last.fm tags
        track_tags = fetch_lastfm_track_tags(t["artist"], t["track"])
        artist_tags = fetch_lastfm_artist_tags(t["artist"])
        lastfm_genres = list(dict.fromkeys(track_tags + artist_tags))
        time.sleep(0.25)

        track_genres[i] = merge_genres(spotify_genres, lastfm_genres)

    genres_found = sum(1 for g in track_genres.values() if g)
    print(f"  {genres_found}/{len(unique_tracks)} tracks have genres")

    # 6. Build feature vector
    # Audio features: weighted mean by play count
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i, t in enumerate(unique_tracks):
        if i not in track_spotify:
            continue
        feat = all_features.get(track_spotify[i]["id"])
        if not feat:
            continue
        pc = t["play_count"]
        for key in AUDIO_FEATURE_KEYS:
            val = feat.get(key)
            if val is not None:
                sums[key] = sums.get(key, 0.0) + val * pc
                counts[key] = counts.get(key, 0) + pc

    audio_features = {}
    for key in AUDIO_FEATURE_KEYS:
        if counts.get(key, 0) > 0:
            audio_features[key] = round(sums[key] / counts[key], 4)
        else:
            audio_features[key] = None

    tracks_with_features = max(counts.values()) if counts else 0

    # Genre vector: count per genre (weighted by play count) / total scrobbles
    genre_counts: Counter[str] = Counter()
    for i, t in enumerate(unique_tracks):
        pc = t["play_count"]
        for g in track_genres.get(i, []):
            genre_counts[g] += pc

    genre_vector = {
        g: round(c / total_scrobbles, 4)
        for g, c in sorted(genre_counts.items(), key=lambda x: -x[1])
    }

    # Artist vector: count per artist (weighted by play count) / total scrobbles
    artist_counts: Counter[str] = Counter()
    for i, t in enumerate(unique_tracks):
        artist_counts[t["artist"].lower()] += t["play_count"]

    artist_vector = {
        a: round(c / total_scrobbles, 4)
        for a, c in sorted(artist_counts.items(), key=lambda x: -x[1])
    }

    return {
        "user": user,
        "total_tracks": total_scrobbles,
        "unique_tracks": len(unique_tracks),
        "tracks_with_features": tracks_with_features,
        "audio_features": audio_features,
        "genre_vector": genre_vector,
        "artist_vector": artist_vector,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default="th_ma_", help="Last.fm username")
    parser.add_argument("--limit", type=int, default=1000, help="Number of recent scrobbles")
    args = parser.parse_args()

    result = build_user_feature_vector(args.user, args.limit)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Summary
    gv, av = result["genre_vector"], result["artist_vector"]
    print(f"\n{'='*60}")
    print(f"User: {result['user']}")
    print(f"Scrobbles: {result['total_tracks']}")
    print(f"Unique tracks: {result['unique_tracks']}")
    print(f"Tracks with features: {result['tracks_with_features']}")
    print(f"Unique genres: {len(gv)}")
    print(f"Unique artists: {len(av)}")
    top5 = sorted(gv.items(), key=lambda x: -x[1])[:5]
    print(f"Top genres: {top5}")
    top5a = sorted(av.items(), key=lambda x: -x[1])[:5]
    print(f"Top artists: {top5a}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
