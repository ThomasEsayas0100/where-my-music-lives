#!/usr/bin/env python3
"""
Unified pipeline: fetch tracks, audio features, and genres for all city playlists.
Outputs a single JSONL file (one line per city) to data/all_cities.jsonl.

Usage:
    python3 collect_all.py                  # process all remaining playlists
    python3 collect_all.py --limit 5        # process only 5 playlists (testing)
    python3 collect_all.py --retry-failed   # re-attempt previously failed playlists
    python3 collect_all.py --finalize       # convert JSONL to single all_cities.json
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fetch_audio_features import (
    AUDIO_FEATURES_BATCH_SIZE,
    CACHE_DIR,
    DATA_DIR,
    PLAYLIST_TRACKS_PAGE_SIZE,
    SPOTIFY_API_BASE,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    RateLimitedClient,
    SpotifyAuth,
    fetch_audio_features_batch,
)
from fetch_artist_genres import (
    ARTIST_GENRES_CACHE,
    fetch_lastfm_artist_tags,
    fetch_lastfm_track_tags,
    fetch_spotify_genres_for_artist_ids,
    get_lastfm_cache_stats,
    merge_genres,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_CSV = Path("/tmp/city_playlists.csv")
OUTPUT_FILE = DATA_DIR / "all_cities.jsonl"
PROGRESS_FILE = DATA_DIR / "collect_progress.json"
REPORT_FILE = DATA_DIR / "collect_report.json"

ARTIST_BATCH_SIZE = 50
# Pause between Last.fm calls to be polite
LASTFM_SLEEP = 0.25
# Persist artist genre cache to disk every N playlists
CACHE_SAVE_INTERVAL = 10


# ---------------------------------------------------------------------------
# Progress management
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"completed": [], "failed": {}, "started_at": None}


def save_progress(progress: dict) -> None:
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.rename(PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Artist genre cache (disk-backed)
# ---------------------------------------------------------------------------

def load_artist_genre_cache() -> dict[str, list[str]]:
    if ARTIST_GENRES_CACHE.exists():
        try:
            return json.loads(ARTIST_GENRES_CACHE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_artist_genre_cache(cache: dict[str, list[str]]) -> None:
    tmp = ARTIST_GENRES_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    tmp.rename(ARTIST_GENRES_CACHE)


# ---------------------------------------------------------------------------
# Track fetching (with artist IDs — key optimization)
# ---------------------------------------------------------------------------

def fetch_playlist_tracks_full(
    client: RateLimitedClient, playlist_id: str
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Fetch all tracks from a playlist, capturing artist IDs alongside names.

    Returns:
        tracks: list of track dicts (id, name, artists, album, etc.)
        track_artists: {track_id: [{"id": "...", "name": "..."}, ...]}
    """
    tracks: list[dict] = []
    track_artists: dict[str, list[dict]] = {}

    url = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks"
    params = {
        "fields": (
            "items(track(id,name,artists(id,name),album(name),"
            "duration_ms,popularity,explicit,external_ids)),next,total"
        ),
        "limit": PLAYLIST_TRACKS_PAGE_SIZE,
        "offset": 0,
    }

    while url:
        data = client.get(url, params=params)
        if not data:
            break

        for item in data.get("items", []):
            track = item.get("track")
            if not track or not track.get("id"):
                continue

            tid = track["id"]

            # Build artist info with IDs
            artists_info = []
            artist_names = []
            for a in track.get("artists", []):
                name = a.get("name", "")
                artist_names.append(name)
                if a.get("id"):
                    artists_info.append({"id": a["id"], "name": name})

            track_artists[tid] = artists_info

            tracks.append({
                "id": tid,
                "name": track.get("name"),
                "artists": artist_names,
                "album": track.get("album", {}).get("name"),
                "duration_ms": track.get("duration_ms"),
                "popularity": track.get("popularity"),
                "explicit": track.get("explicit"),
                "isrc": track.get("external_ids", {}).get("isrc"),
            })

        next_url = data.get("next")
        if next_url:
            url = next_url
            params = None
        else:
            break

    return tracks, track_artists


# ---------------------------------------------------------------------------
# Per-playlist processing (single pass: tracks + features + genres)
# ---------------------------------------------------------------------------

def process_playlist(
    client: RateLimitedClient,
    playlist_id: str,
    playlist_name: str,
    city_name: str,
    country_code: str,
    artist_genre_cache: dict[str, list[str]],
) -> dict:
    """Full pipeline for one playlist: tracks + audio features + genres."""

    # 1. Fetch tracks (with artist IDs)
    tracks, track_artists = fetch_playlist_tracks_full(client, playlist_id)
    total_tracks = len(tracks)

    if total_tracks == 0:
        return {
            "city": city_name,
            "country": country_code,
            "playlist_id": playlist_id,
            "playlist_name": playlist_name,
            "total_tracks": 0,
            "features_found": 0,
            "genres_found": 0,
            "tracks": [],
        }

    # 2. Fetch audio features from ReccoBeats (batches of 40)
    track_ids = [t["id"] for t in tracks]
    all_features: dict[str, dict | None] = {}

    for i in range(0, len(track_ids), AUDIO_FEATURES_BATCH_SIZE):
        batch = track_ids[i : i + AUDIO_FEATURES_BATCH_SIZE]
        batch_features = fetch_audio_features_batch(client, batch)
        all_features.update(batch_features)

    # 3. Fetch genres
    # 3a. Collect unique artist IDs and batch-fetch Spotify genres for uncached ones
    all_artist_ids: set[str] = set()
    for artists_info in track_artists.values():
        for a in artists_info:
            all_artist_ids.add(a["id"])

    uncached_ids = [aid for aid in all_artist_ids if aid not in artist_genre_cache]
    if uncached_ids:
        new_genres = fetch_spotify_genres_for_artist_ids(client, uncached_ids)
        artist_genre_cache.update(new_genres)

    # 3b. Build per-track genres (Spotify + Last.fm)
    features_found = 0
    genres_found = 0

    for track in tracks:
        tid = track["id"]
        feat = all_features.get(tid)

        # Merge audio features
        if feat:
            features_found += 1
            for key in (
                "danceability", "energy", "key", "loudness", "mode",
                "speechiness", "acousticness", "instrumentalness",
                "liveness", "valence", "tempo",
            ):
                track[key] = feat.get(key)

        # Spotify genres from all artists on this track
        artists_info = track_artists.get(tid, [])
        spotify_genres: list[str] = []
        primary_artist_name = None
        for a in artists_info:
            spotify_genres.extend(artist_genre_cache.get(a["id"], []))
            if primary_artist_name is None:
                primary_artist_name = a["name"]

        # Last.fm tags (track + artist level)
        lastfm_genres: list[str] = []
        if primary_artist_name and track.get("name"):
            track_tags = fetch_lastfm_track_tags(primary_artist_name, track["name"])
            artist_tags = fetch_lastfm_artist_tags(primary_artist_name)
            lastfm_genres = list(dict.fromkeys(track_tags + artist_tags))
            time.sleep(LASTFM_SLEEP)

        track["genres"] = merge_genres(spotify_genres, lastfm_genres)
        if track["genres"]:
            genres_found += 1

    return {
        "city": city_name,
        "country": country_code,
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "total_tracks": total_tracks,
        "features_found": features_found,
        "genres_found": genres_found,
        "tracks": tracks,
    }


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def append_jsonl(record: dict) -> None:
    """Append one compact JSON line to the output file."""
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def finalize_to_json() -> None:
    """Convert JSONL output to a single compact JSON array file."""
    if not OUTPUT_FILE.exists():
        print("No JSONL output file found.")
        return

    cities = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cities.append(json.loads(line))

    final = DATA_DIR / "all_cities.json"
    with open(final, "w", encoding="utf-8") as f:
        json.dump(cities, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(cities)} cities to {final} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only N playlists (for testing)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt previously failed playlists")
    parser.add_argument("--finalize", action="store_true",
                        help="Convert JSONL to single all_cities.json")
    args = parser.parse_args()

    if args.finalize:
        finalize_to_json()
        return

    # Setup directories
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # Load input CSV
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        print("Run parse_playlists.py first to generate city_playlists.csv")
        sys.exit(1)

    playlists = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            playlists.append(row)

    print(f"Loaded {len(playlists)} city playlists from {INPUT_CSV}")

    # Load progress
    progress = load_progress()
    if progress["started_at"] is None:
        progress["started_at"] = datetime.now(timezone.utc).isoformat()

    # Handle --retry-failed
    if args.retry_failed and progress["failed"]:
        retry_ids = list(progress["failed"].keys())
        print(f"Retrying {len(retry_ids)} previously failed playlists")
        for rid in retry_ids:
            del progress["failed"][rid]
        save_progress(progress)

    completed_ids = set(progress["completed"])
    remaining = [p for p in playlists if p["playlist_id"] not in completed_ids]

    if completed_ids:
        print(f"Resuming: {len(completed_ids)} done, {len(remaining)} remaining")
    else:
        print(f"Starting fresh: {len(remaining)} playlists to process")

    if args.limit > 0:
        remaining = remaining[: args.limit]
        print(f"Limited to {len(remaining)} playlists")

    if not remaining:
        print("Nothing to process.")
        return

    # Authenticate
    auth = SpotifyAuth(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    client = RateLimitedClient(auth)

    print("Authenticating with Spotify...")
    try:
        auth.get_token()
        print("OK\n")
    except Exception as exc:
        print(f"ERROR: Spotify authentication failed: {exc}")
        sys.exit(1)

    # Load artist genre cache
    artist_genre_cache = load_artist_genre_cache()
    print(f"Artist genre cache: {len(artist_genre_cache)} artists loaded")

    # Process playlists
    start_time = time.time()
    processed = 0
    failed = 0

    for idx, pl in enumerate(remaining, 1):
        pid = pl["playlist_id"]
        pname = pl["playlist_name"]
        city = pl["city_name"]
        code = pl["country_code"]

        print(f"[{idx}/{len(remaining)}] {pname}")

        try:
            result = process_playlist(
                client, pid, pname, city, code, artist_genre_cache
            )

            append_jsonl(result)

            progress["completed"].append(pid)
            if pid in progress["failed"]:
                del progress["failed"][pid]
            save_progress(progress)

            processed += 1
            tc = result["total_tracks"]
            ff = result["features_found"]
            gf = result["genres_found"]
            print(f"  -> {tc} tracks, {ff} features, {gf} with genres")

            # Periodically save artist genre cache
            if processed % CACHE_SAVE_INTERVAL == 0:
                save_artist_genre_cache(artist_genre_cache)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving progress...")
            save_progress(progress)
            save_artist_genre_cache(artist_genre_cache)
            print(f"  Processed this run: {processed}")
            print(f"  Total completed: {len(progress['completed'])}")
            print("  Re-run to resume.")
            sys.exit(0)

        except Exception as exc:
            failed += 1
            progress["failed"][pid] = {
                "name": pname,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_progress(progress)
            print(f"  -> FAILED: {exc}")
            continue

    # Final saves
    save_artist_genre_cache(artist_genre_cache)

    elapsed = time.time() - start_time
    total_done = len(progress["completed"])
    total_failed = len(progress["failed"])

    # Report
    report = {
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "started_at": progress["started_at"],
        "elapsed_seconds": round(elapsed, 1),
        "total_playlists": len(playlists),
        "completed": total_done,
        "failed": total_failed,
        "processed_this_run": processed,
        "failed_this_run": failed,
        "api_requests": client.total_requests,
        "cache_hits": client.cache_hits,
        "lastfm_cache_stats": get_lastfm_cache_stats(),
        "artist_genre_cache_size": len(artist_genre_cache),
        "failed_details": progress["failed"],
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}")
    print("RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Processed this run : {processed}")
    print(f"  Failures this run  : {failed}")
    print(f"  Total completed    : {total_done}/{len(playlists)}")
    print(f"  API requests       : {client.total_requests}")
    print(f"  Cache hits         : {client.cache_hits}")
    print(f"  Artist genres      : {len(artist_genre_cache)} cached")
    print(f"  Last.fm cache      : {get_lastfm_cache_stats()}")
    print(f"  Elapsed            : {elapsed/60:.1f} minutes")
    print(f"  Output             : {OUTPUT_FILE}")
    print(f"  Report             : {REPORT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
