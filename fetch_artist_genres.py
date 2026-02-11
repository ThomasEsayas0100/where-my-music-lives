#!/usr/bin/env python3
"""
Fetch genres for tracks by combining two sources:
  1. Spotify artist genres (structured taxonomy)
  2. Last.fm top tags filtered against a genre whitelist (~5,900 known genres)

Single-track test mode:
    python3 fetch_artist_genres.py "Ink Spots" "I'm Getting Sentimental Over You"

Batch mode (enrich all playlist JSONs in data/playlists/):
    python3 fetch_artist_genres.py --batch
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

from fetch_audio_features import (
    CACHE_DIR,
    DATA_DIR,
    SPOTIFY_API_BASE,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    RateLimitedClient,
    SpotifyAuth,
)

PLAYLISTS_DIR = DATA_DIR / "playlists"
ARTIST_GENRES_CACHE = DATA_DIR / "artist_genres.json"

# Spotify allows up to 50 artist IDs per batch request
ARTIST_BATCH_SIZE = 50

# Last.fm
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

# Reusable session for connection pooling
SESSION = requests.Session()

# Genre whitelist — load once
GENRE_WHITELIST_PATH = Path("/Users/thomasesayas/Documents/lastfm_wrapped/backend/genres.json")


def load_genre_whitelist() -> set[str]:
    """Load the genre whitelist as a set of lowercase strings."""
    try:
        with open(GENRE_WHITELIST_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    except FileNotFoundError:
        print(f"WARNING: Genre whitelist not found at {GENRE_WHITELIST_PATH}")
        return set()


GENRE_WHITELIST = load_genre_whitelist()


# ---------------------------------------------------------------------------
# Last.fm tag fetching (with persistent disk caching)
# ---------------------------------------------------------------------------

# Persistent disk cache paths
LFM_TRACK_CACHE_PATH = CACHE_DIR / "lastfm_track_tags.json"
LFM_ARTIST_CACHE_PATH = CACHE_DIR / "lastfm_artist_tags.json"
_cache_lock = Lock()


def _load_disk_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_disk_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.rename(path)


# Load caches from disk on import
_lastfm_artist_tag_cache: dict[str, list[str]] = _load_disk_cache(LFM_ARTIST_CACHE_PATH)
_lastfm_track_tag_cache: dict[str, list[str]] = _load_disk_cache(LFM_TRACK_CACHE_PATH)
_lastfm_cache_stats = {"artist_hits": 0, "artist_misses": 0,
                       "track_hits": 0, "track_misses": 0}
_cache_dirty = False  # track if cache needs saving


def _lastfm_get(method: str, params: dict) -> dict:
    """Hit Last.fm API and return parsed JSON."""
    payload = {
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        **params,
    }
    r = SESSION.get(LASTFM_BASE, params=payload, timeout=12)
    r.raise_for_status()
    return r.json()


def _filter_tags_to_genres(tags: list[dict]) -> list[str]:
    """Filter raw Last.fm tag dicts to genre whitelist, sorted by count."""
    return [
        t["name"].lower()
        for t in sorted(tags, key=lambda x: int(x.get("count", 0)), reverse=True)
        if t.get("name", "").lower() in GENRE_WHITELIST
    ]


def fetch_lastfm_track_tags(artist: str, track: str) -> list[str]:
    """Fetch top tags for a track from Last.fm, filtered to genre whitelist. Cached."""
    global _cache_dirty
    cache_key = f"{artist.lower()}\t{track.lower()}"
    if cache_key in _lastfm_track_tag_cache:
        _lastfm_cache_stats["track_hits"] += 1
        return _lastfm_track_tag_cache[cache_key]

    _lastfm_cache_stats["track_misses"] += 1
    try:
        data = _lastfm_get("track.getTopTags", {"artist": artist, "track": track})
        result = _filter_tags_to_genres(data.get("toptags", {}).get("tag", []))
    except Exception:
        result = []

    with _cache_lock:
        _lastfm_track_tag_cache[cache_key] = result
        _cache_dirty = True
    return result


def fetch_lastfm_artist_tags(artist: str) -> list[str]:
    """Fetch top tags for an artist from Last.fm, filtered to genre whitelist. Cached."""
    global _cache_dirty
    cache_key = artist.lower()
    if cache_key in _lastfm_artist_tag_cache:
        _lastfm_cache_stats["artist_hits"] += 1
        return _lastfm_artist_tag_cache[cache_key]

    _lastfm_cache_stats["artist_misses"] += 1
    try:
        data = _lastfm_get("artist.getTopTags", {"artist": artist})
        result = _filter_tags_to_genres(data.get("toptags", {}).get("tag", []))
    except Exception:
        result = []

    with _cache_lock:
        _lastfm_artist_tag_cache[cache_key] = result
        _cache_dirty = True
    return result


def get_lastfm_cache_stats() -> dict:
    """Return current cache hit/miss stats."""
    return dict(_lastfm_cache_stats)


def save_lastfm_caches() -> None:
    """Persist in-memory caches to disk."""
    global _cache_dirty
    if not _cache_dirty:
        return
    with _cache_lock:
        _save_disk_cache(LFM_TRACK_CACHE_PATH, _lastfm_track_tag_cache)
        _save_disk_cache(LFM_ARTIST_CACHE_PATH, _lastfm_artist_tag_cache)
        _cache_dirty = False
    stats = get_lastfm_cache_stats()
    print(f"  💾 Saved tag caches (track: {len(_lastfm_track_tag_cache)}, artist: {len(_lastfm_artist_tag_cache)})")
    print(f"     hits: track={stats['track_hits']}, artist={stats['artist_hits']}  "
          f"misses: track={stats['track_misses']}, artist={stats['artist_misses']}")


def prefetch_lastfm_tags_concurrent(
    tracks: list[dict],
    workers: int = 10,
) -> None:
    """
    Concurrently fetch Last.fm tags for a list of tracks.
    Each track dict must have 'artist' and 'track' keys.
    Deduplicates artist-level calls automatically.
    """
    # Identify what's missing from cache
    missing_tracks = []
    missing_artists: set[str] = set()

    for t in tracks:
        tk = f"{t['artist'].lower()}\t{t['track'].lower()}"
        if tk not in _lastfm_track_tag_cache:
            missing_tracks.append(t)
        ak = t["artist"].lower()
        if ak not in _lastfm_artist_tag_cache:
            missing_artists.add(t["artist"])

    total_calls = len(missing_tracks) + len(missing_artists)
    if total_calls == 0:
        print(f"  ⚡ All {len(tracks)} tracks already cached")
        return

    print(f"  🏷️  Fetching tags: {len(missing_tracks)} tracks + {len(missing_artists)} artists "
          f"({total_calls} API calls, {workers} workers)...")

    def _fetch_track(t: dict) -> None:
        fetch_lastfm_track_tags(t["artist"], t["track"])

    def _fetch_artist(artist: str) -> None:
        fetch_lastfm_artist_tags(artist)

    # Fetch all concurrently
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for t in missing_tracks:
            futures.append(executor.submit(_fetch_track, t))
        for a in missing_artists:
            futures.append(executor.submit(_fetch_artist, a))

        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"     ...{done}/{len(futures)} done")
            try:
                f.result()
            except Exception:
                pass  # errors already handled inside fetch functions

    # Save to disk after batch
    save_lastfm_caches()


# ---------------------------------------------------------------------------
# Spotify genre fetching
# ---------------------------------------------------------------------------

def search_track(client: RateLimitedClient, artist: str, track: str) -> dict | None:
    """Search Spotify for a track and return the first matching track object."""
    url = f"{SPOTIFY_API_BASE}/search"
    params = {"q": f"track:{track} artist:{artist}", "type": "track", "limit": 5}
    data = client.get(url, params=params, use_cache=True)
    items = data.get("tracks", {}).get("items", [])
    if not items:
        # Fallback to broad search
        params["q"] = f"{artist} {track}"
        data = client.get(url, params=params, use_cache=True)
        items = data.get("tracks", {}).get("items", [])
    return items[0] if items else None


def fetch_spotify_artist_genres_batch(
    client: RateLimitedClient, artist_ids: list[str]
) -> dict[str, list[str]]:
    """
    Fetch genres for up to 50 artist IDs in one request.
    Returns {artist_id: [genre, ...]}.
    """
    result: dict[str, list[str]] = {}
    if not artist_ids:
        return result

    url = f"{SPOTIFY_API_BASE}/artists"
    params = {"ids": ",".join(artist_ids)}
    data = client.get(url, params=params, use_cache=True)

    for artist in data.get("artists", []):
        if artist and artist.get("id"):
            result[artist["id"]] = artist.get("genres", [])

    return result


def fetch_spotify_genres_for_artist_ids(
    client: RateLimitedClient, artist_ids: list[str]
) -> dict[str, list[str]]:
    """Fetch genres for an arbitrary number of artist IDs, batching by 50."""
    all_genres: dict[str, list[str]] = {}
    unique_ids = list(set(artist_ids))

    for i in range(0, len(unique_ids), ARTIST_BATCH_SIZE):
        batch = unique_ids[i : i + ARTIST_BATCH_SIZE]
        batch_genres = fetch_spotify_artist_genres_batch(client, batch)
        all_genres.update(batch_genres)

    return all_genres


# ---------------------------------------------------------------------------
# Combined genre resolution
# ---------------------------------------------------------------------------

def merge_genres(spotify_genres: list[str], lastfm_genres: list[str]) -> list[str]:
    """
    Merge Spotify and Last.fm genres into one deduplicated list.
    Spotify genres come first (more structured), then Last.fm adds depth.
    All lowercased for consistency.
    """
    seen = set()
    merged = []
    for g in spotify_genres + lastfm_genres:
        g_lower = g.lower()
        if g_lower not in seen:
            seen.add(g_lower)
            merged.append(g_lower)
    return merged


# ---------------------------------------------------------------------------
# Single-track test mode
# ---------------------------------------------------------------------------

def test_single_track(client: RateLimitedClient, artist: str, track: str) -> None:
    """Search for one track and print combined genres from both sources."""
    print(f"Searching Spotify for: {artist} - {track}")
    result = search_track(client, artist, track)

    if not result:
        print("No match found on Spotify.")
        return

    matched_artist = result["artists"][0]["name"]
    matched_track = result["name"]
    print(f"\nMatched: {matched_artist} - {matched_track}")
    print(f"Spotify ID: {result['id']}")
    print(f"Album: {result['album']['name']}")

    # --- Source 1: Spotify artist genres ---
    artist_ids = [a["id"] for a in result.get("artists", []) if a.get("id")]
    artist_names = {a["id"]: a["name"] for a in result.get("artists", []) if a.get("id")}

    spotify_genres: list[str] = []
    if artist_ids:
        genres_map = fetch_spotify_genres_for_artist_ids(client, artist_ids)
        print(f"\n{'='*60}")
        print("SOURCE 1: Spotify Artist Genres")
        print(f"{'='*60}")
        for aid in artist_ids:
            name = artist_names.get(aid, aid)
            genres = genres_map.get(aid, [])
            spotify_genres.extend(genres)
            print(f"  {name}: {genres if genres else '(none)'}")

    # --- Source 2: Last.fm tags (filtered to genre whitelist) ---
    print(f"\n{'='*60}")
    print("SOURCE 2: Last.fm Tags (genre-filtered)")
    print(f"{'='*60}")

    # Track-level tags
    track_tags = fetch_lastfm_track_tags(matched_artist, matched_track)
    print(f"  Track tags: {track_tags[:10] if track_tags else '(none)'}")

    # Artist-level tags (fallback / supplement)
    artist_tags = fetch_lastfm_artist_tags(matched_artist)
    print(f"  Artist tags: {artist_tags[:10] if artist_tags else '(none)'}")

    # Combine Last.fm: track tags first, then artist tags for depth
    lastfm_genres = list(dict.fromkeys(track_tags + artist_tags))

    # --- Merge both sources ---
    combined = merge_genres(spotify_genres, lastfm_genres)

    print(f"\n{'='*60}")
    print(f"COMBINED GENRES ({len(combined)} total)")
    print(f"{'='*60}")
    for i, g in enumerate(combined, 1):
        source = []
        if g in [s.lower() for s in spotify_genres]:
            source.append("spotify")
        if g in lastfm_genres:
            source.append("lastfm")
        print(f"  {i:2d}. {g:<35s}  [{', '.join(source)}]")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Batch mode — enrich playlist JSONs
# ---------------------------------------------------------------------------

def run_batch(client: RateLimitedClient) -> None:
    """Read all playlist JSONs, fetch combined genres, enrich and rewrite."""
    if not PLAYLISTS_DIR.exists():
        print(f"No playlists directory found at {PLAYLISTS_DIR}")
        return

    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))
    if not playlist_files:
        print("No playlist JSON files found.")
        return

    print(f"Found {len(playlist_files)} playlist files")

    # Load existing genre cache
    genre_cache: dict[str, list[str]] = {}
    if ARTIST_GENRES_CACHE.exists():
        try:
            genre_cache = json.loads(ARTIST_GENRES_CACHE.read_text())
            print(f"Loaded {len(genre_cache)} cached artist genres")
        except json.JSONDecodeError:
            pass

    all_artist_ids: set[str] = set()
    track_artist_map: dict[str, list[dict]] = {}  # track_id -> [{id, name}]

    print("Resolving artist IDs from track data...")
    for pf in playlist_files:
        data = json.loads(pf.read_text())
        for track in data.get("tracks", []):
            tid = track.get("track_id")
            if not tid or tid in track_artist_map:
                continue

            url = f"{SPOTIFY_API_BASE}/tracks/{tid}"
            track_data = client.get(url, use_cache=True)
            if not track_data:
                continue

            artists_info = []
            for a in track_data.get("artists", []):
                if a.get("id"):
                    artists_info.append({"id": a["id"], "name": a.get("name", "")})
                    all_artist_ids.add(a["id"])
            track_artist_map[tid] = artists_info

    print(f"Found {len(all_artist_ids)} unique artists across {len(track_artist_map)} tracks")

    # Fetch Spotify genres for uncached artists
    uncached = [aid for aid in all_artist_ids if aid not in genre_cache]
    if uncached:
        print(f"Fetching Spotify genres for {len(uncached)} new artists...")
        new_genres = fetch_spotify_genres_for_artist_ids(client, uncached)
        genre_cache.update(new_genres)

        ARTIST_GENRES_CACHE.write_text(
            json.dumps(genre_cache, indent=2, ensure_ascii=False)
        )
        print(f"Updated genre cache: {len(genre_cache)} artists total")

    # Enrich playlist JSONs with combined genres
    enriched_count = 0
    for pf in playlist_files:
        data = json.loads(pf.read_text())
        modified = False

        for track in data.get("tracks", []):
            tid = track.get("track_id")
            artists_info = track_artist_map.get(tid, [])

            # Spotify genres from all artists on the track
            spotify_genres: list[str] = []
            artist_name = None
            for a in artists_info:
                spotify_genres.extend(genre_cache.get(a["id"], []))
                if artist_name is None:
                    artist_name = a["name"]

            # Last.fm tags (track + artist level, genre-filtered)
            lastfm_genres: list[str] = []
            if artist_name and track.get("name"):
                track_tags = fetch_lastfm_track_tags(artist_name, track["name"])
                artist_tags = fetch_lastfm_artist_tags(artist_name)
                lastfm_genres = list(dict.fromkeys(track_tags + artist_tags))
                time.sleep(0.05)  # be gentle with Last.fm

            track["genres"] = merge_genres(spotify_genres, lastfm_genres)
            modified = True

        if modified:
            tmp = pf.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            tmp.rename(pf)
            enriched_count += 1

    print(f"\nEnriched {enriched_count} playlist files with combined genre data")
    print(f"API requests: {client.total_requests} | Cache hits: {client.cache_hits}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    auth = SpotifyAuth(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    client = RateLimitedClient(auth)

    print("Authenticating with Spotify...")
    auth.get_token()
    print("OK\n")

    print(f"Genre whitelist: {len(GENRE_WHITELIST)} genres loaded")

    if len(sys.argv) >= 3 and sys.argv[1] != "--batch":
        # Single track test: python3 fetch_artist_genres.py "Artist" "Track"
        test_single_track(client, sys.argv[1], sys.argv[2])
    elif "--batch" in sys.argv:
        run_batch(client)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
