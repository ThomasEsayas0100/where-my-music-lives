#!/usr/bin/env python3
"""
Build a feature vector from a user's recent listening history.

Produces the same format as build_city_features.py (audio_features + genre_vector)
so the user vector can be directly compared to city vectors.

Input sources:
  Last.fm:  python3 build_user_features.py --user someone --limit 500
  Spotify:  python3 build_user_features.py --spotify
"""

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
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
    prefetch_lastfm_tags_concurrent,
    save_lastfm_caches,
)

OUTPUT_FILE = DATA_DIR / "user_features.json"

AUDIO_FEATURE_KEYS = (
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
)

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

SPOTIFY_TOKEN_CACHE = CACHE_DIR / "spotify_user_token.json"
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8000/callback"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"


# ---------------------------------------------------------------------------
# Spotify user OAuth (Authorization Code + PKCE)
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _run_oauth_flow() -> dict:
    """
    Run Spotify Authorization Code + PKCE flow.
    Opens browser, captures callback on localhost:8888, returns token dict.
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": "user-read-recently-played",
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    auth_url = f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"

    # Capture the callback
    result = {"code": None, "error": None}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if qs.get("state", [None])[0] != state:
                result["error"] = "State mismatch"
            elif "error" in qs:
                result["error"] = qs["error"][0]
            else:
                result["code"] = qs.get("code", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            msg = "✅ Authenticated! You can close this tab." if result["code"] else f"❌ {result['error']}"
            self.wfile.write(f"<html><body><h2>{msg}</h2></body></html>".encode())

        def log_message(self, *args):
            pass  # suppress server logs

    server = http.server.HTTPServer(("127.0.0.1", 8000), Handler)
    server.timeout = 120

    print("  Opening browser for Spotify login...")
    webbrowser.open(auth_url)

    server.handle_request()  # blocks until callback or timeout
    server.server_close()

    if result["error"]:
        raise RuntimeError(f"Spotify auth failed: {result['error']}")
    if not result["code"]:
        raise RuntimeError("No authorization code received (timeout?)")

    # Exchange code for tokens
    resp = requests.post(SPOTIFY_TOKEN_URL, data={
        "client_id": SPOTIFY_CLIENT_ID,
        "grant_type": "authorization_code",
        "code": result["code"],
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "code_verifier": verifier,
    }, timeout=15)
    resp.raise_for_status()
    tokens = resp.json()
    tokens["obtained_at"] = time.time()
    return tokens


def _refresh_token(refresh_token: str) -> dict:
    """Use a refresh token to get a new access token."""
    resp = requests.post(SPOTIFY_TOKEN_URL, data={
        "client_id": SPOTIFY_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }, timeout=15)
    resp.raise_for_status()
    tokens = resp.json()
    tokens["obtained_at"] = time.time()
    # Spotify may or may not return a new refresh_token
    if "refresh_token" not in tokens:
        tokens["refresh_token"] = refresh_token
    return tokens


def get_spotify_user_token() -> str:
    """
    Get a valid Spotify user access token.
    Uses cached refresh token if available, otherwise runs full OAuth flow.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try cached token
    if SPOTIFY_TOKEN_CACHE.exists():
        try:
            cached = json.loads(SPOTIFY_TOKEN_CACHE.read_text())
            expires_at = cached.get("obtained_at", 0) + cached.get("expires_in", 0) - 60
            if time.time() < expires_at:
                return cached["access_token"]
            # Expired — try refresh
            if cached.get("refresh_token"):
                print("  Refreshing Spotify token...")
                tokens = _refresh_token(cached["refresh_token"])
                SPOTIFY_TOKEN_CACHE.write_text(json.dumps(tokens, indent=2))
                return tokens["access_token"]
        except Exception:
            pass  # fall through to full flow

    # Full OAuth flow
    tokens = _run_oauth_flow()
    SPOTIFY_TOKEN_CACHE.write_text(json.dumps(tokens, indent=2))
    return tokens["access_token"]


# ---------------------------------------------------------------------------
# Spotify recently played
# ---------------------------------------------------------------------------

def fetch_spotify_recent_tracks(limit: int = 50) -> list[dict]:
    """
    Fetch recently played tracks from Spotify (requires user auth).
    Returns list of {artist, track, album, play_count} dicts (deduplicated).
    Spotify API returns at most 50 items.
    """
    token = get_spotify_user_token()
    headers = {"Authorization": f"Bearer {token}"}

    api_limit = min(limit, 50)  # Spotify hard cap
    resp = requests.get(
        f"{SPOTIFY_API_BASE}/me/player/recently-played",
        headers=headers,
        params={"limit": api_limit},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])

    # Reshape to match Last.fm scrobble format
    raw_scrobbles: list[tuple[str, str, str]] = []
    for item in items:
        t = item.get("track", {})
        artist = t.get("artists", [{}])[0].get("name", "") if t.get("artists") else ""
        name = t.get("name", "")
        album = t.get("album", {}).get("name", "")
        if artist and name:
            raw_scrobbles.append((artist, name, album))

    # Deduplicate & count
    play_counts: Counter[tuple[str, str]] = Counter()
    album_map: dict[tuple[str, str], str] = {}
    for artist, track, album in raw_scrobbles:
        key = (artist.lower(), track.lower())
        play_counts[key] += 1
        if key not in album_map:
            album_map[key] = album

    unique_tracks = []
    for (artist_lower, track_lower), count in play_counts.items():
        orig_artist, orig_track = artist_lower, track_lower
        for a, t, _ in raw_scrobbles:
            if a.lower() == artist_lower and t.lower() == track_lower:
                orig_artist, orig_track = a, t
                break
        unique_tracks.append({
            "artist": orig_artist,
            "track": orig_track,
            "album": album_map.get((artist_lower, track_lower), ""),
            "play_count": count,
        })

    return unique_tracks


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

def build_user_feature_vector(
    user: str,
    limit: int = 1000,
    source: str = "lastfm",
) -> dict:
    """
    Build a complete user feature vector.

    Args:
        user: Last.fm username (used when source='lastfm').
        limit: Max scrobbles to fetch.
        source: 'lastfm' or 'spotify'.

    Returns dict with keys: user, total_tracks, unique_tracks,
    tracks_with_features, audio_features, genre_vector, artist_vector.
    """
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # 1. Fetch tracks
    if source == "spotify":
        print(f"Fetching recently played tracks from Spotify...")
        unique_tracks = fetch_spotify_recent_tracks(limit)
        user_label = "spotify_user"
    else:
        print(f"Fetching last {limit} scrobbles for '{user}'...")
        unique_tracks = fetch_recent_tracks(user, limit)
        user_label = user

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

    # 5b. Concurrent Last.fm tag prefetch
    tag_requests = [{"artist": t["artist"], "track": t["track"]} for t in unique_tracks]
    prefetch_lastfm_tags_concurrent(tag_requests, workers=10)

    # 5c. Build per-track genre lists (Spotify + Last.fm)
    track_genres: dict[int, list[str]] = {}
    for i, t in enumerate(unique_tracks):
        spotify_genres: list[str] = []
        if i in track_spotify:
            for aid in track_spotify[i]["artist_ids"]:
                spotify_genres.extend(artist_genre_map.get(aid, []))

        # Last.fm tags (already cached from prefetch)
        track_tags = fetch_lastfm_track_tags(t["artist"], t["track"])
        artist_tags = fetch_lastfm_artist_tags(t["artist"])
        lastfm_genres = list(dict.fromkeys(track_tags + artist_tags))

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
        "user": user_label,
        "source": source,
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--user", default=None, help="Last.fm username")
    group.add_argument("--spotify", action="store_true",
                       help="Use Spotify recently played instead of Last.fm")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max scrobbles (Spotify caps at 50)")
    args = parser.parse_args()

    if args.spotify:
        result = build_user_feature_vector("spotify", args.limit, source="spotify")
    else:
        user = args.user or "th_ma_"
        result = build_user_feature_vector(user, args.limit, source="lastfm")

    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Summary
    gv, av = result["genre_vector"], result["artist_vector"]
    print(f"\n{'='*60}")
    print(f"User: {result['user']}  (source: {result['source']})")
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
