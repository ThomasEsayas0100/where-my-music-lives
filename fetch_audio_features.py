#!/usr/bin/env python3
"""
Fetch Spotify audio features for all tracks in city-based 'Sound of' playlists.

Reads city_playlists.csv (from parse_playlists.py), hits the Spotify Web API to
pull every track in each playlist, fetches audio features in batches of 100,
and writes one JSON file per playlist into data/playlists/.

Designed for resilience:
  - Client Credentials OAuth (no user login needed for public playlists)
  - Automatic token refresh before expiry
  - Disk-based request cache (data/cache/) to avoid duplicate API calls
  - Progress checkpoint (data/progress.json) so the script can resume after
    interruption — just re-run and it picks up where it left off
  - Respects Spotify rate limits via Retry-After headers + adaptive backoff
  - Per-playlist success-rate tracking written into each output file

Usage:
    python3 fetch_audio_features.py

Outputs:
    data/playlists/<playlist_id>.json   — one per playlist
    data/cache/                         — HTTP response cache
    data/progress.json                  — resume checkpoint
    data/run_report.json                — final summary when all playlists complete
"""

import base64
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

INPUT_CSV = Path("/tmp/city_playlists.csv")

# All outputs go under data/ in the project directory
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
CACHE_DIR = DATA_DIR / "cache"
PROGRESS_FILE = DATA_DIR / "progress.json"
REPORT_FILE = DATA_DIR / "run_report.json"

# Rate limiting — Spotify nominally allows ~180 req/30s for most endpoints.
# We stay conservative: ≤80 requests per rolling 30-second window.
RATE_LIMIT_WINDOW = 30  # seconds
RATE_LIMIT_MAX_REQUESTS = 80
# Minimum pause between any two requests (seconds)
MIN_REQUEST_INTERVAL = 0.35
# Back-off ceiling when we get 429s
MAX_BACKOFF = 120

# Audio-features endpoint accepts up to 100 IDs at once
AUDIO_FEATURES_BATCH_SIZE = 100
# Playlist tracks endpoint returns at most 100 per page
PLAYLIST_TRACKS_PAGE_SIZE = 100

# Cache TTL — 7 days (audio features don't change)
CACHE_TTL_SECONDS = 7 * 24 * 3600


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

class SpotifyAuth:
    """Handles Client Credentials flow with automatic refresh."""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: str | None = None
        self.expires_at: float = 0.0  # epoch seconds

    def _request_token(self) -> None:
        """Exchange client credentials for an access token."""
        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        resp = requests.post(
            SPOTIFY_TOKEN_URL,
            headers={
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        self.access_token = body["access_token"]
        # Refresh 60 s before actual expiry to be safe
        self.expires_at = time.time() + body["expires_in"] - 60

    def get_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if self.access_token is None or time.time() >= self.expires_at:
            self._request_token()
        return self.access_token  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(url: str, params: dict | None = None) -> str:
    """Deterministic cache key from URL + query params."""
    raw = url
    if params:
        raw += "?" + urlencode(sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(url: str, params: dict | None = None) -> dict | None:
    """Return cached JSON response or None if miss / expired."""
    key = _cache_key(url, params)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("_cached_at", 0) > CACHE_TTL_SECONDS:
            path.unlink(missing_ok=True)
            return None
        return data["payload"]
    except (json.JSONDecodeError, KeyError):
        path.unlink(missing_ok=True)
        return None


def cache_set(url: str, params: dict | None, payload: dict) -> None:
    """Persist a JSON response to disk cache."""
    key = _cache_key(url, params)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"_cached_at": time.time(), "payload": payload}))


# ---------------------------------------------------------------------------
# Rate-limited HTTP client
# ---------------------------------------------------------------------------

class RateLimitedClient:
    """
    Wraps requests.Session with:
      - rolling-window rate limiting
      - Retry-After / exponential back-off on 429
      - automatic token injection
    """

    def __init__(self, auth: SpotifyAuth):
        self.auth = auth
        self.session = requests.Session()
        self._request_timestamps: list[float] = []
        self._last_request_time: float = 0.0
        self._consecutive_429s: int = 0
        self.total_requests: int = 0
        self.cache_hits: int = 0

    def _wait_for_rate_limit(self) -> None:
        """Block until we're within the rolling-window budget."""
        now = time.time()

        # Enforce minimum interval
        since_last = now - self._last_request_time
        if since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - since_last)

        # Trim old timestamps outside the window
        cutoff = time.time() - RATE_LIMIT_WINDOW
        self._request_timestamps = [
            t for t in self._request_timestamps if t > cutoff
        ]

        # If at capacity, sleep until the oldest request falls out of window
        if len(self._request_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            sleep_until = self._request_timestamps[0] + RATE_LIMIT_WINDOW
            wait = sleep_until - time.time()
            if wait > 0:
                time.sleep(wait)

    def get(self, url: str, params: dict | None = None, use_cache: bool = True) -> dict:
        """
        GET a Spotify API endpoint.  Returns parsed JSON.
        Tries cache first, then hits the network with rate limiting.
        """
        # --- cache check ---
        if use_cache:
            cached = cache_get(url, params)
            if cached is not None:
                self.cache_hits += 1
                return cached

        # --- network ---
        backoff = 1.0
        while True:
            self._wait_for_rate_limit()

            token = self.auth.get_token()
            headers = {"Authorization": f"Bearer {token}"}

            try:
                resp = self.session.get(
                    url, headers=headers, params=params, timeout=30
                )
            except requests.RequestException as exc:
                print(f"  [network error] {exc} — retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            self._last_request_time = time.time()
            self._request_timestamps.append(self._last_request_time)
            self.total_requests += 1

            if resp.status_code == 200:
                self._consecutive_429s = 0
                body = resp.json()
                if use_cache:
                    cache_set(url, params, body)
                return body

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", backoff))
                self._consecutive_429s += 1
                wait = max(retry_after, backoff)
                if self._consecutive_429s > 1:
                    wait *= self._consecutive_429s  # escalate
                wait = min(wait, MAX_BACKOFF)
                print(f"  [429] rate limited — waiting {wait:.0f}s "
                      f"(consecutive: {self._consecutive_429s})")
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            if resp.status_code in (500, 502, 503):
                print(f"  [server {resp.status_code}] retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            if resp.status_code == 404:
                # Playlist removed / track unavailable — return empty
                return {}

            # Unexpected status
            print(f"  [HTTP {resp.status_code}] {url}")
            print(f"    body: {resp.text[:300]}")
            resp.raise_for_status()

        # unreachable, but keeps type checkers happy
        return {}  # pragma: no cover


# ---------------------------------------------------------------------------
# Progress / checkpoint management
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """Load progress checkpoint from disk."""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"completed": [], "failed": {}, "started_at": None}


def save_progress(progress: dict) -> None:
    """Atomically write progress checkpoint."""
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.rename(PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Core data-fetching logic
# ---------------------------------------------------------------------------

def fetch_playlist_tracks(client: RateLimitedClient, playlist_id: str) -> list[dict]:
    """
    Fetch all tracks from a playlist, handling pagination.
    Returns a list of track objects (only those with valid track data).
    """
    tracks = []
    url = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks"
    params = {
        "fields": "items(track(id,name,artists(name),album(name),duration_ms,popularity,explicit,external_ids)),next,total",
        "limit": PLAYLIST_TRACKS_PAGE_SIZE,
        "offset": 0,
    }

    while url:
        data = client.get(url, params=params)
        if not data:
            break

        for item in data.get("items", []):
            track = item.get("track")
            if track and track.get("id"):
                tracks.append(track)

        # Pagination: Spotify returns a full next URL
        next_url = data.get("next")
        if next_url:
            url = next_url
            params = None  # next URL already contains query params
        else:
            break

    return tracks


def fetch_audio_features_batch(
    client: RateLimitedClient, track_ids: list[str]
) -> dict[str, dict | None]:
    """
    Fetch audio features for up to 100 track IDs.
    Returns {track_id: features_dict_or_None}.
    """
    result: dict[str, dict | None] = {}
    if not track_ids:
        return result

    url = f"{SPOTIFY_API_BASE}/audio-features"
    params = {"ids": ",".join(track_ids)}
    data = client.get(url, params=params)

    for feat in data.get("audio_features", []):
        if feat and feat.get("id"):
            result[feat["id"]] = feat
        # Spotify returns null entries for unavailable tracks

    # Mark any requested IDs that got no response
    for tid in track_ids:
        if tid not in result:
            result[tid] = None

    return result


def process_playlist(
    client: RateLimitedClient,
    playlist_id: str,
    playlist_name: str,
    city_name: str,
    country_code: str,
) -> dict:
    """
    Fetch tracks + audio features for one playlist.
    Returns the assembled data dict ready for JSON serialization.
    """
    # -- tracks --
    tracks = fetch_playlist_tracks(client, playlist_id)
    total_tracks = len(tracks)

    if total_tracks == 0:
        return {
            "playlist_id": playlist_id,
            "playlist_name": playlist_name,
            "city_name": city_name,
            "country_code": country_code,
            "total_tracks": 0,
            "features_found": 0,
            "features_missing": 0,
            "success_rate": 0.0,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "tracks": [],
        }

    # -- audio features in batches of 100 --
    track_ids = [t["id"] for t in tracks]
    all_features: dict[str, dict | None] = {}

    for i in range(0, len(track_ids), AUDIO_FEATURES_BATCH_SIZE):
        batch = track_ids[i : i + AUDIO_FEATURES_BATCH_SIZE]
        batch_features = fetch_audio_features_batch(client, batch)
        all_features.update(batch_features)

    # -- assemble per-track records --
    assembled_tracks = []
    features_found = 0
    features_missing = 0

    for track in tracks:
        tid = track["id"]
        feat = all_features.get(tid)

        record = {
            "track_id": tid,
            "name": track.get("name"),
            "artists": [a["name"] for a in track.get("artists", [])],
            "album": track.get("album", {}).get("name"),
            "duration_ms": track.get("duration_ms"),
            "popularity": track.get("popularity"),
            "explicit": track.get("explicit"),
            "isrc": track.get("external_ids", {}).get("isrc"),
        }

        if feat:
            features_found += 1
            # Keep only the meaningful audio feature fields
            for key in (
                "danceability", "energy", "key", "loudness", "mode",
                "speechiness", "acousticness", "instrumentalness",
                "liveness", "valence", "tempo", "time_signature",
            ):
                record[key] = feat.get(key)
        else:
            features_missing += 1

        assembled_tracks.append(record)

    success_rate = (features_found / total_tracks * 100) if total_tracks else 0.0

    return {
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "city_name": city_name,
        "country_code": country_code,
        "total_tracks": total_tracks,
        "features_found": features_found,
        "features_missing": features_missing,
        "success_rate": round(success_rate, 2),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tracks": assembled_tracks,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- setup directories --
    DATA_DIR.mkdir(exist_ok=True)
    PLAYLISTS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # -- load input --
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        print("Run parse_playlists.py first to generate city_playlists.csv")
        sys.exit(1)

    playlists = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            playlists.append(row)

    print(f"Loaded {len(playlists)} city playlists from {INPUT_CSV}")

    # -- load progress --
    progress = load_progress()
    if progress["started_at"] is None:
        progress["started_at"] = datetime.now(timezone.utc).isoformat()
    completed_ids = set(progress["completed"])
    remaining = [p for p in playlists if p["playlist_id"] not in completed_ids]

    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already done, {len(remaining)} remaining")
    else:
        print(f"Starting fresh: {len(remaining)} playlists to process")

    # -- authenticate --
    auth = SpotifyAuth(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    client = RateLimitedClient(auth)

    # Verify credentials before starting the long run
    print("Authenticating with Spotify...")
    try:
        auth.get_token()
        print("Authentication successful.\n")
    except Exception as exc:
        print(f"ERROR: Spotify authentication failed: {exc}")
        sys.exit(1)

    # -- process playlists --
    start_time = time.time()
    processed_this_run = 0
    failed_this_run = 0

    for idx, pl in enumerate(remaining, 1):
        pid = pl["playlist_id"]
        pname = pl["playlist_name"]
        city = pl["city_name"]
        code = pl["country_code"]

        print(f"[{idx}/{len(remaining)}] {pname}")

        try:
            result = process_playlist(client, pid, pname, city, code)

            # Write playlist JSON atomically
            out_path = PLAYLISTS_DIR / f"{pid}.json"
            tmp_path = out_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            tmp_path.rename(out_path)

            # Update progress
            progress["completed"].append(pid)
            if pid in progress["failed"]:
                del progress["failed"][pid]
            save_progress(progress)

            processed_this_run += 1
            sr = result["success_rate"]
            tc = result["total_tracks"]
            ff = result["features_found"]
            print(f"  -> {tc} tracks, {ff} features ({sr}% success)")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            print(f"  Completed this run: {processed_this_run}")
            print(f"  Total completed: {len(progress['completed'])}")
            print(f"  Re-run this script to resume.")
            save_progress(progress)
            sys.exit(0)

        except Exception as exc:
            failed_this_run += 1
            progress["failed"][pid] = {
                "name": pname,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_progress(progress)
            print(f"  -> FAILED: {exc}")
            # Continue to next playlist — don't abort the entire run
            continue

    # -- final report --
    elapsed = time.time() - start_time
    total_done = len(progress["completed"])
    total_failed = len(progress["failed"])

    report = {
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "started_at": progress["started_at"],
        "elapsed_seconds": round(elapsed, 1),
        "total_playlists": len(playlists),
        "completed": total_done,
        "failed": total_failed,
        "processed_this_run": processed_this_run,
        "failed_this_run": failed_this_run,
        "api_requests": client.total_requests,
        "cache_hits": client.cache_hits,
        "failed_details": progress["failed"],
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}")
    print("RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Playlists processed this run : {processed_this_run}")
    print(f"  Failures this run            : {failed_this_run}")
    print(f"  Total completed (all runs)   : {total_done}/{len(playlists)}")
    print(f"  API requests                 : {client.total_requests}")
    print(f"  Cache hits                   : {client.cache_hits}")
    print(f"  Elapsed                      : {elapsed/60:.1f} minutes")
    print(f"  Report                       : {REPORT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
