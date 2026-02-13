#!/usr/bin/env python3
"""
Where Your Music Lives — Web Application

Serves a landing page with Spotify / Last.fm authentication,
then runs the feature-building pipeline and shows city matches.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import Flask, redirect, request, render_template, session, url_for
import requests

from build_user_features import (
    build_user_feature_vector,
    get_spotify_user_token,
    fetch_spotify_recent_tracks,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_REDIRECT_URI,
    SPOTIFY_AUTH_URL,
    SPOTIFY_TOKEN_URL,
    CACHE_DIR,
    DATA_DIR,
)
from fetch_audio_features import SPOTIFY_API_BASE

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")
LASTFM_SHARED_SECRET = os.environ.get("LASTFM_SHARED_SECRET", "")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


# --------------- Spotify Auth ---------------

@app.route("/auth/spotify")
def auth_spotify():
    """Redirect to Spotify OAuth."""
    import secrets, hashlib, base64

    # Generate PKCE
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    state = secrets.token_hex(16)

    session["pkce_verifier"] = verifier
    session["oauth_state"] = state

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": url_for("spotify_callback", _external=True),
        "scope": "user-read-recently-played",
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    import urllib.parse
    return redirect(f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}")


@app.route("/callback/spotify")
def spotify_callback():
    """Handle Spotify OAuth callback."""
    error = request.args.get("error")
    if error:
        return render_template("index.html", error=f"Spotify auth failed: {error}")

    if request.args.get("state") != session.get("oauth_state"):
        return render_template("index.html", error="State mismatch — please try again")

    code = request.args.get("code")
    verifier = session.get("pkce_verifier")

    # Exchange code for tokens
    resp = requests.post(SPOTIFY_TOKEN_URL, data={
        "client_id": SPOTIFY_CLIENT_ID,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": url_for("spotify_callback", _external=True),
        "code_verifier": verifier,
    }, timeout=15)

    if resp.status_code != 200:
        return render_template("index.html", error="Token exchange failed")

    tokens = resp.json()

    # Save token for pipeline use
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokens["obtained_at"] = time.time()
    (CACHE_DIR / "spotify_user_token.json").write_text(json.dumps(tokens, indent=2))

    # Get user display name
    me_resp = requests.get(
        f"{SPOTIFY_API_BASE}/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
        timeout=10,
    )
    spotify_name = me_resp.json().get("display_name", "Spotify User") if me_resp.ok else "Spotify User"

    session["source"] = "spotify"
    session["user_name"] = spotify_name
    session["access_token"] = tokens["access_token"]

    return redirect(url_for("processing"))


# --------------- Last.fm (username) ---------------

@app.route("/auth/lastfm")
def auth_lastfm():
    """Accept Last.fm username and proceed to processing."""
    username = request.args.get("user", "").strip()
    if not username:
        return render_template("index.html", error="Please enter a Last.fm username")

    session["source"] = "lastfm"
    session["user_name"] = username
    return redirect(url_for("processing"))


# --------------- Processing ---------------

@app.route("/processing")
def processing():
    """Show processing page, then redirect to results."""
    source = session.get("source")
    user_name = session.get("user_name", "")
    if not source:
        return redirect(url_for("index"))
    return render_template("processing.html", source=source, user_name=user_name)


@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline():
    """Run the feature pipeline and return results as JSON."""
    source = session.get("source")
    user_name = session.get("user_name", "")

    if not source:
        return {"error": "No auth session"}, 401

    try:
        if source == "spotify":
            result = build_user_feature_vector("spotify", 50, source="spotify")
        else:
            result = build_user_feature_vector(user_name, 1000, source="lastfm")

        # Also run city matching
        from match_user_to_cities import match_user_to_cities
        matches = match_user_to_cities()

        return {
            "user": result,
            "matches": matches[:20],  # top 20 cities
        }
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/results")
def results():
    """Show results page."""
    source = session.get("source")
    user_name = session.get("user_name", "")
    if not source:
        return redirect(url_for("index"))
    return render_template("results.html", source=source, user_name=user_name)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
