#!/usr/bin/env python3
"""Parse Spotify 'Sound of' playlists CSV and extract city-specific playlists."""

import csv
import re
from pathlib import Path


def is_city_playlist(name: str) -> bool:
    """
    Check if playlist name matches city pattern: "The Sound of [CityName] [XX]"
    where XX is exactly 2 uppercase letters at the end.
    """
    # Pattern: "The Sound of " followed by city name, space, then exactly 2 uppercase letters at end
    pattern = r'^The Sound of .+ [A-Z]{2}$'
    return bool(re.match(pattern, name))


def extract_playlist_id(uri: str) -> str:
    """Extract playlist ID from Spotify URI."""
    # URI format: spotify:user:username:playlist:PLAYLIST_ID
    if ':playlist:' in uri:
        return uri.split(':playlist:')[-1]
    return ''


def extract_city_and_country(name: str) -> tuple[str, str]:
    """
    Extract city name and country code from playlist name.
    Assumes format: "The Sound of [CityName] [XX]"
    """
    # Remove "The Sound of " prefix
    if name.startswith("The Sound of "):
        remainder = name[13:]  # len("The Sound of ") = 13

        # Split by last space to separate city from country code
        parts = remainder.rsplit(' ', 1)
        if len(parts) == 2:
            city_name = parts[0]
            country_code = parts[1]
            return city_name, country_code

    return '', ''


def main():
    input_file = Path('/tmp/sound_playlists.csv')
    output_file = Path('/tmp/city_playlists.csv')

    print(f"Reading from: {input_file}")
    print(f"Output to: {output_file}")
    print("-" * 80)

    city_playlists = []
    total_rows = 0

    # Read input CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        # Use csv.reader to properly handle quoted fields with commas
        reader = csv.reader(f)

        for row in reader:
            total_rows += 1

            if len(row) != 3:
                print(f"Warning: Row {total_rows} has {len(row)} columns, skipping")
                continue

            playlist_name, spotify_uri, spotify_url = row

            # Check if it's a city playlist
            if is_city_playlist(playlist_name):
                playlist_id = extract_playlist_id(spotify_uri)
                city_name, country_code = extract_city_and_country(playlist_name)

                if playlist_id and city_name and country_code:
                    city_playlists.append({
                        'playlist_name': playlist_name,
                        'playlist_id': playlist_id,
                        'city_name': city_name,
                        'country_code': country_code
                    })

    # Write output CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['playlist_name', 'playlist_id', 'city_name', 'country_code']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(city_playlists)

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total rows in input file: {total_rows}")
    print(f"City playlists found: {len(city_playlists)}")
    print(f"Percentage: {len(city_playlists)/total_rows*100:.2f}%")

    # Show sample of first 10
    print(f"\n{'='*80}")
    print("FIRST 10 CITY PLAYLISTS")
    print(f"{'='*80}")
    for i, playlist in enumerate(city_playlists[:10], 1):
        print(f"{i:2d}. {playlist['playlist_name']}")
        print(f"    City: {playlist['city_name']}, Code: {playlist['country_code']}, ID: {playlist['playlist_id']}")

    # Show sample of last 10
    if len(city_playlists) > 10:
        print(f"\n{'='*80}")
        print("LAST 10 CITY PLAYLISTS")
        print(f"{'='*80}")
        for i, playlist in enumerate(city_playlists[-10:], len(city_playlists) - 9):
            print(f"{i:2d}. {playlist['playlist_name']}")
            print(f"    City: {playlist['city_name']}, Code: {playlist['country_code']}, ID: {playlist['playlist_id']}")

    print(f"\n{'='*80}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
