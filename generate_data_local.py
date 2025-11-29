"""
Generate Realistic Synthetic Spotify Dataset - LOCAL VERSION
Run this on your Mac to generate the dataset, then upload to S3
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys


def generate_realistic_audio_features(n_rows, seed=42):
    """Generate realistic audio features with proper distributions"""
    np.random.seed(seed)

    print(f"  Generating audio features for {n_rows:,} tracks...")

    # ENERGY - Beta distribution (right-skewed, mean ~0.65)
    energy = np.random.beta(2.5, 2, n_rows)

    # LOUDNESS - Correlated with energy
    loudness = -60 + energy * 55 + np.random.normal(0, 3, n_rows)
    loudness = np.clip(loudness, -60, 0)

    # DANCEABILITY - Beta distribution (right-skewed, mean ~0.60)
    danceability = np.random.beta(2.8, 2.5, n_rows)

    # VALENCE - Correlated with danceability
    valence = 0.3 * danceability + 0.7 * np.random.beta(2, 2, n_rows)
    valence = np.clip(valence, 0, 1)

    # TEMPO - Normal around 120 BPM
    tempo = np.random.normal(120, 28, n_rows)
    tempo = np.clip(tempo, 40, 220)

    # ACOUSTICNESS - Negatively correlated with energy
    acousticness = np.random.beta(1.5, 5, n_rows)
    acousticness = 0.7 * (1 - energy) + 0.3 * acousticness
    acousticness = np.clip(acousticness, 0, 1)

    # INSTRUMENTALNESS - Heavy right skew (most have vocals)
    instrumentalness = np.random.beta(1, 10, n_rows)

    # SPEECHINESS - Heavy right skew
    speechiness = np.random.beta(1, 8, n_rows)

    # LIVENESS - Heavy right skew (studio recordings)
    liveness = np.random.beta(1.2, 6, n_rows)

    # DURATION - Right-skewed around 3-4 minutes
    duration_ms = np.random.gamma(8, 25000, n_rows)
    duration_ms = np.clip(duration_ms, 30000, 600000)

    # KEY - 0-11
    key = np.random.randint(0, 12, n_rows)

    # MODE - 0=minor, 1=major (correlated with valence)
    mode_prob = 0.45 + 0.2 * valence
    mode = (np.random.random(n_rows) < mode_prob).astype(int)

    # TIME_SIGNATURE - Mostly 4/4
    time_signature = np.random.choice([2, 3, 4, 5], n_rows, p=[0.02, 0.08, 0.85, 0.05])

    return {
        "energy": energy,
        "loudness": loudness,
        "danceability": danceability,
        "valence": valence,
        "tempo": tempo,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "speechiness": speechiness,
        "liveness": liveness,
        "duration_ms": duration_ms,
        "key": key,
        "mode": mode,
        "time_signature": time_signature,
    }


def generate_categories(n_rows, seed=42):
    """Generate playlist categories"""
    np.random.seed(seed + 1)

    categories = [
        "workout",
        "party",
        "chill",
        "sleep",
        "focus",
        "study",
        "pop",
        "rock",
        "indie",
    ]
    probs = [0.12, 0.15, 0.18, 0.08, 0.14, 0.12, 0.10, 0.06, 0.05]

    return np.random.choice(categories, n_rows, p=probs)


def adjust_by_category(df):
    """Make categories have distinct audio profiles"""
    print("  Adjusting features by category...")

    # Workout: high energy, high tempo
    mask = df["playlist_category"] == "workout"
    df.loc[mask, "energy"] = np.clip(df.loc[mask, "energy"] + 0.15, 0, 1)
    df.loc[mask, "tempo"] = np.clip(df.loc[mask, "tempo"] + 15, 40, 220)

    # Sleep: low energy, low tempo, high acousticness
    mask = df["playlist_category"] == "sleep"
    df.loc[mask, "energy"] = np.clip(df.loc[mask, "energy"] - 0.25, 0, 1)
    df.loc[mask, "tempo"] = np.clip(df.loc[mask, "tempo"] - 25, 40, 220)
    df.loc[mask, "acousticness"] = np.clip(df.loc[mask, "acousticness"] + 0.2, 0, 1)

    # Party: high everything fun
    mask = df["playlist_category"] == "party"
    df.loc[mask, "energy"] = np.clip(df.loc[mask, "energy"] + 0.12, 0, 1)
    df.loc[mask, "danceability"] = np.clip(df.loc[mask, "danceability"] + 0.15, 0, 1)
    df.loc[mask, "valence"] = np.clip(df.loc[mask, "valence"] + 0.12, 0, 1)

    # Chill: lower energy, higher acousticness
    mask = df["playlist_category"] == "chill"
    df.loc[mask, "energy"] = np.clip(df.loc[mask, "energy"] - 0.12, 0, 1)
    df.loc[mask, "acousticness"] = np.clip(df.loc[mask, "acousticness"] + 0.15, 0, 1)

    # Focus/Study: lower energy
    mask = (df["playlist_category"] == "focus") | (df["playlist_category"] == "study")
    df.loc[mask, "energy"] = np.clip(df.loc[mask, "energy"] - 0.08, 0, 1)
    df.loc[mask, "speechiness"] = np.clip(df.loc[mask, "speechiness"] - 0.05, 0, 1)

    return df


def generate_popularity(df):
    """Generate popularity based on audio features"""
    print("  Calculating popularity scores...")

    popularity = (
        25 * df["energy"]
        + 20 * df["danceability"]
        + 15 * df["valence"]
        + 10 * (1 - df["acousticness"])
        + 10 * (df["loudness"] + 60) / 60
    )

    # Add randomness
    popularity += np.random.normal(0, 15, len(df))

    # Power law (most unpopular, few hits)
    popularity = np.power(popularity / 100, 1.5) * 100
    popularity = np.clip(popularity, 0, 100).astype(int)

    return popularity


def generate_artists(n_rows, seed=42):
    """Generate artist names with power law distribution"""
    np.random.seed(seed + 2)

    # Power law: few artists have many tracks
    n_unique = min(50000, n_rows // 10)
    artist_ids = np.arange(n_unique)
    weights = 1 / np.power(artist_ids + 1, 0.8)
    weights /= weights.sum()

    selected = np.random.choice(artist_ids, n_rows, p=weights, replace=True)

    # Artist name templates
    first = [
        "Drake",
        "Taylor",
        "Ed",
        "Ariana",
        "Post",
        "Billie",
        "The Weeknd",
        "Justin",
        "Dua",
        "Bad",
        "Olivia",
        "Travis",
        "Kanye",
        "Kendrick",
        "Rihanna",
        "Bruno",
        "Eminem",
        "Lil",
        "DJ",
        "Twenty",
    ]
    last = [
        "Swift",
        "Sheeran",
        "Grande",
        "Malone",
        "Eilish",
        "Bieber",
        "Lipa",
        "Bunny",
        "Rodrigo",
        "Scott",
        "West",
        "Lamar",
        "Mars",
        "Nas X",
        "Khalid",
        "One Pilots",
        "Uzi",
        "Baby",
        "Wayne",
        "Cole",
    ]

    artists = []
    for aid in selected:
        if aid < len(first) * len(last):
            artists.append(
                f"{first[aid % len(first)]} {last[aid // len(first) % len(last)]}"
            )
        else:
            artists.append(f"Artist {aid}")

    return artists


def generate_tracks(n_rows, seed=42):
    """Generate track names"""
    np.random.seed(seed + 3)
    adj = ["Beautiful", "Dark", "Sweet", "Lost", "Perfect", "Golden", "Wild", "Falling"]
    nouns = ["Love", "Dreams", "Heart", "Night", "Summer", "Paradise", "Stars", "Fire"]

    tracks = []
    for i in range(n_rows):
        if i % 3 == 0:
            tracks.append(f"{np.random.choice(adj)} {np.random.choice(nouns)}")
        elif i % 3 == 1:
            tracks.append(f"{np.random.choice(nouns)} {i % 100}")
        else:
            tracks.append(f"Track {i}")
    return tracks


def generate_albums(n_rows, seed=42):
    """Generate album names"""
    np.random.seed(seed + 4)
    types = ["Album", "EP", "Single", "Deluxe", "Mixtape"]
    return [
        f"{np.random.choice(types, p=[0.4, 0.15, 0.3, 0.1, 0.05])} {i % 10000}"
        for i in range(n_rows)
    ]


def generate_dates(n_rows, seed=42):
    """Generate release dates 2010-2023"""
    np.random.seed(seed + 5)
    start = datetime(2010, 1, 1)
    days = (datetime(2023, 12, 31) - start).days
    random_days = np.random.beta(2, 1, n_rows) * days
    return [start + timedelta(days=int(d)) for d in random_days]


def generate_followers(n_rows, seed=42):
    """Generate playlist followers"""
    np.random.seed(seed + 6)
    followers = np.random.lognormal(8, 2, n_rows).astype(int)
    return np.clip(followers, 1, 10000000)


def generate_dataset(
    n_rows=5000000, output_path="spotify_data.csv", chunk_size=1000000
):
    """Main generation function with error handling"""
    try:
        print("=" * 80)
        print("🎵 SYNTHETIC SPOTIFY DATASET GENERATOR")
        print("=" * 80)
        print(f"Target rows: {n_rows:,}")
        print(f"Estimated size: ~{(n_rows * 300 / 1024**3):.2f} GB")
        print(f"Chunk size: {chunk_size:,}")
        print("")

        # Validate inputs
        if n_rows <= 0:
            raise ValueError(f"Invalid n_rows: {n_rows}. Must be positive.")
        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}. Must be positive.")
        if chunk_size > n_rows:
            print(
                f"⚠️  Warning: chunk_size ({chunk_size:,}) > n_rows ({n_rows:,}). Adjusting..."
            )
            chunk_size = n_rows

        output_path = Path(output_path)

        # Check disk space
        import shutil

        disk_stats = shutil.disk_usage(
            output_path.parent if output_path.parent.exists() else "."
        )
        available_gb = disk_stats.free / 1024**3
        required_gb = (n_rows * 300 / 1024**3) * 1.2  # 20% buffer

        if available_gb < required_gb:
            raise RuntimeError(
                f"Insufficient disk space! Need {required_gb:.1f} GB, "
                f"only {available_gb:.1f} GB available."
            )

        print(
            f"💾 Disk space check: {available_gb:.1f} GB available (need {required_gb:.1f} GB) ✅"
        )

        # Create output directory
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Cannot create output directory {output_path.parent}: {e}"
            )

        # Check write permissions
        test_file = output_path.parent / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise RuntimeError(f"No write permission in {output_path.parent}: {e}")

        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        chunks_completed = 0

        for chunk_idx in range(n_chunks):
            try:
                start_row = chunk_idx * chunk_size
                end_row = min((chunk_idx + 1) * chunk_size, n_rows)
                chunk_rows = end_row - start_row

                print(f"\n📦 Chunk {chunk_idx + 1}/{n_chunks} ({chunk_rows:,} rows)")

                # Generate all data with error handling
                try:
                    audio = generate_realistic_audio_features(
                        chunk_rows, seed=42 + chunk_idx
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to generate audio features: {e}")

                try:
                    df = pd.DataFrame(
                        {
                            "track_id": [
                                f"track_{start_row + i}" for i in range(chunk_rows)
                            ],
                            "track_name": generate_tracks(
                                chunk_rows, seed=42 + chunk_idx
                            ),
                            "artist_name": generate_artists(
                                chunk_rows, seed=42 + chunk_idx
                            ),
                            "album_name": generate_albums(
                                chunk_rows, seed=42 + chunk_idx
                            ),
                            "release_date": generate_dates(
                                chunk_rows, seed=42 + chunk_idx
                            ),
                            "playlist_name": [
                                f"Playlist {i}" for i in range(chunk_rows)
                            ],
                            "playlist_category": generate_categories(
                                chunk_rows, seed=42 + chunk_idx
                            ),
                            "num_followers": generate_followers(
                                chunk_rows, seed=42 + chunk_idx
                            ),
                            **audio,
                        }
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to create DataFrame: {e}")

                # Adjust by category
                try:
                    df = adjust_by_category(df)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not adjust by category: {e}")
                    # Continue anyway

                # Generate popularity
                try:
                    df["popularity_score"] = generate_popularity(df)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not generate popularity: {e}")
                    df["popularity_score"] = 50  # Default value

                # Validate DataFrame
                if len(df) != chunk_rows:
                    raise RuntimeError(
                        f"DataFrame size mismatch! Expected {chunk_rows}, got {len(df)}"
                    )

                if df.isnull().all().any():
                    null_cols = df.columns[df.isnull().all()].tolist()
                    raise RuntimeError(f"Columns with all NaN values: {null_cols}")

                # Save with error handling
                try:
                    mode = "w" if chunk_idx == 0 else "a"
                    header = chunk_idx == 0
                    df.to_csv(output_path, index=False, mode=mode, header=header)
                except Exception as e:
                    raise RuntimeError(f"Failed to write CSV: {e}")

                print(f"  ✅ Saved chunk {chunk_idx + 1}")
                chunks_completed += 1

                # Memory cleanup
                del df

            except Exception as e:
                print(f"  ❌ Error in chunk {chunk_idx + 1}: {e}")
                if chunks_completed == 0:
                    # If first chunk failed, can't continue
                    raise RuntimeError(f"Failed on first chunk. Cannot continue: {e}")
                else:
                    # Some chunks succeeded, ask user
                    print(
                        f"\n⚠️  {chunks_completed}/{n_chunks} chunks completed before error."
                    )
                    print(f"Partial file saved to: {output_path}")
                    raise

        # Verify final file
        if not output_path.exists():
            raise RuntimeError(f"Output file not created: {output_path}")

        file_size_gb = output_path.stat().st_size / 1024**3

        if file_size_gb < 0.001:  # Less than 1 MB
            raise RuntimeError(f"Output file suspiciously small: {file_size_gb:.6f} GB")

        # Count rows in final file
        try:
            with open(output_path, "r") as f:
                actual_rows = sum(1 for _ in f) - 1  # Subtract header

            if actual_rows != n_rows:
                print(f"\n⚠️  Warning: Expected {n_rows:,} rows but got {actual_rows:,}")
        except Exception as e:
            print(f"\n⚠️  Could not verify row count: {e}")

        print("\n" + "=" * 80)
        print(f"✅ COMPLETE!")
        print(f"📁 File: {output_path}")
        print(f"📊 Rows: {n_rows:,}")
        print(f"💾 Size: {file_size_gb:.2f} GB")
        print("=" * 80)

        return output_path

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user!")
        if output_path.exists():
            print(f"Partial file saved to: {output_path}")
            print(f"You can delete it with: rm {output_path}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        if output_path.exists():
            print(f"\nPartial/corrupted file may exist at: {output_path}")
            print(f"Consider deleting it: rm {output_path}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Spotify dataset")
    parser.add_argument(
        "--rows", type=int, default=5000000, help="Number of rows (default: 5M)"
    )
    parser.add_argument(
        "--output", type=str, default="data/spotify_data.csv", help="Output file path"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000000, help="Chunk size (default: 1M)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.rows <= 0:
        print(f"❌ Error: --rows must be positive (got {args.rows})")
        sys.exit(1)

    if args.chunk_size <= 0:
        print(f"❌ Error: --chunk-size must be positive (got {args.chunk_size})")
        sys.exit(1)

    if args.rows > 100000000:  # 100M limit
        print(f"⚠️  Warning: Generating {args.rows:,} rows is very large!")
        print(f"   Estimated size: ~{(args.rows * 300 / 1024**3):.1f} GB")
        print(f"   Estimated time: ~{(args.rows / 100000):.0f} minutes")
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("Cancelled by user.")
            sys.exit(0)

    try:
        result = generate_dataset(
            n_rows=args.rows, output_path=args.output, chunk_size=args.chunk_size
        )
        print(f"\n✅ Success! Data saved to: {result}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        sys.exit(1)
