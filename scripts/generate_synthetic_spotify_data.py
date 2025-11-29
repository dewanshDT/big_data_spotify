"""
Generate Realistic Synthetic Spotify Dataset
Creates a large-scale dataset with realistic distributions and correlations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logging

def generate_realistic_audio_features(n_rows, seed=42):
    """
    Generate realistic audio features with proper distributions and correlations
    Based on patterns observed in real Spotify data
    """
    np.random.seed(seed)
    logger = setup_logging()
    
    logger.info(f"Generating {n_rows:,} tracks with realistic audio features...")
    
    # 1. ENERGY - Beta distribution (slightly right-skewed)
    # Real Spotify: mean ~0.65, tends toward higher energy
    energy = np.random.beta(2.5, 2, n_rows)
    
    # 2. LOUDNESS - Normal distribution (in dB, typically -60 to 0)
    # Strong positive correlation with energy
    loudness = -60 + energy * 55 + np.random.normal(0, 3, n_rows)
    loudness = np.clip(loudness, -60, 0)
    
    # 3. DANCEABILITY - Beta distribution (slightly right-skewed)
    # Real Spotify: mean ~0.60
    danceability = np.random.beta(2.8, 2.5, n_rows)
    
    # 4. VALENCE (mood: sad to happy) - Nearly uniform but slight right skew
    # Moderate correlation with danceability
    valence = 0.3 * danceability + 0.7 * np.random.beta(2, 2, n_rows)
    valence = np.clip(valence, 0, 1)
    
    # 5. TEMPO - Normal distribution around 120 BPM
    tempo = np.random.normal(120, 28, n_rows)
    tempo = np.clip(tempo, 40, 220)
    
    # 6. ACOUSTICNESS - Exponential-like (most songs have low acousticness)
    # Strong negative correlation with energy
    acousticness = np.random.beta(1.5, 5, n_rows)
    acousticness = 0.7 * (1 - energy) + 0.3 * acousticness
    acousticness = np.clip(acousticness, 0, 1)
    
    # 7. INSTRUMENTALNESS - Heavy right skew (most have vocals)
    instrumentalness = np.random.beta(1, 10, n_rows)
    
    # 8. SPEECHINESS - Heavy right skew (most songs have little speech)
    speechiness = np.random.beta(1, 8, n_rows)
    
    # 9. LIVENESS - Heavy right skew (most are studio recordings)
    liveness = np.random.beta(1.2, 6, n_rows)
    
    # 10. DURATION (in ms) - Right-skewed around 3-4 minutes
    duration_ms = np.random.gamma(8, 25000, n_rows)
    duration_ms = np.clip(duration_ms, 30000, 600000)  # 30s to 10min
    
    # 11. KEY - Discrete (0-11 for musical keys)
    key = np.random.randint(0, 12, n_rows)
    
    # 12. MODE - Binary (0=minor, 1=major)
    # Major keys slightly more common, correlated with valence
    mode_prob = 0.45 + 0.2 * valence
    mode = (np.random.random(n_rows) < mode_prob).astype(int)
    
    # 13. TIME_SIGNATURE - Mostly 4/4 (some 3/4, rare others)
    time_sig_probs = [0.02, 0.08, 0.85, 0.05]  # 2, 3, 4, 5
    time_signature = np.random.choice([2, 3, 4, 5], n_rows, p=time_sig_probs)
    
    logger.info("✅ Audio features generated with realistic distributions")
    
    return {
        'energy': energy,
        'loudness': loudness,
        'danceability': danceability,
        'valence': valence,
        'tempo': tempo,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'speechiness': speechiness,
        'liveness': liveness,
        'duration_ms': duration_ms,
        'key': key,
        'mode': mode,
        'time_signature': time_signature
    }

def generate_popularity_scores(audio_features, n_rows):
    """
    Generate popularity scores based on audio features
    Higher energy + danceability + valence = more popular (generally)
    """
    # Base popularity from features
    popularity = (
        25 * audio_features['energy'] +
        20 * audio_features['danceability'] +
        15 * audio_features['valence'] +
        10 * (1 - audio_features['acousticness']) +
        10 * (audio_features['loudness'] + 60) / 60
    )
    
    # Add randomness
    popularity += np.random.normal(0, 15, n_rows)
    
    # Apply power law (most songs unpopular, few very popular)
    popularity = np.power(popularity / 100, 1.5) * 100
    
    popularity = np.clip(popularity, 0, 100).astype(int)
    
    return popularity

def generate_playlist_categories(n_rows, seed=42):
    """
    Generate playlist categories with realistic distributions
    Different categories have different audio feature profiles
    """
    np.random.seed(seed + 1)
    
    categories = {
        'workout': 0.12,
        'party': 0.15,
        'chill': 0.18,
        'sleep': 0.08,
        'focus': 0.14,
        'study': 0.12,
        'pop': 0.10,
        'rock': 0.06,
        'indie': 0.05
    }
    
    cat_names = list(categories.keys())
    cat_probs = list(categories.values())
    
    return np.random.choice(cat_names, n_rows, p=cat_probs)

def adjust_features_by_category(df):
    """
    Adjust audio features to match category expectations
    """
    logger = setup_logging()
    logger.info("Adjusting features based on playlist categories...")
    
    # Workout: high energy, high tempo
    workout_mask = df['playlist_category'] == 'workout'
    df.loc[workout_mask, 'energy'] = np.clip(
        df.loc[workout_mask, 'energy'] + 0.15, 0, 1
    )
    df.loc[workout_mask, 'tempo'] = np.clip(
        df.loc[workout_mask, 'tempo'] + 15, 40, 220
    )
    
    # Sleep: low energy, low tempo, high acousticness
    sleep_mask = df['playlist_category'] == 'sleep'
    df.loc[sleep_mask, 'energy'] = np.clip(
        df.loc[sleep_mask, 'energy'] - 0.25, 0, 1
    )
    df.loc[sleep_mask, 'tempo'] = np.clip(
        df.loc[sleep_mask, 'tempo'] - 25, 40, 220
    )
    df.loc[sleep_mask, 'acousticness'] = np.clip(
        df.loc[sleep_mask, 'acousticness'] + 0.2, 0, 1
    )
    
    # Party: high energy, high danceability, high valence
    party_mask = df['playlist_category'] == 'party'
    df.loc[party_mask, 'energy'] = np.clip(
        df.loc[party_mask, 'energy'] + 0.12, 0, 1
    )
    df.loc[party_mask, 'danceability'] = np.clip(
        df.loc[party_mask, 'danceability'] + 0.15, 0, 1
    )
    df.loc[party_mask, 'valence'] = np.clip(
        df.loc[party_mask, 'valence'] + 0.12, 0, 1
    )
    
    # Chill: low-medium energy, higher acousticness
    chill_mask = df['playlist_category'] == 'chill'
    df.loc[chill_mask, 'energy'] = np.clip(
        df.loc[chill_mask, 'energy'] - 0.12, 0, 1
    )
    df.loc[chill_mask, 'acousticness'] = np.clip(
        df.loc[chill_mask, 'acousticness'] + 0.15, 0, 1
    )
    
    # Focus/Study: low-medium energy, low speechiness
    focus_mask = (df['playlist_category'] == 'focus') | (df['playlist_category'] == 'study')
    df.loc[focus_mask, 'energy'] = np.clip(
        df.loc[focus_mask, 'energy'] - 0.08, 0, 1
    )
    df.loc[focus_mask, 'speechiness'] = np.clip(
        df.loc[focus_mask, 'speechiness'] - 0.05, 0, 1
    )
    
    logger.info("✅ Features adjusted by category")
    return df

def generate_artist_names(n_rows, n_unique_artists=50000, seed=42):
    """
    Generate realistic artist names with power law distribution
    (Few artists have many tracks, most have few)
    """
    np.random.seed(seed + 2)
    
    # Power law for artist popularity
    artist_ids = np.arange(n_unique_artists)
    artist_weights = 1 / np.power(artist_ids + 1, 0.8)
    artist_weights /= artist_weights.sum()
    
    selected_artists = np.random.choice(
        artist_ids, 
        n_rows, 
        p=artist_weights,
        replace=True
    )
    
    # Generate artist names
    first_names = ['Drake', 'Taylor', 'Ed', 'Ariana', 'Post', 'Billie', 'The Weeknd', 
                   'Justin', 'Dua', 'Bad', 'Olivia', 'Travis', 'Kanye', 'Kendrick',
                   'Rihanna', 'Bruno', 'Eminem', 'Lil', 'DJ', 'Twenty']
    last_names = ['Swift', 'Sheeran', 'Grande', 'Malone', 'Eilish', 'Bieber', 'Lipa',
                  'Bunny', 'Rodrigo', 'Scott', 'West', 'Lamar', 'Mars', 'Nas X',
                  'Khalid', 'One Pilots', 'Uzi', 'Baby', 'Wayne', 'Cole']
    
    artist_names = []
    for artist_id in selected_artists:
        if artist_id < len(first_names) * len(last_names):
            idx1 = artist_id % len(first_names)
            idx2 = artist_id // len(first_names) % len(last_names)
            artist_names.append(f"{first_names[idx1]} {last_names[idx2]}")
        else:
            artist_names.append(f"Artist {artist_id}")
    
    return artist_names

def generate_track_names(n_rows, seed=42):
    """Generate track names"""
    np.random.seed(seed + 3)
    
    adjectives = ['Beautiful', 'Dark', 'Sweet', 'Lost', 'Perfect', 'Golden', 'Wild',
                  'Falling', 'Dancing', 'Dreaming', 'Broken', 'Electric', 'Midnight']
    nouns = ['Love', 'Dreams', 'Heart', 'Night', 'Summer', 'Paradise', 'Stars',
             'Fire', 'Rain', 'Ocean', 'Sky', 'Memories', 'Music']
    
    track_names = []
    for i in range(n_rows):
        if i % 4 == 0:
            name = f"{np.random.choice(adjectives)} {np.random.choice(nouns)}"
        elif i % 4 == 1:
            name = f"{np.random.choice(nouns)} {i % 100}"
        elif i % 4 == 2:
            name = f"Track {i}"
        else:
            name = f"{np.random.choice(adjectives)} {np.random.choice(adjectives)}"
        track_names.append(name)
    
    return track_names

def generate_album_names(n_rows, seed=42):
    """Generate album names"""
    np.random.seed(seed + 4)
    
    album_types = ['Album', 'EP', 'Single', 'Deluxe', 'Mixtape']
    
    album_names = []
    for i in range(n_rows):
        album_type = np.random.choice(album_types, p=[0.4, 0.15, 0.3, 0.1, 0.05])
        album_names.append(f"{album_type} {i % 10000}")
    
    return album_names

def generate_playlist_metadata(n_rows, seed=42):
    """Generate playlist names and metadata"""
    np.random.seed(seed + 5)
    
    # Playlist names by category
    playlist_templates = {
        'workout': ['Workout Mix', 'Gym Motivation', 'Running Tracks', 'Beast Mode'],
        'party': ['Party Hits', 'Turn Up', 'Dance Floor', 'Club Bangers'],
        'chill': ['Chill Vibes', 'Relaxing Music', 'Easy Listening', 'Laid Back'],
        'sleep': ['Sleep Sounds', 'Bedtime Music', 'Deep Sleep', 'Calm Night'],
        'focus': ['Focus Flow', 'Deep Focus', 'Concentration', 'Work Music'],
        'study': ['Study Beats', 'Study Time', 'Study Sessions', 'Exam Prep'],
        'pop': ['Pop Hits', 'Top Pop', 'Pop Anthems', 'Pop Chart'],
        'rock': ['Rock Classics', 'Rock On', 'Alternative Rock', 'Rock Legends'],
        'indie': ['Indie Vibes', 'Indie Mix', 'Alternative Indie', 'Indie Gems']
    }
    
    return [f"Playlist {i}" for i in range(n_rows)]

def generate_release_dates(n_rows, seed=42):
    """Generate realistic release dates (2010-2023)"""
    np.random.seed(seed + 6)
    
    # More recent songs are more common
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate with bias toward recent years
    days_range = (end_date - start_date).days
    
    # Power law: more recent = more common
    random_days = np.random.beta(2, 1, n_rows) * days_range
    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    return dates

def generate_follower_counts(n_rows, seed=42):
    """Generate realistic playlist follower counts"""
    np.random.seed(seed + 7)
    
    # Log-normal distribution (most playlists have few followers)
    followers = np.random.lognormal(8, 2, n_rows).astype(int)
    followers = np.clip(followers, 1, 10000000)
    
    return followers

def generate_dataset(n_rows=50000000, output_path=None, chunk_size=5000000):
    """
    Generate complete synthetic Spotify dataset
    
    Args:
        n_rows: Total number of tracks to generate (default 50M for ~12GB)
        output_path: Where to save the CSV
        chunk_size: Process in chunks to manage memory
    """
    logger = setup_logging()
    logger.info("="*80)
    logger.info("SYNTHETIC SPOTIFY DATASET GENERATOR")
    logger.info("="*80)
    logger.info(f"Target rows: {n_rows:,}")
    logger.info(f"Estimated size: ~{(n_rows * 300 / 1024**3):.1f} GB")
    logger.info(f"Processing in chunks of: {chunk_size:,}")
    logger.info("")
    
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'spotify_synthetic_data.csv'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate in chunks
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min((chunk_idx + 1) * chunk_size, n_rows)
        chunk_rows = end_row - start_row
        
        logger.info(f"\n📦 Chunk {chunk_idx + 1}/{n_chunks}")
        logger.info(f"   Generating rows {start_row:,} to {end_row:,} ({chunk_rows:,} rows)")
        
        # Generate all features
        audio_features = generate_realistic_audio_features(chunk_rows, seed=42 + chunk_idx)
        
        df_chunk = pd.DataFrame({
            'track_id': [f'track_{start_row + i}' for i in range(chunk_rows)],
            'track_name': generate_track_names(chunk_rows, seed=42 + chunk_idx),
            'artist_name': generate_artist_names(chunk_rows, seed=42 + chunk_idx),
            'album_name': generate_album_names(chunk_rows, seed=42 + chunk_idx),
            'release_date': generate_release_dates(chunk_rows, seed=42 + chunk_idx),
            'playlist_name': generate_playlist_metadata(chunk_rows, seed=42 + chunk_idx),
            'playlist_category': generate_playlist_categories(chunk_rows, seed=42 + chunk_idx),
            'num_followers': generate_follower_counts(chunk_rows, seed=42 + chunk_idx),
            **audio_features
        })
        
        # Adjust features based on category
        df_chunk = adjust_features_by_category(df_chunk)
        
        # Generate popularity based on final features
        df_chunk['popularity_score'] = generate_popularity_scores(
            df_chunk[['energy', 'danceability', 'valence', 'acousticness', 'loudness']].to_dict('list'),
            chunk_rows
        )
        
        # Save chunk
        if chunk_idx == 0:
            df_chunk.to_csv(output_path, index=False, mode='w')
        else:
            df_chunk.to_csv(output_path, index=False, mode='a', header=False)
        
        logger.info(f"   ✅ Chunk saved ({len(df_chunk):,} rows)")
        
        # Memory cleanup
        del df_chunk
        
    logger.info("\n" + "="*80)
    logger.info(f"✅ DATASET GENERATION COMPLETE!")
    logger.info(f"📁 Saved to: {output_path}")
    logger.info(f"📊 Total rows: {n_rows:,}")
    logger.info(f"💾 File size: {output_path.stat().st_size / 1024**3:.2f} GB")
    logger.info("="*80)
    
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic Spotify dataset')
    parser.add_argument('--rows', type=int, default=50000000,
                       help='Number of rows to generate (default: 50M for ~12GB)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--chunk-size', type=int, default=5000000,
                       help='Chunk size for processing (default: 5M)')
    
    args = parser.parse_args()
    
    generate_dataset(
        n_rows=args.rows,
        output_path=args.output,
        chunk_size=args.chunk_size
    )

if __name__ == "__main__":
    main()

