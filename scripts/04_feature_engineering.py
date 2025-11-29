"""
Feature Engineering for Spotify Data
Creates new features for ML models
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def load_data(file_path):
    """Load cleaned data (prefer parquet if available)"""
    file_path = Path(file_path)
    parquet_path = file_path.with_suffix('.parquet')
    
    print(f"\n📂 Loading data...")
    
    if parquet_path.exists():
        print(f"   Loading from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"   Loading from CSV: {file_path}")
        df = pd.read_csv(file_path)
    
    print(f"✅ Loaded {len(df):,} rows")
    return df

def create_features(df):
    """Create new features"""
    print("\n" + "="*60)
    print("Feature Engineering")
    print("="*60)
    
    # 1. Duration features
    if 'track_length_min' in df.columns:
        print("\n1. Creating duration features...")
        df['is_long_track'] = (df['track_length_min'] > 5).astype(int)
        df['is_short_track'] = (df['track_length_min'] < 2).astype(int)
    elif 'duration_ms' in df.columns:
        print("\n1. Creating duration features from duration_ms...")
        df['duration_minutes'] = df['duration_ms'] / 60000
        df['is_long_track'] = (df['duration_minutes'] > 5).astype(int)
        df['is_short_track'] = (df['duration_minutes'] < 2).astype(int)
    
    # 2. Mood/Energy categories
    print("\n2. Creating mood/energy categories...")
    
    if 'valence' in df.columns:
        df['mood_category'] = pd.cut(df['valence'], 
                                     bins=[0, 0.33, 0.66, 1.0],
                                     labels=['sad', 'neutral', 'happy'])
    
    if 'energy' in df.columns:
        df['energy_category'] = pd.cut(df['energy'],
                                       bins=[0, 0.33, 0.66, 1.0],
                                       labels=['chill', 'moderate', 'energetic'])
    
    # 3. Danceability categories
    if 'danceability' in df.columns:
        df['danceability_category'] = pd.cut(df['danceability'],
                                             bins=[0, 0.5, 0.7, 1.0],
                                             labels=['low', 'medium', 'high'])
    
    # 4. Popularity tiers
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        print("\n3. Creating popularity tiers...")
        df['popularity_tier'] = pd.cut(df[pop_col],
                                       bins=[0, 30, 60, 100],
                                       labels=['low', 'medium', 'high'])
        df['is_popular'] = (df[pop_col] > 60).astype(int)
    
    # 5. Audio feature interactions
    print("\n4. Creating feature interactions...")
    
    if 'energy' in df.columns and 'danceability' in df.columns:
        df['party_score'] = (df['energy'] + df['danceability']) / 2
    
    if 'acousticness' in df.columns and 'instrumentalness' in df.columns:
        df['chill_score'] = (df['acousticness'] + df['instrumentalness']) / 2
    
    if 'valence' in df.columns and 'energy' in df.columns:
        df['hype_score'] = (df['valence'] + df['energy']) / 2
    
    if 'instrumentalness' in df.columns:
        df['is_instrumental'] = (df['instrumentalness'] > 0.5).astype(int)
    
    # 6. Tempo categories
    if 'tempo' in df.columns:
        print("\n5. Creating tempo categories...")
        df['tempo_category'] = pd.cut(df['tempo'],
                                      bins=[0, 90, 120, 150, 300],
                                      labels=['slow', 'moderate', 'fast', 'very_fast'])
    
    # 7. Playlist follower features (if available)
    if 'num_followers' in df.columns:
        print("\n6. Creating follower-based features...")
        df['high_follower_playlist'] = (df['num_followers'] > df['num_followers'].median()).astype(int)
    
    # 8. Artist/Track features (aggregations) - SKIP FOR LARGE DATASETS
    # This requires too much memory for 62M rows
    # if 'artist_name' in df.columns:
    #     print("\n7. Creating artist-level features...")
    #     pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    #     if pop_col in df.columns:
    #         artist_stats = df.groupby('artist_name').agg({
    #             pop_col: 'mean',
    #             'track_name': 'count'
    #         }).rename(columns={pop_col: 'artist_avg_popularity',
    #                           'track_name': 'artist_track_count'})
    #         df = df.merge(artist_stats, on='artist_name', how='left')
    print("\n7. Skipping artist-level features (too memory intensive for 62M rows)")
    
    # 9. Playlist category features (if available)  
    # SKIP: One-hot encoding 62M rows creates too many columns
    # if 'playlist_category' in df.columns:
    #     print("\n8. Creating playlist category features...")
    #     category_dummies = pd.get_dummies(df['playlist_category'], prefix='category')
    #     df = pd.concat([df, category_dummies], axis=1)
    print("\n8. Skipping one-hot encoding (too memory intensive)")
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Original columns: {df.shape[1] - 20}")  # Approximate
    print(f"   Total columns now: {df.shape[1]}")
    
    return df

def save_features(df, output_path):
    """Save data with engineered features"""
    print(f"\n💾 Saving feature-engineered data...")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save ONLY Parquet (CSV too slow for 62M rows!)
    parquet_path = output_path.with_suffix('.parquet')
    print(f"   Writing to: {parquet_path}")
    df.to_parquet(parquet_path, index=False)
    
    print(f"✅ Saved as Parquet: {parquet_path}")
    print(f"   (Skipped CSV - too slow for this size, use parquet!)")

def main():
    parser = argparse.ArgumentParser(description='Feature engineering for Spotify data')
    parser.add_argument('--input', default='data/processed/spotify_cleaned.csv',
                       help='Input cleaned data file')
    parser.add_argument('--output', default='data/processed/spotify_features.csv',
                       help='Output file with engineered features')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Spotify Feature Engineering")
    print("=" * 60)
    
    # Load data
    df = load_data(args.input)
    
    # Create features
    df_features = create_features(df)
    
    # Save
    save_features(df_features, args.output)
    
    print("\n" + "=" * 60)
    print("✅ Feature Engineering Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Exploratory analysis: jupyter notebook notebooks/01_exploratory_analysis.ipynb")
    print("2. Train ML models: python ml_models/popularity_model.py")
    
    return 0

if __name__ == "__main__":
    exit(main())



