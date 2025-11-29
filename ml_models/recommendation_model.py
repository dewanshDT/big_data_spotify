"""
Music Recommendation System
Recommends similar tracks based on audio features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
import yaml

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """Load feature-engineered data (prefer parquet)"""
    file_path = Path(file_path)
    parquet_path = file_path.with_suffix('.parquet')
    
    print(f"\n📂 Loading data...")
    
    # Try parquet first (faster!)
    if parquet_path.exists():
        print(f"   Loading from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif file_path.exists():
        print(f"   Loading from CSV: {file_path}")
        df = pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Data file not found: {file_path} or {parquet_path}")
    
    print(f"✅ Loaded {len(df):,} rows")
    return df

def prepare_features(df):
    """Prepare features for clustering"""
    print("\n🔧 Preparing features...")
    
    # Select audio features for similarity (based on actual dataset)
    feature_cols = [
        'energy', 'danceability', 'valence', 'tempo',
        'acousticness', 'instrumentalness'
    ]
    
    # Filter columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"   Using {len(feature_cols)} features for recommendations")
    
    # Extract features
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    return X, feature_cols

def train_clustering_model(X, config):
    """Train K-Means clustering model"""
    print("\n🤖 Training clustering model...")
    
    n_clusters = config['ml_models']['recommendation']['n_clusters']
    random_state = config['ml_models']['random_state']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    print(f"✅ Created {n_clusters} clusters")
    
    # Print cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count:,} tracks")
    
    return kmeans, scaler, clusters, X_scaled

def get_recommendations(track_idx, df, X_scaled, clusters, n_recommendations=10):
    """Get song recommendations for a given track"""
    
    # Get track's cluster
    track_cluster = clusters[track_idx]
    
    # Find tracks in same cluster
    cluster_mask = clusters == track_cluster
    cluster_indices = np.where(cluster_mask)[0]
    
    # Calculate cosine similarity only within cluster (for efficiency)
    track_features = X_scaled[track_idx].reshape(1, -1)
    cluster_features = X_scaled[cluster_mask]
    
    similarities = cosine_similarity(track_features, cluster_features)[0]
    
    # Get top N similar tracks (excluding the track itself)
    similar_indices_in_cluster = similarities.argsort()[::-1][1:n_recommendations+1]
    similar_indices = cluster_indices[similar_indices_in_cluster]
    
    # Return recommendations
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    rec_cols = ['track_name', 'artist_name']
    if pop_col in df.columns:
        rec_cols.append(pop_col)
    recommendations = df.iloc[similar_indices][rec_cols].copy()
    recommendations['similarity_score'] = similarities[similar_indices_in_cluster]
    
    return recommendations

def demonstrate_recommendations(df, X_scaled, clusters, config):
    """Demonstrate recommendation system with examples"""
    print("\n" + "="*60)
    print("Recommendation System Demo")
    print("="*60)
    
    n_recommendations = config['ml_models']['recommendation']['n_recommendations']
    
    # Select random tracks to demonstrate
    demo_indices = np.random.choice(len(df), size=5, replace=False)
    
    for idx in demo_indices:
        track_name = df.iloc[idx]['track_name']
        artist_name = df.iloc[idx]['artist_name']
        
        print(f"\n🎵 If you like: '{track_name}' by {artist_name}")
        print(f"   You might also like:")
        
        recommendations = get_recommendations(idx, df, X_scaled, clusters, n_recommendations)
        
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"   {i}. '{row['track_name']}' by {row['artist_name']} "
                  f"(similarity: {row['similarity_score']:.3f})")

def save_recommendation_system(kmeans, scaler, feature_cols, df, clusters):
    """Save recommendation system components"""
    print("\n💾 Saving recommendation system...")
    
    output_dir = Path('../ml_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clustering model
    model_path = output_dir / 'recommendation_kmeans.pkl'
    joblib.dump(kmeans, model_path)
    print(f"✅ K-Means model saved to: {model_path}")
    
    # Save scaler
    scaler_path = output_dir / 'recommendation_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Save track database (for quick lookups)
    db_cols = ['track_name', 'artist_name', 'album_name']
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        db_cols.append(pop_col)
    track_db = df[db_cols].copy()
    track_db['cluster'] = clusters
    
    db_path = output_dir / 'track_database.pkl'
    joblib.dump(track_db, db_path)
    print(f"✅ Track database saved to: {db_path}")
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'n_clusters': kmeans.n_clusters,
        'model_type': 'K-Means Clustering + Cosine Similarity'
    }
    metadata_path = output_dir / 'recommendation_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"✅ Metadata saved to: {metadata_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=5000000, 
                       help='Number of rows to sample (default: 5M)')
    args = parser.parse_args()
    
    print("="*60)
    print("Spotify Music Recommendation System")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Load data
    data_path = '../data/processed/spotify_features.csv'
    df = load_data(data_path)
    
    # Sample for memory efficiency
    if args.sample and args.sample < len(df):
        print(f"\n🎲 Sampling {args.sample:,} rows from {len(df):,} (for memory efficiency)")
        df = df.sample(n=args.sample, random_state=42)
        print(f"✅ Using {len(df):,} rows for recommendations")
    
    # Prepare features
    X, feature_cols = prepare_features(df)
    
    # Train clustering model
    kmeans, scaler, clusters, X_scaled = train_clustering_model(X, config)
    
    # Demonstrate recommendations
    demonstrate_recommendations(df, X_scaled, clusters, config)
    
    # Save system
    save_recommendation_system(kmeans, scaler, feature_cols, df, clusters)
    
    print("\n" + "="*60)
    print("✅ Recommendation System Complete!")
    print("="*60)
    print("\nTo use the recommendation system:")
    print("1. Load the saved models")
    print("2. Find a track index")
    print("3. Call get_recommendations(track_idx, ...)")
    print("4. Get similar track suggestions!")

if __name__ == "__main__":
    main()



