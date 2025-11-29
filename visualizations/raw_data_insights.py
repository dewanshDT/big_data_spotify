"""
Spotify Data Analysis - Using RAW DATA (No Processing)
This bypasses all feature engineering to use original Spotify values
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_raw_data(sample_size=2000000):
    """Load RAW data directly from S3-downloaded CSV"""
    print(f"\n📂 Loading RAW data (sample: {sample_size:,})...")
    
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / 'data/raw/spotify_data.csv',
        Path('/home/ubuntu/spotify_analysis/data/raw/spotify_data.csv'),
        Path(__file__).parent.parent / 'spotify_data.csv',
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if not data_path:
        raise FileNotFoundError("Could not find spotify_data.csv. Please check data location.")
    
    print(f"   Reading from: {data_path}")
    
    # Read with chunking for memory efficiency
    chunks = []
    chunk_size = 500000
    rows_read = 0
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size, 
                             on_bad_lines='skip', engine='python'):
        chunks.append(chunk)
        rows_read += len(chunk)
        print(f"   Loaded {rows_read:,} rows...", end='\r')
        
        if rows_read >= sample_size:
            break
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Sample if we got more than needed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"\n✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"\n📋 Available columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    return df

def analyze_data_quality(df):
    """Check if data has real variation"""
    print("\n🔍 DATA QUALITY CHECK")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols[:10]:  # Check first 10 numeric columns
        unique_count = df[col].nunique()
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        
        print(f"\n{col}:")
        print(f"  Unique values: {unique_count:,}")
        print(f"  Range: [{min_val:.3f} - {max_val:.3f}]")
        print(f"  Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        
        # Check if data is suspiciously uniform
        if std_val < 0.01 and col not in ['year']:
            print(f"  ⚠️  WARNING: Very low variation!")
        elif unique_count < 10:
            print(f"  ⚠️  WARNING: Very few unique values!")
        else:
            print(f"  ✅ Good variation")

def create_real_insights(df, output_dir):
    """Create insights using whatever columns exist"""
    print("\n📊 CREATING INSIGHTS FROM RAW DATA")
    print("="*60)
    
    # Auto-detect audio feature columns
    possible_features = {
        'energy': ['energy', 'nrgy'],
        'danceability': ['danceability', 'dance'],
        'valence': ['valence', 'mood', 'happiness'],
        'tempo': ['tempo', 'bpm'],
        'acousticness': ['acousticness', 'acoustic'],
        'loudness': ['loudness', 'loud'],
        'speechiness': ['speechiness', 'speech'],
        'instrumentalness': ['instrumentalness', 'instrumental'],
        'liveness': ['liveness', 'live'],
    }
    
    feature_map = {}
    for feature, possible_names in possible_features.items():
        for name in possible_names:
            if name in df.columns:
                feature_map[feature] = name
                break
    
    print(f"\n✅ Found {len(feature_map)} audio features:")
    for feature, col_name in feature_map.items():
        print(f"   • {feature} → {col_name}")
    
    if len(feature_map) < 3:
        print("\n⚠️  Not enough audio features found in data!")
        print("\nAvailable columns:")
        print(df.columns.tolist())
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot distributions for each feature
    for feature, col_name in list(feature_map.items())[:9]:
        ax = axes[plot_idx]
        
        data = df[col_name].dropna()
        
        # Histogram + KDE
        ax.hist(data, bins=50, alpha=0.6, color='skyblue', edgecolor='black', density=True)
        data.plot(kind='density', ax=ax, color='red', linewidth=2)
        
        ax.set_title(f'{feature.title()} Distribution', fontweight='bold', fontsize=12)
        ax.set_xlabel(feature.title())
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nRange: [{data.min():.2f}, {data.max():.2f}]'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 9):
        axes[idx].set_visible(False)
    
    plt.suptitle('🎵 RAW Spotify Data - Feature Distributions', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_data_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved: raw_data_distributions.png")
    
    # Correlation matrix if we have enough features
    if len(feature_map) >= 4:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cols_to_correlate = [feature_map[f] for f in list(feature_map.keys())[:8]]
        corr = df[cols_to_correlate].corr()
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=2,
                   cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Feature Correlations (Raw Data)', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'raw_data_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: raw_data_correlations.png")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=2000000,
                       help='Number of rows to sample (default: 2M)')
    args = parser.parse_args()
    
    print("="*80)
    print("🎵 SPOTIFY RAW DATA ANALYSIS")
    print("="*80)
    
    try:
        # Load raw data
        df = load_raw_data(sample_size=args.sample)
        
        # Analyze data quality
        analyze_data_quality(df)
        
        # Create visualizations
        output_dir = Path(__file__).parent
        create_real_insights(df, output_dir)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

