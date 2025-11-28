"""
Create Comprehensive Visualizations for Spotify Data Analysis
Generates charts, graphs, and dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """Load processed data"""
    file_path = Path(file_path)
    parquet_path = file_path.with_suffix('.parquet')
    
    print(f"\n📂 Loading data...")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"✅ Loaded {len(df):,} rows")
    return df

def create_audio_features_dashboard(df, output_dir):
    """Create comprehensive audio features dashboard"""
    print("\n📊 Creating audio features dashboard...")
    
    audio_features = ['energy', 'danceability', 'valence', 'acousticness', 
                     'instrumentalness', 'liveness', 'speechiness']
    available_features = [f for f in audio_features if f in df.columns]
    
    n_features = len(available_features)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_features):
        # Distribution histogram
        axes[idx].hist(df[feature], bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{feature.title()} Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(feature.title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(df[feature].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df[feature].mean():.3f}')
        axes[idx].axvline(df[feature].median(), color='blue', linestyle='--',
                         label=f'Median: {df[feature].median():.3f}')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    # Tempo
    if 'tempo' in df.columns:
        axes[7].hist(df['tempo'], bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[7].set_title('Tempo Distribution', fontsize=12, fontweight='bold')
        axes[7].set_xlabel('Tempo (BPM)')
        axes[7].set_ylabel('Frequency')
        axes[7].grid(True, alpha=0.3)
    
    # Loudness
    if 'loudness' in df.columns:
        axes[8].hist(df['loudness'], bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[8].set_title('Loudness Distribution', fontsize=12, fontweight='bold')
        axes[8].set_xlabel('Loudness (dB)')
        axes[8].set_ylabel('Frequency')
        axes[8].grid(True, alpha=0.3)
    
    plt.suptitle('Spotify Audio Features - Complete Overview', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'audio_features_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: audio_features_dashboard.png")

def create_correlation_heatmap(df, output_dir):
    """Create correlation heatmap"""
    print("\n📊 Creating correlation heatmap...")
    
    numeric_features = ['energy', 'danceability', 'valence', 'acousticness',
                       'instrumentalness', 'liveness', 'speechiness', 
                       'tempo', 'loudness', 'popularity']
    available_features = [f for f in numeric_features if f in df.columns]
    
    corr = df[available_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8})
    plt.title('Audio Features Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: correlation_heatmap.png")

def create_popularity_analysis(df, output_dir):
    """Create popularity analysis visualizations"""
    print("\n📊 Creating popularity analysis...")
    
    if 'popularity' not in df.columns:
        print("   ⚠️  Skipping: 'popularity' column not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Popularity distribution
    axes[0, 0].hist(df['popularity'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 0].set_title('Popularity Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Popularity Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['popularity'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["popularity"].mean():.1f}')
    axes[0, 0].legend()
    
    # 2. Popularity vs Energy
    if 'energy' in df.columns:
        axes[0, 1].scatter(df['energy'], df['popularity'], alpha=0.3)
        axes[0, 1].set_title('Popularity vs Energy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Energy')
        axes[0, 1].set_ylabel('Popularity')
    
    # 3. Popularity vs Danceability
    if 'danceability' in df.columns:
        axes[1, 0].scatter(df['danceability'], df['popularity'], alpha=0.3, color='blue')
        axes[1, 0].set_title('Popularity vs Danceability', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Danceability')
        axes[1, 0].set_ylabel('Popularity')
    
    # 4. Popularity by tier
    if 'popularity_tier' in df.columns:
        tier_counts = df['popularity_tier'].value_counts()
        colors = ['red', 'orange', 'green']
        axes[1, 1].bar(range(len(tier_counts)), tier_counts.values, 
                       color=colors[:len(tier_counts)])
        axes[1, 1].set_xticks(range(len(tier_counts)))
        axes[1, 1].set_xticklabels(tier_counts.index)
        axes[1, 1].set_title('Songs by Popularity Tier', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Tier')
        axes[1, 1].set_ylabel('Count')
        
        for i, count in enumerate(tier_counts.values):
            pct = (count / len(df)) * 100
            axes[1, 1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'popularity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: popularity_analysis.png")

def create_temporal_trends(df, output_dir):
    """Create temporal trends visualizations"""
    print("\n📊 Creating temporal trends...")
    
    if 'year' not in df.columns:
        print("   ⚠️  Skipping: 'year' column not found")
        return
    
    # Filter reasonable years
    df_years = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
    
    if len(df_years) == 0:
        print("   ⚠️  No data in valid year range")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    features_to_plot = [
        ('energy', 'Energy', 'orange'),
        ('danceability', 'Danceability', 'blue'),
        ('valence', 'Mood (Valence)', 'green'),
        ('tempo', 'Tempo (BPM)', 'red')
    ]
    
    for idx, (feature, title, color) in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        if feature in df_years.columns:
            yearly_avg = df_years.groupby('year')[feature].mean()
            ax.plot(yearly_avg.index, yearly_avg.values, marker='o', 
                   linewidth=2, markersize=4, color=color)
            ax.set_title(f'Average {title} Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Music Trends Over Time (1990-2024)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: temporal_trends.png")

def create_category_comparison(df, output_dir):
    """Compare different music categories"""
    print("\n📊 Creating category comparison...")
    
    if 'energy_category' not in df.columns:
        print("   ⚠️  Skipping: 'energy_category' column not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Compare different features across energy categories
    comparisons = [
        ('energy', 'Energy', 'orange'),
        ('danceability', 'Danceability', 'blue'),
        ('valence', 'Mood (Valence)', 'green'),
        ('tempo', 'Tempo (BPM)', 'red')
    ]
    
    for idx, (feature, title, color) in enumerate(comparisons):
        ax = axes[idx // 2, idx % 2]
        if feature in df.columns:
            category_avg = df.groupby('energy_category')[feature].mean()
            category_avg.plot(kind='bar', ax=ax, color=color)
            ax.set_title(f'Average {title} by Energy Category', fontweight='bold')
            ax.set_ylabel(title)
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=0)
    
    plt.suptitle('Music Categories Analysis (Chill vs Energetic)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: category_comparison.png")

def create_top_artists_chart(df, output_dir):
    """Create top artists visualization"""
    print("\n📊 Creating top artists chart...")
    
    if 'artist_name' not in df.columns:
        print("   ⚠️  Skipping: 'artist_name' column not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 20 artists by track count
    top_artists = df['artist_name'].value_counts().head(20)
    axes[0].barh(range(len(top_artists)), top_artists.values, color='teal')
    axes[0].set_yticks(range(len(top_artists)))
    axes[0].set_yticklabels(top_artists.index, fontsize=9)
    axes[0].set_title('Top 20 Artists by Track Count', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Tracks')
    axes[0].invert_yaxis()
    
    # Most popular artists (with at least 5 tracks)
    if 'popularity' in df.columns:
        artist_stats = df.groupby('artist_name').agg({
            'popularity': 'mean',
            'track_name': 'count'
        }).rename(columns={'track_name': 'track_count'})
        
        popular_artists = artist_stats[artist_stats['track_count'] >= 5]\
                         .sort_values('popularity', ascending=False).head(20)
        
        axes[1].barh(range(len(popular_artists)), popular_artists['popularity'].values, 
                    color='purple')
        axes[1].set_yticks(range(len(popular_artists)))
        axes[1].set_yticklabels(popular_artists.index, fontsize=9)
        axes[1].set_title('Top 20 Most Popular Artists (5+ tracks)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Average Popularity')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_artists.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: top_artists.png")

def create_summary_report(df, output_dir):
    """Create a text summary report"""
    print("\n📝 Creating summary report...")
    
    report = []
    report.append("="*70)
    report.append("SPOTIFY DATA ANALYSIS - SUMMARY REPORT")
    report.append("="*70)
    
    # Dataset overview
    report.append("\n1. DATASET OVERVIEW")
    report.append(f"   Total Tracks: {len(df):,}")
    if 'artist_name' in df.columns:
        report.append(f"   Unique Artists: {df['artist_name'].nunique():,}")
    if 'album_name' in df.columns:
        report.append(f"   Unique Albums: {df['album_name'].nunique():,}")
    
    # Audio features
    report.append("\n2. AVERAGE AUDIO FEATURES")
    audio_features = ['energy', 'danceability', 'valence', 'tempo', 'loudness']
    for feature in audio_features:
        if feature in df.columns:
            report.append(f"   {feature.title()}: {df[feature].mean():.3f}")
    
    # Popularity
    if 'popularity' in df.columns:
        report.append("\n3. POPULARITY STATISTICS")
        report.append(f"   Mean Popularity: {df['popularity'].mean():.2f}")
        report.append(f"   Median Popularity: {df['popularity'].median():.2f}")
        report.append(f"   Popular Songs (>60): {(df['popularity'] > 60).sum():,} "
                     f"({(df['popularity'] > 60).mean()*100:.1f}%)")
    
    # Trends
    if 'year' in df.columns:
        df_years = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
        report.append("\n4. TEMPORAL TRENDS (1990-2024)")
        report.append(f"   Tracks in this period: {len(df_years):,}")
        if 'energy' in df_years.columns:
            recent = df_years[df_years['year'] >= 2020]['energy'].mean()
            old = df_years[df_years['year'] < 2000]['energy'].mean()
            change = ((recent - old) / old) * 100
            report.append(f"   Energy change: {change:+.1f}% (2020+ vs pre-2000)")
    
    report.append("\n" + "="*70)
    
    # Write report
    report_text = "\n".join(report)
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(report_text)
    
    print("   ✅ Saved: analysis_summary.txt")
    print("\n" + report_text)

def main():
    print("="*60)
    print("Spotify Data Visualization Generator")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Load data
    data_path = Path('../data/processed/spotify_features.csv')
    df = load_data(data_path)
    
    # Create output directory
    output_dir = Path('../visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    create_audio_features_dashboard(df, output_dir)
    create_correlation_heatmap(df, output_dir)
    create_popularity_analysis(df, output_dir)
    create_temporal_trends(df, output_dir)
    create_category_comparison(df, output_dir)
    create_top_artists_chart(df, output_dir)
    create_summary_report(df, output_dir)
    
    print("\n" + "="*60)
    print("✅ All Visualizations Complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  📊 {file.name}")
    for file in sorted(output_dir.glob('*.txt')):
        print(f"  📝 {file.name}")

if __name__ == "__main__":
    main()



