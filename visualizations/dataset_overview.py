"""
Simple Dataset Overview Visualizations
Shows basic statistics and counts for the Spotify dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_data(sample_size=2000000):
    """Load data"""
    print(f"\n📂 Loading up to {sample_size:,} samples...")
    
    data_path = Path(__file__).parent.parent / 'data/processed/spotify_features.parquet'
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / 'data/raw/spotify_data.csv'
    
    print(f"   Loading from: {data_path}")
    
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, nrows=sample_size)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"✅ Loaded {len(df):,} rows\n")
    return df

def create_overview_charts(df, output_dir):
    """Create comprehensive dataset overview"""
    
    print("📊 Creating dataset overview visualizations...\n")
    
    # Figure 1: Dataset Composition (4 panels)
    fig = plt.figure(figsize=(20, 12))
    
    # Panel 1: Tracks per Category
    ax1 = plt.subplot(2, 3, 1)
    if 'playlist_category' in df.columns:
        cat_counts = df['playlist_category'].value_counts().head(10)
        colors = plt.cm.Set3(np.linspace(0, 1, len(cat_counts)))
        ax1.barh(range(len(cat_counts)), cat_counts.values, color=colors)
        ax1.set_yticks(range(len(cat_counts)))
        ax1.set_yticklabels(cat_counts.index)
        ax1.set_xlabel('Number of Tracks', fontweight='bold')
        ax1.set_title('📊 Tracks per Playlist Category\n(Top 10)', fontweight='bold', fontsize=12)
        ax1.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(cat_counts.values):
            ax1.text(v, i, f' {v:,}', va='center', fontweight='bold')
        
        print(f"✅ Tracks per category: {len(cat_counts)} categories")
    
    # Panel 2: Top 15 Artists
    ax2 = plt.subplot(2, 3, 2)
    if 'artist_name' in df.columns:
        artist_counts = df['artist_name'].value_counts().head(15)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(artist_counts)))
        ax2.barh(range(len(artist_counts)), artist_counts.values, color=colors)
        ax2.set_yticks(range(len(artist_counts)))
        ax2.set_yticklabels(artist_counts.index, fontsize=9)
        ax2.set_xlabel('Number of Tracks', fontweight='bold')
        ax2.set_title('🎤 Top 15 Artists by Track Count', fontweight='bold', fontsize=12)
        ax2.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(artist_counts.values):
            ax2.text(v, i, f' {v:,}', va='center', fontsize=8)
        
        print(f"✅ Top artists: {artist_counts.iloc[0]} tracks by {artist_counts.index[0]}")
    
    # Panel 3: Tracks per Year
    ax3 = plt.subplot(2, 3, 3)
    if 'release_date' in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        year_counts = df['year'].value_counts().sort_index()
        
        ax3.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=4)
        ax3.fill_between(year_counts.index, year_counts.values, alpha=0.3)
        ax3.set_xlabel('Year', fontweight='bold')
        ax3.set_ylabel('Number of Tracks', fontweight='bold')
        ax3.set_title('📅 Tracks per Release Year', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        print(f"✅ Release years: {year_counts.index.min():.0f} to {year_counts.index.max():.0f}")
    
    # Panel 4: Popularity Distribution
    ax4 = plt.subplot(2, 3, 4)
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        ax4.hist(df[pop_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(df[pop_col].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df[pop_col].median():.1f}')
        ax4.axvline(df[pop_col].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {df[pop_col].mean():.1f}')
        ax4.set_xlabel('Popularity Score', fontweight='bold')
        ax4.set_ylabel('Number of Tracks', fontweight='bold')
        ax4.set_title('⭐ Popularity Score Distribution', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        print(f"✅ Popularity: Mean={df[pop_col].mean():.1f}, Median={df[pop_col].median():.1f}")
    
    # Panel 5: Follower Distribution
    ax5 = plt.subplot(2, 3, 5)
    if 'num_followers' in df.columns:
        # Log scale for followers (power law distribution)
        log_followers = np.log10(df['num_followers'] + 1)
        ax5.hist(log_followers, bins=50, color='green', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Log10(Followers + 1)', fontweight='bold')
        ax5.set_ylabel('Number of Playlists', fontweight='bold')
        ax5.set_title('👥 Playlist Follower Distribution\n(Log Scale)', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add reference lines
        median_followers = df['num_followers'].median()
        ax5.axvline(np.log10(median_followers + 1), color='red', linestyle='--', 
                   label=f'Median: {median_followers:,.0f}')
        ax5.legend()
        
        print(f"✅ Followers: Median={median_followers:,.0f}, Max={df['num_followers'].max():,.0f}")
    
    # Panel 6: Track Duration Distribution
    ax6 = plt.subplot(2, 3, 6)
    if 'duration_ms' in df.columns:
        duration_min = df['duration_ms'] / 60000
        ax6.hist(duration_min, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax6.axvline(duration_min.median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {duration_min.median():.1f} min')
        ax6.set_xlabel('Track Duration (minutes)', fontweight='bold')
        ax6.set_ylabel('Number of Tracks', fontweight='bold')
        ax6.set_title('⏱️ Track Duration Distribution', fontweight='bold', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        print(f"✅ Duration: Median={duration_min.median():.1f} min, Mean={duration_min.mean():.1f} min")
    
    plt.suptitle('📊 Dataset Overview: Composition & Statistics', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'overview_1_dataset_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: overview_1_dataset_composition.png")

def create_feature_summaries(df, output_dir):
    """Create simple feature summary charts"""
    
    print("\n📊 Creating feature summary charts...\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    features = [
        ('energy', 'Energy Level', 'red'),
        ('danceability', 'Danceability', 'blue'),
        ('valence', 'Mood (Valence)', 'green'),
        ('tempo', 'Tempo (BPM)', 'orange'),
        ('acousticness', 'Acousticness', 'purple'),
        ('instrumentalness', 'Instrumentalness', 'brown')
    ]
    
    for idx, (feature, title, color) in enumerate(features):
        ax = axes[idx]
        
        if feature in df.columns:
            # Box plot with violin
            parts = ax.violinplot([df[feature].dropna()], positions=[0], showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            
            # Add statistics box
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            std_val = df[feature].std()
            
            stats_text = f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}'
            ax.text(0.5, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10)
            
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_ylabel(title if feature != 'tempo' else 'BPM')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
            
            print(f"✅ {title}: Mean={mean_val:.3f}, Std={std_val:.3f}")
    
    plt.suptitle('🎵 Audio Features Summary Statistics', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'overview_2_feature_summaries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: overview_2_feature_summaries.png")

def create_simple_stats_table(df, output_dir):
    """Create a simple statistics table as text"""
    
    print("\n📝 Creating statistics summary...\n")
    
    stats = []
    stats.append("="*80)
    stats.append("📊 SPOTIFY DATASET - SIMPLE STATISTICS SUMMARY")
    stats.append("="*80)
    stats.append("")
    
    # Basic counts
    stats.append("📈 DATASET SIZE")
    stats.append(f"   Total Tracks: {len(df):,}")
    
    if 'artist_name' in df.columns:
        stats.append(f"   Unique Artists: {df['artist_name'].nunique():,}")
        top_artist = df['artist_name'].value_counts().iloc[0]
        top_artist_name = df['artist_name'].value_counts().index[0]
        stats.append(f"   Most Featured Artist: {top_artist_name} ({top_artist:,} tracks)")
    
    if 'album_name' in df.columns:
        stats.append(f"   Unique Albums: {df['album_name'].nunique():,}")
    
    if 'playlist_category' in df.columns:
        stats.append(f"   Playlist Categories: {df['playlist_category'].nunique()}")
    
    stats.append("")
    
    # Category breakdown
    if 'playlist_category' in df.columns:
        stats.append("🎵 TRACKS PER CATEGORY")
        cat_counts = df['playlist_category'].value_counts()
        for cat, count in cat_counts.items():
            pct = (count / len(df)) * 100
            stats.append(f"   {cat:15s}: {count:8,} tracks ({pct:5.1f}%)")
        stats.append("")
    
    # Audio feature averages
    stats.append("🎯 AUDIO FEATURE AVERAGES")
    features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness', 
                'loudness', 'speechiness', 'instrumentalness', 'liveness']
    
    for feature in features:
        if feature in df.columns:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            min_val = df[feature].min()
            max_val = df[feature].max()
            stats.append(f"   {feature:18s}: {mean_val:6.3f} (±{std_val:.3f})  [range: {min_val:.3f} - {max_val:.3f}]")
    
    stats.append("")
    
    # Popularity breakdown
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        stats.append("⭐ POPULARITY BREAKDOWN")
        stats.append(f"   Average: {df[pop_col].mean():.1f}")
        stats.append(f"   Median: {df[pop_col].median():.1f}")
        stats.append(f"   Unpopular (<30): {(df[pop_col] < 30).sum():,} tracks ({(df[pop_col] < 30).mean()*100:.1f}%)")
        stats.append(f"   Moderate (30-70): {((df[pop_col] >= 30) & (df[pop_col] < 70)).sum():,} tracks ({((df[pop_col] >= 30) & (df[pop_col] < 70)).mean()*100:.1f}%)")
        stats.append(f"   Popular (70+): {(df[pop_col] >= 70).sum():,} tracks ({(df[pop_col] >= 70).mean()*100:.1f}%)")
        stats.append("")
    
    # Year breakdown
    if 'release_date' in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        stats.append("📅 RELEASE YEAR DISTRIBUTION")
        stats.append(f"   Earliest: {df['year'].min():.0f}")
        stats.append(f"   Latest: {df['year'].max():.0f}")
        stats.append(f"   Peak year: {df['year'].mode().iloc[0]:.0f} ({df[df['year'] == df['year'].mode().iloc[0]].shape[0]:,} tracks)")
        stats.append("")
    
    # Top artists
    if 'artist_name' in df.columns:
        stats.append("🎤 TOP 10 ARTISTS (BY TRACK COUNT)")
        top_artists = df['artist_name'].value_counts().head(10)
        for i, (artist, count) in enumerate(top_artists.items(), 1):
            stats.append(f"   {i:2d}. {artist:25s} → {count:6,} tracks")
        stats.append("")
    
    stats.append("="*80)
    
    # Write to file
    stats_text = "\n".join(stats)
    with open(output_dir / 'DATASET_STATISTICS.txt', 'w') as f:
        f.write(stats_text)
    
    print("\n✅ Saved: DATASET_STATISTICS.txt")
    print("\n" + stats_text)

def create_category_pie_chart(df, output_dir):
    """Create simple pie chart of categories"""
    
    print("\n📊 Creating category pie chart...\n")
    
    if 'playlist_category' not in df.columns:
        print("⚠️  Skipping: playlist_category not found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    cat_counts = df['playlist_category'].value_counts()
    
    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_counts)))
    wedges, texts, autotexts = ax.pie(
        cat_counts.values,
        labels=cat_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Make percentage text larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('🎵 Dataset Composition by Playlist Category', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with counts
    legend_labels = [f'{cat}: {count:,} tracks' for cat, count in cat_counts.items()]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overview_3_category_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: overview_3_category_pie.png")

def create_key_metrics_dashboard(df, output_dir):
    """Create a dashboard of key metrics"""
    
    print("\n📊 Creating key metrics dashboard...\n")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create a text-based dashboard
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, '📊 SPOTIFY DATASET - KEY METRICS DASHBOARD', 
           ha='center', fontsize=20, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    # Dataset size
    y_pos = 0.85
    ax.text(0.05, y_pos, '📈 DATASET SIZE', fontsize=16, fontweight='bold')
    y_pos -= 0.05
    ax.text(0.1, y_pos, f'Total Tracks: {len(df):,}', fontsize=13)
    
    if 'artist_name' in df.columns:
        y_pos -= 0.04
        ax.text(0.1, y_pos, f'Unique Artists: {df["artist_name"].nunique():,}', fontsize=13)
    
    if 'album_name' in df.columns:
        y_pos -= 0.04
        ax.text(0.1, y_pos, f'Unique Albums: {df["album_name"].nunique():,}', fontsize=13)
    
    # Audio features
    y_pos -= 0.08
    ax.text(0.05, y_pos, '🎵 AUDIO FEATURES (AVERAGES)', fontsize=16, fontweight='bold')
    
    features = [
        ('energy', 'Energy'),
        ('danceability', 'Danceability'),
        ('valence', 'Mood (Valence)'),
        ('acousticness', 'Acousticness')
    ]
    
    for feature, label in features:
        if feature in df.columns:
            y_pos -= 0.04
            mean_val = df[feature].mean()
            # Create mini bar
            bar_length = mean_val * 0.3
            ax.text(0.1, y_pos, f'{label}:', fontsize=12)
            ax.text(0.3, y_pos, f'{mean_val:.3f}', fontsize=12, fontweight='bold')
            ax.barh(y_pos, bar_length, height=0.02, left=0.38, color='skyblue', alpha=0.7)
    
    # Popularity
    y_pos -= 0.08
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        ax.text(0.05, y_pos, '⭐ POPULARITY', fontsize=16, fontweight='bold')
        y_pos -= 0.04
        ax.text(0.1, y_pos, f'Average: {df[pop_col].mean():.1f}/100', fontsize=13)
        y_pos -= 0.04
        hits = (df[pop_col] >= 70).sum()
        ax.text(0.1, y_pos, f'Hit Songs (70+): {hits:,} ({hits/len(df)*100:.1f}%)', fontsize=13)
    
    # Categories
    y_pos -= 0.08
    if 'playlist_category' in df.columns:
        ax.text(0.05, y_pos, '🎼 TOP 5 CATEGORIES', fontsize=16, fontweight='bold')
        cat_counts = df['playlist_category'].value_counts().head(5)
        for cat, count in cat_counts.items():
            y_pos -= 0.04
            pct = (count / len(df)) * 100
            ax.text(0.1, y_pos, f'{cat}: {count:,} ({pct:.1f}%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overview_4_key_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: overview_4_key_metrics.png")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=2000000,
                       help='Number of rows to sample (default: 2M)')
    args = parser.parse_args()
    
    print("="*80)
    print("📊 SPOTIFY DATASET OVERVIEW GENERATOR")
    print("="*80)
    
    # Load data
    df = load_data(sample_size=args.sample)
    
    # Create output directory
    output_dir = Path(__file__).parent
    
    # Generate overview visualizations
    create_overview_charts(df, output_dir)
    create_feature_summaries(df, output_dir)
    create_category_pie_chart(df, output_dir)
    create_key_metrics_dashboard(df, output_dir)
    create_simple_stats_table(df, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL OVERVIEW VISUALIZATIONS GENERATED!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print("\n📊 Generated Files:")
    print("   • overview_1_dataset_composition.png")
    print("   • overview_2_feature_summaries.png")
    print("   • overview_3_category_pie.png")
    print("   • overview_4_key_metrics.png")
    print("   • DATASET_STATISTICS.txt")

if __name__ == "__main__":
    main()
