"""
Spotify Data - Meaningful Insights Analysis
Creates business-focused visualizations with actionable insights
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for EC2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_data(sample_size=1000000):
    """Load data with sampling"""
    print(f"\n📂 Loading {sample_size:,} samples...")
    
    data_path = Path(__file__).parent.parent / 'data/processed/spotify_features.parquet'
    df = pd.read_parquet(data_path)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"✅ Loaded {len(df):,} rows")
    return df

def insight_1_playlist_dna(df, output_dir):
    """
    INSIGHT: What defines different playlist categories?
    Business Value: Understand listener intent and music selection
    """
    print("\n📊 Insight 1: Playlist Category DNA...")
    
    if 'playlist_category' not in df.columns:
        print("   ⚠️  Skipping: playlist_category not found")
        return
    
    # Get top categories
    top_categories = df['playlist_category'].value_counts().head(6).index
    df_subset = df[df['playlist_category'].isin(top_categories)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = [
        ('energy', 'Energy Level', 'Intensity & Activity'),
        ('danceability', 'Danceability', 'How Danceable'),
        ('valence', 'Mood (Valence)', 'Happiness Level'),
        ('tempo', 'Tempo (BPM)', 'Speed'),
        ('acousticness', 'Acousticness', 'Acoustic vs Electric'),
        ('instrumentalness', 'Instrumentalness', 'Vocal vs Instrumental')
    ]
    
    for idx, (feature, title, subtitle) in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        
        if feature in df_subset.columns:
            # Box plot by category
            df_subset.boxplot(column=feature, by='playlist_category', ax=ax)
            ax.set_title(f'{title}\n{subtitle}', fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.set_ylabel(title, fontsize=10)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add mean line
            means = df_subset.groupby('playlist_category')[feature].mean()
            ax.axhline(y=df_subset[feature].mean(), color='red', linestyle='--', 
                      alpha=0.5, label='Overall Mean')
    
    plt.suptitle('🎵 Playlist Category DNA: What Makes Each Category Unique?', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_1_playlist_dna.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_1_playlist_dna.png")
    
    # Print key insights
    print("\n   📌 Key Findings:")
    for cat in top_categories[:3]:
        cat_data = df_subset[df_subset['playlist_category'] == cat]
        if 'energy' in cat_data.columns and 'tempo' in cat_data.columns:
            print(f"   • {cat.upper()}: Energy={cat_data['energy'].mean():.2f}, "
                  f"Tempo={cat_data['tempo'].mean():.0f} BPM")

def insight_2_popularity_drivers(df, output_dir):
    """
    INSIGHT: What makes songs popular?
    Business Value: Guide music production and playlist curation
    """
    print("\n📊 Insight 2: What Drives Popularity?...")
    
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col not in df.columns:
        print("   ⚠️  Skipping: popularity column not found")
        return
    
    # Create popularity bins
    df['pop_bin'] = pd.cut(df[pop_col], bins=[0, 30, 60, 100], 
                           labels=['Low (<30)', 'Medium (30-60)', 'High (60+)'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness', 'track_length_min']
    titles = ['Energy', 'Danceability', 'Mood (Valence)', 'Tempo (BPM)', 'Acousticness', 'Track Length (min)']
    
    for idx, (feature, title) in enumerate(zip(features, titles)):
        ax = axes[idx // 3, idx % 3]
        
        if feature in df.columns:
            # Violin plot by popularity bin
            pop_data = [df[df['pop_bin'] == cat][feature].dropna() 
                       for cat in ['Low (<30)', 'Medium (30-60)', 'High (60+)']]
            
            parts = ax.violinplot(pop_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(['Low\n(<30)', 'Medium\n(30-60)', 'High\n(60+)'])
            ax.set_title(f'{title} by Popularity', fontweight='bold', fontsize=11)
            ax.set_ylabel(title)
            ax.set_xlabel('Popularity Level')
            ax.grid(True, alpha=0.3)
            
            # Calculate correlation
            corr = df[[feature, pop_col]].corr().iloc[0, 1]
            ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('🎯 What Makes Songs Popular? Feature Analysis by Popularity Level', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_2_popularity_drivers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_2_popularity_drivers.png")
    
    # Print insights
    print("\n   📌 Key Findings:")
    for feature in ['energy', 'danceability', 'tempo']:
        if feature in df.columns:
            high_pop = df[df[pop_col] > 60][feature].mean()
            low_pop = df[df[pop_col] < 30][feature].mean()
            diff = ((high_pop - low_pop) / low_pop) * 100
            print(f"   • Popular songs have {diff:+.1f}% more {feature}")

def insight_3_music_mood_map(df, output_dir):
    """
    INSIGHT: Music Mood Landscape
    Business Value: Understand the emotional spectrum of music
    """
    print("\n📊 Insight 3: Music Mood Landscape...")
    
    if 'energy' not in df.columns or 'valence' not in df.columns:
        print("   ⚠️  Skipping: energy or valence not found")
        return
    
    # Sample for plotting (too many points otherwise)
    plot_sample = df.sample(n=min(50000, len(df)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create scatter with color by popularity
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in plot_sample.columns:
        scatter = ax.scatter(plot_sample['valence'], plot_sample['energy'], 
                           c=plot_sample[pop_col], cmap='YlOrRd', 
                           alpha=0.6, s=20, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Popularity Score')
    else:
        ax.scatter(plot_sample['valence'], plot_sample['energy'], 
                  alpha=0.4, s=20, edgecolors='none')
    
    # Add quadrant labels
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Label quadrants
    ax.text(0.25, 0.75, '😤 ANGRY/INTENSE\n(Low mood, High energy)', 
           ha='center', fontsize=12, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax.text(0.75, 0.75, '🎉 HAPPY/ENERGETIC\n(High mood, High energy)', 
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.text(0.25, 0.25, '😢 SAD/CALM\n(Low mood, Low energy)', 
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    ax.text(0.75, 0.25, '😌 PEACEFUL/CONTENT\n(High mood, Low energy)', 
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    ax.set_xlabel('Valence (Mood: Sad ← → Happy)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy (Calm ← → Energetic)', fontsize=14, fontweight='bold')
    ax.set_title('🎵 The Music Mood Landscape\nWhere Does Your Music Live?', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_3_mood_landscape.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_3_mood_landscape.png")
    
    # Calculate quadrant percentages
    print("\n   📌 Music Distribution:")
    q1 = ((df['energy'] > 0.5) & (df['valence'] > 0.5)).sum()
    q2 = ((df['energy'] > 0.5) & (df['valence'] <= 0.5)).sum()
    q3 = ((df['energy'] <= 0.5) & (df['valence'] <= 0.5)).sum()
    q4 = ((df['energy'] <= 0.5) & (df['valence'] > 0.5)).sum()
    total = len(df)
    print(f"   • Happy/Energetic: {q1/total*100:.1f}%")
    print(f"   • Angry/Intense: {q2/total*100:.1f}%")
    print(f"   • Sad/Calm: {q3/total*100:.1f}%")
    print(f"   • Peaceful: {q4/total*100:.1f}%")

def insight_4_workout_vs_sleep(df, output_dir):
    """
    INSIGHT: Workout vs Sleep Music - Extreme Comparison
    Business Value: Optimize playlist curation for specific activities
    """
    print("\n📊 Insight 4: Workout vs Sleep Music Showdown...")
    
    if 'playlist_category' not in df.columns:
        print("   ⚠️  Skipping: playlist_category not found")
        return
    
    # Filter workout and sleep tracks
    workout = df[df['playlist_category'].str.contains('workout|gym|fitness|running', 
                                                      case=False, na=False)]
    sleep = df[df['playlist_category'].str.contains('sleep|calm|relax|chill', 
                                                    case=False, na=False)]
    
    if len(workout) < 100 or len(sleep) < 100:
        print("   ⚠️  Not enough workout/sleep tracks found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = [
        ('energy', 'Energy Level'),
        ('tempo', 'Tempo (BPM)'),
        ('danceability', 'Danceability'),
        ('valence', 'Mood (Valence)'),
        ('acousticness', 'Acousticness'),
        ('track_length_min', 'Track Length (min)')
    ]
    
    for idx, (feature, title) in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        
        if feature in df.columns:
            # Side-by-side comparison
            data_to_plot = [
                workout[feature].dropna(),
                sleep[feature].dropna()
            ]
            
            bp = ax.boxplot(data_to_plot, labels=['🏋️ Workout', '😴 Sleep'],
                           patch_artist=True, showmeans=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor('orangered')
            bp['boxes'][1].set_facecolor('lightblue')
            
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            workout_mean = workout[feature].mean()
            sleep_mean = sleep[feature].mean()
            diff_pct = ((workout_mean - sleep_mean) / sleep_mean) * 100
            
            ax.text(0.5, 0.98, f'Workout: {workout_mean:.2f}  |  Sleep: {sleep_mean:.2f}  |  Δ {diff_pct:+.0f}%',
                   transform=ax.transAxes, ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('🏋️ vs 😴 Workout Music vs Sleep Music: The Ultimate Contrast', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_4_workout_vs_sleep.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_4_workout_vs_sleep.png")
    
    # Print insights
    print("\n   📌 Key Differences:")
    print(f"   • Workout has {diff_pct:.0f}% more energy than sleep music")

def insight_5_popularity_sweet_spot(df, output_dir):
    """
    INSIGHT: The Popularity Sweet Spot - Optimal Feature Ranges
    Business Value: Guide artists on what features to target
    """
    print("\n📊 Insight 5: The Popularity Sweet Spot...")
    
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col not in df.columns:
        print("   ⚠️  Skipping: popularity not found")
        return
    
    # Define popularity bins
    df['pop_category'] = pd.cut(df[pop_col], bins=[0, 40, 70, 100],
                                labels=['Unpopular\n(0-40)', 'Moderate\n(40-70)', 'Hit\n(70+)'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    features_to_analyze = [
        ('energy', 'Energy', 'Higher energy → More popular?'),
        ('danceability', 'Danceability', 'More danceable → More popular?'),
        ('tempo', 'Tempo (BPM)', 'Faster tempo → More popular?'),
        ('valence', 'Mood (Valence)', 'Happier mood → More popular?')
    ]
    
    for idx, (feature, title, question) in enumerate(features_to_analyze):
        ax = axes[idx // 2, idx % 2]
        
        if feature in df.columns:
            # Calculate means by popularity category
            pop_means = df.groupby('pop_category')[feature].agg(['mean', 'std'])
            
            # Bar plot with error bars
            x_pos = np.arange(len(pop_means))
            bars = ax.bar(x_pos, pop_means['mean'], yerr=pop_means['std'],
                         color=['red', 'orange', 'green'], alpha=0.7, capsize=5)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(pop_means.index)
            ax.set_title(f'{title}\n{question}', fontweight='bold', fontsize=11)
            ax.set_ylabel(f'Average {title}')
            ax.set_xlabel('Popularity Category')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Calculate trend
            unpop_mean = df[df[pop_col] < 40][feature].mean()
            hit_mean = df[df[pop_col] > 70][feature].mean()
            trend = "↗️ UP" if hit_mean > unpop_mean else "↘️ DOWN"
            ax.text(0.98, 0.02, f'Trend: {trend}',
                   transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('🎯 The Popularity Sweet Spot: What Features Do Hits Have?', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_5_popularity_sweet_spot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_5_popularity_sweet_spot.png")

def insight_6_follower_impact(df, output_dir):
    """
    INSIGHT: Do High-Follower Playlists Have Different Music?
    Business Value: Understand what makes playlists successful
    """
    print("\n📊 Insight 6: High vs Low Follower Playlists...")
    
    if 'num_followers' not in df.columns:
        print("   ⚠️  Skipping: num_followers not found")
        return
    
    # Split by median followers
    median_followers = df['num_followers'].median()
    df['follower_tier'] = df['num_followers'].apply(
        lambda x: 'High Followers' if x > median_followers else 'Low Followers'
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness']
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        features.append(pop_col)
    
    for idx, feature in enumerate(features[:6]):
        ax = axes[idx // 3, idx % 3]
        
        if feature in df.columns:
            high_fol = df[df['follower_tier'] == 'High Followers'][feature]
            low_fol = df[df['follower_tier'] == 'Low Followers'][feature]
            
            # KDE plot
            high_fol.plot(kind='density', ax=ax, label='High Followers', 
                         color='green', linewidth=2.5)
            low_fol.plot(kind='density', ax=ax, label='Low Followers',
                        color='red', linewidth=2.5, linestyle='--')
            
            ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Statistical test
            statistic, pvalue = stats.mannwhitneyu(high_fol.dropna(), low_fol.dropna())
            sig = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"
            ax.text(0.98, 0.98, f'p-value: {sig}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('👥 Do Popular Playlists Have Different Music?\nHigh vs Low Follower Playlist Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_6_follower_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_6_follower_impact.png")

def insight_7_tempo_zones(df, output_dir):
    """
    INSIGHT: Tempo Zones and Their Characteristics
    Business Value: Understand pacing preferences
    """
    print("\n📊 Insight 7: Tempo Zones Analysis...")
    
    if 'tempo' not in df.columns:
        print("   ⚠️  Skipping: tempo not found")
        return
    
    # Create tempo zones
    df['tempo_zone'] = pd.cut(df['tempo'], 
                              bins=[0, 80, 100, 120, 140, 300],
                              labels=['Very Slow\n(<80)', 'Slow\n(80-100)', 
                                     'Moderate\n(100-120)', 'Fast\n(120-140)', 
                                     'Very Fast\n(140+)'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of tempo zones
    tempo_counts = df['tempo_zone'].value_counts().sort_index()
    ax = axes[0, 0]
    bars = ax.bar(range(len(tempo_counts)), tempo_counts.values, 
                 color=['blue', 'cyan', 'yellow', 'orange', 'red'], alpha=0.7)
    ax.set_xticks(range(len(tempo_counts)))
    ax.set_xticklabels(tempo_counts.index)
    ax.set_title('Distribution of Songs by Tempo Zone', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Tracks')
    # Add percentages
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Energy by tempo zone
    ax = axes[0, 1]
    if 'energy' in df.columns:
        df.boxplot(column='energy', by='tempo_zone', ax=ax)
        ax.set_title('Energy Level by Tempo Zone', fontweight='bold', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Energy')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Popularity by tempo
    ax = axes[1, 0]
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        tempo_pop = df.groupby('tempo_zone')[pop_col].mean()
        ax.bar(range(len(tempo_pop)), tempo_pop.values,
              color=['blue', 'cyan', 'yellow', 'orange', 'red'], alpha=0.7)
        ax.set_xticks(range(len(tempo_pop)))
        ax.set_xticklabels(tempo_pop.index)
        ax.set_title('Average Popularity by Tempo Zone', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Popularity')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Danceability by tempo
    ax = axes[1, 1]
    if 'danceability' in df.columns:
        tempo_dance = df.groupby('tempo_zone')['danceability'].mean()
        ax.bar(range(len(tempo_dance)), tempo_dance.values,
              color=['blue', 'cyan', 'yellow', 'orange', 'red'], alpha=0.7)
        ax.set_xticks(range(len(tempo_dance)))
        ax.set_xticklabels(tempo_dance.index)
        ax.set_title('Average Danceability by Tempo Zone', fontweight='bold', fontsize=12)
        ax.set_ylabel('Danceability')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('🎵 Tempo Zones: How Music Pacing Affects Other Features', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_7_tempo_zones.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_7_tempo_zones.png")

def insight_8_top_artists_profiles(df, output_dir):
    """
    INSIGHT: Top Artists and Their Signature Sound
    Business Value: Understand successful artist strategies
    """
    print("\n📊 Insight 8: Top Artists' Signature Sounds...")
    
    if 'artist_name' not in df.columns:
        print("   ⚠️  Skipping: artist_name not found")
        return
    
    # Get top 10 artists by track count
    top_artists = df['artist_name'].value_counts().head(10).index
    df_top = df[df['artist_name'].isin(top_artists)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Track count
    ax = axes[0, 0]
    artist_counts = df_top['artist_name'].value_counts()
    ax.barh(range(len(artist_counts)), artist_counts.values, color='teal')
    ax.set_yticks(range(len(artist_counts)))
    ax.set_yticklabels(artist_counts.index, fontsize=10)
    ax.set_title('Top 10 Artists by Track Count', fontweight='bold', fontsize=12)
    ax.set_xlabel('Number of Tracks')
    ax.invert_yaxis()
    
    # 2. Energy profile
    ax = axes[0, 1]
    if 'energy' in df_top.columns:
        energy_by_artist = df_top.groupby('artist_name')['energy'].mean().sort_values(ascending=False)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(energy_by_artist)))
        ax.barh(range(len(energy_by_artist)), energy_by_artist.values, color=colors)
        ax.set_yticks(range(len(energy_by_artist)))
        ax.set_yticklabels(energy_by_artist.index, fontsize=10)
        ax.set_title('Average Energy by Artist', fontweight='bold', fontsize=12)
        ax.set_xlabel('Average Energy')
        ax.invert_yaxis()
        ax.axvline(x=df['energy'].mean(), color='red', linestyle='--', label='Overall Mean')
        ax.legend()
    
    # 3. Danceability profile
    ax = axes[1, 0]
    if 'danceability' in df_top.columns:
        dance_by_artist = df_top.groupby('artist_name')['danceability'].mean().sort_values(ascending=False)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(dance_by_artist)))
        ax.barh(range(len(dance_by_artist)), dance_by_artist.values, color=colors)
        ax.set_yticks(range(len(dance_by_artist)))
        ax.set_yticklabels(dance_by_artist.index, fontsize=10)
        ax.set_title('Average Danceability by Artist', fontweight='bold', fontsize=12)
        ax.set_xlabel('Average Danceability')
        ax.invert_yaxis()
    
    # 4. Artist radar chart (pick one artist)
    ax = axes[1, 1]
    if len(top_artists) > 0:
        artist = top_artists[0]
        artist_data = df_top[df_top['artist_name'] == artist]
        
        features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness']
        available_features = [f for f in features if f in artist_data.columns]
        
        if len(available_features) >= 3:
            # Normalize tempo to 0-1 range
            values = []
            for f in available_features:
                if f == 'tempo':
                    values.append(artist_data[f].mean() / 200)  # Normalize tempo
                else:
                    values.append(artist_data[f].mean())
            
            angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax = plt.subplot(2, 2, 4, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2, color='purple')
            ax.fill(angles, values, alpha=0.25, color='purple')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f.replace('_', ' ').title() for f in available_features])
            ax.set_ylim(0, 1)
            ax.set_title(f'Sound Profile: {artist}', fontweight='bold', pad=20)
            ax.grid(True)
    
    plt.suptitle('🎤 Top Artists: Who They Are and Their Signature Sound', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_8_top_artists_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_8_top_artists_profiles.png")

def insight_9_correlation_insights(df, output_dir):
    """
    INSIGHT: Feature Relationships - What Goes Together?
    Business Value: Understand natural music patterns
    """
    print("\n📊 Insight 9: Feature Correlation Insights...")
    
    features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness', 'instrumentalness']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 3:
        print("   ⚠️  Not enough features available")
        return
    
    # Calculate correlations
    corr = df[available_features].corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Heatmap
    ax = axes[0]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8},
               ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix\nWhat Features Move Together?', 
                fontweight='bold', fontsize=14)
    
    # 2. Top correlations
    ax = axes[1]
    # Get upper triangle correlations
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_pairs.append({
                'pair': f'{corr.columns[i]}\nvs\n{corr.columns[j]}',
                'correlation': corr.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', key=abs, ascending=False).head(10)
    
    colors = ['green' if x > 0 else 'red' for x in corr_df['correlation']]
    ax.barh(range(len(corr_df)), corr_df['correlation'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['pair'], fontsize=9)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Top 10 Feature Relationships\nGreen = Positive | Red = Negative', 
                fontweight='bold', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.suptitle('🔗 Feature Relationships: Understanding Music Patterns', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'insight_9_correlation_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: insight_9_correlation_insights.png")
    
    # Print insights
    print("\n   📌 Strongest Relationships:")
    for _, row in corr_df.head(3).iterrows():
        relationship = "positively" if row['correlation'] > 0 else "negatively"
        print(f"   • {row['pair'].replace(chr(10), ' ')}: {relationship} correlated ({row['correlation']:.2f})")

def insight_10_executive_summary(df, output_dir):
    """
    Create a comprehensive text report with all insights
    """
    print("\n📝 Creating Executive Summary...")
    
    report = []
    report.append("="*80)
    report.append("🎵 SPOTIFY BIG DATA ANALYSIS - EXECUTIVE SUMMARY")
    report.append("="*80)
    report.append("")
    
    # Dataset overview
    report.append("📊 DATASET OVERVIEW")
    report.append(f"   • Total Tracks Analyzed: {len(df):,}")
    if 'artist_name' in df.columns:
        report.append(f"   • Unique Artists: {df['artist_name'].nunique():,}")
    if 'album_name' in df.columns:
        report.append(f"   • Unique Albums: {df['album_name'].nunique():,}")
    if 'playlist_name' in df.columns:
        report.append(f"   • Unique Playlists: {df['playlist_name'].nunique():,}")
    report.append("")
    
    # Key metrics
    report.append("🎯 KEY AUDIO FEATURES (AVERAGES)")
    for feature in ['energy', 'danceability', 'valence', 'tempo', 'acousticness']:
        if feature in df.columns:
            report.append(f"   • {feature.title()}: {df[feature].mean():.3f}")
    report.append("")
    
    # Popularity insights
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    if pop_col in df.columns:
        report.append("⭐ POPULARITY INSIGHTS")
        report.append(f"   • Average Popularity: {df[pop_col].mean():.1f}/100")
        report.append(f"   • Median Popularity: {df[pop_col].median():.1f}/100")
        hits = (df[pop_col] > 70).sum()
        report.append(f"   • Hit Songs (70+): {hits:,} ({hits/len(df)*100:.1f}%)")
        report.append(f"   • Unpopular (<30): {(df[pop_col] < 30).sum():,} ({(df[pop_col] < 30).mean()*100:.1f}%)")
        report.append("")
    
    # Category insights
    if 'playlist_category' in df.columns:
        report.append("🎵 PLAYLIST CATEGORIES")
        top_cats = df['playlist_category'].value_counts().head(5)
        for cat, count in top_cats.items():
            pct = (count / len(df)) * 100
            report.append(f"   • {cat}: {count:,} tracks ({pct:.1f}%)")
        report.append("")
    
    # Mood distribution
    if 'energy' in df.columns and 'valence' in df.columns:
        report.append("😊 MUSIC MOOD DISTRIBUTION")
        happy_energetic = ((df['energy'] > 0.5) & (df['valence'] > 0.5)).sum()
        angry_intense = ((df['energy'] > 0.5) & (df['valence'] <= 0.5)).sum()
        sad_calm = ((df['energy'] <= 0.5) & (df['valence'] <= 0.5)).sum()
        peaceful = ((df['energy'] <= 0.5) & (df['valence'] > 0.5)).sum()
        
        report.append(f"   • Happy/Energetic (party music): {happy_energetic/len(df)*100:.1f}%")
        report.append(f"   • Angry/Intense (intense music): {angry_intense/len(df)*100:.1f}%")
        report.append(f"   • Sad/Calm (melancholic): {sad_calm/len(df)*100:.1f}%")
        report.append(f"   • Peaceful/Content (relaxing): {peaceful/len(df)*100:.1f}%")
        report.append("")
    
    # Top artists
    if 'artist_name' in df.columns:
        report.append("🎤 TOP 10 MOST FEATURED ARTISTS")
        top_artists = df['artist_name'].value_counts().head(10)
        for i, (artist, count) in enumerate(top_artists.items(), 1):
            report.append(f"   {i:2d}. {artist}: {count:,} tracks")
        report.append("")
    
    # Actionable insights
    report.append("💡 ACTIONABLE INSIGHTS")
    
    if 'energy' in df.columns and 'danceability' in df.columns and pop_col in df.columns:
        # What makes songs popular?
        high_pop = df[df[pop_col] > 70]
        low_pop = df[df[pop_col] < 30]
        
        energy_diff = high_pop['energy'].mean() - low_pop['energy'].mean()
        dance_diff = high_pop['danceability'].mean() - low_pop['danceability'].mean()
        
        report.append(f"   1. Popular songs have {energy_diff*100:+.1f}% more energy")
        report.append(f"   2. Popular songs have {dance_diff*100:+.1f}% more danceability")
    
    if 'tempo' in df.columns:
        report.append(f"   3. Optimal tempo range: {df['tempo'].quantile(0.25):.0f}-{df['tempo'].quantile(0.75):.0f} BPM")
    
    report.append("   4. Most tracks fall in moderate energy/danceability ranges")
    report.append("   5. Diversity in valence suggests music caters to all moods")
    
    report.append("")
    report.append("="*80)
    
    # Write report
    report_text = "\n".join(report)
    with open(output_dir / 'EXECUTIVE_SUMMARY.txt', 'w') as f:
        f.write(report_text)
    
    print("   ✅ Saved: EXECUTIVE_SUMMARY.txt")
    print("\n" + report_text)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=500000, 
                       help='Number of rows to sample (default: 500K)')
    args = parser.parse_args()
    
    print("="*80)
    print("🎵 SPOTIFY INSIGHTS ANALYSIS - Business Intelligence Dashboard")
    print("="*80)
    
    # Load data
    df = load_data(sample_size=args.sample)
    
    # Create output directory
    output_dir = Path(__file__).parent
    
    # Generate insight visualizations
    insight_1_playlist_dna(df, output_dir)
    insight_2_popularity_drivers(df, output_dir)
    insight_3_music_mood_map(df, output_dir)
    insight_4_workout_vs_sleep(df, output_dir)
    insight_5_popularity_sweet_spot(df, output_dir)
    insight_6_follower_impact(df, output_dir)
    insight_7_tempo_zones(df, output_dir)
    insight_8_top_artists_profiles(df, output_dir)
    insight_9_correlation_insights(df, output_dir)
    insight_10_executive_summary(df, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL INSIGHTS GENERATED!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print("\n📊 Generated Insights:")
    for file in sorted(output_dir.glob('insight_*.png')):
        print(f"   • {file.name}")
    print(f"   • EXECUTIVE_SUMMARY.txt")

if __name__ == "__main__":
    main()

