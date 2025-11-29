# 🎵 Synthetic Spotify Dataset Generator

## Overview

This script generates a **realistic synthetic Spotify dataset** with proper distributions, correlations, and patterns that mimic real music data.

---

## ✨ Key Features

### Realistic Audio Features
- **Energy**: Beta distribution, right-skewed (mean ~0.65)
- **Loudness**: Normal distribution, correlated with energy (-60 to 0 dB)
- **Danceability**: Beta distribution, right-skewed (mean ~0.60)
- **Valence**: Slightly correlated with danceability
- **Tempo**: Normal distribution around 120 BPM (std: 28)
- **Acousticness**: Negatively correlated with energy
- **Instrumentalness**: Heavy right skew (most vocal tracks)
- **Speechiness**: Heavy right skew
- **Liveness**: Heavy right skew (studio recordings)

### Realistic Correlations
- Energy ↔ Loudness: **Strong positive** (+0.7 to +0.8)
- Energy ↔ Acousticness: **Strong negative** (-0.6 to -0.7)
- Danceability ↔ Valence: **Moderate positive** (+0.3 to +0.4)
- Tempo: **Low correlations** (independent)

### Category-Specific Profiles
Different playlist categories have distinct audio characteristics:

| Category | Energy | Tempo | Acousticness | Danceability | Valence |
|----------|--------|-------|--------------|--------------|---------|
| **Workout** | High ↑ | High ↑ | Low | Medium | Medium |
| **Sleep** | Low ↓ | Low ↓ | High ↑ | Low | Low |
| **Party** | High ↑ | Medium | Low | High ↑ | High ↑ |
| **Chill** | Low-Med ↓ | Medium | High ↑ | Low-Med | Medium |
| **Focus/Study** | Low-Med ↓ | Medium | Medium | Low | Low-Med |

### Realistic Metadata
- **Artists**: Power law distribution (few popular, many niche)
- **Popularity**: Right-skewed (most songs unpopular, few hits)
- **Release Dates**: 2010-2023, biased toward recent years
- **Followers**: Log-normal distribution
- **Track Duration**: 30s to 10min, mean ~3-4 minutes

---

## 🚀 Usage

### Option 1: Generate Small Test Dataset (Quick)

```bash
cd /Users/dewansh/Code/big_data
source venv/bin/activate

# Generate 100K rows (~25 MB) for testing
python scripts/generate_synthetic_spotify_data.py --rows 100000 --output data/spotify_test_data.csv
```

**Time**: ~30 seconds  
**Size**: ~25 MB

---

### Option 2: Generate Medium Dataset (Reasonable)

```bash
# Generate 5M rows (~1.2 GB)
python scripts/generate_synthetic_spotify_data.py --rows 5000000 --output data/spotify_synthetic_data.csv
```

**Time**: ~5-10 minutes  
**Size**: ~1.2 GB

---

### Option 3: Generate Large Dataset (Big Data Scale)

```bash
# Generate 50M rows (~12 GB) - matches your original requirement
python scripts/generate_synthetic_spotify_data.py --rows 50000000 --output data/spotify_synthetic_data.csv --chunk-size 5000000
```

**Time**: ~1-2 hours  
**Size**: ~12 GB

---

### Option 4: Custom Configuration

```bash
python scripts/generate_synthetic_spotify_data.py \
  --rows 20000000 \
  --output data/custom_spotify_data.csv \
  --chunk-size 2000000
```

---

## 📊 Generated Columns

The dataset includes **24 columns**:

### Identifiers
1. `track_id` - Unique track identifier
2. `track_name` - Song title
3. `artist_name` - Artist name
4. `album_name` - Album title

### Metadata
5. `release_date` - Release date (2010-2023)
6. `playlist_name` - Playlist name
7. `playlist_category` - Category (workout, party, chill, etc.)
8. `num_followers` - Playlist follower count
9. `popularity_score` - Popularity (0-100)

### Audio Features (Spotify-like)
10. `energy` - Energy level (0-1)
11. `loudness` - Loudness in dB (-60 to 0)
12. `danceability` - Danceability (0-1)
13. `valence` - Mood/happiness (0-1)
14. `tempo` - Tempo in BPM (40-220)
15. `acousticness` - Acousticness (0-1)
16. `instrumentalness` - Instrumental content (0-1)
17. `speechiness` - Speech content (0-1)
18. `liveness` - Live recording indicator (0-1)
19. `duration_ms` - Track duration in milliseconds
20. `key` - Musical key (0-11)
21. `mode` - Major (1) or Minor (0)
22. `time_signature` - Time signature (2-5)

---

## 🔄 Integration with Your Pipeline

### Step 1: Generate the Data

```bash
cd /Users/dewansh/Code/big_data
source venv/bin/activate

# Generate dataset
python scripts/generate_synthetic_spotify_data.py --rows 5000000
```

### Step 2: Upload to S3

```bash
# The file will be in data/spotify_synthetic_data.csv
# Upload using your existing script (update the filename in the script)

# Update scripts/01_upload_to_s3.py to use the new file:
# Change: 'spotify_data.csv' → 'spotify_synthetic_data.csv'

python scripts/01_upload_to_s3.py
```

### Step 3: Continue with Existing Pipeline

```bash
# On EC2:
python scripts/download_from_s3.py
python scripts/03_data_cleaning.py --chunk-size 500000
python scripts/04_feature_engineering.py
python ml_models/popularity_model.py --sample 5000000
python visualizations/insights_analysis.py --sample 500000
```

---

## 📈 Expected Results

With this realistic synthetic data, you should see:

### ✅ Proper Distributions
- Audio features show natural bell curves and skews
- No more flat uniform distributions

### ✅ Meaningful Correlations
- Energy ↔ Loudness: +0.7 to +0.8
- Energy ↔ Acousticness: -0.6 to -0.7
- Danceability ↔ Valence: +0.3 to +0.4

### ✅ Category Differences
- Workout music: Higher energy and tempo
- Sleep music: Lower energy, higher acousticness
- Party music: High danceability and valence

### ✅ Realistic Popularity Patterns
- Most songs: 10-40 popularity
- Popular songs: 70-90 popularity
- Hits: 90-100 popularity (rare)

---

## 🎯 Quick Start (Recommended)

```bash
# 1. Generate test data (100K rows, ~30 seconds)
cd /Users/dewansh/Code/big_data
source venv/bin/activate
python scripts/generate_synthetic_spotify_data.py --rows 100000 --output data/spotify_test.csv

# 2. Verify it worked
head -20 data/spotify_test.csv
wc -l data/spotify_test.csv

# 3. Test visualizations locally
cd visualizations
python raw_data_insights.py --sample 100000

# 4. If looks good, generate full dataset (5M rows)
python scripts/generate_synthetic_spotify_data.py --rows 5000000

# 5. Upload and continue pipeline
python scripts/01_upload_to_s3.py
```

---

## 💡 Tips

- **Start small**: Generate 100K rows first to verify
- **Memory**: Uses ~500MB RAM per 5M rows
- **Speed**: ~100K rows/second on modern hardware
- **Disk**: Ensure you have 2x target size free space
- **Chunks**: Use `--chunk-size` for very large datasets

---

## 🔍 Verification

After generation, verify the data quality:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load sample
df = pd.read_csv('data/spotify_synthetic_data.csv', nrows=100000)

# Check distributions
df[['energy', 'danceability', 'valence']].hist(bins=50, figsize=(15, 5))
plt.savefig('verification.png')

# Check correlations
print(df[['energy', 'loudness', 'acousticness', 'danceability', 'valence']].corr())

# Check categories
print(df['playlist_category'].value_counts())

# Check popularity distribution
print(df['popularity_score'].describe())
```

Expected output:
- Energy mean: ~0.60-0.70
- Energy ↔ Loudness correlation: +0.70 to +0.80
- Energy ↔ Acousticness correlation: -0.60 to -0.70
- Popularity median: ~35-45

---

## 🆚 Comparison: Old vs New Data

| Metric | Old (Uniform) | New (Realistic) |
|--------|---------------|-----------------|
| Energy Mean | 0.500 | 0.60-0.70 |
| Energy Std | 0.290 | 0.20-0.25 |
| Energy ↔ Loudness | 0.00 | +0.75 |
| Energy ↔ Acousticness | 0.00 | -0.65 |
| Popularity Distribution | Uniform | Right-skewed |
| Category Differences | None | Significant |

---

## 🎓 What Makes This Realistic?

1. **Proper Distributions**: Beta/Normal/Log-normal instead of uniform
2. **Real Correlations**: Energy + Loudness move together
3. **Natural Patterns**: Popular songs have higher energy/danceability
4. **Category Logic**: Workout ≠ Sleep in audio features
5. **Power Laws**: Few popular artists, many niche ones
6. **Temporal Bias**: Recent songs more common

This mimics real Spotify data patterns observed in academic research!

---

## 📚 References

Based on patterns from:
- Spotify Million Playlist Dataset
- Million Song Dataset
- Academic papers on music feature analysis

