# 📊 Your Spotify Dataset Structure

## Dataset Overview

- **File**: `spotify_data.csv`
- **Size**: 13.19 GB
- **Rows**: ~62 million tracks (including header)
- **Columns**: 16

---

## Column Details

| Column | Type | Description | Value Range |
|--------|------|-------------|-------------|
| `playlist_id` | int/str | Unique playlist identifier | - |
| `playlist_name` | string | Name of the playlist | e.g., "Throwbacks" |
| `track_name` | string | Song title | e.g., "Lose Control" |
| `artist_name` | string | Artist name | e.g., "Missy Elliott" |
| `album_name` | string | Album name | e.g., "The Cookbook" |
| `duration_ms` | int | Track duration (milliseconds) | e.g., 226863 |
| `num_followers` | int | Number of playlist followers | e.g., 1, 1000, etc. |
| `tempo` | float | Beats per minute (BPM) | e.g., 163.99 |
| `danceability` | float | How suitable for dancing | 0.0 - 1.0 |
| `energy` | float | Intensity and activity | 0.0 - 1.0 |
| `valence` | float | Musical positivity (mood) | 0.0 - 1.0 |
| `acousticness` | float | Acoustic sound confidence | 0.0 - 1.0 |
| `instrumentalness` | float | Instrumental prediction | 0.0 - 1.0 |
| `popularity_score` | int | Popularity rating | 0 - 100 |
| `playlist_category` | string | Category label | "focus", "sleep", "workout", etc. |
| `track_length_min` | float | Track length (minutes) | e.g., 3.78 |

---

## Sample Data

```csv
playlist_id,playlist_name,track_name,artist_name,album_name,duration_ms,num_followers,tempo,danceability,energy,valence,acousticness,instrumentalness,popularity_score,playlist_category,track_length_min
0,Throwbacks,Lose Control (feat. Ciara & Fat Man Scoop),Missy Elliott,The Cookbook,226863,1,163.99,0.66,0.62,0.72,0.19,0.15,37,focus,3.78
0,Throwbacks,Toxic,Britney Spears,In The Zone,198800,1,74.43,0.86,0.96,0.23,0.29,0.40,43,sleep,3.31
```

---

## What Makes This Dataset Great

### ✅ Available Features

**Audio Features (for ML & Analysis):**
- ✅ Energy, Danceability, Valence (mood)
- ✅ Tempo, Acousticness, Instrumentalness
- ✅ Track length and duration

**Categorical Features:**
- ✅ **Playlist categories** (focus, sleep, workout, etc.)
- ✅ Artist, Track, Album, Playlist names
- ✅ Playlist followers (popularity indicator)

**Target Variable:**
- ✅ `popularity_score` (0-100) - Perfect for prediction models!

### ⚠️ Missing Features (compared to typical Spotify data)

The following are NOT in your dataset:
- ❌ `loudness` (dB)
- ❌ `speechiness` (spoken word detection)
- ❌ `liveness` (audience presence)
- ❌ `year` or `release_date` (can't do temporal analysis)
- ❌ `key`, `mode`, `time_signature` (music theory features)

---

## Analysis Opportunities

### 1. **Playlist Category Analysis** 🎯
Compare different categories:
- **Focus playlists**: Likely lower energy, higher acousticness
- **Workout playlists**: High energy, high danceability, fast tempo
- **Sleep playlists**: Low energy, low tempo, high acousticness
- **Party playlists**: High energy, high danceability, high valence

### 2. **Popularity Prediction** 🎵
Build ML model to predict `popularity_score` based on:
- Audio features (energy, danceability, etc.)
- Track length
- Playlist followers
- Playlist category

### 3. **Artist Analysis** 🎤
- Which artists appear most frequently?
- Which artists have highest average popularity?
- Artist style profiles (energy, danceability, etc.)

### 4. **Playlist Insights** 📊
- Most followed playlists
- Playlist composition (avg features per playlist)
- Popular vs niche playlists

### 5. **Music Recommendations** 🎧
Recommend similar tracks based on:
- Audio feature similarity
- Same category
- Similar tempo/energy/mood

### 6. **Mood & Energy Analysis** 😊⚡
- Distribution of moods (valence) across categories
- Energy patterns in different playlist types
- Correlation between mood and popularity

---

## Key Insights You Can Extract

### Question 1: What makes songs popular?
**Analysis**: Correlation between audio features and `popularity_score`
- Which features correlate most with popularity?
- Do popular songs have higher energy? More danceability?

### Question 2: How do playlist categories differ?
**Analysis**: Compare average features across `playlist_category`
- Workout playlists vs Sleep playlists
- Focus vs Party music characteristics

### Question 3: What defines each music category?
**Analysis**: Profile each category
- Energy, tempo, valence patterns
- Track length differences
- Instrumental vs vocal tracks

### Question 4: Who are the most popular artists?
**Analysis**: Aggregate by `artist_name`
- Artists with most tracks in dataset
- Artists with highest average popularity
- Artists dominating specific categories

### Question 5: Do high-follower playlists have different music?
**Analysis**: Compare tracks in high vs low follower playlists
- Are high-follower playlists more energetic?
- Do they feature more popular tracks?

---

## Recommended Analysis Workflow

### Phase 1: Data Cleaning (30-60 min on EC2)
```bash
python scripts/03_data_cleaning.py
# Output: cleaned dataset, ~60M rows, validated ranges
```

### Phase 2: Feature Engineering (30-60 min)
```bash
python scripts/04_feature_engineering.py
# Creates: mood categories, energy tiers, tempo bins, etc.
```

### Phase 3: Exploratory Analysis (1-2 hours)
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
# Explore distributions, correlations, patterns
```

### Phase 4: ML Models (1-2 hours)
```bash
python ml_models/popularity_model.py
# Train XGBoost to predict popularity_score

python ml_models/recommendation_model.py
# Build K-Means clustering for recommendations
```

### Phase 5: Visualizations (30 min)
```bash
python visualizations/create_charts.py
# Generate 10+ professional charts
```

---

## Expected Results

### Data Quality
- ✅ ~62M tracks successfully processed
- ✅ Missing values handled (<1% loss expected)
- ✅ All features in valid ranges

### ML Model Performance (Estimated)
- **Popularity Prediction**:
  - MAE: 15-20 points (out of 100)
  - R²: 0.3-0.5
  - Key predictors: energy, danceability, num_followers

- **Recommendation System**:
  - 50 clusters of similar tracks
  - Top-10 recommendations per track
  - Cosine similarity > 0.8 for good matches

### Visualizations (10+ charts)
1. Audio features distribution (6 charts)
2. Correlation heatmap
3. Popularity vs features (scatter plots)
4. Category comparison (bar charts)
5. Top artists (horizontal bars)
6. Playlist follower analysis

---

## Memory Considerations

**With 62 million rows and 16 columns:**

- Raw CSV: 13.19 GB
- Loaded in memory (Pandas): ~15-20 GB
- After feature engineering: ~25-30 GB

**Recommended EC2 instance:**
- Minimum: `m5.4xlarge` (64 GB RAM) ✅
- Better: `m5.8xlarge` (128 GB RAM) for comfort
- Budget: `m5.2xlarge` (32 GB RAM) + process in chunks

**Alternative: Use PySpark if memory issues**
```bash
# Already in requirements.txt (commented)
pip install pyspark pyarrow
# Then modify scripts to use Spark DataFrames
```

---

## Cost Estimate (Updated for Your Dataset)

| Resource | Specification | Duration | Cost |
|----------|--------------|----------|------|
| S3 Storage | 20 GB | 1 month | $0.50 |
| EC2 m5.4xlarge | 64 GB RAM | 15 hours | $25.20 |
| Data Transfer | Upload + Download | - | $2.00 |
| **Total** | - | - | **~$27.70** |

**To save money:**
- Use Spot Instances: **$7.56** (70% off)
- Process overnight (lower demand)
- Stop EC2 when idle

---

## Next Steps

1. ✅ Scripts are now updated for your data structure
2. 📤 Upload to S3: `python scripts/01_upload_to_s3.py --file spotify_data.csv`
3. 🚀 Launch EC2 (m5.4xlarge recommended)
4. 🔧 Setup environment: `bash 02_setup_ec2.sh`
5. 📥 Download data: `python scripts/download_from_s3.py`
6. 🧹 Clean data: `python scripts/03_data_cleaning.py`
7. ⚙️ Engineer features: `python scripts/04_feature_engineering.py`
8. 🤖 Train models: `python ml_models/*.py`
9. 📊 Create visualizations: `python visualizations/create_charts.py`
10. 💾 Save results: `python scripts/05_save_results.py --all`

---

**Your dataset is ready for big data analysis! 🚀**

