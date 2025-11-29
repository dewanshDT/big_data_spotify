# 🚀 Quick Start Guide - Synthetic Data Pipeline

Complete end-to-end guide for generating synthetic data and running the full analysis pipeline.

---

## 📋 Overview

1. **Generate synthetic data** (on your Mac)
2. **Upload to S3**
3. **Run master pipeline on EC2** (everything automated)
4. **Download results**

---

## Part 1: Generate Synthetic Data (Local - Mac)

### Step 1.1: Setup

```bash
cd /Users/dewansh/Code/big_data
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy
```

### Step 1.2: Generate Data

**Option A: Quick Test (100K rows, ~25 MB, 30 seconds)**
```bash
python3 generate_data_local.py --rows 100000 --output data/spotify_test.csv
```

**Option B: Medium Dataset (5M rows, ~1.2 GB, 5 minutes)**
```bash
python3 generate_data_local.py --rows 5000000 --output data/spotify_data.csv
```

**Option C: Large Dataset (20M rows, ~5 GB, 20 minutes)**
```bash
python3 generate_data_local.py --rows 20000000 --output data/spotify_data.csv
```

**Option D: Full Scale (50M rows, ~12 GB, 1 hour)**
```bash
python3 generate_data_local.py --rows 50000000 --output data/spotify_data.csv --chunk-size 5000000
```

### Step 1.3: Verify Data

```bash
# Check file size
ls -lh data/spotify_data.csv

# View first few rows
head -20 data/spotify_data.csv

# Count rows
wc -l data/spotify_data.csv
```

Expected columns:
```
track_id,track_name,artist_name,album_name,release_date,playlist_name,
playlist_category,num_followers,energy,loudness,danceability,valence,
tempo,acousticness,instrumentalness,speechiness,liveness,duration_ms,
key,mode,time_signature,popularity_score
```

---

## Part 2: Upload to S3 (Local - Mac)

### Step 2.1: Install AWS CLI (if not installed)

```bash
# Check if installed
aws --version

# If not installed:
brew install awscli  # macOS
```

### Step 2.2: Configure AWS Credentials

```bash
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: eu-north-1
# - Default output format: json
```

### Step 2.3: Upload to S3

```bash
# Upload the data file
aws s3 cp data/spotify_data.csv s3://bigdata-spotifyproject-1/data/spotify_data.csv

# Verify upload
aws s3 ls s3://bigdata-spotifyproject-1/data/ --human-readable
```

You should see your file with size.

---

## Part 3: Run Master Pipeline on EC2

### Step 3.1: Upload Scripts to EC2

```bash
# From your Mac
cd /Users/dewansh/Code/big_data

# Upload the master pipeline
scp -i ~/.ssh/spotify-analyser-key.pem master_pipeline.py ubuntu@YOUR_EC2_IP:~/spotify_analysis/

# Upload all scripts
scp -i ~/.ssh/spotify-analyser-key.pem -r scripts ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem -r ml_models ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem -r visualizations ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem -r utils ubuntu@YOUR_EC2_IP:~/spotify_analysis/
```

### Step 3.2: SSH into EC2

```bash
ssh -i ~/.ssh/spotify-analyser-key.pem ubuntu@YOUR_EC2_IP
```

### Step 3.3: Setup Environment on EC2

```bash
cd ~/spotify_analysis

# Activate virtual environment
source venv/bin/activate

# Verify AWS credentials
aws configure list
```

### Step 3.4: Run the Master Pipeline

**Full Pipeline (Everything Automated):**
```bash
# With 5M sample
python master_pipeline.py --sample 5000000

# With 20M sample
python master_pipeline.py --sample 20000000

# Full dataset (all rows)
python master_pipeline.py
```

**Skip Download (if data already on EC2):**
```bash
python master_pipeline.py --skip-download --sample 5000000
```

**Skip ML Training (Only EDA):**
```bash
python master_pipeline.py --skip-ml --sample 5000000
```

**Run Individual Steps:**
```bash
# Only data cleaning
python master_pipeline.py --step clean --sample 5000000

# Only visualizations
python master_pipeline.py --step viz --sample 500000

# Only ML - Popularity model
python master_pipeline.py --step ml-pop --sample 10000000
```

### Step 3.5: Monitor Progress

**In another terminal (while pipeline runs):**
```bash
# Watch logs
tail -f logs/pipeline_*.log

# Check memory usage
watch -n 5 free -h

# Check disk usage
df -h

# View pipeline summary
cat logs/pipeline_summary.txt
```

---

## Part 4: Download Results (Back to Mac)

### Step 4.1: Download Visualizations

```bash
# From your Mac
cd /Users/dewansh/Code/big_data

# Create results directory
mkdir -p results/visualizations results/models

# Download all visualizations
scp -i ~/.ssh/spotify-analyser-key.pem -r "ubuntu@YOUR_EC2_IP:~/spotify_analysis/visualizations/*.png" results/visualizations/
scp -i ~/.ssh/spotify-analyser-key.pem "ubuntu@YOUR_EC2_IP:~/spotify_analysis/visualizations/EXECUTIVE_SUMMARY.txt" results/

# Download ML models
scp -i ~/.ssh/spotify-analyser-key.pem -r "ubuntu@YOUR_EC2_IP:~/spotify_analysis/ml_models/*.pkl" results/models/

# Download logs
scp -i ~/.ssh/spotify-analyser-key.pem "ubuntu@YOUR_EC2_IP:~/spotify_analysis/logs/pipeline_summary.txt" results/
```

### Step 4.2: View Results

```bash
# Open visualizations
open results/visualizations/

# View summary
cat results/EXECUTIVE_SUMMARY.txt

# View pipeline summary
cat results/pipeline_summary.txt
```

---

## 📊 What the Pipeline Does

### Step 1: Download Data from S3
- Downloads `spotify_data.csv` from S3 to EC2
- Saves to `data/raw/`

### Step 2: Data Cleaning
- Removes duplicates
- Handles missing values
- Fixes data types
- Filters invalid records
- Saves to `data/processed/spotify_cleaned.csv`

### Step 3: Feature Engineering
- Creates derived features
- Normalizes data
- Encodes categorical variables
- Saves to `data/processed/spotify_features.parquet`

### Step 4: Exploratory Analysis
- Generates 9 insight visualizations:
  1. Playlist Category DNA
  2. Popularity Drivers
  3. Music Mood Landscape
  4. Workout vs Sleep
  5. Popularity Sweet Spot
  6. Follower Impact
  7. Tempo Zones
  8. Top Artists Profiles
  9. Feature Correlations
- Creates Executive Summary report

### Step 5: ML - Popularity Prediction
- Trains XGBoost model
- Feature importance analysis
- Model evaluation (R², MAE, RMSE)
- Saves model + visualization

### Step 6: ML - Recommendation System
- K-Means clustering
- Cosine similarity
- Creates song recommendation engine
- Saves model

### Step 7: Upload Results
- Uploads all results back to S3
- Results available at: `s3://bigdata-spotifyproject-1/results/`

---

## ⏱️ Expected Runtime

| Dataset Size | Data Gen | Pipeline (EC2 m5.2xlarge) |
|--------------|----------|---------------------------|
| 100K rows    | 30s      | 5 minutes                 |
| 1M rows      | 2 min    | 10 minutes                |
| 5M rows      | 5 min    | 30 minutes                |
| 20M rows     | 20 min   | 2 hours                   |
| 50M rows     | 1 hour   | 5 hours                   |

---

## 💾 Storage Requirements

| Dataset Size | CSV Size | Processed | Total Needed |
|--------------|----------|-----------|--------------|
| 100K rows    | 25 MB    | 30 MB     | 100 MB       |
| 1M rows      | 250 MB   | 300 MB    | 1 GB         |
| 5M rows      | 1.2 GB   | 1.5 GB    | 5 GB         |
| 20M rows     | 5 GB     | 6 GB      | 20 GB        |
| 50M rows     | 12 GB    | 15 GB     | 50 GB        |

---

## 🔧 Troubleshooting

### "Out of Memory" Error
```bash
# Reduce sample size
python master_pipeline.py --sample 2000000

# Or upgrade EC2 instance
```

### "S3 Access Denied"
```bash
# Check AWS credentials on EC2
aws configure list
aws sts get-caller-identity

# Verify IAM permissions
```

### "Module not found"
```bash
# Activate venv
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Pipeline Stuck
```bash
# Check if process is running
ps aux | grep python

# Check logs
tail -f logs/pipeline_*.log

# Check system resources
htop  # or top
free -h
df -h
```

### Resume After Failure
```bash
# If cleaning failed, start from cleaning
python master_pipeline.py --step clean --sample 5000000

# Then continue with features
python master_pipeline.py --step features

# Then visualizations
python master_pipeline.py --step viz --sample 500000
```

---

## 📚 Files Generated

```
~/spotify_analysis/
├── data/
│   ├── raw/
│   │   └── spotify_data.csv          # Raw data from S3
│   └── processed/
│       ├── spotify_cleaned.csv       # After step 2
│       └── spotify_features.parquet  # After step 3
├── visualizations/
│   ├── insight_*.png                 # 9 visualizations
│   └── EXECUTIVE_SUMMARY.txt         # Summary report
├── ml_models/
│   ├── popularity_model.pkl          # Trained model
│   ├── recommendation_model.pkl      # Trained model
│   └── *.png                         # Model results
└── logs/
    ├── pipeline_*.log                # Detailed logs
    └── pipeline_summary.txt          # Final summary
```

---

## 🎯 Recommended Workflow

### For Testing (Fast)
```bash
# 1. Generate small dataset
python3 generate_data_local.py --rows 100000 --output data/test.csv

# 2. Upload
aws s3 cp data/test.csv s3://bigdata-spotifyproject-1/data/spotify_data.csv

# 3. Run pipeline (5 minutes)
python master_pipeline.py --sample 100000 --skip-ml
```

### For Development (Medium)
```bash
# 1. Generate medium dataset
python3 generate_data_local.py --rows 5000000 --output data/spotify_data.csv

# 2. Upload
aws s3 cp data/spotify_data.csv s3://bigdata-spotifyproject-1/data/

# 3. Run full pipeline (30 minutes)
python master_pipeline.py --sample 5000000
```

### For Production/Demo (Large)
```bash
# 1. Generate large dataset
python3 generate_data_local.py --rows 20000000 --output data/spotify_data.csv

# 2. Upload
aws s3 cp data/spotify_data.csv s3://bigdata-spotifyproject-1/data/

# 3. Run full pipeline (2 hours)
python master_pipeline.py --sample 20000000
```

---

## ✅ Success Criteria

Pipeline is successful when:
- ✅ All 7 steps complete without errors
- ✅ 9 visualization PNGs generated
- ✅ EXECUTIVE_SUMMARY.txt created
- ✅ 2 ML models saved (.pkl files)
- ✅ Results uploaded to S3
- ✅ `pipeline_summary.txt` shows 0 failures

---

## 📞 Need Help?

Check these files for errors:
1. `logs/pipeline_summary.txt` - High-level summary
2. `logs/pipeline_*.log` - Detailed logs
3. Terminal output - Real-time progress

Common issues are usually:
- Memory (reduce `--sample`)
- AWS credentials (check `aws configure`)
- Missing files (check S3 upload)

