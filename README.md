# 🎵 Spotify Music Trends Analysis - Simple AWS Pipeline

**Goal**: Analyze 12.3 GB Spotify dataset to understand what makes songs popular, genre evolution, and mood/energy patterns.

## 🏗️ Simple Architecture

```
Local Machine → S3 (Storage) → EC2 (Processing) → Results
                                    ↓
                            All Analysis Happens Here:
                            - Data Cleaning (Pandas)
                            - Feature Engineering
                            - ML Models (Scikit-learn)
                            - Visualizations
```

**No Lambda. No Glue. No Athena. Just the essentials.**

---

## 📁 Project Structure

```
big_data/
├── data/
│   └── sample/                 # Sample data for testing
├── scripts/
│   ├── 01_upload_to_s3.py     # Upload CSV to S3
│   ├── 02_setup_ec2.sh        # EC2 bootstrap script
│   ├── 03_data_cleaning.py    # Clean and process data
│   ├── 04_feature_engineering.py
│   └── 05_save_results.py     # Save back to S3
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_popularity_prediction.ipynb
│   └── 03_music_recommendations.ipynb
├── ml_models/
│   ├── popularity_model.py
│   └── recommendation_model.py
├── visualizations/
│   └── create_charts.py
├── config.yaml                 # Simple config
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### **Step 1: Setup AWS**

```bash
# Install AWS CLI
pip install awscli boto3

# Configure credentials
aws configure
# Enter: Access Key, Secret Key, Region (e.g., us-east-1)
```

### **Step 2: Create S3 Bucket**

```bash
# Create bucket
aws s3 mb s3://spotify-data-analysis-bucket

# Or use Python script
python scripts/create_s3_bucket.py
```

### **Step 3: Upload Data to S3**

```bash
# Upload your 12.3 GB CSV
python scripts/01_upload_to_s3.py --file /path/to/spotify_data.csv
```

### **Step 4: Launch EC2 Instance**

```bash
# Option A: Use AWS Console
# - Go to EC2 Dashboard
# - Launch Instance
# - Choose: Ubuntu 22.04
# - Instance Type: m5.4xlarge (16 vCPU, 64 GB RAM)
# - Storage: 100 GB
# - Create/select key pair
# - Launch

# Option B: Use AWS CLI (see docs/ec2_setup.md)
```

### **Step 5: Connect to EC2 & Setup Environment**

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Run setup script (installs Python, packages, etc.)
bash setup_ec2.sh
```

### **Step 6: Run Analysis on EC2**

```bash
# Download code to EC2
git clone <your-repo> # or scp your files

# Run the pipeline
python scripts/03_data_cleaning.py
python scripts/04_feature_engineering.py

# Start Jupyter for interactive analysis
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Access from your browser: http://your-ec2-ip:8888
```

### **Step 7: Train ML Models**

```bash
python ml_models/popularity_model.py
python ml_models/recommendation_model.py
```

### **Step 8: Create Visualizations**

```bash
python visualizations/create_charts.py
```

### **Step 9: Save Results & Stop EC2**

```bash
# Upload results to S3
python scripts/05_save_results.py

# Stop EC2 from AWS Console to avoid charges
# Or: aws ec2 stop-instances --instance-ids i-xxxxx
```

---

## 📊 Dataset Features

**Audio Features:**

- `energy`: Intensity and activity (0-1)
- `danceability`: How suitable for dancing (0-1)
- `valence`: Musical positivity/mood (0-1)
- `tempo`: Beats per minute
- `loudness`: Overall loudness in dB
- `speechiness`: Presence of spoken words
- `acousticness`: Confidence of acoustic sound
- `instrumentalness`: Predicts no vocals
- `liveness`: Presence of audience

**Metadata:**

- `track_name`, `artist_name`, `album_name`
- `popularity`: 0-100 score
- `duration_ms`: Track length
- `year`: Release year
- `playlist_name`, `playlist_category`

---

## 🎯 Analysis Goals

### 1. **Exploratory Data Analysis**

- Distribution of audio features
- Trends over time (2010-2020)
- Correlation between features
- Playlist category comparisons

### 2. **What Makes Songs Popular?**

- Feature importance for popularity
- Correlation analysis
- Regression modeling

### 3. **Music Mood Analysis**

- High energy vs. chill tracks
- Workout vs. focus playlists
- Mood changes over years

### 4. **ML Models**

- **Popularity Prediction**: Predict if a song will be popular
- **Recommendation System**: Suggest similar tracks
- **Playlist Classification**: Auto-categorize playlists

---

## 💰 Cost Estimate

| Service            | Usage                  | Monthly Cost |
| ------------------ | ---------------------- | ------------ |
| S3 Storage (20 GB) | Store data & results   | $0.50        |
| EC2 m5.4xlarge     | 20 hours processing    | $33.60       |
| Data Transfer      | Minimal                | $1.00        |
| **Total**          | For project completion | **~$35**     |

**Tips to Save Money:**

- Use **Spot Instances** (70% cheaper)
- Stop EC2 when not in use
- Delete S3 data after project
- Use **AWS Free Tier** if eligible

---

## 🔧 Key Python Libraries

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost
- **AWS**: boto3
- **Big Data (optional)**: pyspark (if data is too large)

---

## 📈 Expected Insights

1. **Workout playlists** have highest energy and tempo
2. **Chill playlists** have high acousticness, low energy
3. **Popular songs** tend to have high danceability and energy
4. **Temporal trends**: Music has become more energetic over time
5. **Valence (mood)** varies significantly by genre and region

---

## 🐛 Troubleshooting

**Out of Memory on EC2?**

- Use larger instance type (m5.8xlarge)
- Process data in chunks with `pd.read_csv(chunksize=10000)`
- Use PySpark for distributed processing

**Slow S3 uploads?**

- Use multipart upload for large files
- Increase network bandwidth
- Compress data before upload

**EC2 connection issues?**

- Check security group (port 22 for SSH, 8888 for Jupyter)
- Verify key pair permissions: `chmod 400 your-key.pem`
- Check EC2 public IP address

---

## 📚 Additional Resources

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## ✅ Checklist

- [ ] AWS account created and configured
- [ ] S3 bucket created
- [ ] Data uploaded to S3
- [ ] EC2 instance launched and configured
- [ ] Environment setup complete
- [ ] Data cleaning done
- [ ] Exploratory analysis complete
- [ ] ML models trained
- [ ] Visualizations created
- [ ] Results saved to S3
- [ ] EC2 instance stopped/terminated

---

**Let's keep it simple and get insights! 🚀**


