# 🚀 Quick Start Guide

Get up and running with Spotify data analysis on AWS in under 30 minutes!

## Prerequisites

- AWS account
- Python 3.8+
- 12.3 GB Spotify CSV file

## Step-by-Step Workflow

### 1. Setup Local Environment (5 min)

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key  
# Region: us-east-1
```

### 2. Create S3 Bucket (2 min)

```bash
# Create bucket
python scripts/create_s3_bucket.py

# Follow prompts and confirm
```

### 3. Upload Data to S3 (10-30 min depending on internet speed)

```bash
# Upload your CSV
python scripts/01_upload_to_s3.py --file /path/to/spotify_data.csv
```

### 4. Launch EC2 Instance (5 min)

**Via AWS Console:**
1. Go to EC2 Dashboard
2. Click "Launch Instance"
3. Choose: Ubuntu Server 22.04 LTS
4. Instance type: `m5.4xlarge` (16 vCPU, 64 GB RAM)
5. Storage: 100 GB
6. Security Group: 
   - Allow SSH (port 22) from your IP
   - Allow port 8888 (for Jupyter) from your IP
7. Create/select key pair
8. Launch!

### 5. Connect & Setup EC2 (10 min)

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Run setup script
wget https://raw.githubusercontent.com/YOUR_REPO/scripts/02_setup_ec2.sh
bash 02_setup_ec2.sh

# Follow the interactive prompts
```

### 6. Download Data on EC2 (10-20 min)

```bash
# Activate virtual environment
source ~/spotify_analysis/venv/bin/activate

# Download from S3
python scripts/download_from_s3.py
```

### 7. Process Data (30-60 min)

```bash
# Clean data
python scripts/03_data_cleaning.py

# Engineer features
python scripts/04_feature_engineering.py
```

### 8. Analyze & Train Models (1-2 hours)

```bash
# Option A: Use Jupyter for interactive analysis
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
# Access at: http://YOUR_EC2_IP:8888

# Option B: Run ML models directly
python ml_models/popularity_model.py
python ml_models/recommendation_model.py

# Create visualizations
python visualizations/create_charts.py
```

### 9. Save Results to S3 (5 min)

```bash
# Upload all results back to S3
python scripts/05_save_results.py --all
```

### 10. Stop EC2 & Download Results Locally (5 min)

```bash
# From your local machine
aws s3 sync s3://spotify-data-analysis-bucket/results ./results
aws s3 sync s3://spotify-data-analysis-bucket/visualizations ./visualizations

# Stop EC2 instance (to avoid charges)
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID
```

---

## Estimated Costs

| Item | Cost |
|------|------|
| S3 Storage (20 GB, 1 month) | $0.50 |
| S3 Data Transfer | $1.00 |
| EC2 m5.4xlarge (10 hours) | $16.80 |
| **Total** | **~$18.30** |

**To reduce costs:**
- Use Spot Instances (70% cheaper)
- Delete S3 data after project
- Stop EC2 when not in use

---

## Troubleshooting

### Out of Memory on EC2?
```bash
# Use larger instance: m5.8xlarge or m5.12xlarge
# Or process data in chunks (already implemented in scripts)
```

### Can't connect to Jupyter?
```bash
# Check security group allows port 8888
# Verify Jupyter is running: ps aux | grep jupyter
```

### S3 upload too slow?
```bash
# Compress first
gzip spotify_data.csv
python scripts/01_upload_to_s3.py --file spotify_data.csv.gz
```

---

## What You'll Get

✅ Cleaned and processed dataset  
✅ 10+ visualizations showing music trends  
✅ Popularity prediction model (MAE ~15)  
✅ Song recommendation system  
✅ Comprehensive analysis report  

---

## Next Steps

1. Review visualizations in `visualizations/` folder
2. Experiment with ML models
3. Try different feature engineering approaches
4. Build a web app to showcase results!

---

**Questions?** Check the main README.md for detailed documentation.



