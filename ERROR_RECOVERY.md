# 🔧 Error Recovery Guide

Common errors and how to recover from them.

---

## Data Generation Errors

### "Insufficient disk space"
```bash
# Check available space
df -h

# Clear temp files
rm -rf /tmp/*

# Or reduce dataset size
python3 generate_data_local.py --rows 1000000  # Generate smaller dataset
```

### "No write permission"
```bash
# Check permissions
ls -la data/

# Fix permissions
chmod 755 data/
```

### "Generation interrupted"
```bash
# Check if partial file exists
ls -lh data/spotify_data.csv

# Delete partial file and restart
rm data/spotify_data.csv
python3 generate_data_local.py --rows 5000000
```

---

## Pipeline Errors

### Step 1: "S3 Access Denied"
```bash
# Check AWS credentials
aws sts get-caller-identity

# Reconfigure if needed
aws configure

# Verify bucket access
aws s3 ls s3://bigdata-spotifyproject-1/

# If persists, skip download if file exists locally
python master_pipeline.py --skip-download
```

### Step 2: "Data Cleaning - Out of Memory"
```bash
# Option 1: Reduce sample size
python master_pipeline.py --sample 1000000

# Option 2: Reduce chunk size
# Edit scripts/03_data_cleaning.py
# Change: --chunk-size 500000 → 250000

# Option 3: Upgrade EC2 instance
# m5.xlarge (16GB) → m5.2xlarge (32GB)
```

### Step 2: "Data Cleaning - File Not Found"
```bash
# Check if raw data exists
ls -lh data/raw/spotify_data.csv

# If not, download it
python scripts/download_from_s3.py

# Or upload from local
aws s3 cp data/spotify_data.csv s3://bigdata-spotifyproject-1/data/
```

### Step 3: "Feature Engineering - KeyError"
```bash
# Usually means cleaned data is corrupted
# Check cleaned data
head -20 data/processed/spotify_cleaned.csv

# If corrupted, re-run cleaning
python scripts/03_data_cleaning.py --chunk-size 500000

# Then retry feature engineering
python master_pipeline.py --step features
```

### Step 4: "Visualization - matplotlib backend error"
```bash
# Already fixed in code with matplotlib.use('Agg')
# If still occurs, check:
echo $DISPLAY  # Should be empty on headless server

# Manual fix:
export MPLBACKEND=Agg
python visualizations/insights_analysis.py --sample 500000
```

### Step 5/6: "ML - Out of Memory"
```bash
# Option 1: Reduce sample size
python ml_models/popularity_model.py --sample 2000000

# Option 2: Run with pipeline
python master_pipeline.py --sample 2000000

# Option 3: Skip ML
python master_pipeline.py --skip-ml
```

### Step 7: "Upload to S3 Failed"
```bash
# Check AWS credentials
aws configure list

# Manual upload
aws s3 sync visualizations/ s3://bigdata-spotifyproject-1/results/visualizations/ --exclude "*.py"
aws s3 sync ml_models/ s3://bigdata-spotifyproject-1/results/models/ --exclude "*.py"
```

---

## Recovery Strategies

### Resume from Specific Step

If pipeline failed at step 4:
```bash
# Steps 1-3 already complete, start from step 4
python master_pipeline.py --skip-download --step viz

# Or run remaining steps
python master_pipeline.py --step viz
python master_pipeline.py --step ml-pop
python master_pipeline.py --step ml-rec
python master_pipeline.py --step upload
```

### Clean Restart

```bash
# Remove all processed data
rm -rf data/processed/*

# Remove logs
rm -rf logs/*

# Start fresh
python master_pipeline.py
```

### Partial Success Strategy

If some steps succeed:
```bash
# Check what completed
cat logs/pipeline_summary.txt

# Run only failed steps
# Example: if only ML failed
python ml_models/popularity_model.py --sample 5000000
python ml_models/recommendation_model.py --sample 3000000
```

---

## Debugging Commands

### Check Pipeline Status
```bash
# View summary
cat logs/pipeline_summary.txt

# View detailed logs
ls -lh logs/
tail -100 logs/pipeline_*.log
```

### Check Data at Each Stage
```bash
# Raw data
ls -lh data/raw/
head -20 data/raw/spotify_data.csv

# Cleaned data
ls -lh data/processed/
head -20 data/processed/spotify_cleaned.csv

# Features
ls -lh data/processed/*.parquet
```

### Check System Resources
```bash
# Memory usage
free -h

# Disk usage
df -h

# Running processes
ps aux | grep python

# Kill stuck process
pkill -f master_pipeline
```

---

## Emergency Procedures

### Pipeline Frozen/Hung

```bash
# 1. Open new SSH session
ssh -i ~/.ssh/spotify-analyser-key.pem ubuntu@YOUR_EC2_IP

# 2. Check what's running
ps aux | grep python

# 3. Check memory
free -h
# If swap is 100%, it's out of memory

# 4. Kill process
pkill -f master_pipeline

# 5. Restart with smaller sample
python master_pipeline.py --sample 1000000
```

### EC2 Instance Unresponsive

```bash
# From AWS Console:
# 1. Go to EC2 Dashboard
# 2. Select instance
# 3. Actions → Instance State → Stop
# 4. Wait 1 minute
# 5. Actions → Instance State → Start
# 6. Get new IP address
# 7. SSH in and resume
```

### Data Corrupted

```bash
# 1. Verify corruption
head -100 data/processed/spotify_cleaned.csv

# 2. Delete corrupted files
rm data/processed/*

# 3. Re-download raw data
python scripts/download_from_s3.py

# 4. Restart from cleaning
python scripts/03_data_cleaning.py --chunk-size 500000
```

---

## Prevention Best Practices

### Before Running Pipeline

```bash
# 1. Check available resources
free -h
df -h

# 2. Verify data exists
ls -lh data/raw/

# 3. Test with small sample first
python master_pipeline.py --sample 100000 --skip-ml

# 4. If successful, run full
python master_pipeline.py --sample 5000000
```

### During Pipeline Execution

```bash
# Monitor in separate terminal
watch -n 5 'free -h && df -h'

# Or
htop  # Interactive process monitor
```

### After Errors

```bash
# Always check logs first
cat logs/pipeline_summary.txt
tail -50 logs/pipeline_*.log

# Document what went wrong
# Adjust parameters accordingly
```

---

## Quick Reference

| Error Type | Quick Fix |
|------------|-----------|
| Out of Memory | `--sample 1000000` |
| Disk Full | `df -h && rm -rf /tmp/*` |
| AWS Access | `aws configure` |
| File Not Found | `--skip-download` or re-upload |
| Module Not Found | `pip install -r requirements.txt` |
| Process Hung | `pkill -f master_pipeline` |
| Corruption | Delete processed/, restart |
| Timeout | Increase timeout or reduce sample |

---

## Getting Help

### Collect Debug Info

```bash
# System info
uname -a
free -h
df -h

# Python info
python --version
pip list

# Pipeline status
cat logs/pipeline_summary.txt
tail -100 logs/pipeline_*.log

# AWS status
aws sts get-caller-identity
aws s3 ls s3://bigdata-spotifyproject-1/
```

### Share this with support/debugging

---

## Success Checklist

After recovery, verify:
- [ ] All 7 steps show ✅ in summary
- [ ] 9 PNG visualizations exist
- [ ] EXECUTIVE_SUMMARY.txt created
- [ ] 2 .pkl model files saved
- [ ] Files uploaded to S3
- [ ] No ❌ in pipeline_summary.txt

