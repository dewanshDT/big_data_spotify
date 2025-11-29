# 🎯 Complete Workflow - Synthetic Data Pipeline

## Overview

This guide walks you through the **complete end-to-end workflow** with all error handling in place.

---

## 🛡️ Error Handling Features

### ✅ What's Protected:

1. **Pre-flight Validation**
   - Disk space checks
   - Memory availability
   - Python version verification
   - Package installation verification
   - AWS credentials validation
   - File permissions

2. **Graceful Degradation**
   - Critical steps (cleaning, features) must pass
   - Non-critical steps (viz, ML) can fail without stopping
   - Partial success is tracked and reported

3. **Recovery Options**
   - Resume from any step
   - Skip completed steps
   - Adjust sample sizes on failure
   - Clear error messages with solutions

4. **Comprehensive Logging**
   - Timestamped execution
   - Step-by-step progress
   - Detailed error messages
   - Execution time tracking
   - Final summary report

---

## 📋 Workflow Steps

### Step 0: Validate Setup (RECOMMENDED)

```bash
cd /Users/dewansh/Code/big_data

# Run validation script
python3 validate_setup.py
```

**What it checks:**
- ✅ Python 3.7+
- ✅ All required packages
- ✅ Directory structure
- ✅ All scripts present and valid
- ✅ config.yaml correct
- ✅ AWS credentials
- ✅ Disk space (>10 GB recommended)
- ✅ Memory (>4 GB recommended)

**Output:**
```
🔍 SETUP VALIDATION
============================================================
🐍 Python Environment
✅ Python 3.11.5
✅ Running in virtual environment

📦 Python Packages
✅ pandas          - Data processing
✅ numpy           - Numerical computations
✅ boto3           - AWS SDK
...

📋 VALIDATION SUMMARY
✅ All checks passed! Ready to run pipeline.
```

---

### Step 1: Generate Synthetic Data (Local)

```bash
# Quick test (100K rows, 30 seconds)
python3 generate_data_local.py --rows 100000 --output data/test.csv

# Production (5M rows, 5 minutes)
python3 generate_data_local.py --rows 5000000 --output data/spotify_data.csv
```

**Error Handling:**
- ✅ Validates input arguments
- ✅ Checks disk space before starting
- ✅ Checks write permissions
- ✅ Processes in chunks (memory safe)
- ✅ Validates each chunk
- ✅ Handles keyboard interrupts
- ✅ Verifies final file

**If Error Occurs:**
```bash
# Check error message
# Common issues:
# - Insufficient disk space → Free up space or reduce --rows
# - No write permission → chmod 755 data/
# - Interrupted → Delete partial file, restart
```

---

### Step 2: Verify Generated Data

```bash
# Check file size
ls -lh data/spotify_data.csv

# View first rows
head -20 data/spotify_data.csv

# Count rows (should match --rows + 1 header)
wc -l data/spotify_data.csv

# Quick quality check
python3 visualizations/raw_data_insights.py --sample 100000
```

**Expected:**
- 22 columns
- No uniform distributions (should see curves)
- Correlations between features (energy ↔ loudness ~0.7)

---

### Step 3: Upload to S3

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Upload data
aws s3 cp data/spotify_data.csv s3://bigdata-spotifyproject-1/data/spotify_data.csv

# Verify upload
aws s3 ls s3://bigdata-spotifyproject-1/data/ --human-readable
```

**Error Handling:**
- If "Access Denied": Run `aws configure`
- If "Bucket not found": Create bucket or update bucket name
- If upload slow: Use `--storage-class STANDARD_IA` for cheaper storage

---

### Step 4: Setup EC2

```bash
# SSH into EC2
ssh -i ~/.ssh/spotify-analyser-key.pem ubuntu@YOUR_EC2_IP

# Navigate to project
cd ~/spotify_analysis

# Activate virtual environment
source venv/bin/activate

# Verify AWS credentials on EC2
aws sts get-caller-identity
```

---

### Step 5: Upload Scripts to EC2

```bash
# From your Mac, upload all scripts
cd /Users/dewansh/Code/big_data

scp -i ~/.ssh/spotify-analyser-key.pem master_pipeline.py ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem validate_setup.py ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem -r scripts ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem -r ml_models ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem -r visualizations ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem -r utils ubuntu@YOUR_EC2_IP:~/spotify_analysis/
scp -i ~/.ssh/spotify-analyser-key.pem config.yaml ubuntu@YOUR_EC2_IP:~/spotify_analysis/
```

---

### Step 6: Validate EC2 Setup

```bash
# On EC2, run validation
cd ~/spotify_analysis
source venv/bin/activate
python validate_setup.py
```

**If validation fails:**
- Missing packages: `pip install -r requirements.txt`
- Missing directories: Pipeline will create them
- AWS credentials: `aws configure`
- Low memory: Upgrade instance or reduce sample size

---

### Step 7: Run Master Pipeline

**Option A: Full Pipeline (Recommended)**
```bash
# With 5M sample (30 minutes)
python master_pipeline.py --sample 5000000

# Monitor in another terminal
tail -f logs/pipeline_*.log
```

**Option B: Test Run First**
```bash
# Quick test with 100K sample (5 minutes)
python master_pipeline.py --sample 100000 --skip-ml

# If successful, run full
python master_pipeline.py --sample 5000000
```

**Option C: Step by Step**
```bash
# Download data
python master_pipeline.py --step download

# Clean data
python master_pipeline.py --step clean --sample 5000000

# Features
python master_pipeline.py --step features

# Visualizations
python master_pipeline.py --step viz --sample 500000

# ML models
python master_pipeline.py --step ml-pop --sample 10000000
python master_pipeline.py --step ml-rec --sample 5000000

# Upload results
python master_pipeline.py --step upload
```

---

### Step 8: Monitor Execution

**In Real-Time:**
```bash
# Terminal 1: Run pipeline
python master_pipeline.py --sample 5000000

# Terminal 2: Monitor
tail -f logs/pipeline_*.log

# Terminal 3: Watch resources
watch -n 5 'free -h && df -h'
```

**Pipeline Output:**
```
============================================================
🎵 SPOTIFY BIG DATA ANALYSIS - MASTER PIPELINE
============================================================
[2024-01-15 10:00:00] 📍 Running: Step 1: Download Data from S3
[2024-01-15 10:00:05] ✅ Step 1 completed in 5.2s

[2024-01-15 10:00:05] 📍 Running: Step 2: Data Cleaning & Preprocessing
[2024-01-15 10:05:30] ✅ Step 2 completed in 325.1s
...
```

---

### Step 9: Handle Errors (If Any)

**If Pipeline Fails:**

1. **Check the summary:**
```bash
cat logs/pipeline_summary.txt
```

2. **Identify the failed step:**
```
❌ Failed Steps:
  1. Step 2: Data Cleaning & Preprocessing - Exit code 137
```

3. **Look up the error:**
```bash
# Exit code 137 = Out of Memory
# Solution: Reduce sample size or upgrade instance
python master_pipeline.py --sample 2000000
```

4. **Common Errors and Solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| Exit code 137 | Out of memory | Reduce `--sample` or upgrade EC2 |
| Exit code 1 | Script error | Check logs, fix script issue |
| S3 Access Denied | AWS credentials | Run `aws configure` |
| File not found | Missing data | Re-run download step |
| Module not found | Missing package | `pip install <package>` |

5. **Resume from failure point:**
```bash
# If failed at step 4, resume from there
python master_pipeline.py --skip-download --step viz
```

---

### Step 10: Verify Results

```bash
# Check what was generated
ls -lh visualizations/*.png
ls -lh ml_models/*.pkl
cat visualizations/EXECUTIVE_SUMMARY.txt

# Check summary
cat logs/pipeline_summary.txt
```

**Expected Files:**
- 9 PNG visualizations (insight_1 through insight_9)
- 1 EXECUTIVE_SUMMARY.txt
- 2 PKL model files
- pipeline_summary.txt showing all ✅

---

### Step 11: Download Results

```bash
# From your Mac
cd /Users/dewansh/Code/big_data
mkdir -p results/{visualizations,models,reports}

# Download visualizations
scp -i ~/.ssh/spotify-analyser-key.pem "ubuntu@YOUR_EC2_IP:~/spotify_analysis/visualizations/*.png" results/visualizations/

# Download models
scp -i ~/.ssh/spotify-analyser-key.pem "ubuntu@YOUR_EC2_IP:~/spotify_analysis/ml_models/*.pkl" results/models/

# Download reports
scp -i ~/.ssh/spotify-analyser-key.pem ubuntu@YOUR_EC2_IP:~/spotify_analysis/visualizations/EXECUTIVE_SUMMARY.txt results/reports/
scp -i ~/.ssh/spotify-analyser-key.pem ubuntu@YOUR_EC2_IP:~/spotify_analysis/logs/pipeline_summary.txt results/reports/

# View results
open results/visualizations/
cat results/reports/EXECUTIVE_SUMMARY.txt
```

---

## 🎯 Success Criteria

Your pipeline is successful when:

- [ ] `validate_setup.py` passes all checks
- [ ] Data generated without errors
- [ ] All 7 pipeline steps show ✅
- [ ] 9 PNG files in visualizations/
- [ ] EXECUTIVE_SUMMARY.txt created
- [ ] 2 ML models (.pkl) saved
- [ ] `pipeline_summary.txt` shows 0 critical failures
- [ ] Results downloaded to local machine

---

## 🔄 Iterative Development

### First Run (Testing)
```bash
# 1. Generate small dataset
python3 generate_data_local.py --rows 100000

# 2. Upload
aws s3 cp data/spotify_data.csv s3://YOUR_BUCKET/data/

# 3. Test pipeline (skip ML for speed)
python master_pipeline.py --sample 100000 --skip-ml

# 4. Verify outputs look good
```

### Second Run (Development)
```bash
# 1. Generate medium dataset
python3 generate_data_local.py --rows 1000000

# 2. Upload
aws s3 cp data/spotify_data.csv s3://YOUR_BUCKET/data/

# 3. Run full pipeline
python master_pipeline.py --sample 1000000

# 4. Analyze results, tune parameters
```

### Final Run (Production)
```bash
# 1. Generate large dataset
python3 generate_data_local.py --rows 20000000

# 2. Upload
aws s3 cp data/spotify_data.csv s3://YOUR_BUCKET/data/

# 3. Run full pipeline with larger samples
python master_pipeline.py --sample 20000000

# 4. Present results
```

---

## 📞 Emergency Procedures

### Pipeline Frozen
```bash
# 1. Check if running
ps aux | grep python

# 2. Check resources
free -h
df -h

# 3. Kill if needed
pkill -f master_pipeline

# 4. Resume from last successful step
python master_pipeline.py --step <next_step>
```

### Out of Memory
```bash
# Immediate: Kill process
pkill -f master_pipeline

# Solution: Reduce sample size
python master_pipeline.py --sample 1000000

# Or: Upgrade instance (AWS Console)
# m5.xlarge → m5.2xlarge
```

### EC2 Connection Lost
```bash
# If SSH times out, the pipeline is still running
# Just reconnect and monitor:
ssh -i ~/.ssh/spotify-analyser-key.pem ubuntu@YOUR_EC2_IP
cd ~/spotify_analysis
tail -f logs/pipeline_*.log
```

---

## 💡 Tips for Success

1. **Always validate first**: `python3 validate_setup.py`
2. **Start small**: Test with 100K rows first
3. **Monitor resources**: Use `htop` or `watch`
4. **Use screen/tmux**: For persistent sessions
5. **Check logs frequently**: `tail -f logs/*.log`
6. **Save intermediate results**: Don't delete processed data
7. **Document issues**: Keep notes on what failed and why
8. **Budget time**: 5M rows takes ~30 min, 20M takes ~2 hrs

---

## 📚 Reference Documents

- `QUICKSTART_SYNTHETIC.md` - Quick start guide
- `ERROR_RECOVERY.md` - Error handling guide
- `SYNTHETIC_DATA_GUIDE.md` - Data generation details
- `MONITORING_GUIDE.md` - Resource monitoring
- `EC2_SIZING_GUIDE.md` - Instance selection

---

## ✅ Final Checklist

Before considering the project complete:

- [ ] Validated setup passes
- [ ] Data generated with realistic distributions
- [ ] Pipeline runs without critical errors
- [ ] All visualizations generated
- [ ] ML models trained and saved
- [ ] Results downloaded
- [ ] Executive summary reviewed
- [ ] Project documented
- [ ] Ready for presentation

**Congratulations! Your pipeline is production-ready! 🎉**

