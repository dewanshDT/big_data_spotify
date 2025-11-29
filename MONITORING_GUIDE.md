# 📊 Monitoring Guide - Track Your Big Data Processing

## 🎯 Overview

All scripts now log their progress to both:
1. **Console** (what you see on screen)
2. **Log files** (saved in `logs/` directory)

---

## 📁 Log Files Location

```
~/spotify_analysis/logs/
├── data_cleaning_20241129_143022.log
├── feature_engineering_20241129_150533.log
├── popularity_model_20241129_153012.log
└── recommendation_model_20241129_160155.log
```

Each script creates a timestamped log file so you can review progress later.

---

## 🔍 How to Monitor Progress

### Option 1: Watch Logs in Real-Time (Recommended)

```bash
# In a separate SSH session or terminal, tail the logs
tail -f logs/*.log

# Or for a specific script
tail -f logs/data_cleaning*.log
```

### Option 2: Check Logs Periodically

```bash
# View last 50 lines
tail -n 50 logs/data_cleaning*.log

# Search for errors
grep -i error logs/*.log

# Search for completion
grep -i "complete" logs/*.log
```

### Option 3: Monitor System Resources

```bash
# Watch memory usage (updates every 5 seconds)
watch -n 5 free -h

# Check disk space
df -h

# Monitor CPU and RAM usage
htop  # or: top
```

---

## 📊 What Gets Logged

### Data Cleaning Script
```
2024-11-29 14:30:22 - INFO - Loading data from: data/raw/spotify_data.csv
2024-11-29 14:30:45 - INFO - Loaded 62,000,000 rows, 16 columns
2024-11-29 14:31:10 - INFO - Step 1: Removing duplicates...
2024-11-29 14:32:05 - INFO - Removed 125,430 duplicate rows
2024-11-29 14:32:06 - INFO - Step 2: Handling missing values...
2024-11-29 14:35:22 - INFO - Filled 1,245 missing values in 'energy' with median
2024-11-29 14:40:15 - INFO - Cleaning complete! Final rows: 61,874,570
2024-11-29 14:42:30 - INFO - Saved as CSV: data/processed/spotify_cleaned.csv
2024-11-29 14:42:35 - INFO - Saved as Parquet: data/processed/spotify_cleaned.parquet
```

### Feature Engineering Script
```
2024-11-29 15:05:33 - INFO - Loading data...
2024-11-29 15:06:12 - INFO - Loaded 61,874,570 rows from parquet
2024-11-29 15:06:15 - INFO - Creating duration features...
2024-11-29 15:08:22 - INFO - Creating mood/energy categories...
2024-11-29 15:12:45 - INFO - Creating popularity tiers...
2024-11-29 15:15:10 - INFO - Feature engineering complete!
2024-11-29 15:15:11 - INFO - Original columns: 16
2024-11-29 15:15:11 - INFO - Total columns now: 28
```

### ML Model Training
```
2024-11-29 15:30:12 - INFO - Training XGBoost model...
2024-11-29 15:45:33 - INFO - Model training complete!
2024-11-29 15:45:35 - INFO - Train MAE: 14.23 | Test MAE: 15.87
2024-11-29 15:45:35 - INFO - Train RMSE: 18.45 | Test RMSE: 19.92
2024-11-29 15:45:35 - INFO - Train R²: 0.425 | Test R²: 0.392
2024-11-29 15:45:40 - INFO - Model saved to: ml_models/popularity_model.pkl
```

---

## 🚨 Error Detection

### Check for Errors
```bash
# Find any errors in logs
grep -i "error" logs/*.log

# Find warnings
grep -i "warning" logs/*.log

# Find out of memory issues
grep -i "memory" logs/*.log
```

### Common Issues and Solutions

**Out of Memory:**
```bash
# Restart with chunked processing
python scripts/03_data_cleaning.py --chunk-size 100000
```

**Process Killed:**
```bash
# Check system logs
dmesg | tail

# Usually means out of memory - use smaller chunks or larger instance
```

---

## 📈 Progress Estimation

### Typical Timeline (m5.xlarge, 62M rows):

| Task | Duration | Log Indicators |
|------|----------|----------------|
| Download from S3 | 15 min | "Download complete" |
| Data Cleaning | 45 min | "Cleaning complete! Final rows: X" |
| Feature Engineering | 45 min | "Feature engineering complete!" |
| Popularity Model | 60 min | "Model training complete!" |
| Recommendation Model | 60 min | "Created X clusters" |
| Visualizations | 30 min | "All Visualizations Complete!" |
| Upload to S3 | 10 min | "Upload Complete! X files uploaded" |

**Total: ~4-5 hours**

---

## 🔄 Using `screen` for Long Processes

Highly recommended to avoid losing progress if SSH disconnects!

```bash
# Start a screen session
screen -S spotify

# Run your scripts
python scripts/03_data_cleaning.py --input data/raw/spotify_data.csv

# Detach (keeps running in background)
# Press: Ctrl+A, then D

# Check running screens
screen -ls

# Reattach to see progress
screen -r spotify

# Kill a screen session
screen -X -S spotify quit
```

---

## 📊 Real-Time Dashboard (Optional)

### Create a simple monitoring script:

```bash
# Create monitor.sh
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=========================================="
    echo "Spotify Analysis - Live Monitor"
    echo "=========================================="
    echo ""
    echo "Current Time: $(date)"
    echo ""
    echo "Memory Usage:"
    free -h | grep -v "+"
    echo ""
    echo "Disk Space:"
    df -h | grep -E "Filesystem|/dev/root"
    echo ""
    echo "Latest Log Entries:"
    tail -n 10 logs/*.log 2>/dev/null | tail -n 10
    echo ""
    echo "=========================================="
    sleep 10
done
EOF

chmod +x monitor.sh
./monitor.sh
```

---

## 📧 Get Notified When Complete

### Email notification (if you have `mail` configured):

```bash
python scripts/03_data_cleaning.py && echo "Data cleaning complete!" | mail -s "Spotify Analysis Update" your@email.com
```

### Slack/Discord webhook (if you have one):

```bash
python scripts/03_data_cleaning.py && curl -X POST -H 'Content-type: application/json' --data '{"text":"Data cleaning complete!"}' YOUR_WEBHOOK_URL
```

---

## ✅ Checklist for Monitoring

- [ ] Start script in `screen` session
- [ ] Open second SSH session for monitoring
- [ ] Run `tail -f logs/*.log` in second session
- [ ] Monitor `free -h` periodically
- [ ] Check disk space with `df -h`
- [ ] Set reminders to check progress every hour
- [ ] Review logs for errors: `grep -i error logs/*.log`

---

## 📝 Example Monitoring Session

**Terminal 1 (Running Scripts):**
```bash
screen -S spotify
source ~/spotify_analysis/venv/bin/activate
python scripts/03_data_cleaning.py --input data/raw/spotify_data.csv
# Ctrl+A, D to detach
```

**Terminal 2 (Monitoring):**
```bash
# Watch logs
tail -f ~/spotify_analysis/logs/*.log
```

**Terminal 3 (System Monitoring):**
```bash
# Watch resources
watch -n 5 'free -h; echo ""; df -h | grep root'
```

---

**All logs are timestamped and saved for later review!** 📊

You can always check what happened by reading the log files, even after the scripts complete.

