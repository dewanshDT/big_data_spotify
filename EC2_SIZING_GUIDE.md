# 🖥️ EC2 Instance Sizing Guide - Budget-Friendly Options

## Your Data: 13.19 GB CSV, 62 Million Rows

When loaded into Pandas: **~20-30 GB RAM needed**

---

## ❌ Bad News: Free Tier Won't Work

**AWS Free Tier instances:**
- `t2.micro`: 1 GB RAM ❌ (Too small)
- `t3.micro`: 1 GB RAM ❌ (Too small)

Your dataset is too large for free tier instances. **BUT** we have budget-friendly alternatives!

---

## ✅ Budget-Friendly Solutions

### **Option 1: Chunked Processing on Small Instance** 💰 Best Value!

**Instance:** `m5.large` (8 GB RAM)
- **Cost**: $0.096/hour → **~$2** for entire project (20 hours)
- **Works?** ✅ YES with chunked processing (already in scripts!)

```bash
# The scripts already support chunking!
python scripts/03_data_cleaning.py  # Auto-chunks if needed
python scripts/04_feature_engineering.py  # Works with chunks
```

**How it works:**
- Reads CSV in 100K row chunks (configurable)
- Processes each chunk
- Saves incrementally
- Total time: 4-6 hours (slower but works!)

---

### **Option 2: Medium Instance with Chunking** 💰💰 Good Balance

**Instance:** `m5.xlarge` (16 GB RAM)
- **Cost**: $0.192/hour → **~$4** for project (20 hours)
- **Works?** ✅ YES, faster than m5.large
- **Processing time**: 2-3 hours

---

### **Option 3: Large Instance (Original Recommendation)** 💰💰💰

**Instance:** `m5.4xlarge` (64 GB RAM)
- **Cost**: $0.768/hour → **~$15** for project (20 hours)
- **Works?** ✅ YES, loads entire dataset in memory
- **Processing time**: 1-2 hours
- **Only needed if**: You want fastest processing

---

### **Option 4: Spot Instances (70% Cheaper!)** 💰 RECOMMENDED!

Use **Spot Instances** for any of the above:

| Instance | On-Demand | Spot Price | Savings |
|----------|-----------|------------|---------|
| m5.large | $0.096/hr | ~$0.029/hr | 70% off |
| m5.xlarge | $0.192/hr | ~$0.058/hr | 70% off |
| m5.2xlarge | $0.384/hr | ~$0.115/hr | 70% off |
| m5.4xlarge | $0.768/hr | ~$0.230/hr | 70% off |

**20-hour project costs:**
- m5.large Spot: **$0.58** 🎉
- m5.xlarge Spot: **$1.16** 🎉
- m5.4xlarge Spot: **$4.60** 🎉

**Caveat**: Spot instances can be interrupted, but for batch processing this is fine!

---

### **Option 5: Process Locally (FREE!)** 💰 $0

If your computer has 32+ GB RAM:

```bash
# Just run locally - no EC2 needed!
python scripts/03_data_cleaning.py --input spotify_data.csv
python scripts/04_feature_engineering.py
python ml_models/popularity_model.py
```

**Pros:**
- ✅ Completely free
- ✅ No AWS setup needed
- ✅ Keep data local

**Cons:**
- ❌ Might take 6-12 hours
- ❌ Ties up your computer
- ❌ Need good RAM (32GB+)

---

### **Option 6: Subsample the Data** 💰 $0-2

Work with 10% of data (6 million rows ≈ 1.3 GB):

```bash
# Create 10% sample
head -n 6200001 spotify_data.csv > spotify_sample.csv

# Use any small instance (even free tier!)
# t2.medium (4 GB RAM) - $0.0464/hr = $0.93 for project
```

**Pros:**
- ✅ Very cheap
- ✅ Faster iteration
- ✅ Great for learning/testing

**Cons:**
- ❌ Not full dataset insights
- ❌ ML models less accurate

---

## 🎯 My Recommendation

### For Budget-Conscious (Best Value):

**Use m5.large with Spot pricing**
```
Cost: $0.029/hr × 20 hours = $0.58
Total with S3: ~$2
```

**Why?**
- ✅ Super cheap (<$1!)
- ✅ Scripts handle chunking automatically
- ✅ You'll learn big data techniques (chunking, memory management)
- ✅ Still processes full 62M rows

---

### For Learning Experience:

**Use m5.xlarge with Spot pricing**
```
Cost: $0.058/hr × 15 hours = $0.87
Total with S3: ~$2.50
```

**Why?**
- ✅ Still very cheap
- ✅ Faster than m5.large
- ✅ More comfortable RAM headroom
- ✅ Better for ML training

---

## 📋 How to Enable Chunked Processing

Good news: **Already built into the scripts!** Just use the flags:

```bash
# Auto-detect and use chunking if needed
python scripts/03_data_cleaning.py

# Or force chunking with explicit size
python scripts/03_data_cleaning.py --chunk-size 100000
```

The scripts will:
1. Try to load full data
2. If memory error → automatically switch to chunking
3. Process 100K rows at a time
4. Save results incrementally

---

## 🚀 Launching Spot Instances

### Via AWS Console:
1. Go to EC2 → Launch Instance
2. Choose instance type (m5.large or m5.xlarge)
3. **Request Type**: Select "Spot Instances" ✅
4. Set max price: $0.10/hr (way above spot price for safety)
5. Launch!

### Via AWS CLI:
```bash
aws ec2 request-spot-instances \
  --spot-price "0.10" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification \
    "ImageId=ami-0c55b159cbfafe1f0,\
     InstanceType=m5.large,\
     KeyName=your-key,\
     SecurityGroupIds=sg-xxxxx"
```

---

## 💡 Cost Comparison Summary

| Approach | Instance | Type | Hours | Total Cost |
|----------|----------|------|-------|------------|
| **Best Value** | m5.large | Spot | 20 | **~$2** ✅ |
| Good Balance | m5.xlarge | Spot | 15 | **~$2.50** ✅ |
| Fast | m5.2xlarge | Spot | 10 | **~$3.50** |
| Fastest | m5.4xlarge | Spot | 10 | **~$5** |
| Learning Only | t2.medium + 10% sample | On-Demand | 5 | **~$1** |
| DIY | Local machine | - | - | **$0** ✅ |

*(All costs include S3 storage/transfer)*

---

## ⚠️ Memory Management Tips

### If You Get Out of Memory Errors:

```python
# Option 1: Use smaller chunks
python scripts/03_data_cleaning.py --chunk-size 50000

# Option 2: Process subset first
python scripts/03_data_cleaning.py --sample 1000000  # 1M rows

# Option 3: Use data types optimization
# (already in scripts - converts to optimal dtypes)
```

### Monitor Memory Usage:

```bash
# On EC2, check memory
free -h

# While script runs
watch -n 1 free -h
```

---

## 🎓 What You'll Learn

### Using Small Instance (m5.large):
- ✅ Chunked data processing
- ✅ Memory-efficient programming
- ✅ Real big data constraints
- ✅ Optimization techniques

### Using Large Instance (m5.4xlarge):
- ✅ In-memory processing
- ✅ Faster iteration
- ✅ Full dataset in RAM

Both are valuable learning experiences!

---

## 📊 Processing Time Estimates

| Instance | RAM | Load Time | Clean | Features | ML | Total |
|----------|-----|-----------|-------|----------|-----|-------|
| m5.large | 8 GB | 2 hrs | 1.5 hrs | 1 hr | 2 hrs | ~6.5 hrs |
| m5.xlarge | 16 GB | 1 hr | 1 hr | 45 min | 1.5 hrs | ~4 hrs |
| m5.2xlarge | 32 GB | 30 min | 30 min | 30 min | 1 hr | ~2.5 hrs |
| m5.4xlarge | 64 GB | 15 min | 20 min | 20 min | 45 min | ~1.5 hrs |

---

## ✅ Final Recommendation

**For someone watching budget:**

```
Instance: m5.xlarge (Spot)
Cost: ~$2.50 total
Time: ~4 hours processing
```

This gives you:
- ✅ Reasonable speed
- ✅ Very low cost
- ✅ Processes full dataset
- ✅ Good learning experience

**Launch command:**
```bash
# Request spot instance
aws ec2 request-spot-instances \
  --spot-price "0.10" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

---

**Bottom line: You can do this entire project for under $3! 🎉**

