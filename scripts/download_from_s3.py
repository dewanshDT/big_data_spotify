"""
Download Spotify Data from S3 to EC2
"""

import boto3
import yaml
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        # If running on EC2, config might not exist yet
        return {
            'aws': {'region': 'us-east-1', 's3_bucket': 'spotify-data-analysis-bucket'},
            's3_paths': {'raw_data': 'raw/spotify_data.csv'}
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DownloadProgress:
    """Progress callback for download"""
    
    def __init__(self, filename, size):
        self._filename = filename
        self._size = size
        self._seen_so_far = 0
        self._pbar = tqdm(total=size, unit='B', unit_scale=True, 
                         desc=f"Downloading {Path(filename).name}")
    
    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)
    
    def __del__(self):
        if hasattr(self, '_pbar'):
            self._pbar.close()

def download_from_s3(bucket_name, s3_key, local_path, region):
    """Download file from S3 with progress bar"""
    
    # Create directory if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        s3_client = boto3.client('s3', region_name=region)
        
        # Get file size
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response['ContentLength']
        file_size_gb = file_size / (1024 ** 3)
        
        print(f"\n📦 Source: s3://{bucket_name}/{s3_key}")
        print(f"📊 Size: {file_size_gb:.2f} GB")
        print(f"💾 Destination: {local_path}")
        
        # Check disk space
        stat = os.statvfs(os.path.dirname(local_path) or '.')
        free_space = stat.f_bavail * stat.f_frsize
        free_space_gb = free_space / (1024 ** 3)
        
        print(f"💿 Free disk space: {free_space_gb:.2f} GB")
        
        if file_size > free_space * 0.9:  # Leave 10% buffer
            print("\n❌ Error: Not enough disk space!")
            return False
        
        print("\n🚀 Starting download...")
        
        # Download with progress
        s3_client.download_file(
            bucket_name,
            s3_key,
            local_path,
            Callback=DownloadProgress(local_path, file_size)
        )
        
        print(f"\n✅ Download complete!")
        print(f"   File saved to: {local_path}")
        return True
        
    except s3_client.exceptions.NoSuchKey:
        print(f"\n❌ Error: File not found in S3: {s3_key}")
        return False
    
    except s3_client.exceptions.NoSuchBucket:
        print(f"\n❌ Error: Bucket not found: {bucket_name}")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download Spotify data from S3')
    parser.add_argument('--bucket', help='S3 bucket name', default=None)
    parser.add_argument('--key', help='S3 key (path in bucket)', default=None)
    parser.add_argument('--output', help='Local output path', default='data/raw/spotify_data.csv')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    bucket_name = args.bucket or config['aws']['s3_bucket']
    s3_key = args.key or config['s3_paths']['raw_data']
    local_path = args.output
    region = config['aws']['region']
    
    print("=" * 60)
    print("Download Spotify Data from S3")
    print("=" * 60)
    
    # Download
    success = download_from_s3(bucket_name, s3_key, local_path, region)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Download Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print(f"1. Verify data: head {local_path}")
        print("2. Start data cleaning: python scripts/03_data_cleaning.py")
        print("3. Or explore in Jupyter: jupyter notebook")
    else:
        print("\n❌ Download failed. Please check the error and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())



