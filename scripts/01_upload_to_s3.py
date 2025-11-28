"""
Upload Spotify Dataset to S3
Handles large file uploads with progress tracking
"""

import boto3
import yaml
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import os

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ProgressPercentage:
    """Progress callback for upload"""
    
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._pbar = tqdm(total=self._size, unit='B', unit_scale=True, 
                         desc=f"Uploading {Path(filename).name}")
    
    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)
    
    def __del__(self):
        if hasattr(self, '_pbar'):
            self._pbar.close()

def upload_to_s3(file_path, bucket_name, s3_key, region):
    """Upload file to S3 with progress bar"""
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    file_size_gb = file_size / (1024 ** 3)
    
    print(f"\n📦 File: {file_path}")
    print(f"📊 Size: {file_size_gb:.2f} GB")
    print(f"🎯 Destination: s3://{bucket_name}/{s3_key}")
    
    # Confirm for large files
    if file_size_gb > 1:
        response = input(f"\nThis is a large file ({file_size_gb:.2f} GB). Continue? (y/n): ")
        if response.lower() != 'y':
            print("Upload cancelled.")
            return False
    
    try:
        s3_client = boto3.client('s3', region_name=region)
        
        print("\n🚀 Starting upload...")
        
        # Use multipart upload for large files
        s3_client.upload_file(
            file_path,
            bucket_name,
            s3_key,
            Callback=ProgressPercentage(file_path)
        )
        
        print(f"\n✅ Upload complete!")
        print(f"   File available at: s3://{bucket_name}/{s3_key}")
        return True
        
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {file_path}")
        return False
    
    except boto3.exceptions.S3UploadFailedError as e:
        print(f"\n❌ Upload failed: {e}")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload Spotify data to S3')
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--key', help='S3 key (path in bucket)', default=None)
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    bucket_name = config['aws']['s3_bucket']
    region = config['aws']['region']
    
    # Determine S3 key
    if args.key:
        s3_key = args.key
    else:
        s3_key = config['s3_paths']['raw_data']
    
    print("=" * 60)
    print("Spotify Data Upload to S3")
    print("=" * 60)
    
    # Upload
    success = upload_to_s3(args.file, bucket_name, s3_key, region)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Upload Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Launch EC2 instance (see README.md)")
        print("2. SSH into EC2 and run setup script")
        print("3. Start data processing")
    else:
        print("\n❌ Upload failed. Please check the error and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()



