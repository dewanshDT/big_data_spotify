"""
Create S3 Bucket for Spotify Data Analysis
Simple script to create and configure S3 bucket
"""

import boto3
import yaml
import sys
from pathlib import Path

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_bucket(bucket_name, region):
    """Create S3 bucket"""
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        if region == 'us-east-1':
            # us-east-1 doesn't need LocationConstraint
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"✅ Bucket '{bucket_name}' created successfully!")
        return True
    
    except s3_client.exceptions.BucketAlreadyExists:
        print(f"❌ Bucket '{bucket_name}' already exists (owned by someone else)")
        print("   Try a different bucket name")
        return False
    
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"ℹ️  Bucket '{bucket_name}' already exists and is owned by you")
        return True
    
    except Exception as e:
        print(f"❌ Error creating bucket: {e}")
        return False

def setup_bucket_structure(bucket_name, region):
    """Create folder structure in S3 bucket"""
    s3_client = boto3.client('s3', region_name=region)
    
    folders = ['raw/', 'processed/', 'results/', 'models/', 'visualizations/']
    
    for folder in folders:
        try:
            s3_client.put_object(Bucket=bucket_name, Key=folder)
            print(f"  Created folder: {folder}")
        except Exception as e:
            print(f"  Warning: Could not create folder {folder}: {e}")

def main():
    print("=" * 60)
    print("Creating S3 Bucket for Spotify Data Analysis")
    print("=" * 60)
    
    # Load config
    config = load_config()
    bucket_name = config['aws']['s3_bucket']
    region = config['aws']['region']
    
    print(f"\nBucket name: {bucket_name}")
    print(f"Region: {region}")
    
    # Confirm
    response = input("\nProceed with bucket creation? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Create bucket
    print("\n🔨 Creating bucket...")
    if create_bucket(bucket_name, region):
        print("\n📁 Setting up folder structure...")
        setup_bucket_structure(bucket_name, region)
        
        print("\n" + "=" * 60)
        print("✅ Setup complete!")
        print("=" * 60)
        print(f"\nYour S3 bucket is ready: s3://{bucket_name}")
        print("\nNext step: Upload your data")
        print(f"  python scripts/01_upload_to_s3.py --file /path/to/spotify_data.csv")
    else:
        print("\n❌ Bucket creation failed. Please check the error and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()



