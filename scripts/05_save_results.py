"""
Save Results Back to S3
Upload processed data, models, and visualizations to S3
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
        return {
            'aws': {'region': 'us-east-1', 's3_bucket': 'spotify-data-analysis-bucket'},
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class UploadProgress:
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

def upload_directory(local_dir, bucket, s3_prefix, region):
    """Upload entire directory to S3"""
    s3_client = boto3.client('s3', region_name=region)
    local_dir = Path(local_dir)
    
    files_to_upload = list(local_dir.rglob('*'))
    files_to_upload = [f for f in files_to_upload if f.is_file()]
    
    print(f"\n📦 Found {len(files_to_upload)} files to upload from {local_dir}")
    
    uploaded = 0
    for file_path in files_to_upload:
        relative_path = file_path.relative_to(local_dir)
        s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
        
        try:
            s3_client.upload_file(
                str(file_path),
                bucket,
                s3_key,
                Callback=UploadProgress(str(file_path))
            )
            uploaded += 1
        except Exception as e:
            print(f"\n⚠️  Failed to upload {file_path}: {e}")
    
    return uploaded

def main():
    parser = argparse.ArgumentParser(description='Upload results to S3')
    parser.add_argument('--processed-data', action='store_true',
                       help='Upload processed data')
    parser.add_argument('--models', action='store_true',
                       help='Upload ML models')
    parser.add_argument('--visualizations', action='store_true',
                       help='Upload visualizations')
    parser.add_argument('--all', action='store_true',
                       help='Upload everything')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    bucket = config['aws']['s3_bucket']
    region = config['aws']['region']
    
    print("=" * 60)
    print("Upload Results to S3")
    print("=" * 60)
    print(f"\nBucket: {bucket}")
    print(f"Region: {region}")
    
    total_uploaded = 0
    
    # Upload processed data
    if args.processed_data or args.all:
        if Path('data/processed').exists():
            print("\n📊 Uploading processed data...")
            count = upload_directory('data/processed', bucket, 'processed', region)
            print(f"✅ Uploaded {count} processed data files")
            total_uploaded += count
    
    # Upload ML models
    if args.models or args.all:
        if Path('ml_models').exists():
            print("\n🤖 Uploading ML models...")
            # Upload saved models (pkl, h5, etc.)
            model_files = list(Path('ml_models').glob('*.pkl')) + \
                         list(Path('ml_models').glob('*.h5')) + \
                         list(Path('ml_models').glob('*.joblib'))
            
            if model_files:
                s3_client = boto3.client('s3', region_name=region)
                for model_file in model_files:
                    s3_key = f"models/{model_file.name}"
                    try:
                        s3_client.upload_file(
                            str(model_file),
                            bucket,
                            s3_key,
                            Callback=UploadProgress(str(model_file))
                        )
                        total_uploaded += 1
                    except Exception as e:
                        print(f"\n⚠️  Failed to upload {model_file}: {e}")
                print(f"✅ Uploaded {len(model_files)} model files")
    
    # Upload visualizations
    if args.visualizations or args.all:
        if Path('visualizations').exists():
            print("\n📈 Uploading visualizations...")
            count = upload_directory('visualizations', bucket, 'visualizations', region)
            print(f"✅ Uploaded {count} visualization files")
            total_uploaded += count
    
    # Upload results summary (if exists)
    if Path('results').exists() and (args.all or args.processed_data):
        print("\n📄 Uploading results...")
        count = upload_directory('results', bucket, 'results', region)
        print(f"✅ Uploaded {count} result files")
        total_uploaded += count
    
    print("\n" + "=" * 60)
    print(f"✅ Upload Complete! {total_uploaded} files uploaded")
    print("=" * 60)
    print(f"\nYour results are available at:")
    print(f"  s3://{bucket}/")
    print("\nTo download later:")
    print(f"  aws s3 sync s3://{bucket}/results ./results")
    print(f"  aws s3 sync s3://{bucket}/visualizations ./visualizations")
    
    return 0

if __name__ == "__main__":
    exit(main())



