"""
Setup Validation Script
Run this before starting the pipeline to catch issues early
"""

import sys
import subprocess
from pathlib import Path
import importlib.util

class SetupValidator:
    """Validates the entire setup before running pipeline"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.base_dir = Path(__file__).parent
        
    def check(self, condition, success_msg, error_msg, is_critical=True):
        """Check a condition and record result"""
        if condition:
            print(f"✅ {success_msg}")
            return True
        else:
            print(f"{'❌' if is_critical else '⚠️ '} {error_msg}")
            if is_critical:
                self.errors.append(error_msg)
            else:
                self.warnings.append(error_msg)
            return False
    
    def validate_python(self):
        """Validate Python version"""
        print("\n🐍 Python Environment")
        print("-" * 60)
        
        version = sys.version_info
        self.check(
            version >= (3, 7),
            f"Python {version.major}.{version.minor}.{version.micro}",
            f"Python 3.7+ required (have {version.major}.{version.minor})",
            is_critical=True
        )
        
        # Check if in virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        self.check(
            in_venv,
            "Running in virtual environment",
            "Not in virtual environment (recommended)",
            is_critical=False
        )
    
    def validate_packages(self):
        """Validate required Python packages"""
        print("\n📦 Python Packages")
        print("-" * 60)
        
        required = {
            'pandas': 'Data processing',
            'numpy': 'Numerical computations',
            'boto3': 'AWS SDK',
            'yaml': 'Config parsing',
            'sklearn': 'Machine learning (scikit-learn)',
            'xgboost': 'ML models',
            'matplotlib': 'Visualizations',
            'seaborn': 'Statistical plots'
        }
        
        all_installed = True
        for package, description in required.items():
            # Handle special case for sklearn
            import_name = 'sklearn' if package == 'sklearn' else package
            
            try:
                __import__(import_name)
                print(f"✅ {package:15s} - {description}")
            except ImportError:
                print(f"❌ {package:15s} - {description} [MISSING]")
                self.errors.append(f"Package '{package}' not installed")
                all_installed = False
        
        if not all_installed:
            print("\n💡 Install missing packages:")
            print("   pip install -r requirements.txt")
    
    def validate_structure(self):
        """Validate directory structure"""
        print("\n📁 Directory Structure")
        print("-" * 60)
        
        required_dirs = [
            ('scripts', 'Pipeline scripts'),
            ('ml_models', 'ML model scripts'),
            ('visualizations', 'Visualization scripts'),
            ('utils', 'Utility modules'),
            ('data', 'Data directory'),
            ('logs', 'Log files')
        ]
        
        for dir_name, description in required_dirs:
            dir_path = self.base_dir / dir_name
            self.check(
                dir_path.exists() and dir_path.is_dir(),
                f"{dir_name:20s} - {description}",
                f"Directory '{dir_name}' missing",
                is_critical=True
            )
    
    def validate_scripts(self):
        """Validate required scripts exist"""
        print("\n📄 Required Scripts")
        print("-" * 60)
        
        required_scripts = [
            ('generate_data_local.py', 'Data generation'),
            ('master_pipeline.py', 'Pipeline orchestrator'),
            ('scripts/download_from_s3.py', 'S3 download'),
            ('scripts/03_data_cleaning.py', 'Data cleaning'),
            ('scripts/04_feature_engineering.py', 'Feature engineering'),
            ('scripts/05_save_results.py', 'Save to S3'),
            ('ml_models/popularity_model.py', 'Popularity model'),
            ('ml_models/recommendation_model.py', 'Recommendation model'),
            ('visualizations/insights_analysis.py', 'Insights visualization'),
            ('utils/logger.py', 'Logging utility')
        ]
        
        for script, description in required_scripts:
            script_path = self.base_dir / script
            exists = script_path.exists() and script_path.is_file()
            
            if exists:
                # Check if Python file is parseable
                try:
                    with open(script_path, 'r') as f:
                        compile(f.read(), script_path, 'exec')
                    print(f"✅ {script:40s} - {description}")
                except SyntaxError as e:
                    print(f"❌ {script:40s} - Syntax error: {e}")
                    self.errors.append(f"Script '{script}' has syntax error")
            else:
                print(f"❌ {script:40s} - {description} [MISSING]")
                self.errors.append(f"Script '{script}' not found")
    
    def validate_config(self):
        """Validate configuration files"""
        print("\n⚙️  Configuration Files")
        print("-" * 60)
        
        # Check config.yaml
        config_path = self.base_dir / 'config.yaml'
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_keys = ['s3_bucket_name', 'aws_region']
                all_present = all(key in config for key in required_keys)
                
                self.check(
                    all_present,
                    "config.yaml valid",
                    f"config.yaml missing keys: {required_keys}",
                    is_critical=True
                )
                
                if all_present:
                    print(f"   Bucket: {config.get('s3_bucket_name')}")
                    print(f"   Region: {config.get('aws_region')}")
                    
            except Exception as e:
                print(f"❌ config.yaml invalid: {e}")
                self.errors.append(f"config.yaml error: {e}")
        else:
            print(f"❌ config.yaml not found")
            self.errors.append("config.yaml not found")
        
        # Check requirements.txt
        req_path = self.base_dir / 'requirements.txt'
        self.check(
            req_path.exists(),
            "requirements.txt found",
            "requirements.txt missing",
            is_critical=False
        )
    
    def validate_aws(self):
        """Validate AWS configuration"""
        print("\n☁️  AWS Configuration")
        print("-" * 60)
        
        # Check AWS CLI
        try:
            result = subprocess.run(
                'aws --version',
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            aws_installed = result.returncode == 0
            self.check(
                aws_installed,
                f"AWS CLI installed: {result.stdout.strip()}",
                "AWS CLI not installed",
                is_critical=False
            )
        except Exception:
            print(f"⚠️  Could not check AWS CLI")
            self.warnings.append("Could not verify AWS CLI")
        
        # Check AWS credentials
        try:
            result = subprocess.run(
                'aws sts get-caller-identity',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✅ AWS credentials configured")
                # Parse identity
                import json
                identity = json.loads(result.stdout)
                print(f"   Account: {identity.get('Account')}")
                print(f"   ARN: {identity.get('Arn')}")
            else:
                print(f"⚠️  AWS credentials not configured")
                self.warnings.append("AWS credentials not configured")
                print(f"   Run: aws configure")
        except Exception as e:
            print(f"⚠️  Could not verify AWS credentials: {e}")
            self.warnings.append("Could not verify AWS credentials")
    
    def validate_resources(self):
        """Validate system resources"""
        print("\n💻 System Resources")
        print("-" * 60)
        
        # Check disk space
        try:
            import shutil
            disk = shutil.disk_usage(self.base_dir)
            available_gb = disk.free / 1024**3
            total_gb = disk.total / 1024**3
            used_pct = (disk.used / disk.total) * 100
            
            print(f"   Total: {total_gb:.1f} GB")
            print(f"   Used: {used_pct:.1f}%")
            print(f"   Available: {available_gb:.1f} GB")
            
            self.check(
                available_gb >= 10,
                f"Sufficient disk space ({available_gb:.1f} GB)",
                f"Low disk space ({available_gb:.1f} GB)",
                is_critical=available_gb < 5
            )
        except Exception as e:
            print(f"⚠️  Could not check disk space: {e}")
        
        # Check memory (Linux only)
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = None
            mem_available = None
            
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    mem_total = int(line.split()[1]) / 1024 / 1024  # GB
                if 'MemAvailable:' in line:
                    mem_available = int(line.split()[1]) / 1024 / 1024  # GB
            
            if mem_total and mem_available:
                print(f"   Total RAM: {mem_total:.1f} GB")
                print(f"   Available RAM: {mem_available:.1f} GB")
                
                self.check(
                    mem_available >= 4,
                    f"Sufficient memory ({mem_available:.1f} GB available)",
                    f"Low memory ({mem_available:.1f} GB available)",
                    is_critical=mem_available < 2
                )
        except Exception:
            print(f"   (Memory check not available on this system)")
    
    def validate_data(self):
        """Check for existing data"""
        print("\n📊 Data Files")
        print("-" * 60)
        
        # Check for raw data
        raw_data = self.base_dir / 'data' / 'raw' / 'spotify_data.csv'
        if raw_data.exists():
            size_mb = raw_data.stat().st_size / 1024 / 1024
            print(f"✅ Raw data found: {size_mb:.1f} MB")
        else:
            print(f"⚠️  No raw data found (will need to generate or download)")
        
        # Check for processed data
        processed_data = self.base_dir / 'data' / 'processed' / 'spotify_cleaned.csv'
        if processed_data.exists():
            size_mb = processed_data.stat().st_size / 1024 / 1024
            print(f"✅ Processed data found: {size_mb:.1f} MB")
        else:
            print(f"   No processed data (will be generated)")
    
    def generate_report(self):
        """Generate final report"""
        print("\n" + "="*60)
        print("📋 VALIDATION SUMMARY")
        print("="*60)
        
        total_checks = len(self.errors) + len(self.warnings)
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            print("\n✅ All checks passed! Ready to run pipeline.")
            print("\nNext steps:")
            print("  1. Generate data:")
            print("     python3 generate_data_local.py --rows 5000000")
            print("\n  2. Upload to S3:")
            print("     aws s3 cp data/spotify_data.csv s3://YOUR_BUCKET/data/")
            print("\n  3. Run pipeline on EC2:")
            print("     python master_pipeline.py --sample 5000000")
            return True
            
        if len(self.errors) > 0:
            print(f"\n❌ {len(self.errors)} CRITICAL ISSUE(S) FOUND:")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if len(self.warnings) > 0:
            print(f"\n⚠️  {len(self.warnings)} WARNING(S):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if len(self.errors) > 0:
            print("\n❌ Fix critical issues before continuing!")
            return False
        else:
            print("\n⚠️  You can proceed, but address warnings for best results.")
            return True
    
    def run_all_validations(self):
        """Run all validation checks"""
        print("="*60)
        print("🔍 SETUP VALIDATION")
        print("="*60)
        
        self.validate_python()
        self.validate_packages()
        self.validate_structure()
        self.validate_scripts()
        self.validate_config()
        self.validate_aws()
        self.validate_resources()
        self.validate_data()
        
        return self.generate_report()

def main():
    validator = SetupValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

