"""
Master Pipeline - Run Everything End-to-End
Orchestrates the entire Spotify data analysis pipeline on EC2
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


class PipelineRunner:
    """Orchestrates the entire data pipeline"""

    def __init__(self, data_sample_size=None):
        self.base_dir = Path(__file__).parent
        self.start_time = datetime.now()
        self.data_sample_size = data_sample_size
        self.steps_completed = []
        self.steps_failed = []

    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "STEP": "📍",
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")

    def run_command(self, cmd, step_name, cwd=None, timeout=None):
        """Run a shell command and handle errors"""
        self.log(f"Running: {step_name}", "STEP")
        self.log(f"Command: {cmd}")

        step_start = datetime.now()

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.base_dir,
                timeout=timeout,
            )

            # Show output in real-time if available
            if result.stdout:
                for line in result.stdout.split("\n"):
                    if line.strip():
                        print(f"  {line}")

            elapsed = (datetime.now() - step_start).total_seconds()
            self.log(f"{step_name} completed in {elapsed:.1f}s", "SUCCESS")
            self.steps_completed.append((step_name, elapsed))
            return True

        except subprocess.TimeoutExpired as e:
            self.log(f"{step_name} TIMEOUT after {timeout}s!", "ERROR")
            self.log(f"Output: {e.stdout if e.stdout else 'None'}", "ERROR")
            self.steps_failed.append((step_name, "Timeout"))
            return False

        except subprocess.CalledProcessError as e:
            elapsed = (datetime.now() - step_start).total_seconds()
            self.log(f"{step_name} FAILED after {elapsed:.1f}s!", "ERROR")

            # Show both stdout and stderr
            if e.stdout:
                self.log(f"Output: {e.stdout}", "ERROR")
            if e.stderr:
                self.log(f"Error: {e.stderr}", "ERROR")

            self.steps_failed.append((step_name, e.returncode))
            return False

        except KeyboardInterrupt:
            self.log(f"{step_name} interrupted by user!", "ERROR")
            self.steps_failed.append((step_name, "Interrupted"))
            raise

        except Exception as e:
            self.log(f"{step_name} UNEXPECTED ERROR: {e}", "ERROR")
            self.steps_failed.append((step_name, str(e)))
            return False

    def check_prerequisites(self):
        """Check if required files and directories exist"""
        self.log("Checking prerequisites...", "STEP")

        checks_passed = True

        # Check Python version
        if sys.version_info < (3, 7):
            self.log(
                f"Python 3.7+ required (have {sys.version_info.major}.{sys.version_info.minor})",
                "ERROR",
            )
            checks_passed = False
        else:
            self.log(
                f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✅"
            )

        # Check required directories
        required_dirs = [
            "data",
            "scripts",
            "ml_models",
            "visualizations",
            "logs",
            "utils",
        ]
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.log(f"Created directory: {dir_name}")
                except Exception as e:
                    self.log(f"Failed to create directory {dir_name}: {e}", "ERROR")
                    checks_passed = False

        # Check required Python files
        required_scripts = [
            "scripts/download_from_s3.py",
            "scripts/03_data_cleaning.py",
            "scripts/04_feature_engineering.py",
            "scripts/05_save_results.py",
            "ml_models/popularity_model.py",
            "ml_models/recommendation_model.py",
            "visualizations/insights_analysis.py",
            "utils/logger.py",
        ]

        missing_scripts = []
        for script in required_scripts:
            script_path = self.base_dir / script
            if not script_path.exists():
                missing_scripts.append(script)

        if missing_scripts:
            self.log("Missing required scripts:", "ERROR")
            for script in missing_scripts:
                self.log(f"  - {script}", "ERROR")
            checks_passed = False
        else:
            self.log(f"All {len(required_scripts)} required scripts found ✅")

        # Check disk space
        try:
            import shutil

            disk_usage = shutil.disk_usage(self.base_dir)
            available_gb = disk_usage.free / 1024**3

            if available_gb < 10:
                self.log(f"Low disk space: {available_gb:.1f} GB available", "ERROR")
                checks_passed = False
            elif available_gb < 50:
                self.log(f"⚠️  Limited disk space: {available_gb:.1f} GB available")
            else:
                self.log(f"Disk space: {available_gb:.1f} GB available ✅")
        except Exception as e:
            self.log(f"Could not check disk space: {e}", "ERROR")

        # Check for config file
        config_file = self.base_dir / "config.yaml"
        if not config_file.exists():
            self.log("Warning: config.yaml not found", "ERROR")
            checks_passed = False
        else:
            self.log("config.yaml found ✅")

        # Check AWS credentials (optional but recommended)
        try:
            result = subprocess.run(
                "aws sts get-caller-identity",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.log("AWS credentials configured ✅")
            else:
                self.log(
                    "AWS credentials not configured (some steps may fail)", "ERROR"
                )
        except Exception:
            self.log("Could not verify AWS credentials", "ERROR")

        # Check for virtual environment
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            self.log("Virtual environment active ✅")
        else:
            self.log("⚠️  Not running in virtual environment")

        # Check required Python packages
        required_packages = [
            "pandas",
            "numpy",
            "boto3",
            "yaml",
            "sklearn",
            "xgboost",
            "matplotlib",
            "seaborn",
        ]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.log("Missing required packages:", "ERROR")
            for pkg in missing_packages:
                self.log(f"  - {pkg}", "ERROR")
            self.log("Install with: pip install -r requirements.txt", "INFO")
            checks_passed = False
        else:
            self.log(f"All {len(required_packages)} required packages installed ✅")

        if checks_passed:
            self.log("All prerequisites check passed!", "SUCCESS")
        else:
            self.log(
                "Prerequisites check FAILED - fix issues before continuing", "ERROR"
            )

        return checks_passed

    def step_1_download_data(self):
        """Step 1: Download data from S3"""
        return self.run_command(
            "python scripts/download_from_s3.py", "Step 1: Download Data from S3"
        )

    def step_2_data_cleaning(self):
        """Step 2: Clean and preprocess data"""
        cmd = "python scripts/03_data_cleaning.py --chunk-size 500000"
        if self.data_sample_size:
            cmd += f" --sample {self.data_sample_size}"

        return self.run_command(cmd, "Step 2: Data Cleaning & Preprocessing")

    def step_3_feature_engineering(self):
        """Step 3: Feature engineering"""
        return self.run_command(
            "python scripts/04_feature_engineering.py", "Step 3: Feature Engineering"
        )

    def step_4_exploratory_analysis(self):
        """Step 4: Create visualizations"""
        sample_arg = f"--sample {min(500000, self.data_sample_size or 500000)}"
        return self.run_command(
            f"python visualizations/insights_analysis.py {sample_arg}",
            "Step 4: Exploratory Analysis & Visualizations",
        )

    def step_5_ml_popularity_model(self):
        """Step 5: Train popularity prediction model"""
        sample_arg = f"--sample {min(10000000, self.data_sample_size or 10000000)}"
        return self.run_command(
            f"python ml_models/popularity_model.py {sample_arg}",
            "Step 5: ML - Popularity Prediction Model",
        )

    def step_6_ml_recommendation_model(self):
        """Step 6: Train recommendation model"""
        sample_arg = f"--sample {min(5000000, self.data_sample_size or 5000000)}"
        return self.run_command(
            f"python ml_models/recommendation_model.py {sample_arg}",
            "Step 6: ML - Recommendation System",
        )

    def step_7_save_results(self):
        """Step 7: Upload results back to S3"""
        return self.run_command(
            "python scripts/05_save_results.py", "Step 7: Upload Results to S3"
        )

    def generate_summary_report(self):
        """Generate final summary report"""
        self.log("=" * 80)
        self.log("PIPELINE EXECUTION SUMMARY", "INFO")
        self.log("=" * 80)

        elapsed = datetime.now() - self.start_time
        total_seconds = elapsed.total_seconds()
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        self.log(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Total execution time: {hours}h {minutes}m {seconds}s")
        self.log(f"Steps completed: {len(self.steps_completed)}")
        self.log(f"Steps failed: {len(self.steps_failed)}")

        if self.steps_completed:
            self.log("\n✅ Completed Steps:")
            for i, step_info in enumerate(self.steps_completed, 1):
                if isinstance(step_info, tuple):
                    step_name, step_time = step_info
                    self.log(f"  {i}. {step_name} ({step_time:.1f}s)")
                else:
                    self.log(f"  {i}. {step_info}")

        if self.steps_failed:
            self.log("\n❌ Failed Steps:")
            for i, step_info in enumerate(self.steps_failed, 1):
                if isinstance(step_info, tuple):
                    step_name, error_info = step_info
                    self.log(f"  {i}. {step_name} - {error_info}")
                else:
                    self.log(f"  {i}. {step_info}")

        self.log("=" * 80)

        # Write to file with error handling
        try:
            report_path = self.base_dir / "logs" / "pipeline_summary.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                f.write(f"Pipeline Execution Summary\n")
                f.write(f"=" * 80 + "\n\n")
                f.write(
                    f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {hours}h {minutes}m {seconds}s\n")
                f.write(f"\nSteps completed: {len(self.steps_completed)}\n")
                f.write(f"Steps failed: {len(self.steps_failed)}\n")

                if self.steps_completed:
                    f.write(f"\n✅ Completed Steps:\n")
                    for step_info in self.steps_completed:
                        if isinstance(step_info, tuple):
                            step_name, step_time = step_info
                            f.write(f"  - {step_name} ({step_time:.1f}s)\n")
                        else:
                            f.write(f"  - {step_info}\n")

                if self.steps_failed:
                    f.write(f"\n❌ Failed Steps:\n")
                    for step_info in self.steps_failed:
                        if isinstance(step_info, tuple):
                            step_name, error_info = step_info
                            f.write(f"  - {step_name}: {error_info}\n")
                        else:
                            f.write(f"  - {step_info}\n")

                f.write(f"\n" + "=" * 80 + "\n")

                # Add recommendations
                if self.steps_failed:
                    f.write("\n💡 TROUBLESHOOTING RECOMMENDATIONS:\n\n")
                    for step_info in self.steps_failed:
                        step_name = (
                            step_info[0] if isinstance(step_info, tuple) else step_info
                        )

                        if "Download" in step_name:
                            f.write(
                                "- Download failed: Check AWS credentials and S3 bucket access\n"
                            )
                        elif "Cleaning" in step_name:
                            f.write(
                                "- Cleaning failed: Check input data format and memory availability\n"
                            )
                        elif "Feature" in step_name:
                            f.write(
                                "- Feature engineering failed: Verify cleaned data exists\n"
                            )
                        elif "ML" in step_name or "Model" in step_name:
                            f.write(
                                f"- {step_name} failed: Check memory, reduce sample size\n"
                            )
                        elif "Visualization" in step_name:
                            f.write(
                                "- Visualization failed: Check matplotlib backend and data availability\n"
                            )

            self.log(f"Summary saved to: {report_path}")

        except Exception as e:
            self.log(f"Failed to write summary report: {e}", "ERROR")

    def run_full_pipeline(self, skip_download=False, skip_ml=False):
        """Run the complete pipeline with comprehensive error handling"""
        try:
            self.log("=" * 80)
            self.log("🎵 SPOTIFY BIG DATA ANALYSIS - MASTER PIPELINE")
            self.log("=" * 80)
            self.log(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if self.data_sample_size:
                self.log(f"Data sample size: {self.data_sample_size:,}")
            self.log("")

            # Check prerequisites
            self.log("\n🔍 Checking prerequisites...")
            prereq_passed = self.check_prerequisites()

            if not prereq_passed:
                if skip_download:
                    self.log(
                        "⚠️  Prerequisites check failed, but continuing as requested...",
                        "ERROR",
                    )
                else:
                    self.log(
                        "❌ Prerequisites check failed! Fix issues before continuing.",
                        "ERROR",
                    )
                    self.generate_summary_report()
                    return False

            # Step 1: Download data (optional)
            if not skip_download:
                self.log("\n📥 Step 1: Downloading data from S3...")
                if not self.step_1_download_data():
                    self.log(
                        "⚠️  Download failed, checking if data exists locally...",
                        "ERROR",
                    )
                    data_file = self.base_dir / "data" / "raw" / "spotify_data.csv"
                    if not data_file.exists():
                        self.log("❌ No local data found. Cannot continue.", "ERROR")
                        self.generate_summary_report()
                        return False
                    else:
                        self.log("✅ Found local data, continuing...")
            else:
                self.log("\n⏭️  Skipping download step (as requested)")

            # Step 2: Data cleaning (CRITICAL)
            self.log("\n🧹 Step 2: Data cleaning...")
            if not self.step_2_data_cleaning():
                self.log("❌ Data cleaning FAILED! Cannot continue.", "ERROR")
                self.generate_summary_report()
                return False

            # Step 3: Feature engineering (CRITICAL)
            self.log("\n⚙️  Step 3: Feature engineering...")
            if not self.step_3_feature_engineering():
                self.log("❌ Feature engineering FAILED! Cannot continue.", "ERROR")
                self.generate_summary_report()
                return False

            # Step 4: Visualizations (NON-CRITICAL)
            self.log("\n📊 Step 4: Creating visualizations...")
            if not self.step_4_exploratory_analysis():
                self.log(
                    "⚠️  Visualization generation failed, but continuing...", "ERROR"
                )

            # Step 5 & 6: ML models (NON-CRITICAL, optional)
            if skip_ml:
                self.log("\n⏭️  Skipping ML training (as requested)")
            else:
                self.log("\n🤖 Step 5: Training popularity model...")
                if not self.step_5_ml_popularity_model():
                    self.log(
                        "⚠️  Popularity model training failed, but continuing...",
                        "ERROR",
                    )

                self.log("\n🤖 Step 6: Training recommendation model...")
                if not self.step_6_ml_recommendation_model():
                    self.log(
                        "⚠️  Recommendation model training failed, but continuing...",
                        "ERROR",
                    )

            # Step 7: Save results (NON-CRITICAL)
            self.log("\n☁️  Step 7: Uploading results to S3...")
            if not self.step_7_save_results():
                self.log("⚠️  Failed to upload results. Check manually.", "ERROR")

            # Generate summary
            self.generate_summary_report()

            # Final status
            critical_failures = [
                f
                for f in self.steps_failed
                if "Cleaning" in str(f) or "Feature" in str(f)
            ]

            if len(self.steps_failed) == 0:
                self.log("\n🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉", "SUCCESS")
                return True
            elif len(critical_failures) > 0:
                self.log(f"\n❌ Pipeline FAILED with critical errors", "ERROR")
                return False
            else:
                self.log(
                    f"\n⚠️  Pipeline completed with {len(self.steps_failed)} non-critical failures",
                    "ERROR",
                )
                return True  # Still return True if only non-critical steps failed

        except KeyboardInterrupt:
            self.log("\n\n⚠️  Pipeline interrupted by user!", "ERROR")
            self.generate_summary_report()
            return False

        except Exception as e:
            self.log(f"\n❌ FATAL ERROR: {e}", "ERROR")
            import traceback

            traceback.print_exc()
            self.generate_summary_report()
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the complete Spotify analysis pipeline"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for data processing (default: use all data)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading data from S3 (assume already present)",
    )
    parser.add_argument(
        "--skip-ml", action="store_true", help="Skip ML model training (only do EDA)"
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=["download", "clean", "features", "viz", "ml-pop", "ml-rec", "upload"],
        help="Run only a specific step",
    )

    args = parser.parse_args()

    pipeline = PipelineRunner(data_sample_size=args.sample)

    # Run specific step or full pipeline
    if args.step:
        step_map = {
            "download": pipeline.step_1_download_data,
            "clean": pipeline.step_2_data_cleaning,
            "features": pipeline.step_3_feature_engineering,
            "viz": pipeline.step_4_exploratory_analysis,
            "ml-pop": pipeline.step_5_ml_popularity_model,
            "ml-rec": pipeline.step_6_ml_recommendation_model,
            "upload": pipeline.step_7_save_results,
        }
        success = step_map[args.step]()
        sys.exit(0 if success else 1)
    else:
        success = pipeline.run_full_pipeline(
            skip_download=args.skip_download, skip_ml=args.skip_ml
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
