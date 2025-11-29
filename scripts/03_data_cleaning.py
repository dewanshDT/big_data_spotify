"""
Data Cleaning and Preprocessing
Handles missing values, duplicates, and data type conversions
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime


# Setup logging
def setup_logging(log_file="logs/data_cleaning.log"):
    """Setup logging configuration"""
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(file_path, sample_size=None, chunk_size=None, logger=None):
    """Load CSV data with options for sampling and chunking"""
    if logger:
        logger.info(f"Loading data from: {file_path}")
    print(f"\n📂 Loading data from: {file_path}")

    if sample_size:
        msg = f"Loading sample of {sample_size:,} rows for testing..."
        if logger:
            logger.info(msg)
        print(f"   {msg}")
        df = pd.read_csv(
            file_path, nrows=sample_size, on_bad_lines="skip", engine="python"
        )
    elif chunk_size:
        msg = f"Loading in chunks of {chunk_size:,} rows..."
        if logger:
            logger.info(msg)
        print(f"   {msg}")
        # Return iterator for chunk processing with error handling
        return pd.read_csv(
            file_path, chunksize=chunk_size, on_bad_lines="skip", engine="python"
        )
    else:
        if logger:
            logger.info("Loading full dataset...")
        print("   Loading full dataset...")
        df = pd.read_csv(file_path, on_bad_lines="skip", engine="python")

    msg = f"Loaded {len(df):,} rows, {len(df.columns)} columns"
    if logger:
        logger.info(msg)
    print(f"✅ {msg}")
    return df


def explore_data(df):
    """Quick data exploration"""
    print("\n" + "=" * 60)
    print("Data Exploration")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing": missing, "Percentage": missing_pct})
    print(missing_df[missing_df["Missing"] > 0])

    print("\nBasic Statistics:")
    print(df.describe())

    print("\nMemory Usage:")
    print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def clean_data(df, logger=None):
    """Clean and preprocess data"""
    if logger:
        logger.info("=" * 60)
        logger.info("Starting Data Cleaning")
        logger.info("=" * 60)
    print("\n" + "=" * 60)
    print("Data Cleaning")
    print("=" * 60)

    original_rows = len(df)

    # 1. Remove duplicates
    if logger:
        logger.info("Step 1: Removing duplicates...")
    print("\n1. Removing duplicates...")
    df = df.drop_duplicates()
    removed = original_rows - len(df)
    if logger:
        logger.info(f"Removed {removed:,} duplicate rows")
    print(f"   Removed {removed:,} duplicate rows")

    # 2. Handle missing values for audio features
    print("\n2. Handling missing values...")
    audio_features = [
        "energy",
        "danceability",
        "valence",
        "tempo",
        "acousticness",
        "instrumentalness",
    ]

    for feature in audio_features:
        if feature in df.columns:
            missing_count = df[feature].isnull().sum()
            if missing_count > 0:
                # Fill with median for audio features
                df[feature].fillna(df[feature].median(), inplace=True)
                print(
                    f"   Filled {missing_count:,} missing values in '{feature}' with median"
                )

    # 3. Handle missing text fields
    print("\n3. Handling missing text fields...")
    text_fields = ["track_name", "artist_name", "album_name", "playlist_name"]
    for field in text_fields:
        if field in df.columns:
            missing_count = df[field].isnull().sum()
            if missing_count > 0:
                df[field].fillna("Unknown", inplace=True)
                print(
                    f"   Filled {missing_count:,} missing values in '{field}' with 'Unknown'"
                )

    # 4. Remove rows with critical missing data
    print("\n4. Removing rows with critical missing data...")
    critical_columns = ["track_name", "artist_name"]
    df_before = len(df)
    df = df.dropna(subset=[col for col in critical_columns if col in df.columns])
    print(f"   Removed {df_before - len(df):,} rows with missing critical data")

    # 5. Data type conversions
    print("\n5. Converting data types...")

    # Ensure audio features are numeric
    for feature in audio_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")

    # Convert popularity to int (column is 'popularity_score' in this dataset)
    if "popularity_score" in df.columns:
        df["popularity_score"] = (
            pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0).astype(int)
        )
    elif "popularity" in df.columns:
        df["popularity"] = (
            pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)
        )

    # Convert duration_ms to int
    if "duration_ms" in df.columns:
        df["duration_ms"] = (
            pd.to_numeric(df["duration_ms"], errors="coerce").fillna(0).astype(int)
        )

    # 6. Remove outliers (optional - commented out for now)
    # print("\n6. Removing outliers...")
    # ... outlier removal logic if needed

    # 7. Validate value ranges
    print("\n6. Validating value ranges...")

    # Audio features should be between 0 and 1
    features_0_1 = [
        "energy",
        "danceability",
        "valence",
        "acousticness",
        "instrumentalness",
    ]
    for feature in features_0_1:
        if feature in df.columns:
            invalid_count = ((df[feature] < 0) | (df[feature] > 1)).sum()
            if invalid_count > 0:
                print(f"   Warning: {invalid_count:,} invalid values in '{feature}'")
                # Clip to valid range
                df[feature] = df[feature].clip(0, 1)

    # Popularity should be 0-100
    if "popularity_score" in df.columns:
        df["popularity_score"] = df["popularity_score"].clip(0, 100)
    elif "popularity" in df.columns:
        df["popularity"] = df["popularity"].clip(0, 100)

    final_msg = f"Cleaning complete! Final rows: {len(df):,}"
    if logger:
        logger.info(final_msg)
    print(f"\n✅ {final_msg}")

    return df


def save_cleaned_data(df, output_path):
    """Save cleaned data"""
    print(f"\n💾 Saving cleaned data to: {output_path}")

    # Create directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df.to_csv(output_path, index=False)

    # Also save as parquet for faster loading
    parquet_path = output_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)

    print(f"✅ Saved as CSV: {output_path}")
    print(f"✅ Saved as Parquet: {parquet_path}")

    # Print file sizes
    csv_size = output_path.stat().st_size / 1024**2
    parquet_size = parquet_path.stat().st_size / 1024**2
    print(f"\n   CSV size: {csv_size:.2f} MB")
    print(f"   Parquet size: {parquet_size:.2f} MB")
    print(f"   Compression: {(1 - parquet_size/csv_size)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Clean Spotify data")
    parser.add_argument(
        "--input", default="data/raw/spotify_data.csv", help="Input CSV file"
    )
    parser.add_argument(
        "--output", default="data/processed/spotify_cleaned.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Use sample of N rows for testing"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Process in chunks of N rows (for memory efficiency)",
    )
    parser.add_argument(
        "--explore", action="store_true", help="Show data exploration before cleaning"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Spotify Data Cleaning Started")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info("=" * 60)

    print("=" * 60)
    print("Spotify Data Cleaning")
    print("=" * 60)

    # Load data
    if args.chunk_size:
        logger.info(f"Using chunked processing (chunk size: {args.chunk_size:,} rows)")
        print(f"\n⚙️  Using chunked processing (chunk size: {args.chunk_size:,} rows)")
        print("   This will take longer but use less memory")

    df = load_data(
        args.input, sample_size=args.sample, chunk_size=args.chunk_size, logger=logger
    )

    # Handle chunked processing
    if args.chunk_size and hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame):
        print("\n⚙️  Processing in chunks...")
        chunks_cleaned = []
        for i, chunk in enumerate(df):
            print(f"\nProcessing chunk {i+1}...")
            chunk_cleaned = clean_data(chunk)
            chunks_cleaned.append(chunk_cleaned)
        df_cleaned = pd.concat(chunks_cleaned, ignore_index=True)
        print(f"\n✅ Concatenated {len(chunks_cleaned)} chunks")
    else:
        # Explore if requested
        if args.explore:
            explore_data(df)
            response = input("\nContinue with cleaning? (y/n): ")
            if response.lower() != "y":
                print("Aborted.")
                sys.exit(0)

        # Clean data
        df_cleaned = clean_data(df)

    # Save cleaned data
    save_cleaned_data(df_cleaned, args.output)

    logger.info("=" * 60)
    logger.info("Data Cleaning Complete!")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {datetime.now()}")

    print("\n" + "=" * 60)
    print("✅ Data Cleaning Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Feature engineering: python scripts/04_feature_engineering.py")
    print("2. Start analysis: jupyter notebook notebooks/01_exploratory_analysis.ipynb")
    print(f"\n📄 Logs saved to: logs/data_cleaning.log")

    return 0


if __name__ == "__main__":
    exit(main())
