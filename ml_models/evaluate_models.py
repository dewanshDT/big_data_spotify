#!/usr/bin/env python3
"""
ML Model Evaluation and Visualization Script

This script loads the trained ML models and generates comprehensive
evaluation visualizations including:
- Popularity Model: actual vs predicted, residuals, feature importance
- Recommendation Model: cluster visualization, characteristics, distribution
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_data(data_path, sample_size=None):
    """Load the processed data"""
    logger = setup_logger("model_evaluation")
    logger.info(f"Loading data from {data_path}")

    data_path = Path(data_path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size:,} rows from {len(df):,}")
        df = df.sample(n=sample_size, random_state=42)

    logger.info(f"Loaded {len(df):,} rows")
    return df


def evaluate_popularity_model(model_dir, data_path, output_dir, sample_size=None):
    """Evaluate and visualize the popularity prediction model"""
    logger = setup_logger("popularity_evaluation")
    logger.info("=" * 80)
    logger.info("EVALUATING POPULARITY PREDICTION MODEL")
    logger.info("=" * 80)

    # Load model and metadata
    model_path = model_dir / "popularity_model.pkl"
    metadata_path = model_dir / "popularity_model_metadata.pkl"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    logger.info("Loading model...")
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    feature_columns = metadata["feature_cols"]
    logger.info(f"Model uses {len(feature_columns)} features")

    # Load data
    df = load_data(data_path, sample_size)

    # Handle column name mismatches (legacy compatibility)
    if "track_length_min" in feature_columns and "track_length_min" not in df.columns:
        if "duration_minutes" in df.columns:
            df["track_length_min"] = df["duration_minutes"]
            logger.info("Mapped duration_minutes -> track_length_min")

    # Prepare features
    X = df[feature_columns]
    y_true = (
        df["popularity_score"] if "popularity_score" in df.columns else df["popularity"]
    )

    # Make predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    logger.info(f"MAE: {mae:.3f}")
    logger.info(f"RMSE: {rmse:.3f}")
    logger.info(f"R² Score: {r2:.3f}")

    # Create visualizations
    fig = plt.figure(figsize=(20, 12))

    # 1. Actual vs Predicted Scatter Plot
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(y_true, y_pred, alpha=0.3, s=10, c=y_true, cmap="viridis")
    ax1.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    ax1.set_xlabel("Actual Popularity", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Popularity", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Actual vs Predicted Popularity\n(Regression Model)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Actual Popularity")

    # 2. Residual Plot
    ax2 = plt.subplot(2, 3, 2)
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10, c="steelblue")
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Popularity", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Residuals (Actual - Predicted)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Residual Plot\n(Error Distribution)", fontsize=14, fontweight="bold", pad=15
    )
    ax2.grid(True, alpha=0.3)

    # 3. Error Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="coral")
    ax3.axvline(x=0, color="r", linestyle="--", lw=2, label="Zero Error")
    ax3.set_xlabel("Residual Value", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax3.set_title(
        "Distribution of Prediction Errors\n(Histogram)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Feature Importance
    ax4 = plt.subplot(2, 3, 4)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features

        ax4.barh(range(len(indices)), importances[indices], color="teal", alpha=0.7)
        ax4.set_yticks(range(len(indices)))
        ax4.set_yticklabels([feature_columns[i] for i in indices])
        ax4.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax4.set_title(
            "Top 15 Feature Importances\n(XGBoost)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax4.grid(True, alpha=0.3, axis="x")
        ax4.invert_yaxis()

    # 5. Metrics Table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("off")

    metrics_data = [
        ["Metric", "Value", "Interpretation"],
        ["Mean Absolute Error (MAE)", f"{mae:.3f}", "Avg prediction error"],
        ["Root Mean Squared Error (RMSE)", f"{rmse:.3f}", "Error magnitude"],
        ["R² Score", f"{r2:.3f}", "Variance explained"],
        ["Sample Size", f"{len(y_true):,}", "Tracks evaluated"],
        ["Min Actual", f"{y_true.min():.1f}", "Lowest popularity"],
        ["Max Actual", f"{y_true.max():.1f}", "Highest popularity"],
        ["Mean Actual", f"{y_true.mean():.1f}", "Average popularity"],
        ["Min Predicted", f"{y_pred.min():.1f}", "Lowest prediction"],
        ["Max Predicted", f"{y_pred.max():.1f}", "Highest prediction"],
    ]

    table = ax5.table(
        cellText=metrics_data, cellLoc="left", loc="center", colWidths=[0.4, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax5.set_title(
        "Model Performance Metrics\n(Regression Statistics)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # 6. Prediction Accuracy by Range
    ax6 = plt.subplot(2, 3, 6)

    # Bin predictions into ranges
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    y_true_binned = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)

    mae_by_range = []
    for label in labels:
        mask = y_true_binned == label
        if mask.sum() > 0:
            mae_range = mean_absolute_error(y_true[mask], y_pred[mask])
            mae_by_range.append(mae_range)
        else:
            mae_by_range.append(0)

    bars = ax6.bar(labels, mae_by_range, color="purple", alpha=0.7, edgecolor="black")
    ax6.set_xlabel("Actual Popularity Range", fontsize=12, fontweight="bold")
    ax6.set_ylabel("Mean Absolute Error", fontsize=12, fontweight="bold")
    ax6.set_title(
        "Prediction Error by Popularity Range\n(MAE Distribution)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax6.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.suptitle(
        "🎯 Popularity Prediction Model - Comprehensive Evaluation",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = output_dir / "ml_eval_1_popularity_model.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "sample_size": len(y_true)}


def evaluate_recommendation_model(model_dir, data_path, output_dir, sample_size=None):
    """Evaluate and visualize the recommendation clustering model"""
    logger = setup_logger("recommendation_evaluation")
    logger.info("=" * 80)
    logger.info("EVALUATING RECOMMENDATION CLUSTERING MODEL")
    logger.info("=" * 80)

    # Load model and metadata
    kmeans_path = model_dir / "recommendation_kmeans.pkl"
    scaler_path = model_dir / "recommendation_scaler.pkl"
    metadata_path = model_dir / "recommendation_metadata.pkl"

    if not kmeans_path.exists():
        logger.error(f"Model not found at {kmeans_path}")
        return

    logger.info("Loading model...")
    kmeans = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)

    feature_columns = metadata["feature_cols"]
    n_clusters = metadata["n_clusters"]
    logger.info(f"Model uses {len(feature_columns)} features, {n_clusters} clusters")

    # Load data
    df = load_data(data_path, sample_size)

    # Prepare features
    X = df[feature_columns].values
    X_scaled = scaler.transform(X)

    # Get cluster assignments
    logger.info("Assigning clusters...")
    clusters = kmeans.predict(X_scaled)
    df["cluster"] = clusters

    # Calculate silhouette score (sample for speed - it's O(n²)!)
    logger.info("Calculating silhouette score...")
    if len(X_scaled) > 50000:
        logger.info(f"Using 50K sample for silhouette score (O(n²) complexity)")
        sample_idx = np.random.choice(len(X_scaled), 50000, replace=False)
        silhouette_avg = silhouette_score(X_scaled[sample_idx], clusters[sample_idx])
    else:
        silhouette_avg = silhouette_score(X_scaled, clusters)
    logger.info(f"Silhouette Score: {silhouette_avg:.3f}")

    # Create visualizations
    fig = plt.figure(figsize=(20, 12))

    # 1. Cluster Size Distribution
    ax1 = plt.subplot(2, 3, 1)
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    bars = ax1.bar(
        cluster_counts.index,
        cluster_counts.values,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_xlabel("Cluster ID", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Tracks", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Cluster Size Distribution\n(K-Means)", fontsize=14, fontweight="bold", pad=15
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 2. PCA Visualization (2D)
    ax2 = plt.subplot(2, 3, 2)
    logger.info("Performing PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    scatter = ax2.scatter(
        X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.5, s=10
    )
    ax2.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_title(
        "Cluster Visualization (PCA Projection)\n(2D Projection)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.colorbar(scatter, ax=ax2, label="Cluster ID")
    ax2.grid(True, alpha=0.3)

    # 3. Cluster Characteristics Heatmap
    ax3 = plt.subplot(2, 3, 3)

    # Calculate mean features per cluster
    cluster_features = df.groupby("cluster")[feature_columns].mean()

    # Normalize for better visualization
    cluster_features_norm = (cluster_features - cluster_features.min()) / (
        cluster_features.max() - cluster_features.min()
    )

    sns.heatmap(
        cluster_features_norm.T,
        annot=False,
        cmap="YlOrRd",
        cbar_kws={"label": "Normalized Value"},
        ax=ax3,
    )
    ax3.set_xlabel("Cluster ID", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Audio Features", fontsize=12, fontweight="bold")
    ax3.set_title(
        "Cluster Characteristics Heatmap\n(Feature Profiles)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # 4. Top Features by Cluster Variance
    ax4 = plt.subplot(2, 3, 4)

    feature_variance = cluster_features.var().sort_values(ascending=False)[:10]
    ax4.barh(
        range(len(feature_variance)),
        feature_variance.values,
        color="coral",
        alpha=0.7,
        edgecolor="black",
    )
    ax4.set_yticks(range(len(feature_variance)))
    ax4.set_yticklabels(feature_variance.index)
    ax4.set_xlabel("Variance Across Clusters", fontsize=12, fontweight="bold")
    ax4.set_title(
        "Features with Highest Cluster Variance\n(Distinguishing Features)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax4.grid(True, alpha=0.3, axis="x")
    ax4.invert_yaxis()

    # 5. Silhouette Analysis
    ax5 = plt.subplot(2, 3, 5)

    # Calculate silhouette samples (sample for speed)
    if len(X_scaled) > 10000:
        sample_idx = np.random.choice(len(X_scaled), 10000, replace=False)
        silhouette_vals = silhouette_samples(X_scaled[sample_idx], clusters[sample_idx])
        clusters_sample = clusters[sample_idx]
    else:
        silhouette_vals = silhouette_samples(X_scaled, clusters)
        clusters_sample = clusters

    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[clusters_sample == i]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.tab10(i / n_clusters)
        ax5.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax5.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i,
            str(i),
            fontsize=10,
            fontweight="bold",
        )
        y_lower = y_upper + 10

    ax5.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Average: {silhouette_avg:.3f}",
    )
    ax5.set_xlabel("Silhouette Coefficient", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Cluster", fontsize=12, fontweight="bold")
    ax5.set_title(
        "Silhouette Analysis\n(Cluster Quality)", fontsize=14, fontweight="bold", pad=15
    )
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="x")

    # 6. Metrics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    metrics_data = [
        ["Metric", "Value", "Interpretation"],
        ["Number of Clusters", f"{n_clusters}", "K-Means clusters"],
        ["Silhouette Score", f"{silhouette_avg:.3f}", "Quality (-1 to 1)"],
        ["Total Tracks", f"{len(df):,}", "Samples clustered"],
        ["Features Used", f"{len(feature_columns)}", "Audio features"],
        ["Inertia", f"{kmeans.inertia_:,.0f}", "Within-cluster sum"],
        ["Min Cluster Size", f"{cluster_counts.min():,}", "Smallest cluster"],
        ["Max Cluster Size", f"{cluster_counts.max():,}", "Largest cluster"],
        ["Mean Cluster Size", f"{cluster_counts.mean():,.0f}", "Average size"],
        ["Std Cluster Size", f"{cluster_counts.std():,.0f}", "Size variation"],
    ]

    table = ax6.table(
        cellText=metrics_data, cellLoc="left", loc="center", colWidths=[0.4, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor("#2196F3")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax6.set_title(
        "Model Performance Metrics\n(Clustering Statistics)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.suptitle(
        "🎵 Recommendation Clustering Model - Comprehensive Evaluation",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = output_dir / "ml_eval_2_recommendation_model.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")

    return {
        "silhouette_score": silhouette_avg,
        "n_clusters": n_clusters,
        "inertia": kmeans.inertia_,
        "sample_size": len(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize ML models")
    parser.add_argument(
        "--input",
        default="data/processed/spotify_features.parquet",
        help="Path to processed data file",
    )
    parser.add_argument(
        "--model-dir", default="ml_models", help="Directory containing trained models"
    )
    parser.add_argument(
        "--output", default="visualizations", help="Output directory for visualizations"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of samples to use for evaluation (default: all)",
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_path = project_root / args.input
    model_dir = project_root / args.model_dir
    output_dir = project_root / args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("model_evaluation")

    logger.info("=" * 80)
    logger.info("🤖 ML MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Data: {data_path}")
    logger.info(f"Models: {model_dir}")
    logger.info(f"Output: {output_dir}")
    if args.sample:
        logger.info(f"Sample size: {args.sample:,}")
    logger.info("=" * 80)

    # Evaluate both models
    results = {}

    try:
        logger.info("\n📊 Evaluating Popularity Prediction Model...")
        pop_results = evaluate_popularity_model(
            model_dir, data_path, output_dir, args.sample
        )
        results["popularity"] = pop_results
    except Exception as e:
        logger.error(f"Error evaluating popularity model: {e}")

    try:
        logger.info("\n🎵 Evaluating Recommendation Clustering Model...")
        rec_results = evaluate_recommendation_model(
            model_dir, data_path, output_dir, args.sample
        )
        results["recommendation"] = rec_results
    except Exception as e:
        logger.error(f"Error evaluating recommendation model: {e}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 80)

    if "popularity" in results:
        logger.info("\n🎯 Popularity Prediction Model:")
        logger.info(f"  MAE: {results['popularity']['mae']:.3f}")
        logger.info(f"  RMSE: {results['popularity']['rmse']:.3f}")
        logger.info(f"  R² Score: {results['popularity']['r2']:.3f}")
        logger.info(f"  Samples: {results['popularity']['sample_size']:,}")

    if "recommendation" in results:
        logger.info("\n🎵 Recommendation Clustering Model:")
        logger.info(
            f"  Silhouette Score: {results['recommendation']['silhouette_score']:.3f}"
        )
        logger.info(f"  Clusters: {results['recommendation']['n_clusters']}")
        logger.info(f"  Inertia: {results['recommendation']['inertia']:,.0f}")
        logger.info(f"  Samples: {results['recommendation']['sample_size']:,}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ MODEL EVALUATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
