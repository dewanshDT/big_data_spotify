"""
Popularity Prediction Model
Predicts song popularity based on audio features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import yaml

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """Load feature-engineered data"""
    file_path = Path(file_path)
    parquet_path = file_path.with_suffix('.parquet')
    
    print(f"\n📂 Loading data...")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"✅ Loaded {len(df):,} rows")
    return df

def prepare_features(df):
    """Prepare features for modeling"""
    print("\n🔧 Preparing features...")
    
    # Select numeric audio features (based on actual dataset)
    feature_cols = [
        'energy', 'danceability', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 'track_length_min'
    ]
    
    # Add num_followers if available
    if 'num_followers' in df.columns:
        feature_cols.append('num_followers')
    
    # Add engineered features if available
    if 'party_score' in df.columns:
        feature_cols.extend(['party_score', 'chill_score', 'hype_score'])
    
    if 'high_follower_playlist' in df.columns:
        feature_cols.extend(['high_follower_playlist'])
    
    if 'artist_avg_popularity' in df.columns:
        feature_cols.extend(['artist_avg_popularity', 'artist_track_count'])
    
    # Filter columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"   Using {len(feature_cols)} features: {feature_cols}")
    
    # Extract features and target
    X = df[feature_cols].copy()
    # Use 'popularity_score' if available, otherwise 'popularity'
    pop_col = 'popularity_score' if 'popularity_score' in df.columns else 'popularity'
    y = df[pop_col].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target variable shape: {y.shape}")
    
    return X, y, feature_cols

def train_model(X_train, y_train, X_test, y_test, config):
    """Train XGBoost model"""
    print("\n🤖 Training XGBoost model...")
    
    # Model parameters from config
    params = {
        'max_depth': config['ml_models']['popularity_prediction']['max_depth'],
        'n_estimators': config['ml_models']['popularity_prediction']['n_estimators'],
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'random_state': config['ml_models']['random_state'],
        'n_jobs': -1
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             early_stopping_rounds=10,
             verbose=False)
    
    print("✅ Model training complete!")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nModel Performance:")
    print("="*50)
    print(f"Train MAE:  {train_mae:.2f}  |  Test MAE:  {test_mae:.2f}")
    print(f"Train RMSE: {train_rmse:.2f} |  Test RMSE: {test_rmse:.2f}")
    print(f"Train R²:   {train_r2:.3f} |  Test R²:   {test_r2:.3f}")
    print("="*50)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

def plot_results(y_test, y_test_pred, feature_importance):
    """Create visualizations"""
    print("\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Popularity', fontsize=12)
    axes[0].set_ylabel('Predicted Popularity', fontsize=12)
    axes[0].set_title('Actual vs Predicted Popularity', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Feature Importance
    top_features = feature_importance.head(10)
    axes[1].barh(range(len(top_features)), top_features['importance'])
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features['feature'])
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('../visualizations/popularity_model_results.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    plt.close()

def save_model(model, scaler, feature_cols, metrics):
    """Save trained model"""
    print("\n💾 Saving model...")
    
    output_dir = Path('../ml_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'popularity_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save scaler if used
    if scaler:
        scaler_path = output_dir / 'popularity_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved to: {scaler_path}")
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'metrics': metrics,
        'model_type': 'XGBoost Regressor'
    }
    metadata_path = output_dir / 'popularity_model_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"✅ Metadata saved to: {metadata_path}")

def main():
    print("="*60)
    print("Spotify Popularity Prediction Model")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Load data
    data_path = '../data/processed/spotify_features.csv'
    df = load_data(data_path)
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Train-test split
    test_size = config['ml_models']['test_size']
    random_state = config['ml_models']['random_state']
    
    print(f"\n✂️  Splitting data (test size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"   Train size: {len(X_train):,}")
    print(f"   Test size: {len(X_test):,}")
    
    # Optional: Scale features (XGBoost doesn't strictly need it, but can help)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    scaler = None
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test, config)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Plot results
    y_test_pred = model.predict(X_test)
    plot_results(y_test, y_test_pred, metrics['feature_importance'])
    
    # Save model
    save_model(model, scaler, feature_cols, metrics)
    
    print("\n" + "="*60)
    print("✅ Model Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review model performance metrics")
    print("2. Use model for predictions: model.predict(new_data)")
    print("3. Deploy model or integrate into pipeline")

if __name__ == "__main__":
    main()



