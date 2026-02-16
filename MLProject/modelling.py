"""
CI/CD Model Training Pipeline with MLflow
House Prices - Random Forest Model
Author: Amirullah
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CI
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

import mlflow
import mlflow.sklearn
import os
import sys
import joblib
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("CI/CD MODEL TRAINING PIPELINE")
print("House Prices - Random Forest")
print("Author: Amirullah")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================
print("\n[1/7] Loading configuration...")

# Parse command line arguments from MLProject
if len(sys.argv) >= 4:
    N_ESTIMATORS = int(sys.argv[1])
    MAX_DEPTH = int(sys.argv[2])
    RANDOM_STATE = int(sys.argv[3])
else:
    # Default values
    N_ESTIMATORS = 200
    MAX_DEPTH = 15
    RANDOM_STATE = 42

# Detect CI environment
IS_CI = os.getenv('CI', 'false').lower() == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'

# Paths
TRAIN_DATA_PATH = 'dataset_preprocessing/train_processed.csv'
TEST_DATA_PATH = 'dataset_preprocessing/test_processed.csv'

# Experiment name
EXPERIMENT_NAME = "CI_House_Prices_Training"

print(f"   Environment: {'CI/CD' if IS_CI else 'Local'}")
print(f"   N_Estimators: {N_ESTIMATORS}")
print(f"   Max_Depth: {MAX_DEPTH}")
print(f"   Random_State: {RANDOM_STATE}")
print("[OK] Configuration loaded")

# ============================================================
# SETUP MLFLOW
# ============================================================
print("\n[2/7] Setting up MLflow...")

# Set tracking URI based on environment
mlflow.set_tracking_uri("file:./mlruns")
print("   Using file-based tracking")

# Set experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[OK] Experiment: {EXPERIMENT_NAME}")

# ============================================================
# LOAD DATA
# ============================================================
print("\n[3/7] Loading preprocessed data...")

try:
    # Load training data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Separate features and target
    X_train = train_df.drop('SalePrice', axis=1)
    y_train = train_df['SalePrice']

    X_test = test_df.drop('SalePrice', axis=1)
    y_test = test_df['SalePrice']

    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    print(f"   Features: {X_train.shape[1]}")
    print("[OK] Data loaded successfully")

except FileNotFoundError as e:
    print(f"[ERROR] Data file not found: {e}")
    print("   Please ensure preprocessed data is in dataset_preprocessing/ folder")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    sys.exit(1)

# ============================================================
# MODEL TRAINING
# ============================================================
print("\n[4/7] Training model...")

# Start MLflow run
with mlflow.start_run(run_name=f"CI_RF_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

    # Log parameters
    params = {
        'model_type': 'RandomForest',
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'environment': 'CI' if IS_CI else 'Local'
    }

    mlflow.log_params(params)
    print("   Parameters logged")

    # Train model
    print(f"   Training RandomForest...")
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
        n_jobs=params['n_jobs']
    )

    model.fit(X_train, y_train)
    print("[OK] Model trained")

    # ============================================================
    # PREDICTIONS & METRICS
    # ============================================================
    print("\n[5/7] Evaluating model...")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    correlation, _ = pearsonr(y_test, y_test_pred)

    # Print metrics
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print("\nTraining Metrics:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R2:   {train_r2:.4f}")

    print("\nTest Metrics:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    print(f"  Correlation: {correlation:.4f}")

    # Log metrics
    mlflow.log_metrics({
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'correlation': correlation
    })

    print("\n[OK] Metrics logged")

    # ============================================================
    # CREATE ARTIFACTS
    # ============================================================
    print("\n[6/7] Creating artifacts...")

    # Create artifacts directory
    os.makedirs('artifacts', exist_ok=True)

    # Feature Importance
    print("   Creating feature importance plot...")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
    plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=120, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact('artifacts/feature_importance.png')

    # Actual vs Predicted
    print("   Creating actual vs predicted plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Sale Price (log)')
    plt.ylabel('Predicted Sale Price (log)')
    plt.title(f'Actual vs Predicted (R2={test_r2:.4f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/actual_vs_predicted.png', dpi=120, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact('artifacts/actual_vs_predicted.png')

    # Metrics Summary
    print("   Creating metrics summary...")
    metrics_summary = {
        'model': 'RandomForest',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'environment': 'CI' if IS_CI else 'Local',
        'parameters': params,
        'training_metrics': {
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2)
        },
        'test_metrics': {
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'mape': float(test_mape),
            'correlation': float(correlation)
        },
        'data_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1]
        }
    }

    with open('artifacts/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    mlflow.log_artifact('artifacts/metrics_summary.json')

    # Save model
    print("   Saving model...")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/model.pkl'
    joblib.dump(model, model_path)

    # Log model file as artifact
    mlflow.log_artifact(model_path)

    # Log model to MLflow Model Registry with signature
    print("   Logging model to MLflow registry...")
    from mlflow.models.signature import infer_signature

    # Infer model signature from training data
    signature = infer_signature(X_train, y_train_pred)

    # Create conda environment spec
    conda_env = {
        'channels': ['conda-forge', 'defaults'],
        'dependencies': [
            'python=3.12',
            'pip',
            {
                'pip': [
                    'mlflow==2.11.3',
                    'scikit-learn==1.4.0',
                    'pandas==2.2.0',
                    'numpy==1.26.3',
                    'cloudpickle==3.0.0',
                ]
            }
        ],
        'name': 'mlflow-env'
    }

    mlflow.sklearn.log_model(
        model,
        "model",
        conda_env=conda_env,
        signature=signature,
        registered_model_name="RandomForest_CI_HousePrices",
        input_example=X_train.iloc[:5]  # Add input example
    )

    print("[OK] Artifacts created and logged")

    # ============================================================
    # COMPLETION
    # ============================================================
    print("\n[7/7] Training completed!")

    run = mlflow.active_run()
    run_id = run.info.run_id

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n[OK] Model: RandomForest")
    print(f"[OK] Test R2: {test_r2:.4f}")
    print(f"[OK] Test RMSE: {test_rmse:.4f}")
    print(f"[OK] MLflow Run ID: {run_id}")
    print(f"[OK] Environment: {'CI/CD Pipeline' if IS_CI else 'Local Development'}")
    print(f"[OK] Artifacts saved to: artifacts/ and models/")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Exit with success
    sys.exit(0)
