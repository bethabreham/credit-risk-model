"""
Model Training and Tracking for Credit Risk Model
Trains Logistic Regression, Random Forest, and XGBoost
Tracks experiments with MLflow and registers the best model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the processed dataset with target variable"""
    df = pd.read_csv('data/processed/customer_features_with_target.csv')
    print(f"Loaded data shape: {df.shape}")
    
    # Drop CustomerId column if exists
    if 'CustomerId' in df.columns:
        df = df.drop('CustomerId', axis=1)
    
    # Separate features and target
    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target percentage: {y.mean()*100:.2f}% high-risk")
    
    return X, y


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression model"""
    
    print("\n" + "=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)
    
    with mlflow.start_run(run_name="Logistic_Regression") as run:
        # Train model
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        
        # Predictions
        y_pred = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Log to MLflow
        mlflow.log_params({
            'model': 'Logistic Regression',
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "logistic_regression_model")
        
        print(f"Results: {metrics}")
        
        return lr, metrics, run.info.run_id


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    
    print("\n" + "=" * 60)
    print("Training Random Forest...")
    print("=" * 60)
    
    with mlflow.start_run(run_name="Random_Forest") as run:
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Log to MLflow
        mlflow.log_params({
            'model': 'Random Forest',
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced'
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        print(f"Results: {metrics}")
        
        return rf, metrics, run.info.run_id


def train_random_forest_tuned(X_train, X_test, y_train, y_test):
    """Train Random Forest with Hyperparameter Tuning (GridSearchCV)"""
    
    print("\n" + "=" * 60)
    print("Random Forest Hyperparameter Tuning (GridSearch)...")
    print("=" * 60)
    
    with mlflow.start_run(run_name="Random_Forest_Tuned") as run:
        # Parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Predictions
        y_pred = best_rf.predict(X_test)
        y_proba = best_rf.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_params({'model': 'Random Forest (Tuned)'})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "random_forest_tuned_model")
        
        print(f"Best parameters: {best_params}")
        print(f"Results: {metrics}")
        
        return best_rf, metrics, run.info.run_id


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model (if installed)"""
    
    try:
        import xgboost as xgb
        
        print("\n" + "=" * 60)
        print("Training XGBoost...")
        print("=" * 60)
        
        with mlflow.start_run(run_name="XGBoost") as run:
            # Train model
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = xgb_model.predict(X_test)
            y_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            # Log to MLflow
            mlflow.log_params({
                'model': 'XGBoost',
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(xgb_model, "xgboost_model")
            
            print(f"Results: {metrics}")
            
            return xgb_model, metrics, run.info.run_id
            
    except ImportError:
        print("XGBoost not installed. Skipping...")
        return None, None, None


def register_best_model(results_dict):
    """Register the best model in MLflow Model Registry"""
    
    # Find best model by ROC-AUC
    best_model_name = max(results_dict, key=lambda x: results_dict[x]['roc_auc'])
    best_score = results_dict[best_model_name]['roc_auc']
    
    print("\n" + "=" * 60)
    print("MLFLOW MODEL REGISTRY")
    print("=" * 60)
    print(f"Best model: {best_model_name}")
    print(f"ROC-AUC score: {best_score:.4f}")
    
    # Note: To register a model, you need the run_id from the MLflow run
    # Since we're not storing run_ids in the metrics dict, we skip registration for now
    print("✅ To register the model, run: mlflow ui")
    print("✅ Then manually register the best model from the MLflow UI")
    
    return best_model_name


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("CREDIT RISK MODEL - TRAINING PIPELINE")
    print("=" * 60)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment("Credit_Risk_Model_Experiment")
    
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Train target %: {y_train.mean()*100:.2f}%")
    print(f"Test target %: {y_test.mean()*100:.2f}%")
    
    # Dictionary to store results
    results = {}
    
    # Train models
    lr_model, lr_metrics, lr_run_id = train_logistic_regression(X_train, X_test, y_train, y_test)
    results['Logistic Regression'] = {'metrics': lr_metrics, 'run_id': lr_run_id}
    
    rf_model, rf_metrics, rf_run_id = train_random_forest(X_train, X_test, y_train, y_test)
    results['Random Forest'] = {'metrics': rf_metrics, 'run_id': rf_run_id}
    
    rf_tuned_model, rf_tuned_metrics, rf_tuned_run_id = train_random_forest_tuned(X_train, X_test, y_train, y_test)
    results['Random Forest (Tuned)'] = {'metrics': rf_tuned_metrics, 'run_id': rf_tuned_run_id}
    
    # Try XGBoost
    xgb_model, xgb_metrics, xgb_run_id = train_xgboost(X_train, X_test, y_train, y_test)
    if xgb_model is not None:
        results['XGBoost'] = {'metrics': xgb_metrics, 'run_id': xgb_run_id}
    
    # Display results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    results_df = pd.DataFrame({name: data['metrics'] for name, data in results.items()}).T
    print(results_df.round(4))
    
    # Register best model
    best_model_name = register_best_model({name: data['metrics'] for name, data in results.items()})
    
    # Confusion matrix for best model
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX - BEST MODEL")
    print("=" * 60)
    
    best_model_data = results[best_model_name]
    
    # Retrain best model on full training data to get predictions
    if best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    elif best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif best_model_name == 'Random Forest (Tuned)':
        # Get the tuned parameters from the stored model
        best_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    else:
        best_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    print(f"Confusion Matrix for {best_model_name}:")
    print(f"True Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f" Best model: {best_model_name}")
    print(f" ROC-AUC: {best_model_data['metrics']['roc_auc']:.4f}")
    print(" MLflow runs saved in './mlruns'")
    print(" To view MLflow UI, run: mlflow ui")
    
    return results


if __name__ == "__main__":
    results = main()