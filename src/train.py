# Model training
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def load_data(filepath):
    """Loads preprocessed data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates evaluation metrics for classification."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    }
    return metrics

def train_and_track():
    # Setup MLflow Experiment
    mlflow.set_experiment("Model_Training_and_Tracking")
    
    # Define paths (Adjust these paths to fit your project setup)
    data_path = "data/processed_data.csv" 
    target_col = "target" # Replace with your actual target column name
    
    # 1. Data Prep
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, target_column=target_col)
    
    # Define models and hyperparameter spaces
    model_definitions = [
        {
            "name": "Random_Forest",
            "model": RandomForestClassifier(random_state=42),
            "param_dist": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        {
            "name": "Gradient_Boosting",
            "model": GradientBoostingClassifier(random_state=42),
            "param_dist": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        }
    ]
    
    best_overall_f1 = 0
    best_run_id = None
    best_model_name = None

    for m_def in model_definitions:
        # Start MLflow parent run for this model architecture
        with mlflow.start_run(run_name=f"Hyperparameter_Tuning_{m_def['name']}") as parent_run:
            
            print(f"Running Random Search for {m_def['name']}...")
            search = RandomizedSearchCV(
                estimator=m_def["model"],
                param_distributions=m_def["param_dist"],
                n_iter=4,
                cv=3,
                scoring='f1',
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            
            # Extract best model and params
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            # Predict and evaluate
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            metrics = evaluate_model(y_test, y_pred, y_prob)
            
            # Log Parameters & Metrics to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", m_def["name"])
            
            # Save and Log Sample Test Data as an Artifact
            test_summary_path = "test_summary.csv"
            X_test.head().to_csv(test_summary_path, index=False)
            mlflow.log_artifact(test_summary_path)
            if os.path.exists(test_summary_path):
                os.remove(test_summary_path)
            
            # Log the Model Framework artifact
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            
            print(f"{m_def['name']} Results -> F1-Score: {metrics['f1_score']:.4f}")
            
            # Track best model globally across loops
            if metrics["f1_score"] > best_overall_f1:
                best_overall_f1 = metrics["f1_score"]
                best_run_id = parent_run.info.run_id
                best_model_name = m_def["name"]

    # 3. Register the absolute best model in the MLflow Model Registry
    if best_run_id:
        print(f"\nRegistering the best model: {best_model_name} (Run ID: {best_run_id})")
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "Best_Production_Classifier")

if __name__ == "__main__":
    train_and_track()