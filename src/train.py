# Model training
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data():
    # Placeholder: Replace with your actual data ingestion logic
    # Example: df = pd.read_csv("data/processed_data.csv")
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
    y = np.random.randint(0, 2, size=100)
    return X, y

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    return metrics

def train_and_track():
    # Set the MLflow experiment name
    mlflow.set_experiment("Model_Training_and_Tracking")
    
    # 1. Data Preparation
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Define Model Search Spaces
    model_definitions = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1.0, 10.0]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
        }
    }
    
    best_overall_score = -1
    best_model_uri = None
    best_model_name = None

    # 3. Model Selection, Tuning, and Experiment Tracking
    for model_name, config in model_definitions.items():
        # Start an MLflow parent run for this specific algorithm
        with mlflow.start_run(run_name=f"Tuning_{model_name}"):
            
            print(f"Running Hyperparameter Tuning for {model_name}...")
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Extract best model and params
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Evaluate on unseen test data
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_prob)
            
            # Log Parameters and Metrics to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("algorithm", model_name)
            
            # Log the Model Artifact
            # input_example helps MLflow understand the expected data schema
            input_example = X_train.head(1)
            model_info = mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                input_example=input_example
            )
            
            print(f"{model_name} Metrics: {metrics}")
            
            # Track the absolute best model based on F1 Score
            if metrics["f1_score"] > best_overall_score:
                best_overall_score = metrics["f1_score"]
                best_model_uri = model_info.model_uri
                best_model_name = model_name

    # 4. Model Registry
    if best_model_uri:
        print(f"\nRegistering the best model ({best_model_name}) in the MLflow Registry...")
        registered_model_name = "Best_Classification_Model"
        mlflow.register_model(model_uri=best_model_uri, name=registered_model_name)
        print(f"Model successfully registered as '{registered_model_name}'!")

if __name__ == "__main__":
    train_and_track()