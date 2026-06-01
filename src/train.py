# Model training
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

from data_processing import build_feature_pipeline, engineer_proxy_target

def run_training_lifecycle(data_path="./data/raw/data.csv"):
    df = pd.read_csv(data_path)
    
    # Generate Target Labels
    targets = engineer_proxy_target(df)
    
    # Build and fit pipeline
    pipeline = build_feature_pipeline()
    
    # Align X and y mapping indices by CustomerId
    y_mapped = targets['is_high_risk']
    
    # Fit-Transform the pipeline data
    X_transformed = pipeline.fit_transform(df, y_mapped)
    y_final = y_mapped.loc[X_transformed.index]
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_final, test_size=0.2, stratify=y_final, random_state=42
    )
    
    mlflow.set_experiment("Bati_Bank_Credit_Risk")
    
    # Model 1: Logistic Regression (Baseline)
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        param_grid_lr = {'C': [0.01, 0.1, 1.0, 10.0]}
        grid_lr = GridSearchCV(lr, param_grid_lr, cv=3, scoring='roc_auc')
        grid_lr.fit(X_train, y_train)
        
        best_lr = grid_lr.best_estimator_
        preds = best_lr.predict(X_test)
        probs = best_lr.predict_proba(X_test)[:, 1]
        
        # Log Metrics
        mlflow.log_params(grid_lr.best_params_)
        mlflow.log_metric("ROC_AUC", roc_auc_score(y_test, probs))
        mlflow.log_metric("F1_Score", f1_score(y_test, preds))
        mlflow.sklearn.log_model(best_lr, "model", registered_model_name="LR_Baseline")
        
    # Model 2: LightGBM (Challenger)
    with mlflow.start_run(run_name="LightGBM_Challenger"):
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        param_grid_lgb = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100]
        }
        grid_lgb = GridSearchCV(lgb_model, param_grid_lgb, cv=3, scoring='roc_auc')
        grid_lgb.fit(X_train, y_train)
        
        best_lgb = grid_lgb.best_estimator_
        lgb_preds = best_lgb.predict(X_test)
        lgb_probs = best_lgb.predict_proba(X_test)[:, 1]
        
        lgb_auc = roc_auc_score(y_test, lgb_probs)
        mlflow.log_params(grid_lgb.best_params_)
        mlflow.log_metric("ROC_AUC", lgb_auc)
        mlflow.log_metric("F1_Score", f1_score(y_test, lgb_preds))
        mlflow.lightgbm.log_model(best_lgb, "model")
        
        # Save champion model mapping configurations locally for container usage
        os.makedirs("models", exist_ok=True)
        with open("models/pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        with open("models/champion_model.pkl", "wb") as f:
            pickle.dump(best_lgb, f)
            
        # Programmatic production promotion simulation mapping
        print(f"Champion Model Trained successfully with ROC-AUC: {lgb_auc}")

if __name__ == "__main__":
    run_training_lifecycle()