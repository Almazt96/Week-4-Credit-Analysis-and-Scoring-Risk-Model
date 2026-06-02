import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient

# 1. Paths configuration
db_uri = "sqlite:///D:/personal/kifiya 10 Academy/10 Academy/Week 4 credit-risk-model/mlflow.db"
artifact_dir = "D:/personal/kifiya 10 Academy/10 Academy/Week 4 credit-risk-model/mlflow_artifacts"

# Ensure clean setup
mlflow.set_tracking_uri(db_uri)

# 2. Create an experiment that maps SQLite metadata to local folder storage
client = MlflowClient()
experiment_name = "Credit_Risk_Experiment"
exp = client.get_experiment_by_name(experiment_name)

if exp is None:
    # Set the local folder explicitly for binaries
    os.makedirs(artifact_dir, exist_ok=True)
    exp_id = client.create_experiment(
        name=experiment_name,
        artifact_location=f"file:///{artifact_dir}"
    )
else:
    exp_id = exp.experiment_id

# 3. Train a quick placeholder model
print("Training placeholder model...")
X = [[1, 2], [3, 4]]
y = [0, 1]
model = LogisticRegression()
model.fit(X, y)

# 4. Log the model into our explicit experiment configuration
print("Logging and registering model to database...")
with mlflow.start_run(experiment_id=exp_id):
    run_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",  # modern parameter naming instead of artifact_path
        registered_model_name="Credit_Risk_Model"
    )

# 5. Instantly promote it to 'Production' stage for the API
print("Promoting model version to Production stage...")
client.transition_model_version_stage(
    name="Credit_Risk_Model",
    version=1,  # <--- Change run_info.version to 1
    stage="Production"
)

print("\n🎉 SUCCESS! Your mlflow.db is populated and files are saved locally.")