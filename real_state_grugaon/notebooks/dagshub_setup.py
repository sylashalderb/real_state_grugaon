import mlflow
import dagshub
dagshub.init(repo_owner='sylashalderb', repo_name='project_kalke', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sylashalderb/project_kalke.mlflow")
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)