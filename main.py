import mlflow
import mlflow.sklearn
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main(max_iter, C, solver):
    # End any active MLflow runs
    if mlflow.active_run():
        mlflow.end_run()

    # Load data
    data = pd.read_csv('Device_Dataset.csv')
    X = data.drop('Stressed State', axis=1)  
    y = data['Stressed State']   

    # Convert 'Activity Type' to numerical using get_dummies
    X = pd.get_dummies(X, columns=['Activity Type'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with specified hyperparameters
    model = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model and parameters to MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model logged in run {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations for Logistic Regression")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength; smaller values specify stronger regularization.")
    parser.add_argument("--solver", type=str, default="lbfgs", help="Algorithm to use in the optimization problem (e.g., 'lbfgs', 'saga').")
    
    args = parser.parse_args()
    main(args.max_iter, args.C, args.solver)