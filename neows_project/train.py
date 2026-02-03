#There are three objectives for this train.py file: final training of the model, threshold tuning, and testing the model
        #We will split our data into train, val, test then train the model using the train section 
        #Then we will tune the threshold using the validation set
        #And finally we will test our final model and threshold against the training data


from .pipeline import build_final_pipe
#The final pipeline to fit on the training data containing preprocessing, feature eng, scaler, and placeholder for model

import mlflow
import mlflow.sklearn
#Additionally we will log the params, metrics, pipeline, and model to mlflow for analysis

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
#These are the three possible models to choose from in this project,
# making the choice now should make it easier to change models should the need arise

import json
from pathlib import Path
import tempfile
#This needed to read the model name and hyperparameters from the chosen model logged in models/params

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#These imports are for loading and partitionioning the data to be used in training

from sklearn.metrics import fbeta_score, precision_score, accuracy_score, roc_auc_score, recall_score
#Importing some metrics to view how well our model did

#Setting the MLflow experiment up
mlflow.set_experiment(experiment_name="Final_Model")

#We will load the best model and hyperparameters from the params folder using json
def load_model():
    best_route = Path("models")/"params"/"best_RF.json"
    with open(best_route, "r") as f:
        cfg = json.load(f)
    cfg["_path"] = str(best_route)
    return cfg

#This next function is the training functionality
def train(X, y, cfg: dict):
    model_name = cfg["model_name"]
    best_params = cfg["best_params"]

    #Next function is to define the dictionary to transform the model_name into the proper name
    models = {
        "XG" : XGBClassifier,
        "RF" : RandomForestClassifier,
        "LG" : LogisticRegression
    }

    #These next few lines are for defining the chosen model (ch_model) building the pipeline with it and fitting it on X,y

    ch_model = models[model_name](**best_params)

    pipeline = build_final_pipe(ch_model)

    pipeline.fit(X,y)

    return pipeline, model_name, best_params

def threshold(X, y, pipeline):
    thresholds = np.linspace(0.01, 0.99, 200)

    best_thresh = 0.5
    best_score = -1

    prob = pipeline.predict_proba(X)[:,1]

    for t in thresholds:
        predict = (prob >= t).astype(int)
        score = fbeta_score(y, predict, beta=2)

        if score > best_score:
            best_score = score
            best_thresh = t
    
    return best_score, best_thresh



def main():

    data = pd.read_csv('data/raw/asteroids_data.csv')
    y = data['is_potentially_hazardous_asteroid'].astype(int)
    X = data.drop(columns='is_potentially_hazardous_asteroid')


    #Splitting the original dataset into train, val, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size= 0.15, random_state=42, shuffle=True, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, shuffle=True, stratify=y_train_val)


    cfg = load_model()

    pipeline, model_name, best_params = train(X_train, y_train, cfg)

    best_score, best_thresh = threshold(X_val, y_val, pipeline)

    test_prob = pipeline.predict_proba(X_test)[:,1]
    test_predict = (test_prob >= best_thresh).astype(int)

    f2_score = fbeta_score(y_test, test_predict, beta=2)
    accuracy = accuracy_score(y_test, test_predict)
    precision = precision_score(y_test, test_predict)
    auc_score = roc_auc_score(y_test, test_prob)
    recall = recall_score(y_test, test_predict)

    with mlflow.start_run(run_name=f"train_final_{model_name}") as run:
        mlflow.set_tag("stage", "final_train")
        mlflow.set_tag("model_type", model_name)
        mlflow.log_params(best_params)
        mlflow.log_artifact(cfg["_path"], artifact_path="config")
        #mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_metric("F2_score", f2_score)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("auc_score", auc_score)
        mlflow.log_metric("Recall", recall)
        mlflow.log_param("Best_threshold", best_thresh)

        p = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        with open(p, "w") as f:
            json.dump({"decision_threshold": best_thresh}, f)
        mlflow.log_artifact(p, artifact_path="inference_config")

if __name__ == "__main__":
    main()














