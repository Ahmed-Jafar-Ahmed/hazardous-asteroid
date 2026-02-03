from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from .preprocess import NeoWsPreprocessor
from .feature_engineering import NeoWsFeatureEngineer
from sklearn.ensemble import RandomForestClassifier

def build_pipeline():
    # Constructs the pipeline with preprocessing, feature engineering, and scaling used to tune hyperparameters and choose model

    return Pipeline(steps=[
        ('preprocessor', NeoWsPreprocessor()),
        ('feature', NeoWsFeatureEngineer()),
        ('scaler', StandardScaler())
    ])

#Now we will define the final pipeline used for training te model

def build_final_pipe(model):

    return Pipeline(steps=[
        ('preprocessor', NeoWsPreprocessor()),
        ('feature', NeoWsFeatureEngineer()),
        ('scaler', StandardScaler()),
        ("model", model)
    ])