from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from src.preprocess import NeoWsPreprocessor
from src.feature_engineering import NeoWsFeatureEngineer

def build_pipeline():
    # Constructs the pipeline with preprocessing, feature engineering, and scaling

    return Pipeline(steps=[
        ('preprocessor', NeoWsPreprocessor()),
        ('feature', NeoWsFeatureEngineer()),
        ('scaler', StandardScaler())
    ])
