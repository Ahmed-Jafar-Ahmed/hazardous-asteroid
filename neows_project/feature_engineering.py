import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NeoWsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        # Drop id column if it exists
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
        # Convert 'close_approach_date' to datetime
        X['close_approach_date'] = pd.to_datetime(X['close_approach_date'])
        # Extract year, month, and day
        X['year'] = X['close_approach_date'].dt.year
        X['month'] = X['close_approach_date'].dt.month
        X['day'] = X['close_approach_date'].dt.day
        
        # Combine 'estimated_diameter_min' and 'estimated_diameter_max' into 'estimated_diameter_range'
        X['estimated_diameter_range'] = X['estimated_diameter_max'] - X['estimated_diameter_min']

        # Return only the relevant features for modeling
        X = X[['absolute_magnitude_h', 'estimated_diameter_range', 'year', 'month', 'day', 'relative_velocity', 'miss_distance_kilometers']]
        
        return X
    
