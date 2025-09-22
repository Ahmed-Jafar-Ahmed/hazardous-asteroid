import pandas as pd

class NeoWsFeatureEngineer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        # Convert 'close_approach_date' to datetime
        X['close_approach_date'] = pd.to_datetime(X['close_approach_date'])
        # Extract year, month, and day
        X['year'] = X['close_approach_date'].dt.year
        X['month'] = X['close_approach_date'].dt.month
        X['day'] = X['close_approach_date'].dt.day
        
        # Combine 'estimated_diameter_min' and 'estimated_diameter_max' into 'estimated_diameter_range'
        X['estimated_diameter_range'] = X['estimated_diameter_max'] - X['estimated_diameter_min']

        # Return only the relevant features for modeling
        X = X[['absolute_magnitude_h', 'estimated_diameter_range', 'is_potentially_hazardous_asteroid', 'year', 'month', 'day', 'relative_velocity_km_per_s', 'miss_distance_kilometers']]
        
        return X
    
