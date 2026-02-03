import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

class NeoWsPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        
        # Select useful columns that exist in the dataframe
        useful_data = ['id','absolute_magnitude_h', 'estimated_diameter_kilometers_estimated_diameter_min', 
                       'estimated_diameter_kilometers_estimated_diameter_max','close_approach_data']
        # Only select columns that actually exist
        cols_to_keep = [col for col in useful_data if col in df.columns]
        if cols_to_keep:
            df = df[cols_to_keep]
        df = df.rename(columns={
            'estimated_diameter_kilometers_estimated_diameter_min': 'estimated_diameter_min',
            'estimated_diameter_kilometers_estimated_diameter_max': 'estimated_diameter_max'
        })
        #Flatten nested JSON in 'close_approach_data'
        # Parse the string representation of list using eval (data is controlled/trusted)
        df['close_approach_data'] = df['close_approach_data'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df['close_approach_date'] = df['close_approach_data'].apply(lambda x: x[0]['close_approach_date'] if len(x) > 0 else np.nan)
        df['relative_velocity'] = df['close_approach_data'].apply(lambda x: float(x[0]['relative_velocity']['kilometers_per_hour']) if len(x) > 0 else np.nan)
        df['miss_distance_kilometers'] = df['close_approach_data'].apply(lambda x: float(x[0]['miss_distance']['kilometers']) if len(x) > 0 else np.nan)
        df = df.drop(columns=['close_approach_data'])
        
        # Convert data types
        df['close_approach_date'] = pd.to_datetime(df['close_approach_date'])        
        # Handle missing values
        for col in df.select_dtypes(include="number").columns:
            if col != "id":
                df[col] = df[col].fillna(df[col].median())
        median_DT = df['close_approach_date'].median()
        df['close_approach_date'] = df['close_approach_date'].fillna(median_DT)
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        return df
