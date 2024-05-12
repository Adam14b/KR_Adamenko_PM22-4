import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, data_sep=',', col_name_sep='_'):
        self.data_sep     = data_sep
        self.col_name_sep = col_name_sep
        self.columns      = []
        self.dummy_cols   = []
        self.dummy_prefix = []
        
    def fit(self, X, y  = None): 
        object_cols       = X.select_dtypes(include="O").columns
        self.dummy_cols   = [col for col in object_cols if X[col].str.contains(self.data_sep, regex=True).any()]
        self.dummy_prefix = [col[:2] if self.col_name_sep not in col else ''.join(map(lambda x: x[0], col.split(self.col_name_sep))) for col in self.dummy_cols]
        
        if len(self.dummy_cols):
            dummy_frames_df = pd.concat([X[col].str.get_dummies(sep=self.data_sep).add_prefix(pre+self.col_name_sep) for pre, col in zip(self.dummy_prefix, self.dummy_cols)], axis=1)
            self.columns    = X.join(dummy_frames_df).drop(columns=self.dummy_cols).columns.tolist()
        else:
            self.columns = X.columns.tolist()
        return self
    
    def transform(self, X, y = None):
        if len(self.dummy_prefix) or len(self.dummy_cols):
            dummy_frames_df = pd.concat([X[col].str.get_dummies(sep=self.data_sep).add_prefix(pre+self.col_name_sep) for pre, col in zip(self.dummy_prefix, self.dummy_cols)], axis=1)
            X = X.join(dummy_frames_df).reindex(columns=self.columns, fill_value=0)
        return X
          
    def get_feature_names_out(self, input_features=None):
        return self.columns
