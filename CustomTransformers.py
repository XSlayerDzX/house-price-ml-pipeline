# IMPORTING
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# 1. Binary features
class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['HasAlley'] = X['Alley'].notnull().astype(int)
        X['HasMiscFeature'] = X['MiscFeature'].notnull().astype(int)
        X['IsCentralAir'] = (X['CentralAir'] == 'Y').astype(int)
        X['PavedDrive'] = X['PavedDrive'].apply(lambda c: 'N' if c in ['N', 'P'] else c)
        X['IsPaved'] = (X['PavedDrive'] == 'Y').astype(int)
        X['Remodadded'] = (X['YearRemodAdd'] != X['YearBuilt']).astype(int)
        X['GarageYrBlt_missing'] = X['GarageYrBlt'].isna().astype(int)
        X['HasNonNormCond2'] = (X['Condition2'] != 'Norm').astype(int)
        return X.drop(columns=['Alley', 'MiscFeature', 'CentralAir', 'PavedDrive','Condition2',"Id"], errors='ignore')

# 2. Ordinal mapping
class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict): self.mapping_dict = mapping_dict
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col, mapping in self.mapping_dict.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0)
        return X

# 3. Grouping-based categorical mapping
class GroupMapper(BaseEstimator, TransformerMixin):
    def __init__(self, grouping_dict):
        self.grouping_dict = grouping_dict
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col, groupings in self.grouping_dict.items():
            if col == "MSSubClass":
                X[col] = X[col].astype(str)

            def map_value(val):
                for group_name, values in groupings.items():
                    if val in values:
                        return group_name
                return val

            X[col] = X[col].map(map_value)
        return X

# 4. Simple categorical imputation
class SimpleCatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_values=None): self.fill_values = fill_values
    def fit(self, X, y=None):
        self.fill_values_ = self.fill_values or {col: X[col].mode()[0] for col in X.columns}
        self.fill_values_['MSZoning'] = 'NoMSZoning'
        self.fill_values_['Fence'] = 'NoFence'
        self.fill_values_['GarageType'] = 'NoGarage'
        return self
    def transform(self, X):
        X = X.copy()
        for col, val in self.fill_values_.items():
            X[col] = X[col].fillna(val)
        return X

# 5. Numeric value imputer
class ValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy_dict): self.strategy_dict = strategy_dict
    def fit(self, X, y=None):
        self.fill_values_ = {}
        for col, strategy in self.strategy_dict.items():
            if strategy == 'mean':
                self.fill_values_[col] = X[col].mean()
            elif strategy == 'median':
                self.fill_values_[col] = X[col].median()
            else:
                self.fill_values_[col] = strategy
        return self
    def transform(self, X):
        X = X.copy()
        for col, val in self.fill_values_.items():
            X[col] = X[col].fillna(val)
        return X

# 6. Masonry handler
class MasonryHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['MasVnrType'] = X['MasVnrType'].fillna('NoMasonry')
        X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
        X['HasA_Masonry_Veneer'] = X['MasVnrType'].notnull().astype(int)
        return X

# 7. Column dropper
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns): self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.columns, errors='ignore')
