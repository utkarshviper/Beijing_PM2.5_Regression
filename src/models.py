"""
models.py - Model Definitions, Chronological Splitting, Feature Scaling & Training

Step 4: Identify X and y
Step 5: Chronological Anti-Leakage Split (Train: 2013-03-01 to 2016-02-29, Test: 2016-03-01 to 2017-02-28)
Step 6: Feature Scaling & Encoding (StandardScaler fit STRICTLY on X_train)
Step 7: Baseline & Comparison Models:
        - Naive Baseline (DummyRegressor)
        - Parametric Baseline (Linear / Ridge Regression)
        - Comparison 1: Decision Tree Regressor
        - Comparison 2: K-Nearest Neighbors Regressor (KNN)
        - Comparison 3: Support Vector Regressor (SVR)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Continuous features requiring StandardScaler in Step 6
NUMERIC_FEATURES = ['DEWP', 'TEMP', 'PRES', 'WSPM', 'RAIN', 'temp_dew_spread']

# Categorical features requiring One-Hot Encoding
CATEGORICAL_FEATURES = ['season', 'time_of_day', 'wd_sector', 'station']


def prepare_feature_matrices(df: pd.DataFrame, drop_first: bool = True):
    """
    Step 4: Separates feature matrix X and target y.
    Step 5: Splits chronologically:
            Train: 2013-03-01 to 2016-02-29 (First 3 seasonal years)
            Test:  2016-03-01 to 2017-02-28 (Final held-out seasonal year)
    Step 6: Encodes dummies and applies StandardScaler fit strictly on Train.
    """
    # Exclude companion pollutants to prevent leakage/collinearity
    excluded_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'year', 'month', 'day', 'hour', 'wd', 'zone_type', 'is_rural']
    
    # Feature columns include: NUMERIC_FEATURES + is_urban + is_weekend + CATEGORICAL_FEATURES
    feature_cols = NUMERIC_FEATURES + ['is_urban', 'is_weekend'] + CATEGORICAL_FEATURES
    
    # One-hot encode categoricals across the full domain structure
    df_encoded = pd.get_dummies(df[feature_cols + ['datetime', 'PM2.5']], columns=CATEGORICAL_FEATURES, drop_first=drop_first)
    
    # Chronological Split
    split_date = pd.Timestamp("2016-03-01 00:00:00")
    train_mask = df_encoded['datetime'] < split_date
    test_mask = df_encoded['datetime'] >= split_date
    
    train_df = df_encoded[train_mask].drop(columns=['datetime']).reset_index(drop=True)
    test_df = df_encoded[test_mask].drop(columns=['datetime']).reset_index(drop=True)
    
    X_train_raw = train_df.drop(columns=['PM2.5'])
    y_train = train_df['PM2.5'].values
    
    X_test_raw = test_df.drop(columns=['PM2.5'])
    y_test = test_df['PM2.5'].values
    
    # Step 6: Feature Scaling - fit StandardScaler strictly on training continuous variables
    scaler = StandardScaler()
    
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()
    
    X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X_train_raw[NUMERIC_FEATURES])
    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test_raw[NUMERIC_FEATURES])
    
    return {
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X_train_raw.columns)
    }


def train_baseline_models(data_dict: dict):
    """
    Trains naive baseline (DummyRegressor) and parametric baseline (Linear & Ridge Regression).
    """
    # 1. Naive Baseline
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    # 2. Linear Regression (OLS)
    lr = LinearRegression()
    lr.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    # 3. Ridge Regression (alpha=1.0)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    return {
        'dummy': dummy,
        'linear_regression': lr,
        'ridge': ridge
    }


def train_decision_tree(data_dict: dict, max_depth: int = 8):
    """
    Trains DecisionTreeRegressor on scaled (or unscaled) features with specified max_depth.
    """
    dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=20, random_state=42)
    dt.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    return dt


def train_knn_regressor(data_dict: dict, n_neighbors: int = 15, weights: str = 'distance', metric: str = 'euclidean'):
    """
    Trains KNeighborsRegressor with distance weighting.
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1)
    knn.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    return knn


def train_svr_regressor(data_dict: dict, sample_size: int = 15000, C: float = 10.0, epsilon: float = 0.1):
    """
    Trains SVR with RBF kernel on a representative chronological/stratified subsample
    due to O(N^2) quadratic solver scaling.
    """
    np.random.seed(42)
    n_train = len(data_dict['y_train'])
    if n_train > sample_size:
        idx = np.random.choice(n_train, size=sample_size, replace=False)
        X_sub = data_dict['X_train_scaled'].iloc[idx]
        y_sub = data_dict['y_train'][idx]
    else:
        X_sub = data_dict['X_train_scaled']
        y_sub = data_dict['y_train']
        
    svr = SVR(kernel='rbf', C=C, epsilon=epsilon)
    svr.fit(X_sub, y_sub)
    return svr
