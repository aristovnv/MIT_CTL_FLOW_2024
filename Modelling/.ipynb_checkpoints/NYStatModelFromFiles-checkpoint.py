import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import datetime
from joblib import dump
from tensorflow.keras.models import load_model
from joblib import load
import pickle

import NYStatModel as ny

userName = 'naristov'
pathData = f"/home/gridsan/{userName}/Modelling/Data"
pathModel = f"/home/gridsan/{userName}/Modelling/Model"


for terminal in ny.terminals:
    model_autoarima_length = joblib.load(f'{pathModel}/model_autoarima_length_{terminal}.joblib')        
    model_autoarima_count = joblib.load(f'{pathModel}/model_autoarima_count_{terminal}.joblib')
    
    # Load Exponential Smoothing models
    es_models_length = [load(f'{pathModel}/model_es_length_{terminal}.joblib') for terminal in terminals]
    es_models_count = [load(f'{pathModel}/model_es_count_{terminal}.joblib') for terminal in terminals]

    # Load LSTM models
    lstm_models_length = [load_model(f'{pathModel}/model_lstm_length_{terminal}.h5') for terminal in terminals]
    lstm_models_count = [load_model(f'{pathModel}/model_lstm_count_{terminal}.h5') for terminal in terminals]

    df_merged, features_columns = dataPreparation()
    train_size = int(0.8 * len(df_merged))
    train, test = df_merged.iloc[:train_size], df_merged.iloc[train_size:]

    length_forecasts = pd.DataFrame(index=test.index)
    count_forecasts = pd.DataFrame(index=test.index)

    target_columns = [f'Length_NY_{terminal}', f'count_NY_{terminal}']

    X_train, y_train = train[features_columns], train[target_columns]
    X_test, y_test = test[features_columns], test[target_columns]

    length_forecast_autoarima = model_autoarima_length.predict(n_periods=len(X_test), exogenous=X_test)

    count_forecast_autoarima = model_autoarima_count.predict(n_periods=len(X_test), exogenous=X_test)

    result_es_length = model_es_length.fit()
    length_forecast_es = result_es_length.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

    result_es_count = model_es_count.fit()
    count_forecast_es = result_es_count.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
    
    scaler_length = StandardScaler()
    X_train_scaled_length = scaler_length.fit_transform(X_train)
    X_test_scaled_length = scaler_length.transform(X_test)
    
    X_train_length_reshaped = X_train_scaled_length.reshape((X_train_scaled_length.shape[0], 1, X_train_scaled_length.shape[1]))
    X_test_length_reshaped = X_test_scaled_length.reshape((X_test_scaled_length.shape[0], 1, X_test_scaled_length.shape[1])) 
    length_forecast_lstm = model_lstm_length.predict(X_test_length_reshaped)

    scaler_count = StandardScaler()
    X_train_scaled_count = scaler_count.fit_transform(X_train)
    X_test_scaled_count = scaler_count.transform(X_test)

    X_train_count_reshaped = X_train_scaled_count.reshape((X_train_scaled_count.shape[0], 1, X_train_scaled_count.shape[1]))
    X_test_count_reshaped = X_test_scaled_count.reshape((X_test_scaled_count.shape[0], 1, X_test_scaled_count.shape[1])) 

    count_forecast_lstm = model_lstm_count.predict(X_test_count_reshaped)
    
    
