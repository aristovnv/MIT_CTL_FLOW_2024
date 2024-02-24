import os
import pickle
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Replace with your model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt
import datetime
from joblib import dump
from tensorflow.keras.models import load_model
from joblib import load

model_desc = 'My Model'
#train_size_ratio = 0.8
train_size_ratio = 0.98

userName = 'naristov'
pathData = f"/home/gridsan/{userName}/Modelling/Data"
pathLogs = f"/home/gridsan/{userName}/Modelling/Logs"
pathModels = f"/home/gridsan/{userName}/Modelling/Models"
pathOutput = f"/home/gridsan/{userName}/Modelling/Output"


terminals = ['NY_APM', 'NY_LibertyB', 'NY_LibertyNY', 'NY_Maher', 'NY_Newark', 'NY_RedHook', 'Boston']


def calculate_mape_for_sets(df_true, df_pred):
    assert df_true.shape == df_pred.shape, "Input DataFrames must have the same shape"
    epsilon = 0.01
    df_true_nonzero = df_true.replace(0, epsilon)

    # Calculate MAPE for each column
    #mape_per_column = ((df_true_nonzero - df_pred).abs() / df_true_nonzero).mean() * 100
    mape_per_column = ((df_true + epsilon - df_pred).abs() / (df_true + epsilon)).mean() * 100

    #mape_per_set = ((df_true - df_pred).abs() / df_true).mean() * 100
    return mape_per_column

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Ensure both arrays have the same shape
    if y_true.shape != y_pred.shape:
        # Reshape y_pred to match the shape of y_true
        y_pred = y_pred.reshape(y_true.shape)

    # Avoid division by zero
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 



###
def dataPreparation(modelType, targetPrefix = "count" , daily = True):
    # modelType: 0 = All, 1: NY Only, 2: Boston Only, ....
    
    
    shipsData = pd.read_csv(f"{pathData}/shipsData.csv")
    anchorStatsBoston = pd.read_csv(f"{pathData}/ResultedAnchorStatsBoston.csv")
    anchorStatsBostonDaily = pd.read_csv(f"{pathData}/ResultedAnchorStatsBostondaily.csv")

    anchorStatsNY = pd.read_csv(f"{pathData}/ResultedAnchorStatsNY.csv")
    anchorStatsNYDaily = pd.read_csv(f"{pathData}/ResultedAnchorStatsNYdaily.csv")
    berthStatsBoston = pd.read_csv(f"{pathData}/ResultedBerthStatsBoston.csv")
    berthStatsBostonDaily = pd.read_csv(f"{pathData}/ResultedBerthStatsBostondaily.csv")
    berthStatsNY = pd.read_csv(f"{pathData}/ResultedBerthStatsNY.csv")
    berthStatsNYDaily = pd.read_csv(f"{pathData}/ResultedBerthStatsNYdaily.csv")
    harborStatsBoston = pd.read_csv(f"{pathData}/ResultedHarborStatsBoston.csv")
    harborStatsBostonDaily = pd.read_csv(f"{pathData}/ResultedHarborStatsBostondaily.csv")    
    harborStatsNY = pd.read_csv(f"{pathData}/ResultedHarborStatsNY.csv")
    harborStatsNYDaily = pd.read_csv(f"{pathData}/ResultedHarborStatsNYdaily.csv")    

    BerthMergeAnchor = pd.read_csv(f"{pathData}/BerthMergeAnchor.csv")
    VesselnOnBerth = pd.read_csv(f"{pathData}/ResultedVesselOnBerth.csv")
    BerthMergeAnchor = pd.merge(BerthMergeAnchor, shipsData,
                  how='left', left_on='MMSI', right_on='MMSI')

    # Create columns dynamically based on operators
    operator_columns = pd.get_dummies(BerthMergeAnchor['Operator'], prefix='operator', dummy_na=False)
    BerthMergeAnchor = pd.concat([BerthMergeAnchor, operator_columns], axis=1)
    BerthMergeAnchorBoston = BerthMergeAnchor[BerthMergeAnchor['groupedPort'] == 'Boston']
    BerthMergeAnchorNY = BerthMergeAnchor[BerthMergeAnchor['groupedPort'] == 'NY']
    
    berthStatsAll = pd.concat([berthStatsNY, berthStatsBoston])
    berthStatsAllDaily = pd.concat([berthStatsNYDaily, berthStatsBostonDaily])
    
    
    df_pivoted = berthStatsAll.pivot_table(index='Hour', columns='nearestPort', values=['count', 'Length'], aggfunc='sum', fill_value=0)

    df_pivotedDaily = berthStatsAllDaily.pivot_table(index='timeDiscrepancy', columns='nearestPort', values=['count', 'Length'], aggfunc='sum', fill_value=0)
    
    
    # Flatten the multi-level columns
    df_pivoted.columns = [f'{agg}_{terminal}' for agg, terminal in df_pivoted.columns]
    df_pivotedDaily.columns = [f'{agg}_{terminal}' for agg, terminal in df_pivotedDaily.columns]
    # Reset the index to make 'hour' a regular column
    df_pivoted.reset_index(inplace=True)
    df_pivotedDaily.reset_index(inplace=True)
    # Merge the dataframes on the common time-related column (hour)
    df_merged = pd.merge(df_pivoted, anchorStatsNY, on='Hour', how='left', suffixes=('', '_anchor_NY'))
    df_merged = pd.merge(df_merged, harborStatsNY, on='Hour', how='left', suffixes=('', '_harbor_NY'))
    df_merged = pd.merge(df_merged, anchorStatsBoston, on='Hour', how='left', suffixes=('', '_anchor_Boston'))
    df_merged = pd.merge(df_merged, harborStatsBoston, on='Hour', how='left', suffixes=('', '_harbor_Boston'))

    # Merge the dataframes on the common time-related column (Dayly)
    df_mergedDaily = pd.merge(df_pivotedDaily, anchorStatsNYDaily, on='timeDiscrepancy', how='left', suffixes=('', '_anchor_NY'))
    df_mergedDaily = pd.merge(df_mergedDaily, harborStatsNYDaily, on='timeDiscrepancy', how='left', suffixes=('', '_harbor_NY'))
    df_mergedDaily = pd.merge(df_mergedDaily, anchorStatsBostonDaily, on='timeDiscrepancy', how='left', suffixes=('', '_anchor_Boston'))
    df_mergedDaily = pd.merge(df_mergedDaily, harborStatsBostonDaily, on='timeDiscrepancy', how='left', suffixes=('', '_harbor_Boston'))
    
    
    # Handle missing values if any
    df_merged.fillna(0, inplace=True)
    df_merged.fillna(0, inplace=True)
    df_mergedDaily.fillna(0, inplace=True)
    df_mergedDaily.fillna(0, inplace=True)
    # Feature Engineering
    df_merged['Hour']= pd.to_datetime(df_merged['Hour'])
    df_merged['Hour']= pd.to_datetime(df_merged['Hour'])
    df_mergedDaily['timeDiscrepancy']= pd.to_datetime(df_mergedDaily['timeDiscrepancy'])
    df_mergedDaily['timeDiscrepancy']= pd.to_datetime(df_mergedDaily['timeDiscrepancy'])
    # Extract hour and create a new column
    df_merged['hour_of_day'] = df_merged['Hour'].dt.hour
    df_merged['day_of_week'] = df_merged['Hour'].dt.day_of_week
    df_merged['month'] = df_merged['Hour'].dt.month
    df_merged['Hour']= df_merged['Hour'].astype(int)

    # Extract hour and create a new column
    df_mergedDaily['day_of_week'] = df_mergedDaily['timeDiscrepancy'].dt.day_of_week
    df_mergedDaily['month'] = df_mergedDaily['timeDiscrepancy'].dt.month
    
    
    
    
    df_merged.drop(['Unnamed: 0.1','index','Unnamed: 0','Unnamed: 0.1_harbor_NY','index_harbor_NY','Unnamed: 0_harbor_NY','Unnamed: 0.1_anchor_Boston','index_anchor_Boston','Unnamed: 0_anchor_Boston','Unnamed: 0.1_harbor_Boston','index_harbor_Boston','Unnamed: 0_harbor_Boston'
], axis = 1, inplace=True)
    df_mergedDaily.drop(['Unnamed: 0.1','index','Unnamed: 0','Unnamed: 0.1_harbor_NY','index_harbor_NY','Unnamed: 0_harbor_NY','Unnamed: 0.1_anchor_Boston','index_anchor_Boston','Unnamed: 0_anchor_Boston','Unnamed: 0.1_harbor_Boston','index_harbor_Boston','Unnamed: 0_harbor_Boston'
], axis = 1, inplace=True)

    
    # add to drop columns for any new port  
    df_merged.rename(columns={'count': 'count_anchor_NY', 'Length': 'Length_harbor_NY', 'Width': 'Width_harbor_NY'}, inplace=True)
    df_mergedDaily.rename(columns={'count': 'count_anchor_NY', 'Length': 'Length_harbor_NY', 'Width': 'Width_harbor_NY'}, inplace=True)

    
    features_columns = [ 'day_of_week', 'month', 'hour_of_day']
    features_columnsDaily = [ 'day_of_week', 'month']
    selected_columns = []
    #target_columns = ['Length_NY_APM', 'Length_NY_LibertyB', 'Length_NY_LibertyNY', 'Length_NY_Maher',
    #                  'Length_NY_Newark', 'Length_NY_RedHook', 'count_NY_APM', 'count_NY_LibertyB',
    #                  'count_NY_LibertyNY', 'count_NY_Maher', 'count_NY_Newark', 'count_NY_RedHook']  
    df_merged.sort_values(by='Hour', inplace=True)    
    df_mergedDaily.sort_values(by='timeDiscrepancy', inplace=True)    
    
    for terminal in terminals:
        count_column = f'count_{terminal}'
        diff_column = f'diff_to_prev_{terminal}'
        # Calculate the difference and assign it to the new column
        df_merged[diff_column] = df_merged[count_column].diff().fillna(0)
        df_mergedDaily[diff_column] = df_mergedDaily[count_column].diff().fillna(0)

    #adding the rest of ports for fun
    allBerthStatsDaily = pd.read_csv(f"{pathData}/ResultedVesselOnBerth.csv")

    allBerthStatsDaily['timeDiscrepancy'] = allBerthStatsDaily.apply(lambda row: pd.date_range(row['start_time'], row['end_time'], freq='D'), axis=1)
    allBerthStatsDaily = allBerthStatsDaily.explode('timeDiscrepancy')
    allBerthStatsDaily['timeDiscrepancy'] = pd.to_datetime(allBerthStatsDaily['timeDiscrepancy']).dt.normalize()
    
    # Create a pivot table to count the number of ships in each port for each day
    pivot_allBerthStatsDaily = pd.pivot_table(allBerthStatsDaily, values='MMSI', index='timeDiscrepancy', columns='nearestPort', aggfunc='count', fill_value=0)

    # Reset index to make 'Day' a regular column
    pivot_allBerthStatsDaily.reset_index(inplace=True)
    # Rename the columns to match your desired format
    pivot_allBerthStatsDaily.columns.name = None  # Remove the 'port' name from the columns
    pivot_allBerthStatsDaily.columns = ['timeDiscrepancy'] + [f'port{port}' for port in pivot_allBerthStatsDaily.columns[1:]]
    port_columns = pivot_allBerthStatsDaily.filter(regex='^port')
    #features_columnsDaily += port_columns.columns.tolist()
# Assuming df_mergedDaily and allBerthStatsDaily are your DataFrames
# and you want to join them on the 'timeDiscrepancy' column

    df_mergedDaily = pd.merge(df_mergedDaily, pivot_allBerthStatsDaily, on='timeDiscrepancy', how='left')
    df_mergedDaily['timeDiscrepancy']= df_mergedDaily['timeDiscrepancy'].astype(int)


    # Identify columns to shift
    columns_to_shift = [col for col in df_mergedDaily.columns if col.startswith('count_') or col.startswith('port')]

    # Specify the number of steps to shift
    num_steps = 10

    # List to store the names of newly added columns
    new_columns_list = []

    # Iterate through each column and add shifted columns
    for column in columns_to_shift:
        for step in range(1, num_steps + 1):
            new_column_name = f'{column}_p{step}'
            df_mergedDaily[new_column_name] = df_mergedDaily[column].shift(-step)
            new_columns_list.append(new_column_name)

    features_columnsDaily += new_columns_list
    
    if modelType in [0, 1]:        
        features_columns += [ 'count_harbor_NY', 'Length_harbor_NY', 'Width_harbor_NY']
        features_columnsDaily += [ 'count_harbor_NY', 'Length_harbor_NY', 'Width_harbor_NY']
        selected_columns += [f'{targetPrefix}_{terminal}' for terminal in terminals if terminal.startswith('NY')]
    if modelType in [0, 2]:
        features_columns += [ 'count_harbor_Boston', 'Length_harbor_Boston', 'Width_harbor_Boston']
        features_columnsDaily += [ 'count_harbor_Boston', 'Length_harbor_Boston', 'Width_harbor_Boston']
        selected_columns += [f'{targetPrefix}_{terminal}' for terminal in terminals if terminal.startswith('Boston')]
        
    # add more types 
    if daily:
        df_mergedDaily.fillna(0, inplace=True)
        return df_mergedDaily, features_columnsDaily, selected_columns
    return df_merged, features_columns, selected_columns



#####
def runXGBoost(X_train, X_test, y_train, y_test, terminal, targetPrefix = "count"):
    
    dtrain_count = xgb.DMatrix(X_train, label=y_train[f'{targetPrefix}_{terminal}'])
    dtest_count = xgb.DMatrix(X_test, label=y_test[f'{targetPrefix}_{terminal}'])
    
    params = {
    'objective': 'reg:squarederror',  # for regression tasks
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100}
    
    print(f"XGBoost - Forecasting for {targetPrefix}_{terminal}")
    model_XGBoost_count = xgb.train(params, dtrain_count, num_boost_round=10)    
    count_forecast_XGBoost = model_XGBoost_count.predict(dtest_count)
    model_XGBoost_count.save_model(f'{pathModels}/model_XGBoost_{targetPrefix}_{terminal}.json')
    return count_forecast_XGBoost

#####
def runArima(X_train, X_test, y_train, y_test, terminal, targetPrefix = "count"):
    
    print(f"AutoARIMA - Forecasting for {targetPrefix}_{terminal}")
    model_autoarima_count = auto_arima(y_train[f'{targetPrefix}_{terminal}'], exogenous=X_train, suppress_warnings=True, trace=True)
    count_forecast_autoarima = model_autoarima_count.predict(n_periods=len(X_test), exogenous=X_test)
    dump(model_autoarima_count, f'{pathModels}/model_autoarima_{targetPrefix}_{terminal}.joblib')
    return count_forecast_autoarima

#### 
def runExpSmooth(X_train, X_test, y_train, y_test, terminal, targetPrefix = "count"):
    
    print(f"Exponential Smoothing - Forecasting for {targetPrefix}_{terminal}")
    model_es_count = ExponentialSmoothing(y_train[f'{targetPrefix}_{terminal}'], trend='add', seasonal='add', seasonal_periods=24)
    result_es_count = model_es_count.fit()
    count_forecast_es = result_es_count.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
    dump(model_es_count, f'{pathModels}/model_es_{targetPrefix}_{terminal}.joblib')
    return count_forecast_es

####
def runLSTM(X_train, X_test, y_train, y_test, terminal, targetPrefix = "count"):

    print(f"LSTM - Forecasting for {targetPrefix}_{terminal}")
        # Assuming you have standardized features and targets
    scaler_count = StandardScaler()
    X_train_scaled_count = scaler_count.fit_transform(X_train)
    X_test_scaled_count = scaler_count.transform(X_test)

    X_train_count_reshaped = X_train_scaled_count.reshape((X_train_scaled_count.shape[0], 1, X_train_scaled_count.shape[1]))
    X_test_count_reshaped = X_test_scaled_count.reshape((X_test_scaled_count.shape[0], 1, X_test_scaled_count.shape[1])) 
    
    model_lstm_count = Sequential()
    model_lstm_count.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled_count.shape[1])))
    model_lstm_count.add(Dense(1))
    model_lstm_count.compile(optimizer='adam', loss='mse')

    model_lstm_count.fit(X_train_count_reshaped, y_train[f'{targetPrefix}_{terminal}'], epochs=50, verbose=1)

    model_lstm_count.save(f'{pathModels}/model_lstm_{targetPrefix}_{terminal}.h5')

    count_forecast_lstm = model_lstm_count.predict(X_test_count_reshaped)
        
    return count_forecast_lstm

###
def startModelPerTerminal(model, modelType, terminal, targetPrefix = "count", roundQty = 0):
    df_merged, features_columns, selected_columns = dataPreparation(modelType, targetPrefix)
    train_size = int(train_size_ratio * len(df_merged))
    train, test = df_merged.iloc[:train_size], df_merged.iloc[train_size:]

    count_forecasts = pd.DataFrame(index=test.index)
    target_columns = [f'{targetPrefix}_{terminal}']

    X_train, y_train = train[features_columns], train[target_columns]
    X_test, y_test = test[features_columns], test[target_columns]
    prediction = y_test.copy()
        # AutoARIMA
    if model == 'arima':
        prediction = runArima(X_train, X_test, y_train, y_test, terminal, targetPrefix)   
        
        # Exponential Smoothing
    if model == 'expSmooth':
        prediction = runExpSmooth(X_train, X_test, y_train, y_test, terminal, targetPrefix)
        
        # LSTM (Long Short-Term Memory)
    if model == 'LSTM':
        prediction = runLSTM(X_train, X_test, y_train, y_test, terminal, targetPrefix)
        index_for_forecast = y_test.index        
        prediction = pd.DataFrame(prediction, columns = [f'{targetPrefix}_{terminal}'], index=index_for_forecast)    

            
    if model == 'XGBoost':
        prediction = runXGBoost(X_train, X_test, y_train, y_test, terminal, targetPrefix)
        index_for_forecast = y_test.index
        prediction = pd.DataFrame(prediction, columns = [f'{targetPrefix}_{terminal}'], index=index_for_forecast)   
        
    if model =='PoissonRegressor':
        prediction = runPoissonRegressor(X_train, X_test, y_train, y_test, terminal, targetPrefix)

        
    saveError(f'{terminal}_{model}_{targetPrefix}', f'{terminal}', f'{model}', f'{targetPrefix}', f'one terminal', y_test, prediction)
    if roundQty == 1:
        saveError(f'{terminal}_{model}_{targetPrefix}', f'{terminal}', f'{model}', f'{targetPrefix}', f'one terminal', y_test, prediction, roundQty)

    saveResults(f'{terminal}_{model}_{targetPrefix}', y_test, prediction)    
    return 0
                  
def runXGBoostAll(X_train, X_test, y_train, y_test, selected_columns):
    print(f"XGBoost - Forecasting ")
    dtrain_length = xgb.DMatrix(X_train, label=y_train[selected_columns])
    dtest_length = xgb.DMatrix(X_test, label=y_test[selected_columns])
    
    params = {
    'objective': 'reg:squarederror',  # for regression tasks
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100}    
    
    model_XGBoost_length = xgb.train(params, dtrain_length, num_boost_round=10)
    length_forecast_XGBoost = model_XGBoost_length.predict(dtest_length)
    
    return length_forecast_XGBoost 


def runLSTMAll(X_train, X_test, y_train, y_test, selected_columns):
    print(f"LSTM - Forecasting ")
    scaler_length = StandardScaler()
    X_train_scaled_length = scaler_length.fit_transform(X_train)
    X_test_scaled_length = scaler_length.transform(X_test)
    
    X_train_length_reshaped = X_train_scaled_length.reshape((X_train_scaled_length.shape[0], 1, X_train_scaled_length.shape[1]))
    X_test_length_reshaped = X_test_scaled_length.reshape((X_test_scaled_length.shape[0], 1, X_test_scaled_length.shape[1])) 
        
    model_lstm_length = Sequential()
    model_lstm_length.add(LSTM(50, activation='relu', input_shape= (1, X_train_scaled_length.shape[1])))
    model_lstm_length.add(Dense(units=y_train.shape[1],activation='linear'))
    model_lstm_length.compile(optimizer='adam', loss='mean_squared_error')    
    model_lstm_length.fit(X_train_length_reshaped, y_train[selected_columns], epochs=50, verbose=1)

    length_forecast_lstm = model_lstm_length.predict(X_test_length_reshaped)

    return length_forecast_lstm

###
def runLSTNML(X_train, X_test, y_train, y_test, selected_columns):
    # Feature normalization
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.values)
    X_test_scaled = scaler_X.transform(X_test.values)

    # Target normalization
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    # Reshape input for LSTM
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Define the model
    model = Sequential()

    # Input layer
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(1, X_train_scaled.shape[1])))
    model.add(BatchNormalization())

    # Hidden layers
    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(units=50, activation='relu'))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(units=y_train.shape[1], activation='linear'))  # Assuming y_train.shape[1] is the number of berths

    # Compile the model
    model.compile(optimizer='adam', loss='poisson')

    # Print the model summary
    model.summary()

    # Train the model
    model.fit(X_train_reshaped, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2)

    # Predict using the trained model
    predictions_scaled = model.predict(X_test_reshaped)

    # Inverse transform predictions to the original scale
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Inverse transform the true labels for evaluation
    #y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    print("Shape of y_test:", y_test.shape)
    print("Shape of predictions:", predictions.shape)
    return predictions

###
def runProphet():
    return 0
###
def runVAR(X_train, X_test, y_train, y_test, selected_columns):# (Vector Autoregression):

    model = VAR(y_train, exog=X_train)
    results = model.fit()

    forecast = results.forecast(y_train.values, steps=len(X_test), exog_future=X_test)
    
    return forecast
##
def MLM():
    #Consider exploring other machine learning models like Random Forests, Gradient Boosting Machines, or even neural networks designed for regression tasks.
    return 0
###
def runDeepAR():
    return 0

def crossValidation(model, actual):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predicted_cv = cross_val_predict(model, pd.DataFrame(np.random.rand(100, 10)), actual, cv=cv)


def saveError(fileName, terminal, model, varName, description, actual, prediction, roundQty = 0):
    output = prediction.copy()
    if roundQty == 1:
        output = output.round()
    err = {
        "roundQty" : [roundQty],
        "terminal": [f"{terminal}"],
        "model": [f"{model}"],
        "varName": [f"{varName}"],
        "description": [f"{description}"],
        "mse": [mean_squared_error(actual, output)],
        "mae": [mean_absolute_error(actual, output)], 
        "RSquared": [r2_score(actual, output)],
        "mape": [mean_absolute_percentage_error(actual, output)] 
    }

    pd.DataFrame(err).to_csv(f"{pathLogs}/{fileName}{roundQty}.csv")

def saveResults(fileName, actual, prediction):
    actual.to_csv(f'{pathOutput}/{fileName}_actual.csv', index=True)
    prediction.to_csv(f'{pathOutput}/{fileName}_prediction.csv', index=True)
    
###
def startModelForGroup(model, modelType, targetPrefix = "count", roundQty = 0):
    df_merged, features_columns, selected_columns = dataPreparation(modelType, targetPrefix)
    train_size = int(train_size_ratio * len(df_merged))
    train, test = df_merged.iloc[:train_size], df_merged.iloc[train_size:]

    count_forecasts = pd.DataFrame(index=test.index)
    #target_columns = [f'{targetPrefix}_{terminal}']

    X_train, y_train = train[features_columns], train[selected_columns]
    X_test, y_test = test[features_columns], test[selected_columns]
    prediction = y_test.copy()
    
    if model == 'LSTM':
        prediction = runLSTMAll(X_train, X_test, y_train, y_test, selected_columns)
        
    if model == 'XGBoost':
        prediction = runXGBoostAll(X_train, X_test, y_train, y_test, selected_columns)

    if model == 'VAR':
        prediction = runVAR(X_train, X_test, y_train, y_test, selected_columns)
    
    if model == '1DCNN':
        prediction = run1DCNN(X_train, X_test, y_train, y_test, selected_columns)
        
    if model == 'RandomForestClassifier':
        prediction = runRandomForestClassifier(X_train, X_test, y_train, y_test, selected_columns)
    if model == 'RandomForestRegressor':
        prediction = runRandomForestRegressor(X_train, X_test, y_train, y_test, selected_columns)
    if model == 'LSTNML':
        prediction = runLSTNML(X_train, X_test, y_train, y_test, selected_columns)
    
    if model == 'SVR':
        prediction = runSVR(X_train, X_test, y_train, y_test, selected_columns)
        
    #new
    saveError(f'{modelType}_{model}_{targetPrefix}', f'{modelType}', f'{model}', f'{targetPrefix}', f'several terminals', y_test, prediction)
    if roundQty == 1:
        saveError(f'{modelType}_{model}_{targetPrefix}', f'{modelType}', f'{model}', f'{targetPrefix}', f'several terminals', y_test, prediction, roundQty)


    index_for_forecast = y_test.index
    forecasted_columns = {f'{terminal}': prediction[:, i] for i, terminal in enumerate(selected_columns)}
    forecasted_df = pd.DataFrame(forecasted_columns, index=index_for_forecast)    
    saveResults(f'{modelType}_{model}_{targetPrefix}', y_test, forecasted_df)
    return 0


def mergeLog():
    # List to store DataFrames from each CSV file
    dfs = []

    # Iterate through each file in the folder
    for filename in os.listdir(pathLogs):
        if filename.endswith('.csv'):
            file_path = os.path.join(pathLogs, filename)
        
        # Read the CSV file into a DataFrame and append it to the list
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate the list of DataFrames into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    result_df.to_csv(f'{pathLogs}/AllModels.csv', index=False)

###
def run1DCNN(X_train, X_test, y_train, y_test, selected_columns):

    # Normalize the data
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Create separate scalers for y_train and y_test
    scaler_y_train = MinMaxScaler()
    y_train_scaled = scaler_y_train.fit_transform(y_train)

    scaler_y_test = MinMaxScaler()
    y_test_scaled = scaler_y_test.fit_transform(y_test)

    # Build a feedforward neural network
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(selected_columns), activation='linear'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mse')  # Mean Squared Error is used for regression tasks

    # Train the model
    model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=train_size_ratio)

    # Predict on the test set
    y_pred_scaled = model.predict(X_test_scaled)

    # Inverse transform the predictions to the original scale
    y_pred_original = scaler_y_test.inverse_transform(y_pred_scaled)

    return y_pred_original

def runPoissonRegressor(X_train, X_test, y_train, y_test, terminal, targetPrefix):
    
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    # Extract values from DataFrame, reshape, and then ravel
    y_train_scaled = scaler_y.fit_transform(y_train[f'{targetPrefix}_{terminal}'].values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test[f'{targetPrefix}_{terminal}'].values.reshape(-1, 1)).ravel()

    # Create and fit the Poisson Regressor model
    poisson_model = PoissonRegressor()
    poisson_model.fit(X_train_scaled, y_train_scaled)

    # Predictions on the test set
    y_pred_scaled = poisson_model.predict(X_test_scaled)

    # Inverse transform the predictions to the original scale
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    return y_pred_original
    
def runRandomForestClassifier(X_train, X_test, y_train, y_test, selected_columns):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate Hamming loss
    hamming_distance = np.sum(y_test != y_pred) / float(y_test.size)

    # Calculate Hamming loss
    hamming_loss_value = hamming_distance / y_test.shape[0]
    print(f'Hamming Loss: {hamming_loss_value}')
    
    # Classification report
#    print('Classification Report:')
#    print(classification_report(y_test, y_pred))
    return y_pred

def runRandomForestRegressor(X_train, X_test, y_train, y_test, selected_columns):
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    regr_multirf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = regr_multirf.predict(X_test)
    return y_pred

def runSVR(X_train, X_test, y_train, y_test, selected_columns):
    svm_regressor = SVR(kernel='linear')  # You can choose different kernels (linear, rbf, poly, etc.)
    svm_regressor.fit(X_train, y_train)
    y_pred_regression = svm_regressor.predict(X_test)    
    return y_pred_regression


def checkNoise(df):    
    for column in df.columns:
        time_series = df[column]
        result = adfuller(time_series)
    
        # Display the results
        print(f'Column: {column}')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value}')
        print('\n')

def callCheckNoise(model, modelType, targetPrefix = "count", roundQty = 0):    
    df_merged, features_columns, selected_columns = dataPreparation(modelType, targetPrefix)
    df_merged.to_csv(f"{pathData}/test.csv")
    print(features_columns)
    #checkNoise(df_merged)
        
#startModelForGroup('LSTNML', 0, targetPrefix = "count", roundQty = 1)        
#startModelForGroup('SVR', 0, targetPrefix = "count", roundQty = 1)
#startModelForGroup('SVR', 0, targetPrefix = "diff_to_prev", roundQty = 1)
callCheckNoise('SVR', 0, targetPrefix = "diff_to_prev", roundQty = 1)
