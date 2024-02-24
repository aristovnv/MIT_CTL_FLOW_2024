import pickle
import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


import NYStatModel as ny


def get_initial_data():
    df_merged, features_columns = ny.dataPreparation()
    train_size = int(0.8 * len(df_merged))
    train, test = df_merged.iloc[:train_size], df_merged.iloc[train_size:]
    return df_merged
#for now assume we have this as forecast 

df_stats = get_initial_data()
df = ny.BerthMergeAnchorNY.copy()
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['start_time_anchor'] = pd.to_datetime(df['start_time_anchor'])
df['end_time_anchor'] = pd.to_datetime(df['end_time_anchor'])

df['isAnchor'] = np.where(df['start_time_anchor'].isnull(), 0 , 1)

# Extract hour from 'start_time' if 'start_time_anchor' is null, else use 'start_time_anchor'
df['Hour'] = np.where(df['start_time_anchor'].isnull(), df['start_time'], df['start_time_anchor'])
df['Hour'] = pd.to_datetime(df['Hour'])
df['Hour'] = pd.to_datetime(df['Hour'].dt.strftime('%Y-%m-%d %H:00:00'))
# Subtract 2 hours from 'Hour'
df['Hour'] = df['Hour'] - pd.to_timedelta(2, unit='H') 
# Convert 'Hour' to datetime format
df['Hour'] = df['Hour'].astype(int)

merged_df = pd.merge(df, df_stats, on='Hour', how='left')

# Create a label encoder
label_encoder = LabelEncoder()

# Fit and transform the terminal variable
merged_df['terminal_encoded'] = label_encoder.fit_transform(merged_df['nearestPort'])

merged_df.drop(['Unnamed: 0.2_x','index','Unnamed: 0.1_x','Unnamed: 0_x','groupedPort', 'Unnamed: 0.1_anchor', 'Unnamed: 0_anchor','VesselName_anchor','nearestPort_anchor', 'groupedPort_anchor','Unnamed: 0_anchor.1', 'index_anchor','nearestPort_anchor.1','group','start_time_anchor.1','end_time_anchor.1','TimeSpent_anchor.1','Unnamed: 0.8','Unnamed: 0.7','Unnamed: 0.6','Unnamed: 0.5','Unnamed: 0.4','Unnamed: 0.3','Unnamed: 0.2_y','Unnamed: 0.1_y','Unnamed: 0_y','Call Sign','Cargo Type','IMO Number','Name','Operator','Registered Owner','Ship Type','Vessel Type'], axis = 1, inplace = True) 

merged_df['start_time_anchor_int'] = merged_df['start_time_anchor'].astype(int) 
merged_df['end_time_anchor_int'] = merged_df['end_time_anchor'].astype(int)

# Assuming you have a target variable 'terminal'
X = merged_df.drop(['isAnchor','start_time', 'end_time', 'VesselName','nearestPort', 'terminal_encoded', 'start_time_anchor', 'end_time_anchor', 'start_time_anchor_int', 'end_time_anchor_int'], axis = 1)
X = X.fillna(-5)
y = merged_df[['terminal_encoded', 'isAnchor']]
#y = df['nearestPort']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MultiOutputRegressor with RandomForestRegressor
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
regr_multirf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regr_multirf.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'MAPE: {ny.mean_absolute_percentage_error(y_test, y_pred)}')

# Now, you can use the trained model to make predictions for a new arrived vessel
#new_vessel_features = pd.DataFrame({'additional_feature1': [22], 'additional_feature2': [14]})
#new_predictions = regr_multirf.predict(new_vessel_features)

# Extracting predictions for the new vessel
#predicted_terminal, predicted_start_time_anchor, predicted_end_time_anchor = new_predictions[0]
