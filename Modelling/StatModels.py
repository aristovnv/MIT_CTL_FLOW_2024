import os
import sys
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
sys.path.append("../Coding")
import Constants as con
import graphviz

model_desc = 'My Model'
#train_size_ratio = 0.8
train_size_ratio = 0.98

userName = 'naristov'
pathData = f"/home/gridsan/{userName}/Modelling/Data"
pathLogs = f"/home/gridsan/{userName}/Modelling/Logs"
pathModels = f"/home/gridsan/{userName}/Modelling/Models"
pathOutput = f"/home/gridsan/{userName}/Modelling/Output"


terminals = ['NY_APM', 'NY_LibertyB', 'NY_LibertyNY', 'NY_Maher', 'NY_Newark', 'NY_RedHook', 'Boston', 'Baltimore', 'Norfolk', 'Portland']

def preprocess_routes():
    file_name = "Routes2.csv"
    df = pd.read_csv(f"{con.pathData}/{file_name}")
    ships_file_name = "ships_grouped.csv"
    ships = pd.read_csv(f"{con.pathData}/{ships_file_name}")
    ships = ships[['IMO Number', 'Grouped', 'size_group']]
    ships.rename(columns={'Grouped': 'Operator'}, inplace=True)
    ships['IMO Number'] = 'IMO' + ships['IMO Number'].astype(str)
    #ships = pd.get_dummies(ships, columns=['Operator', 'size_group'])
    from_frames = []
    to_frames = []
    from_city_columns = df.columns[df.columns.str.startswith('nearestPort_')][:-1]
    to_city_columns = df.columns[df.columns.str.startswith('nearestPort_')][1:]
    # Iterate over columns starting with 'city_'
    for col_name in from_city_columns:
        # Extract the suffix from the column name
        suffix = col_name.split('_')[-1]

        # Filter rows where the city matches 'NY'
        df_sub = df[['IMO', col_name, 'end_time_' + suffix, 'nearestPort_' + str(int(suffix) + 1)]]

        # Rename the columns
        df_sub.columns = ['IMO', 'src_city', 'date', 'dest_city']

        # Convert 'end_time' to date format
        df_sub['date'] = pd.to_datetime(df_sub['date']).dt.date

        # Create dummy columns for destination cities
        df_sub = pd.merge(df_sub, ships, left_on='IMO', right_on='IMO Number', how='inner')
        #df_new = pd.get_dummies(df_sub, columns=['dest_city'], prefix='', prefix_sep='')
        #df_new.drop(columns=['IMO'], inplace=True)

        df_sub['date'] = pd.to_datetime(df_sub['date'])
        # Append the DataFrame to the list of frames
        from_frames.append(df_sub)

    # Concatenate all DataFrames
    from_result = pd.concat(from_frames)
    from_result['operator_size'] = from_result['dest_city'] + '_' + from_result['Operator'] + '_' + from_result['size_group'].astype(str)
    #from_result['Operator'] = from_result['dest_city'] + '_' + from_result['Operator']
    #from_result['size_group'] = from_result['dest_city'] + '_' + from_result['size_group'].astype(str)
    from_result = pd.get_dummies(from_result, 
                             columns=['operator_size'], 
                             prefix={'operator_size': 'opsize'}, 
                             prefix_sep='')
    #from_result.info()
    dummy_columns = [col for col in from_result.columns if col.startswith(('oper','opsize','size'))]

    from_result = from_result.groupby(['date', 'src_city'])[dummy_columns].sum().reset_index()
        
    for col_name in to_city_columns:
        # Extract the suffix from the column name
        suffix = col_name.split('_')[-1]

        # Filter rows where the city matches 'NY'
        df_sub = df[['IMO', col_name, 'end_time_' + suffix, 'nearestPort_' + str(int(suffix) - 1)]]

        # Rename the columns
        df_sub.columns = ['IMO', 'dest_city', 'date', 'src_city']

        # Convert 'end_time' to date format
        df_sub['date'] = pd.to_datetime(df_sub['date']).dt.date

        # Create dummy columns for destination cities
        df_sub = pd.merge(df_sub, ships, left_on='IMO', right_on='IMO Number', how='inner')
        #df_new = pd.get_dummies(df_sub, columns=['dest_city'], prefix='', prefix_sep='')
        #df_new.drop(columns=['IMO'], inplace=True)

        df_sub['date'] = pd.to_datetime(df_sub['date'])
        # Append the DataFrame to the list of frames
        to_frames.append(df_sub)

    # Concatenate all DataFrames
    to_result = pd.concat(to_frames)
    to_result['operator_size'] = to_result['src_city'] + '_' + to_result['Operator'] + '_' + to_result['size_group'].astype(str)
    #to_result['Operator'] = to_result['src_city'] + '_' + to_result['Operator']
    #to_result['size_group'] = to_result['src_city'] + '_' + to_result['size_group'].astype(str)
    to_result = pd.get_dummies(to_result, 
                             columns=['operator_size'], 
                             prefix={'operator_size': 'opsize'}, 
                             prefix_sep='')
    #from_result.info()
    dummy_columns = [col for col in to_result.columns if col.startswith(('oper','opsize','size'))]

    to_result = to_result.groupby(['date', 'dest_city'])[dummy_columns].sum().reset_index()
    from_result.to_csv(f'{con.pathOutput}/from_result.csv', index=False)
    to_result.to_csv(f'{con.pathOutput}/to_result.csv', index=False)
    return from_result, to_result

from_routes_df, to_routes_df = preprocess_routes()

# Create instances of nodes
def construct_features_for_main_nodes(node_data, location):
    # Add features based on the location (e.g., 'New_York')
    file_name = "ResultedHarborStatsdaily.csv"
    df = pd.read_csv(f"{con.pathData}/{file_name}")    
    df['timeDiscrepancy'] = pd.to_datetime(df['timeDiscrepancy'])    
    df['coordinates'] = node_data['coordinates']   
    df = df[df['nearestPort'] == location] # check !!!
    df['month'] = df['timeDiscrepancy'].dt.month
    df['dow'] = df['timeDiscrepancy'].dt.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)  # Encoding DOW using cosine as well for periodicity
    # You can also include numerical encoding for day of week
    df['dow_num'] = df['dow']
    node_data['occupation'] = df[['timeDiscrepancy', 'count', 'Length', 'Width','month_sin','dow_sin','dow_cos','dow_num']]

def construct_features_for_main_nodes_sum(node_data, location):
    # Add features based on the location (e.g., 'New_York')
    file_name = "ResultedBerthStats.csv"
    df = pd.read_csv(f"{con.pathData}/{file_name}")
    df['timeDiscrepancy'] = pd.to_datetime(df['timeDiscrepancy'])
    df['count'] = df['count']
    #file_name = 'BerthMergeAnchor.csv' #ResultedVesselOnBerth.csv
    #df = df[df['nearestPort'] == location] # check !!!
    df['is_location'] = df['nearestPort'].apply(lambda x: 1 if x == location or x.split('_')[0] == location else 0)
    filtered_df = df[df['is_location'] == 1]

    # Group by 'timeDiscrepancy' and aggregate sum of 'count', and average of 'Length' and 'Width'
    df_grouped = filtered_df.groupby('timeDiscrepancy').agg({'count':'sum', 'Length':'mean', 'Width':'mean'}).reset_index()
    df_grouped['month'] = df_grouped['timeDiscrepancy'].dt.month
    df_grouped['dow'] = df_grouped['timeDiscrepancy'].dt.dayofweek
    df_grouped['month_sin'] = np.sin(2 * np.pi * df_grouped['month'] / 12)
    df_grouped['dow_sin'] = np.sin(2 * np.pi * df_grouped['dow'] / 7)
    df_grouped['dow_cos'] = np.cos(2 * np.pi * df_grouped['dow'] / 7)  # Encoding DOW using cosine as well for periodicity
    df_grouped['dow_num'] = df_grouped['dow']
    node_data['occupation'] = df_grouped[['timeDiscrepancy', 'count', 'Length', 'Width','month_sin','dow_sin','dow_cos','dow_num']]    


def construct_features_for_terminal_nodes(node_data, location, terminal):
    file_name = "ResultedBerthStats.csv"
    df = pd.read_csv(f"{con.pathData}/{file_name}")
    df['timeDiscrepancy'] = pd.to_datetime(df['timeDiscrepancy'])
    #file_name = 'BerthMergeAnchor.csv' #ResultedVesselOnBerth.csv
    file_name = "ResultedVesselOnBerth.csv"
    df2 = pd.read_csv(f"{con.pathData}/{file_name}")
    df2['start_time'] = pd.to_datetime(df2['start_time'])
    df2['end_time'] = pd.to_datetime(df2['end_time'])    
    local_location = location
    if terminal == 'Redhook':
        local_location = 'NY_RedHook'
    elif terminal == 'Maher':
        local_location = 'NY_Maher'
    elif terminal == 'LibertyBayonne':
        local_location =  'NY_LibertyB'
    elif terminal == 'LibertyNewYork':
        local_location =  'NY_LibertyNY'
    elif terminal == 'Newark':
        local_location =  'NY_Newark'
    elif terminal == 'APM':
        local_location =  'NY_APM'
    df = df[df['nearestPort'] == local_location]
    df2 = df2[df2['nearestPort'] == local_location]
    local_frdf = from_routes_df[from_routes_df['src_city'] == local_location]
    local_trdf = to_routes_df[to_routes_df['dest_city'] == local_location]
    df2['end_time'] += pd.Timedelta(days=1)
    print(df['count'])
    ships = pd.read_csv(f"{con.pathData}/ships_grouped.csv")
    ships.drop('Operator', axis=1, inplace=True)    
    ships.rename(columns={'Grouped': 'Operator'}, inplace=True)

    ships['IMO Number'] = 'IMO' + ships['IMO Number'].astype(str)

    # Step 2: Merge df and df2
    df2_ships = pd.merge(df2, ships[['IMO Number', 'Operator', 'size_group']], left_on='IMO', right_on='IMO Number', how='inner')

    # Drop the redundant 'id' column after the merge
    df2_ships.drop('IMO Number', axis=1, inplace=True)    

    df2_transformed = df2_ships[['IMO', 'nearestPort', 'start_time', 'end_time', 'Operator', 'size_group']].assign(timeDiscrepancy=lambda x: [pd.date_range(start, end, freq='D').date for start, end in zip(x['start_time'], x['end_time'])]).explode('timeDiscrepancy') 
    
    
    df2_dummies = pd.get_dummies(df2_transformed, columns=['Operator', 'size_group'], prefix={'Operator': 'opcount', 'size_group':'szcount'})

    # Step 2: Group by 'day' and 'location' and dynamically aggregate counts for each user and group
    grouped_df2 = df2_dummies.groupby(['timeDiscrepancy', 'nearestPort']).sum().reset_index()

    #df2_pivot = df2_transformed.pivot_table(index=['timeDiscrepancy','nearestPort','Length', 'Width'],columns='VesselName',aggfunc ='size', fill_value=0)
    # Reset index
    #df2_pivot = df2_pivot.reset_index()
# Merge df1 and df2_transformed
    grouped_df2['timeDiscrepancy'] = pd.to_datetime(grouped_df2['timeDiscrepancy'])
    merged_df = pd.merge(df, grouped_df2, how='left', on=['timeDiscrepancy', 'nearestPort'], suffixes=('', '_df2'))
    print(merged_df.info())
    merged_df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "nearestPort"], inplace=True)
    merged_df['month'] = merged_df['timeDiscrepancy'].dt.month
    merged_df['dow'] = merged_df['timeDiscrepancy'].dt.dayofweek
    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
    merged_df['dow_sin'] = np.sin(2 * np.pi * merged_df['dow'] / 7)
    merged_df['dow_cos'] = np.cos(2 * np.pi * merged_df['dow'] / 7)  # Encoding DOW using cosine as well for periodicity
    merged_df['dow_num'] = merged_df['dow']
    merged_df.drop(columns=['month','dow'], inplace=True)
#    merged_df = pd.merge(merged_df, local_frdf, how='left', left_on='timeDiscrepancy', right_on='date')
#    merged_df.drop(columns=['date','src_city'], inplace=True)
    merged_df = pd.merge(merged_df, local_trdf, how='left', left_on='timeDiscrepancy', right_on='date')
# Fill NaN values with 0
    merged_df.fillna(0, inplace=True)
    merged_df.drop(columns=['date','dest_city'], inplace=True)
    #polygons = [node_data['coordinates'] for _ in range(len(merged_df))]
    #merged_df['coordinates'] =  polygons
    print(f"-----------------------------------")
    print(f"{merged_df.info()}")
    node_data['occupation'] = merged_df


def construct_features_for_wait_nodes(node_data, location):    
    file_name = "ResultedAnchorStatsdaily.csv"
    df = pd.read_csv(f"{con.pathData}/{file_name}")    
    df = df[df['nearestPort']==location]
    df['timeDiscrepancy'] = pd.to_datetime(df['timeDiscrepancy'])
    df['count'] = df['count']
    #file_name = 'BerthMergeAnchor.csv' #ResultedVesselOnBerth.csv
    file_name = 'ResultedVesselAnchor.csv' #ResultedVesselOnBerth.csv
    df2 = pd.read_csv(f"{con.pathData}/{file_name}")
    #df2= df2[df2['nearestPort'].notna()]
    df2 = df2[df2['nearestPort'] == location]    
    df2['start_time'] = pd.to_datetime(df2['start_time'])
    df2['end_time'] = pd.to_datetime(df2['end_time'])
    df2['end_time'] += pd.Timedelta(days=1)    

    ships = pd.read_csv(f"{con.pathData}/ships_grouped.csv")
    ships['IMO Number'] = 'IMO' + ships['IMO Number'].astype(str)
    df2_ships = pd.merge(df2, ships[['IMO Number', 'Operator', 'size_group']], left_on='IMO', right_on='IMO Number', how='inner')
    df2_ships.drop('IMO Number', axis=1, inplace=True)    

    df2_transformed = df2_ships[['IMO', 'nearestPort', 'start_time', 'end_time', 'Operator', 'size_group']].assign(timeDiscrepancy=lambda x: [pd.date_range(start, end, freq='D').date for start, end in zip(x['start_time'], x['end_time'])]).explode('timeDiscrepancy') 

    df2_dummies = pd.get_dummies(df2_transformed, columns=['Operator', 'size_group'], prefix=['operator', 'size_group'])
    grouped_df2 = df2_dummies.groupby(['timeDiscrepancy', 'nearestPort']).sum().reset_index()
    grouped_df2['timeDiscrepancy'] = pd.to_datetime(grouped_df2['timeDiscrepancy'])


#    df2_pivot = df2_transformed.pivot_table(index=['timeDiscrepancy', 'Length', 'Width'],columns='VesselName',aggfunc ='size', fill_value=0)
#    df2_pivot = df2_pivot.reset_index()
#    df2_pivot['timeDiscrepancy'] = pd.to_datetime(df2_pivot['timeDiscrepancy'])
    merged_df = pd.merge(df, grouped_df2, how='left', on=['timeDiscrepancy', 'nearestPort'], suffixes=('', '_df2'))
    polygons = [node_data['coordinates'] for _ in range(len(merged_df))]
    merged_df['coordinates'] =  polygons
    print(merged_df.info())
    merged_df.drop(columns=["Unnamed: 0.1", "index", "Unnamed: 0"], inplace=True)
    merged_df['month'] = merged_df['timeDiscrepancy'].dt.month
    merged_df['dow'] = merged_df['timeDiscrepancy'].dt.dayofweek
    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
    merged_df['dow_sin'] = np.sin(2 * np.pi * merged_df['dow'] / 7)
    merged_df['dow_cos'] = np.cos(2 * np.pi * merged_df['dow'] / 7)  # Encoding DOW using cosine as well for periodicity
    merged_df['dow_num'] = merged_df['dow']
    node_data['occupation'] = merged_df[['timeDiscrepancy', 'count','month_sin','dow_sin','dow_cos','dow_num']]

# Sample nodes with different prefixes
nodes = {
    'NY_Main': {'coordinates': con.NYPortAquatory},
    'NY_Redhook': {'coordinates': con.NY_RedHook_PortCoordinates, 'resources': 10},
    'NY_LibertyBayonne': {'coordinates': con.NY_LibertyBayonne_PortCoordinates, 'resources': 15},
    'NY_LibertyNewYork': {'coordinates': con.NY_LibertyNewYork_PortCoordinates, 'resources': 10},
    'NY_Newark': {'coordinates': con.NY_Newark_PortCoordinates, 'resources': 15},
    'NY_Maher': {'coordinates': con.NY_Maher_PortCoordinates, 'resources': 10},
    'NY_APM': {'coordinates': con.NY_APM_PortCoordinates, 'resources': 15},
    'NY_WaitingArea': {'coordinates': con.NYPortAnchorageArea},

    'Boston_Main': {'coordinates': con.BostonPortAquatory},
    'Boston_Terminal': {'coordinates': con.BostonPortCoordinates, 'resources': 9},
    'Boston_WaitingArea': {'coordinates': con.BostonPortAnchorageArea},

    'Savanna_Main': {'coordinates': con.SavannaPortAquatory},
    'Savanna_Terminal': {'coordinates': con.SavannaGardenCityGE_PortCoordinates, 'resources': 9},
    'Savanna_WaitingArea': {'coordinates': con.SavannaPortAnchorageArea},

    'Norfolk_Main': {'coordinates': con.NorfolkPortAquatory},
    'Norfolk_Terminal': {'coordinates': con.NorfolkPortCoordinates, 'resources': 9},
    'Norfolk_WaitingArea': {'coordinates': con.NorfolkPortAnchorageArea},

    'Baltimore_Main': {'coordinates': con.BaltimorPortAquatory},
    'Baltimore_Terminal': {'coordinates': con.BaltimorPortCoordinates, 'resources': 9},
    'Baltimore_WaitingArea': {'coordinates': con.BaltimorPortAnchorageArea},    
}
def construct_agg_dict(cols, avg_cols):
    agg_dict = []
    for col in cols:
        if col in avg_cols:
            agg_dict.append(col)
    return agg_dict


def process_nodes():
    ila_file_name = "ilaContractCountdown.csv"
    df_ila = pd.read_csv(f"{con.pathData}/{ila_file_name}")    
    df_ila['date'] = pd.to_datetime(df_ila['date'])
    shape_max = 0
    num_features=3300
    len_t_input = 7 #number_of_days
    len_predictions = 30 #number_of_days  
    start_date = '2015-01-02'
    end_date = '2023-09-30'
    time_discrepancy = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Process nodes dynamically
    for node1_name, node1_data in nodes.items():
        node1_port = node1_name.split('_')[0]
        node1_zone = node1_name.split('_')[-1]
        for node2_name, node2_data in nodes.items(): 
            node2_port = node2_name.split('_')[0]
            node2_zone = node2_name.split('_')[-1]                 
            if node1_name != node2_name:
                continue 
            else:
                #terminals
                if (node1_zone != 'Main') and (node1_zone != 'WaitingArea'):
                    construct_features_for_terminal_nodes(nodes[node1_name], node1_port, node1_zone)
                    print(f"------------------------{node1_name}---------------------")
                    print(f"{nodes[node1_name]['occupation']['count']}")
                #main node
                if (node1_zone == 'Main'):
                    #construct_features_for_main_nodes(nodes[node1_name], node1_port)
                    construct_features_for_main_nodes_sum(nodes[node1_name], node1_port)
                #wait zone
                if (node1_zone == 'WaitingArea'):
                    construct_features_for_wait_nodes(nodes[node1_name], node1_port)

    # Iterate over nodes
    for node, info in nodes.items():
        nodes_data = info['occupation']

        # Define columns for which to calculate averages
        avg_cols = ['Length', 'Width', 'Length_df2', 'Width_df2', 'month_sin','dow_sin','dow_cos','dow_num']

        # Calculate aggregate functions for average columns
        agg_dict = construct_agg_dict(nodes_data.columns.tolist(), avg_cols)

        # Get columns for sum aggregation
        sum_cols = nodes_data.columns.difference(avg_cols + ['count', 'timeDiscrepancy'])

        # Perform aggregation
        result = nodes_data

        if len(agg_dict) > 0:
            result = nodes_data.groupby(['timeDiscrepancy', 'count']).agg({col: 'mean' for col in agg_dict})
        if len(sum_cols) > 0:
            result_sum = nodes_data.groupby(['timeDiscrepancy', 'count']).agg({col: 'sum' for col in sum_cols})
            result = pd.concat([result, result_sum], axis=1)

        result.reset_index(inplace=True)

        # Specify the number of previous days (lag)
        result['timeDiscrepancy'] = result['timeDiscrepancy'].dt.date
        result.set_index('timeDiscrepancy', inplace=True)
        # Assuming df is your DataFrame with 'timeDiscrepancy' as the index
        result.index = pd.to_datetime(result.index)

        # Extract unique dates from 2015/01/01 to 2023/09/30
        all_dates = pd.date_range(start='2015-01-02', end='2023-09-30')

        # Reindex the DataFrame to include all_dates
        result = result.reindex(all_dates, fill_value=0)

        # Reset index to convert 'timeDiscrepancy' back to a column
        result.reset_index(inplace=True)

        # Rename the 'index' column to 'timeDiscrepancy'
        result.rename(columns={'index': 'timeDiscrepancy'}, inplace=True)

        # Create lagged versions of all columns
        for col in result.columns:        
            #if col != 'timeDiscrepancy':
            if col == 'count' or col.startswith(('opcount', 'szcount')):
                #for i in range(1, len_t_input + 1):
                    result[f'{col}_lag_{1}'] = result[col].shift(1).fillna(0)
        for col in result.columns:
            if col.startswith(('opsize', 'oper', 'size')):
                for i in range(1, len_t_input + 1):
                    result[f'{col}_lag_{i}'] = result[col].shift(i).fillna(0)                
        #for col in result.columns:
        #    if col.startswith(('opsize', 'oper', 'size')):
        #        for i in range(1, len_t_input + 1):
        #            result[f'{col}_lag_{i}'] = result[col].shift(i).fillna(0)                
    # Merge the DataFrames based on timeDiscrepancy and date columns
        result = pd.merge(result, df_ila[['date', 'ila_contract_countdown']], how='left', left_on='timeDiscrepancy', right_on='date')
        result.drop(columns=['date'], inplace=True)
        columns_to_drop = [col for col in result.columns if col.startswith(('oper', 'opsize', 'size','opcount','szcount')) and 'leap' not in col and 'lag' not in col]
        # Drop the columns
        result.drop(columns=columns_to_drop, inplace=True)    

        print(f"For node {node}, we have shape {result.shape}")
        _, kk = result.shape
        if kk > shape_max:
            shape_max = kk
        result.to_csv(f"{con.pathOutput}/gr_node_{node}.csv")    
        nodes[node]['occupation'] = result

    all_columns = set()
    for node, info in nodes.items():
        df = info['occupation']
        all_columns.update(df.columns)

    # Step 3: Construct the matrix for each node
    X_data = []
    y_data = []
    
    for node, info in nodes.items():
        df = info['occupation']
        df['timeDiscrepancy'] = pd.to_datetime(df['timeDiscrepancy'])

    # Set the "date" column as the index
        df.set_index('timeDiscrepancy', inplace=True)
        # Remove 'count' from the list of columns
        columns_without_count = [col for col in all_columns if col != 'count']
        # Extract data without the 'count' column
        X_matrix = pd.DataFrame(index=time_discrepancy, columns= list(all_columns), dtype=np.float64)
        y_vector = pd.DataFrame(index=time_discrepancy, columns=['count'], dtype=np.float64)
        
        #X_matrix = np.zeros((len(time_discrepancy), num_features), dtype=np.float64)
        # Extract 'count' column data
        #y_vector = np.zeros((len(time_discrepancy), 1), dtype=np.float64)
        # Fill X_matrix and y_vector with values from the corresponding DataFrame
        for date in time_discrepancy:
            if date in df.index:
                X_matrix.loc[date] = df.loc[date]
                # Set 'count' column to 0 in X_matrix
                X_matrix.loc[date, 'count'] = 0
                y_vector.loc[date, 'count'] = df.loc[date]['count']        
        '''
        for i, date in enumerate(time_discrepancy):
            if date in df.index:
                num_cols = min(num_features, len(df.columns))  # Number of columns to consider
                X_matrix[i, :num_cols] = df.loc[date].values[:num_cols]  
                #X_matrix[i, :num_cols] = df_slice.values  # Assign values to X_matrix            
                #X_matrix[i, :len(df)] = df.loc[date].values
                # Set 'count' column in X_matrix to 0
                X_matrix[i, df.columns.get_loc('count')] = 0
                y_vector[i] = df.loc[date]['count']            
        '''
        X_matrix['Terminal'] = node
        print(f"appending {node}")
        y_vector['Terminal'] = node
        X_matrix['timeDiscrepancy'] = X_matrix.index  # Copy the index to a new column
        X_matrix.reset_index(drop=True, inplace=True)
        y_vector['timeDiscrepancy'] = y_vector.index  # Copy the index to a new column
        y_vector.reset_index(drop=True, inplace=True)

        #print(f"{node} and {info}: {y_vector}")
        X_data.append(X_matrix)
        y_data.append(y_vector)
        
        
    print(f"X_data is {X_data}")   
    X_data_reset = [df.reset_index(drop=True) for df in X_data]
    y_data_reset = [df.reset_index(drop=True) for df in y_data]
    return pd.concat(X_data_reset, ignore_index=True), pd.concat(y_data_reset, ignore_index=True), nodes

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

def calculate_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Ensure that both inputs are numpy arrays of the same shape
    if y_true.shape != y_pred.shape:
        y_pred = y_pred.reshape(y_true.shape)

    # Calculate the absolute difference and the average of absolute values
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Avoid division by zero by replacing zeros in the denominator with a small value
    denominator[denominator == 0] = 1e-10  # Small constant to prevent division by zero

    # Calculate SMAPE for each element and then average over all elements
    smape = np.mean(numerator / denominator) * 100

    return smape

###############################################################################################################
def startModelPerTerminal(model, roundQty = 0):
    #df_merged, features_columns, selected_columns = dataPreparation(modelType, targetPrefix)
    X_data, y_data, nd = process_nodes()
    X_data = X_data.fillna(0)
    y_data = y_data.fillna(0)
    
    for terminal, xx in nodes.items():
        print(f'Working with terminal {terminal}')
        X = X_data[X_data['Terminal'] == terminal] #nodes[terminal]['occupation']
        y = y_data[y_data['Terminal'] == terminal]

        X.drop('Terminal', axis=1, inplace=True)
        y.drop('Terminal', axis=1, inplace=True)

        train_size = len(X) - 30
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            # AutoARIMA
        if model == 'arima':
            prediction = runArima(X_train, X_test, y_train, y_test, terminal)   

            # Exponential Smoothing
        if model == 'expSmooth':
            prediction = runExpSmooth(X_train, X_test, y_train, y_test, terminal)

            # LSTM (Long Short-Term Memory)
        if model == 'LSTM':
            prediction = runLSTM(X_train, X_test, y_train, y_test, terminal)
            index_for_forecast = y_test.index        
            prediction = pd.DataFrame(prediction, columns = [f'{targetPrefix}_{terminal}'], index=index_for_forecast)    


        if model == 'XGBoost':
            prediction = runXGBoost(X_train, X_test, y_train, y_test, terminal)
            index_for_forecast = y_test.index
            prediction = pd.DataFrame(prediction, columns = [f'{targetPrefix}_{terminal}'], index=index_for_forecast)   

        if model =='PoissonRegressor':
            prediction = runPoissonRegressor(X_train, X_test, y_train, y_test, terminal)


        saveError_n(f'{terminal}_{model}', f'{terminal}', f'{model}', f'one terminal', y_test, prediction)
        if roundQty == 1:
            saveError_n(f'{terminal}_{model}', f'{terminal}', f'{model}', f'one terminal', y_test, prediction, roundQty)

        saveResults_n(f'{terminal}_{model}', y_test, prediction)    
    return 0

def saveError_n(fileName, terminal, model, description, actual, prediction, roundQty = 0):
    output = prediction.copy()
    y_true = np.nan_to_num(actual, nan=0.0)#actual.fillna(0)  # Fill NaNs with zero
    y_pred = np.nan_to_num(prediction, nan=0.0)#prediction.fillna(0) #y_pred.mean()  
    if roundQty == 1:
        output = output.round()
    err = {
        "roundQty" : [roundQty],
        "terminal": [f"{terminal}"],
        "model": [f"{model}"],
        "description": [f"{description}"],
        "mse": [mean_squared_error(y_true, y_pred)],
        "mae": [mean_absolute_error(y_true, y_pred)], 
        "RSquared": [r2_score(y_true, y_pred)],
        "mape": [mean_absolute_percentage_error(y_true, y_pred)], 
        "smape":[calculate_smape(y_true, y_pred)] 
    }

    pd.DataFrame(err).to_csv(f"{pathLogs}/{fileName}{roundQty}.csv")

def saveError_bt(fileName, model, description, actual, prediction, roundQty = 0):
    output = prediction.copy()
    #y_true = np.nan_to_num(actual, nan=0.0)
    y_true = actual.fillna(0)  # Fill NaNs with zero
    y_pred = np.nan_to_num(prediction, nan=0.0)#prediction.fillna(0) #y_pred.mean()  
    if roundQty == 1:
        output = output.round()
    error_df = pd.DataFrame()
    for col_idx in range(y_true.shape[1]):
        # Get the column name from y_test
        column_name = y_true.columns[col_idx] 
        y_true_col = y_true[column_name]
        y_pred_col = y_pred[:, col_idx]#y_pred.iloc[:, col_idx]
        err = {
            "roundQty" : roundQty,
            "terminal": f"{column_name}",
            "model": f"{model}",
            "description": f"{description}",
            "mse": mean_squared_error(y_true_col, y_pred_col),
            "mae": mean_absolute_error(y_true_col, y_pred_col), 
            "RSquared": r2_score(y_true_col, y_pred_col),
            "mape": mean_absolute_percentage_error(y_true_col, y_pred_col), 
            "smape":calculate_smape(y_true_col, y_pred_col) 
        }    
        error_df = error_df.append(err, ignore_index=True)  # Append the row to the DataFrame
    error_df.to_csv(f"{pathLogs}/{fileName}_error_br.csv")
    
    
def saveResults_n(fileName, actual, prediction):
    actual.to_csv(f'{pathOutput}/{fileName}_actual.csv', index=True)
    prediction.to_csv(f'{pathOutput}/{fileName}_prediction.csv', index=True)

###
def startModelForGroup(roundQty = 0):
    X_data, y_data, nd = process_nodes()
    #X_data.to_csv(f'{pathOutput}/xdata.csv')
    #print(f"X_data info {X_data.info()}")
    X_data = X_data.fillna(0)
    y_data = y_data.fillna(0)

    duplicates = X_data[X_data.duplicated(subset=['timeDiscrepancy', 'Terminal'], keep=False)]
    print("Duplicate Entries:")
    print(duplicates)
    columns_to_pivot = [col for col in X_data.columns if col not in ['Terminal', 'timeDiscrepancy']]
    X = X_data.pivot(index='timeDiscrepancy', columns='Terminal', values=columns_to_pivot)
    y = y_data.pivot(index='timeDiscrepancy', columns='Terminal', values='count')
    
    
    #X.drop('Terminal', axis=1, inplace=True)
    #y.drop('Terminal', axis=1, inplace=True)
    X.to_csv(f'{pathOutput}/INPUT_actual.csv')
    train_size = len(X) - 30
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    #,'VAR','1DCNN','RandomForestClassifier','LSTNML', 'SVR', 'XGBoost', 'LSTM','RandomForestRegressor'
    model_names = ['XGBoost' ]
            # AutoARIMA
    for model in model_names:
        if model == 'LSTM':
            prediction = runLSTMAll(X_train, X_test, y_train, y_test)

        if model == 'XGBoost':
            prediction = runXGBoostAll(X_train, X_test, y_train, y_test)

        if model == 'VAR':
            prediction = runVAR(X_train, X_test, y_train, y_test)

        if model == '1DCNN':
            prediction = run1DCNN(X_train, X_test, y_train, y_test)

        if model == 'RandomForestClassifier':
            prediction = runRandomForestClassifier(X_train, X_test, y_train, y_test)
        if model == 'RandomForestRegressor':
            prediction = runRandomForestRegressor(X_train, X_test, y_train, y_test)
        if model == 'LSTNML':
            prediction = runLSTNML(X_train, X_test, y_train, y_test)

        if model == 'SVR':
            prediction = runSVR(X_train, X_test, y_train, y_test)

            #new
        saveError_bt(f'{model}', f'{model}', f'several terminals', y_test, prediction)
        if roundQty == 1:
            saveError_bt(f'{model}', f'{model}', f'several terminals', y_test, prediction, roundQty)

        index_for_forecast = y_test.index
        forecasted_columns = {col: prediction[:, i] for i, col in enumerate(y_test.columns)}

        forecasted_df = pd.DataFrame(forecasted_columns, index=index_for_forecast)    
        saveResults_n(f'group_{model}_1', y_test, forecasted_df)
    return 0

    
#####
def runArima(X_train, X_test, y_train, y_test, terminal):
    
    print(f"AutoARIMA - Forecasting for {terminal}")
    model_autoarima_count = auto_arima(y_train, exogenous=X_train, suppress_warnings=True, trace=True)
    count_forecast_autoarima = model_autoarima_count.predict(n_periods=len(X_test), exogenous=X_test)
    dump(model_autoarima_count, f'{pathModels}/model_autoarima_{terminal}.joblib')
    return count_forecast_autoarima

def runLSTMAll(X_train, X_test, y_train, y_test):
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
    model_lstm_length.fit(X_train_length_reshaped, y_train, epochs=50, verbose=1)

    length_forecast_lstm = model_lstm_length.predict(X_test_length_reshaped)

    return length_forecast_lstm



########################################################


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

def runXGBoostAll(X_train, X_test, y_train, y_test):
    print(f"XGBoost - Forecasting ")
    dtrain_length = xgb.DMatrix(X_train, label=y_train)
    dtest_length = xgb.DMatrix(X_test, label=y_test)
    
    params = {
    'objective': 'reg:squarederror',  # for regression tasks
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100}    
    
    model_XGBoost_length = xgb.train(params, dtrain_length, num_boost_round=10)
    length_forecast_XGBoost = model_XGBoost_length.predict(dtest_length)
    
    #xgb.plot_importance(model_XGBoost_length)
    #graph = xgb.to_graphviz(model_XGBoost_length, num_trees=3)
    #graph.render(f'{pathOutput}/XGB_importance_tree', format="png", cleanup=True)  # Saves as PNG and cleans up temporary files
    
    feature_importance = model_XGBoost_length.get_score(importance_type='gain')

    # Convert feature importance to a DataFrame
    feature_importance_df = pd.DataFrame(
        list(feature_importance.items()),  # Convert dictionary to list of tuples
        columns=["Feature", "Importance"]  # Define column names
    )

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Save the DataFrame to a CSV file
    feature_importance_df.to_csv(f'{pathOutput}/XGB_ft_importance_tree.csv', index=False)    
    return length_forecast_XGBoost 



###
def runLSTNML(X_train, X_test, y_train, y_test):
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
def runVAR(X_train, X_test, y_train, y_test):# (Vector Autoregression):

    model = VAR(y_train, exog=X_train)
    results = model.fit()

    forecast = results.forecast(y_train.values, steps=len(X_test), exog_future=X_test)
    
    return forecast
##

def crossValidation(model, actual):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predicted_cv = cross_val_predict(model, pd.DataFrame(np.random.rand(100, 10)), actual, cv=cv)





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
def run1DCNN(X_train, X_test, y_train, y_test):

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
    model.add(Dense(y_test.shape[1], activation='linear'))  # Output layer

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

def runRandomForestClassifier(X_train, X_test, y_train, y_test):
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
'''
def runRandomForestRegressor(X_train, X_test, y_train, y_test):
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    regr_multirf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = regr_multirf.predict(X_test)
    return y_pred
'''
def runRandomForestRegressor(X_train, X_test, y_train, y_test):
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    regr_multirf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = []
    
    feature_importances = []

    for i, estimator in enumerate(regr_multirf.estimators_):
        # Get feature importances from each RandomForestRegressor
        importances = estimator.feature_importances_

        # Create a DataFrame to store the feature importances with context
        importance_df = pd.DataFrame({
            'Feature': [f'Feature_{j}' for j in range(X_train.shape[1])],
            'Importance': importances,
            'Output_Target': [f'Target_{i}'] * X_train.shape[1]  # Label for each output target
        })

        feature_importances.append(importance_df)

    # Concatenate all feature importance DataFrames
    final_importance_df = pd.concat(feature_importances)
    final_importance_df.to_csv(f'{pathOutput}/RF_importance.csv', index=False)  
    # Iterate through each row in X_test
    for index, row in X_test.iterrows():
        # Predict for each row and append to y_pred list
        # The row is converted to a DataFrame before prediction to maintain shape
        prediction = regr_multirf.predict(row.values.reshape(1, -1))
        y_pred.append(prediction)    
    #y_pred = regr_multirf.predict(X_test)
    y_pred = np.vstack(y_pred)
    return y_pred

def runSVR(X_train, X_test, y_train, y_test):
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

def load_data():
    userName = 'naristov'
    pathOutput = f"/home/gridsan/{userName}/Modelling/Output"
    inp = pd.read_csv(f'{pathOutput}/INPUT_actual.csv')
    patterns = ['Length', 'Width', 'count']

    # Create a regex pattern to match columns that start with any of these patterns
    regex_pattern = '^(' + '|'.join(patterns) + ')'
    filtered_df = inp.filter(regex=regex_pattern)
    filtered_df.to_csv(f'{pathOutput}/INPUT_somecols.csv')
    #inp.head(10).to_csv(f'{pathOutput}/INPUT_small.csv')
#startModelPerTerminal('arima')    
startModelForGroup(roundQty = 1)        
#startModelForGroup('SVR', 0, targetPrefix = "count", roundQty = 1)
#startModelForGroup('SVR', 0, targetPrefix = "diff_to_prev", roundQty = 1)
#callCheckNoise('SVR', 0, targetPrefix = "diff_to_prev", roundQty = 1)
#load_data()