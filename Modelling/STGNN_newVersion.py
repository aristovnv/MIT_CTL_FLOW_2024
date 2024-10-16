import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, diags, identity, diags
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import math
import scipy
import time
sys.path.append("../Coding")
import Constants as con
from sklearn.model_selection import train_test_split

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




def create_A_hat(all_edges):
    """
    Create the adjacency matrix A, the degree matrix D of A, and calculate A_hat

    Return:
    A_hat -- A Pytorch sparse tensor A_hat = D^(-1/2)AD^(-1/2)

    """
    # Step 1: Extract unique node names
    nodes = set()
    for edge in all_edges:
        nodes.update(edge)

    # Convert the set of node names into a list
    nodes_list = list(nodes)

    # Step 2: Create an empty adjacency matrix
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Step 3: Populate the adjacency matrix based on the edges
    node_index = {node: i for i, node in enumerate(nodes_list)}  # Mapping from node name to index
    for edge, attributes in all_edges.items():
        node1, node2 = edge
        distance = attributes.get('distance', 0)  # If distance is not specified, default to 0
        adj_matrix[node_index[node1], node_index[node2]] = distance
    A = pd.DataFrame(adj_matrix, index=nodes_list, columns=nodes_list)
    #A = adj_matrix #pd.read_csv('../Data/adj_matrix_dist.csv', index_col = 0)
    print("---------------------A is------------------")
    print(A)
    A.to_csv(f'{con.pathOutput}/adj.csv')
    num_sensors = A.shape[0]

    #sigma_sq = 10**2
    #sigma_sq = 0.01**2 <- works
    #sigma_sq = 0.001**2
    sigma = A.values.std()
    sigma_sq = sigma**2  # should be according to matrix
    print(f"sigma is {sigma}")
    squared = np.power(A.values, 2)
    data_weighted = np.exp(-squared/sigma_sq)
    A = coo_matrix(data_weighted) + identity(num_sensors, dtype=np.float32)

    diagonals = A.sum(axis = 1)
    D = diags(diagonals.flatten(), [0], shape=(num_sensors, num_sensors))

    A_hat = ((D.power(-1/2)).dot(A)).dot(D.power(-1/2))
    values = A_hat.data
    indices = A_hat.nonzero()
    A_hat = torch.sparse_coo_tensor(indices, values, dtype = torch.float)

    return A_hat


'''
class Node:
    def __init__(self, coordinates, resources, utilization_data):
        self.coordinates = coordinates
        self.resources = resources
        self.utilization_data = utilization_data

class Edge:
    def __init__(self, distance, direction_data):
        self.distance = distance
        self.direction_data = direction_data
'''
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



# Create a dictionary to store edges
edges = {}
distance = pd.read_csv(f"{con.pathData}/DistanceStatic.csv", index_col=0)
shape_max = 0
# Process nodes dynamically
for node1_name, node1_data in nodes.items():
    node1_port = node1_name.split('_')[0]
    node1_zone = node1_name.split('_')[-1]
    for node2_name, node2_data in nodes.items(): 
        node2_port = node2_name.split('_')[0]
        node2_zone = node2_name.split('_')[-1]                 
        if node1_name != node2_name:
            # between terminals in one port 
            if (node1_port == node2_port) and (node1_zone != 'Main') and (node1_zone != 'WaitingArea') and (node2_zone != 'Main') and (node2_zone != 'WaitingArea'):
                edges[(node1_name, node2_name)] = {'distance': 2}
            # wait zone to terminals
            if (node1_port == node2_port) and (node2_zone != 'Main') and (node2_zone != 'WaitingArea') and (node1_zone == 'WaitingArea'):
                edges[(node1_name, node2_name)] = {'distance': 3}
            # main node to terminals 
            if (node1_port == node2_port) and (node1_zone == 'Main') and (node2_zone != 'Main') and (node2_zone != 'WaitingArea'):
                edges[(node1_name, node2_name)] = {'distance': 3}
            # main node to wait zone 
            if (node1_port == node2_port) and (node1_zone == 'Main') and (node2_zone == 'WaitingArea'):
                edges[(node1_name, node2_name)] = {'distance': 0.1}            
            # terminal to main node
            if (node1_port == node2_port) and (node1_zone != 'Main') and (node1_zone != 'WaitingArea') and (node2_zone == 'Main'):
                edges[(node1_name, node2_name)] = {'distance': 3}            
            # main node to main node
            if (node1_port != node2_port) and (node1_zone == 'Main') and (node2_zone == 'Main'):
                value = 5
                try: 
                    value = distance.loc[node1_port, node2_port]
                    if pd.isnull(value) or value == 0:
                        value = 5
                except KeyError:
                    print(f"could fin distance from {node1_port} to {node2_port}")
                    value = 5
                edges[(node1_name, node2_name)] = {'distance': value}
            # Example: Build edges between nodes based on some condition
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

                

number_of_nodes = len(nodes) 
number_heads = 4 # was 4
num_features=3300
len_t_input = 7 #number_of_days
len_predictions = 30 #number_of_days
batch_size = 1
epoch_number = 20
hidden_size = 256

number_of_days = 3195 # 6*365 + 2*366 + 5*31+3*30+28 = 273 + 732 + 2190 = 1005 + 2190 - 1 (excluding 2015-01-01)
print(f"number of nodes is {number_of_nodes}")
class GCNLayer(nn.Module):
    """
    A graph convolutional layer

    Args:
    in_size -- The number of expected features in the input X
    out_size -- The number of features of the output tensor

    Inputs:
    A_hat -- Pytorch sparse tensor A_hat = D^(-1/2)AD^(-1/2)
    X -- Pytorch tensor X_t with shape [number of sensors, number of features] or batch of matrices X_t with shape [batch size, number of sensors, number of features]

    Outputs:
    out -- Tensor containing the result of the matrix multiplication (A_hat*X_t*W)
    """
    def __init__(self, in_size, out_size):
        super(GCNLayer, self).__init__()
        W = torch.empty((in_size, out_size))
        self.W = nn.parameter.Parameter(torch.nn.init.kaiming_uniform_(W))

    def forward(self, A_hat, X):
        if X.dim() == 2: #  GRU iterates over all time steps, so no time dimension
            out = torch.matmul(torch.spmm(A_hat, X), self.W)
            return out
        else:
            #out = torch.matmul(torch.spmm(A_hat, X.reshape(109,-1)).reshape(109,-1,8), self.W)
            out = torch.matmul(torch.spmm(A_hat, X.reshape(A_hat.shape[0],-1)).reshape(A_hat.shape[0],-1,num_features), self.W)
            return out


class PositionalEncoding(nn.Module):# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    Calculate a positional encoding for each of the sensor matrices

    Args:
    d_model -- The number of features in the model (model dimensions)
    max_len -- The number of times t in the model
    number_nodes -- The number of nodes in the dataset

    Outputs:
    positional_encodings -- Pytorch tensor containing the positional encoding for all sensors
    """
    def __init__(self, d_model, max_len, number_nodes):
        super(PositionalEncoding, self).__init__()
        positional_encodings = torch.zeros(number_nodes, max_len, d_model)
        for i in range(number_nodes):
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = 10000**(2*i/d_model)
            #div_term = 100**(2*i/d_model)
            pe[0::2, :] = torch.sin(position / div_term)[0::2, :]
            pe[1::2, :] = torch.cos(position / div_term)[1::2, :]
            positional_encodings[i] = pe
        self.register_buffer('pe', positional_encodings)

    def forward(self):
        positional_encodings = self.pe
        return positional_encodings

class MLP(nn.Module):
    """
    Multi-layer perceptron/multi-layer feedforward network which applies a linear transformation to the input multiple times

    Args:
    in_size -- The number of expected features in the input x
    hidden_size -- Size of the hidden layer
    out_size -- The number of features of the output tensor

    Inputs:
    x -- Pytorch tensor of size [*, in_size], where * indicates it can be any dimension.

    Outputs:
    out -- Tensor containing an output of size [*, out_size]
    """
    def __init__(self, in_size, hidden_size , out_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, hidden_size)
        self.fc22 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        #x = F.selu(self.fc1(x))#
        x = F.relu(self.fc1(x))
        #x = F.selu(self.fc2(x))#
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc21(x))
        #x = F.relu(self.fc22(x))
        out = self.fc3(x)
        return out

class STGNN(nn.Module): #maybe create functions for loops
    """
    Multi-layer perceptron/multi-layer feedforward network which applies a linear transformation to the input multiple times

    Args:
    in_size -- The number of expected features in the input X
    hidden_size -- Size of the hidden layer
    out_size -- The number of features of the output tensor
    len_t -- The number of times t in the input X
    num_features -- The number of features in the input X

    Inputs:
    A_hat -- Pytorch sparse tensor A_hat = D^(-1/2)AD^(-1/2)
    X -- Pytorch tensor X_t with shape [number of sensors, number of features] or batch of matrices X_t with shape [batch size, number of sensors, number of features]
    num_features -- The number of features in the input X
    device -- The device the calculations are run on ('cpu' or 'cuda:0')
    H_tilde -- The first hidden layer to the GRU

    Outputs:
    out -- Tensor containing the predictions of the network of size [number of sensors, out_size]
    """
    def __init__(self, in_size, hidden_size, out_size, len_t, num_features):
        super(STGNN, self).__init__()
        self.GCN = GCNLayer(in_size, in_size)        
        self.GRU = nn.GRUCell(in_size, in_size) #outputs hidden layer for each time step
        self.pos_encoding = PositionalEncoding(d_model = num_features, max_len = len_t, number_nodes = number_of_nodes)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=number_heads, dim_feedforward=hidden_size, dropout=0)
        #self.transformer_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=number_heads, dim_feedforward=hidden_size, dropout=0.1)
        self.encoder_norm = nn.LayerNorm(num_features)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1, norm = self.encoder_norm)
        self.MLP = MLP(len_t*num_features, hidden_size, out_size)

    def forward(self, A_hat, X, num_features, device, H_tilde = None):
        #X_tilde = F.selu(self.GCN(A_hat.to(device), X))
        X_tilde = F.relu(self.GCN(A_hat.to(device), X))
        H_tilde_list = []
        for t in range(0, X.size(1)): #X has Xt for t, ..., t+len(X)
            if H_tilde is None:
                H_tilde = torch.zeros([number_of_nodes, num_features]).to(device)
            GRU_out = self.GRU(X_tilde.transpose(0,1)[t], H_tilde)
            H_tilde_list.append(GRU_out)
            #H_tilde = F.selu(self.GCN(A_hat.to(device), GRU_out))#
            H_tilde = F.relu(self.GCN(A_hat.to(device), GRU_out))
        H_tilde_list = torch.stack(H_tilde_list)
        positional_encodings = self.pos_encoding().to(device)
        H_matrices = torch.transpose(H_tilde_list, 0, 1) + positional_encodings
        out = self.transformer(H_matrices)
        out = self.MLP(out.reshape((-1, out.size(1)*out.size(2))))
        return out

def train_STGNN_network(train_loader, validation_loader, A_hat, num_features, device, len_t_input, len_predictions, epoch_number = 20, hidden_size = 64):
    """
    Train the Spatial Temporal Graph NN

    Arguments:
    train_loader -- The Pytorch DataLoader that iterates over the training data
    validation_loader -- The Pytorch DataLoader that iterates over the validation data
    A_hat -- The Pytorch sparse tensor calculated by A_hat = D^(-1/2)AD^(-1/2)
    num_features -- The number of features in the features matrices X (the number of columns)
    device -- The device the calculations are run on ('cpu' or 'cuda:0')
    len_t_input -- The length of the time interval that the network uses as input
    len_predictions -- The length of the time interval of predictions

    Return:
    net -- The trained network
    optimizer -- The optimizer that was used to train the network
    mean_loss -- List containing the mean loss per epoch
    std_loss -- List containing the standard deviation of the loss per epoch
    val_mean_loss -- List containing the mean loss per epoch
    val_std_loss -- List containing the standard deviation of the loss per epoch

    """
    net = STGNN(in_size = num_features, hidden_size = hidden_size, out_size = len_predictions, len_t = len_t_input, num_features = num_features)
    #net.load_state_dict(torch.load('../Networks/freight_april_00001.pt'))
    net.train()
    net.to(device)
    criterion =  nn.MSELoss()
    #optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00039910003644642216, amsgrad=False)#0.00001,
    optimizer = optim.AdamW(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00039910003644642216, amsgrad=False)#0.00001,
    loss_list = []
    loss_list = np.array(loss_list)
    mean_loss = []
    mean_loss = np.array(mean_loss)
    std_loss = []
    std_loss = np.array(std_loss)

    val_loss_list = []
    val_loss_list = np.array(val_loss_list)
    val_mean_loss = []
    val_mean_loss = np.array(val_mean_loss)
    val_std_loss = []
    val_std_loss = np.array(val_std_loss)

    A_hat = A_hat.to(device)
    #print(f"----------------------A -HAT")
    #print(f"{A_hat}")
    for epoch in range(epoch_number):  # loop over the dataset multiple times
        print('epoch', epoch+1)
        start = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            inputs = torch.transpose(x.squeeze(0), 0, 1).to(device)
            targets = torch.transpose(y.squeeze(0), 0, 1).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(A_hat, inputs, num_features, device = device)
            loss = torch.sqrt(criterion(outputs, targets)) #mean loss of the batch
            #loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            #print("-------------------- in training")
            #print(f"batch_idx: {batch_idx}")
            #print(f"{outputs}")
            loss_list = np.append(loss_list, loss.item())

        print('Training loss:', loss_list.mean())
        mean_loss = np.append(mean_loss, loss_list.mean())
        std_loss = np.append(std_loss, loss_list.std())

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(validation_loader):
                inputs = torch.transpose(x.squeeze(0), 0, 1).to(device)
                targets = torch.transpose(y.squeeze(0), 0, 1).to(device)
                val_outputs = net(A_hat, inputs, num_features, device)
                val_loss = torch.sqrt(criterion(val_outputs, targets))  #mean loss of the batch
                val_loss_list = np.append(val_loss_list, val_loss.item())
        print('Validation loss:', val_loss_list.mean())
        val_mean_loss = np.append(val_mean_loss, val_loss_list.mean())
        val_std_loss = np.append(val_std_loss, val_loss_list.std())
        end = time.time()
        print('Duration: ', end-start)
    print('Finished Training')
    return net, optimizer, mean_loss, std_loss, val_mean_loss, val_std_loss    

def train(model, train_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) #0.001

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}')

    return model    

def test_RMSE(net, A_hat, test_loader, num_features, device="cpu"):
    criterion = nn.MSELoss() #SMAPELoss() #
    A_hat.to(device)
    net.to(device)
    loss_list = []
    loss_list = np.array(loss_list)
    mean_loss = []
    mean_loss = np.array(mean_loss)
    std_loss = []
    std_loss = np.array(std_loss)
    i = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            inputs = torch.transpose(x.squeeze(0), 0, 1).to(device)
            targets = torch.transpose(y.squeeze(0), 0, 1).to(device)
            outputs = net(A_hat, inputs, num_features, device)
            loss = torch.sqrt(criterion(outputs, targets))  #mean loss of the batch
            #loss = criterion(outputs, targets)
            loss_list = np.append(loss_list, loss.item())
            #print(f"{i}: {loss}")
            #print(f"{batch_idx}------outputs-------")
            #print(outputs)
            #print(f"{batch_idx}----targets----")
            #print(targets)                    
            i+=1
        mean_loss = np.append(mean_loss, loss_list.mean())
        std_loss = np.append(std_loss, loss_list.std())
        print("------outputs-------")
        print(outputs)
        print("----targets----")
        print(targets)        
    return mean_loss, std_loss, outputs, targets

class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, actual, forecast):
        """
        Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).
        
        Args:
        actual (torch.Tensor): Tensor of actual values.
        forecast (torch.Tensor): Tensor of forecasted values.
        
        Returns:
        torch.Tensor: SMAPE loss.
        """
        denominator = (torch.abs(actual) + torch.abs(forecast)) / 2.0
        diff = torch.abs(actual - forecast) / denominator
        diff[denominator == 0] = 0  # Handle division by zero
        return torch.mean(diff) * 100

def sliding_window(window_size_x, window_size_y, len_data, X, y):
    """
    Create a batch of size window_size_x for the feature matrices X and create a batch of size window_size_y for the targets y

    Arguments:
    window_size_x -- Int which indicates how many hours are in a batch of feature matrices X
    window_size_y -- Int which indicates how many hours are in a batch of targets y
    X -- A Pytorch tensor containing the feature matrices
    y -- A Pytorch tensor containing the targets

    Return:
    X_data -- A Pytorch tensor containing batches of feature matrices X
    y_data -- A Pytorch tensor containing batches of targets y

    """
    X_data = []
    y_data = []
    for i in range(window_size_x, len_data-window_size_y):
        Xt = X[i-window_size_x:i]
        yt = y[i:i+window_size_y]
        #yt = torch.cat(tuple(yt), 1)
        #data.append((Xt, yt))
        X_data.append(Xt)
        y_data.append(yt)
    X_data = torch.stack(X_data)
    y_data = torch.stack(y_data)
    return X_data, y_data

def create_data_loaders(X, y, window_size_x, window_size_y, indices = [0.7, 0.1, 0.2], batch_size = 1):
    """
    Create the train, validation, and test splits and create a Pytorch DataLoader for each of these splits

    Arguments:
    X -- A Pytorch tensor containing the batches of feature matrices X
    y -- A Pytorch tensor containing the batches of targets y
    indices -- A list of 3 floats indicating how big the train, validation, and test split should be, respectively. Should satisfy sum(indices) = 1

    Return:
    train_loader -- The Pytorch DataLoader that loads the training
    validation_loader -- The Pytorch DataLoader that loads the validation data
    test_loader -- The Pytorch DataLoader that loads the test data

    """
    train_index = round(X.shape[0]*indices[0])
    validation_index = round(X.shape[0]*(indices[0]+indices[1]))
    test_index = round(X.shape[0]*sum(indices))

    X_train = X[0:train_index]
    X_validation = X[train_index:validation_index]
    X_test = X[validation_index:test_index]
    y_train = y[0:train_index]
    y_validation = y[train_index:validation_index]
    y_test = y[validation_index:test_index]

    X_train_data, y_train_data = sliding_window(window_size_x, window_size_y, X_train.shape[0], X_train, y_train)
    X_validation_data, y_validation_data = sliding_window(window_size_x, window_size_y, X_validation.shape[0], X_validation, y_validation)
    X_test_data, y_test_data = sliding_window(window_size_x, window_size_y, X_test.shape[0], X_test, y_test)

    train_set = TensorDataset(X_train_data, y_train_data)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    validation_set = TensorDataset(X_validation_data, y_validation_data)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    test_set = TensorDataset(X_test_data, y_test_data)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, validation_loader, test_loader
#print(all_nodes)

A_hat = create_A_hat(edges)

def construct_agg_dict(cols, avg_cols):
    agg_dict = []
    for col in cols:
        if col in avg_cols:
            agg_dict.append(col)
    return agg_dict

# Step 1: Create timeDiscrepancy column ranging from '2015-01-01' to '2023-09-30'
start_date = '2015-01-02'
end_date = '2023-09-30'
time_discrepancy = pd.date_range(start=start_date, end=end_date, freq='D')

# List to store reshaped DataFrames
reshaped_dfs = []
ila_file_name = "ilaContractCountdown.csv"
df_ila = pd.read_csv(f"{con.pathData}/{ila_file_name}")    
df_ila['date'] = pd.to_datetime(df_ila['date'])

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

num_features=((shape_max + 3) // 4) * 4
# Step 2: Extract all column names from the occupation DataFrames
all_columns = set()
for node, info in nodes.items():
    df = info['occupation']
    all_columns.update(df.columns)

# Step 3: Construct the matrix for each node
X_data = []
y_data = []

for node, info in nodes.items():
    df = info['occupation']
    df.fillna(0)
    df['timeDiscrepancy'] = pd.to_datetime(df['timeDiscrepancy'])

# Set the "date" column as the index
    df.set_index('timeDiscrepancy', inplace=True)
    # Remove 'count' from the list of columns
    columns_without_count = [col for col in all_columns if col != 'count']
    # Extract data without the 'count' column
    X_matrix = np.zeros((len(time_discrepancy), num_features), dtype=np.float64)
    # Extract 'count' column data
    y_vector = np.zeros((len(time_discrepancy), 1), dtype=np.float64)
    # Fill X_matrix and y_vector with values from the corresponding DataFrame
    for i, date in enumerate(time_discrepancy):
        if date in df.index:
            num_cols = min(num_features, len(df.columns))  # Number of columns to consider
            X_matrix[i, :num_cols] = df.loc[date].values[:num_cols]  
            #X_matrix[i, :num_cols] = df_slice.values  # Assign values to X_matrix            
            #X_matrix[i, :len(df)] = df.loc[date].values
            # Set 'count' column in X_matrix to 0
            X_matrix[i, df.columns.get_loc('count')] = 0
            y_vector[i] = df.loc[date]['count']            
    #print(f"{node} and {info}: {y_vector}")
    X_data.append(X_matrix)
    y_data.append(y_vector)

# Convert lists to 3D numpy arrays
X_array = np.array(X_data)
X_array = np.transpose(X_array, (1, 0, 2))
y_array = np.array(y_data)
y_array = np.transpose(y_array, (1, 0, 2))[:, :, 0] #np.squeeze(y_array, axis=2) 
#print("------------------X_data----------------")
#print(X_data)
#print("------------------y_data----------------")
#print(y_data)

'''
with open(f'{con.pathOutput}/columns_info.txt', 'w') as f:
    sys.stdout = f  # Redirect standard output to the file
    nodes_df.info(verbose=True)
    sys.stdout = sys.__stdout__  # Reset standard output
    , 'coordinates'
'''

X_temp = torch.tensor(X_array, dtype = torch.float)
'''
original_shape = torch.Size([3195, 11, 97])
desired_shape = list(original_shape)

add = 3
for i in range(add):
# Desired shape of X
    print(i)
    desired_shape[-1] += 1  # Increment the last dimension by 1
# Create a new tensor with the desired shape filled with zeros
    X = torch.zeros(desired_shape)
    X[:, :, :-1] = X_temp
    X_temp = torch.tensor(X, dtype = torch.float)
#print(X_temp.shape)
#print(X.shape)
# Copy the values from the original tensor to the new tensor
#X[:, :, :-1] = X_temp
#X[:, :, 97] = X_temp

'''
X = X_temp
y = torch.tensor(y_array, dtype = torch.float)
#print("------------------X_temp----------------")
#print(X_temp)
#print("------------------y----------------")
#print(y)


#print(y)
print(f"y shape is {y.shape}")
#y = torch.transpose(y, 0, 1)
print(f"X shape is {X.shape}")


train_loader, validation_loader, test_loader = create_data_loaders(X, y, window_size_x = len_t_input, window_size_y = len_predictions, indices = [0.85, 0.05, 0.1], batch_size = batch_size)
#device = 'cpu'
torch.save(train_loader, f"{con.pathOutput}/train_loader.pt")
torch.save(validation_loader, f"{con.pathOutput}/validation_loader.pt")
torch.save(test_loader, f"{con.pathOutput}/test_loader.pt")
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = "cpu"
print(f"device is {device}")

network, optimizer, mean_loss, std_loss, val_mean_loss, val_std_loss = train_STGNN_network(train_loader, validation_loader, A_hat, num_features, device, len_t_input = len_t_input, len_predictions = len_predictions, epoch_number = epoch_number, hidden_size = hidden_size)
plt.figure(figsize=(12,7.5))
plt.plot(range(1, len(mean_loss)+1), mean_loss, lw=2, label='mean train RMSE', color='blue')
plt.fill_between(range(1, len(mean_loss)+1), mean_loss+std_loss, mean_loss-std_loss, facecolor='blue', alpha=0.2)
plt.plot(range(1, len(val_mean_loss)+1), val_mean_loss, lw=2, label='mean validation RMSE', color='orange')
plt.fill_between(range(1, len(val_mean_loss)+1), val_mean_loss+val_std_loss, val_mean_loss-val_std_loss, facecolor='orange', alpha=0.2)
plt.title(r'Mean and std. dev. of RMSE per epoch',fontsize=20)
plt.legend(fontsize=20)
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('RMSE',fontsize=20)
plt.savefig(f'{con.pathOutput}/model_loss1.png')
plt.show()
torch.save(network.state_dict(), f'{con.pathOutput}/myModel_gr.pt')
test_mean_loss, test_std_loss, outputs, targets = test_RMSE(network, A_hat, test_loader, num_features, device)
print(test_mean_loss)

k = 0

dataframes = []
dataframes_pred = []
for node, info in nodes.items():
    output_values = outputs[k].cpu().numpy()
    target_values = targets[k].cpu().numpy()
    node_df = pd.DataFrame([output_values], columns=[f'day{i}' for i in range(1, 31)])
    pred_df = pd.DataFrame([target_values], columns=[f'day{i}' for i in range(1, 31)])
        
    # Add a 'Node' column with the node name
    node_df['Node'] = node
    pred_df['Node'] = node
    # Set the 'Node' column as the index
    node_df = node_df.set_index('Node')    
    pred_df = pred_df.set_index('Node')    
    dataframes.append(node_df)
    dataframes_pred.append(pred_df)
    k += 1

result_df = pd.concat(dataframes)
result_df_pred = pd.concat(dataframes_pred)
result_df.reset_index(inplace=True)
result_df_pred.reset_index(inplace=True)

# Save the DataFrame to a CSV file
result_df.to_csv(f'{con.pathOutput}/prediction_gr_5.csv', index=False)
result_df_pred.to_csv(f'{con.pathOutput}/target_gr_5.csv', index=False)
