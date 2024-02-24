import Constants as con
# imports 
import pandas as pd
import os
import matplotlib.pyplot as plt # plotting
import numpy as np  # numercial operations
import seaborn as sns   #plotting
import scipy.stats as stats
import statsmodels.api as sm #statistical models (including regression)
import datetime
#from datetime import datetime #this is to parse (or interpret) the date time format in most files
from math import sin, cos, atan2, sqrt, degrees, radians, pi
from sklearn.metrics import mean_squared_log_error #evaluation metric
from sklearn.metrics import mean_squared_error , mean_absolute_percentage_error #evaluation metric
from sklearn.model_selection import train_test_split    #train test split

# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# ARIMA and SARIMA example
#from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#packages for decomposition
from statsmodels.tsa.seasonal import seasonal_decompose #seasonal decomposition
#from pmdarima import auto_arima #auto arima model
import warnings
warnings.filterwarnings('ignore')
# geo
from geopy.distance import geodesic
from geopy.distance import great_circle as distance
import geopy.point
import shapely
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points


def BerthVesselFileMerger(daily = False):
    # berth vessel file merger 
    ais_folder = f"/home/gridsan/{con.userName}/Coding/BerthOnly"
    files = os.listdir(ais_folder)    
    frames = []
    for f in files:
        ais_file = ais_folder + "/" + f
        if f.startswith(f"VesselOnBerth"):
            df = pd.read_csv(ais_file, error_bad_lines=False )
            frames.append(df)
    resulted = pd.concat(frames, ignore_index=True)
    resulted = resulted.sort_values(by=['nearestPort', 'IMO', 'VesselName', 'start_time'])
    
    resulted['time_diff'] = (pd.to_datetime(resulted["start_time"]).shift(-1) - pd.to_datetime(resulted["end_time"])).dt.total_seconds() / 3600
    resulted['group'] = (resulted['time_diff'] > 1).cumsum()

    last_row_mask = resulted.duplicated(['nearestPort', 'IMO', 'VesselName', 'group'], keep='last') | (resulted['group'] == resulted['group'].max())

# Group by 'Port', 'Vessel', and 'group', then aggregate 'start_time' and 'end_time'
    result = resulted[~last_row_mask].groupby(['nearestPort', 'IMO', 'VesselName', 'Length', 'Width', 'group']).agg({'start_time': 'min', 'end_time': 'max', 'TimeSpent': 'sum'}).reset_index()
    result.reset_index(inplace=True)  
    result.to_csv(f"{ais_folder}/ResultedVesselOnBerth.csv")
    return 0


def AnchorVesselFileMerger(daily = False):
    # anchorage vessel file merger
    ais_folder = f"/home/gridsan/{con.userName}/Coding/Anchor2"
    files = os.listdir(ais_folder)    
    frames = []
    for f in files:
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AnchorVessel") and f.endswith(f".csv"):
            df = pd.read_csv(ais_file, error_bad_lines=False )
            frames.append(df)
            
    resulted = pd.concat(frames, ignore_index=True)
    resulted = resulted.sort_values(by=['nearestPort', 'IMO', 'VesselName', 'start_time'])
    
    resulted['time_diff'] = (pd.to_datetime(resulted["start_time"]).shift(-1) - pd.to_datetime(resulted["end_time"])).dt.total_seconds() / 3600
    resulted['group'] = (resulted['time_diff'] > 1).cumsum()

    last_row_mask = resulted.duplicated(['nearestPort', 'IMO', 'VesselName', 'group'], keep='last') | (resulted['group'] == resulted['group'].max())

# Group by 'Port', 'Vessel', and 'group', then aggregate 'start_time' and 'end_time'
    result = resulted[~last_row_mask].groupby(['nearestPort', 'IMO', 'VesselName', 'group']).agg({'start_time': 'min', 'end_time': 'max', 'TimeSpent': 'sum'}).reset_index()
    result.reset_index(inplace=True)  
    result.to_csv(f"{ais_folder}/ResultedVesselAnchor.csv")
    return 0


def BerthStatsFileMerger(daily = False):
    # berth vessel file merger 
    ais_folder = f"/home/gridsan/{con.userName}/Coding/BerthOnly"
    files = os.listdir(ais_folder)    
    frames = []
    tdSuffix = ''
    if daily:
        tdSuffix = 'daily'
    for f in files:
        ais_file = ais_folder + "/" + f
        if f.startswith(f"BerthOnly") and f.endswith(f"{tdSuffix}.csv"):
            df = pd.read_csv(ais_file, error_bad_lines=False )
            frames.append(df)
    resulted = pd.concat(frames, ignore_index=True)
    resulted = resulted.sort_values(by=['timeDiscrepancy'])
    #resultNY = resulted[resulted['nearestPort'].str.startswith('NY_')]
    #resultNY.reset_index(inplace=True)  
    #resultNY.to_csv(f"{ais_folder}/ResultedBerthStatsNY.csv")
    #resultBoston = resulted[resulted['nearestPort'] == 'Boston']
    #resultBoston.reset_index(inplace=True)  
    #resultBoston.to_csv(f"{ais_folder}/ResultedBerthStatsBoston.csv")
    resulted.to_csv(f"{ais_folder}/ResultedBerthStats.csv")
    return 0

def AnchorStatsFileMerger(daily = False):
    # berth vessel file merger 
    ais_folder = f"/home/gridsan/{con.userName}/Coding/Anchor2"
    files = os.listdir(ais_folder)    
    framesBoston = []
    framesNY = []
    tdSuffix = ''
    if daily:
        tdSuffix = 'daily'    
    for f in files:
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AnchorStats") and f.endswith(f"{tdSuffix}.csv"):
            df = pd.read_csv(ais_file, error_bad_lines=False )
            framesBoston.append(df)
        #if f.startswith(f"NYAnchorStats") and f.endswith(f"{tdSuffix}.csv"):
        #    df = pd.read_csv(ais_file, error_bad_lines=False )
        #    framesNY.append(df)
            
    resultedBoston = pd.concat(framesBoston, ignore_index=True)
    #resultedNY = pd.concat(framesNY, ignore_index=True)
    resultedBoston = resultedBoston.sort_values(by='timeDiscrepancy')
    #resultedNY = resultedNY.sort_values(by='timeDiscrepancy')
    resultedBoston.reset_index(inplace=True)  
    #resultedNY.reset_index(inplace=True)  
    #resultedNY.to_csv(f"{ais_folder}/ResultedAnchorStatsNY{tdSuffix}.csv")
    resultedBoston.to_csv(f"{ais_folder}/ResultedAnchorStats{tdSuffix}.csv")

    return 0

def HarborStatsFileMerger(daily = False):
    # berth vessel file merger 
    ais_folder = f"/home/gridsan/{con.userName}/Coding/Harbour2"
    files = os.listdir(ais_folder)    
    framesBoston = []
    framesNY = []
    tdSuffix = ''
    if daily:
        tdSuffix = 'daily'    
    for f in files:
        ais_file = ais_folder + "/" + f
        if f.endswith(f".csv"):
            df = pd.read_csv(ais_file, error_bad_lines=False )
            framesBoston.append(df)
#        if f.startswith(f"NY") and f.endswith(f"{tdSuffix}.csv"):
#            df = pd.read_csv(ais_file, error_bad_lines=False )
#            framesNY.append(df)
            
    resultedBoston = pd.concat(framesBoston, ignore_index=True)
#    resultedNY = pd.concat(framesNY, ignore_index=True)
    resultedBoston = resultedBoston.sort_values(by='timeDiscrepancy')
#    resultedNY = resultedNY.sort_values(by='timeDiscrepancy')
    resultedBoston.reset_index(inplace=True)  
#    resultedNY.reset_index(inplace=True)  
#    resultedNY.to_csv(f"{ais_folder}/ResultedHarborStatsNY{tdSuffix}.csv")
    resultedBoston.to_csv(f"{ais_folder}/ResultedHarborStats{tdSuffix}.csv")

    return 0



BerthVesselFileMerger(True)
AnchorVesselFileMerger(True)

BerthStatsFileMerger(True)
AnchorStatsFileMerger(True)
HarborStatsFileMerger(True)