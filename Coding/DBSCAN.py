# imports 
from geopy.distance import great_circle
from shapely.geometry import Point
from shapely.geometry import MultiPoint
import geopandas
import pandas as pd
import os
import matplotlib.pyplot as plt # plotting
import numpy as np  # numercial operations
import seaborn as sns   #plotting
import scipy.stats as stats
import statsmodels.api as sm #statistical models (including regression)
import datetime
#from datetime import datetime #this is to parse (or interpret) the date time format in most files

from sklearn.metrics import mean_squared_log_error #evaluation metric
from sklearn.metrics import mean_squared_error , mean_absolute_percentage_error #evaluation metric
from sklearn.model_selection import train_test_split    #train test split
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

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
import geopy.distance


# AIS columns
# MMSI	BaseDateTime	LAT	LON	SOG	COG	Heading	VesselName	IMO	CallSign	VesselType	Status	Length	Width	Draft	Cargo	TransceiverClass


# definitions
ports = {'Boston, MA', 'Portland, ME', 'Newark, NJ', 'Portland, OR'}
portCodes = {'Boston, MA':'0401', 'Portland, OR' : '2904', 'Newark, NJ':'4601', 'Portland, ME' : '0101'}
#ports coordinates

portsCoordinates = {'0401':(42.340, -71.019), '4601': (40.667663996, -74.040666504), '0101': (43.680031, -70.310425), '2904':(45.523064, -122.676483)}
userName = "naristov"
#ports precision
pp = 25
portArea =400
#ports - nearest ports
portsNearest = {'0401':('4601','0101')}
waitCellRadius = 50
outerGridCellSize = 10
outerAreaDistance = 400
portRadius = 10 

def getDistanceMiles(fromPoint, toPoint):
    return geopy.distance.distance(fromPoint, toPoint).miles 

def getDistanceKm(fromPoint, toPoint):
    return geopy.distance.distance(fromPoint, toPoint).km

def logTime(lastTime , message):
    currTime = datetime.datetime.now().replace(microsecond=0) 
    print(currTime - lastTime , message)
    return currTime

def readAISData(year, month, shipList):
    ais_folder = f"/home/gridsan/{userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    year2 = year
    month2 = f"{(int(month) - 1):02}"
    ais_folder2 = ais_folder    
    files = os.listdir(ais_folder)    

    if month == "01":
        year2 = year - 1
        month2 = "12"
        ais_folder2 = f"/home/gridsan/{userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year2}"
        files = [*files, *os.listdir(ais_folder2)]

    frames = []
    waitingFrames = []
    #waitCellRadius = 50
    #outerGridCellSize = 10
    #outerAreaDistance = 400
    #portRadius = 10 
    #prepare boundaries for working with df
    kkk = 0
    currTime = datetime.datetime.now().replace(microsecond=0) 
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        ais_file2 = ais_folder2 + "/" + f
        if (f.startswith(f"AIS_{year}_{month}") or f.startswith(f"AIS_{year2}_{month2}")) and not f.startswith("AIS_2015_01_01"):
                currTime = logTime(currTime, f"file is {ais_file}")
            #try:
                try:
                    df = pd.read_csv(ais_file)
                except:
                    print(f"didn't find {ais_file}, trying {ais_file2}")
                    df = pd.read_csv(ais_file2, error_bad_lines=False )
                #filtering boston area roughly
                df = df[abs(df['LAT'] - portsCoordinates['0401'][0]) < 7 ]
                df = df[abs(df['LON'] - portsCoordinates['0401'][1]) < 7 ]
                
                df_waiting = df[abs(df['LON'] - portsCoordinates['0401'][1]) < 1 ]
                df_waiting = df_waiting[abs(df_waiting['LAT'] - portsCoordinates['0401'][0]) < 1 ]
                waitingFrames.append(df_waiting)
                #return df_waiting 
                df = df[df['MMSI'].isin(shipList)]
                
                if len(df) > 0:
                    #print(f" df is not empty for {ais_file}")
                    for p in portsCoordinates:      
                        if p == '0401':
                            df[p] = df.apply(lambda x: getDistanceMiles((x['LAT'], x['LON']),portsCoordinates[p]), 
                                             axis=1)
                #  looking for a square algorithm (should it be number or 2 numbers???)
                #only Boston for now
                
                    df['outerCell'] = df.apply(lambda x: findOuterCell((x['LAT'], x['LON']),
                                                                       portsCoordinates['0401'], 
                                                                       outerAreaDistance, 
                                                                       outerGridCellSize), axis=1)
                    df['waitZone'] = df.apply(lambda x: pointIsIn((x['LAT'], x['LON']), getAreaBoundaries(getWaitCells(portsCoordinates['0401'])[0],25)), axis=1)
                    df['BostonPortZone'] = df.apply(lambda x: pointIsIn((x['LAT'], x['LON']), getAreaBoundaries(portsCoordinates['0401'],5)), axis=1)
                
                    frames.append(df)
                else: 
                    print(f"is epmpty {ais_file}" )
        kkk = kkk + 1
            #except:
            #    print(f"failed zip file is {f}")
    currTime = logTime(currTime, "finished files")
    df_full = pd.concat(frames, ignore_index=True)
    dfWaiting = pd.concat(waitingFrames, ignore_index=True)
    # Take all bostonies
    df_in_port = df_full[df_full["BostonPortZone"]].copy()
    df_waitZone = df_full[df_full["waitZone"]].copy()    
    currTime = logTime(currTime, "starting updating frames")
    df_first = df_full[df_full["outerCell"] < 64000]
    df_first = df_first.sort_values('BaseDateTime').groupby(["MMSI","VesselType", "VesselName", "IMO", "CallSign"]).apply(pd.DataFrame.head, n=1).reset_index(drop=True)

    df_in_port["YearMonth"] = pd.to_datetime(df_in_port["BaseDateTime"]).dt.strftime('%Y-%m')
    df_in_port = df_in_port[df_in_port["YearMonth"]==f"{year}-{month}"]
    df_in_port = df_in_port.groupby(["MMSI","VesselType", "VesselName", "IMO", "CallSign"]).agg(
        start_time =("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
    currTime = logTime(currTime, "df_in_ports updates finished")
    #first date appeared in dataset
    df_in_port = pd.merge(df_in_port, df_first[["MMSI","VesselType", "VesselName", "IMO", "CallSign", "BaseDateTime","outerCell"]], left_on=["MMSI","VesselType", "VesselName", "IMO", "CallSign"],
                              right_on=["MMSI","VesselType", "VesselName", "IMO", "CallSign"], how='left',
                         suffixes=('', '_firstEntry'))
    currTime = logTime(currTime, "merge of first date performed")
    #first date appeared in waitZone
    df_waitZone = df_waitZone.sort_values('BaseDateTime').groupby(["MMSI","VesselType", "VesselName", "IMO", "CallSign"]).apply(pd.DataFrame.head, n=1).reset_index(drop=True)
    currTime = logTime(currTime, "df_waitZone applied")
    df_in_port = pd.merge(df_in_port, df_waitZone[["MMSI","VesselType", "VesselName", "IMO", "CallSign","BaseDateTime"]], left_on=["MMSI","VesselType", "VesselName", "IMO", "CallSign"],
                              right_on=["MMSI","VesselType", "VesselName", "IMO", "CallSign"], how='left',
                         suffixes=('', '_waitEntry'))
    currTime = logTime(currTime, "df_port calculated")
    dfWaiting["HOURS"] = pd.to_datetime(dfWaiting["BaseDateTime"]).dt.strftime('%Y-%m-%dT%H')
    currTime = logTime(currTime, "df_waiting passed")
    dfWaitingGrouped = dfWaiting.groupby(["MMSI","HOURS","VesselType", "VesselName", "IMO", "CallSign"]).agg(start_time = ("BaseDateTime", np.min), end_time =("BaseDateTime", np.max))
    currTime = logTime(currTime, "df_waiting grouped")
    dfWaitingGrouped["percent"] = (pd.to_datetime(dfWaitingGrouped["end_time"]).dt.strftime('%M').astype(int) - pd.to_datetime(dfWaitingGrouped["start_time"]).dt.strftime('%M').astype(int))/60
    currTime = logTime(currTime, "percent calculated")
    dfWaitingGrouped.reset_index(inplace=True)
    currTime = logTime(currTime, "index reseted")
    dfWaitingGrouped = dfWaitingGrouped.groupby(["HOURS","VesselType"]).agg(count = ("VesselName", np.size))
    currTime = logTime(currTime, "grouped")
    dfWaitingGrouped.reset_index(inplace=True)      
    #  number of ships by types there atm 
    #number of ships inside the port by types
    currTime = logTime(currTime, "end")
    
    return df_in_port, dfWaitingGrouped, dfWaiting, df_waitZone, df_full

def getShipList():
    return pd.read_csv(f"/home/gridsan/{userName}/shipList.csv")

def findOuterCell(point, centralPoint, radiusMi, outerGridCellSize):
    getLatDistance = point[0] - centralPoint[0]
    getLongDistance = point[1] - centralPoint[1]
    first_part = int(((radiusMi - getDistanceMiles((centralPoint[0], point[1]), centralPoint))/outerGridCellSize )//1) - 1 
    second_part = int (2 * radiusMi / outerGridCellSize) * (int(((radiusMi - getDistanceMiles((point[0], centralPoint[1]), centralPoint))/outerGridCellSize )//1) - 1)
    dist = first_part + second_part
    if dist > (int (2 * radiusMi / outerGridCellSize))**2 or first_part < 0 or second_part < 0:
        return 100000
    return dist

def getAreaBoundaries(centralPoint, radiusMi = 400):    
    maxLong = geopy.distance.distance(miles=radiusMi).destination(centralPoint, bearing=90).longitude #East
    minLong = geopy.distance.distance(miles=radiusMi).destination(centralPoint, bearing=270).longitude  #West
    maxLat = geopy.distance.distance(miles=radiusMi).destination(centralPoint, bearing=0).latitude  #North
    minLat = geopy.distance.distance(miles=radiusMi).destination(centralPoint, bearing=180).latitude #South
    return (minLat, maxLat, minLong, maxLong)

def pointIsIn(point, boundaries):
    return (point[0] <= boundaries[1]) & (point[0] > boundaries[0]) & (point[1] <= boundaries[3]) & (point[1] > boundaries[2])    

def getOuterCells(centralPoint, radiusMi = 400, gridSize=10):
    minLat, maxLat, minLong, maxLong = getAreaBoundaries(centralPoint, radiusMi)

    maxLong = geopy.distance.distance(miles=gridSize/2).destination((minLat,maxLong), bearing=270).longitude #East
    minLong = geopy.distance.distance(miles=gridSize/2).destination((minLat,minLong), bearing=90).longitude  #West
    maxLat = geopy.distance.distance(miles=gridSize/2).destination((maxLat,minLong), bearing=180).latitude  #North
    minLat = geopy.distance.distance(miles=gridSize/2).destination((minLat,minLong), bearing=0).latitude #South
    
    OuterCells = {}
    k = int(2*radiusMi/gridSize)
    for num in range(k-1):
        OuterCells[num] = (minLat, geopy.distance.distance(miles=(num) * gridSize/2).destination((minLat,minLong), bearing=90))  # north going to east
        OuterCells[4 * (k - 1) - num - 1] = ( geopy.distance.distance(miles=(num + 1) * gridSize/2).destination((minLat,minLong), bearing=180), minLong)   # west go to south
        OuterCells[3 * (k - 1) - num - 1] = (maxLat, geopy.distance.distance(miles=(num + 1) * gridSize/2).destination((minLat,minLong), bearing=90))   # south go to east
        OuterCells[k - 1 + num] = ( geopy.distance.distance(miles=(num) * gridSize/2).destination((minLat,minLong), bearing=180), maxLong)   # go to # east go to south
        
    return OuterCells

def getWaitCells(centralPoint, radiusMi = 50, gridSize = 5):
    return {0 : centralPoint}

def run_month(year,month):
    try:
        print(f"running {month} and year is {year}")
        df_port, df_waiting, df_waiting_src, df_waitZone, df_full = readAISData(year,month,getShipList()["MMSI"])
        print(f"saving {month} and year is {year}, the size are {len(df_port)} and  {len(df_waiting)}")
        df_port.to_csv(f"/home/gridsan/{userName}/RunAIS/BostonAreaAISResult/port{year}{month}.csv")
        df_waiting.to_csv(f"/home/gridsan/{userName}/RunAIS/BostonAreaAISResult/wait{year}{month}.csv")            
    except Exception as error:
        print(f"An exception occurred for {year} and {month}:", error)
    return 0


def runDBSCAN(year, month, shipList, part):
    ais_folder = f"/home/gridsan/{userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)   
    frames = []
    fileList = []
    for k in range(5):
        fileList.append(f"AIS_{year}_{month}_{(k + 1 + part * 5):02}.zip")
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f in fileList:
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df = df[df['MMSI'].isin(shipList)]
            frames.append(df)

    df_full0 = pd.concat(frames, ignore_index=True)
    
    print(f"df count is {len(df_full0)}  for {year} and {month} and part {part}")
    coord0 = df_full0[["LAT", "LON"]]
    db0 = DBSCAN(eps=10/6371., min_samples=100,algorithm='ball_tree', metric='haversine').fit(np.radians(coord0))
    cluster_labels0 = db0.labels_
    num_clusters0 = len(set(cluster_labels0))
    clusters0 = pd.Series([coord0[cluster_labels0 == n] for n in range(num_clusters0)])
    
    kd = pd.DataFrame(columns=list('xy'))
    i = 0
    for t in clusters0:
        s = geopandas.GeoSeries(MultiPoint(list (zip (t.LAT,t.LON))))
        k = s.centroid[0]
        if not k.is_empty:
            kd.loc[i] = [k.x, k.y]
            i+=1
    kd.to_csv(f"/home/gridsan/{userName}/RunAIS/BostonAreaAISResult/dbscan_{year}_{month}_{part}.csv")       

def runDBSCAN_speedlimit(year, month, shipList, part, speedLimit = 5, totalParts = 5):
    ais_folder = f"/home/gridsan/{userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)   
    frames = []
    fileList = []
    for k in range(totalParts):
        fileList.append(f"AIS_{year}_{month}_{(k + 1 + part * totalParts):02}.zip")
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f in fileList:
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df = df[df['MMSI'].isin(shipList)]
            df = df[df['SOG'] < speedLimit]
            frames.append(df)

    df_full0 = pd.concat(frames, ignore_index=True)
    
    print(f"df count is {len(df_full0)}  for {year} and {month} and part {part}")
    coord0 = df_full0[["LAT", "LON"]]
    db0 = DBSCAN(eps=2/6371., min_samples=200,algorithm='ball_tree', metric='haversine').fit(np.radians(coord0))
    cluster_labels0 = db0.labels_
    num_clusters0 = len(set(cluster_labels0))
    clusters0 = pd.Series([coord0[cluster_labels0 == n] for n in range(num_clusters0)])
    
    kd = pd.DataFrame(columns=list('xy'))
    i = 0
    for t in clusters0:
        s = geopandas.GeoSeries(MultiPoint(list (zip (t.LAT,t.LON))))
        k = s.centroid[0]
        if not k.is_empty:
            kd.loc[i] = [k.x, k.y]
            i+=1
    kd.to_csv(f"/home/gridsan/{userName}/RunAIS/BostonAreaAISResult/dbscan_{year}_{month}_{part}.csv")       

            
def run_monthDBSCAN(year,month, part):
    try:
        print(f"running {month} and year is {year}")
        #runDBSCAN(year,month,getShipList()["MMSI"], part)
        runDBSCAN_speedlimit(year,month,getShipList()["MMSI"], part, 2, 7)
    except Exception as error:
        print(f"An exception occurred for {year} and {month} and part # {part}:", error)
    return 0
