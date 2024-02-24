# imports 
import Constants as con
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

def getShipHistory():
    
    #read berth file fo the month (?)
    berthFolder = f"/home/gridsan/{con.userName}/Coding/BerthOnly"
    berth = pd.read_csv(f"{berthFolder}/ResultedVesselOnBerth.csv")
    # read anchor file for the month and prev month (?)
    anchorFolder = f"/home/gridsan/{con.userName}/Coding/Anchor"
    anchor = pd.read_csv(f"{anchorFolder}/ResultedVesselAnchor.csv")
        
    #filtering now for NY and Boston
    berth = berth[(berth["nearestPort"] == 'Boston') | (berth["nearestPort"].str.startswith(f"NY_"))] 
    berth['groupedPort'] = np.where(berth['nearestPort'].str.startswith('NY'), 'NY', berth['nearestPort']) 
        
    #join anchor 
    berth['start_time'] = pd.to_datetime(berth['start_time'])
    berth['end_time'] = pd.to_datetime(berth['end_time'])
    berth.reset_index(inplace=True)
    anchor['start_time'] = pd.to_datetime(anchor['start_time'])
    anchor['end_time'] = pd.to_datetime(anchor['end_time'])

    berthAndAnchor = pd.merge(berth, anchor, left_on=['groupedPort','MMSI', 'VesselName'], right_on = ['nearestPort','MMSI', 'VesselName'], how='left', suffixes=('', '_anchor'))
    berthAndAnchor['Date_difference'] = (berthAndAnchor['start_time'] - berthAndAnchor['end_time_anchor']).dt.total_seconds() / 3600
# Filter rows based on the condition (date difference <= 6 hours)
    berthAndAnchor = berthAndAnchor[((berthAndAnchor['end_time_anchor'] >= berthAndAnchor['start_time'] - pd.Timedelta(hours=6)) & (berthAndAnchor['end_time_anchor'] < berthAndAnchor['start_time']))]

    finalResult = pd.merge(berth, berthAndAnchor, how='left', on=['index'], suffixes=('', '_anchor'))

    finalResult.to_csv(f"/home/gridsan/{con.userName}/Coding/IDK/BerthMergeAnchor.csv")
    # read AIS files for a few days before as a day and poistion of entry.
    
    #probably, need to make loop, but don't need atm, I think.
    
    return 0

    
def getBearing(pointA, pointB):
    start_lat, start_lon = pointA
    end_lat, end_lon = pointB

    start_lat = radians(start_lat)
    start_lon = radians(start_lon)
    end_lat = radians(end_lat)
    end_lon = radians(end_lon)

    d_lon = end_lon - start_lon

    x = sin(d_lon) * cos(end_lat)
    y = cos(start_lat) * sin(end_lat) - (sin(start_lat) * cos(end_lat) * cos(d_lon))

    initial_bearing = atan2(x, y)

    # Normalize the angle to a compass bearing
    compass_bearing = (degrees(initial_bearing) + 360) % 360
    
    if compass_bearing >= 180:
        adds_bearing = compass_bearing
        compass_bearing = compass_bearing - 180
    else:
        adds_bearing = compass_bearing + 180
    
    return compass_bearing, adds_bearing


def getDistanceMiles(fromPoint, toPoint):
    return geopy.distance.distance(fromPoint, toPoint).miles 

def getDistanceKm(fromPoint, toPoint):
    return geopy.distance.distance(fromPoint, toPoint).km

def getDistanceToLineM(point, line):
    a, b = nearest_points(line, Point(point))
    return getDistanceKm(a.coords[0], b.coords[0]) * 1000

def getClosestPort(point):
    minDistance = 3000000
    portName = 'NONE'
    bearing = (-180, -180)
    for port in PortCoordinates:
        for i in range(len(PortCoordinates[port]) - 1):
            dis = getDistanceToLineM(point, LineString([PortCoordinates[port][i], PortCoordinates[port][i + 1]]) )
            if dis < minDistance:                
                minDistance = dis
                portName = port
                bearing = getBearing(PortCoordinates[port][i], PortCoordinates[port][i + 1])
    return portName, minDistance, bearing[0], bearing[1]



def getShipsData(year, month, ships, SOGLimit = 1):
    ais_folder = f"/home/gridsan/{userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)    
    frames = []
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_{year}_{month}"):
            print(f"processing file {f} started on {datetime.datetime.now().replace(microsecond=0)}" )
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df = df[df['MMSI'].isin(ships["MMSI"])]
            df = df[df['SOG'] < SOGLimit]  # want only near terminal
            #results = df.apply(lambda x: getClosestPort((x['LAT'], x['LON'])), axis=1)
            #print(results)
            df[["nearestPort", "nearestPortDistance", "nearestPortBearingMin", "nearestPortBearingMax"]] = df.apply(lambda x: getClosestPort((x['LAT'], x['LON'])), 
                                             axis=1, result_type='expand')
            
            #df = df[df["nearestPortDistance"] < df["Width"] if df["Width"] > 0 else 0]
            #df["head_diff"] = df.apply(lambda x: abs(x["Heading"] - x["nearestPortBearingMin"]))
            #df = df[df["head_diff"] < 3 or (df["head_diff"] < 183 and df["head_diff"] < 177)]
            if len(df) > 0:
                frames.append(df)
            #    break
            '''                
                dfGrouped = df.groupby(["MMSI","VesselType", "VesselName", "IMO", "CallSign"]).agg(Length = ("Length", np.min), Width =("Width", np.min))
                dfGrouped.reset_index(inplace=True)
                shipsLeftNew = pd.merge(shipsLeft, dfGrouped[["MMSI", "Length", "Width"]], left_on=["MMSI"],
                                  right_on=["MMSI"],
                             suffixes=('', '_extra'))
                if len(shipsLeftNew) > 0:
                    frames.append(shipsLeftNew)
                shipsLeft = shipsLeft[~shipsLeft["MMSI"].isin(shipsLeftNew["MMSI"])]
                print(f"left {len(shipsLeft)}");
                if len(shipsLeft) == 0:
                    break
               # group by df by fields, join to shipsLeft and 
            '''            
    resulted = pd.concat(frames, ignore_index=True)
    return resulted

def getHarbourData(year, month):
    ais_folder = f"/home/gridsan/{userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)    
    framesNY = []
    framesBoston = []
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_{year}_{month}"):
            print(f"processing file {f} started on {datetime.datetime.now().replace(microsecond=0)}" )
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df["Hour"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime('%Y-%m-%dT%H')
            
            dfBoston = df[df["LAT"] < 42.36]
            dfBoston = dfBoston[dfBoston["LAT"] > 42.30]
            dfBoston = dfBoston[dfBoston["LON"] > -71.1]
            dfBoston = dfBoston[dfBoston["LON"] < -70.8]

            dfNY = df[df["LAT"] < 40.75]
            dfNY = dfNY[dfNY["LAT"] > 40.45]
            dfNY = dfNY[dfNY["LON"] > -74.3]
            dfNY = dfNY[dfNY["LON"] < -73.9]

            dfBoston[f"BostonHarbour"] = dfBoston.apply(lambda x: BostonPortAquatory.contains(Point(x['LAT'], x['LON'])), 
                                                        axis=1)            
            dfNY[f"NYHarbour"] = dfNY.apply(lambda x: NYPortAquatory.contains(Point(x['LAT'], x['LON'])), 
                                             axis=1)            
            dfBoston = dfBoston[dfBoston[f"BostonHarbour"]]
            dfNY = dfNY[dfNY[f"NYHarbour"]]
            if len(dfBoston) > 0:
                dfBostonGrouped = dfBoston.groupby(["Hour","VesselType", "MMSI"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean))
                dfBostonGrouped.reset_index(inplace=True)
                dfBostonGrouped = dfBostonGrouped.groupby(["Hour"]).agg(count = ("MMSI", np.size), Length = ("Length", np.mean), Width =("Width", np.mean))
                dfBostonGrouped.reset_index(inplace=True)
                if len(dfBostonGrouped) > 0:
                    framesBoston.append(dfBostonGrouped)                

            if len(dfNY)>0:
                dfNYGrouped = dfNY.groupby(["Hour","VesselType", "MMSI"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean))
                dfNYGrouped.reset_index(inplace=True)
                dfNYGrouped = dfNYGrouped.groupby(["Hour"]).agg(count = ("MMSI", np.size), Length = ("Length", np.mean), Width =("Width", np.mean))
                dfNYGrouped.reset_index(inplace=True)

                if len(dfNYGrouped) > 0:
                    framesNY.append(dfNYGrouped)
    resNY = pd.concat(framesNY, ignore_index=True)
    resBoston = pd.concat(framesBoston, ignore_index=True)
    resNY.to_csv(f"/home/gridsan/{userName}/Coding/Harbour/NY{year}{month}.csv")
    resBoston.to_csv(f"/home/gridsan/{userName}/Coding/Harbour/Boston{year}{month}.csv")
    return 0

def runGetHarbourData(year, month):
    getHarbourData(year, month)
    return 0

def getShipList():
    return pd.read_csv(f"/home/gridsan/{userName}/shipList.csv")

def getShipsDataFile():
    return pd.read_csv(f"/home/gridsan/{userName}/shipsData.csv")

def saveResultedDataFile(df, year, month):
    df.to_csv(f"/home/gridsan/{userName}/Coding/PortStatsInitial/berthInitial{year}{month}.csv")

def runInitial(year,month):
    saveResultedDataFile(getShipsData(year, month, getShipsDataFile()), year, month)
    return 0


def runFilteringBerths(year, month):
    filteringBerth(year, month)
    return 0

def runAnchorageAreas(year, month):
    getAnchorageAreas(year, month)
    return 0

def filteringBerth(year, month):
    df = pd.read_csv(f"/home/gridsan/{userName}/Coding/PortStatsInitial/berthInitial{year}{month}.csv")
    
    df = df[df["nearestPortDistance"] < 60]
    df["Hour"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime('%Y-%m-%dT%H')
    #df = df[df["nearestPortDistance"] > 1000]
    #df = df[df["nearestPortDistance"] < 350000]
    dfPortStatsGrouped = df.groupby(["Hour", "MMSI", "nearestPort"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean))
    dfPortStatsGrouped.reset_index(inplace=True)
    dfPortStatsGrouped = dfPortStatsGrouped.groupby(["Hour", "nearestPort"]).agg(count = ("MMSI", np.size), Length = ("Length", np.sum), Width =("Width", np.mean))
    dfPortStatsGrouped.reset_index(inplace=True)
    
    dfPortVesselStats = df.groupby(["MMSI", "VesselName", "nearestPort"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean), start_time = ("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
    dfPortVesselStats.reset_index(inplace=True)
    dfPortVesselStats["TimeSpent"] = dfPortVesselStats.apply(lambda x: (pd.to_datetime(x["end_time"])- pd.to_datetime(x["start_time"])).total_seconds()/3600, axis=1)
    
    '''
    df["head_diff"] = df["Heading"] % 180 - df["nearestPortBearingMin"] 
    df["head_diff2"] = df["Heading"] % 180 - df["nearestPortBearingMax"] 
    df["head_diff3"] = df["Heading"] % 180 - df["nearestPortBearingMax"] % 180 
    df["head_diff"] = df.apply(lambda x: abs(x["head_diff"]), axis=1)
    df["head_diff2"] = df.apply(lambda x: abs(x["head_diff2"]), axis=1)
    df["head_diff3"] = df.apply(lambda x: abs(x["head_diff3"]), axis=1)
    
    df = df[ (df["head_diff"] < 12 ) | (df["head_diff2"]  < 12) | ( df["head_diff3"] < 12)] 
    '''
    dfPortStatsGrouped.to_csv(f"/home/gridsan/{userName}/Coding/BerthOnly/BerthOnly{year}{month}.csv")
    dfPortVesselStats.to_csv(f"/home/gridsan/{userName}/Coding/BerthOnly/VesselOnBerth{year}{month}.csv")
    return 0

def getAnchorageAreas(year, month):
    ais_folder = f"/home/gridsan/{userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)
    ships = getShipList()
    framesNY = []
    framesBoston = []
    framesVesselBoston = []
    framesVesselNY = []
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_{year}_{month}"):
            print(f"processing file {f} started on {datetime.datetime.now().replace(microsecond=0)}" )
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df = df[df['MMSI'].isin(ships["MMSI"])]
            df["Hour"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime('%Y-%m-%dT%H')
            
            dfBoston = df[df["LAT"] < 42.45]
            dfBoston = dfBoston[dfBoston["LAT"] > 42.40]
            dfBoston = dfBoston[dfBoston["LON"] > -70.85]
            dfBoston = dfBoston[dfBoston["LON"] < -70.65]

            dfNY = df[df["LAT"] < 40.52]
            dfNY = dfNY[dfNY["LAT"] > 40.40]
            dfNY = dfNY[dfNY["LON"] > -73.75]
            dfNY = dfNY[dfNY["LON"] < -73.45]

            dfBoston[f"BostonAnchor"] = dfBoston.apply(lambda x: BostonPortAnchorageArea.contains(Point(x['LAT'], x['LON'])), 
                                                        axis=1)            
            dfNY[f"NYAnchor"] = dfNY.apply(lambda x: NYPortAnchorageArea.contains(Point(x['LAT'], x['LON'])), 
                                             axis=1)            
            dfBoston = dfBoston[dfBoston[f"BostonAnchor"]]
            dfNY = dfNY[dfNY[f"NYAnchor"]]
            if len(dfBoston) > 0:
                dfBostonGrouped = dfBoston.groupby(["Hour","VesselType", "MMSI"]).agg(Length = ("Length", np.mean))
                dfBostonGrouped.reset_index(inplace=True)
                dfBostonGrouped = dfBostonGrouped.groupby(["Hour"]).agg(count = ("MMSI", np.size))
                dfBostonGrouped.reset_index(inplace=True)
                if len(dfBostonGrouped) > 0:
                    framesBoston.append(dfBostonGrouped)                
                dfBostonVesselGrouped = dfBoston.groupby(["MMSI", "VesselName"]).agg(start_time = ("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
                dfBostonVesselGrouped.reset_index(inplace=True)
                dfBostonVesselGrouped["TimeSpent"] = dfBostonVesselGrouped.apply(lambda x: (pd.to_datetime(x["end_time"])- pd.to_datetime(x["start_time"])).total_seconds()/3600, axis=1)
                if len(dfBostonVesselGrouped) > 0:
                    framesVesselBoston.append(dfBostonVesselGrouped)                

            if len(dfNY)>0:
                dfNYGrouped = dfNY.groupby(["Hour","VesselType", "MMSI"]).agg(Length = ("Length", np.mean))
                dfNYGrouped.reset_index(inplace=True)
                dfNYGrouped = dfNYGrouped.groupby(["Hour"]).agg(count = ("MMSI", np.size))
                dfNYGrouped.reset_index(inplace=True)
                if len(dfNYGrouped) > 0:
                    framesNY.append(dfNYGrouped)     
                dfNYVesselGrouped = dfNY.groupby(["MMSI", "VesselName"]).agg(start_time = ("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
                dfNYVesselGrouped.reset_index(inplace=True)
                dfNYVesselGrouped["TimeSpent"] = dfNYVesselGrouped.apply(lambda x: (pd.to_datetime(x["end_time"])- pd.to_datetime(x["start_time"])).total_seconds()/3600, axis=1)
                if len(dfNYVesselGrouped) > 0:
                    framesVesselNY.append(dfNYVesselGrouped)  
    if len(framesNY) > 0:
        resNY = pd.concat(framesNY, ignore_index=True)
        resNY.to_csv(f"/home/gridsan/{userName}/Coding/Anchor/NYAnchorStats{year}{month}.csv")
    if len(framesVesselNY) > 0:
        resVNY = pd.concat(framesVesselNY, ignore_index=True)
        resVNY.to_csv(f"/home/gridsan/{userName}/Coding/Anchor/NYAnchorVessels{year}{month}.csv")
    if len(framesBoston) > 0:
        resBoston = pd.concat(framesBoston, ignore_index=True)
        resBoston.to_csv(f"/home/gridsan/{userName}/Coding/Anchor/BostonAnchorStats{year}{month}.csv")
    if len(framesVesselBoston) > 0:
        resVBoston = pd.concat(framesVesselBoston, ignore_index=True)
        resVBoston.to_csv(f"/home/gridsan/{userName}/Coding/Anchor/BostonAnchorVessels{year}{month}.csv")
    
    return 0

getShipHistory()