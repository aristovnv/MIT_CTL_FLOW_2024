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
    for port in con.PortCoordinates:
        for i in range(len(con.PortCoordinates[port]) - 1):
            dis = getDistanceToLineM(point, LineString([con.PortCoordinates[port][i], con.PortCoordinates[port][i + 1]]) )
            if dis < minDistance:                
                minDistance = dis
                portName = port
                bearing = getBearing(con.PortCoordinates[port][i], con.PortCoordinates[port][i + 1])
    return portName, minDistance, bearing[0], bearing[1]
            


def getShipsData(year, month, ships, SOGLimit = 1):
    ais_folder = f"/home/gridsan/{con.userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)    
    frames = []
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_{year}_{month}"):
            print(f"processing file {f} started on {datetime.datetime.now().replace(microsecond=0)}" )
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df = df[df['IMO'].isin(ships["IMO"])]
            df = df[df['SOG'] < SOGLimit]  # want only near terminal
            #results = df.apply(lambda x: getClosestPort((x['LAT'], x['LON'])), axis=1)
            #print(results)            
            df[["nearestPort", "nearestPortDistance", "nearestPortBearingMin", "nearestPortBearingMax"]] = df.apply(lambda x: getClosestPort((x['LAT'], x['LON'])), 
                                             axis=1, result_type='expand')
            
            df = pd.merge(df, ships[["IMO", "Length", "Width"]], on='IMO', how='left', suffixes=('', '_shipsdata'))
            df["Length"] = df["Length_shipsdata"]  
            df["Width"] = df["Width_shipsdata"]  
            df.drop(['Length_shipsdata', 'Width_shipsdata'],axis=1, inplace=True)            
            if len(df) > 0:
                frames.append(df)
            #    break
    resulted = pd.concat(frames, ignore_index=True)
    return resulted

def getShipsData_daily(df, ships, SOGLimit = 1):
    df = df[df['IMO'].isin(ships["IMO"])]
    df = df[df['SOG'] < SOGLimit]  # want only near terminal
    df[["nearestPort", "nearestPortDistance", "nearestPortBearingMin", "nearestPortBearingMax"]] = df.apply(lambda x: getClosestPort((x['LAT'], x['LON'])), 
                                             axis=1, result_type='expand')
    df = pd.merge(df, ships[["IMO", "Length", "Width"]], on='IMO', how='left', suffixes=('', '_shipsdata'))
    df["Length"] = df["Length_shipsdata"]  
    df["Width"] = df["Width_shipsdata"]  
    df.drop(['Length_shipsdata', 'Width_shipsdata'],axis=1, inplace=True)            
    return df


def getHarbourData(year, month, daily = False):
    ais_folder = f"/home/gridsan/{con.userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)
    timePattern = '%Y-%m-%dT%H'
    tdSuffix = ''
    if daily:
        timePattern = '%Y-%m-%d'
        tdSuffix = 'daily'
    framesNY = []
    framesBoston = []
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_{year}_{month}"):
            print(f"processing file {f} started on {datetime.datetime.now().replace(microsecond=0)}" )
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df["timeDiscrepancy"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime(timePattern)
            
            dfBoston = df[df["LAT"] < 42.36]
            dfBoston = dfBoston[dfBoston["LAT"] > 42.30]
            dfBoston = dfBoston[dfBoston["LON"] > -71.1]
            dfBoston = dfBoston[dfBoston["LON"] < -70.8]

            dfNY = df[df["LAT"] < 40.75]
            dfNY = dfNY[dfNY["LAT"] > 40.45]
            dfNY = dfNY[dfNY["LON"] > -74.3]
            dfNY = dfNY[dfNY["LON"] < -73.9]

            dfBoston[f"BostonHarbour"] = dfBoston.apply(lambda x: con.BostonPortAquatory.contains(Point(x['LAT'], x['LON'])), 
                                                        axis=1)            
            dfNY[f"NYHarbour"] = dfNY.apply(lambda x: con.NYPortAquatory.contains(Point(x['LAT'], x['LON'])), 
                                             axis=1)            
            dfBoston = dfBoston[dfBoston[f"BostonHarbour"]]
            dfNY = dfNY[dfNY[f"NYHarbour"]]
            if len(dfBoston) > 0:
                dfBostonGrouped = dfBoston.groupby(["timeDiscrepancy","VesselType", "IMO"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean))
                dfBostonGrouped.reset_index(inplace=True)
                dfBostonGrouped = dfBostonGrouped.groupby(["timeDiscrepancy"]).agg(count = ("IMO", np.size), Length = ("Length", np.mean), Width =("Width", np.mean))
                dfBostonGrouped.reset_index(inplace=True)
                if len(dfBostonGrouped) > 0:
                    framesBoston.append(dfBostonGrouped)                

            if len(dfNY)>0:
                dfNYGrouped = dfNY.groupby(["timeDiscrepancy","VesselType", "IMO"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean))
                dfNYGrouped.reset_index(inplace=True)
                dfNYGrouped = dfNYGrouped.groupby(["timeDiscrepancy"]).agg(count = ("IMO", np.size), Length = ("Length", np.mean), Width =("Width", np.mean))
                dfNYGrouped.reset_index(inplace=True)

                if len(dfNYGrouped) > 0:
                    framesNY.append(dfNYGrouped)
    resNY = pd.concat(framesNY, ignore_index=True)
    resBoston = pd.concat(framesBoston, ignore_index=True)
    resNY.to_csv(f"/home/gridsan/{con.userName}/Coding/Harbour/NY{year}{month}{tdSuffix}.csv")
    resBoston.to_csv(f"/home/gridsan/{con.userName}/Coding/Harbour/Boston{year}{month}{tdSuffix}.csv")
    return 0

def getHarbourData_daily(df, daily = False):
    timePattern = '%Y-%m-%dT%H'
    dfHarbourStats = pd.DataFrame()
    tdSuffix = ''
    if daily:
        timePattern = '%Y-%m-%d'
        tdSuffix = 'daily'
    framesHarbour = []
    df["timeDiscrepancy"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime(timePattern)
    for harbour in con.portsList:       
        df_harbour = df[df["LAT"] < con.PortLimitsCoordinates[harbour]['maxLAT']]
        df_harbour = df_harbour[df_harbour["LAT"] > con.PortLimitsCoordinates[harbour]['minLAT']]
        df_harbour = df_harbour[df_harbour["LON"] > con.PortLimitsCoordinates[harbour]['minLON']]
        df_harbour = df_harbour[df_harbour["LON"] < con.PortLimitsCoordinates[harbour]['maxLON']]
        df_harbour[f"Harbour"] = df_harbour.apply(lambda x: con.PortHarbour[harbour].contains(Point(x['LAT'], x['LON'])), axis=1)            
        df_harbour = df_harbour[df_harbour[f"Harbour"]]
        if len(df_harbour) > 0:
            dfHarbourGrouped = df_harbour.groupby(["timeDiscrepancy","VesselType", "IMO"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean))
            dfHarbourGrouped.reset_index(inplace=True)
            dfHarbourGrouped = dfHarbourGrouped.groupby(["timeDiscrepancy"]).agg(count = ("IMO", np.size), Length = ("Length", np.mean), Width =("Width", np.mean))
            dfHarbourGrouped.reset_index(inplace=True)
            if len(dfHarbourGrouped) > 0:
                dfHarbourGrouped["nearestPort"] = harbour
                framesHarbour.append(dfHarbourGrouped)                
    if len(framesHarbour) > 0:
        dfHarbourStats = pd.concat(framesHarbour)    
    return dfHarbourStats

def runGetHarbourData(year, month, daily = False):
    getHarbourData(year, month, daily)
    return 0

def getShipList():
    return pd.read_csv(f"/home/gridsan/{con.userName}/shipList.csv")

def getShipsDataFile():
    df = pd.read_csv(f"/home/gridsan/{con.userName}/shipsData.csv")
    df["IMO"] = df['IMO Number'].apply(lambda x: 'IMO' + str(x))
    return df

def saveResultedDataFile(df, year, month):
    df.to_csv(f"/home/gridsan/{con.userName}/Coding/PortStatsInitial2/berthInitial{year}{month}.csv")

def runInitial(year,month):
    saveResultedDataFile(getShipsData(year, month, getShipsDataFile()), year, month)
    return 0


def runFilteringBerths(year, month, daily = False):
    filteringBerth(year, month, daily)
    return 0

def runAnchorageAreas(year, month, daily = False):
    getAnchorageAreas(year, month, daily)
    return 0

def filteringBerth(year, month, daily = False):
    df = pd.read_csv(f"/home/gridsan/{con.userName}/Coding/PortStatsInitial2/berthInitial{year}{month}.csv")
    tdSuffix = ''
    if daily:
        timePattern = '%Y-%m-%d'
        tdSuffix = 'daily'    
    df = df[df["nearestPortDistance"] < 60]
    df["timeDiscrepancy"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime(timePattern)
    #df = df[df["nearestPortDistance"] > 1000]
    #df = df[df["nearestPortDistance"] < 350000]
    dfPortStatsGrouped = df.groupby(["timeDiscrepancy", "IMO", "nearestPort"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean))
    dfPortStatsGrouped.reset_index(inplace=True)
    dfPortStatsGrouped = dfPortStatsGrouped.groupby(["timeDiscrepancy", "nearestPort"]).agg(count = ("IMO", np.size), Length = ("Length", np.sum), Width =("Width", np.mean))
    dfPortStatsGrouped.reset_index(inplace=True)
    
    dfPortVesselStats = df.groupby(["IMO", "VesselName", "nearestPort"]).agg(Length = ("Length", np.mean), Width =("Width", np.mean), start_time = ("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
    dfPortVesselStats.reset_index(inplace=True)
    dfPortVesselStats["TimeSpent"] = dfPortVesselStats.apply(lambda x: (pd.to_datetime(x["end_time"])- pd.to_datetime(x["start_time"])).total_seconds()/3600, axis=1)
    
    dfPortStatsGrouped.to_csv(f"/home/gridsan/{con.userName}/Coding/BerthOnly/BerthOnly{year}{month}{tdSuffix}.csv")
    dfPortVesselStats.to_csv(f"/home/gridsan/{con.userName}/Coding/BerthOnly/VesselOnBerth{year}{month}.csv")
    return 0

def getAnchorageAreas(year, month, daily = False):
    ais_folder = f"/home/gridsan/{con.userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    tdSuffix = ''
    if daily:
        timePattern = '%Y-%m-%d'
        tdSuffix = 'daily'        
    files = os.listdir(ais_folder)
    ships = getShipsDataFile()
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
            df = df[df['IMO'].isin(ships["IMO"])]
            df["timeDiscrepancy"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime(timePattern)
            
            dfBoston = df[df["LAT"] < 42.45]
            dfBoston = dfBoston[dfBoston["LAT"] > 42.40]
            dfBoston = dfBoston[dfBoston["LON"] > -70.85]
            dfBoston = dfBoston[dfBoston["LON"] < -70.65]

            dfNY = df[df["LAT"] < 40.52]
            dfNY = dfNY[dfNY["LAT"] > 40.40]
            dfNY = dfNY[dfNY["LON"] > -73.75]
            dfNY = dfNY[dfNY["LON"] < -73.45]

            dfBoston[f"BostonAnchor"] = dfBoston.apply(lambda x: con.BostonPortAnchorageArea.contains(Point(x['LAT'], x['LON'])), 
                                                        axis=1)            
            dfNY[f"NYAnchor"] = dfNY.apply(lambda x: con.NYPortAnchorageArea.contains(Point(x['LAT'], x['LON'])), 
                                             axis=1)            
            dfBoston = dfBoston[dfBoston[f"BostonAnchor"]]
            dfNY = dfNY[dfNY[f"NYAnchor"]]
            if len(dfBoston) > 0:
                dfBostonGrouped = dfBoston.groupby(["timeDiscrepancy","VesselType", "IMO"]).agg(Length = ("Length", np.mean))
                dfBostonGrouped.reset_index(inplace=True)
                dfBostonGrouped = dfBostonGrouped.groupby(["timeDiscrepancy"]).agg(count = ("IMO", np.size))
                dfBostonGrouped.reset_index(inplace=True)
                if len(dfBostonGrouped) > 0:
                    framesBoston.append(dfBostonGrouped)                
                dfBostonVesselGrouped = dfBoston.groupby(["IMO", "VesselName"]).agg(start_time = ("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
                dfBostonVesselGrouped.reset_index(inplace=True)
                dfBostonVesselGrouped["TimeSpent"] = dfBostonVesselGrouped.apply(lambda x: (pd.to_datetime(x["end_time"])- pd.to_datetime(x["start_time"])).total_seconds()/3600, axis=1)
                if len(dfBostonVesselGrouped) > 0:
                    framesVesselBoston.append(dfBostonVesselGrouped)                

            if len(dfNY)>0:
                dfNYGrouped = dfNY.groupby(["timeDiscrepancy","VesselType", "IMO"]).agg(Length = ("Length", np.mean))
                dfNYGrouped.reset_index(inplace=True)
                dfNYGrouped = dfNYGrouped.groupby(["timeDiscrepancy"]).agg(count = ("IMO", np.size))
                dfNYGrouped.reset_index(inplace=True)
                if len(dfNYGrouped) > 0:
                    framesNY.append(dfNYGrouped)     
                dfNYVesselGrouped = dfNY.groupby(["IMO", "VesselName"]).agg(start_time = ("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
                dfNYVesselGrouped.reset_index(inplace=True)
                dfNYVesselGrouped["TimeSpent"] = dfNYVesselGrouped.apply(lambda x: (pd.to_datetime(x["end_time"])- pd.to_datetime(x["start_time"])).total_seconds()/3600, axis=1)
                if len(dfNYVesselGrouped) > 0:
                    framesVesselNY.append(dfNYVesselGrouped)  
    if len(framesNY) > 0:
        resNY = pd.concat(framesNY, ignore_index=True)
        resNY.to_csv(f"/home/gridsan/{con.userName}/Coding/Anchor/NYAnchorStats{year}{month}{tdSuffix}.csv")
    if len(framesVesselNY) > 0:
        resVNY = pd.concat(framesVesselNY, ignore_index=True)
        resVNY.to_csv(f"/home/gridsan/{con.userName}/Coding/Anchor/NYAnchorVessels{year}{month}.csv")
    if len(framesBoston) > 0:
        resBoston = pd.concat(framesBoston, ignore_index=True)
        resBoston.to_csv(f"/home/gridsan/{con.userName}/Coding/Anchor/BostonAnchorStats{year}{month}{tdSuffix}.csv")
    if len(framesVesselBoston) > 0:
        resVBoston = pd.concat(framesVesselBoston, ignore_index=True)
        resVBoston.to_csv(f"/home/gridsan/{con.userName}/Coding/Anchor/BostonAnchorVessels{year}{month}.csv")
    
    return 0


def getAnchorageAreas_daily(df, ships, daily = True):
    tdSuffix = ''
    dfAnchorStats = pd.DataFrame()
    dfAnchorVessels = pd.DataFrame()
    if daily:
        timePattern = '%Y-%m-%d'
        tdSuffix = 'daily'        
        df = df[df['IMO'].isin(ships["IMO"])]
        df["timeDiscrepancy"] = pd.to_datetime(df["BaseDateTime"]).dt.strftime(timePattern)
        anchorGrouped = []
        anchorVessels = []
        for anchor in con.portsList:
            df_anchor = df[df["LAT"] < con.PortLimitsCoordinates[anchor]['maxLAT']]
            df_anchor = df_anchor[df_anchor["LAT"] > con.PortLimitsCoordinates[anchor]['minLAT']]
            df_anchor = df_anchor[df_anchor["LON"] > con.PortLimitsCoordinates[anchor]['minLON']]
            df_anchor = df_anchor[df_anchor["LON"] < con.PortLimitsCoordinates[anchor]['maxLON']]
            df_anchor[f"Anchor"] = df_anchor.apply(lambda x: con.PortAnchorages[anchor].contains(Point(x['LAT'], x['LON'])), axis=1)            
            df_anchor = df_anchor[df_anchor[f"Anchor"]]
            if len(df_anchor) > 0:
                df_anchorGrouped = df_anchor.groupby(["timeDiscrepancy","VesselType", "IMO"]).agg(Length = ("Length", np.mean))
                df_anchorGrouped.reset_index(inplace=True)
                df_anchorGrouped = df_anchorGrouped.groupby(["timeDiscrepancy"]).agg(count = ("IMO", np.size))
                df_anchorGrouped.reset_index(inplace=True)
                if len(df_anchorGrouped) > 0:
                    df_anchorGrouped["nearestPort"] = anchor
                    anchorGrouped.append(df_anchorGrouped)
                dfAnchorVesselGrouped = df_anchor.groupby(["IMO", "VesselName"]).agg(start_time = ("BaseDateTime", np.min), end_time = ("BaseDateTime", np.max))
                dfAnchorVesselGrouped.reset_index(inplace=True)
                dfAnchorVesselGrouped["TimeSpent"] = dfAnchorVesselGrouped.apply(lambda x: (pd.to_datetime(x["end_time"])- pd.to_datetime(x["start_time"])).total_seconds()/3600, axis=1)                
                
                if len(dfAnchorVesselGrouped) > 0:
                    dfAnchorVesselGrouped["nearestPort"] = anchor
                    anchorVessels.append(dfAnchorVesselGrouped)  
        if len(anchorGrouped) > 0:
            dfAnchorStats = pd.concat(anchorGrouped)
        if len(anchorVessels) > 0:
            dfAnchorVessels = pd.concat(anchorVessels)
    return dfAnchorStats, dfAnchorVessels




def buildPoligon():
    df = df[df["y"]<-70.65]
    df = df[df["y"] > -70.85]
    df = df[df["x"] > 42.35]
    df = df[df["x"] < 42.50]
    gdf = geopandas.GeoDataFrame(
         df,    geometry=gpd.points_from_xy(df["y"], df["x"]))
    gdf["a"] = "a"

    res = gdf.dissolve("a").convex_hull
    return res

#runFilteringBerths(2023,'01')
#getAnchorageAreas(2023, '09')

def run_all(year, month):
    runHarbour = True
    runAnchor = False
    runInitial = False 
    runFiltering = True
    tdSuffix = 'daily'
    #runInitial(year, month)
    ais_folder = f"/home/gridsan/{con.userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)
    ships = getShipsDataFile()
    framesHarbourStats = []
    framesAnchorStats = []
    framesAnchorVessels = []
    framesShipsData = []
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_{year}_{month}"):
            print(f"processing file {f} started on {datetime.datetime.now().replace(microsecond=0)}" )
            df = pd.read_csv(ais_file, error_bad_lines=False )
            if runInitial:
                dfShipsData = getShipsData_daily(df.copy(), ships)
                framesShipsData.append(dfShipsData)
            if runHarbour:
                dfHarbourStats = getHarbourData_daily(df.copy(), True)
                framesHarbourStats.append(dfHarbourStats)
            if runAnchor:
                dfAnchorStats, dfAnchorVessels = getAnchorageAreas_daily(df.copy(), ships, True)
                framesAnchorStats.append(dfAnchorStats)
                framesAnchorVessels.append(dfAnchorVessels)
            
    if  runHarbour and len(framesHarbourStats) > 0 :
        harbourStats = pd.concat(framesHarbourStats, ignore_index=True)
        harbourStats.to_csv(f"/home/gridsan/{con.userName}/Coding/Harbour2/HarbourStats{year}{month}.csv")            
        
    if runAnchor and len(framesAnchorStats) > 0:
        anchorStats = pd.concat(framesAnchorStats, ignore_index=True)
        anchorStats.to_csv(f"/home/gridsan/{con.userName}/Coding/Anchor2/AnchorStats{year}{month}{tdSuffix}.csv")
        
    if runAnchor and len(framesAnchorVessels) > 0:
        anchorVessels = pd.concat(framesAnchorVessels, ignore_index=True)
        anchorVessels.to_csv(f"/home/gridsan/{con.userName}/Coding/Anchor2/AnchorVessels{year}{month}.csv")
    if runInitial and len(framesShipsData) > 0:
        ships_data = pd.concat(framesShipsData, ignore_index=True)
        saveResultedDataFile(ships_data, year, month)        
    if runInitial and runFiltering:
        runFilteringBerths(year,month, True)

#run_all(2015, '01')