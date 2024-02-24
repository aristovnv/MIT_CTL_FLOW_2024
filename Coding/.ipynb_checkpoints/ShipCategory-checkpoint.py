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
import geopy.distance


def getShipsData(year, shipList, readyShips):
    ais_folder = f"/home/gridsan/{con.userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)    
    frames = []
    shipsLeft = shipList
    if  not(readyShips is None):
        frames.append(readyShips)
        shipsLeft = shipList[~shipList["MMSI"].isin(readyShips["MMSI"])]
    print(f"original {len(shipsLeft)}");
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_"):
            print(f"processing file {f}")
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df = df[df['MMSI'].isin(shipsLeft["MMSI"])]                
            if len(df) > 0:
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
            
    updatedShipList = pd.concat(frames, ignore_index=True)
    return updatedShipList

def getShipList():
    return pd.read_csv(f"/home/gridsan/{con.userName}/shipList.csv")

def getShipsDataFile():
    return pd.read_csv(f"/home/gridsan/{con.userName}/shipsData.csv")

def saveShipsDataFile(df):
    df.to_csv(f"/home/gridsan/{con.userName}/shipsData.csv")

def getIMODataFile():
    return pd.read_csv(f"/home/gridsan/{con.userName}/shipsIMOData.csv")

def saveIMODataFile(df):
    df.to_csv(f"/home/gridsan/{con.userName}/shipsIMOData.csv")

    
    
def getIMOList(year, shipList, readyShips):
    ais_folder = f"/home/gridsan/{con.userName}/flow_shared/data/Marine_Cadaster_2015-2023/{year}"
    files = os.listdir(ais_folder)    
    frames = []
    shipList["IMO Number"] = shipList["IMO Number"].astype(str).fillna('')
    shipList["IMO Number"] = "IMO" + shipList["IMO Number"].astype(str).str.split('.').str[0]
    shipsLeft = shipList
    if  not(readyShips is None):
        readyShips['date'] = pd.to_datetime(readyShips['date'])
        frames.append(readyShips)
    print(f"original {len(shipsLeft)}");
    for f in files:
        #print(f)
        ais_file = ais_folder + "/" + f
        if f.startswith(f"AIS_"):
            print(f"processing file {f}")
            df = pd.read_csv(ais_file, error_bad_lines=False )
            df = df[df['IMO'].isin(shipList["IMO Number"])]      
            if len(df) > 0:
                dfUniqueList = df[["MMSI", "IMO"]].drop_duplicates()  
                dfUniqueList.reset_index(inplace=True)
                dfUniqueList = dfUniqueList[["MMSI", "IMO"]].drop_duplicates()  
                condition = pd.merge(dfUniqueList,shipList, left_on = 'IMO', right_on = 'IMO Number', how = 'left', suffixes=('', '_ships'))
                condition["yesno"] = condition["MMSI"] == condition["MMSI_ships"]
                condition = condition[~condition["yesno"]]
                dfUniqueList = dfUniqueList[dfUniqueList["IMO"].isin(condition["IMO"])][['MMSI', 'IMO']]                
                dateYear = f.split('_')[1]
                dateMonth = f.split('_')[2]
                dateDate = f.split('_')[3].split('.')[0]                
                dfUniqueList["date"] = datetime.datetime.strptime(dateYear + dateMonth + dateDate, '%Y%m%d')
                if len(dfUniqueList) > 0:
                    frames.append(dfUniqueList)

               # group by df by fields, join to shipsLeft and 
            
    updatedShipList = pd.concat(frames, ignore_index=True)
    updatedShipList = updatedShipList.groupby(["MMSI", "IMO"]).agg(date = ("date","max"))
    updatedShipList.reset_index(inplace=True)
    return updatedShipList

year = 2019
for y in range(9):

    currentIMOData = getIMODataFile()
    df = getIMOList(y + 2015, getShipList(), currentIMOData)
    print(f"done year {y + 2015}")
    saveIMODataFile(df)
    