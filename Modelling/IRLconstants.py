from pathlib import Path
import datetime as dt

HOME = str(Path.home())
DATA_PATH = f"{HOME}/Modelling/Data" #path
MODELS_PATH = f"{HOME}/Modelling/Models" #pathModels
BERTH_FILE = f"{DATA_PATH}/ResultedVesselOnBerth.csv" #file_berth
ANCHOR_FILE = f"{DATA_PATH}/ResultedVesselAnchor.csv" #file_anchor
SHIP_FILE = f"{DATA_PATH}/ships_grouped_cleaned.csv" #file_ship
OPERATORS_FILE = f"{DATA_PATH}/ship_operator_changes.csv" #file_operators
 #file_spots
VOYAGES_FILE = f"{DATA_PATH}/voyages_origin.csv" #file_voyages

MIN_BERTH_TIME = 2 #min_berth_time
MIN_ANCHOR_TIME = 1 #min_anchor_time
HOURS = 6 #hours
TIMEFRAMES_PER_DAY = int(24 / HOURS) #timeframes_per_day
INCOMING_DAYS = 5 #7 #incoming_days
EXTRA_FEATURES_NUM = 10

LEARNING_RATE = 0.3
NUM_ITERATIONS = 101 #100 #10
SOFTMAX_THRESHOLD = -500
TEMPERATURE = 0.001#00
STAY_MULTIPLICATOR = 0#10

TIME_FEATURES = 5
#TIME_SPENT_FEATURES = 4  #????

START_DATE = dt.datetime(2015, 1, 1)
END_DATE = dt.datetime(2023, 12, 31)

USE_LOG = True

L1_REG = 0.2
L2_REG = 0.1
ZERO_TO_NEGATIVE = True

REMOVE_WRONG_ACTIONS = False
SCALE_FEATURES = True

NOISE_MEAN = 0
NOISE_SIGMA = 0.8