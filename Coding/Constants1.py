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
userName = "naristov"
pathData = f"/home/gridsan/{userName}/Modelling/Data"
pathLogs = f"/home/gridsan/{userName}/Modelling/Logs"
pathModels = f"/home/gridsan/{userName}/Modelling/Models"
pathOutput = f"/home/gridsan/{userName}/Modelling/Output"
pathRoot = f"/home/gridsan/{userName}"

BostonPortCoordinates = {0:(42.341804273594036, -71.02980795209466), 1:(42.3419874765817, -71.01790720327652)}

BostonPortAquatory = Polygon([(42.35556621842422, -70.93735034537563), (42.345045834649085, -70.89915833306084), (42.31064986013085, -70.84777271649186), (42.30987958576217, -70.88110392723931), (42.307825474695846, -70.98630556116096), (42.342479620136736, -71.01477597058098), (42.34350611845346, -70.95401595098927)])

NYPortAquatory = Polygon([(40.542318916329435, -73.94098865509673), (40.467400716140666, -73.95037257035008), (40.46829309161308, -74.2612147631174), (40.49951876274485, -74.27059867837075), (40.70613293456646, -74.15564571651717), (40.70257603305626, -74.10403418262375), (40.64652978034017, -74.14508881185715), (40.6554291564861, -74.08878532033704), (40.70079751106782, -74.05828759576363), (40.69901894159107, -73.99260018899017)])

BostonPortAnchorageArea =  Polygon([(42.40257923076923, -70.74652182186232), (42.410670250896, -70.81194559139784), (42.42132250403876, -70.80426483844913), (42.44617486486489, -70.73918181467178), (42.42861562790697, -70.71013260465114), (42.41704204142008, -70.69880633136088), (42.40257923076923, -70.74652182186232)])

NYPortAnchorageArea = Polygon([(40.42859534743203, -73.73900649546825), (40.4895569283276, -73.74402249146756), (40.49942771140417, -73.72887617532966), (40.50685044515099, -73.7025290779015), (40.509943707414855, -73.66836805611223), (40.504027586592095, -73.48565681564249), (40.49688553072632, -73.50451608938542), (40.42859534743203, -73.73900649546825)])

PortlandMePortAnchorageArea = Polygon([(43.52, -70.01), (43.72, -70.01), (43.72, -69.81), (43.52, -69.81)])
PortlandMePortAquatory = Polygon([(43.64644319006879, -70.2581543218201), (43.644330907081205, -70.25583406846671), (43.651451244134975, -70.24614504227551), (43.6558123784631, -70.23494542868313), (43.624073433777646, -70.2067024092657), (43.631054116143936, -70.18752930253157), (43.6631584969416, -70.22197321868047), (43.66549618945906, -70.23844368898702)])

NorfolkPortAnchorageArea = Polygon([(36.60, -75.70), (36.60, -75.10), (37.00, -75.10), (37.00, -75.70)])
NorfolkPortAquatory = Polygon([(37.082820430439234, -75.96220653940925), (37.00729528735854, -76.3085481901116), (36.96145472645158, -76.4205657812427), (36.85641236803843, -76.33882435476065), (36.862835772058006, -76.31109065648994), (36.96495364693104, -76.32860667645038), (36.9252898004366, -75.99799179969706)])

SavannaPortAnchorageArea = Polygon([(31.90, -80.60), (31.90, -80.30), (32.00, -80.30), (32.00, -80.60)])
SavannaPortAquatory = Polygon([(32.1414953899058, -81.14379605586558),(32.09086770043659, -81.12869678385786), (32.07588864802821, -81.07872527773053), (32.00693190482501, -80.8396663700918), (32.10063296319859, -80.86253908304639), (32.038866728689364, -80.89218424776254), (32.10589822752294, -81.00965162612401)])

BaltimorPortAnchorageArea = NorfolkPortAnchorageArea
BaltimorPortAquatory = Polygon([(37.110248816780306, -75.98512780182773), (38.983005080232786, -76.33492701626639), (39.15253344150625, -76.27733039707667), (39.20542852387246, -76.49778581830513), (39.25768192856102, -76.5530079026545), (39.244787625087284, -76.57068902477157), (39.20715461352019, -76.52931863262651), (39.13240478200626, -76.43479855575838), (39.01176192561259, -76.39733965692012), (37.0502411672974, -76.34578838571211), (36.93199070437279, -75.99984860866181)])


'''
PhyladelphiaPortAnchorageArea = Polygon([])
PhiladelphiaPortAquatory = Polygon([])



'''
PortlandMePortCoordinates = {0:(43.647945426743846, -70.25649769586678), 1:(43.64626934334997, -70.25830825140272)}

NY_RedHook_PortCoordinates = {0:(40.684142416340876, -74.01212973883557), 1:(40.6864053326207, -74.00837571534996)}
NY_LibertyBayonne_PortCoordinates = {0:(40.67350929700549, -74.0852786121706), 1:(40.66991941453956, -74.07693686476857)}
NY_LibertyNewYork_PortCoordinates = {0:(40.637570914682286, -74.19370023585414), 1:(40.64357746376455, -74.18629826969683)}
NY_Newark_PortCoordinates = {0:(40.687541630213616, -74.15631645763351), 1:(40.680458743651435, -74.14490409434313)}
NY_Maher_PortCoordinates = {0:(40.68705964668085, -74.16014246307424), 1:(40.67390460908992, -74.13890175359724), 2:(40.66828384048358, -74.14285113899098)}
NY_APM_PortCoordinates = {0:(40.66828384048358, -74.14285113899098), 1:(40.65955776568647, -74.14884245579564), 2:(40.662757562613606, -74.15678125197019)}

PhilPortCoordinates = {0:(39.8975668651021, -75.13667209908292), 1:(39.90529752776159, -75.13282973938465)}
PhilTiogaPortCoordinates = {0:(39.97956703588732, -75.07992988202021), 1:(39.97783995531592, -75.08967362084782)}
WilmingtonDePortCoordinates = {0:(39.72114172193462, -75.52978141494508), 1: (39.71741201469553, -75.51853425134732)}

ChesterPAPortCoordinates = {0:(39.85096404317711, -75.3387148586414), 1:(39.84958462331735, -75.3424336303261)}
SalemNJPortCoordinates = {0:(39.57383968539546, -75.4848122957912), 1:(39.57343782740635, -75.4866699392343)}

BaltimorPortCoordinates = {0:(39.25043228527255, -76.54056861104614), 1:(39.257703809879295, -76.552742710426)}

NorfolkPortCoordinates = {0:(36.90374823431183, -76.32600481354972), 1:(36.91519880959283, -76.32794726764261)}

Norfolk_VI_PortCoordinates = {0:(36.87003423172875, -76.34813571272258), 1:(36.880074899172385, -76.35101417961141)} #VIG Portsmouth


WilmingtonNC_PortCoordinates = {0:(34.18938801465766, -77.95599336068304), 1:(34.19663096232355, -77.9553832566425)}

WandoWelshSC_PortCoordinates = {0:(32.82862322398931, -79.89371622964143), 1:(32.83818825292812, -79.88929519921504)}
NorthCharlestonSC_PortCoordinates = {0:(32.89983128954796, -79.96180156097704), 1:(32.90496579243429, -79.95665171962114)}
HughKLeathermanSC_PortCoordinates = {0:(32.83591793206249, -79.93311327243495), 1:(32.8397689984232, -79.93288074020272)}
SavannaGardenCityGE_PortCoordinates = {0:(32.14149362872214, -81.14360565545472), 1:(32.13252134206184, -81.14242645209573), 2:(32.12743473358055, -81.13740430523177), 3:(32.12157092610278, -81.1344582693252), 4:(32.11763120002451, -81.13144167489067)}

LA_WBCT_CHINASHIPPING_Coordinates = {0:(33.75044474985812, -118.27322720653954), 1:(33.75641065841185, -118.27728908454198)}
LA_WBCT_ECT_Coordinates = {0:(33.76452999000073, -118.27776874548991), 1:(33.76082381783848, -118.27561839508998), 2:(33.75871317240361, -118.27749887627596)}

LA_Trapac_Coordinates = {0:(33.766267569037936, -118.27732949912492), 1:(33.76704032251859, -118.27066089270973), 2:(33.75828764181882, -118.27401540376734)}

LA_ETS_Coordinates = {0:(33.748636506964885, -118.27002077073244), 1:(33.74071604830348, -118.27480943779489)}

LA_Yusen_Coordinates = {0:(33.752795761893324, -118.26468591151824), 1:(33.75956630382521, -118.25784750259838), 2:(33.76070315450625, -118.25455552957324)}

LA_Fenix_Coordinates = {0:(33.731236024532855, -118.26043488999994), 1:(33.73485973981179, -118.24797628112059)}
LA_APM_Coordinates = {0:(33.723712471669145, -118.26032644609487), 1:(33.72894452222493, -118.25828193679041), 2:(33.733740288452054, -118.24179480406946)}

LB_A_Coordinates = {0:(33.76747585096194, -118.23821223496846), 1:(33.7700985547214, -118.2269199387691)}
LB_B_Coordinates = {0:(33.762885823766204, -118.21386987983296), 1:(33.75101648010919, -118.21389120151316)}
LB_T_Coordinates = {0:(33.75558376867356, -118.22882505880654), 1:(33.75102950196576, -118.24436985321692)}

LB_G_Coordinates = {0:(33.74385388204616, -118.2037048167187), 1:(33.74382794949853, -118.19929711466533), 2:(33.748149932582955, -118.19730117413098), 3:(33.744772275415805, -118.19728657075503), 4:(33.743864641301826, -118.19664204828494), 5:(33.7438041320194, -118.19248383880064)}
LB_J_Coordinates = {0:(33.740620454682855, -118.19560676124748), 1:(33.74062045466779, -118.20240543368145), 2:(33.73656933377884, -118.19461326569127), 3:(33.73656933377884, -118.18574588396599)}


PortLimitsCoordinates = {'Boston':{'maxLAT':42.45, 'minLAT':42.30,'maxLON':-70.65, 'minLON':-71.1}, 'NY':{'maxLAT':40.75, 'minLAT':40.40,'maxLON':-73.45, 'minLON':-74.3}, 'Portland_Me': {'maxLAT':43.7, 'minLAT':43.5,'maxLON':-69.9, 'minLON':-70.3}, 'Norfolk':{'maxLAT':37.10, 'minLAT':36.70,'maxLON':-75.60, 'minLON':-76.50} , 'Savanna':{'maxLAT':32.20, 'minLAT':32.00,'maxLON':-80.7, 'minLON':-81.2}, 'Baltimor':{'maxLAT':39.4, 'minLAT':36.9,'maxLON':-74.8, 'minLON':-76.7}}

PortAnchorages = {'Boston':BostonPortAnchorageArea, 'NY': NYPortAnchorageArea, 'Portland_Me': PortlandMePortAnchorageArea, 'Norfolk': NorfolkPortAnchorageArea, 'Savanna':SavannaPortAnchorageArea, 'Baltimor': BaltimorPortAnchorageArea}
PortHarbour = {'Boston':BostonPortAquatory, 'NY': NYPortAquatory, 'Portland_Me':PortlandMePortAquatory, 'Norfolk':NorfolkPortAquatory, 'Savanna':SavannaPortAquatory, 'Baltimor':BaltimorPortAquatory}

portsList = ['Boston', 'NY', 'Portland_Me', 'Norfolk', 'Savanna', 'Baltimor']


PortCoordinates = {'Boston': BostonPortCoordinates, 'NY_RedHook': NY_RedHook_PortCoordinates, 'NY_LibertyB': NY_LibertyBayonne_PortCoordinates, 'NY_LibertyNY': NY_LibertyNewYork_PortCoordinates,'NY_Newark': NY_Newark_PortCoordinates, 'NY_Maher': NY_Maher_PortCoordinates, 'NY_APM': NY_APM_PortCoordinates, 'Portland_Me': PortlandMePortCoordinates, 'Wilmington_DE': WilmingtonDePortCoordinates, 'Philadelphia': PhilPortCoordinates, 'Philadelphia_Tioga': PhilTiogaPortCoordinates, 'Chester_PA': ChesterPAPortCoordinates, 'Salem_NJ': SalemNJPortCoordinates, 'LA_WBCT_ChinaShipping': LA_WBCT_CHINASHIPPING_Coordinates, 'LA_WBCT_ECT': LA_WBCT_ECT_Coordinates, 'LA_TRAPAC': LA_Trapac_Coordinates, 'LA_ETS': LA_ETS_Coordinates, 'LA_Yusen': LA_Yusen_Coordinates, 'LA_Fenix': LA_Fenix_Coordinates, 'LA_APM': LA_APM_Coordinates, 'LB_J':LB_J_Coordinates, 'LB_A':LB_A_Coordinates, 'LB_B':LB_B_Coordinates, 'LB_T':LB_T_Coordinates, 'LB_G':LB_G_Coordinates, 'Baltimore':BaltimorPortCoordinates, 'Norfolk': NorfolkPortCoordinates, 'Norfolk_Virginia_International':Norfolk_VI_PortCoordinates, 'Wilmington_NC':WilmingtonNC_PortCoordinates, 'Wando_Welch_SC':WandoWelshSC_PortCoordinates, 'North_Charleston_SC':NorthCharlestonSC_PortCoordinates, 'HughKLeatherman_SC':HughKLeathermanSC_PortCoordinates, 'Savanna':SavannaGardenCityGE_PortCoordinates}

