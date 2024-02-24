import os
import numpy as np
from mpi4py import MPI
import sys
import NYStatModel as ny

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

name = MPI.Get_processor_name()
pid = os.getpid()

print("Hello World! I am process %d of %d on %s with pid %d.\n" % (rank, size, name, pid))

# Load the file names

idx = range(84)
myidx = idx[rank:len(idx):size]

# portList = {'1703', '5204','10000','5203', '5201', '1821', '1801', '1901', '1902', '2002', '5301', '2301', '10001', '10002', '10003', '10004', '4909', '10005', '5101', '5104', '10006', '10007', '2501', '2704', '2713', '2809', '3002', '3001', '20000', '20001', '3007'}

# portList = {'10000','5311', '1805', '1701', '20002', '1103', '1816'}

portList = {'10008', '10009'}

mods = ['count', 'diff_to_prev']
terminals = ['NY_APM', 'NY_LibertyB', 'NY_LibertyNY', 'NY_Maher', 'NY_Newark', 'NY_RedHook', 'Boston']
modelTypeTerminal = {'Boston':2, 'NY_APM':1, 'NY_LibertyB':1, 'NY_LibertyNY':1, 'NY_Maher':1, 'NY_Newark':1, 'NY_RedHook':1}
Models_terminal = ['arima', 'expSmooth', 'LSTM', 'XGBoost'] 

Models_grouped = ['LSTM', 'XGBoost', 'VAR', '1DCNN', 'RandomForestClassifier', 'RandomForestRegressor', 'LSTNML'] 

NYterminals = ['NY_APM', 'NY_LibertyB', 'NY_LibertyNY', 'NY_Maher', 'NY_Newark', 'NY_RedHook']
EastCostTerminals = ['NY_APM', 'NY_LibertyB', 'NY_LibertyNY', 'NY_Maher', 'NY_Newark', 'NY_RedHook', 'Boston']

GroupedTerminals = [NYterminals, EastCostTerminals]
ModelTypeGroupedTerminals = [1, 0]
models_per_terminal = len(Models_terminal)
variables_count = len(mods)
single_terminal_models = models_per_terminal * len(terminals)
grouped_terminal_models = len(Models_grouped) * len (GroupedTerminals)

roundQty = 1 

#currently 84. every single terminal adds 8 (4 model * 2), every grouped port adds 8 per terminal to single model and 2 to grouped (if it is East Cost), for west coast need to create also grouped west + probably total. (4 * 7 + 7 * 2) * 2
total_iterations = (single_terminal_models + grouped_terminal_models) * variables_count

for i in myidx:
    if i < single_terminal_models * variables_count:
        mod = mods [i // single_terminal_models]
        k = i % single_terminal_models
        terminal = terminals[k // models_per_terminal]
        model = Models_terminal[k % models_per_terminal]
        modelType = modelTypeTerminal[terminal]
        print (f"{i}: {mod}, {terminal}, {model} and model type {modelType}")
        try:
            ny.startModelPerTerminal(model, modelType, terminal, mod, roundQty)
        except Exception as error:
            print(f"An exception occurred for {terminal} and {model}:", error)
        
    else:
        k = i - single_terminal_models * variables_count
        mod = mods [k // grouped_terminal_models]
        l = k % grouped_terminal_models
        groupedTerminal = GroupedTerminals[l // len(Models_grouped)]
        modelType = ModelTypeGroupedTerminals[l // len(Models_grouped)]
        model = Models_grouped[l % len(Models_grouped)]
        print (f"{i}: {mod}, {groupedTerminal}, {model}, and model type {modelType} ")
        try:
            ny.startModelForGroup(model, modelType, mod, roundQty)
        except Exception as error:
            print(f"An exception occurred for {groupedTerminal} and {model}:", error)
