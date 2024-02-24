import os
import numpy as np
from mpi4py import MPI
import sys
from PortStats import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

name = MPI.Get_processor_name()
pid = os.getpid()

print("Hello World! I am process %d of %d on %s with pid %d.\n" % (rank, size, name, pid))

# Load the file names

idx = range(105)
myidx = idx[rank:len(idx):size]

# portList = {'1703', '5204','10000','5203', '5201', '1821', '1801', '1901', '1902', '2002', '5301', '2301', '10001', '10002', '10003', '10004', '4909', '10005', '5101', '5104', '10006', '10007', '2501', '2704', '2713', '2809', '3002', '3001', '20000', '20001', '3007'}

# portList = {'10000','5311', '1805', '1701', '20002', '1103', '1816'}

#portList = {'10008', '10009'}

#for i in range(108):
#    month = f"{(i % 12 + 1):02}"
#    year = i // 12 
#    print(f"{i} and {year} and {month}")


for i in myidx:
    month = f"{(i % 12 + 1)  :02}"
    year = i // 12
    yearStart = 2015
    yearCount = 9
    #if  year == year: #not((year == 2015 and (i==1 or i ==0)) or (year ==2023 and i>5) or (year == 2020)):
    #if (1==1):
    print(f"saving {month} and year is {year + yearStart} ")
    try:
        run_all(year + yearStart,month)
        #runInitial(year + yearStart,month)
        #runGetHarbourData(year + yearStart,month, True)
        #runFilteringBerths(year + yearStart,month, True)
        #runAnchorageAreas(year + yearStart,month, True)
    except Exception as error:
        print(f"An exception occurred for {year + yearStart} and {month}:", error)

                

