Here you can find the code used for US DOT FLOW project.
This project used mostly AIS data for all ships around US coast from 01/01/2015 to /09/30/2023. 
Unfortunately, import data was not use for prediction of congestion, that could be potential next step.
Additional open-source data was considered being used.

There are 2 folders (some kegacy names are still in use): 

  Coding - this folder contains mostly files related to initial AID data processing
  Modelling - different models with usage of data from Coding.

Due to huge volume of data, data processing and models training was done on mit supercloud (https://supercloud.mit.edu/) with different parameters in run.sh

Coding folder - as it was mentioned, this folder contains initial processing of AIS data:
  Finding container ships on berth and ships in waiting area, calculating statistics on berth, harbor and anchorage area occupation.
  Performing DBSCAN to find points of conjestions, ports and ancorage areas (1)
  (Categorizing (clustering) containership by size was done in colab)
    run.sh - shell file to execute processing with required resources
    runProcess.py - envelope to run in parallel
    DBSCAN.py - performing DBSCAN on set of data
    ShipCategory.py - Processin ships list with adding Length and Width data from AIS
    PortStats.py - Gathering all ships/harbor stats on monthy basis
    AdditionalDataProcessing.py - merge monthly data into one file (one per category)
    Constant.py - constants file
  
Modelling folder - contains several models to predict congestion: XGBoost, RandomForest, LSTM, ST-GNN; Contains code for Inverse Reinforcement Learning
  run.sh - file for running oython code with required resources
  StatModels.py - for XGBoost, RandomForest and LSTM
  ST-GNN_newVersion.py - last version of ST-GNN model
  FindingSpot.py - algorithm for finding exact berth position based on historical bertging data of ships
  IRL_dataset - main IRL file, prepapres datasets, executes IRL training and prediction for terminals in the model (need to make refactor and speedup!!!!)
  Utils.py - some IRL functions
  IRLConstants.py - Constants for IRL (play here to change learning rates etc)
  IRL_env_def.py - IRL environment definitions
  

Related works:
1) Publication, DBSCAN: https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050924X00083/1-s2.0-S1877050924012547/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCfoPzLF1lZQHLcsEqRMNMfdd0t9wP6fiV3Yu2%2B%2BVoz0QIgPlFxUVHfw5Ps%2BQBmgtlFRMxwIdUH0vpvOcn6bUwKQP8qswUIGRAFGgwwNTkwMDM1NDY4NjUiDAUwiQ2YiLJlUcxkmiqQBXQuopB0N4dktDLDolP2d6K%2FFYf%2FP6KJV3HafbTAXDoTO%2FWrpJVmmac%2BITucaI2wx%2FTBlrsc6l7ni6W3mZsDnesDspJxIWiCRyF43VLTHK3JX3lV1pGJrhdA8FwRh2faRE4CUvi28rs%2F%2BBATI8SLWJ%2B5vo%2Fu95vr6ZBS9sQ8YoRVphBEb5w42msp%2FXLFFf%2FVu5zTx8ccn54vKz8B1iyeNjqzeFkQazmrTGvvNhEZy1ELyZeSNdqEmColtTYK0dEvMuKH%2Fp4WXS4GEEpH%2B5hLhE93LZljYjqTg%2FwtnB3ZEpLa8DBxA3NiLSJcouOWpaqClP5iTDbcV99%2FLg8oBmxcw%2BR0fbfyD9nhVEIEaQHwPM4p8M0Ph9pkVrmWc%2FwtcAGzka89sTP4SY4Fh5grBKQMSN8umazxFymN9MsyaDvorBmok7lyedIazO9raFqVrlrg2DtNV0yHoP3Gn29pf4pyx%2BS2fVPhaBjkPTW9cAb3leC5d5a8DuUpEjni7a01wydJBUZqXZngYV46R%2B%2B%2FpQjctl6ig9KpOuM8IqQ5ti0tMUSQE7zB%2Bmh6BKGWx9C895RqZo8lNT2VcBU0pYU1wrPDl74BeqgoNaXsDfL%2FcWAvaLSG5t7JmrwfkwyQkT7NFFf1u6RuoJVV7klgHqnbMWJAqsTC0jjDaOtNMBKoyilGwgCo6Yu22uNWxaPZ2jDvfAKDk4ZPalkM4AOXlzh2gl84PyrTvvxpSFOZpodt%2F%2FeL926JKThZCmCmvJxZp118ggX7EmTCqkDsh4oDyRLvWHhwFMHlCMe%2BzY0bNTu2xjpRqh46Hq4ce0ZYzIrbH%2BUzKmA36BB9zn1s2E%2FjHvwJRyB3eb9q3lQNge4HxIKExV0rqwkxMK26v7gGOrEBNRZhP%2Bodz8yNmvskCX%2FMw%2FIvgM5ACv4uTIhm4s5MZ3mhpGnI6He8uAc1jf2NLcvecNe3KlnhQ4RGOvcdoFv0nGfrtodpwGFFrlL1UqMcvURkuzGLjtU2fc%2Br9AYrgLz1%2Bb5Ccs9cv%2FN8homY0Orvmj%2Bgv2ze5mOm6kzR26tkSO7%2B23StVSMh2DuX5qwlvWBFErR2AraT1gqsc%2BR4CVfOZ%2Fg5C5wdKjLfDyRlL1jPfe%2FZ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241016T172354Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZUQ3ZDH7%2F20241016%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=450e2977fe248bd043e6a4dd6d2872ef14fca43e3f8f9cd03b9a134a659e8998&hash=7c27c5c1aa108b7f46e77a01c16af1dbcb6e152daa2349cb81643f6d28bce7d7&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050924012547&tid=spdf-e249f5cf-3ae5-4c39-b504-bf79a10e7bb0&sid=e62037bf11dad34eac2b1cb13119a4f414a2gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=13155b04545b01065750&rr=8d39bd46fb138fdb&cc=us)
2) Publication, Capstone (including DBSCAN): https://ctl.mit.edu/sites/ctl.mit.edu/files/theses/ELUCIDATING%20US%20IMPORT%20SUPPLY%20CHAIN%20DYNAMICS.pdf
3) Publication (1,2 and additional works, temporal link on ieee-hpec conference): https://ieee-hpec.org/wp-content/uploads/2024/09/139.pdf
4) Paper on current IRL status (submitted, not rewieved yet) in repository: Modelling/FLOW_IRL_TRB_AI.pdf.
