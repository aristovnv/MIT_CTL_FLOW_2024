#!/bin/bash

#Slurm sbatch options
#SBATCH -o runModels.log-%j
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --exclusive

#was 84

# Load Anaconda and MPI module
module load anaconda/2023a
module load mpi/openmpi-4.1.5

# Call your script as you would from the command line
#mpirun python RunModel.py
mpirun python STGNN.py

#python -c "import NYStatModel as ny; ny.mergeLog()"


