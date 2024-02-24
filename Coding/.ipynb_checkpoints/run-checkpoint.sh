#!/bin/bash

#Slurm sbatch options
#SBATCH -o runAISTransformation.log-%j
#SBATCH -n 105
#SBATCH -c 2
#SBATCH --exclusive
#n was 105!!
# Load Anaconda and MPI module
module load anaconda/2023a
module load mpi/openmpi-4.1.5

# Call your script as you would from the command line
mpirun python runProcess.py
#mpirun python ShipCategory.py
#mpirun python PortStats.py

