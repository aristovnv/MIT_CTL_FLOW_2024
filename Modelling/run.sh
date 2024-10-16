#!/bin/bash

#Slurm sbatch options
#SBATCH -o Logs/runModels.log-%j
#SBATCH --job-name pytorch
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --exclusive
# SBATCH -N 1
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:volta:2
# Load Anaconda and MPI module
module load anaconda/2023a
module load mpi/openmpi-4.1.5
#module load cuda/11.6
#module load nccl/2.11.4-cuda11.6


#was 84

# Load Anaconda and MPI module

# Call your script as you would from the command line
#mpirun python RunModel.py
#mpirun python STGNN_newVersion.py
#mpirun python StatModels.py
#mpirun python FindingSpots.py
#mpirun python IRL_torch.py
#mpirun python IRL_0107_speed_load.py
#mpirun python IRL_0107.py
#mpirun python IRL_0107_speed.py

#mpirun python IRL_0107_speed_load_constr.py
#mpirun python IRL_0107_predict.py
mpirun python IRL_run.py
#python -c "import NYStatModel as ny; ny.mergeLog()"


