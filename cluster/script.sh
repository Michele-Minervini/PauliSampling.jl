#!/bin/bash -l

#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
## # SBATCH --exclusive
#SBATCH --cpus-per-task=18  # its 7000 per cpu # 72 is max
#SBATCH --time=2-00:00:00

echo STARTING AT `date`

export JULIA_NUM_THREADS=18

srun julia --project=./ main.jl "$@"

echo FINISHED at `date`
