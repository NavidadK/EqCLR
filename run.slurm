#!/bin/bash
#SBATCH -p gpu                 # partition (cpu, gpu, gpu-big, etc.)
#SBATCH -J eqres_rs       # job name
#SBATCH --mem=40G              # memory (check partition limits!)
#SBATCH -t 0-12:00:00          # time (days-hh:mm:ss)
#SBATCH -o slurm_%j.out        # stdout file
#SBATCH -e slurm_%j.err        # stderr file

# Load modules if needed (depends on your cluster)
# module load python

# Activate your conda environment if needed
source ~/.bashrc
conda activate eq_env2

# Run python script
python eqCLR_pmnist.py