#!/bin/bash
#SBATCH -n 1 -c 1 -G 1
env_scripts=/home/pnzhizhin/anaconda3/bin

module load Python/Anaconda_v10.2019
source $env_scripts/deactivate
source $env_scripts/activate hedge_env

echo "Starting classification script"
./run_classification.sh
