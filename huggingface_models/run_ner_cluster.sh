#!/bin/bash
#SBATCH -n 1 -c 1 -G 1
#SBATCH --output=output.out
env_scripts=/home/pnzhizhin/anaconda3/bin

module load Python/Anaconda_v10.2019 
source $env_scripts/deactivate
source $env_scripts/activate hedge_env

echo "Starting NER script"
./run_ner.sh
