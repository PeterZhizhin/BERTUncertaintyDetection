#!/bin/bash
echo "Training initial models"
sbatch --wait ./train_all_models_on_wiki_and_bio_slurm.sh
echo "Performing transfer"
sbatch --wait ./transfer_all_models_on_wiki_and_bio_slurm.sh