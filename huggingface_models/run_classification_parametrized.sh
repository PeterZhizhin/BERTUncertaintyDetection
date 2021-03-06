#!/bin/bash
#SBATCH -n 1 -c 1 -G 1
# The python script is in this directory as well, so make it cwd.
cd "$(dirname "$0")"

machine_name=$(uname -n)
if [[ $machine_name == cn-* ]];
then
  echo "Executing on HPC cluster, setting up env"
  env_scripts=/home/pnzhizhin/anaconda3/bin

  module load Python/Anaconda_v10.2019
  source $env_scripts/deactivate
  source $env_scripts/activate hedge_env
fi

BERT_MODEL=$1
OUTPUT_DIR=$2
DATA_DIR=$3
SEED=$4

MAYBE_DO_TRAIN="--do_train"
if [ ! -z $5 ]
then
  MAYBE_DO_TRAIN=""
fi

mkdir -p classification_results
export BATCH_SIZE=32
export NUM_EPOCHS=5
export SAVE_STEPS=750
export MAX_LENGTH=512

python3 run_classification.py --data_dir $DATA_DIR \
    --task_name hedge \
    --model_type bert \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --cache_dir transformers_cache \
    --do_eval \
    $MAYBE_DO_TRAIN
