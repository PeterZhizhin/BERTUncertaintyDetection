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

export MAX_LENGTH=512

export BATCH_SIZE=32
export NUM_EPOCHS=5
export SAVE_STEPS=750
export EVAL_STEPS=0

python run_ner.py --data_dir $DATA_DIR \
    --model_type bert \
    --labels ../labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --cache_dir transformers_cache \
    --do_predict \
    $(MAYBE_DO_TRAIN)
