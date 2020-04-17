#!/bin/bash
# The python script is in this directory as well, so make it cwd.
cd "$(dirname "$0")"

export MAX_LENGTH=512
export BERT_MODEL=allenai/scibert_scivocab_cased

export OUTPUT_DIR=scibert_scivocab_cased_wiki_hedge_50percent
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py --data_dir ../uncertainty_dataset/output_datasets/result \
    --model_type bert \
    --labels ../labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --cache_dir transformers_cache \
    --do_train \
    --do_eval

