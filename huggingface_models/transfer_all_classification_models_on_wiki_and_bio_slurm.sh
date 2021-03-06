#!/bin/bash
#SBATCH -n 1 -c 1 -G 1
#SBATCH --error=ner_experiments.%A_%a.error
#SBATCH --output=ner_experiments.%A_%a.out
#SBATCH --array=0-39

BASE_DATA_DIR=../uncertainty_dataset/output_datasets/result/classification
BASE_OUTPUT_DIR="classification_experiment"

mkdir -p $BASE_OUTPUT_DIR

# 30 runs in total 1 * 2 * 20 = 40 (training only bert large)
declare -a model_paths=("bert-large-cased" "google/electra-small-discriminator" "bert-base-cased" "allenai/scibert_scivocab_cased" "biobert_base")
declare -a output_base_paths=("bert_large_cased" "electra_small_discriminator" "bert_base_cased" "scibert_scivocab_cased" "biobert_base")
declare -a dataset_paths=("wiki" "bio")
declare -a dataset_transfers=("bio" "wiki")
seeds_start=1
number_of_seeds=20

let seeds_end=$seeds_start+$number_of_seeds-1
echo $seeds_start $seeds_end

all_arguments_list=()

for (( i=0; i<${#model_paths[*]}; ++i ));
do
  current_output_base_path=${output_base_paths[$i]}
  for (( j=0; j<${#dataset_paths[*]}; ++j ));
  do
    dataset_part=${dataset_paths[$j]}
    dataset_transfer=${dataset_transfers[$j]}
    for seed in $(seq $seeds_start $seeds_end)
    do
      current_model_path="$BASE_OUTPUT_DIR/${current_output_base_path}_${dataset_part}_${seed}"
      full_output_dir="$BASE_OUTPUT_DIR/${current_output_base_path}_${dataset_part}_to_${dataset_transfer}_${seed}"
      full_data_dir="$BASE_DATA_DIR/$dataset_transfer"
      full_argument_list="$current_model_path $full_output_dir $full_data_dir $seed"
      all_arguments_list+=("$full_argument_list")
    done
  done
done

current_task_arguments=${all_arguments_list[${SLURM_ARRAY_TASK_ID}]}
echo "Current task ID: $SLURM_ARRAY_TASK_ID"
echo "Executing arguments: $current_task_arguments"
./run_classification_parametrized.sh $current_task_arguments
