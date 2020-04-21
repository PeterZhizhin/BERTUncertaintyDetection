#!/bin/bash
#SBATCH -n 1 -c 1 -G 1
#SBATCH --error=ner_experiments.%A_%a.error
#SBATCH --output=ner_experiments.%A_%a.out
#SBATCH --array=0-29

BASE_DATA_DIR=../uncertainty_dataset/output_datasets/result

# 30 runs in total 3 * 2 * 5 = 30
declare -a model_paths=("bert-base-cased" "allenai/scibert_scivocab_cased" "biobert_base")
declare -a output_base_paths=("bert_base_cased" "scibert_scivocab_cased" "biobert_base")
declare -a dataset_paths=("wiki" "bio")
declare -a dataset_transfers=("bio" "wiki")
number_of_seeds=5

all_arguments_list=()

for (( i=0; i<${#model_paths[*]}; ++i ));
do
  current_output_base_path=${output_base_paths[$i]}
  for (( j=0; j<${#dataset_paths[*]}; ++j ));
  do
    dataset_part=${dataset_paths[$j]}
    dataset_transfer=${dataset_transfers[$j]}
    for seed in $(seq 1 $number_of_seeds)
    do
      current_model_path="${current_output_base_path}_${dataset_part}_${seed}"
      full_output_dir="${current_output_base_path}_${dataset_part}_to_${dataset_transfer}_${seed}"
      full_data_dir="$BASE_DATA_DIR/$dataset_transfer"
      full_argument_list="$current_model_path $full_output_dir $full_data_dir $seed"
      all_arguments_list+=("$full_argument_list")
    done
  done
done

current_task_arguments=${all_arguments_list[${SLURM_ARRAY_TASK_ID}]}
echo "Current task ID: $SLURM_ARRAY_TASK_ID"
echo "Executing arguments: $current_task_arguments"
./run_ner_parametrized.sh $current_task_arguments
