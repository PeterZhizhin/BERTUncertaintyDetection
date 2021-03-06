#!/bin/bash
#SBATCH -n 1 -c 1 -G 1
#SBATCH --error=model_eval.%A_%a.error
#SBATCH --output=model_eval.%A_%a.out
#SBATCH --time="00:10:00"
#SBATCH --array=0-959

DO_GENERATE_SYMLINKS=$1

BASE_DATA_NER_DIR=../uncertainty_dataset/output_datasets/result
BASE_DATA_CLASSIFICATION_DIR=../uncertainty_dataset/output_datasets/result/classification

BASE_OUTPUT_CLASSIFICATION_DIR="classification_experiment"
BASE_OUTPUT_NER_DIR="ner_experiments"

# Total number of models: 2 * 3 * 8 * 20 = 960
declare -a used_programs=("./run_ner_parametrized.sh" "./run_classification_parametrized.sh")
# declare -a used_programs=("echo" "echo")
declare -a tasks_data_dir=("$BASE_DATA_NER_DIR" "$BASE_DATA_CLASSIFICATION_DIR")
declare -a tasks_output_dir=("$BASE_OUTPUT_NER_DIR" "$BASE_OUTPUT_CLASSIFICATION_DIR")

declare -a base_model_paths=("bert_base_cased" "scibert_scivocab_cased" "biobert_base")

declare -a model_types=("wiki" "bio" "wiki_to_bio" "bio_to_wiki" "wiki" "bio" "wiki" "bio")
declare -a used_datasets=("wiki" "bio" "bio" "wiki" "factbank" "factbank" "bio" "wiki")

declare -a model_files=("config.json" "pytorch_model.bin" "special_tokens_map.json" "tokenizer_config.json" "training_args.bin" "vocab.txt")

seeds_start=1
number_of_seeds=20

let seeds_end=$seeds_start+$number_of_seeds-1
echo $seeds_start $seeds_end

all_arguments_list=()
used_programs_list=()
for (( task_number=0; task_number<${#tasks_data_dir[*]}; ++task_number ));
do
  BASE_DATA_DIR=${tasks_data_dir[$task_number]}
  BASE_OUTPUT_DIR=${tasks_output_dir[$task_number]}

  used_program=${used_programs[$task_number]}
  for base_model_path in "${base_model_paths[@]}"
  do
    for (( model_type_number=0; model_type_number<${#model_types[*]}; ++model_type_number ));
    do
      model_type=${model_types[$model_type_number]}
      evaluated_dataset=${used_datasets[$model_type_number]}
      for seed in $(seq $seeds_start $seeds_end)
      do
        full_model_dir="$BASE_OUTPUT_DIR/${base_model_path}_${model_type}_${seed}"
        full_output_dir="$BASE_OUTPUT_DIR/results/${base_model_path}_${model_type}_at_${evaluated_dataset}_${seed}"

	if [ ! -z $DO_GENERATE_SYMLINKS ];
	then
          echo "Generating a symlink"
          mkdir -p $full_output_dir
          for model_file in "${model_files[@]}"
          do
            ln -sf $(realpath "$full_model_dir/$model_file") "$full_output_dir/$model_file"
          done
	fi

        full_data_dir="$BASE_DATA_DIR/$evaluated_dataset"
        full_argument_list="$full_model_dir $full_output_dir $full_data_dir $seed --no_train"

        all_arguments_list+=("$full_argument_list")
        used_programs_list+=("$used_program")
      done
    done
  done
done

current_task_arguments=${all_arguments_list[${SLURM_ARRAY_TASK_ID}]}
current_used_program=${used_programs_list[${SLURM_ARRAY_TASK_ID}]}
echo "Current task ID: $SLURM_ARRAY_TASK_ID"
echo "Executing arguments: $current_used_program $current_task_arguments"
$current_used_program $current_task_arguments
