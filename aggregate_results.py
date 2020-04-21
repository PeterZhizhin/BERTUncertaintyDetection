import argparse
import re
import os
import collections

FOLDER_RE = re.compile(r'([a-z_]+)_(wiki|bio)_(\d+)$')


def parse_eval_results_file(path):
    assert os.path.isfile(path)
    with open(path, 'r') as f:
        all_lines = f.readlines()
        metric_value = [line.strip().split(' = ') for line in all_lines]
        return {metric: float(value) for metric, value in metric_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_folder', type=str, help='Folder with all results.')

    args = parser.parse_args()
    all_files = os.listdir(args.results_folder)

    re_matches = [FOLDER_RE.match(result) for result in all_files]

    metric_model_and_dataset_to_results = collections.defaultdict(list)
    for filename, re_match in zip(all_files, re_matches):
        if re_match is None:
            continue
        eval_results_path = os.path.join(args.results_folder, filename, 'eval_results.txt')
        metrics = parse_eval_results_file(eval_results_path)
        model_name, dataset, seed = re_match.groups()

        for metric, value in metrics.items():
            metric_model_and_dataset_to_results[(model_name, dataset, metric)].append(value)

    for (model_name, dataset, metric), all_values in metric_model_and_dataset_to_results.items():
        average_metric = sum(all_values) / len(all_values)
        print(f'{model_name} {dataset} {metric}: {average_metric}')


if __name__ == "__main__":
    main()
