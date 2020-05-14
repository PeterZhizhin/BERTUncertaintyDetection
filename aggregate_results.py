import argparse
import re
import os
import collections
import csv

FOLDER_RE = re.compile(r'([a-z_]+?)_(wiki|bio|wiki_to_bio|bio_to_wiki)_(\d+)$')
FOLDER_RE_RESULTS = re.compile(r'([a-z_]+?)_(wiki|bio|wiki_to_bio|bio_to_wiki)_at_(wiki|bio|factbank)_(\d+)$')


def parse_eval_results_file(path):
    assert os.path.isfile(path)
    with open(path, 'r') as f:
        all_lines = f.readlines()
        metric_value = [line.strip().split(' = ') for line in all_lines]
        return {metric: float(value) for metric, value in metric_value}


def export_all_metrics_csv(metric_model_dataset_all_values, output_file_path):
    with open(output_file_path, 'w') as output_f:
        output_csv = csv.writer(output_f)
        output_csv.writerow(['model', 'dataset', 'evaluated_at', 'metric', 'run_id', 'value'])
        for (model_name, dataset, evaluated_at, metric), all_values in metric_model_dataset_all_values.items():
            for seed, value in all_values:
                output_csv.writerow([model_name, dataset, evaluated_at, metric, seed, value])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_folder', type=str, help='Folder with all results.')
    parser.add_argument('--export_csv', action='store_true', help='Whether to export CSV with all results or not.')
    parser.add_argument('--csv_output', type=str, default='aggregate_output.csv', help='File for CSV results.')
    parser.add_argument('--eval_all_script_format', action='store_true',
                        help='Whether to parse results in the evaluate_all_models.sh format.')

    args = parser.parse_args()

    model_output_file = 'test_results.txt' if args.eval_all_script_format else 'eval_results.txt'

    all_files = os.listdir(args.results_folder)
    target_re = FOLDER_RE_RESULTS if args.eval_all_script_format else FOLDER_RE
    re_matches = [target_re.match(result) for result in all_files]

    metric_model_and_dataset_to_results = collections.defaultdict(list)
    for filename, re_match in zip(all_files, re_matches):
        if re_match is None:
            continue
        eval_results_path = os.path.join(args.results_folder, filename, model_output_file)
        metrics = parse_eval_results_file(eval_results_path)
        if args.eval_all_script_format:
            model_name, dataset, evaluated_at, seed = re_match.groups()
        else:
            model_name, dataset, seed = re_match.groups()
            evaluated_at = dataset.split('_')[-1]

        for metric, value in metrics.items():
            metric_model_and_dataset_to_results[(model_name, dataset, evaluated_at, metric)].append((seed, value))

    if args.export_csv:
        print('Exporting CSV to {}'.format(args.csv_output))
        export_all_metrics_csv(metric_model_and_dataset_to_results, args.csv_output)
    else:
        for (model_name, dataset, evaluated_at, metric), all_values in metric_model_and_dataset_to_results.items():
            average_metric = sum(i[1] for i in all_values) / len(all_values)
            print(f'{model_name} {dataset} (eval at: {evaluated_at}) {metric}: {average_metric}')


if __name__ == "__main__":
    main()
