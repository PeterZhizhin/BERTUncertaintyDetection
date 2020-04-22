import argparse
from data import hedge_dataset


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_xml', help='Path to an XML file with the dataset to convert')
    arg_parser.add_argument('dataset_out', type=str, help='Target path where to save the result')
    arg_parser.add_argument('--output_type', default='xml',
                            help=('Dataset output type (valid values are: csv, ner, multiner, classification, '
                                  'multi_classification)'))
    return arg_parser.parse_args()


def main():
    args = parse_args()
    dataset = hedge_dataset.HedgeDataset.from_xml_file(args.dataset_xml)
    if args.output_type == 'multiner':
        dataset.convert_to_multiple_ner_files(args.dataset_out)
        return
    elif args.output_type == 'multi_classification':
        dataset.convert_to_multiple_classification_tsv_files(args.dataset_out)
        return
    with open(args.dataset_out, 'w', encoding='utf-8') as dataset_out:
        if args.output_type == 'csv':
            dataset.convert_to_csv(dataset_out)
        elif args.output_type == 'ner':
            dataset.convert_to_ner_txt(dataset_out)
        elif args.output_type == 'classification':
            dataset.convert_to_sequence_classification_tsv(dataset_out)
        else:
            raise ValueError(
                'Wrong dataset output type, expected: csv, ner, multiner, classification, multi_classification')


if __name__ == "__main__":
    main()
