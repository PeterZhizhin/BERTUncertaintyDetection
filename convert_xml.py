import argparse
from data import hedge_dataset


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_xml', help='Path to an XML file with the dataset to convert')
    arg_parser.add_argument('dataset_out', type=str, help='Target path where to save the result')
    arg_parser.add_argument('--output_type', default='xml',
                            help='Dataset output type (valid values are: xml, ner, multiner)')
    return arg_parser.parse_args()


def main():
    args = parse_args()
    dataset = hedge_dataset.HedgeDataset.from_xml_file(args.dataset_xml)
    if args.output_type == 'multiner':
        dataset.convert_to_multiple_ner_files(args.dataset_out)
        return
    with open(args.dataset_out, 'w', encoding='utf-8') as dataset_out:
        if args.output_type == 'xml':
            dataset.convert_to_csv(dataset_out)
        elif args.output_type == 'ner':
            dataset.convert_to_ner_txt(dataset_out)
        else:
            raise ValueError('Wrong dataset output type, expected: xml, ner, multiner')


if __name__ == "__main__":
    main()
