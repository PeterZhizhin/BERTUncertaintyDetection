import argparse
from data import hedge_dataset


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_xml', help='Path to an XML file with the dataset to convert')
    arg_parser.add_argument('dataset_csv', help='Target path where to save the result')
    return arg_parser.parse_args()


def main():
    args = parse_args()
    dataset = hedge_dataset.HedgeDataset.from_xml_file(args.dataset_xml)
    with open(args.dataset_csv, 'w', encoding='utf-8') as dataset_csv:
        dataset.convert_to_csv(dataset_csv)


if __name__ == "__main__":
    main()
