#!/bin/bash

mkdir -p uncertainty_dataset/output_datasets/by_document/classification
mkdir -p uncertainty_dataset/output_datasets/result/classification/wiki

python3 convert_xml.py --output_type multi_classification \
  uncertainty_dataset/wiki.xml \
  "uncertainty_dataset/output_datasets/by_document/classification/wiki_{}.tsv"

./train_test_split.sh uncertainty_dataset/output_datasets/result/classification/wiki/train.tsv \
  uncertainty_dataset/output_datasets/result/classification/wiki/dev.tsv \
  75 \
  uncertainty_dataset/output_datasets/by_document/classification/wiki_*.tsv
