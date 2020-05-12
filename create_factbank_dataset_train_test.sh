#!/bin/bash

mkdir -p uncertainty_dataset/output_datasets/by_document/classification
mkdir -p uncertainty_dataset/output_datasets/result/classification/factbank

python3 convert_xml.py --output_type multi_classification \
  uncertainty_dataset/factbank.xml \
  "uncertainty_dataset/output_datasets/by_document/classification/factbank_{}.tsv"

./train_test_split.sh uncertainty_dataset/output_datasets/result/classification/factbank/train.tsv \
  uncertainty_dataset/output_datasets/result/classification/factbank/dev.tsv \
  0 \
  uncertainty_dataset/output_datasets/by_document/classification/factbank_*.tsv

mkdir -p uncertainty_dataset/output_datasets/by_document
mkdir -p uncertainty_dataset/output_datasets/result/factbank

python3 convert_xml.py --output_type multiner \
  uncertainty_dataset/factbank.xml \
  "uncertainty_dataset/output_datasets/by_document/factbank_{}.txt"

./train_test_split.sh uncertainty_dataset/output_datasets/result/factbank/train.txt \
  uncertainty_dataset/output_datasets/result/factbank/dev.txt \
  0 \
  uncertainty_dataset/output_datasets/by_document/factbank_*.txt
