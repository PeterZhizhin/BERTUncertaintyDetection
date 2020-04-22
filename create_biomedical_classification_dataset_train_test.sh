#!/bin/bash

mkdir -p uncertainty_dataset/output_datasets/by_document/classification
mkdir -p uncertainty_dataset/output_datasets/result/classification/bio

python3 convert_xml.py --output_type multi_classification \
  uncertainty_dataset/bio_bmc.xml \
  "uncertainty_dataset/output_datasets/by_document/classification/bio_bmc_{}.tsv"

python3 convert_xml.py --output_type multi_classification \
  uncertainty_dataset/bio_fly.xml \
  "uncertainty_dataset/output_datasets/by_document/classification/bio_fly_{}.tsv"

python3 convert_xml.py --output_type multi_classification \
  uncertainty_dataset/bio_hbc.xml \
  "uncertainty_dataset/output_datasets/by_document/classification/bio_hbc_{}.tsv"

./train_test_split.sh uncertainty_dataset/output_datasets/result/classification/bio/train.tsv \
  uncertainty_dataset/output_datasets/result/classification/bio/dev.tsv \
  75 \
  uncertainty_dataset/output_datasets/by_document/classification/bio_*.tsv
