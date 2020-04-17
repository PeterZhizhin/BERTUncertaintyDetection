#!/bin/bash

mkdir -p uncertainty_dataset/output_datasets/by_document
mkdir -p uncertainty_dataset/output_datasets/result/bio

python3 convert_xml.py --output_type multiner \
  uncertainty_dataset/bio_bmc.xml \
  "uncertainty_dataset/output_datasets/by_document/bio_bmc_{}.txt"

python3 convert_xml.py --output_type multiner \
  uncertainty_dataset/bio_fly.xml \
  "uncertainty_dataset/output_datasets/by_document/bio_fly_{}.txt"

python3 convert_xml.py --output_type multiner \
  uncertainty_dataset/bio_hbc.xml \
  "uncertainty_dataset/output_datasets/by_document/bio_hbc_{}.txt"

./train_test_split.sh uncertainty_dataset/output_datasets/result/bio/train.txt \
  uncertainty_dataset/output_datasets/result/bio/dev.txt \
  75 \
  uncertainty_dataset/output_datasets/by_document/bio_*.txt
