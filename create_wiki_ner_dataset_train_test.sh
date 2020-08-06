#!/bin/bash

mkdir -p uncertainty_dataset/output_datasets/by_document
mkdir -p uncertainty_dataset/output_datasets/result/wiki

python3 convert_xml.py --output_type multiner \
  uncertainty_dataset/wiki.xml \
  "uncertainty_dataset/output_datasets/by_document/wiki_{}.txt"

./train_test_split.sh uncertainty_dataset/output_datasets/result/wiki/train.txt \
  uncertainty_dataset/output_datasets/result/wiki/dev.txt \
  75 \
  uncertainty_dataset/output_datasets/by_document/wiki_*.txt
