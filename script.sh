#!/bin/bash
DATASET_PATH='/media/ivan/a68c0147-4423-4f62-8e54-388f4ace9ec54/Datasets/SHREC2022/dataset/test'

for i in $(find $DATASET_PATH -name '*.txt');
do
    python evaluation.py --file=$i --outf=./output
done