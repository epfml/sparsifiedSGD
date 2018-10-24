#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage: $0 data_dir output_dir"
    exit
fi

set -x

models=("rand" "top" "sgd")
cores=(1 2 3 5 8 10 12 15 18 20 24)
repeat=3

for i in `seq 1 $repeat`;
do
  for model in "${models[@]}"
  do
    for core in "${cores[@]}"
    do
      python parallel_experiment.py \
        --dataset_file="$1/rcv1.pickle" \
        --output_directory="$2" \
        --num_epochs=5 \
        --num_cores=${core} \
        --model=${model} \
        --k=50 \
        --initial_lr=1. \
        --lr=bottou || break 3;
    done
  done
done
