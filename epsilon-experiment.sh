#!/bin/bash
set -x

models=("top" "sgd")
cores=(1 2 3 5 10 15)
for model in "${models[@]}"
do
  for core in "${cores[@]}"
  do
    python parallel_experiment.py \
      --dataset_file='/mlodata1/jb/data/epsilon.pickle' \
      --output_directory='results/new_par_epsilon/' \
      --num_epochs=10 \
      --num_cores=${core} \
      --model=${model} \
      --k=10 \
      --initial_lr=1. \
      --lr=bottou || break;
  done
done
