#!/bin/bash

sampling_rates=(0.7 0.8 0.9 1.0)

encoder=FLUID

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for sampling_rate in "${sampling_rates[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_test.py input_sampling_rate=$sampling_rate > ./${encoder}/test_${sampling_rate}.txt
done