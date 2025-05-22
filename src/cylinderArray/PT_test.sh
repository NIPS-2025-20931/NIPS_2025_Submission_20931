#!/bin/bash

sampling_rates=(0.7 0.8 0.9 1.0)

encoder=PT
enc_layers=3

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for sr in "${sampling_rates[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_test.py enc_layers=$enc_layers input_sampling_rate=$sr > ./${encoder}/test_${sr}_${enc_layers}.txt
done