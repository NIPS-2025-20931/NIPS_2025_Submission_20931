#!/bin/bash

sampling_rates=(0.4 0.6 0.8 1.0)

encoder=GINO
gno_transform_types=(nonlinear)

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for sr in "${sampling_rates[@]}"
do
    for gtt in "${gno_transform_types[@]}"
    do
        stdbuf -o0 -e0 python3 -u ${encoder}_test.py gno_transform_type=$gtt input_sampling_rate=$sr > ./${encoder}/test_${sr}_${gtt}.txt
    done
done