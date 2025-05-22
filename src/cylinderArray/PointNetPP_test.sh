#!/bin/bash

sampling_rates=(0.7 0.8 0.9 1.0)

encoder=PointNetPP
sampling_method=FPS

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for sr in "${sampling_rates[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_test.py sampling_method=$sampling_method input_sampling_rate=$sr > ./${encoder}/test_${sr}_${sampling_method}.txt
done