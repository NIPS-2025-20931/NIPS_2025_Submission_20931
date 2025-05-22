#!/bin/bash

sampling_rates=(0.4 0.6 0.8 1.0)

encoder=PointGNN
gnlayers=4

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for sr in "${sampling_rates[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_test.py graph_net_layers=$gnlayers input_sampling_rate=$sr > ./${encoder}/test_${sr}_${gnlayers}.txt
done
