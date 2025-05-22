#!/bin/bash

DEFAULTSR=0.4
sampling_rate="${1:-$DEFAULTSR}"

encoder=PointGNN
gnlayers=4

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py graph_net_layers=$gnlayers input_sampling_rate=$sampling_rate >> ./${encoder}/${sampling_rate}_${gnlayers}.txt


