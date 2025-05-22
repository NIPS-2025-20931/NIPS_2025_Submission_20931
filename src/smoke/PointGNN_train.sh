#!/bin/bash

encoder=PointGNN
gnlayers=4

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py graph_net_layers=$gnlayers >> ./${encoder}/train_${gnlayers}.txt


