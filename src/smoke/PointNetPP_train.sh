#!/bin/bash

encoder=PointNetPP
sampling_method=FPS

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py sampling_method=$sampling_method >> ./${encoder}/train_${sampling_method}.txt