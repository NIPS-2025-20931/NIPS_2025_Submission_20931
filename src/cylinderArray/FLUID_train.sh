#!/bin/bash

DEFAULTSR=0.8
sampling_rate="${1:-$DEFAULTSR}"

encoder=FLUID

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py input_sampling_rate=$sampling_rate >> ./${encoder}/train_${sampling_rate}.txt