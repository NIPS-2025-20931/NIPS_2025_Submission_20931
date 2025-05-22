#!/bin/bash

DEFAULTSR=0.9
sampling_rate="${1:-$DEFAULTSR}"

encoder=RIGNO

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py input_sampling_rate=$sampling_rate >> ./${encoder}/train_${sampling_rate}.txt