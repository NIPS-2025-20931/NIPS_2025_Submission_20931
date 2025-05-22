#!/bin/bash
DEFAULTSR=0.4
sampling_rate="${1:-$DEFAULTSR}"

encoder=PT
enc_layers=4

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py enc_layers=$enc_layers input_sampling_rate=$sampling_rate >> ./${encoder}/${sampling_rate}_${enc_layers}.txt