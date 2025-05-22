#!/bin/bash
DEFAULTSR=0.9
sampling_rate="${1:-$DEFAULTSR}"

encoder=PT
enc_layers=3

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py enc_layers=$enc_layers input_sampling_rate=$sampling_rate >> ./${encoder}/${sampling_rate}_${enc_layers}.txt