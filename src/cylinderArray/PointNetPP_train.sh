#!/bin/bash
DEFAULTSR=0.9
sampling_rate="${1:-$DEFAULTSR}"

encoder=PointNetPP
sampling_method=FPS

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py sampling_method=$sampling_method input_sampling_rate=$sampling_rate >> ./${encoder}/${sampling_rate}_${sampling_method}.txt
