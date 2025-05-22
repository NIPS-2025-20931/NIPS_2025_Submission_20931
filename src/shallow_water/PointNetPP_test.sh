#!/bin/bash

encoder=PointNetPP
sampling_method=FPS

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_test.py sampling_method=$sampling_method > ./${encoder}/test_${sampling_method}.txt