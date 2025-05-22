#!/bin/bash

encoder=FNO_DSE

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py >> ./${encoder}/train.txt