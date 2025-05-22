#!/bin/bash

encoder=GINO
# gno_transform_types=(linear nonlinear_kernelonly nonlinear)
gno_transform_types=(nonlinear)

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for gno_trans in "${gno_transform_types[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_train.py gno_transform_type=$gno_trans >> ./${encoder}/train_${gno_trans}.txt
done