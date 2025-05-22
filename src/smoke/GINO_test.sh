#!/bin/bash

encoder=GINO
gno_transform_types=(nonlinear)

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for gtt in "${gno_transform_types[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_test.py gno_transform_type=$gtt > ./${encoder}/test_${gtt}.txt
done