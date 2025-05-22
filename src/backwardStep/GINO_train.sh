#!/bin/bash

DEFAULTSR=1.0
sampling_rate="${1:-$DEFAULTSR}"

encoder=GINO
gno_transform_types=(nonlinear)

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for gno_trans in "${gno_transform_types[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_train.py gno_transform_type=$gno_trans input_sampling_rate=$sampling_rate >> ./${encoder}/${sampling_rate}_${gno_trans}.txt
done