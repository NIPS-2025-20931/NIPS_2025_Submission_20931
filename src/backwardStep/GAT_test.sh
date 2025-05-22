#!/bin/bash

sampling_rates=(0.4 0.6 0.8 1.0)

encoder=GAT
latent_fuse=MaxPooling
message_passing_steps=3

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

for sr in "${sampling_rates[@]}"
do
    stdbuf -o0 -e0 python3 -u ${encoder}_test.py msg_passing_steps=$message_passing_steps latent_fuse=$latent_fuse input_sampling_rate=$sr > ./${encoder}/test_${sr}_${latent_fuse}_${message_passing_steps}.txt
done