#!/bin/bash

DEFAULTSR=0.9
sampling_rate="${1:-$DEFAULTSR}"

encoder=GCN
latent_fuse=MaxPooling
message_passing_steps=8

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py msg_passing_steps=$message_passing_steps latent_fuse=$latent_fuse input_sampling_rate=$sampling_rate >> ./${encoder}/${sampling_rate}_${latent_fuse}_${message_passing_steps}.txt