#!/bin/bash

DEFAULTSR=0.9
sampling_rate="${1:-$DEFAULTSR}"

encoder=MGN
latent_fuse=MaxPooling
enc_layers=3
process_layer=3
message_passing_steps=4
dec_layers=3

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights


stdbuf -o0 -e0 python3 -u ${encoder}_train.py proc_layers=$process_layer msg_passing_steps=$message_passing_steps latent_fuse=$latent_fuse enc_layers=$enc_layers dec_layers=$dec_layers input_sampling_rate=$sampling_rate >> ./${encoder}/${sampling_rate}_${enc_layers}_${process_layer}_${message_passing_steps}_${dec_layers}.txt
