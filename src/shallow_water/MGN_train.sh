#!/bin/bash

encoder=MGN
latent_fuse=MaxPooling
enc_layers=3
process_layer=3
message_passing_steps=4
dec_layers=3
upload_log=False

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_train.py proc_layers=$process_layer msg_passing_steps=$message_passing_steps latent_fuse=$latent_fuse enc_layers=$enc_layers dec_layers=$dec_layers upload_log=$upload_log >> ./${encoder}/train_${enc_layers}_${process_layer}_${message_passing_steps}_${dec_layers}.txt
