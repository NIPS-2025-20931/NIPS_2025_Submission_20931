#!/bin/bash

encoder=GCN
latent_fuse=MaxPooling
message_passing_steps=5

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_test.py msg_passing_steps=$message_passing_steps latent_fuse=$latent_fuse > ./${encoder}/test_${latent_fuse}_${message_passing_steps}.txt