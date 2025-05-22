#!/bin/bash

encoder=PT
enc_layers=4

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_test.py enc_layers=$enc_layers > ./${encoder}/test_${enc_layers}.txt
