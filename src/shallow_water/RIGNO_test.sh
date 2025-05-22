#!/bin/bash

encoder=RIGNO

mkdir -p ${encoder}
mkdir -p ${encoder}/outputs
mkdir -p ${encoder}/weights

stdbuf -o0 -e0 python3 -u ${encoder}_test.py > ./${encoder}/test.txt
