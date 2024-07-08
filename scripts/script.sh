#!/bin/bash

cd ./../src/flowms
python train.py --n_features 64 --init_channels 128 --channel_scale_factors 1 2 2 4 --batch_size 64 --n_epochs 400 --lr 1e-4
poweroff