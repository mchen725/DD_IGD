#!/bin/bash

spec=woof
net=resnet # resnet_ap resnet
d=10
ipc=50
path=../results/dit-distillation/woof-50-dit-igd-ckpts-convnet6-k5-gamma120-r1-gi200-low30-high45

python train.py -d imagenet --imagenet_dir ${path} ../imagenet/ -n ${net} --depth ${d} --nclass 10 --norm_type instance --ipc ${ipc} --tag test --slct_type random --spec ${spec}

