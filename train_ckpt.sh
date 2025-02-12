#!/bin/bash
model=convnet6
spec=woof

python train_ckpts.py -d imagenet --imagenet_dir ../imagenet/train/ ../imagenet/ \
    -n ${model} --nclass 10 --norm_type instance --tag test --slct_type random --spec ${spec} --mixup vanilla --repeat 1 \
    --ckpt-dir ./ckpts/${spec}/${model}/ --epochs 50 --depth 6 --ipc -1
