#!/bin/bash

k=10
low=30
high=45
gi=200
cp=ckpts
r=1
spec=woof
nsample=50
ntype=convnet6
gamma=100
phase=0 # end 7
nclass=10
tart_ncls=10
d=6

python sample_mp.py --model DiT-XL/2 --image-size 256 --ckpt path/to/minimax/checkpoint \
    --save-dir "../results/dit-distillation/${spec}-${nsample}-minimax-igd-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path ../imagenet/train/ \
    --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth ${d} \
    --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --dev-scale ${gamma} \
    --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls}
    

