"""
Implementation of the gradient-similarity-based checkpoint selection algorithm.
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data import ImageFolder
from collections import OrderedDict, defaultdict
from PIL import Image
import numpy as np
import gc
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
import time
from reparam_module import ReparamModule
import torch.nn as nn

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """

    args.size = 256
    args.width = 1.0
    args.norm_type = 'instance'
    args.nch = 3

    if size == None:
        size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.spec,
                          args.depth,
                          nclass,
                          norm_type=args.norm_type,
                          size=size,
                          nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.spec,
                              args.depth,
                              nclass,
                              width=args.width,
                              norm_type=args.norm_type,
                              size=size,
                              nch=args.nch)

    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        width = int(128 * args.width)
        model = CN.ConvNet(channel=4, num_classes=nclass, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))

        model.classifier = nn.Linear(2048, nclass)
    elif args.net_type == 'convnet6':
        width = int(128 * args.width)
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=6, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
    elif args.net_type == 'convnet4':
        width = int(128 * args.width)
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=4, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")

    return model

def rand_ckpts(args):
    expert_path = './%s/%s/%s/'%(args.ckpt_path, args.spec, args.net_type)
    expert_files = os.listdir(expert_path)
    # rand_id1 = np.random.choice(len(expert_files))
    rand_id1 = 0
    state = torch.load(expert_path + expert_files[rand_id1])
    # ckpts = state[np.random.choice(len(state))]
    ckpts = state[0]
    print('expert_path',expert_path)
    print(' file', expert_files[rand_id1])

    # select_idxs = np.arange(0, len(ckpts), 20).tolist()
    # # select_idxs = np.random.choice(int(len(ckpts)*0.6),size=5)
    # # print('select_idxs',select_idxs)
    # ckpts = [ckpts[idx] for idx in select_idxs]

    return ckpts


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    # torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == '1k':
        file_list = './misc/class_indices.txt'
    else:
        file_list = './misc/class100.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))


    batch_size = 1

    # Setup data:
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    print('spec',args.spec)
    dataset = ImageFolder(args.data_path, transform=transform, nclass=args.nclass,
                          ipc=args.real_ipc, spec=args.spec, phase=args.phase,
                          seed=0, return_origin=True)
    # dataset_real = ImageFolder(args.data_path, transform=transform, nclass=args.nclass,
    #                       ipc=args.finetune_ipc, spec=args.spec, phase=args.phase,
    #                       seed=0, slct_type='loss', return_origin=True)

    real_loader = DataLoader(
        dataset,
        batch_size=500,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    print(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    # define gm resources 
    args.device = 'cuda'
    surrogate = define_model(args,args.nclass).to(args.device)  
    surrogate = ReparamModule(surrogate)
    # if args.eval:
    surrogate.eval()
    # surrogate.train()
    ckpts = rand_ckpts(args)

    grad_memory = defaultdict(list)
    # real_imgs =  defaultdict(list)
    correspond_labels = defaultdict(list)


    print('load real grad memory ')
    for x, ry, y in real_loader: # ry 是0-1的，y是在ori 1000个里的真实index
        # ry = ry.numpy()
        x = x.to(device)
        y = y.numpy()

        y_set = set(y)
        
        for c in y_set:
            if len(grad_memory[c]) > args.grad_ipc:
                continue
            # Update the auxiliary memories
            grad_memory[c].extend(x[y == c].detach().cpu().split(1))
            # real_imgs[c].extend(x[y == c].detach().cpu().split(1))
            correspond_labels[c] = ry[y==c][0].cpu().numpy()
        all_len = [len(lst) for lst in grad_memory.values()]
        print(all_len)
        # print('all_len',all_len)
        # if all_len>=args.nclass*args.grad_ipc:
        if min(all_len)>args.grad_ipc:
            break

    for y in grad_memory.keys():
        grad_memory[y] = grad_memory[y][:args.grad_ipc]
    all_len = [len(lst) for lst in grad_memory.values()]
    print('grad memory', all_len)

    
    # gap = args.grad_ipc // 4 if args.grad_ipc <= 100 else 25
    gap = 100
    gap_idxs = np.arange(0, min(all_len), gap).tolist()
    print('gap_idxs',gap_idxs)
    print('start obtain real grad memory for influence function')

    cur_params = torch.cat([p.data.to('cuda').reshape(-1) for p in ckpts[0]], 0).requires_grad_(True)
    pre_grads = cal_each_cls_grad(surrogate,cur_params,grad_memory,correspond_labels,gap_idxs,gap)
    comp_idx = 0
    threshold = 0.7
    print('threshold',threshold)
    for idx in [int(i) for i in range(1,len(ckpts))]:
        cls_grad_sim = 0
        ckpt = ckpts[idx]
        cur_params = torch.cat([p.data.to('cuda').reshape(-1) for p in ckpt], 0).requires_grad_(True)
        cur_grads = cal_each_cls_grad(surrogate,cur_params,grad_memory,correspond_labels,gap_idxs,gap)
        for y in grad_memory.keys():
            pre_grad = pre_grads[y].cuda()
            cur_grad = cur_grads[y].cuda()
            cls_grad_sim += cos_simf(pre_grad, cur_grad)
        cls_grad_sim /= len(grad_memory.keys())

        print('cos sim between:',comp_idx,',',idx)
        print('grad cos sim', cls_grad_sim)
        # comp_idx = idx
        # pre_grads = cur_grads

        if cls_grad_sim < threshold:
            comp_idx = idx
            pre_grads = cur_grads
            print('!!!! lower than threshold %s !!!!'%threshold)

def cos_simf(pg, rg):
    inv_gm_dist = torch.sum(pg * rg) / (torch.norm(pg) * torch.norm(rg) + 0.000001)
    return inv_gm_dist

def cal_each_cls_grad(surrogate,cur_params,grad_memory,correspond_labels,gap_idxs,gap):
    real_gradients =  defaultdict(list)
    criterion_ce = nn.CrossEntropyLoss().to(args.device)

    for y in grad_memory.keys():
        # idxs = np.arange()
        correspond_y = correspond_labels[y]   
        for gi in gap_idxs:
            # print(gi)
            # print(grad_memory[y][gi:gi+gap])
            cur_embd0 = torch.stack(grad_memory[y][gi:gi+gap]).cpu().numpy()
            cur_embds = torch.from_numpy(cur_embd0).squeeze(1).to('cuda').requires_grad_(True)
            # print('111',cur_imgs.shape)
            if gi == 0:
                acc_grad = torch.zeros(cur_params.shape)
            real_pred = surrogate(cur_embds, flat_param=cur_params)
            real_target = torch.tensor([np.ones(len(cur_embds))*correspond_y], dtype=torch.long, device=args.device).view(-1) 
            # print('real_pred',real_pred)
            real_loss = criterion_ce(real_pred, real_target)
            # print('real_loss',real_loss)
            real_grad = torch.autograd.grad(real_loss, cur_params)[0] #.detach().clone().requires_grad_(False)
            # print('real_grad',real_grad)
            acc_grad += real_grad.detach().data.cpu()
            surrogate.zero_grad()
        real_gradients[y]=acc_grad / len(gap_idxs)

    return real_gradients


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument("--memory-size", type=int, default=64, help='the memory size')
    parser.add_argument("--real_ipc", type=int, default=500, help='the number of samples participating in the fine-tuning')
    parser.add_argument("--grad-ipc", type=int, default=80, help='the number of samples participating in the fine-tuning')
    parser.add_argument('--lambda-pos', default=0.03, type=float, help='weight for representativeness constraint')
    parser.add_argument('--lambda-neg', default=0.01, type=float, help='weight for diversity constraint')
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--net-type", type=str, default='convnet6')
    parser.add_argument("--gm-scale", type=float, default=0.02)
    parser.add_argument("--depth", type=int, default=10, help='the depth of the network')
    parser.add_argument("--low", type=int, default=500, help='allowed lowest time step for gm guidance')
    parser.add_argument("--high", type=int, default=800, help='allowed highest time step for gm guidance')
    args = parser.parse_args()
    main(args)
