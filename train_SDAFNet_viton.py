### Based on https://github.com/OFA-Sys/DAFlow

import os
import math
import tqdm
import random
import torch
import argparse
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from datasets import VITONDataset
from networks.sdafnet import SDAFNet_Tryon
from networks import external_function
from utils import lpips
from utils.utils import AverageMeter, weights_init
from torchvision import transforms, utils
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="name to identify training iteration")
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=192)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard')
    parser.add_argument('--dataset_imgpath', type=str, default='train')
    parser.add_argument('--dataset_list', type=str, default='train_data.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--finetune', action="store_true", help="finetune model using pretrained weights")
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--display_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--multi_flows', type=int, default=6)

    opt = parser.parse_args()
    return opt

def train(opt, net, gpus):
    train_dataset = VITONDataset(opt.dataset_dir, opt.dataset_list, (opt.load_height, opt.load_width), True)
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=opt.shuffle, num_workers=opt.workers)
    #Tensorboard
    summary = SummaryWriter(os.path.join(opt.tensorboard_dir, opt.name))
    #criterion
    criterion_L1 = nn.L1Loss()
    criterion_percept = lpips.exportPerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
    criterion_style = external_function.VGGLoss().cuda()
    #optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=opt.lr)
    #scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    epoches = 200
    iterations = 0
    for epoch in range(epoches):
        loss_l1_avg = AverageMeter()
        loss_vgg_avg = AverageMeter()
        for i, inputs in enumerate(tqdm.tqdm(train_loader, position=0, leave=True)):
            iterations+=1
            img = inputs['img'].cuda()
            img_agnostic = inputs['img_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            cloth_img = inputs['cloth'].cuda()
            img =  F.interpolate(img, size=(256, 192), mode='bilinear')
            ref_input = torch.cat((pose, img_agnostic), dim=1)
            
            return_all = True                                                                       
            inputs = (ref_input, cloth_img, img_agnostic, return_all)                               
            result_tryon, results_all = nn.parallel.data_parallel(net, inputs, gpus)    

            epsilon = 0.001
            loss_all = 0
            num_layer = 5
            for num in range(num_layer):
                cur_img = F.interpolate(img, scale_factor=0.5**(4-num), mode='bilinear')
                loss_l1 = criterion_L1(results_all[num], cur_img.cuda())
                if num == 0:
                    cur_img = F.interpolate(cur_img, scale_factor=2, mode='bilinear')
                    results_all[num] = F.interpolate(results_all[num], scale_factor=2, mode='bilinear')
                loss_perceptual = criterion_percept(cur_img.cuda(),results_all[num]).mean()
                loss_content, loss_style = criterion_style(results_all[num], cur_img.cuda())
                loss_vgg = loss_perceptual + 100*loss_style + 0.1*loss_content
                loss_all = loss_all + (num+1) * loss_l1 + (num + 1)  * loss_vgg
            loss = loss_all
            loss_l1_avg.update(loss.item())
            loss_vgg_avg.update(loss_vgg.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations%opt.display_freq ==0:
                parse_pred = torch.cat([cloth_img.cpu(), 
                                        pose.cpu(),
                                        img_agnostic.cpu(),
                                        img.cpu(),
                                        result_tryon.cpu(), 
                                        ], 2)
                parse_pred = F.interpolate(parse_pred, size=(1280, 192), mode='bilinear')
                utils.save_image(
                        parse_pred,
                        f"{os.path.join(opt.save_dir, opt.name, 'log_sample')}/{str(iterations).zfill(6)}_daf_viton_sample_{str(opt.name)}.jpg",
                        nrow=6,
                        normalize=True,
                        range=(-1, 1),
                    )
                summary.add_scalars('Losses', {'L1': loss_l1.item(), 'VGG': loss_vgg.item()}, iterations)
                summary.add_scalars('Total Loss', {'Total': loss_all.item()}, iterations)

            print("[%d %d][%d] l1_loss:%.4f l1_loss_avg:%.4f vgg_loss:%.4f vgg_loss_avg:%.4f "%(epoch,epoches,iterations,loss_l1.item(),loss_l1_avg.avg,loss_vgg.item(),loss_vgg_avg.avg))

        if (epoch+1)%opt.save_freq ==0:
            torch.save(
                {
                    "epoch": epoch,
                    "opt": opt,
                    "state_dict": net.state_dict()
                },
                os.path.join(opt.save_dir, f"saved_checkpoints/{str(iterations).zfill(6)}_daf_viton_s1fine_{str(opt.name)}.pt"),
            )
            
        scheduler.step()

def main():
    opt = get_opt()
    print(opt)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.tensorboard_dir, opt.name))
        os.makedirs(os.path.join(opt.save_dir, 'saved_checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(opt.save_dir, opt.name, 'log_sample'), exist_ok=True)
    
    gpus = list(range(torch.cuda.device_count()))   
    torch.cuda.set_device(gpus[0])                  
    cudnn.benchmark = True                          

    sdafnet = SDAFNet_Tryon(ref_in_channel=6)
    if opt.finetune:
        sdafnet.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, "ckpt_viton.pt"))) 
        print(f"Weights loaded from: {opt.checkpoint_dir}/ckpt_viton.pt")
    # else:
    #     sdafnet.apply(weights_init)
    sdafnet.cuda()
    sdafnet.train()
    
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    train(opt, sdafnet, gpus)   

if __name__ == '__main__':
    main()
