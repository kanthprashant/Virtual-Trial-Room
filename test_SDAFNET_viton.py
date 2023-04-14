### Based on https://github.com/OFA-Sys/DAFlow

import argparse
import os
import torch.backends.cudnn as cudnn
import torch
from torch.nn import functional as F
import tqdm
from datasets import VITONDataset
from networks.sdafnet import SDAFNet_Tryon
from torch.utils import data
from torchvision.utils import save_image
from collections import OrderedDict
cudnn.benchmark = True

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default = 192)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--add_compare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--dataset_imgpath', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_data.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')
    opt = parser.parse_args()
    return opt

def test(opt, net):
    test_dataset = VITONDataset(opt.dataset_dir, opt.dataset_list, (opt.load_height, opt.load_width), opt.train)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)
    with torch.no_grad():
        for i, inputs in enumerate(tqdm.tqdm(test_loader)):
            img_names = inputs['img_name']
            img = inputs['img'].cuda()
            img_agnostic = inputs['img_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            cloth_img = inputs['cloth'].cuda()
            img =  F.interpolate(img, size=(256, 192), mode='bilinear')
            ref_input = torch.cat((pose, img_agnostic), dim=1)
            tryon_result = net(ref_input, cloth_img, img_agnostic).detach().cpu()
            if opt.add_compare:
                tryon_result = torch.cat([img_agnostic.detach().cpu(), tryon_result],2)
                save_image(tryon_result, os.path.join(opt.save_dir, opt.name, "vis_custom_out", "agn"+img_names[0]), nrow=10, normalize=True, range=(-1,1))
                tryon_result = torch.cat([img.detach().cpu(), cloth_img.detach().cpu(), tryon_result],2)
                save_image(tryon_result, os.path.join(opt.save_dir, opt.name, "vis_custom_out", img_names[0]), nrow=10, normalize=True, range=(-1,1))
            else:
                for j in range(tryon_result.shape[0]):
                    save_image(tryon_result[j:j+1], os.path.join(opt.save_dir, opt.name, "vis_custom_out", img_names[j]), nrow=1, normalize=True, range=(-1,1))

def main():
    opt = get_opt()
    print(opt)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name, "vis_custom_out")):
        os.makedirs(os.path.join(opt.save_dir, opt.name, "vis_custom_out"))
    sdafnet = SDAFNet_Tryon(ref_in_channel=6)
    ### two ways to load model
    ### first
    sdafnet = sdafnet.cuda()
    state_dict = torch.load("./checkpoints/openpose_finetune.pt")['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    sdafnet.load_state_dict(new_state_dict)
    ### second
    # state_dict = torch.load("./checkpoints/mediapipe_finetune.pt")['state_dict']
    # sdafnet.load_state_dict(state_dict)
    # sdafnet.cuda()
    sdafnet.eval()
    test(opt, sdafnet)

if __name__ == '__main__':
    main()
