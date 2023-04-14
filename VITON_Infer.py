import os
import cv2
from copy import deepcopy
import base64
import torch
import argparse
import networks
import numpy as np
import torch.nn as nn
import mediapipe as mp
import torch.nn.functional as F
from io import BytesIO
from PIL import Image, ImageOps
from utils.transforms import transform_logits
from input_formatter import human_parser_input, blazepose_keypoints, get_img_agnostic, draw_keypoints
from torchvision import transforms
from torchvision.utils import save_image
from collections import OrderedDict

class InferVITON:
    def __init__(self,):
        self.output_size = [256, 192]
        self.lip_input_size = [473, 473]
        self.lip_num_classes = 20
        self.lip_labels = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                           'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                           'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
        self.devices = list(range(torch.cuda.device_count()))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.devices[0])
        
        self.parser_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
                ])
        self.parser = human_parser_input.HumanParsing(self.lip_input_size, self.parser_transform)
        self.parser_model =networks.init_model('resnet101', num_classes=self.lip_num_classes, pretrained=None)
        state_dict = torch.load('./checkpoints/lip.pth')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.parser_model.load_state_dict(new_state_dict)

        self.mp_pose = mp.solutions.pose

        self.sdafnet_transform = transforms.Compose([
                transforms.Resize(self.output_size, interpolation=Image.Resampling.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        self.sdafnet = networks.init_model('sdafnet', ref_in_channel=6)
        self.sdafnet.cuda()
        # self.sdafnet.load_state_dict(torch.load('./checkpoints/ckpt_viton.pt'))
        # self.sdafnet.load_state_dict(torch.load('./checkpoints/mediapipe_train_0409.pt')['state_dict'])
        sdafnet_state_dict = torch.load('./checkpoints/openpose_finetune.pt')['state_dict']
        new_sdafnet_state_dict = OrderedDict()
        for k, v in sdafnet_state_dict.items():
            name = k[7:]  # remove `module.`
            new_sdafnet_state_dict[name] = v
        self.sdafnet.load_state_dict(new_sdafnet_state_dict)

        self.parser_model.cuda()
        self.parser_model.eval()
        
        self.sdafnet.cuda()
        self.sdafnet.eval()
        
        print(f"Model ready to infer!!")
    
    def tensor2img(self, img_tensor):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.clamp(0,255).permute(1,2,0)
        array = tensor.numpy().astype('uint8')
        im = Image.fromarray(array)
        return im

    def resize(self, img):
        """
            PIL Image
        """
        print(img.mode)
        img = np.asarray(img)
        img = cv2.resize(img, (192, 256), cv2.INTER_AREA)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def get_palette(self, num_cls):
        """ Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def get_agnostic_initials(self, img_name, img_path):
        image, meta = self.parser.get_input(img_path)
        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']

        # palette = self.get_palette(self.lip_num_classes)
        with torch.no_grad():
            output = self.parser_model(image.unsqueeze(0).cuda())
            upsample = nn.Upsample(size=self.lip_input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.lip_input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parse = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            # parse.putpalette(palette)
            # parse.save("./Output/parsed_image.png") # put back

        parse = parse.resize(self.output_size[::-1], Image.Resampling.LANCZOS).convert('L')

        pose_data = blazepose_keypoints.make_json(self.mp_pose, [img_path], save_json=False)
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))
        return parse, pose_data
    
    def infer(self, img_path, cloth_img_path):
        img_name = img_path.split('/')[-1]
        
        parse, pose_data = self.get_agnostic_initials(img_name, img_path)
        # parse = parse.convert('L')
        parse = transforms.Resize(self.output_size[1], interpolation=0)(parse)

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img = img.resize(self.output_size[::-1], Image.Resampling.LANCZOS)

        # img.save(os.path.join(args.out_path, "resize_image.jpg"))

        cloth_img = Image.open(cloth_img_path)
        cloth_img = cloth_img.resize(self.output_size[::-1], Image.Resampling.LANCZOS)

        agnostic = get_img_agnostic.get_img_agnostic(img, parse, pose_data[:, :2])
        agnostic = agnostic.convert('RGB')
        agnostic.save(os.path.join('./Output', 'agnostic_'+img_name))

        pose = draw_keypoints.get_poseimg(pose_data)
        pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
        # pose = Image.fromarray(pose.astype(np.uint8))
        pose = Image.fromarray(pose)

        pose.save(os.path.join('./Output', 'pose_'+img_name))

        agnostic = self.sdafnet_transform(agnostic).unsqueeze(0)
        pose = self.sdafnet_transform(pose).unsqueeze(0)
        cloth_img = self.sdafnet_transform(cloth_img).unsqueeze(0)
        with torch.no_grad():
            ref_input = torch.cat((pose, agnostic), dim=1)
            tryon_result = self.sdafnet(ref_input.cuda(), cloth_img.cuda(), agnostic.cuda()).detach().cpu()
        save_image(tryon_result, os.path.join('./Output', img_name), nrow=1, normalize=True, range=(-1,1))

        im_out = self.tensor2img(tryon_result.squeeze(0))
        # im_out.save(os.path.join(args.out_path, "saved_befoer_encoding.jpg"))
        im_file = BytesIO()
        im_out.save(im_file, format="JPEG")
        im_b64 = base64.b64encode(im_file.getvalue())
        return im_b64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='', help="path of the person image")
    parser.add_argument('--cloth_path',  type=str, default='', help="path of the cloth image")
    parser.add_argument('--out_path', type=str, default='./Output', help="path to save intermediate images")
    args = parser.parse_args()

    os.makedirs("./Output", exist_ok=True)
    os.makedirs(args.out_path, exist_ok=True)
    
    inferviton = InferVITON()

    im_b64 = inferviton.infer(args.img_path, args.cloth_path)
    im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    out = Image.open(im_file)   # img is now PIL Image object
    out.save(os.path.join(args.out_path, "decoded_output.jpg"))
    print(f"Output generated!")
