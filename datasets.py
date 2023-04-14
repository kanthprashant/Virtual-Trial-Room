import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class VITONDataset(Dataset):
    def __init__(self, dataPath, imagesList, imageSize, train=True):
        self.train = train
        self.dataPath = dataPath
        self.imageList = []
        self.clothList = []
        if self.train:
            with open(os.path.join(self.dataPath, imagesList), 'r') as f:
                for line in f:
                    self.imageList.append(line.strip())
                    self.clothList.append(line.strip())
        else:
            with open(os.path.join(self.dataPath, imagesList), 'r') as f:
                for line in f:
                    img_name, c_name = line.strip().split()
                    self.imageList.append(img_name.strip())
                    self.clothList.append(c_name.strip())
        self.transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.Resampling.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, index):
        img_name = self.imageList[index]
        cloth_name = self.clothList[index]
        if self.train:
            im_path = os.path.join(self.dataPath, 'train')
        else:
            im_path = os.path.join(self.dataPath, 'test')
        img = Image.open(os.path.join(im_path, 'image', img_name))
        img = self.transform(img)
        cimg = Image.open(os.path.join(im_path, 'clothes', cloth_name))
        cimg = self.transform(cimg)
        pose = Image.open(os.path.join(im_path, 'vis_pose', img_name.replace('.jpg', '_keypoints.jpg')))
        pose = self.transform(pose)
        img_agnostic = Image.open(os.path.join(im_path, 'img_agnostic', img_name))
        img_agnostic = self.transform(img_agnostic)

        return {
                'img_name': img_name,
                'img': img,
                'cloth': cimg,
                'img_agnostic': img_agnostic,
                'pose': pose}