import json
import os
import platform

import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

data_transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_root = '../data/'


class COCODataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        # 加载COCO注释
        self.modename = ''
        if mode == 'train':
            self.modename = 'train2017'
        elif mode == 'val':
            self.modename = 'val2017'
        else:
            raise RuntimeError('mode not in [train, val, test]')
        self.annotation_file = os.path.join(root_dir, 'annotations', 'instances_{}.json'.format(self.modename))
        self.image_dir = os.path.join(root_dir, '{}'.format(self.modename))
        self.coco = COCO(self.annotation_file)
        self.transform = data_transform[mode]

        # 获取所有图像ID
        self.image_ids = list(self.coco.imgs.keys())

        # 创建类别映射表（ID -> 类别名称）
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat2label = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.label2cat = {i: cat['name'] for i, cat in enumerate(self.categories)}

        self.classes = self.label2cat
        with open('classes.json', 'w', encoding='utf-8') as file:
            json.dump(self.classes, file, ensure_ascii=False, indent=4)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        # 获取图像文件路径,并载入图片
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        # 应用图像变换
        if self.transform is not None:
            image = self.transform(image)

        # 获取标注信息 并载入（如果一张图像有多个类别，选择最主要的类别）
        annotation_ids = self.coco.getAnnIds(imgIds=self.image_ids[index])
        annotations = self.coco.loadAnns(annotation_ids)

        # 假设每张图像对应一个类别，选择第一个类别（可以根据任务调整）
        if len(annotations) > 0:
            main_category_id = annotations[0]['category_id']
            label = self.cat2label[main_category_id]
        else:
            # 如果没有标注，则选择一个默认的类别，例如背景
            label = -1  # 或者其他有效类别ID

        return image, label


if __name__ == '__main__':

    root_dir = '../../datasets/COCO2017'
    os_name = platform.system()
    if os_name == 'Windows':
        root_dir = root_dir.replace('/', '\\')
    if not os.path.isdir(root_dir):
        raise OSError("未找到COCO2017")

    COCO_train = COCODataset(root_dir, mode='train')
    train_loader = torch.utils.data.DataLoader(COCO_train, batch_size=1, shuffle=True)
    COCO_val = COCODataset(root_dir, mode='val')
    val_loader = torch.utils.data.DataLoader(COCO_val, batch_size=1, shuffle=True)
    pass
