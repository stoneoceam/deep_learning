import os
import platform

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from VOCSegDataset.utils import voc_colorize

base_size = 520
crop_size = 480

min_size = int(0.5 * base_size)
max_size = int(2.0 * base_size)

data_transform = {
    'train': transforms.Compose([
        transforms.RandomCrop(min_size, max_size),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.RandomCrop(base_size, base_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", mode='train'):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_name = mode + '.txt'
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        img = data_transform['train'](img)
        target = data_transform['val'](target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    root_dir = "../../datasets"
    os_name = platform.system()
    if os_name == 'Windows':
        root_dir = root_dir.replace('/', '\\')

    VOC_train = VOCSegmentation(voc_root=root_dir, mode='train')
    train_loader = torch.utils.data.DataLoader(VOC_train, batch_size=4, collate_fn=VOCSegmentation.collate_fn)

    VOC_val = VOCSegmentation(voc_root=root_dir, mode='val')
    val_loader = torch.utils.data.DataLoader(VOC_val, batch_size=4, collate_fn=VOCSegmentation.collate_fn)

    example_image = train_loader.dataset.images[0]
    example_mask = train_loader.dataset.masks[0]
    image = np.array(Image.open(example_image))
    mask = np.array(Image.open(example_mask))  # 加载mask图像，像素值为类别索引
    color_mask = voc_colorize(mask)  # 转换为彩色图

    # 可视化原始mask和彩色mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title("Original Mask (Grayscale)")
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Colorized Mask")
    plt.imshow(color_mask)

    plt.show()
