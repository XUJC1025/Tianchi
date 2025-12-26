import os
import glob
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


# RLE 解码
def rle_decode(mask_rle, shape=(512, 512)):
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None, csv_path=None):
        self.data_dir = data_dir

        # 读取 mask CSV
        csv_file = csv_path if csv_path else os.path.join(data_dir, "train_mask.csv")
        self.masks_df = pd.read_csv(csv_file, sep="\t", header=None, names=["img", "rle"])

        # 生成快速索引的 RLE 字典
        self.rle_dict = dict(zip(self.masks_df["img"], self.masks_df["rle"]))

        # 根据 CSV 构建图片路径列表
        self.img_dir = [os.path.join(data_dir, "train", img_name) for img_name in self.masks_df["img"]]

    def __getitem__(self, index):
        img_path = self.img_dir[index]
        img_name = os.path.basename(img_path)
        print(img_name)
        # 读取 image（cv2 读入是 BGR）
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        image = image.astype(np.float32) / 255.0

        # 生成 mask
        rle = self.rle_dict.get(img_name, "")
        mask = rle_decode(rle)
        mask = mask.astype(np.float32)

        # HWC → CHW
        image = image.transpose(2, 0, 1)
        mask = mask.reshape(512, 512)
        mask = np.expand_dims(mask, 0)

        image = torch.from_numpy(image).float().contiguous()
        mask = torch.from_numpy(mask).float().contiguous()

        return image, mask, img_path

    def __len__(self):
        return len(self.img_dir)
