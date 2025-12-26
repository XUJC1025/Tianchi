import os
import cv2
import torch
import numpy as np
from utils.look_dataset  import MyDataset  # 你的 MyDataset 类所在文件

def save_image_mask_pairs(data_dir, save_dir="saved_pairs", num=10):
    os.makedirs(save_dir, exist_ok=True)

    dataset = MyDataset(data_dir=data_dir)

    for i in range(0, len(dataset)):
        image, mask, img_path = dataset[i]  # image: (3,512,512), mask: (1,512,512)

        # --------------------------
        # 原图 (从 CHW → HWC，*255)
        # --------------------------
        img_np = image.numpy().transpose(1, 2, 0) * 255
        img_np = img_np[:, :, ::-1].astype("uint8")  # RGB→BGR 原样显示

        # --------------------------
        # mask 图 (单通道 → 3通道灰度)
        # --------------------------

        mask_np = mask.squeeze(0).numpy() * 255
        mask_np = mask_np.astype("uint8")
        mask_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)

        # --------------------------
        # 左右拼接
        # --------------------------
        pair = np.hstack([img_np, mask_3ch])
        img_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, f"pair_{img_name}.png")
        cv2.imwrite(save_path, pair)

        print(f"Saved: {save_path}")

    print("All saved successfully!")

if __name__ == "__main__":
    save_image_mask_pairs("./data", num=10)
