import os
import cv2
from utils.look_dataset  import MyDataset  # 你的 MyDataset 类所在文件
import pandas as pd




def save_image_mask_pairs(data_dir, save_dir="./data/train"):
    os.makedirs(save_dir, exist_ok=True)

    dataset = MyDataset(data_dir=data_dir)

    for i in range(0, len(dataset)):
        image, mask, img_path = dataset[i]  # image: (3,512,512), mask: (1,512,512)

        # --------------------------
        # 原图 (从 CHW → HWC，*255)
        # --------------------------
        img_np = image.numpy().transpose(1, 2, 0) * 255
        img_np = img_np[:, :, ::-1].astype("uint8")  # RGB→BGR 原样显示

        img_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img_np)

        print(f"Saved: {save_path}")

    print("All saved successfully!")




if __name__ == "__main__":
    csv_path = "origin_data/train_mask.csv"
    save_path = "data/train_mask.csv"

    # 读取 CSV
    df = pd.read_csv(csv_path, sep="\t", header=None, names=["img", "rle"])

    # 保留前 10000 行
    df_top1000 = df.iloc[:5000]

    # 保存回 CSV（保持原来的制表符格式）
    df_top1000.to_csv(save_path, sep="\t", index=False, header=False)

    print(f"前5000行已保存到 {save_path}")
    save_image_mask_pairs("./origin_data")
