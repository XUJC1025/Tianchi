import os
import torch
import cv2
import numpy as np
import pandas as pd
from model.Unet import Unet

# ---------- RLE 编码 ----------
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)




# ---------- 测试推理函数 ----------
def predict(weight_path, test_csv, image_dir, save_csv, input_size=(512, 512), device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    net = Unet(in_channels=3, n_classes=1)
    state_dict = torch.load(weight_path, map_location=device)
    print(state_dict.keys())
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # 创建 mask 输出文件夹 -------------------- 新增
    save_mask_dir = "test_predict"
    os.makedirs(save_mask_dir, exist_ok=True)
    # ----------------------------------------

    # 读取 CSV
    test_df = pd.read_csv(test_csv, sep=None, engine='python', header=None)
    test_images = test_df.iloc[:, 0].astype(str).str.strip().tolist()

    print(f"开始测试，共 {len(test_images)} 张图片")

    results = []
    processed = 0

    for img_name in test_images:
        processed += 1
        print(f"[{processed}/{len(test_images)}] processing {img_name}", end='\r')

        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            results.append({"image_name": img_name, "rle_mask": ""})
            continue

        # 读取图片
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        h, w, _ = img.shape

        # resize
        img_input = img.transpose(2, 0, 1)  # (3,512,512)
        img_input = np.expand_dims(img_input, 0)  # (1,3,512,512)
        img_tensor = torch.from_numpy(img_input).float().to(device)

        # 推理
        with torch.no_grad():
            pred = net(img_tensor)
            pred_sigmoid = torch.sigmoid(pred)[0, 0].cpu().numpy()

        # 阈值化 mask
        mask = (pred_sigmoid > 0.5).astype(np.uint8)

        # ---------- 保存 mask 图片（新增） ----------
        mask_path = os.path.join(save_mask_dir, img_name.replace('.jpg', '.png'))
        cv2.imwrite(mask_path, mask * 255)  # 转成可视化白色区域
        # ------------------------------------------------

        # RLE 编码
        rle = rle_encode(mask) if mask.sum() > 0 else ""

        results.append({"image_name": img_name, "rle_mask": rle})

    print("\n测试完成，正在保存 train_mask.csv")

    # 保存 train_mask.csv
    result_df = pd.DataFrame(results)
    result_df.to_csv(
        save_csv,
        index=False,
        header=False,
        sep='\t',  # 这里是制表符
        quoting=3  # csv.QUOTE_NONE
    )
    print(f"结果已保存到 {save_csv}")

# ---------- 主运行 ----------
if __name__ == "__main__":
    predict(
        weight_path="run/Unet_best_model.pth",
        test_csv="data/test_a_samplesubmit.csv",
        image_dir="data/test_a/",
        save_csv="result.csv",
        input_size=(512, 512)
    )
