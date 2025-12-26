
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import os
import matplotlib.pyplot as plt
from utils.dataset import MyDataset
from model.Unet import Unet

import warnings
warnings.filterwarnings("ignore")  # 屏蔽警告

import numpy as np
import cv2

def tensor_to_img(t):
    """
    把Tensor (C,H,W) 转成 uint8 (H,W,3)
    """
    t = t.detach().cpu().numpy()
    if t.ndim == 3:
        t = np.transpose(t, (1, 2, 0))
    t = (t * 255).clip(0, 255).astype(np.uint8)
    return t

def save_concat_image(image, gt_mask, pred_mask, save_path):
    """
    image: 原图  Tensor
    gt_mask: 真实 mask Tensor
    pred_mask: 预测 mask Tensor
    """
    img = tensor_to_img(image)

    gt = tensor_to_img(gt_mask)
    pred = tensor_to_img(pred_mask)

    # mask 是单通道的，转成 3 通道便于拼接
    if gt.shape[2] == 1:
        gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    if pred.shape[2] == 1:
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    # 水平拼接
    concat_img = np.hstack([img, gt, pred])

    cv2.imwrite(save_path, concat_img)



def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() +mooth)

def train_net(net, device, data_path, save_path, epochs=40, batch_size=3, lr=1e-5):

    run_dir = save_path
    os.makedirs(run_dir, exist_ok=True)

    sts_dataset = MyDataset(data_path)

    train_idx, val_idx = train_test_split(range(len(sts_dataset)), test_size=0.3, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(sts_dataset, batch_size=batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(sts_dataset, batch_size=batch_size,
                            sampler=val_sampler)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    train_loss_list, val_loss_list, dice_list = [], [], []

    print_interval = 10  # 🚀 每 10 个 batch 打印一次即可

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        for i, (image, label) in enumerate(train_loader, 1):
            image = image.to(device)
            label = label.to(device)

            pred = net(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_interval == 0:  # 🚀 只每 10 个 batch 打印
                print(f"[Train] Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.5f}")

        # epoch loss
        epoch_train_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_train_loss)
        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.5f}")

        # ====== validation ======
        net.eval()
        val_loss_total, dice_total = 0.0, 0.0
        val_steps = 0

        with torch.no_grad():
            for i, (image, label) in enumerate(val_loader, 1):
                image = image.to(device)
                label = label.to(device)

                pred = net(image)
                val_loss_total += criterion(pred, label).item()
                dice_total += dice_coeff(pred, label).item()
                val_steps += 1

                # 只保存第一张图
                if i == 1:
                    pred_sigmoid = torch.sigmoid(pred)
                    pred_binary = (pred_sigmoid > 0.5).float()

                    save_path_epoch = os.path.join(run_dir, f"epoch_{epoch+1}_sample.png")
                    save_concat_image(
                        image[0],
                        label[0],
                        pred_binary[0],
                        save_path_epoch
                    )


        epoch_val_loss = val_loss_total / val_steps
        epoch_dice = dice_total / val_steps

        val_loss_list.append(epoch_val_loss)
        dice_list.append(epoch_dice)

        print(f"Epoch {epoch+1} Val Loss: {epoch_val_loss:.5f}, Dice: {epoch_dice:.5f}")

        pd.DataFrame({
            "epoch": range(1, len(train_loss_list)+1),
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "val_dice": dice_list
        }).to_csv(os.path.join(run_dir, "train_loss_log.csv"), index=False)

        #  只有当性能变好才保存模型
        if epoch_dice == max(dice_list):
            torch.save(net.state_dict(), os.path.join(run_dir, "Unet_best_model.pth"))
            print("Best model saved.")

    torch.save(net.state_dict(), os.path.join(run_dir, "Unet_last_model.pth"))

    #  训练结束后再画图（一次即可）
    plt.figure()
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))

    plt.figure()
    plt.plot(dice_list, label="dice")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "dice_curve.png"))

    print("训练结束。")




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Unet().to(device)
    data_path = "./data"  # 改成你的数据路径
    save_path = "./run1"
    train_net(net, device, data_path, save_path, epochs=40, batch_size=4, lr=1e-5)


