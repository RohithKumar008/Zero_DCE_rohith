import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import model_denoise
import Myloss
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #this is for the filter model
    # DCE_net = model_denoise.enhance_net_denoise().cuda()
    #this is for the gpt model
    DCE_net = model_denoise.enhance_net_refine().cuda()
    DCE_net.apply(weights_init)
    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
        print(f"✅ Loaded pretrained weights from {config.pretrain_dir}")

    # Dataset & Dataloader
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Loss functions
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    DCE_net.train()

    best_loss = float('inf')
    print("🚀 Starting training...")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        DCE_net.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        start_time = time.time()

        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()

            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
            Loss_TV = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            loss = Loss_TV + loss_spa + loss_col + loss_exp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time

        print(f"🟡 [Epoch {epoch+1}/{config.num_epochs}] "
              f"Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(DCE_net.state_dict(), os.path.join(config.snapshots_folder, "best.pth"))
            print(f"✅ Saved new best model with loss {best_loss:.6f}")

    # Save last model at the end
    last_path = os.path.join(config.snapshots_folder, "last.pth")
    torch.save(DCE_net.state_dict(), last_path)
    print(f"💾 Saved final model to {last_path}")
    print("=" * 60)
    print(f"🏁 Training complete. Best Loss: {best_loss:.6f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--display_iter', type=int, default=10)  # unused but kept for compatibility
    parser.add_argument('--snapshot_iter', type=int, default=10)  # unused but kept for compatibility
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch99.pth")

    config = parser.parse_args()
    os.makedirs(config.snapshots_folder, exist_ok=True)
    train(config)