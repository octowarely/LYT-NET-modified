import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
from model import LYT
from losses import CombinedLoss
from dataloader import create_dataloaders
import os
import numpy as np

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
    
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The SSIM value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim


    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim

def main():
    # Hyperparameters
    train_low = 'data/LOLv1/Train/input'
    train_high = 'data/LOLv1/Train/target'
    test_low = 'data/LOLv1/Test/input'
    test_high = 'data/LOLv1/Test/target'
    
    # train_low = 'data/LOLv2/Real_captured/Train/Low'
    # train_high = 'data/LOLv2/Real_captured/Train/Normal'
    # test_low = 'data/LOLv2/Real_captured/Test/Low'
    # test_high = 'data/LOLv2/Real_captured/Test/Normal'

    # train_low = 'data/LOLv2/Synthetic/Train/Low'
    # train_high = 'data/LOLv2/Synthetic/Train/Normal'
    # test_low = 'data/LOLv2/Synthetic/Test/Low'
    # test_high = 'data/LOLv2/Synthetic/Test/Normal'
    learning_rate = 2e-4 
    
    # change original 1500 to 100
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    # Model, loss, optimizer, and scheduler
    model = LYT().to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0
    print('Training started.')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_psnr, avg_ssim = validate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            # 改保存的模型名称
            torch.save(model.state_dict(), 'best_model3_3.pth')
            print(f'Saving model with PSNR: {best_psnr:.6f}')

if __name__ == '__main__':
    main()
