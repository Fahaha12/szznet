import argparse
import logging
import math
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import MultiModalDataset  # 自定义的数据集类
from model.SwinTransformerForRegression import SwinTransformerForRegression
import matplotlib.pyplot as plt
from utils.classification_data import LungDatasetForRegression
import os
import re
from sklearn.metrics import r2_score, mean_absolute_error
from utils.regutils import calculate_metrics, plot_loss, save_checkpoint, load_checkpoint, find_latest_checkpoint
from torch.utils.tensorboard import SummaryWriter


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='MultiViewBreastCancerDetection')
    parser.add_argument('--data-path', default='BraTS', type=str, help='data path')
    parser.add_argument('--out-path', default='result', help='directory to output the result')
    parser.add_argument('--txt-path', default='Output', help='directory to output the result')
    parser.add_argument('--gpu-id', default=0, type=int, help='visible gpu id(s)')
    parser.add_argument('--num-workers', default=10, type=int, help='number of workers')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total steps to run')
    parser.add_argument('--labeled_ratio', default=0.1, type=float, help='labeled data ratio')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', action='store_true', help='use consistency loss')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS', help='length of the consistency loss ramp-up')
    parser.add_argument('--warm_up_epochs', default=10, type=int, help='number of warm up epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='train batch_size')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--input-width', default=224, type=int, help='the width of input images')
    parser.add_argument('--input-height', default=224, type=int, help='the height of input images')
    parser.add_argument('--model_type', default='swin_transformer_reg', type=str, help='the type of model used',
                        choices=['swin_transformer', 'phenotypic', 'multi_modal', 'multi_modal_semi', 'multi_modal_image_main'])
    parser.add_argument('--cosine_lr', default=True, type=bool, help='whether use cosine scheduler')
    parser.add_argument('--save-path', default='model_checkpoint.pth', type=str, help='path to save the model checkpoint')
    parser.add_argument('--log-path', default='training.log', type=str, help='path to save the training log')
    args = parser.parse_args()

    # 设置日志格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info('Hyperparameters:')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # 确定使用的设备
    device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

    # 输入图像的shape
    input_shape = (args.input_height, args.input_width)

    # 初始化模型和优化器
    student_model = SwinTransformerForRegression(pretrain_path='pre_train_pth/swin_tiny_patch4_window7_224.pth')
    student_model = student_model.to(device)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = nn.MSELoss()  # 确保在使用之前定义损失函数

    # 查找最新的检查点文件
    latest_checkpoint_path = find_latest_checkpoint(args.out_path)
    if latest_checkpoint_path:
        # 加载最新的模型和优化器状态
        start_epoch = load_checkpoint(student_model, optimizer, latest_checkpoint_path)
        logger.info(f'Loaded checkpoint from epoch {start_epoch}')
    else:
        start_epoch = 0  # 如果没有找到权重文件，从头开始训练

    # 划分训练集和验证集
    k = 10
    random.seed(444)
    train_val_case_names = []
    case_names_file_path = 'brats_class copy.txt'
    with open(case_names_file_path) as f:
        case_names = f.readlines()
        for case_name in case_names:
            case_name = case_name.strip()
            train_val_case_names.append(case_name)

    random.shuffle(train_val_case_names)
    test_num = math.ceil(len(train_val_case_names) // 10)
    test_case_names = train_val_case_names[:test_num]
    train_val_case_names = list(set(train_val_case_names) - set(test_case_names))

    val_num = math.ceil(len(train_val_case_names) // k)

    mse_scores = []
    train_losses = []
    val_losses = []
    r2_scores = []  # 用于记录每个 epoch 的 R² 分数
    mae_scores = []  # 用于记录每个 epoch 的 MAE

    for run_num in range(k):
        if run_num < k - 1:
            val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
        else:
            val_case_names = train_val_case_names[run_num * val_num:]
        train_case_names = list(set(train_val_case_names) - set(val_case_names))

        # 创建训练和验证数据集
        train_dataset = LungDatasetForRegression(
            root_dir=args.data_path, 
            input_shape=input_shape,
            case_names=train_case_names, 
            is_train=True
        )
        val_dataset = LungDatasetForRegression(
            root_dir=args.data_path, 
            input_shape=input_shape,
            case_names=val_case_names, 
            is_train=False
        )

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )

        # 学习率调整策略
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5) if args.cosine_lr else None

        best_val_loss = float('inf')  # 初始化最好的验证损失为无穷大

        writer = SummaryWriter(log_dir=args.out_path + '/tensorboard_logs') # 初始化 SummaryWriter

        for epoch in range(start_epoch, args.epochs):
            student_model.train()
            pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} - Training", leave=False)
            
            epoch_train_loss = 0
            for image, phenotypes, target in pbar_train:
                image, phenotypes, target = image.to(device), phenotypes.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = student_model(image, phenotypes)
                
                output = output.squeeze()
                target = target.squeeze()
                
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                pbar_train.set_postfix(loss=loss.item())

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)

            # 获取并记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epoch)
            logger.info(f'Epoch [{epoch+1}/{args.epochs}] - Current Learning Rate: {current_lr:.6f}')


            student_model.eval()
            total_val_loss = 0
            pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation", leave=False)
            all_targets = []
            all_predictions = []

            with torch.no_grad():
                for image, phenotypes, target in pbar_val:
                    image, phenotypes, target = image.to(device), phenotypes.to(device), target.to(device)
                    output = student_model(image, phenotypes)
                    
                    output = output.squeeze()
                    target = target.squeeze()
                    
                    val_loss = loss_function(output, target)
                    total_val_loss += val_loss.item()

                    # 收集所有的目标值和预测值
                    all_targets.append(target)
                    all_predictions.append(output)

                    pbar_val.set_postfix(val_loss=val_loss.item())

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

            # 计算 R² 分数和均方误差 (MAE)
            all_targets = torch.cat(all_targets)  # 合并所有目标值
            all_predictions = torch.cat(all_predictions)  # 合并所有预测值
            r2, mae = calculate_metrics(all_targets, all_predictions)
            r2_scores.append(r2)
            mae_scores.append(mae)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Metric/R2_Score', r2, epoch)
            writer.add_scalar('Metric/MAE', mae, epoch)

            # 记录日志
            logger.info(f'Epoch [{epoch+1}/{args.epochs}] - Training MSE: {avg_train_loss:.4f} - Validation MSE: {avg_val_loss:.4f}')
            logger.info(f'Epoch [{epoch+1}/{args.epochs}] - R² Score: {r2:.4f} - MAE: {mae:.4f}')

            # 更新学习率调度器
            if lr_scheduler:
                lr_scheduler.step()

            # 保存最新权重和当前epoch
            if (epoch + 1) % 50 == 0:
                latest_model_save_path = os.path.join(args.out_path, f'latest_model_checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(student_model, optimizer, epoch + 1, latest_model_save_path)
                logger.info(f'Latest model checkpoint saved at {latest_model_save_path}')

            # 保存最佳权重和当前epoch
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_save_path = os.path.join(args.out_path, 'best_model_checkpoint.pth')
                save_checkpoint(student_model, optimizer, epoch + 1, best_model_save_path)
                logger.info(f'Best model checkpoint saved at {best_model_save_path}')

        # 绘制损失图和指标图
        plot_loss(train_losses, val_losses, r2_scores, mae_scores, args.out_path, epoch + 1)
        logger.info(f'Loss plot saved at {os.path.join(args.out_path, "loss_plot_epoch_{epoch+1}.jpg")}')

        # 关闭 SummaryWriter
        writer.close()

if __name__ == '__main__':
    main()