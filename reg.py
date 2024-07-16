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
from utils.classification_data import LungDatasetForRegression  # 自定义的数据集类
from model.SwinTransformerForRegression import SwinTransformerForRegression  # 自定义的模型类
import matplotlib.pyplot as plt
import os


def main():

    parser = argparse.ArgumentParser(description='MultiViewBreastCancerDetection')

    parser.add_argument('--data-path', default='B1', type=str, help='data path')
    parser.add_argument('--out-path', default='result', help='directory to output the result')
    parser.add_argument('--txt-path', default='Output', help='directory to output the result')
    parser.add_argument('--gpu-id', default=0, type=int, help='visible gpu id(s)')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', default=300, type=int, help='number of total steps to run')
    parser.add_argument('--labeled_ratio', default=0.1, type=float, help='labeled data ratio')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=True, type=bool, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS', help='length of the consistency loss ramp-up')
    parser.add_argument('--warm_up_epochs', default=10, type=int, help='number of total steps to run')
    parser.add_argument('--batch-size', default=128, type=int, help='train batch_size')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--input-width', default=224, type=int, help='the width of input images')
    parser.add_argument('--input-height', default=224, type=int, help='the height of input images')
    parser.add_argument('--model_type', default='swin_transformer_reg', type=str, help='the type of model used',
                        choices=['swin_transformer', 'phenotypic', 'multi_modal', 'multi_modal_semi', 'multi_modal_image_main'])
    parser.add_argument('--cosine_lr', default=False, type=bool, help='whether use cosine scheduler')
    parser.add_argument('--save-path', default='model_checkpoint.pth', type=str, help='path to save the model checkpoint')
    parser.add_argument('--log-path', default='training.log', type=str, help='path to save the training log')

    # 获取命令行输入的一些参数
    args = parser.parse_args()

    # 设置日志格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_path),  # Save log to file
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger()

    # Log hyperparameters
    logger.info('Hyperparameters:')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # 确定使用的设备
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        device = torch.device('cpu')

    # 输入图像的shape
    input_shape = (args.input_height, args.input_width)

    if args.model_type == 'swin_transformer_reg':
        # 划分训练集和验证集
        k = 10
        random.seed(444)

        train_val_case_names = []
        case_names_file_path = os.path.join('brats_class copy.txt')
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

        for run_num in range(k):
            if run_num < k - 1:
                val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
            else:
                val_case_names = train_val_case_names[run_num * val_num:]
            train_case_names = list(set(train_val_case_names) - set(val_case_names))

            # 创建模型
            student_model = SwinTransformerForRegression(pretrain_path='pre_train_pth/swin_tiny_patch4_window7_224.pth')
            student_model = student_model.to(device)

            # 定义优化器
            optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # 定义loss函数
            loss_function = nn.MSELoss()

            # 创建数据集
            train_dataset = LungDatasetForRegression(
                root_dir=args.data_path, 
                input_shape=(args.input_height, args.input_width),  # 提供正确的 input_shape
                case_names=train_case_names, 
                is_train=True
            )
            val_dataset = LungDatasetForRegression(
                root_dir=args.data_path, 
                input_shape=(args.input_height, args.input_width),  # 提供正确的 input_shape
                case_names=val_case_names, 
                is_train=False
            )

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

            # 学习率调整策略
            if args.cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = None

            train_losses = []
            val_losses = []

            for epoch in range(args.epochs):
                student_model.train()
                
                # Initialize tqdm for training progress
                pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} - Training", leave=False)
                
                epoch_train_loss = 0
                for image, phenotypes, target in pbar_train:
                    image, phenotypes, target = image.to(device), phenotypes.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = student_model(image, phenotypes)  # Pass both image and phenotypes to the model
                    
                    # Ensure output and target have the same shape
                    output = output.squeeze()  # Remove extra dimension if necessary
                    target = target.squeeze()  # Ensure target is the correct shape
                    
                    # 计算损失
                    loss = loss_function(output, target)
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    pbar_train.set_postfix(loss=loss.item())

                # Average training loss for the epoch
                avg_train_loss = epoch_train_loss / len(train_dataloader)
                train_losses.append(avg_train_loss)

                # Validation
                student_model.eval()
                total_val_loss = 0
                pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation", leave=False)
                with torch.no_grad():
                    for image, phenotypes, target in pbar_val:
                        image, phenotypes, target = image.to(device), phenotypes.to(device), target.to(device)
                        output = student_model(image, phenotypes)  # Pass both image and phenotypes to the model
                        
                        # Ensure output and target have the same shape
                        output = output.squeeze()  # Remove extra dimension if necessary
                        target = target.squeeze()  # Ensure target is the correct shape
                        
                        val_loss = loss_function(output, target)
                        total_val_loss += val_loss.item()

                        pbar_val.set_postfix(val_loss=val_loss.item())

                avg_val_loss = total_val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)

                logger.info(f'Epoch [{epoch+1}/{args.epochs}] - Training MSE: {avg_train_loss:.4f} - Validation MSE: {avg_val_loss:.4f}')

                if lr_scheduler:
                    lr_scheduler.step()

            # Save model checkpoint after each run
            model_save_path = os.path.join(args.out_path, f'model_checkpoint_run_{run_num}.pth')
            torch.save(student_model.state_dict(), model_save_path)
            logger.info(f'Model checkpoint saved at {model_save_path}')

            # Optionally, you can calculate the average MSE for the current run
            avg_mse = np.mean(val_losses)
            mse_std = np.std(val_losses)
            mse_scores.append(avg_mse)
            logger.info(f'Run {run_num + 1} - Average Validation MSE: {avg_mse:.4f}, Std: {mse_std:.4f}')

        # After all runs, plot the MSE scores
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.out_path, 'mse_plot.jpg'))  # Save the plot as a .jpg file
        logger.info(f'MSE plot saved at {os.path.join(args.out_path, "mse_plot.jpg")}')

if __name__ == '__main__':
    main()