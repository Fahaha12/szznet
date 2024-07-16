import argparse
import logging
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model.lung_classification_model import ModelSwinTransformer, OnlyPhenotypicModel, MultiModalModelSwinTBackbone
from model.SwinTransformer import SwinTransformerForRegression
from utils import ramps
from utils.classification_data import LungDatasetForClassificationWithPhenotypic, \
    lung_classification_with_phenotypic_collate, LungDatasetForSemiSupervisedClassificationWithPhenotypic, LungDatasetForRegression
from utils.meter import AverageMeter

logger = logging.getLogger(__name__)
global_step = 0

def main():
    parser = argparse.ArgumentParser(description='MultiViewBreastCancerDetection')

    parser.add_argument('--data-path', default=r'B1', type=str, help='data path')
    parser.add_argument('--out-path', default=r'result', help='directory to output the result')
    parser.add_argument('--txt-path', default=r'Output',
                        help='directory to output the result')
    parser.add_argument('--gpu-id', default=0, type=int, help='visible gpu id(s)')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', default=300, type=int, help='number of total steps to run')
    parser.add_argument('--labeled_ratio', default=0.1, type=int, help='labeled data ratio')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=True, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--warm_up_epochs', default=10, type=int, help='number of total steps to run')
    parser.add_argument('--batch-size', default=128, type=int, help='train batch_size')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--input-width', default=224, type=int, help='the width of input images')
    parser.add_argument('--input-height', default=224, type=int, help='the height of input images')
    parser.add_argument('--model_type', default='swin_transformer_reg', type=str,
                        help='the type of model used',
                        choices=['swin_transformer',
                                 'phenotypic',
                                 'multi_modal',
                                 'multi_modal_semi',
                                 'multi_modal_image_main'])
    parser.add_argument('--cosine_lr', default=False, type=bool, help='whether use cosine scheduler')

    # 设置日志格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # 获取命令行输入的一些参数
    args = parser.parse_args()

    # 确定使用的设备
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        device = torch.device('cpu')

    # device = torch.device('cpu')

    # 输入图像的shape
    input_shape = (args.input_height, args.input_width)

    dataset_name = args.data_path.split('/')[-1]
    if dataset_name == 'B1':
        if args.model_type == 'swin_transformer':
            # 划分训练集和验证集
            k = 10
            random.seed(444)

            train_val_case_names = []
            case_names_file_path = 'brats_class.txt'
            with open(case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    train_val_case_names.append(case_name)

            random.shuffle(train_val_case_names)
            test_num = math.ceil(len(train_val_case_names) // 10)
            test_case_names = train_val_case_names[0: test_num]
            train_val_case_names = list(set(train_val_case_names) - set(test_case_names))

            val_num = math.ceil(len(train_val_case_names) // k)

            student_val_acc = []
            student_test_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ModelSwinTransformer(img_size=img_size, patch_size=patch_size,
                                                     num_classes=5, window_size=window_size, pretrain=True,
                                                     pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                class_loss_function = nn.NLLLoss()

                # 创建数据集和数据加载器
                train_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                           input_shape=input_shape,
                                                                           case_names=train_case_names,
                                                                           is_train=True, only_phenotypic=True)
                val_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                         input_shape=input_shape,
                                                                         case_names=val_case_names,
                                                                         is_train=False, only_phenotypic=True)
                test_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                          input_shape=input_shape,
                                                                          case_names=test_case_names,
                                                                          is_train=False, only_phenotypic=True)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=lung_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=lung_classification_with_phenotypic_collate)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True, pin_memory=True, drop_last=False,
                                             collate_fn=lung_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 1 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                file_path = os.path.join(args.txt_path, args.model_type + f'_({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, student_best_acc, student_best_class_acc = train_sv_sm_model(args, student_model,
                                                                                            train_dataloader,
                                                                                            val_dataloader,
                                                                                            class_loss_function,
                                                                                            optimizer, lr_scheduler,
                                                                                            device,
                                                                                            class_names, train_dataset,
                                                                                            run_num, k)


                student_val_acc.append(student_best_acc)

                test_loss, test_acc, test_class_acc, test_class_all_num = evaluate_sv_sm_model(args, student_model,
                                                                                           test_dataloader,
                                                                                           class_loss_function,
                                                                                           device, class_names,
                                                                                           run_num, k)
                logger.info('******第' + str(run_num) + '次训练结果******')
                logger.info('Best Val Acc: {:.4f}'.format(student_best_acc))
                logger.info(student_best_class_acc)
                logger.info('Test Acc: {:.4f}'.format(test_acc))
                logger.info(test_class_acc)
                student_test_acc.append(student_test_acc)

                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_val_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_best_acc,
                        acc_1=student_best_class_acc['level_1'],
                        acc_2=student_best_class_acc['level_2'],
                        acc_3=student_best_class_acc['level_3'],
                        acc_4=student_best_class_acc['level_4'],
                        acc_5=student_best_class_acc['level_5'])
                    f.writelines(lines)
                    lines = 'test_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_test_acc,
                        acc_1=test_class_acc['level_1'],
                        acc_2=test_class_acc['level_2'],
                        acc_3=test_class_acc['level_3'],
                        acc_4=test_class_acc['level_4'],
                        acc_5=test_class_acc['level_5'])
                    f.writelines(lines)
                    if run_num == k - 1:
                        avg_val_acc = np.mean(student_val_acc)
                        acc_val_std = np.std(student_val_acc)
                        lines = '{k}次训练的平均值：\navg_val_acc: {avg_val_acc}    acc_val_std: {acc_val_std}\n'.format(
                            k=k,
                            avg_val_acc=avg_val_acc,
                            acc_val_std=acc_val_std)
                        f.writelines(lines)
                        avg_test_acc = np.mean(student_test_acc)
                        acc_test_std = np.std(student_test_acc)
                        lines = '{k}次训练的平均值：\navg_test_acc: {avg_test_acc}    acc_test_std: {acc_test_std}\n\n\n'.format(
                            k=k,
                            avg_test_acc=avg_test_acc,
                            acc_test_std=acc_test_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Val Acc: {:.4f}'.format(avg_val_acc))
                        logger.info('Avg Test Acc: {:.4f}'.format(avg_test_acc))
                # break

        elif args.model_type == 'swin_transformer_reg':    
                # 划分训练集和验证集
            k = 10
            random.seed(444)

            train_val_case_names = []
            case_names_file_path = 'brats_class.txt'
            with open(case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    train_val_case_names.append(case_name)

            random.shuffle(train_val_case_names)
            test_num = math.ceil(len(train_val_case_names) // 10)
            test_case_names = train_val_case_names[0: test_num]
            train_val_case_names = list(set(train_val_case_names) - set(test_case_names))

            val_num = math.ceil(len(train_val_case_names) // k)

            mse_scores = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                student_model = SwinTransformerForRegression(pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                # 定义loss函数
                loss_function = nn.MSELoss()

                # 创建数据集和数据加载器
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
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

                for epoch in range(args.num_epochs):
                    student_model.train()
                    for data, target in train_dataloader:
                        optimizer.zero_grad()
                        output = student_model(data.to(device))
                        loss = loss_function(output, target.to(device))
                        loss.backward()
                        optimizer.step()

                    # Validation
                    student_model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for data, target in val_dataloader:
                            output = student_model(data.to(device))
                            val_loss = loss_function(output, target.to(device))
                            total_val_loss += val_loss.item()

                    avg_val_loss = total_val_loss / len(val_dataloader)
                    logger.info(f'Validation MSE: {avg_val_loss:.4f}')
                    mse_scores.append(avg_val_loss)

            avg_mse = np.mean(mse_scores)
            mse_std = np.std(mse_scores)
            logger.info(f'Average MSE: {avg_mse:.4f}, Std: {mse_std:.4f}')

        elif args.model_type == 'phenotypic':
            k = 10
            random.seed(444)

            train_val_case_names = []
            case_names_file_path = os.path.join('data', 'lung_classification_case_names.txt')
            with open(case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    train_val_case_names.append(case_name)

            random.shuffle(train_val_case_names)
            test_num = math.ceil(len(train_val_case_names) // 10)
            test_case_names = train_val_case_names[0: test_num]
            train_val_case_names = list(set(train_val_case_names) - set(test_case_names))

            val_num = math.ceil(len(train_val_case_names) // k)

            student_val_acc = []
            student_test_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                student_model = OnlyPhenotypicModel(phenotypic_dim=8, num_classes=5)
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                class_loss_function = nn.NLLLoss()

                # 创建数据集和数据加载器
                train_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                           input_shape=input_shape,
                                                                           case_names=train_case_names,
                                                                           is_train=True, only_phenotypic=True)
                val_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                         input_shape=input_shape,
                                                                         case_names=val_case_names,
                                                                         is_train=False, only_phenotypic=True)
                test_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                          input_shape=input_shape,
                                                                          case_names=test_case_names,
                                                                          is_train=False, only_phenotypic=True)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=lung_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=lung_classification_with_phenotypic_collate)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True, pin_memory=True, drop_last=False,
                                             collate_fn=lung_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 1 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                logger.info(f"  ImageSize = {img_size}")

                file_path = os.path.join(args.txt_path, args.model_type + f'_({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, student_best_acc, student_best_class_acc = train_sv_sm_model(args, student_model,
                                                                                            train_dataloader,
                                                                                            val_dataloader,
                                                                                            class_loss_function,
                                                                                            optimizer, lr_scheduler,
                                                                                            device,
                                                                                            class_names, train_dataset,
                                                                                            run_num, k)


                student_val_acc.append(student_best_acc)

                test_loss, test_acc, test_class_acc, test_class_all_num = evaluate_sv_sm_model(args, student_model,
                                                                                           test_dataloader,
                                                                                           class_loss_function,
                                                                                           device, class_names,
                                                                                           run_num, k)
                logger.info('******第' + str(run_num) + '次训练结果******')
                logger.info('Best Val Acc: {:.4f}'.format(student_best_acc))
                logger.info(student_best_class_acc)
                logger.info('Test Acc: {:.4f}'.format(test_acc))
                logger.info(test_class_acc)
                student_test_acc.append(test_acc)

                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_val_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_best_acc,
                        acc_1=student_best_class_acc['level_1'],
                        acc_2=student_best_class_acc['level_2'],
                        acc_3=student_best_class_acc['level_3'],
                        acc_4=student_best_class_acc['level_4'],
                        acc_5=student_best_class_acc['level_5'])
                    f.writelines(lines)
                    lines = 'test_acc: {test_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        test_acc=test_acc,
                        acc_1=test_class_acc['level_1'],
                        acc_2=test_class_acc['level_2'],
                        acc_3=test_class_acc['level_3'],
                        acc_4=test_class_acc['level_4'],
                        acc_5=test_class_acc['level_5'])
                    f.writelines(lines)
                    if run_num == k - 1:
                        avg_val_acc = np.mean(student_val_acc)
                        acc_val_std = np.std(student_val_acc)
                        lines = '{k}次训练的平均值：\navg_val_acc: {avg_val_acc}    acc_val_std: {acc_val_std}\n'.format(
                            k=k,
                            avg_val_acc=avg_val_acc,
                            acc_val_std=acc_val_std)
                        f.writelines(lines)
                        avg_test_acc = np.mean(student_test_acc)
                        acc_test_std = np.std(student_test_acc)
                        lines = '{k}次训练的平均值：\navg_test_acc: {avg_test_acc}    acc_test_std: {acc_test_std}\n\n\n'.format(
                            k=k,
                            avg_test_acc=avg_test_acc,
                            acc_test_std=acc_test_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Val Acc: {:.4f}'.format(avg_val_acc))
                        logger.info('Avg Test Acc: {:.4f}'.format(avg_test_acc))
                # break
        elif args.model_type == 'multi_modal':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            train_val_case_names = []
            case_names_file_path = 'brats_class.txt'
            with open(case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    train_val_case_names.append(case_name)

            random.shuffle(train_val_case_names)
            test_num = math.ceil(len(train_val_case_names) // 10)
            test_case_names = train_val_case_names[0: test_num]
            train_val_case_names = list(set(train_val_case_names) - set(test_case_names))

            val_num = math.ceil(len(train_val_case_names) // k)

            student_val_acc = []
            student_test_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))
                # print(len(train_case_names))
                # print(len(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = MultiModalModelSwinTBackbone(phenotypic_dim=8, img_size=img_size, patch_size=patch_size,
                                                             class_num=5, window_size=window_size, pretrain=True,
                                                             pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                class_loss_function = nn.NLLLoss()

                # 创建数据集和数据加载器
                train_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                           input_shape=input_shape,
                                                                           case_names=train_case_names,
                                                                           is_train=True, only_phenotypic=True)
                val_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                         input_shape=input_shape,
                                                                         case_names=val_case_names,
                                                                         is_train=False, only_phenotypic=True)
                test_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                          input_shape=input_shape,
                                                                          case_names=test_case_names,
                                                                          is_train=False, only_phenotypic=True)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=lung_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=lung_classification_with_phenotypic_collate)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True, pin_memory=True, drop_last=False,
                                             collate_fn=lung_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 1 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                logger.info(f"  ImageSize = {img_size}")

                file_path = os.path.join(args.txt_path, args.model_type + f'_({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, student_best_acc, student_best_class_acc = train_sv_sm_model(args, student_model,
                                                                                            train_dataloader,
                                                                                            val_dataloader,
                                                                                            class_loss_function,
                                                                                            optimizer, lr_scheduler,
                                                                                            device,
                                                                                            class_names, train_dataset,
                                                                                            run_num, k)


                student_val_acc.append(student_best_acc)

                test_loss, test_acc, test_class_acc, test_class_all_num = evaluate_sv_sm_model(args, student_model,
                                                                                           test_dataloader,
                                                                                           class_loss_function,
                                                                                           device, class_names,
                                                                                           run_num, k)
                logger.info('******第' + str(run_num) + '次训练结果******')
                logger.info('Best Val Acc: {:.4f}'.format(student_best_acc))
                logger.info(student_best_class_acc)
                logger.info('Test Acc: {:.4f}'.format(test_acc))
                logger.info(test_class_acc)
                student_test_acc.append(student_test_acc)

                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_val_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_best_acc,
                        acc_1=student_best_class_acc['level_1'],
                        acc_2=student_best_class_acc['level_2'],
                        acc_3=student_best_class_acc['level_3'],
                        acc_4=student_best_class_acc['level_4'],
                        acc_5=student_best_class_acc['level_5'])
                    f.writelines(lines)
                    lines = 'test_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_test_acc,
                        acc_1=test_class_acc['level_1'],
                        acc_2=test_class_acc['level_2'],
                        acc_3=test_class_acc['level_3'],
                        acc_4=test_class_acc['level_4'],
                        acc_5=test_class_acc['level_5'])
                    f.writelines(lines)
                    if run_num == k - 1:
                        avg_val_acc = np.mean(student_val_acc)
                        acc_val_std = np.std(student_val_acc)
                        lines = '{k}次训练的平均值：\navg_val_acc: {avg_val_acc}    acc_val_std: {acc_val_std}\n'.format(
                            k=k,
                            avg_val_acc=avg_val_acc,
                            acc_val_std=acc_val_std)
                        f.writelines(lines)
                        avg_test_acc = np.mean(student_test_acc)
                        acc_test_std = np.std(student_test_acc)
                        lines = '{k}次训练的平均值：\navg_test_acc: {avg_test_acc}    acc_test_std: {acc_test_std}\n\n\n'.format(
                            k=k,
                            avg_test_acc=avg_test_acc,
                            acc_test_std=acc_test_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Val Acc: {:.4f}'.format(avg_val_acc))
                        logger.info('Avg Test Acc: {:.4f}'.format(avg_test_acc))
                # break
        elif args.model_type == 'multi_modal_only_labeled':
            # 划分训练集和验证集
            k = 10
            random.seed(444)

            train_val_case_names = []
            case_names_file_path = os.path.join('data', 'lung_classification_case_names.txt')
            with open(case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    train_val_case_names.append(case_name)

            random.shuffle(train_val_case_names)
            test_num = math.ceil(len(train_val_case_names) // 10)
            test_case_names = train_val_case_names[0: test_num]
            train_val_case_names = list(set(train_val_case_names) - set(test_case_names))

            val_num = math.ceil(len(train_val_case_names) // k)

            student_val_acc = []
            student_test_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))
                labeled_case_names = train_case_names[0: math.ceil(len(train_case_names) * args.labeled_ratio)]
                # print(len(train_case_names))
                # print(len(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = MultiModalModelSwinTBackbone(phenotypic_dim=8, img_size=img_size, patch_size=patch_size,
                                                             class_num=5, window_size=window_size, pretrain=True,
                                                             pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                class_loss_function = nn.NLLLoss(ignore_index=-1)

                # 创建数据集和数据加载器
                train_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                           input_shape=input_shape,
                                                                           case_names=train_case_names,
                                                                           is_train=True, only_phenotypic=True)
                val_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                         input_shape=input_shape,
                                                                         case_names=val_case_names,
                                                                         is_train=False, only_phenotypic=True)
                test_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                          input_shape=input_shape,
                                                                          case_names=test_case_names,
                                                                          is_train=False, only_phenotypic=True)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=lung_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=lung_classification_with_phenotypic_collate)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True, pin_memory=True, drop_last=False,
                                             collate_fn=lung_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 1 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                logger.info(f"  ImageSize = {img_size}")

                file_path = os.path.join(args.txt_path, args.model_type + f'_({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}_'
                                                                          f'labeled_ratio={args.labeled_ratio}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, student_best_acc, student_best_class_acc = train_sv_sm_model(args, student_model,
                                                                                            train_dataloader,
                                                                                            val_dataloader,
                                                                                            class_loss_function,
                                                                                            optimizer, lr_scheduler,
                                                                                            device,
                                                                                            class_names, train_dataset,
                                                                                            run_num, k)


                student_val_acc.append(student_best_acc)

                test_loss, test_acc, test_class_acc, test_class_all_num = evaluate_sv_sm_model(args, student_model,
                                                                                           test_dataloader,
                                                                                           class_loss_function,
                                                                                           device, class_names,
                                                                                           run_num, k)
                logger.info('******第' + str(run_num) + '次训练结果******')
                logger.info('Best Val Acc: {:.4f}'.format(student_best_acc))
                logger.info(student_best_class_acc)
                logger.info('Test Acc: {:.4f}'.format(test_acc))
                logger.info(test_class_acc)
                student_test_acc.append(student_test_acc)

                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_val_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_best_acc,
                        acc_1=student_best_class_acc['level_1'],
                        acc_2=student_best_class_acc['level_2'],
                        acc_3=student_best_class_acc['level_3'],
                        acc_4=student_best_class_acc['level_4'],
                        acc_5=student_best_class_acc['level_5'])
                    f.writelines(lines)
                    lines = 'test_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_test_acc,
                        acc_1=test_class_acc['level_1'],
                        acc_2=test_class_acc['level_2'],
                        acc_3=test_class_acc['level_3'],
                        acc_4=test_class_acc['level_4'],
                        acc_5=test_class_acc['level_5'])
                    f.writelines(lines)
                    if run_num == k - 1:
                        avg_val_acc = np.mean(student_val_acc)
                        acc_val_std = np.std(student_val_acc)
                        lines = '{k}次训练的平均值：\navg_val_acc: {avg_val_acc}    acc_val_std: {acc_val_std}\n'.format(
                            k=k,
                            avg_val_acc=avg_val_acc,
                            acc_val_std=acc_val_std)
                        f.writelines(lines)
                        avg_test_acc = np.mean(student_test_acc)
                        acc_test_std = np.std(student_test_acc)
                        lines = '{k}次训练的平均值：\navg_test_acc: {avg_test_acc}    acc_test_std: {acc_test_std}\n\n\n'.format(
                            k=k,
                            avg_test_acc=avg_test_acc,
                            acc_test_std=acc_test_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Val Acc: {:.4f}'.format(avg_val_acc))
                        logger.info('Avg Test Acc: {:.4f}'.format(avg_test_acc))
                # break
        elif args.model_type == 'multi_modal_semi':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            train_val_case_names = []
            case_names_file_path = os.path.join('data', 'lung_classification_case_names.txt')
            with open(case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    train_val_case_names.append(case_name)

            random.shuffle(train_val_case_names)

            val_num = math.ceil(len(train_val_case_names) // k)

            student_val_acc = []
            teacher_val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))
                labeled_case_names = train_case_names[0: math.ceil(len(train_case_names) * args.labeled_ratio)]
                # print(len(train_case_names))
                # print(len(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = MultiModalModelSwinTBackbone(phenotypic_dim=8, img_size=img_size, patch_size=patch_size,
                                                             class_num=5, window_size=window_size, pretrain=True,
                                                             pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)
                teacher_model = MultiModalModelSwinTBackbone(phenotypic_dim=8, img_size=img_size, patch_size=patch_size,
                                                             class_num=5, window_size=window_size, pretrain=True,
                                                             pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                for param in teacher_model.parameters():
                    param.detach_()
                teacher_model = teacher_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                class_loss_function = nn.NLLLoss(ignore_index=-1)
                consistency_loss_function = softmax_mse_loss

                # 创建数据集和数据加载器
                train_dataset = LungDatasetForSemiSupervisedClassificationWithPhenotypic(args.data_path,
                                                                                         input_shape=input_shape,
                                                                                         case_names=train_case_names,
                                                                                         labeled_case_names=labeled_case_names,
                                                                                         is_train=True,
                                                                                         only_phenotypic=True)
                val_dataset = LungDatasetForClassificationWithPhenotypic(args.data_path,
                                                                         input_shape=input_shape,
                                                                         case_names=val_case_names,
                                                                         is_train=False,
                                                                         only_phenotypic=True)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=lung_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=lung_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 1 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                logger.info(f"  ImageSize = {img_size}")

                file_path = os.path.join(args.txt_path, args.model_type + f'_({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}_'
                                                                          f'labeled_ratio={args.labeled_ratio}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, student_best_acc, student_best_class_acc \
                    , teacher_best_acc, teacher_best_class_acc = train_sv_sm_semi_model(args, student_model,
                                                                                        teacher_model,
                                                                                        train_dataloader,
                                                                                        val_dataloader,
                                                                                        class_loss_function,
                                                                                        consistency_loss_function,
                                                                                        optimizer,
                                                                                        lr_scheduler, device,
                                                                                        class_names,
                                                                                        train_dataset, run_num, k)
                student_val_acc.append(student_best_acc)
                teacher_val_acc.append(teacher_best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nstudent_best_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=student_best_acc,
                        acc_1=student_best_class_acc['level_1'],
                        acc_2=student_best_class_acc['level_2'],
                        acc_3=student_best_class_acc['level_3'],
                        acc_4=student_best_class_acc['level_4'],
                        acc_5=student_best_class_acc['level_5'])
                    f.writelines(lines)
                    lines = 'teacher_best_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=teacher_best_acc,
                        acc_1=teacher_best_class_acc['level_1'],
                        acc_2=teacher_best_class_acc['level_2'],
                        acc_3=teacher_best_class_acc['level_3'],
                        acc_4=teacher_best_class_acc['level_4'],
                        acc_5=teacher_best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Student Acc: {:.4f}'.format(student_best_acc))
                    logger.info(student_best_class_acc)
                    logger.info('Best Teacher Acc: {:.4f}'.format(teacher_best_acc))
                    logger.info(teacher_best_class_acc)
                    if run_num == k - 1:
                        avg_val_acc = np.mean(student_val_acc)
                        acc_val_std = np.std(student_val_acc)
                        lines = '{k}次训练的平均值：\nstudent_avg_acc: {avg_acc}    student_acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_acc=avg_val_acc,
                            acc_std=acc_val_std)
                        f.writelines(lines)

                        logger.info('******总训练结果******')
                        logger.info('Avg Student Acc: {:.4f}'.format(avg_val_acc))

                        avg_val_acc = np.mean(teacher_val_acc)
                        acc_val_std = np.std(teacher_val_acc)
                        lines = 'teacher_avg_acc: {avg_acc}    teacher_acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_val_acc,
                            acc_std=acc_val_std)
                        f.writelines(lines)
                        logger.info('Avg Teacher Acc: {:.4f}'.format(avg_val_acc))
                # break


def train_sv_sm_model(args, model, train_dataloader, val_dataloader, loss_function, optimizer, lr_scheduler,
                      device, class_names, train_dataset, run_num=0, k=1):
    logger.info("***** training start*****")
    logger.info(f"  Model Type = {model.__class__.__name__}")
    logger.info(f"  GPU ID = {args.gpu_id}")
    logger.info(f"  Num Workers = {args.num_workers}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch Size = {args.batch_size}")
    logger.info(f"  Learning Rate = {args.lr}")
    logger.info(f"  Input Size = (w:{args.input_width},h:{args.input_height})")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    best_acc = 0
    best_val_class_acc = []

    unchanged_epoch = 0

    for epoch in range(args.epochs):
        losses.reset()
        model.train()

        with tqdm(train_dataloader, position=0) as p_bar:
            for batch_index, (images, phenotypes, labels) in enumerate(train_dataloader):
                data_time.update(time.time() - end)

                images = images.type(torch.FloatTensor).to(device)
                phenotypes = phenotypes.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                output = model(images, phenotypes)

                # print(output)
                # print(labels)
                loss = loss_function(output, labels)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                losses.update(loss.item())
                batch_time.update(time.time() - end)
                end = time.time()

                p_bar.set_description(
                    "Run Num: {run_num:2}/{k:2}. Train Epoch: {epoch:3}/{epochs:3}. Batch: {batch:3}/{iter:3}. "
                    "LR: {lr:.8f}. Avg Data: {data:.3f}s. Avg Batch: {bt:.3f}s. Loss: {loss:.4f}".format(
                        run_num=run_num + 1,
                        k=k,
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_index + 1,
                        iter=len(train_dataloader),
                        lr=lr_scheduler.get_last_lr()[-1],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        # 每代训练结束后评估模型，并获得一些评价指标，如果当前模型的指标高于best，则保存模型
        val_loss, val_acc, val_class_acc, val_class_all_num = evaluate_sv_sm_model(args, model,
                                                                                   val_dataloader,
                                                                                   loss_function,
                                                                                   device, class_names,
                                                                                   run_num, k)

        if val_acc > best_acc:
            unchanged_epoch = 0
            best_acc = val_acc
            best_val_class_acc = val_class_acc
            torch.save(model.state_dict(), os.path.join(args.out_path, 'brats' +'_best_acc_model.pt'))
        else:
            unchanged_epoch += 1

        # 打印日志，以便实时查看训练中的一些评价指标
        logger.info('Best Acc: {:.4f}'.format(best_acc))
        logger.info(best_val_class_acc)
        logger.info('Val Acc: {:.4f}'.format(val_acc))
        logger.info('Val Class Acc:')
        logger.info(val_class_acc)
        logger.info(val_class_all_num)

        # 如果30代内auc没有提升，则结束训练
        if unchanged_epoch >= 30:
            break

    return model, best_acc, best_val_class_acc


def evaluate_sv_sm_model(args, model, eval_dataloader, loss_function, device, class_names, run_num=0, k=1):
    losses = AverageMeter()
    acc = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    model.eval()

    with tqdm(eval_dataloader, position=0) as p_bar:

        class_correct_num = {cls_name: 0 for cls_name in class_names}
        class_all_num = {cls_name: 0 for cls_name in class_names}

        for batch_index, (images, phenotypes, labels) in enumerate(eval_dataloader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                images = images.type(torch.FloatTensor).to(device)
                phenotypes = phenotypes.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.LongTensor).to(device)

                output = model(images, phenotypes)

                loss = loss_function(output, labels)

                acc, class_correct_num_batch, class_all_num_batch = accuracy(output, labels, class_names)

                for cls_name in class_names:
                    class_correct_num[cls_name] += class_correct_num_batch[cls_name]
                    class_all_num[cls_name] += class_all_num_batch[cls_name]

                losses.update(loss.item())
                batch_time.update(time.time() - end)
                end = time.time()

                p_bar.set_description(
                    "Run Num: {run_num:2}/{k:2}. Evaluate Batch: {batch:3}/{iter:3}. Avg Data: {data:.3f}s. "
                    "Avg Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                        run_num=run_num + 1,
                        k=k,
                        batch=batch_index + 1,
                        iter=len(eval_dataloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()
        class_acc = {}
        total_correct_num = 0
        total_all_num = 0
        for cls_name in class_names:
            if class_all_num[cls_name] == 0:
                class_acc[cls_name] = -1
            else:
                class_acc[cls_name] = round(class_correct_num[cls_name] * 100.0 / class_all_num[cls_name], 2)
            total_correct_num += class_correct_num[cls_name]
            total_all_num += class_all_num[cls_name]

        total_acc = total_correct_num * 100.0 / total_all_num

    return losses.avg, total_acc, class_acc, class_all_num


def train_sv_sm_semi_model(args, student_model, teacher_model, train_dataloader, val_dataloader, class_loss_function,
                           consistency_loss_function, optimizer, lr_scheduler, device, class_names, train_dataset,
                           run_num=0, k=1):
    logger.info("***** training start*****")
    logger.info(f"  Model Type = {student_model.__class__.__name__}")
    logger.info(f"  GPU ID = {args.gpu_id}")
    logger.info(f"  Num Workers = {args.num_workers}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch Size = {args.batch_size}")
    logger.info(f"  Learning Rate = {args.lr}")
    logger.info(f"  Input Size = (w:{args.input_width},h:{args.input_height})")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    student_best_acc = 0
    student_best_val_class_acc = []
    teacher_best_acc = 0
    teacher_best_val_class_acc = []

    unchanged_epoch = 0

    global global_step

    for epoch in range(args.epochs):
        losses.reset()
        student_model.train()

        with tqdm(train_dataloader, position=0) as p_bar:
            for batch_index, (images, phenotypes, labels) in enumerate(train_dataloader):
                data_time.update(time.time() - end)

                images = images.type(torch.FloatTensor).to(device)
                phenotypes = phenotypes.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                student_output = student_model(images, phenotypes)
                teacher_output = teacher_model(images, phenotypes)
                teacher_output = Variable(teacher_output.detach().data, requires_grad=False)

                student_class_loss = class_loss_function(student_output, labels)
                # teacher_class_loss = class_loss_function(teacher_output, labels)
                consistency_loss = consistency_loss_function(student_output, teacher_output)
                loss = student_class_loss + get_current_consistency_weight(epoch, args) * consistency_loss

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                global_step += 1
                update_ema_variables(student_model, teacher_model, args.ema_decay, global_step)

                losses.update(loss.item())
                batch_time.update(time.time() - end)
                end = time.time()

                p_bar.set_description(
                    "Run Num: {run_num:2}/{k:2}. Train Epoch: {epoch:3}/{epochs:3}. Batch: {batch:3}/{iter:3}. "
                    "LR: {lr:.8f}. Avg Data: {data:.3f}s. Avg Batch: {bt:.3f}s. Loss: {loss:.4f}".format(
                        run_num=run_num + 1,
                        k=k,
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_index + 1,
                        iter=len(train_dataloader),
                        lr=lr_scheduler.get_last_lr()[-1],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        # 每代训练结束后评估模型，并获得一些评价指标，如果当前模型的指标高于best，则保存模型
        student_val_loss, student_val_acc, student_val_class_acc, \
        teacher_val_loss, teacher_val_acc, teacher_val_class_acc, \
        val_class_all_num = evaluate_sv_sm_semi_model(args, student_model,
                                                      teacher_model,
                                                      val_dataloader,
                                                      class_loss_function,
                                                      device, class_names,
                                                      run_num, k)

        # if student_val_acc > student_best_acc:
        #     unchanged_epoch = 0
        #     student_best_acc = student_val_acc
        #     student_best_val_class_acc = student_val_class_acc
        #     # torch.save(model.state_dict(), os.path.join(args.out_path, comment + '_best_acc_model.pt'))
        # else:
        #     unchanged_epoch += 1

        if student_val_acc <= student_best_acc and teacher_val_acc <= teacher_best_acc:
            unchanged_epoch += 1
        else:
            if student_val_acc > student_best_acc:
                unchanged_epoch = 0
                student_best_acc = student_val_acc
                student_best_val_class_acc = student_val_class_acc
            if teacher_val_acc > teacher_best_acc:
                unchanged_epoch = 0
                teacher_best_acc = teacher_val_acc
                teacher_best_val_class_acc = teacher_val_class_acc

        # 打印日志，以便实时查看训练中的一些评价指标
        logger.info('Best Student Acc: {:.4f}'.format(student_best_acc))
        logger.info(student_best_val_class_acc)
        logger.info('Student Val Acc: {:.4f}'.format(student_val_acc))
        logger.info('Student Val Class Acc:')
        logger.info(student_val_class_acc)
        logger.info('\n')
        logger.info('Best Teacher Acc: {:.4f}'.format(teacher_best_acc))
        logger.info(teacher_best_val_class_acc)
        logger.info('Teacher Val Acc: {:.4f}'.format(teacher_val_acc))
        logger.info('Teacher Val Class Acc:')
        logger.info(teacher_val_class_acc)
        logger.info('\n')
        logger.info(val_class_all_num)

        # 如果30代内auc没有提升，则结束训练
        if unchanged_epoch >= 30:
            break

    return student_model, student_best_acc, student_best_val_class_acc, teacher_best_acc, teacher_best_val_class_acc


def evaluate_sv_sm_semi_model(args, student_model, teacher_model, eval_dataloader, class_loss_function, device,
                              class_names,
                              run_num=0, k=1):
    student_losses = AverageMeter()
    teacher_losses = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    student_model.eval()

    with tqdm(eval_dataloader, position=0) as p_bar:

        student_class_correct_num = {cls_name: 0 for cls_name in class_names}
        teacher_class_correct_num = {cls_name: 0 for cls_name in class_names}
        class_all_num = {cls_name: 0 for cls_name in class_names}

        for batch_index, (images, phenotypes, labels) in enumerate(eval_dataloader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                images = images.type(torch.FloatTensor).to(device)
                phenotypes = phenotypes.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.LongTensor).to(device)

                student_output = student_model(images, phenotypes)
                teacher_output = teacher_model(images, phenotypes)

                student_loss = class_loss_function(student_output, labels)
                teacher_loss = class_loss_function(teacher_output, labels)

                student_acc, student_class_correct_num_batch, class_all_num_batch = accuracy(student_output, labels,
                                                                                             class_names)
                teacher_acc, teacher_class_correct_num_batch, _ = accuracy(teacher_output, labels, class_names)

                for cls_name in class_names:
                    student_class_correct_num[cls_name] += student_class_correct_num_batch[cls_name]
                    teacher_class_correct_num[cls_name] += teacher_class_correct_num_batch[cls_name]
                    class_all_num[cls_name] += class_all_num_batch[cls_name]

                student_losses.update(student_loss.item())
                teacher_losses.update(teacher_loss.item())
                batch_time.update(time.time() - end)
                end = time.time()

                p_bar.set_description(
                    "Run Num: {run_num:2}/{k:2}. Evaluate Batch: {batch:3}/{iter:3}. Avg Data: {data:.3f}s. "
                    "Avg Batch: {bt:.3f}s. Student Loss: {student_loss:.4f}. Teacher Loss: {teacher_loss:.4f}".format(
                        run_num=run_num + 1,
                        k=k,
                        batch=batch_index + 1,
                        iter=len(eval_dataloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        student_loss=student_losses.avg,
                        teacher_loss=teacher_losses.avg))
                p_bar.update()
        student_class_acc = {}
        teacher_class_acc = {}
        student_total_correct_num = 0
        teacher_total_correct_num = 0
        total_all_num = 0
        for cls_name in class_names:
            if class_all_num[cls_name] == 0:
                student_class_acc[cls_name] = -1
                teacher_class_acc[cls_name] = -1
            else:
                student_class_acc[cls_name] = round(
                    student_class_correct_num[cls_name] * 100.0 / class_all_num[cls_name], 2)
                teacher_class_acc[cls_name] = round(
                    teacher_class_correct_num[cls_name] * 100.0 / class_all_num[cls_name], 2)
            student_total_correct_num += student_class_correct_num[cls_name]
            teacher_total_correct_num += teacher_class_correct_num[cls_name]
            total_all_num += class_all_num[cls_name]

        student_total_acc = student_total_correct_num * 100.0 / total_all_num
        teacher_total_acc = teacher_total_correct_num * 100.0 / total_all_num

    return student_losses.avg, student_total_acc, student_class_acc, \
           teacher_losses.avg, teacher_total_acc, teacher_class_acc, class_all_num


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax)


def accuracy(output, target, class_names, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    class_correct_num = {cls_name: 0 for cls_name in class_names}
    class_all_num = {cls_name: 0 for cls_name in class_names}

    for i in range(target.size(0)):
        target_class_name = class_names[target[i].item()]
        correct_num = correct[0, i].reshape(-1).sum(0)
        class_correct_num[target_class_name] = class_correct_num[target_class_name] + correct_num.item()
        class_all_num[target_class_name] = class_all_num[target_class_name] + 1

    return res[0], class_correct_num, class_all_num


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == '__main__':
    main()
