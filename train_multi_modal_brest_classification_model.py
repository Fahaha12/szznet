import argparse
import logging
import math
import os
import random
import time
import torch

import numpy as np
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torchvision
from sklearn.metrics import confusion_matrix

from model.birads_classification_models import *
from utils import ramps
from utils.classification_data import DDSMMVDatasetForBIRADSClassification, ddsm_mv_dataset_for_birads_classification_collate, \
    DDSMMVDatasetForBIRADSClassificationWithPhenotypic, \
    ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate, \
    DDSMMVDatasetForSemiSupervisedBIRADSClassificationWithPhenotypic
from utils.constants import VIEWS
from utils.meter import AverageMeter

logger = logging.getLogger(__name__)
global_step = 0


def main():
    parser = argparse.ArgumentParser(description='MultiModalBreastCancerClassification')

    parser.add_argument('--data-path', default=r'data/DDSM', type=str, help='data path')
    parser.add_argument('--out-path', default=r'result', help='directory to output the result')
    parser.add_argument('--txt-path', default=r'birads_classification_result_224_224',
                        help='directory to output the result')

    parser.add_argument('--gpu-id', default=0, type=int, help='visible gpu id(s)')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', default=300, type=int, help='number of total steps to run')
    parser.add_argument('--warm_up_epochs', default=20, type=int, help='number of total steps to run')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--batch-size', default=16, type=int, help='train batch_size')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--input-width', default=224, type=int, help='the width of input images')
    parser.add_argument('--input-height', default=224, type=int, help='the height of input images')
    parser.add_argument('--model_type', default='single_swin_transformer', type=str,
                        help='the type of student_model used',

                        choices=['single_resnet50', 'single_swin_transformer',
                                 'phenotypic_0', 'phenotypic_1',
                                 'multi_modal_single_swin_0',

                                 'view_wise_0',
                                 'view_wise_swin_transformer_last_stage_cva',
                                 'view_wise_phenotypic_last_stage_cva',

                                 'view_wise_multi_modal_swin_transformer_last_stage_cva',
                                 'view_wise_multi_modal_only_image',
                                 'view_wise_multi_modal_multi_view_first',

                                 'view_wise_multi_modal_semi',
                                 'view_wise_multi_modal_only_image_semi'])
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

    # 输入图像的shape
    input_shape = (args.input_height, args.input_width)

    dataset_name = args.data_path.split('/')[-1]
    if dataset_name == 'DDSM':
        if args.model_type == 'single_resnet50':
            # 划分训练集和验证集
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)

            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                backbone = 'resnet50'
                student_model = SVModelResnet(pretrained=True, backbone=backbone, class_num=5)
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=args.weight_decay)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(
                                                                                       args.input_height,
                                                                                       args.input_width),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(
                                                                                     args.input_height,
                                                                                     args.input_width),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']

                logger.info(f"  Backbone = {backbone}")
                logger.info(f"  LW = {with_loss_weight}")
                student_model, best_acc, best_class_acc = train_sv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_{backbone}_'
                                                                          f'_lw={with_loss_weight}_'
                                                                          f'({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'single_swin_transformer':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)

            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = SVModelSwinTransformer(img_size=img_size, patch_size=patch_size,
                                                       num_classes=5, window_size=window_size, pretrain=True,
                                                       pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(img_size, img_size),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(img_size, img_size),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_sv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'phenotypic_0':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)

            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = OnlyPhenotypicModel0(phenotypic_dim=5, num_classes=5,
                                                     hidden_layer_dims=[96, 192, 384, 768, 1536])
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(img_size, img_size),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True, only_phenotypic=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(img_size, img_size),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False, only_phenotypic=True)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_sv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'phenotypic_1':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)

            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                student_model = OnlyPhenotypicModel1(phenotypic_dim=5, num_classes=5)
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(img_size, img_size),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True, only_phenotypic=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(img_size, img_size),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False, only_phenotypic=True)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                logger.info(f"  ImageSize = {img_size}")

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_sv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_0':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                backbone = 'resnet50'
                student_model = ViewWiseResNet50LastStageConcat(pretrained=True, backbone=backbone, class_num=5)
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(
                                                                                       args.input_height,
                                                                                       args.input_width),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(
                                                                                     args.input_height,
                                                                                     args.input_width),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
                    gamma = 0.95

                    def step_with_warm_up(step: int):
                        if step < warm_up_steps:
                            return (step + 1) / warm_up_steps
                        else:
                            return gamma ** ((step - warm_up_steps) // step_size)

                    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, step_with_warm_up)

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']

                student_model, best_acc, best_class_acc = train_mv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'_{backbone}_'
                                                                          f'({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'multi_modal_single_swin_0':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)

            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = MultiModalModelSwinTBackbone(phenotypic_dim=5, device=device, img_size=img_size,
                                                             patch_size=patch_size,
                                                             class_num=5, window_size=window_size, pretrain=True,
                                                             pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(img_size, img_size),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(img_size, img_size),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({args.input_width},{args.input_height})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_sv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ViewWiseSwinTransLastStagesCVA(img_size=img_size, patch_size=patch_size,
                                                               num_classes=5, window_size=window_size, pretrain=True,
                                                               pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(
                                                                                       args.input_height,
                                                                                       args.input_width),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(
                                                                                     args.input_height,
                                                                                     args.input_width),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_mv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_phenotypic_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ViewWiseOnlyPhenotypicModel1LastStagesCVA(phenotypic_dim=5, num_classes=5,
                                                                          window_size=window_size)
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(
                                                                                       args.input_height,
                                                                                       args.input_width),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(
                                                                                     args.input_height,
                                                                                     args.input_width),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_mv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_multi_modal_swin_transformer_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ViewWiseMultiModalSwinTransLastStagesCVA(phenotypic_dim=5, device=device,
                                                                         img_size=img_size,
                                                                         patch_size=patch_size, num_classes=5,
                                                                         window_size=window_size, pretrain=True,
                                                                         pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(
                                                                                       args.input_height,
                                                                                       args.input_width),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(
                                                                                     args.input_height,
                                                                                     args.input_width),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_mv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_multi_modal_only_image':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ViewWiseMultiModalOnlyImageInput(phenotypic_dim=5, device=device, img_size=img_size,
                                                                 patch_size=patch_size, num_classes=5,
                                                                 window_size=window_size, pretrain=True,
                                                                 pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(
                                                                                       args.input_height,
                                                                                       args.input_width),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(
                                                                                     args.input_height,
                                                                                     args.input_width),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                use_final_loss_weight = True
                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'flw={use_final_loss_weight}_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_mvmm_only_image_model(args, student_model,
                                                                                      train_dataloader,
                                                                                      val_dataloader,
                                                                                      class_loss_function,
                                                                                      optimizer, lr_scheduler, device,
                                                                                      class_names, train_dataset,
                                                                                      run_num, k, use_final_loss_weight)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_multi_modal_only_image_semi':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)
            unlabeled_case_names = []
            unlabeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_unlabeled_case_names.txt')
            with open(unlabeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    unlabeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)

            student_val_acc = []
            teacher_val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))
                train_case_names = labeled_train_case_names.copy()
                train_case_names.extend(unlabeled_case_names)

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ViewWiseMultiModalOnlyImageInput(phenotypic_dim=5, device=device, img_size=img_size,
                                                                 patch_size=patch_size, num_classes=5,
                                                                 window_size=window_size, pretrain=True,
                                                                 pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)
                teacher_model = ViewWiseMultiModalOnlyImageInput(phenotypic_dim=5, device=device, img_size=img_size,
                                                                 patch_size=patch_size, num_classes=5,
                                                                 window_size=window_size, pretrain=True,
                                                                 pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                for param in teacher_model.parameters():
                    param.detach_()
                teacher_model = teacher_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=args.weight_decay)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss(ignore_index=-1)
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight, ignore_index=-1)
                consistency_loss_function = softmax_mse_loss

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForSemiSupervisedBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                                 input_shape=(
                                                                                                     args.input_height,
                                                                                                     args.input_width),
                                                                                                 case_names=train_case_names,
                                                                                                 labeled_case_names=labeled_case_names,
                                                                                                 is_train=True)
                val_dataset = DDSMMVDatasetForSemiSupervisedBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                               input_shape=(
                                                                                                   args.input_height,
                                                                                                   args.input_width),
                                                                                               case_names=val_case_names,
                                                                                               labeled_case_names=labeled_case_names,
                                                                                               is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']

                # student_model, best_acc, best_class_acc = train_mvmm_only_image_model(args, student_model,
                #                                                                       train_dataloader,
                #                                                                       val_dataloader,
                #                                                                       class_loss_function,
                #                                                                       optimizer, lr_scheduler, device,
                #                                                                       class_names, train_dataset,
                #                                                                       run_num, k)

                student_model, student_best_acc, student_best_class_acc \
                    , teacher_best_acc, teacher_best_class_acc = train_mvmm_semi_only_image_model(args, student_model,
                                                                                                  teacher_model,
                                                                                                  train_dataloader,
                                                                                                  val_dataloader,
                                                                                                  class_loss_function,
                                                                                                  consistency_loss_function,
                                                                                                  optimizer,
                                                                                                  lr_scheduler, device,
                                                                                                  class_names,
                                                                                                  train_dataset, run_num,
                                                                                                  k,
                                                                                                  use_final_loss_weight=True)
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
                        avg_acc = np.mean(student_val_acc)
                        acc_std = np.std(student_val_acc)
                        lines = '{k}次训练的平均值：\nstudent_avg_acc: {avg_acc}    student_acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)

                        logger.info('******总训练结果******')
                        logger.info('Avg Student Acc: {:.4f}'.format(avg_acc))

                        avg_acc = np.mean(teacher_val_acc)
                        acc_std = np.std(teacher_val_acc)
                        lines = 'teacher_avg_acc: {avg_acc}    teacher_acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('Avg Teacher Acc: {:.4f}'.format(avg_acc))
        elif args.model_type == 'view_wise_multi_modal_multi_view_first':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ViewWiseMultiModalMultiViewFirst(phenotypic_dim=5, device=device, img_size=img_size,
                                                                 patch_size=patch_size, num_classes=5,
                                                                 window_size=window_size, pretrain=True,
                                                                 pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss()
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                   input_shape=(
                                                                                       args.input_height,
                                                                                       args.input_width),
                                                                                   case_names=labeled_train_case_names,
                                                                                   is_train=True)
                val_dataset = DDSMMVDatasetForBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                 input_shape=(
                                                                                     args.input_height,
                                                                                     args.input_width),
                                                                                 case_names=val_case_names,
                                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
                student_model, best_acc, best_class_acc = train_mv_model(args, student_model, train_dataloader,
                                                                         val_dataloader, class_loss_function,
                                                                         optimizer, lr_scheduler, device,
                                                                         class_names, train_dataset, run_num, k)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_acc: {best_acc}    1_acc: {acc_1}    2_acc: {acc_2}    ' \
                            '3_acc: {acc_3}    4_acc: {acc_4}    5_acc: {acc_5}\n'.format(
                        run_num=run_num,
                        best_acc=best_acc,
                        acc_1=best_class_acc['level_1'],
                        acc_2=best_class_acc['level_2'],
                        acc_3=best_class_acc['level_3'],
                        acc_4=best_class_acc['level_4'],
                        acc_5=best_class_acc['level_5'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_acc = np.mean(val_acc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_acc: {avg_acc}    acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_multi_modal_semi':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 10
            random.seed(444)

            labeled_case_names = []
            labeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_new_case_names.txt')
            with open(labeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    labeled_case_names.append(case_name)
            unlabeled_case_names = []
            unlabeled_case_names_file_path = os.path.join('data', 'ddsm_birads_classification_unlabeled_case_names.txt')
            with open(unlabeled_case_names_file_path) as f:
                case_names = f.readlines()
                for case_name in case_names:
                    case_name = case_name.replace('\n', '')
                    unlabeled_case_names.append(case_name)

            random.shuffle(labeled_case_names)

            val_num = math.ceil(len(labeled_case_names) // k)

            student_val_acc = []
            teacher_val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = labeled_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = labeled_case_names[run_num * val_num: len(labeled_case_names)]
                labeled_train_case_names = list(set(labeled_case_names) - set(val_case_names))
                train_case_names = labeled_train_case_names.copy()
                train_case_names.extend(unlabeled_case_names)

                # 创建模型
                img_size = 224
                patch_size = 4
                window_size = 7
                student_model = ViewWiseMultiModalSwinTransLastStagesCVA(phenotypic_dim=5, device=device,
                                                                         img_size=img_size,
                                                                         patch_size=patch_size, num_classes=5,
                                                                         window_size=window_size, pretrain=True,
                                                                         pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                student_model = student_model.to(device)
                teacher_model = ViewWiseMultiModalSwinTransLastStagesCVA(phenotypic_dim=5, device=device,
                                                                         img_size=img_size,
                                                                         patch_size=patch_size, num_classes=5,
                                                                         window_size=window_size, pretrain=True,
                                                                         pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                for param in teacher_model.parameters():
                    param.detach_()
                teacher_model = teacher_model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(student_model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                with_loss_weight = False
                malignant_weight = torch.FloatTensor([1, 50, 6, 1.7, 2.4]).to(device)
                class_loss_function = nn.NLLLoss(ignore_index=-1)
                if with_loss_weight:
                    class_loss_function = nn.NLLLoss(malignant_weight, ignore_index=-1)
                consistency_loss_function = softmax_mse_loss

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForSemiSupervisedBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                                 input_shape=(
                                                                                                     args.input_height,
                                                                                                     args.input_width),
                                                                                                 case_names=train_case_names,
                                                                                                 labeled_case_names=labeled_train_case_names,
                                                                                                 is_train=True)
                val_dataset = DDSMMVDatasetForSemiSupervisedBIRADSClassificationWithPhenotypic(args.data_path,
                                                                                               input_shape=(
                                                                                                   args.input_height,
                                                                                                   args.input_width),
                                                                                               case_names=val_case_names,
                                                                                               labeled_case_names=labeled_train_case_names,
                                                                                               is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate)

                # 学习率调整策略
                if args.cosine_lr:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                else:
                    warm_up_steps = args.warm_up_epochs * len(train_dataloader)
                    step_size = 5 * len(train_dataloader)
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

                file_path = os.path.join(args.txt_path, args.model_type + f'_lw={with_loss_weight}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'lr={args.lr}'
                                                                          f'.txt')

                class_names = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']

                student_model, student_best_acc, student_best_class_acc \
                    , teacher_best_acc, teacher_best_class_acc = train_mv_semi_model(args, student_model,
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
                        avg_acc = np.mean(student_val_acc)
                        acc_std = np.std(student_val_acc)
                        lines = '{k}次训练的平均值：\nstudent_avg_acc: {avg_acc}    student_acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)

                        logger.info('******总训练结果******')
                        logger.info('Avg Student Acc: {:.4f}'.format(avg_acc))

                        avg_acc = np.mean(teacher_val_acc)
                        acc_std = np.std(teacher_val_acc)
                        lines = 'teacher_avg_acc: {avg_acc}    teacher_acc_std: {acc_std}\n\n\n'.format(
                            k=k,
                            avg_acc=avg_acc,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('Avg Teacher Acc: {:.4f}'.format(avg_acc))


def train_sv_model(args, model, train_dataloader, val_dataloader, loss_function, optimizer, lr_scheduler,
                   device, class_names, train_dataset, run_num=0, k=1):
    logger.info("***** training start*****")
    logger.info(f"  Model Type = {model.__class__.__name__}")
    logger.info(f"  GPU ID = {args.gpu_id}")
    logger.info(f"  Num Workers = {args.num_workers}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch Size = {args.batch_size}")
    logger.info(f"  Learning Rate = {args.lr}")
    logger.info(f"  Input Size = (w:{args.input_width},h:{args.input_height})")

    comment = f'_{model.__class__.__name__}_{args.epochs}_{train_dataset.__class__.__name__}_{args.lr}_' \
              f'{optimizer.__class__.__name__}_({args.input_width},{args.input_height})'
    # writer = SummaryWriter(comment=comment)

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

                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                    # print('phenotypic[' + view + '] shape: ' + str(phenotypes[view].shape))
                    # print('images[' + view + '] shape: ' + str(images[view].shape))
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)
                # print('phenotypic[case] shape: ' + str(case_phenotypes.shape))
                # print('labels[l_breast] shape: ' + str(l_labels.shape))
                # print('labels[r_breast] shape: ' + str(r_labels.shape))

                optimizer.zero_grad()

                l_cc_outputs = torch.log(
                    F.softmax(model(images[VIEWS.L_CC], phenotypes[VIEWS.L_CC], case_phenotypes), 1))
                r_cc_outputs = torch.log(
                    F.softmax(model(images[VIEWS.R_CC], phenotypes[VIEWS.R_CC], case_phenotypes), 1))
                l_mlo_outputs = torch.log(
                    F.softmax(model(images[VIEWS.L_MLO], phenotypes[VIEWS.L_MLO], case_phenotypes), 1))
                r_mlo_outputs = torch.log(
                    F.softmax(model(images[VIEWS.R_MLO], phenotypes[VIEWS.R_MLO], case_phenotypes), 1))

                l_cc_loss = loss_function(l_cc_outputs, l_labels)
                r_cc_loss = loss_function(r_cc_outputs, r_labels)
                l_mlo_loss = loss_function(l_mlo_outputs, l_labels)
                r_mlo_loss = loss_function(r_mlo_outputs, r_labels)
                loss = (l_cc_loss + r_cc_loss + l_mlo_loss + r_mlo_loss) / 4

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

        # lr_scheduler.step()

        # 每代训练结束后评估模型，并获得一些评价指标，如果当前模型的指标高于best，则保存模型
        val_loss, val_acc, val_class_acc, val_class_all_num = evaluate_sv_model(args, model,
                                                                                val_dataloader,
                                                                                loss_function,
                                                                                device, class_names,
                                                                                run_num, k)

        if val_acc > best_acc:
            unchanged_epoch = 0
            best_acc = val_acc
            best_val_class_acc = val_class_acc
            # torch.save(model.state_dict(), os.path.join(args.out_path, comment + '_best_acc_model.pt'))
        else:
            unchanged_epoch += 1

        # train_dataset.is_train = False
        # train_loss, train_acc, train_auc, train_class_acc, train_class_all_num = evaluate_m_nm_sv_model(args, model,
        #                                                                                                 train_dataloader,
        #                                                                                                 loss_function,
        #                                                                                                 device,
        #                                                                                                 class_names,
        #                                                                                                 run_num,
        #                                                                                                 k)
        # train_dataset.is_train = True

        # 打印日志，以便实时查看训练中的一些评价指标
        logger.info('Best Acc: {:.4f}'.format(best_acc))
        logger.info(best_val_class_acc)
        logger.info('Val Acc: {:.4f}'.format(val_acc))
        logger.info('Val Class Acc:')
        logger.info(val_class_acc)
        logger.info(val_class_all_num)
        # logger.info('Train Auc: {:.4f}'.format(train_auc))
        # logger.info('Train Acc: {:.4f}'.format(train_acc))
        # logger.info('Train Class Acc:')
        # logger.info(train_class_acc)
        # logger.info(train_class_all_num)

        # 使用tensorboard记录训练数据
        # writer.add_scalar('train_loss', losses.avg, epoch)
        # writer.add_scalar('train_auc', train_auc, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('val_auc', val_auc, epoch)
        # writer.add_scalar('val_acc', val_acc, epoch)

        # 如果30代内auc没有提升，则结束训练
        if unchanged_epoch >= 30:
            break

    # writer.close()
    return model, best_acc, best_val_class_acc


def evaluate_sv_model(args, model, eval_dataloader, loss_function, device, class_names, run_num=0, k=1):
    losses = AverageMeter()
    acc = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    model.eval()

    with tqdm(eval_dataloader, position=0) as p_bar:

        class_correct_num = {cls_name: 0 for cls_name in class_names}
        class_all_num = {cls_name: 0 for cls_name in class_names}

        for batch_index, (images, phenotypes, malignant_labels) in enumerate(eval_dataloader):
            data_time.update(time.time() - end)

            with torch.no_grad():

                for view in VIEWS.LIST:
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                m_l_labels = malignant_labels['l_breast'].type(torch.LongTensor).to(device)
                m_r_labels = malignant_labels['r_breast'].type(torch.LongTensor).to(device)

                l_cc_outputs = torch.log(
                    F.softmax(model(images[VIEWS.L_CC], phenotypes[VIEWS.L_CC], case_phenotypes), 1))
                r_cc_outputs = torch.log(
                    F.softmax(model(images[VIEWS.R_CC], phenotypes[VIEWS.R_CC], case_phenotypes), 1))
                l_mlo_outputs = torch.log(
                    F.softmax(model(images[VIEWS.L_MLO], phenotypes[VIEWS.L_MLO], case_phenotypes), 1))
                r_mlo_outputs = torch.log(
                    F.softmax(model(images[VIEWS.R_MLO], phenotypes[VIEWS.R_MLO], case_phenotypes), 1))

                l_cc_loss = loss_function(l_cc_outputs, m_l_labels)
                r_cc_loss = loss_function(r_cc_outputs, m_r_labels)
                l_mlo_loss = loss_function(l_mlo_outputs, m_l_labels)
                r_mlo_loss = loss_function(r_mlo_outputs, m_r_labels)
                loss = (l_cc_loss + r_cc_loss + l_mlo_loss + r_mlo_loss) / 4

                l_cc_acc, l_cc_correct_num, l_cc_all_num = accuracy(l_cc_outputs, m_l_labels, class_names)
                r_cc_acc, r_cc_correct_num, r_cc_all_num = accuracy(r_cc_outputs, m_r_labels, class_names)
                l_mlo_acc, l_mlo_correct_num, l_mlo_all_num = accuracy(l_mlo_outputs, m_l_labels, class_names)
                r_mlo_acc, r_mlo_correct_num, r_mlo_all_num = accuracy(r_mlo_outputs, m_r_labels, class_names)

                for cls_name in class_names:
                    class_correct_num[cls_name] += l_cc_correct_num[cls_name]
                    class_correct_num[cls_name] += r_cc_correct_num[cls_name]
                    class_correct_num[cls_name] += l_mlo_correct_num[cls_name]
                    class_correct_num[cls_name] += r_mlo_correct_num[cls_name]
                    class_all_num[cls_name] += l_cc_all_num[cls_name]
                    class_all_num[cls_name] += r_cc_all_num[cls_name]
                    class_all_num[cls_name] += l_mlo_all_num[cls_name]
                    class_all_num[cls_name] += r_mlo_all_num[cls_name]

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


def train_mv_model(args, model, train_dataloader, val_dataloader,
                   loss_function, optimizer, lr_scheduler, device, class_names, train_dataset, run_num=0, k=1):
    logger.info("***** training start*****")
    logger.info(f"  Model Type = {model.__class__.__name__}")
    logger.info(f"  Num Workers = {args.num_workers}")
    logger.info(f"  GPU ID = {args.gpu_id}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch Size = {args.batch_size}")
    logger.info(f"  Learning Rate = {args.lr}")
    logger.info(f"  Input Size = (w:{args.input_width},h:{args.input_height})")

    comment = f' {model.__class__.__name__} {args.epochs} {train_dataset.__class__.__name__} {args.lr} ' \
              f'{optimizer.__class__.__name__} ({args.input_width},{args.input_height})'
    # writer = SummaryWriter(comment=comment)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    best_val_acc = 0
    best_val_class_acc = []
    unchanged_epoch = 0

    for epoch in range(args.epochs):
        losses.reset()
        model.train()

        with tqdm(train_dataloader, position=0) as p_bar:
            for batch_index, (images, phenotypes, labels) in enumerate(train_dataloader):
                data_time.update(time.time() - end)

                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                l_output, r_output = model(images, phenotypes, case_phenotypes)
                l_loss = loss_function(l_output, l_labels)
                r_loss = loss_function(r_output, r_labels)

                loss = (l_loss + r_loss) / 2

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

        # lr_scheduler.step()

        # 每代训练结束后评估模型，并获得一些评价指标，如果当前模型的指标高于best，则保存模型
        val_loss, val_acc, val_class_acc, val_all_num = evaluate_mv_model(args, model, val_dataloader,
                                                                          loss_function,
                                                                          device, class_names,
                                                                          run_num, k)

        if val_acc > best_val_acc:
            unchanged_epoch = 0
            best_val_acc = val_acc
            best_val_class_acc = val_class_acc
            torch.save(model.state_dict(), os.path.join(args.out_path, comment + '_best_acc_model.pt'))
        else:
            unchanged_epoch += 1

        # train_dataset.is_train = False
        # train_loss, train_acc, train_auc, train_class_acc, train_all_num = evaluate_m_nm_mv_model(args,
        #                                                                                           model,
        #                                                                                           train_dataloader,
        #                                                                                           loss_function,
        #                                                                                           device,
        #                                                                                           class_names,
        #                                                                                           run_num,
        #                                                                                           k)
        # train_dataset.is_train = True

        # 打印日志，以便实时查看训练中的一些评价指标
        logger.info('Best Acc: {:.4f}'.format(best_val_acc))
        logger.info(best_val_class_acc)
        logger.info('Val Acc: {:.4f}'.format(val_acc))
        # logger.info('Val Class Acc:')
        logger.info(val_class_acc)
        logger.info(val_all_num)
        # logger.info('Train Auc: {:.4f}'.format(train_auc))
        # logger.info('Train Acc: {:.4f}'.format(train_acc))
        # # logger.info('Train Class Acc:')
        # logger.info(train_class_acc)
        # logger.info(train_all_num)

        # 使用tensorboard记录训练数据
        # writer.add_scalar('train_loss', losses.avg, epoch)
        # writer.add_scalar('train_auc', train_auc, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('val_auc', val_auc, epoch)
        # writer.add_scalar('val_acc', val_acc, epoch)

        # 如果20代内auc没有提升，则结束训练
        if unchanged_epoch >= 30:
            break

    # writer.close()
    return model, best_val_acc, best_val_class_acc


def evaluate_mv_model(args, model, eval_dataloader, loss_function, device, class_names, run_num=0, k=1):
    losses = AverageMeter()
    # acc = AverageMeter()
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
                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                l_output, r_output = model(images, phenotypes, case_phenotypes)

                l_loss = loss_function(l_output, l_labels)
                r_loss = loss_function(r_output, r_labels)

                loss = (l_loss + r_loss) / 2
                l_acc, l_correct_num, l_all_num = accuracy(l_output, l_labels, class_names)
                r_acc, r_correct_num, r_all_num = accuracy(r_output, r_labels, class_names)

                for cls_name in class_names:
                    class_correct_num[cls_name] += l_correct_num[cls_name]
                    class_correct_num[cls_name] += r_correct_num[cls_name]
                    class_all_num[cls_name] += l_all_num[cls_name]
                    class_all_num[cls_name] += r_all_num[cls_name]

                losses.update(loss.item())
                # acc.update(eval_acc)
                batch_time.update(time.time() - end)
                end = time.time()

                p_bar.set_description(
                    "Run Num: {run_num:2}/{k:2}. Evaluate Batch: {batch:3}/{iter:3}. Avg Data: {data:.3f}s. "
                    "Avg Batch: {bt:.3f}s. Loss: {loss:.4f}".format(
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


def train_mv_semi_model(args, student_model, teacher_model, train_dataloader, val_dataloader, class_loss_function,
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

                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                student_l_output, student_r_output = student_model(images, phenotypes, case_phenotypes)
                teacher_l_output, teacher_r_output = teacher_model(images, phenotypes, case_phenotypes)
                teacher_l_output = Variable(teacher_l_output.detach().data, requires_grad=False)
                teacher_r_output = Variable(teacher_r_output.detach().data, requires_grad=False)

                student_l_class_loss = class_loss_function(student_l_output, l_labels)
                student_r_class_loss = class_loss_function(student_r_output, r_labels)
                consistency_l_loss = consistency_loss_function(student_l_output, teacher_l_output)
                consistency_r_loss = consistency_loss_function(student_r_output, teacher_r_output)
                loss = (student_l_class_loss + student_r_class_loss + get_current_consistency_weight(epoch, args) *
                        (consistency_l_loss + consistency_r_loss)) / 2

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
        val_class_all_num = evaluate_mv_semi_model(args, student_model,
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


def evaluate_mv_semi_model(args, student_model, teacher_model, eval_dataloader, class_loss_function, device,
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
                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                student_l_output, student_r_output = student_model(images, phenotypes, case_phenotypes)
                teacher_l_output, teacher_r_output = teacher_model(images, phenotypes, case_phenotypes)

                student_l_class_loss = class_loss_function(student_l_output, l_labels)
                student_r_class_loss = class_loss_function(student_r_output, r_labels)
                student_loss = (student_l_class_loss + student_r_class_loss) / 2
                teacher_l_class_loss = class_loss_function(teacher_l_output, l_labels)
                teacher_r_class_loss = class_loss_function(teacher_r_output, r_labels)
                teacher_loss = (teacher_l_class_loss + teacher_r_class_loss) / 2

                student_l_acc, student_l_class_correct_num_batch, class_l_all_num_batch = accuracy(student_l_output,
                                                                                                   l_labels,
                                                                                                   class_names)
                student_r_acc, student_r_class_correct_num_batch, class_r_all_num_batch = accuracy(student_r_output,
                                                                                                   r_labels,
                                                                                                   class_names)
                teacher_l_acc, teacher_l_class_correct_num_batch, _ = accuracy(teacher_l_output, l_labels, class_names)
                teacher_r_acc, teacher_r_class_correct_num_batch, _ = accuracy(teacher_r_output, r_labels, class_names)

                for cls_name in class_names:
                    student_class_correct_num[cls_name] += student_l_class_correct_num_batch[cls_name]
                    student_class_correct_num[cls_name] += student_r_class_correct_num_batch[cls_name]
                    teacher_class_correct_num[cls_name] += teacher_l_class_correct_num_batch[cls_name]
                    teacher_class_correct_num[cls_name] += teacher_r_class_correct_num_batch[cls_name]
                    class_all_num[cls_name] += class_l_all_num_batch[cls_name]
                    class_all_num[cls_name] += class_r_all_num_batch[cls_name]

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


def train_mvmm_only_image_model(args, model, train_dataloader, val_dataloader,
                                loss_function, optimizer, lr_scheduler, device, class_names, train_dataset, run_num=0,
                                k=1, use_final_loss_weight=False):
    logger.info("***** training start*****")
    logger.info(f"  Model Type = {model.__class__.__name__}")
    logger.info(f"  Num Workers = {args.num_workers}")
    logger.info(f"  GPU ID = {args.gpu_id}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch Size = {args.batch_size}")
    logger.info(f"  Learning Rate = {args.lr}")
    logger.info(f"  Input Size = (w:{args.input_width},h:{args.input_height})")

    comment = f' {model.__class__.__name__} {args.epochs} {train_dataset.__class__.__name__} {args.lr} ' \
              f'{optimizer.__class__.__name__} ({args.input_width},{args.input_height})'
    # writer = SummaryWriter(comment=comment)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    best_val_acc = 0
    best_val_class_acc = []
    unchanged_epoch = 0

    for epoch in range(args.epochs):
        losses.reset()
        model.train()

        with tqdm(train_dataloader, position=0) as p_bar:
            for batch_index, (images, phenotypes, labels) in enumerate(train_dataloader):
                data_time.update(time.time() - end)

                l_labels_shape, r_labels_shape, l_labels_margin, r_labels_margin, l_labels_subtlety, r_labels_subtlety \
                    = torch.zeros(args.batch_size), torch.zeros(args.batch_size), torch.zeros(args.batch_size), \
                      torch.zeros(args.batch_size), torch.zeros(args.batch_size), torch.zeros(args.batch_size)
                l_labels_shape = l_labels_shape.type(torch.FloatTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.FloatTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.FloatTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.FloatTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.FloatTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.FloatTensor).to(device)

                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)

                    phenotypes_split = torch.chunk(phenotypes[view], 3, 1)
                    phenotypes_shape, phenotypes_margin, phenotypes_subtlety = \
                        phenotypes_split[0].squeeze(1), phenotypes_split[1].squeeze(1), phenotypes_split[2].squeeze(1)
                    if view == VIEWS.L_CC or view == VIEWS.L_MLO:
                        l_labels_shape = torch.where(torch.gt(phenotypes_shape, l_labels_shape), phenotypes_shape,
                                                     l_labels_shape)
                        l_labels_margin = torch.where(torch.gt(phenotypes_margin, l_labels_margin), phenotypes_margin,
                                                      l_labels_margin)
                        l_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, l_labels_subtlety),
                                                        phenotypes_subtlety, l_labels_subtlety)
                    else:
                        r_labels_shape = torch.where(torch.gt(phenotypes_shape, r_labels_shape), phenotypes_shape,
                                                     r_labels_shape)
                        r_labels_margin = torch.where(torch.gt(phenotypes_margin, r_labels_margin), phenotypes_margin,
                                                      r_labels_margin)
                        r_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, r_labels_subtlety),
                                                        phenotypes_subtlety, r_labels_subtlety)

                l_labels_shape = l_labels_shape.type(torch.LongTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.LongTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.LongTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.LongTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.LongTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.LongTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                l_output, r_output, l_output_shape, r_output_shape, l_output_margin, r_output_margin, \
                l_output_subtlety, r_output_subtlety = model(images, case_phenotypes)

                l_shape_loss = loss_function(l_output_shape, l_labels_shape)
                r_shape_loss = loss_function(r_output_shape, r_labels_shape)
                l_margin_loss = loss_function(l_output_margin, l_labels_margin)
                r_margin_loss = loss_function(r_output_margin, r_labels_margin)
                l_subtlety_loss = loss_function(l_output_subtlety, l_labels_subtlety)
                r_subtlety_loss = loss_function(r_output_subtlety, r_labels_subtlety)
                l_loss = loss_function(l_output, l_labels)
                r_loss = loss_function(r_output, r_labels)

                if use_final_loss_weight:
                    final_loss_weight = ramps.sigmoid_rampup(epoch, args.consistency_rampup)
                else:
                    final_loss_weight = 1
                loss = (final_loss_weight * (l_loss + r_loss) + l_shape_loss + r_shape_loss + l_margin_loss +
                        r_margin_loss + l_subtlety_loss + r_subtlety_loss) / 8

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

        # lr_scheduler.step()

        # 每代训练结束后评估模型，并获得一些评价指标，如果当前模型的指标高于best，则保存模型
        val_loss, val_acc, val_class_acc, val_all_num, acc_shape, acc_margin, acc_subtlety = \
            evaluate_mvmm_only_image_model(args, model, val_dataloader,
                                           loss_function,
                                           device, class_names,
                                           run_num, k)

        if val_acc > best_val_acc:
            unchanged_epoch = 0
            best_val_acc = val_acc
            best_val_class_acc = val_class_acc
            torch.save(model.state_dict(), os.path.join(args.out_path, comment + '_best_acc_model.pt'))
        else:
            unchanged_epoch += 1

        # train_dataset.is_train = False
        # train_loss, train_acc, train_auc, train_class_acc, train_all_num = evaluate_m_nm_mv_model(args,
        #                                                                                           model,
        #                                                                                           train_dataloader,
        #                                                                                           loss_function,
        #                                                                                           device,
        #                                                                                           class_names,
        #                                                                                           run_num,
        #                                                                                           k)
        # train_dataset.is_train = True

        # 打印日志，以便实时查看训练中的一些评价指标
        logger.info('Best Acc: {:.4f}'.format(best_val_acc))
        logger.info(best_val_class_acc)
        logger.info('Val Acc: {:.4f}'.format(val_acc))
        logger.info('Shape Acc: {:.4f}'.format(acc_shape))
        logger.info('Margin Acc: {:.4f}'.format(acc_margin))
        logger.info('Subtlety Acc: {:.4f}'.format(acc_subtlety))
        # logger.info('Val Class Acc:')
        logger.info(val_class_acc)
        logger.info(val_all_num)
        # logger.info('Train Auc: {:.4f}'.format(train_auc))
        # logger.info('Train Acc: {:.4f}'.format(train_acc))
        # # logger.info('Train Class Acc:')
        # logger.info(train_class_acc)
        # logger.info(train_all_num)

        # 使用tensorboard记录训练数据
        # writer.add_scalar('train_loss', losses.avg, epoch)
        # writer.add_scalar('train_auc', train_auc, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('val_auc', val_auc, epoch)
        # writer.add_scalar('val_acc', val_acc, epoch)

        # 如果20代内auc没有提升，则结束训练
        if unchanged_epoch >= 30:
            break

    # writer.close()
    return model, best_val_acc, best_val_class_acc


def evaluate_mvmm_only_image_model(args, model, eval_dataloader, loss_function, device, class_names, run_num=0, k=1):
    losses = AverageMeter()
    acc_shape = AverageMeter()
    acc_margin = AverageMeter()
    acc_subtlety = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    model.eval()

    with tqdm(eval_dataloader, position=0) as p_bar:

        class_names_shape = ['none', 'OVAL', 'LOBULATED', 'ROUND', 'IRREGULAR', 'ARCHITECTURAL_DISTORTION']
        class_names_margin = ['none', 'ILL_DEFINED', 'MICROLOBULATED', 'CIRCUMSCRIBED', 'OBSCURED', 'SPICULATED']
        class_names_subtlety = ['0', '1', '2', '3', '4', '5']

        class_correct_num = {cls_name: 0 for cls_name in class_names}
        class_all_num = {cls_name: 0 for cls_name in class_names}
        # class_correct_num_shape = {cls_name: 0 for cls_name in class_names_shape}
        # class_all_num_shape = {cls_name: 0 for cls_name in class_names_shape}
        # class_correct_num_margin = {cls_name: 0 for cls_name in class_names_margin}
        # class_all_num_margin = {cls_name: 0 for cls_name in class_names_margin}
        # class_correct_num_subtlety = {cls_name: 0 for cls_name in class_names_subtlety}
        # class_all_num_subtlety = {cls_name: 0 for cls_name in class_names_subtlety}

        for batch_index, (images, phenotypes, labels) in enumerate(eval_dataloader):
            data_time.update(time.time() - end)

            with torch.no_grad():

                l_labels_shape, r_labels_shape, l_labels_margin, r_labels_margin, l_labels_subtlety, r_labels_subtlety \
                    = torch.zeros(labels['l_breast'].size(0)), torch.zeros(labels['l_breast'].size(0)), torch.zeros(
                    labels['l_breast'].size(0)), \
                      torch.zeros(labels['l_breast'].size(0)), torch.zeros(labels['l_breast'].size(0)), torch.zeros(
                    labels['l_breast'].size(0))
                l_labels_shape = l_labels_shape.type(torch.FloatTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.FloatTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.FloatTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.FloatTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.FloatTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.FloatTensor).to(device)

                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)

                    phenotypes_split = torch.chunk(phenotypes[view], 3, 1)
                    phenotypes_shape, phenotypes_margin, phenotypes_subtlety = \
                        phenotypes_split[0].squeeze(1), phenotypes_split[1].squeeze(1), phenotypes_split[2].squeeze(1)
                    if view == VIEWS.L_CC or view == VIEWS.L_MLO:
                        l_labels_shape = torch.where(torch.gt(phenotypes_shape, l_labels_shape), phenotypes_shape,
                                                     l_labels_shape)
                        l_labels_margin = torch.where(torch.gt(phenotypes_margin, l_labels_margin), phenotypes_margin,
                                                      l_labels_margin)
                        l_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, l_labels_subtlety),
                                                        phenotypes_subtlety, l_labels_subtlety)
                    else:
                        r_labels_shape = torch.where(torch.gt(phenotypes_shape, r_labels_shape), phenotypes_shape,
                                                     r_labels_shape)
                        r_labels_margin = torch.where(torch.gt(phenotypes_margin, r_labels_margin), phenotypes_margin,
                                                      r_labels_margin)
                        r_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, r_labels_subtlety),
                                                        phenotypes_subtlety, r_labels_subtlety)

                l_labels_shape = l_labels_shape.type(torch.LongTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.LongTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.LongTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.LongTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.LongTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.LongTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                l_output, r_output, l_output_shape, r_output_shape, l_output_margin, r_output_margin, \
                l_output_subtlety, r_output_subtlety = model(images, case_phenotypes)

                l_shape_loss = loss_function(l_output_shape, l_labels_shape)
                r_shape_loss = loss_function(r_output_shape, r_labels_shape)
                l_margin_loss = loss_function(l_output_margin, l_labels_margin)
                r_margin_loss = loss_function(r_output_margin, r_labels_margin)
                l_subtlety_loss = loss_function(l_output_subtlety, l_labels_subtlety)
                r_subtlety_loss = loss_function(r_output_subtlety, r_labels_subtlety)
                l_loss = loss_function(l_output, l_labels)
                r_loss = loss_function(r_output, r_labels)

                loss = (l_loss + r_loss + l_shape_loss + r_shape_loss + l_margin_loss +
                        r_margin_loss + l_subtlety_loss + r_subtlety_loss) / 8

                l_acc, l_correct_num, l_all_num = accuracy(l_output, l_labels, class_names)
                r_acc, r_correct_num, r_all_num = accuracy(r_output, r_labels, class_names)
                l_shape_acc, _, _ = accuracy(l_output_shape, l_labels_shape, class_names_shape)
                r_shape_acc, _, _ = accuracy(r_output_shape, r_labels_shape, class_names_shape)
                l_margin_acc, _, _ = accuracy(l_output_margin, l_labels_margin, class_names_margin)
                r_margin_acc, _, _ = accuracy(r_output_margin, r_labels_margin, class_names_margin)
                l_subtlety_acc, _, _ = accuracy(l_output_subtlety, l_labels_subtlety, class_names_subtlety)
                r_subtlety_acc, _, _ = accuracy(r_output_subtlety, r_labels_subtlety, class_names_subtlety)
                acc_shape.update(l_shape_acc, labels['l_breast'].size(0))
                acc_shape.update(r_margin_acc, labels['l_breast'].size(0))
                acc_margin.update(l_margin_acc, labels['l_breast'].size(0))
                acc_margin.update(r_margin_acc, labels['l_breast'].size(0))
                acc_subtlety.update(l_subtlety_acc, labels['l_breast'].size(0))
                acc_subtlety.update(r_subtlety_acc, labels['l_breast'].size(0))

                # l_shape_acc, l_shape_correct_num, l_shape_all_num = accuracy(l_output_shape, l_labels_shape, class_names_shape)
                # r_shape_acc, r_shape_correct_num, r_shape_all_num = accuracy(r_output_shape, r_labels_shape, class_names_shape)
                # l_margin_acc, l_margin_correct_num, l_margin_all_num = accuracy(l_output_margin, l_labels_margin, class_names_margin)
                # r_margin_acc, r_margin_correct_num, r_margin_all_num = accuracy(r_output_margin, r_labels_margin, class_names_margin)
                # l_subtlety_acc, l_subtlety_correct_num, l_subtlety_all_num = accuracy(l_output_subtlety, l_labels_subtlety, class_names_subtlety)
                # r_subtlety_acc, r_subtlety_correct_num, r_subtlety_all_num = accuracy(r_output_subtlety, r_labels_subtlety, class_names_subtlety)

                for cls_name in class_names:
                    class_correct_num[cls_name] += l_correct_num[cls_name]
                    class_correct_num[cls_name] += r_correct_num[cls_name]
                    class_all_num[cls_name] += l_all_num[cls_name]
                    class_all_num[cls_name] += r_all_num[cls_name]
                # for cls_name in class_names_shape:
                #     class_correct_num_shape[cls_name] += l_shape_correct_num[cls_name]
                #     class_correct_num_shape[cls_name] += r_shape_correct_num[cls_name]
                #     class_all_num_shape[cls_name] += l_shape_all_num[cls_name]
                #     class_all_num_shape[cls_name] += r_shape_all_num[cls_name]
                # for cls_name in class_names_margin:
                #     class_correct_num_margin[cls_name] += l_margin_correct_num[cls_name]
                #     class_correct_num_margin[cls_name] += r_margin_correct_num[cls_name]
                #     class_all_num_margin[cls_name] += l_margin_all_num[cls_name]
                #     class_all_num_margin[cls_name] += r_margin_all_num[cls_name]
                # for cls_name in class_names_subtlety:
                #     class_correct_num_subtlety[cls_name] += l_subtlety_correct_num[cls_name]
                #     class_correct_num_subtlety[cls_name] += r_subtlety_correct_num[cls_name]
                #     class_all_num_subtlety[cls_name] += l_subtlety_all_num[cls_name]
                #     class_all_num_subtlety[cls_name] += r_subtlety_all_num[cls_name]

                losses.update(loss.item())
                # acc.update(eval_acc)
                batch_time.update(time.time() - end)
                end = time.time()

                p_bar.set_description(
                    "Run Num: {run_num:2}/{k:2}. Evaluate Batch: {batch:3}/{iter:3}. Avg Data: {data:.3f}s. "
                    "Avg Batch: {bt:.3f}s. Loss: {loss:.4f}".format(
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

    return losses.avg, total_acc, class_acc, class_all_num, acc_shape.avg, acc_margin.avg, acc_subtlety.avg


def train_mvmm_semi_only_image_model(args, student_model, teacher_model, train_dataloader, val_dataloader,
                                     class_loss_function,
                                     consistency_loss_function, optimizer, lr_scheduler, device, class_names,
                                     train_dataset,
                                     run_num=0, k=1, use_final_loss_weight=False):
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

                l_labels_shape, r_labels_shape, l_labels_margin, r_labels_margin, l_labels_subtlety, r_labels_subtlety \
                    = torch.zeros(labels['l_breast'].size(0)), torch.zeros(labels['l_breast'].size(0)), torch.zeros(
                    labels['l_breast'].size(0)), \
                      torch.zeros(labels['l_breast'].size(0)), torch.zeros(labels['l_breast'].size(0)), torch.zeros(
                    labels['l_breast'].size(0))
                l_labels_shape = l_labels_shape.type(torch.FloatTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.FloatTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.FloatTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.FloatTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.FloatTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.FloatTensor).to(device)

                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)

                    phenotypes_split = torch.chunk(phenotypes[view], 3, 1)
                    phenotypes_shape, phenotypes_margin, phenotypes_subtlety = \
                        phenotypes_split[0].squeeze(1), phenotypes_split[1].squeeze(1), phenotypes_split[2].squeeze(1)
                    if view == VIEWS.L_CC or view == VIEWS.L_MLO:
                        l_labels_shape = torch.where(torch.gt(phenotypes_shape, l_labels_shape), phenotypes_shape,
                                                     l_labels_shape)
                        l_labels_margin = torch.where(torch.gt(phenotypes_margin, l_labels_margin), phenotypes_margin,
                                                      l_labels_margin)
                        l_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, l_labels_subtlety),
                                                        phenotypes_subtlety, l_labels_subtlety)
                    else:
                        r_labels_shape = torch.where(torch.gt(phenotypes_shape, r_labels_shape), phenotypes_shape,
                                                     r_labels_shape)
                        r_labels_margin = torch.where(torch.gt(phenotypes_margin, r_labels_margin), phenotypes_margin,
                                                      r_labels_margin)
                        r_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, r_labels_subtlety),
                                                        phenotypes_subtlety, r_labels_subtlety)

                l_labels_shape = l_labels_shape.type(torch.LongTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.LongTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.LongTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.LongTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.LongTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.LongTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                student_l_output, student_r_output, \
                student_l_output_shape, student_r_output_shape, \
                student_l_output_margin, student_r_output_margin, \
                student_l_output_subtlety, student_r_output_subtlety \
                    = student_model(images, case_phenotypes)

                teacher_l_output, teacher_r_output, \
                teacher_l_output_shape, teacher_r_output_shape, \
                teacher_l_output_margin, teacher_r_output_margin, \
                teacher_l_output_subtlety, teacher_r_output_subtlety \
                    = teacher_model(images, case_phenotypes)

                teacher_l_output = Variable(teacher_l_output.detach().data, requires_grad=False)
                teacher_r_output = Variable(teacher_r_output.detach().data, requires_grad=False)
                teacher_l_output_shape = Variable(teacher_l_output_shape.detach().data, requires_grad=False)
                teacher_r_output_shape = Variable(teacher_r_output_shape.detach().data, requires_grad=False)
                teacher_l_output_margin = Variable(teacher_l_output_margin.detach().data, requires_grad=False)
                teacher_r_output_margin = Variable(teacher_r_output_margin.detach().data, requires_grad=False)
                teacher_l_output_subtlety = Variable(teacher_l_output_subtlety.detach().data, requires_grad=False)
                teacher_r_output_subtlety = Variable(teacher_r_output_subtlety.detach().data, requires_grad=False)

                student_l_class_loss = class_loss_function(student_l_output, l_labels)
                student_r_class_loss = class_loss_function(student_r_output, r_labels)
                student_l_shape_class_loss = class_loss_function(student_l_output_shape, l_labels_shape)
                student_r_shape_class_loss = class_loss_function(student_r_output_shape, r_labels_shape)
                student_l_margin_class_loss = class_loss_function(student_l_output_margin, l_labels_margin)
                student_r_margin_class_loss = class_loss_function(student_r_output_margin, r_labels_margin)
                student_l_subtlety_class_loss = class_loss_function(student_l_output_subtlety, l_labels_subtlety)
                student_r_subtlety_class_loss = class_loss_function(student_r_output_subtlety, r_labels_subtlety)

                consistency_l_loss = consistency_loss_function(student_l_output, teacher_l_output)
                consistency_r_loss = consistency_loss_function(student_r_output, teacher_r_output)
                consistency_l_shape_loss = consistency_loss_function(student_l_output_shape, teacher_l_output_shape)
                consistency_r_shape_loss = consistency_loss_function(student_r_output_shape, teacher_r_output_shape)
                consistency_l_margin_loss = consistency_loss_function(student_l_output_margin, teacher_l_output_margin)
                consistency_r_margin_loss = consistency_loss_function(student_r_output_margin, teacher_r_output_margin)
                consistency_l_subtlety_loss = consistency_loss_function(student_l_output_subtlety,
                                                                        teacher_l_output_subtlety)
                consistency_r_subtlety_loss = consistency_loss_function(student_r_output_subtlety,
                                                                        teacher_r_output_subtlety)

                if use_final_loss_weight:
                    final_loss_weight = ramps.sigmoid_rampup(epoch, args.consistency_rampup)
                else:
                    final_loss_weight = 1

                class_loss = (final_loss_weight * (student_l_class_loss + student_r_class_loss) +
                              student_l_shape_class_loss + student_r_shape_class_loss + student_l_margin_class_loss +
                              student_r_margin_class_loss + student_l_subtlety_class_loss +
                              student_r_subtlety_class_loss) / 8
                consistency_loss = get_current_consistency_weight(epoch, args) * (
                        consistency_l_loss + consistency_r_loss +
                        consistency_l_shape_loss + consistency_r_shape_loss +
                        consistency_l_margin_loss + consistency_r_margin_loss +
                        consistency_l_subtlety_loss + consistency_r_subtlety_loss) / 8
                loss = class_loss + consistency_loss

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
        student_acc_shape, student_acc_margin, student_acc_subtlety, \
        teacher_val_loss, teacher_val_acc, teacher_val_class_acc, \
        teacher_acc_shape, teacher_acc_margin, teacher_acc_subtlety, \
        val_class_all_num = evaluate_mvmm_semi_only_image_model(args, student_model,
                                                                teacher_model,
                                                                val_dataloader,
                                                                class_loss_function,
                                                                device, class_names,
                                                                run_num, k)

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
        # logger.info('Student Val Class Acc:')
        logger.info('Student Shape Acc: {:.4f}'.format(student_acc_shape))
        logger.info('Student Margin Acc: {:.4f}'.format(student_acc_margin))
        logger.info('Student Subtlety Acc: {:.4f}'.format(student_acc_subtlety))
        logger.info(student_val_class_acc)
        logger.info('')
        logger.info('Best Teacher Acc: {:.4f}'.format(teacher_best_acc))
        logger.info(teacher_best_val_class_acc)
        logger.info('Teacher Val Acc: {:.4f}'.format(teacher_val_acc))
        # logger.info('Teacher Val Class Acc:')
        logger.info('Teacher Shape Acc: {:.4f}'.format(teacher_acc_shape))
        logger.info('Teacher Margin Acc: {:.4f}'.format(teacher_acc_margin))
        logger.info('Teacher Subtlety Acc: {:.4f}'.format(teacher_acc_subtlety))
        logger.info(teacher_val_class_acc)
        logger.info('')
        logger.info(val_class_all_num)

        # 如果30代内auc没有提升，则结束训练
        if unchanged_epoch >= 30:
            break

    return student_model, student_best_acc, student_best_val_class_acc, teacher_best_acc, teacher_best_val_class_acc


def evaluate_mvmm_semi_only_image_model(args, student_model, teacher_model, eval_dataloader, class_loss_function, device,
                                        class_names,
                                        run_num=0, k=1):
    student_losses = AverageMeter()
    teacher_losses = AverageMeter()
    student_acc_shape = AverageMeter()
    student_acc_margin = AverageMeter()
    student_acc_subtlety = AverageMeter()
    teacher_acc_shape = AverageMeter()
    teacher_acc_margin = AverageMeter()
    teacher_acc_subtlety = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    student_model.eval()

    with tqdm(eval_dataloader, position=0) as p_bar:

        student_class_correct_num = {cls_name: 0 for cls_name in class_names}
        teacher_class_correct_num = {cls_name: 0 for cls_name in class_names}
        class_all_num = {cls_name: 0 for cls_name in class_names}

        class_names_shape = ['none', 'OVAL', 'LOBULATED', 'ROUND', 'IRREGULAR', 'ARCHITECTURAL_DISTORTION']
        class_names_margin = ['none', 'ILL_DEFINED', 'MICROLOBULATED', 'CIRCUMSCRIBED', 'OBSCURED', 'SPICULATED']
        class_names_subtlety = ['0', '1', '2', '3', '4', '5']

        for batch_index, (images, phenotypes, labels) in enumerate(eval_dataloader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                l_labels_shape, r_labels_shape, l_labels_margin, r_labels_margin, l_labels_subtlety, r_labels_subtlety \
                    = torch.zeros(labels['l_breast'].size(0)), torch.zeros(labels['l_breast'].size(0)), torch.zeros(
                    labels['l_breast'].size(0)), \
                      torch.zeros(labels['l_breast'].size(0)), torch.zeros(labels['l_breast'].size(0)), torch.zeros(
                    labels['l_breast'].size(0))
                l_labels_shape = l_labels_shape.type(torch.FloatTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.FloatTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.FloatTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.FloatTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.FloatTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.FloatTensor).to(device)

                for view in VIEWS.LIST:
                    phenotypes[view] = phenotypes[view].type(torch.FloatTensor).to(device)
                    images[view] = images[view].type(torch.FloatTensor).to(device)

                    phenotypes_split = torch.chunk(phenotypes[view], 3, 1)
                    phenotypes_shape, phenotypes_margin, phenotypes_subtlety = \
                        phenotypes_split[0].squeeze(1), phenotypes_split[1].squeeze(1), phenotypes_split[2].squeeze(1)
                    if view == VIEWS.L_CC or view == VIEWS.L_MLO:
                        l_labels_shape = torch.where(torch.gt(phenotypes_shape, l_labels_shape), phenotypes_shape,
                                                     l_labels_shape)
                        l_labels_margin = torch.where(torch.gt(phenotypes_margin, l_labels_margin), phenotypes_margin,
                                                      l_labels_margin)
                        l_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, l_labels_subtlety),
                                                        phenotypes_subtlety, l_labels_subtlety)
                    else:
                        r_labels_shape = torch.where(torch.gt(phenotypes_shape, r_labels_shape), phenotypes_shape,
                                                     r_labels_shape)
                        r_labels_margin = torch.where(torch.gt(phenotypes_margin, r_labels_margin), phenotypes_margin,
                                                      r_labels_margin)
                        r_labels_subtlety = torch.where(torch.gt(phenotypes_subtlety, r_labels_subtlety),
                                                        phenotypes_subtlety, r_labels_subtlety)

                l_labels_shape = l_labels_shape.type(torch.LongTensor).to(device)
                r_labels_shape = r_labels_shape.type(torch.LongTensor).to(device)
                l_labels_margin = l_labels_margin.type(torch.LongTensor).to(device)
                r_labels_margin = r_labels_margin.type(torch.LongTensor).to(device)
                l_labels_subtlety = l_labels_subtlety.type(torch.LongTensor).to(device)
                r_labels_subtlety = r_labels_subtlety.type(torch.LongTensor).to(device)
                case_phenotypes = phenotypes['case'].type(torch.LongTensor).to(device)
                l_labels = labels['l_breast'].type(torch.LongTensor).to(device)
                r_labels = labels['r_breast'].type(torch.LongTensor).to(device)

                student_l_output, student_r_output, \
                student_l_output_shape, student_r_output_shape, \
                student_l_output_margin, student_r_output_margin, \
                student_l_output_subtlety, student_r_output_subtlety \
                    = student_model(images, case_phenotypes)

                teacher_l_output, teacher_r_output, \
                teacher_l_output_shape, teacher_r_output_shape, \
                teacher_l_output_margin, teacher_r_output_margin, \
                teacher_l_output_subtlety, teacher_r_output_subtlety \
                    = teacher_model(images, case_phenotypes)

                student_l_class_loss = class_loss_function(student_l_output, l_labels)
                student_r_class_loss = class_loss_function(student_r_output, r_labels)
                student_l_shape_class_loss = class_loss_function(student_l_output_shape, l_labels_shape)
                student_r_shape_class_loss = class_loss_function(student_r_output_shape, r_labels_shape)
                student_l_margin_class_loss = class_loss_function(student_l_output_margin, l_labels_margin)
                student_r_margin_class_loss = class_loss_function(student_r_output_margin, r_labels_margin)
                student_l_subtlety_class_loss = class_loss_function(student_l_output_subtlety, l_labels_subtlety)
                student_r_subtlety_class_loss = class_loss_function(student_r_output_subtlety, r_labels_subtlety)

                teacher_l_class_loss = class_loss_function(teacher_l_output, l_labels)
                teacher_r_class_loss = class_loss_function(teacher_r_output, r_labels)
                teacher_l_shape_class_loss = class_loss_function(teacher_l_output_shape, l_labels_shape)
                teacher_r_shape_class_loss = class_loss_function(teacher_r_output_shape, r_labels_shape)
                teacher_l_margin_class_loss = class_loss_function(teacher_l_output_margin, l_labels_margin)
                teacher_r_margin_class_loss = class_loss_function(teacher_r_output_margin, r_labels_margin)
                teacher_l_subtlety_class_loss = class_loss_function(teacher_l_output_subtlety, l_labels_subtlety)
                teacher_r_subtlety_class_loss = class_loss_function(teacher_r_output_subtlety, r_labels_subtlety)

                student_loss = (student_l_class_loss + student_r_class_loss +
                                student_l_shape_class_loss + student_r_shape_class_loss +
                                student_l_margin_class_loss + student_r_margin_class_loss +
                                student_l_subtlety_class_loss + student_r_subtlety_class_loss) / 8
                teacher_loss = (teacher_l_class_loss + teacher_r_class_loss +
                                teacher_l_shape_class_loss + teacher_r_shape_class_loss +
                                teacher_l_margin_class_loss + teacher_r_margin_class_loss +
                                teacher_l_subtlety_class_loss + teacher_r_subtlety_class_loss) / 8

                student_l_acc, student_l_correct_num, l_all_num = accuracy(student_l_output, l_labels, class_names)
                student_r_acc, student_r_correct_num, r_all_num = accuracy(student_r_output, r_labels, class_names)
                student_l_shape_acc, _, _ = accuracy(student_l_output_shape, l_labels_shape, class_names_shape)
                student_r_shape_acc, _, _ = accuracy(student_r_output_shape, r_labels_shape, class_names_shape)
                student_l_margin_acc, _, _ = accuracy(student_l_output_margin, l_labels_margin, class_names_margin)
                student_r_margin_acc, _, _ = accuracy(student_r_output_margin, r_labels_margin, class_names_margin)
                student_l_subtlety_acc, _, _ = accuracy(student_l_output_subtlety, l_labels_subtlety,
                                                        class_names_subtlety)
                student_r_subtlety_acc, _, _ = accuracy(student_r_output_subtlety, r_labels_subtlety,
                                                        class_names_subtlety)
                student_acc_shape.update(student_l_shape_acc, labels['l_breast'].size(0))
                student_acc_shape.update(student_r_shape_acc, labels['l_breast'].size(0))
                student_acc_margin.update(student_l_margin_acc, labels['l_breast'].size(0))
                student_acc_margin.update(student_r_margin_acc, labels['l_breast'].size(0))
                student_acc_subtlety.update(student_l_subtlety_acc, labels['l_breast'].size(0))
                student_acc_subtlety.update(student_r_subtlety_acc, labels['l_breast'].size(0))

                teacher_l_acc, teacher_l_correct_num, _ = accuracy(teacher_l_output, l_labels, class_names)
                teacher_r_acc, teacher_r_correct_num, _ = accuracy(teacher_r_output, r_labels, class_names)
                teacher_l_shape_acc, _, _ = accuracy(teacher_l_output_shape, l_labels_shape, class_names_shape)
                teacher_r_shape_acc, _, _ = accuracy(teacher_r_output_shape, r_labels_shape, class_names_shape)
                teacher_l_margin_acc, _, _ = accuracy(teacher_l_output_margin, l_labels_margin, class_names_margin)
                teacher_r_margin_acc, _, _ = accuracy(teacher_r_output_margin, r_labels_margin, class_names_margin)
                teacher_l_subtlety_acc, _, _ = accuracy(teacher_l_output_subtlety, l_labels_subtlety,
                                                        class_names_subtlety)
                teacher_r_subtlety_acc, _, _ = accuracy(teacher_r_output_subtlety, r_labels_subtlety,
                                                        class_names_subtlety)
                teacher_acc_shape.update(teacher_l_shape_acc, labels['l_breast'].size(0))
                teacher_acc_shape.update(teacher_r_shape_acc, labels['l_breast'].size(0))
                teacher_acc_margin.update(teacher_l_margin_acc, labels['l_breast'].size(0))
                teacher_acc_margin.update(teacher_r_margin_acc, labels['l_breast'].size(0))
                teacher_acc_subtlety.update(teacher_l_subtlety_acc, labels['l_breast'].size(0))
                teacher_acc_subtlety.update(teacher_r_subtlety_acc, labels['l_breast'].size(0))

                for cls_name in class_names:
                    student_class_correct_num[cls_name] += student_l_correct_num[cls_name]
                    student_class_correct_num[cls_name] += student_r_correct_num[cls_name]
                    teacher_class_correct_num[cls_name] += teacher_l_correct_num[cls_name]
                    teacher_class_correct_num[cls_name] += teacher_r_correct_num[cls_name]
                    class_all_num[cls_name] += l_all_num[cls_name]
                    class_all_num[cls_name] += r_all_num[cls_name]

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
           student_acc_shape.avg, student_acc_margin.avg, student_acc_subtlety.avg, \
           teacher_losses.avg, teacher_total_acc, teacher_class_acc, \
           teacher_acc_shape.avg, teacher_acc_margin.avg, teacher_acc_subtlety.avg, class_all_num


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


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


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


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == '__main__':
    main()
