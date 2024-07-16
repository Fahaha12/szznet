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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torchvision
from sklearn.metrics import confusion_matrix

from model.classification_models import *
from utils.classification_data import DDSMMVDatasetForMNmClassification, ddsm_mv_dataset_for_m_nm_classification_collate
from utils.constants import VIEWS
from utils.meter import AverageMeter

logger = logging.getLogger(__name__)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def main():
    parser = argparse.ArgumentParser(description='MultiViewBreastCancerClassification')

    parser.add_argument('--data-path', default=r'data/DDSM', type=str, help='data path')
    parser.add_argument('--out-path', default=r'result', help='directory to output the result')
    parser.add_argument('--txt-path', default=r'classification_result', help='directory to output the result')

    parser.add_argument('--gpu-id', default=0, type=int, help='visible gpu id(s)')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', default=300, type=int, help='number of total steps to run')
    parser.add_argument('--warm_up_epochs', default=20, type=int, help='number of total steps to run')
    parser.add_argument('--batch-size', default=16, type=int, help='train batch_size')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--input-width', default=224, type=int, help='the width of input images')
    parser.add_argument('--input-height', default=224, type=int, help='the height of input images')
    parser.add_argument('--model_type', default='single_resnet50', type=str,
                        help='the type of model used',

                        choices=['single_resnet50', 'single_swin_transformer',

                                 'view_wise_resnet50_last_stage_concat',
                                 'view_wise_swin_transformer_last_stage_concat',
                                 'view_wise_resnet50_last_stage_cva',
                                 'view_wise_swin_transformer_all_stage_cva',
                                 'view_wise_swin_transformer_last_stage_cva',
                                 'view_wise_swin_transformer_three_stage_cva',
                                 'view_wise_swin_transformer_two_stage_cva',
                                 'view_wise_swin_transformer_one_stage_cva',
                                 'view_wise_swin_transformer_before_stage_cva',

                                 'breast_wise_resnet50_last_stage_concat',
                                 'breast_wise_swin_transformer_last_stage_concat',
                                 'breast_wise_resnet50_last_stage_cva',
                                 'breast_wise_swin_transformer_last_stage_cva',

                                 'joint_resnet50_last_stage_concat',
                                 'joint_swin_transformer_last_stage_concat',
                                 'joint_resnet50_last_stage_cva',
                                 'joint_swin_transformer_last_stage_cva', ])
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

    dataset_name = args.data_path.split('/')[-1]
    if dataset_name == 'DDSM':
        if args.model_type == 'single_resnet50':
            # 划分训练集和验证集
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)
            test_num = math.ceil(len(train_val_case_names) // 10)
            test_case_names = train_val_case_names[0: test_num]
            train_val_case_names = list(set(train_val_case_names) - set(test_case_names))

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []
            test_acc_ = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                backbone = 'resnet50'
                model = SVModelResnet(pretrained=True, backbone=backbone)
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

                # 定义loss函数
                malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(args.input_height, args.input_width),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(args.input_height, args.input_width),
                                                                case_names=val_case_names, is_train=False)
                test_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                 input_shape=(args.input_height, args.input_width),
                                                                 case_names=test_case_names,
                                                                 is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True, pin_memory=True, drop_last=False,
                                             collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  Backbone = {backbone}")
                model, best_auc, best_acc, best_class_acc = train_sv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)
                test_loss, test_acc, test_auc, test_class_acc, _ = evaluate_sv_model(args, model,
                                                                                     test_dataloader,
                                                                                     loss_function,
                                                                                     device, class_names,
                                                                                     run_num, k)
                logger.info('******第' + str(run_num) + '次训练结果******')
                logger.info('Best Auc: {:.4f}'.format(best_auc))
                logger.info('Best Acc: {:.4f}'.format(best_acc))
                logger.info(best_class_acc)
                logger.info('Test Auc: {:.4f}'.format(test_auc))
                logger.info('Test Acc: {:.4f}'.format(test_acc))
                logger.info(test_class_acc)
                test_acc_.append(test_acc)

                file_path = os.path.join(args.txt_path, args.model_type + f'_{backbone}_'
                                                                          f'({args.input_width},{args.input_height})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    lines = 'test_auc: {test_auc}    test_acc: {test_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        test_auc=test_auc,
                        test_acc=test_acc,
                        nm_acc=test_class_acc['not_malignant'],
                        m_acc=test_class_acc['malignant'])
                    f.writelines(lines)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        avg_test_auc = np.mean(test_auc)
                        avg_test_acc = np.mean(test_acc)
                        auc_test_std = np.std(test_auc)
                        acc_test_std = np.std(test_acc)
                        lines = 'avg_test_auc: {avg_test_auc}    avg_test_acc: {avg_test_acc}    auc_test_std: {auc_test_std}' \
                                '    acc_test_std: {acc_test_std}\n'.format(
                            avg_test_auc=avg_test_auc,
                            avg_test_acc=avg_test_acc,
                            auc_test_std=auc_test_std,
                            acc_test_std=acc_test_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                        logger.info('Avg Test Auc: {:.4f}'.format(avg_test_auc))
                        logger.info('Avg Test Acc: {:.4f}'.format(avg_test_acc))
                # break
        elif args.model_type == 'single_swin_transformer':
            # 划分训练集和验证集
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = SVModelSwinTransformer(img_size=img_size, patch_size=patch_size,
                                               num_classes=2, window_size=window_size, pretrain=True,
                                               pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)

                # 定义loss函数
                malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                loss_function = nn.NLLLoss(malignant_weight)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_sv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
        elif args.model_type == 'view_wise_resnet50_last_stage_concat':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                backbone = 'resnet50'
                model = ViewWiseResnet50LastStageConcat(pretrained=True, backbone=backbone)
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(args.input_height, args.input_width),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(args.input_height, args.input_width),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)
                file_path = os.path.join(args.txt_path, args.model_type + f'_{backbone}_'
                                                                          f'({args.input_width},{args.input_height})'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_last_stage_concat':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                pretrain = True
                model = ViewWiseSwinTransLastStagesConcat(img_size=img_size, patch_size=patch_size,
                                                          num_classes=2, window_size=window_size, pretrain=pretrain,
                                                          pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")
                logger.info(f"  PreTrain = {pretrain}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'pretrain={pretrain}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_resnet50_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                img_size = 224
                window_size = 7
                model = ViewWiseResnet50LastStagesCVA(num_classes=2, window_size=window_size, pretrain=True)
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_all_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = ViewWiseSwinTransAllStagesCVA(img_size=img_size, patch_size=patch_size,
                                                      num_classes=2, window_size=window_size, pretrain=True,
                                                      pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                pretrain = True
                model = ViewWiseSwinTransLastStagesCVA(img_size=img_size, patch_size=patch_size,
                                                       num_classes=2, window_size=window_size, pretrain=pretrain,
                                                       pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")
                logger.info(f"  PreTrain = {pretrain}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                # f'pretrain={pretrain}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_three_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = ViewWiseSwinTransThreeStagesCVA(img_size=img_size, patch_size=patch_size,
                                                        num_classes=2, window_size=window_size, pretrain=True,
                                                        pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_two_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = ViewWiseSwinTransTwoStagesCVA(img_size=img_size, patch_size=patch_size,
                                                      num_classes=2, window_size=window_size, pretrain=True,
                                                      pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_one_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = ViewWiseSwinTransOneStagesCVA(img_size=img_size, patch_size=patch_size,
                                                      num_classes=2, window_size=window_size, pretrain=True,
                                                      pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'view_wise_swin_transformer_before_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = ViewWiseSwinTransBeforeStagesCVA(img_size=img_size, patch_size=patch_size,
                                                         num_classes=2, window_size=window_size, pretrain=True,
                                                         pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'breast_wise_resnet50_last_stage_concat':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                backbone = 'resnet50'
                split = False
                model = BreastWiseResnet50LastStageConcat(pretrained=True, backbone=backbone, split=split)
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(args.input_height, args.input_width),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(args.input_height, args.input_width),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  Backbone = {backbone}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)
                file_path = os.path.join(args.txt_path, args.model_type + f'_{backbone}_'
                                                                          f'{split}_'
                                                                          f'({args.input_width},{args.input_height})'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
        elif args.model_type == 'breast_wise_swin_transformer_last_stage_concat':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = BreastWiseSwinTransLastStagesConcat(img_size=img_size, patch_size=patch_size,
                                                            num_classes=2, window_size=window_size, pretrain=True,
                                                            pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'breast_wise_resnet50_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = BreastWiseResnet50LastStagesCVA(num_classes=2, window_size=window_size, pretrain=True)
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'breast_wise_swin_transformer_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = BreastWiseSwinTransLastStagesCVA(img_size=img_size, patch_size=patch_size,
                                                         num_classes=2, window_size=window_size, pretrain=True,
                                                         pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'breast_wise_swin_transformer_last_stage_cva_with_class_token':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = BreastWiseSwinTransLastStagesCVAWithClassToken(img_size=img_size, patch_size=patch_size,
                                                                       num_classes=2, window_size=window_size,
                                                                       pretrain=True,
                                                                       pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                loss_function = nn.NLLLoss(malignant_weight)
                # loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'joint_resnet50_last_stage_concat':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

            for run_num in range(k):
                if run_num < k - 1:
                    val_case_names = train_val_case_names[run_num * val_num: (run_num + 1) * val_num]
                else:
                    val_case_names = train_val_case_names[run_num * val_num: len(train_val_case_names)]
                train_case_names = list(set(train_val_case_names) - set(val_case_names))

                # 创建模型
                backbone = 'resnet50'
                model = JointResnet50LastStageConcat(pretrained=True, backbone=backbone)
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(args.input_height, args.input_width),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(args.input_height, args.input_width),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  Backbone = {backbone}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)
                file_path = os.path.join(args.txt_path, args.model_type + f'_{backbone}_'
                                                                          f'({args.input_width},{args.input_height})'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
        elif args.model_type == 'joint_swin_transformer_last_stage_concat':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = JointSwinTransLastStagesConcat(img_size=img_size, patch_size=patch_size,
                                                       num_classes=2, window_size=window_size, pretrain=True,
                                                       pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'joint_resnet50_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = JointResnet50LastStagesCVA(num_classes=2, window_size=window_size, pretrain=True)
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'joint_swin_transformer_last_stage_cva':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                model = JointSwinTransLastStagesCVA(img_size=img_size, patch_size=patch_size,
                                                    num_classes=2, window_size=window_size, pretrain=True,
                                                    pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                # loss_function = nn.CrossEntropyLoss()
                # malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                # malignant_weight = torch.FloatTensor([1, 1]).to(device)
                # loss_function = nn.NLLLoss(malignant_weight)
                loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break
        elif args.model_type == 'joint_swin_transformer_last_stage_cva_with_class_token':
            # 划分训练集和验证集
            # train_ratio = 0.8
            k = 5
            random.seed(444)

            train_val_dir = args.data_path
            train_val_case_names = os.listdir(train_val_dir)
            normal_image_names = []
            for name in train_val_case_names:
                if name.startswith('n'):
                    normal_image_names.append(name)
                if name.startswith('b'):
                    normal_image_names.append(name)
            for name in normal_image_names:
                train_val_case_names.remove(name)

            random.shuffle(train_val_case_names)

            # train_num = int(len(train_val_case_names) * train_ratio)
            # train_case_names = train_val_case_names[:train_num]
            # val_case_names = train_val_case_names[train_num:]

            val_num = math.ceil(len(train_val_case_names) // k)

            val_auc = []
            val_acc = []

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
                pretrain = True
                model = JointSwinTransLastStagesCVAWithClassToken(img_size=img_size, patch_size=patch_size,
                                                                  num_classes=2, window_size=window_size,
                                                                  pretrain=pretrain,
                                                                  pretrain_path=r'pre_train_pth/swin_tiny_patch4_window7_224.pth')
                model = model.to(device)

                # 定义优化器
                optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                #                       weight_decay=args.weight_decay, nesterov=args.nesterov)

                # 定义loss函数
                malignant_weight = torch.FloatTensor([1, 2.31]).to(device)
                loss_function = nn.NLLLoss(malignant_weight)
                # loss_function = nn.NLLLoss()

                # 学习率调整策略
                # if args.cosine_lr:
                #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                # else:
                #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                # 创建数据集和数据加载器
                train_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                  input_shape=(img_size, img_size),
                                                                  case_names=train_case_names, is_train=True)
                val_dataset = DDSMMVDatasetForMNmClassification(args.data_path,
                                                                input_shape=(img_size, img_size),
                                                                case_names=val_case_names, is_train=False)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=False,
                                            collate_fn=ddsm_mv_dataset_for_m_nm_classification_collate)

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

                class_names = ['not_malignant', 'malignant']

                logger.info(f"  ImageSize = {img_size}")
                logger.info(f"  PatchSize = {patch_size}")
                logger.info(f"  WinSize = {window_size}")
                logger.info(f"  PreTrain = {pretrain}")

                model, best_auc, best_acc, best_class_acc = train_mv_model(args, model, train_dataloader,
                                                                           val_dataloader, loss_function,
                                                                           optimizer, lr_scheduler, device,
                                                                           class_names, train_dataset, run_num, k)

                file_path = os.path.join(args.txt_path, args.model_type + f'_winsize={window_size}_'
                                                                          f'patchsize={patch_size}_'
                                                                          f'pretrain={pretrain}_'
                                                                          f'({img_size},{img_size})_'
                                                                          f'.txt')
                val_auc.append(best_auc)
                val_acc.append(best_acc)
                with open(file_path, 'a', encoding='utf-8') as f:
                    lines = '第{run_num}次训练：\nbest_auc: {best_auc}    best_acc: {best_acc}    nm_acc: {nm_acc}' \
                            '    m_acc: {m_acc}\n'.format(
                        run_num=run_num,
                        best_auc=best_auc,
                        best_acc=best_acc,
                        nm_acc=best_class_acc['not_malignant'],
                        m_acc=best_class_acc['malignant'])
                    f.writelines(lines)
                    logger.info('******第' + str(run_num) + '次训练结果******')
                    logger.info('Best Auc: {:.4f}'.format(best_auc))
                    logger.info('Best Acc: {:.4f}'.format(best_acc))
                    logger.info(best_class_acc)
                    if run_num == k - 1:
                        avg_auc = np.mean(val_auc)
                        avg_acc = np.mean(val_acc)
                        auc_std = np.std(val_auc)
                        acc_std = np.std(val_acc)
                        lines = '{k}次训练的平均值：\navg_auc: {avg_auc}    avg_acc: {avg_acc}    auc_std: {auc_std}' \
                                '    acc_std: {acc_std}\n'.format(
                            k=k,
                            avg_auc=avg_auc,
                            avg_acc=avg_acc,
                            auc_std=auc_std,
                            acc_std=acc_std)
                        f.writelines(lines)
                        logger.info('******总训练结果******')
                        logger.info('Avg Auc: {:.4f}'.format(avg_auc))
                        logger.info('Avg Acc: {:.4f}'.format(avg_acc))
                # break


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

    best_auc = 0

    unchanged_epoch = 0

    for epoch in range(args.epochs):
        losses.reset()
        model.train()

        with tqdm(train_dataloader, position=0) as p_bar:
            for batch_index, (images, malignant_labels) in enumerate(train_dataloader):
                data_time.update(time.time() - end)

                for view in VIEWS.LIST:
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                m_l_labels = malignant_labels['l_breast'].type(torch.LongTensor).to(device)
                m_r_labels = malignant_labels['r_breast'].type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                # CC_MLO Avg
                # l_cc_outputs = F.softmax(model(images[VIEWS.L_CC]), 1)
                # r_cc_outputs = F.softmax(model(images[VIEWS.R_CC]), 1)
                # l_mlo_outputs = F.softmax(model(images[VIEWS.L_MLO]), 1)
                # r_mlo_outputs = F.softmax(model(images[VIEWS.R_MLO]), 1)
                #
                # l_outputs = torch.log((l_cc_outputs + l_mlo_outputs) / 2)
                # r_outputs = torch.log((r_cc_outputs + r_mlo_outputs) / 2)
                # l_loss = loss_function(l_outputs, m_l_labels)
                # r_loss = loss_function(r_outputs, m_r_labels)
                # loss = (l_loss + r_loss) / 2

                l_cc_outputs = torch.log(F.softmax(model(images[VIEWS.L_CC]), 1))
                r_cc_outputs = torch.log(F.softmax(model(images[VIEWS.R_CC]), 1))
                l_mlo_outputs = torch.log(F.softmax(model(images[VIEWS.L_MLO]), 1))
                r_mlo_outputs = torch.log(F.softmax(model(images[VIEWS.R_MLO]), 1))

                l_cc_loss = loss_function(l_cc_outputs, m_l_labels)
                r_cc_loss = loss_function(r_cc_outputs, m_r_labels)
                l_mlo_loss = loss_function(l_mlo_outputs, m_l_labels)
                r_mlo_loss = loss_function(r_mlo_outputs, m_r_labels)
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
        val_loss, val_acc, val_auc, val_class_acc, val_class_all_num = evaluate_sv_model(args, model,
                                                                                         val_dataloader,
                                                                                         loss_function,
                                                                                         device, class_names,
                                                                                         run_num, k)

        if val_auc > best_auc:
            unchanged_epoch = 0
            best_auc = val_auc
            best_acc = val_acc
            best_val_class_acc = val_class_acc
            # torch.save(model.state_dict(), os.path.join(args.out_path, comment + '_best_auc_model.pt'))
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
        logger.info('Best Auc: {:.4f}'.format(best_auc))
        logger.info('Best Acc: {:.4f}'.format(best_acc))
        logger.info(best_val_class_acc)
        logger.info('Val Auc: {:.4f}'.format(val_auc))
        logger.info('Val Acc: {:.4f}'.format(val_acc))
        # logger.info('Val Class Acc:')
        logger.info(val_class_acc)
        logger.info(val_class_all_num)
        # logger.info('Train Auc: {:.4f}'.format(train_auc))
        # logger.info('Train Acc: {:.4f}'.format(train_acc))
        # # logger.info('Train Class Acc:')
        # logger.info(train_class_acc)
        # logger.info(train_class_all_num)

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
    return model, best_auc, best_acc, best_val_class_acc


def evaluate_sv_model(args, model, eval_dataloader, loss_function, device, class_names, run_num=0, k=1):
    losses = AverageMeter()
    acc = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    model.eval()

    y_trues = torch.empty(0)
    y_predictions = torch.empty(0)

    with tqdm(eval_dataloader, position=0) as p_bar:

        class_correct_num = {cls_name: 0 for cls_name in class_names}
        class_all_num = {cls_name: 0 for cls_name in class_names}

        for batch_index, (images, malignant_labels) in enumerate(eval_dataloader):
            data_time.update(time.time() - end)

            with torch.no_grad():

                for view in VIEWS.LIST:
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                m_l_labels = malignant_labels['l_breast'].type(torch.LongTensor).to(device)
                m_r_labels = malignant_labels['r_breast'].type(torch.LongTensor).to(device)

                l_cc_outputs = F.softmax(model(images[VIEWS.L_CC]), 1)
                r_cc_outputs = F.softmax(model(images[VIEWS.R_CC]), 1)
                l_mlo_outputs = F.softmax(model(images[VIEWS.L_MLO]), 1)
                r_mlo_outputs = F.softmax(model(images[VIEWS.R_MLO]), 1)

                # l_outputs = (l_cc_outputs + l_mlo_outputs) / 2
                # r_outputs = (r_cc_outputs + r_mlo_outputs) / 2

                y_trues = torch.cat([y_trues, m_l_labels.cpu().detach()])
                y_trues = torch.cat([y_trues, m_r_labels.cpu().detach()])
                y_trues = torch.cat([y_trues, m_l_labels.cpu().detach()])
                y_trues = torch.cat([y_trues, m_r_labels.cpu().detach()])

                # y_predictions = torch.cat([y_predictions, l_outputs[:, 1].cpu().detach()])
                # y_predictions = torch.cat([y_predictions, r_outputs[:, 1].cpu().detach()])
                y_predictions = torch.cat([y_predictions, l_cc_outputs[:, 1].cpu().detach()])
                y_predictions = torch.cat([y_predictions, r_cc_outputs[:, 1].cpu().detach()])
                y_predictions = torch.cat([y_predictions, l_mlo_outputs[:, 1].cpu().detach()])
                y_predictions = torch.cat([y_predictions, r_mlo_outputs[:, 1].cpu().detach()])

                # l_outputs = torch.log(l_outputs)
                # r_outputs = torch.log(r_outputs)
                # l_loss = loss_function(l_outputs, m_l_labels)
                # r_loss = loss_function(r_outputs, m_r_labels)
                # loss = (l_loss + r_loss) / 2

                l_cc_outputs = torch.log(l_cc_outputs)
                r_cc_outputs = torch.log(r_cc_outputs)
                l_mlo_outputs = torch.log(l_mlo_outputs)
                r_mlo_outputs = torch.log(r_mlo_outputs)

                l_cc_loss = loss_function(l_cc_outputs, m_l_labels)
                r_cc_loss = loss_function(r_cc_outputs, m_r_labels)
                l_mlo_loss = loss_function(l_mlo_outputs, m_l_labels)
                r_mlo_loss = loss_function(r_mlo_outputs, m_r_labels)
                loss = (l_cc_loss + r_cc_loss + l_mlo_loss + r_mlo_loss) / 4

                # m_l_acc, m_l_correct_num, m_l_all_num = accuracy(l_outputs, m_l_labels, class_names)
                # m_r_acc, m_r_correct_num, m_r_all_num = accuracy(r_outputs, m_r_labels, class_names)
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

                # for cls_name in class_names:
                #     class_correct_num[cls_name] += m_l_correct_num[cls_name]
                #     class_correct_num[cls_name] += m_r_correct_num[cls_name]
                #     class_all_num[cls_name] += m_l_all_num[cls_name]
                #     class_all_num[cls_name] += m_r_all_num[cls_name]

                losses.update(loss.item())
                # acc.update(eval_acc)
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
            class_acc[cls_name] = round(class_correct_num[cls_name] * 100.0 / class_all_num[cls_name], 2)
            total_correct_num += class_correct_num[cls_name]
            total_all_num += class_all_num[cls_name]

        total_acc = total_correct_num * 100.0 / total_all_num

    val_auc = metrics.roc_auc_score(y_trues, y_predictions, multi_class='ovo')
    return losses.avg, total_acc, val_auc, class_acc, class_all_num


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

    best_val_auc = 0
    best_val_acc = 0
    best_val_class_acc = []
    unchanged_epoch = 0

    for epoch in range(args.epochs):
        losses.reset()
        model.train()

        with tqdm(train_dataloader, position=0) as p_bar:
            for batch_index, (images, malignant_labels) in enumerate(train_dataloader):
                data_time.update(time.time() - end)

                for view in VIEWS.LIST:
                    images[view] = images[view].type(torch.FloatTensor).to(device)
                m_l_labels = malignant_labels['l_breast'].type(torch.LongTensor).to(device)
                m_r_labels = malignant_labels['r_breast'].type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                m_l_output, m_r_output = model(images)
                m_l_loss = loss_function(m_l_output, m_l_labels)
                m_r_loss = loss_function(m_r_output, m_r_labels)

                loss = (m_l_loss + m_r_loss) / 2

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
        val_loss, val_acc, val_auc, val_class_acc, val_all_num = evaluate_mv_model(args, model, val_dataloader,
                                                                                   loss_function,
                                                                                   device, class_names,
                                                                                   run_num, k)

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_val_class_acc = val_class_acc
        #     torch.save(model.state_dict(), os.path.join(args.out_path, comment + '_best_acc_model.pt'))
        if val_auc > best_val_auc:
            unchanged_epoch = 0
            best_val_auc = val_auc
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
        logger.info('Best Auc: {:.4f}'.format(best_val_auc))
        logger.info('Best Acc: {:.4f}'.format(best_val_acc))
        logger.info(best_val_class_acc)
        logger.info('Val Auc: {:.4f}'.format(val_auc))
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
    return model, best_val_auc, best_val_acc, best_val_class_acc


def evaluate_mv_model(args, model, eval_dataloader, loss_function, device, class_names, run_num=0, k=1):
    losses = AverageMeter()
    # acc = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    model.eval()

    y_trues = torch.empty(0)
    y_predictions = torch.empty(0)
    # r_y_trues = torch.empty(0)
    # r_y_predictions = torch.empty(0)

    with tqdm(eval_dataloader, position=0) as p_bar:

        class_correct_num = {cls_name: 0 for cls_name in class_names}
        class_all_num = {cls_name: 0 for cls_name in class_names}

        for batch_index, (images, malignant_labels) in enumerate(eval_dataloader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                for view in VIEWS.LIST:
                    images[view] = images[view].type(torch.FloatTensor).to(device)

                m_l_labels = malignant_labels['l_breast'].type(torch.LongTensor).to(device)
                m_r_labels = malignant_labels['r_breast'].type(torch.LongTensor).to(device)

                m_l_output, m_r_output = model(images)

                # 用于计算auc
                l_outputs_softmax = F.softmax(m_l_output, dim=1)
                y_trues = torch.cat([y_trues, m_l_labels.cpu().detach()])
                y_predictions = torch.cat([y_predictions, l_outputs_softmax[:, 1].cpu().detach()])

                r_outputs_softmax = F.softmax(m_r_output, dim=1)
                y_trues = torch.cat([y_trues, m_r_labels.cpu().detach()])
                y_predictions = torch.cat([y_predictions, r_outputs_softmax[:, 1].cpu().detach()])

                m_l_loss = loss_function(m_l_output, m_l_labels)
                m_r_loss = loss_function(m_r_output, m_r_labels)

                loss = (m_l_loss + m_r_loss) / 2
                # print(m_l_output.shape)
                # print(m_l_labels.shape)
                m_l_acc, m_l_correct_num, m_l_all_num = accuracy(m_l_output, m_l_labels, class_names)
                m_r_acc, m_r_correct_num, m_r_all_num = accuracy(m_r_output, m_r_labels, class_names)

                for cls_name in class_names:
                    class_correct_num[cls_name] += m_l_correct_num[cls_name]
                    class_correct_num[cls_name] += m_r_correct_num[cls_name]
                    class_all_num[cls_name] += m_l_all_num[cls_name]
                    class_all_num[cls_name] += m_r_all_num[cls_name]

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
            class_acc[cls_name] = round(class_correct_num[cls_name] * 100.0 / class_all_num[cls_name], 2)
            total_correct_num += class_correct_num[cls_name]
            total_all_num += class_all_num[cls_name]

        total_acc = total_correct_num * 100.0 / total_all_num
    val_auc = metrics.roc_auc_score(y_trues, y_predictions, multi_class='ovo')

    return losses.avg, total_acc, val_auc, class_acc, class_all_num


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


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


if __name__ == '__main__':
    main()
