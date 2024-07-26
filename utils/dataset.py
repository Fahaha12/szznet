import csv
import math
import os
import random

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from xml.dom.minidom import parse
import xml.dom.minidom
import re

class MultiModalDataset(Dataset):
    def __init__(self, root_dir, input_shape, case_names, is_train=False, augment=True, only_phenotypic=False):
        super(MultiModalDataset, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.case_names = case_names
        self.length = len(self.case_names)
        self.is_train = is_train
        self.augment = augment
        self.only_phenotypic = only_phenotypic

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        case_name = self.case_names[index]

        # 初始化表型特征（如果需要）
        phenotypes = [0, 0, 0, 0, 0, 0, 0, 0]

        # 图像和注释的路径
        image_dir = os.path.join(self.root_dir, 'JPEGImages')
        annotation_dir = os.path.join(self.root_dir, 'Annotations')
        image_path = os.path.join(image_dir, case_name + '.jpg')
        annotation_path = os.path.join(annotation_dir, case_name + '.xml')

        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = torch.from_numpy(image.transpose((2, 0, 1)))  # 转换为 CxHxW
        image = image / 255.0  # 归一化

        # 解析 XML 获取表型数据
        collection = xml.dom.minidom.parse(annotation_path).documentElement
        objects = collection.getElementsByTagName("object")
        characteristic = objects[0].getElementsByTagName("characteristics")[0]

        # 转换表型数据
        for i, label in enumerate(['Age', 'Extent_of_Resection', 'VoxelVolume', 'VoxelNum', 'Elongation', 'Flatness', 'MajorAxisLength', 'MinorAxisLength']):
            value = characteristic.getElementsByTagName(label)[0].childNodes[0].data
            phenotypes[i] = float(value) if value != 'NA' else 0.0  # 处理 'NA'，设置默认值

        # 转换目标值
        survival_days_str = characteristic.getElementsByTagName('Survival_days')[0].childNodes[0].data

        # 从字符串中提取数值，处理各种情况
        match = re.search(r'\d+', survival_days_str)
        target = float(match.group()) if match else 0.0  # 若未找到数字，则默认值为 0.0

        # 可选地转换为 PyTorch 张量
        phenotypes = torch.tensor(phenotypes, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return image, phenotypes, target