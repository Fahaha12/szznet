import csv
import math
import os
import random

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from utils.constants import VIEWS
from utils import data_augment
from xml.dom.minidom import parse
import xml.dom.minidom
import re

class DDSMSVDatasetForMNmClassification(Dataset):
    def __init__(self, root_dir, input_shape, image_names, is_train=False):
        super(DDSMSVDatasetForMNmClassification, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.image_names = image_names
        self.length = len(self.image_names)
        # self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_name = self.image_names[index]
        annotation_name = image_name[:-4] + '.txt'

        # 样例名， 图像类型（l_cc, r_cc, l_mlo, r_mlo），样例文件夹路径， 图像文件夹路径， 标注文件夹路径
        case_name = image_name.split('.')[0]
        case_dir = os.path.join(self.root_dir, case_name)
        image_dir = os.path.join(case_dir, 'Images')
        annotation_dir = os.path.join(case_dir, 'Annotations')
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, annotation_name)
        annotation_names = os.listdir(annotation_dir)

        # 读取图像并进行一些处理
        image = Image.open(image_path)
        image = np.array(image)

        # 读取目标框信息
        resized_annotation_name = image_name[0:-4] + '_resized.txt'
        resized_annotation_path = os.path.join(annotation_dir, resized_annotation_name)
        box = np.zeros((0, 0))
        if resized_annotation_name in annotation_names:
            with open(resized_annotation_path) as f:
                lines = f.readlines()
                object_num = len(lines)
                box = np.zeros((object_num, 5))
                for index, line in enumerate(lines):
                    line = line.replace('\n', '')
                    num_list = line.split(' ')
                    box[index, 0] = int(num_list[0])
                    box[index, 1] = int(num_list[1])
                    box[index, 2] = int(num_list[2])
                    box[index, 3] = int(num_list[3])
                    box[index, 4] = int(num_list[4])

        # 从文件中读取标签信息
        malignant_label = 0
        if annotation_name in annotation_names:
            with open(annotation_path) as f:
                lines = f.readlines()
            for line in lines:
                if 'PATHOLOGY' in line:
                    line = line.replace('\n', '')
                    image_class = line.split(' ')[1].lower()
                    if image_class == 'malignant':
                        malignant_label = 1

        # 对图像进行数据增强并resize

        # if self.is_train and malignant_label == 1:
        # image, box = data_augment.RandomHorizontalFilp()(np.copy(image), np.copy(box))
        # image, box = data_augment.RandomCrop()(np.copy(image), np.copy(box))
        # image, box = data_augment.RandomAffine()(np.copy(image), np.copy(box))

        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        # image, box, image_w, image_h = data_augment.Resize(self.input_shape, True)(np.copy(image), np.copy(box))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image / 255.0

        return image, malignant_label


class DDSMMVDatasetForMNmClassification(Dataset):
    def __init__(self, root_dir, input_shape, case_names, is_train=False, augment=True):
        super(DDSMMVDatasetForMNmClassification, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.case_names = case_names
        self.length = len(self.case_names)
        self.is_train = is_train
        self.augment = augment

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        case_name = self.case_names[index]

        images = {view: [] for view in VIEWS.LIST}
        malignant_labels = {view: [] for view in VIEWS.LIST}

        case_dir = os.path.join(self.root_dir, case_name)
        image_dir = os.path.join(case_dir, 'Images')
        annotation_dir = os.path.join(case_dir, 'Annotations')

        image_names = os.listdir(image_dir)
        annotation_names = os.listdir(annotation_dir)

        for image_name in image_names:

            image_path = os.path.join(image_dir, image_name)
            # 读取图像并进行一些处理
            image = Image.open(image_path)
            # image = image.resize((self.input_shape[1], self.input_shape[0]))
            image = np.array(image)

            # 读取目标框信息
            resized_annotation_name = image_name[0:-4] + '_resized.txt'
            resized_annotation_path = os.path.join(annotation_dir, resized_annotation_name)
            box = np.zeros((0, 0))
            if resized_annotation_name in annotation_names:
                with open(resized_annotation_path) as f:
                    lines = f.readlines()
                    object_num = len(lines)
                    box = np.zeros((object_num, 5))
                    for index, line in enumerate(lines):
                        line = line.replace('\n', '')
                        num_list = line.split(' ')
                        box[index, 0] = int(num_list[0])
                        box[index, 1] = int(num_list[1])
                        box[index, 2] = int(num_list[2])
                        box[index, 3] = int(num_list[3])
                        box[index, 4] = int(num_list[4])

            # 对数据进行增强并resize
            if self.is_train and self.augment:
                image, box = data_augment.RandomCrop()(np.copy(image), np.copy(box))
                image, box = data_augment.RandomAffine()(np.copy(image), np.copy(box))

            image_type = image_name.split('.')[1]
            if image_type == 'RIGHT_CC' or image_type == 'RIGHT_MLO':
                image = cv2.flip(image, 1)
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            image = image / 255.0

            # 读取标签
            malignant_label = 0
            annotation_name = image_name[:-4] + '.txt'
            annotation_path = os.path.join(annotation_dir, annotation_name)
            if annotation_name in annotation_names:
                with open(annotation_path) as f:
                    lines = f.readlines()
                # 从文件中读取标签信息
                for line in lines:
                    if 'PATHOLOGY' in line:
                        line = line.replace('\n', '')
                        image_class = line.split(' ')[1].lower()
                        if image_class == 'malignant':
                            malignant_label = 1

            # 添加到数据字典中
            image_type = image_name.split('.')[1]
            image_key = ''
            if image_type == 'LEFT_CC':
                image_key = VIEWS.L_CC
            elif image_type == 'LEFT_MLO':
                image_key = VIEWS.L_MLO
            elif image_type == 'RIGHT_CC':
                image_key = VIEWS.R_CC
            elif image_type == 'RIGHT_MLO':
                image_key = VIEWS.R_MLO

            images[image_key] = image
            malignant_labels[image_key] = malignant_label

        malignant_labels['l_breast'] = self.label_or(malignant_labels[VIEWS.L_CC], malignant_labels[VIEWS.L_MLO])
        malignant_labels['r_breast'] = self.label_or(malignant_labels[VIEWS.R_CC], malignant_labels[VIEWS.R_MLO])

        return images, malignant_labels

    def label_or(self, label_a, label_b):
        if label_a == 0 and label_b == 0:
            label = 0
        else:
            label = 1
        return label


# 多视图恶性/非恶性分类时使用
def ddsm_mv_dataset_for_m_nm_classification_collate(batch):
    images_batch = {view: [] for view in VIEWS.LIST}
    malignant_labels_batch = {view: [] for view in VIEWS.LIST}
    malignant_labels_batch['l_breast'] = []
    malignant_labels_batch['r_breast'] = []
    for images, malignant_labels in batch:
        for view in VIEWS.LIST:
            images_batch[view].append(images[view].unsqueeze(0))
        malignant_labels_batch['l_breast'].append(malignant_labels['l_breast'])
        malignant_labels_batch['r_breast'].append(malignant_labels['r_breast'])

    for view in VIEWS.LIST:
        images_batch[view] = torch.cat(images_batch[view])
    malignant_labels_batch['l_breast'] = torch.from_numpy(np.array(malignant_labels_batch['l_breast']))
    malignant_labels_batch['r_breast'] = torch.from_numpy(np.array(malignant_labels_batch['r_breast']))

    return images_batch, malignant_labels_batch


# 多视图BIRADS分级时使用(无表型信息)
class DDSMMVDatasetForBIRADSClassification(Dataset):
    def __init__(self, root_dir, input_shape, case_names, is_train=False, augment=True):
        super(DDSMMVDatasetForBIRADSClassification, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.case_names = case_names
        self.length = len(self.case_names)
        self.is_train = is_train
        self.augment = augment

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        case_name = self.case_names[index]

        images = {view: [] for view in VIEWS.LIST}
        labels = {view: [] for view in VIEWS.LIST}

        case_dir = os.path.join(self.root_dir, case_name)
        image_dir = os.path.join(case_dir, 'Images')
        annotation_dir = os.path.join(case_dir, 'Annotations')

        image_names = os.listdir(image_dir)
        annotation_names = os.listdir(annotation_dir)

        for image_name in image_names:

            image_path = os.path.join(image_dir, image_name)
            # 读取图像并进行一些处理
            image = Image.open(image_path)
            # image = image.resize((self.input_shape[1], self.input_shape[0]))
            image = np.array(image)

            # 读取目标框信息
            resized_annotation_name = image_name[0:-4] + '_resized.txt'
            resized_annotation_path = os.path.join(annotation_dir, resized_annotation_name)
            box = np.zeros((0, 0))
            if resized_annotation_name in annotation_names:
                with open(resized_annotation_path) as f:
                    lines = f.readlines()
                    object_num = len(lines)
                    box = np.zeros((object_num, 5))
                    for index, line in enumerate(lines):
                        line = line.replace('\n', '')
                        num_list = line.split(' ')
                        box[index, 0] = int(num_list[0])
                        box[index, 1] = int(num_list[1])
                        box[index, 2] = int(num_list[2])
                        box[index, 3] = int(num_list[3])
                        box[index, 4] = int(num_list[4])

            # 对数据进行增强并resize
            if self.is_train and self.augment:
                image, box = data_augment.RandomCrop()(np.copy(image), np.copy(box))
                image, box = data_augment.RandomAffine()(np.copy(image), np.copy(box))

            image_type = image_name.split('.')[1]
            if image_type == 'RIGHT_CC' or image_type == 'RIGHT_MLO':
                image = cv2.flip(image, 1)
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            image = image / 255.0

            # 读取标签
            label = 0
            annotation_name = image_name[:-4] + '.txt'
            annotation_path = os.path.join(annotation_dir, annotation_name)
            if annotation_name in annotation_names:
                with open(annotation_path) as f:
                    lines = f.readlines()
                # 从文件中读取标签信息
                for line in lines:
                    if 'ASSESSMENT' in line:
                        line = line.replace('\n', '')
                        # print(line.split(' '))
                        image_class = int(line.split(' ')[1]) - 1
                        if image_class > label:
                            # 取最高等级作为label
                            label = image_class

            # 添加到数据字典中
            image_type = image_name.split('.')[1]
            image_key = ''
            if image_type == 'LEFT_CC':
                image_key = VIEWS.L_CC
            elif image_type == 'LEFT_MLO':
                image_key = VIEWS.L_MLO
            elif image_type == 'RIGHT_CC':
                image_key = VIEWS.R_CC
            elif image_type == 'RIGHT_MLO':
                image_key = VIEWS.R_MLO

            images[image_key] = image
            labels[image_key] = label

        labels['l_breast'] = self.label_max(labels[VIEWS.L_CC], labels[VIEWS.L_MLO])
        labels['r_breast'] = self.label_max(labels[VIEWS.R_CC], labels[VIEWS.R_MLO])
        labels['case'] = self.label_max(labels['l_breast'], labels['r_breast'])

        return images, labels

    def label_max(self, label_a, label_b):
        if label_a > label_b:
            label = label_a
        else:
            label = label_b
        return label


# 多视图BIRADS分级时使用(无表型信息)
def ddsm_mv_dataset_for_birads_classification_collate(batch):
    images_batch = {view: [] for view in VIEWS.LIST}
    labels_batch = {view: [] for view in VIEWS.LIST}
    labels_batch['l_breast'] = []
    labels_batch['r_breast'] = []
    labels_batch['case'] = []
    for images, labels in batch:
        for view in VIEWS.LIST:
            images_batch[view].append(images[view].unsqueeze(0))
        labels_batch['l_breast'].append(labels['l_breast'])
        labels_batch['r_breast'].append(labels['r_breast'])
        labels_batch['case'].append(labels['case'])

    for view in VIEWS.LIST:
        images_batch[view] = torch.cat(images_batch[view])
    labels_batch['l_breast'] = torch.from_numpy(np.array(labels_batch['l_breast']))
    labels_batch['r_breast'] = torch.from_numpy(np.array(labels_batch['r_breast']))
    labels_batch['case'] = torch.from_numpy(np.array(labels_batch['case']))

    return images_batch, labels_batch


# 多视图BIRADS分级时使用(有表型信息)
class DDSMMVDatasetForBIRADSClassificationWithPhenotypic(Dataset):
    def __init__(self, root_dir, input_shape, case_names, is_train=False, augment=True, only_phenotypic=False):
        super(DDSMMVDatasetForBIRADSClassificationWithPhenotypic, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.case_names = case_names
        self.length = len(self.case_names)
        self.is_train = is_train
        self.augment = augment
        self.only_phenotypic = only_phenotypic
        self.phenotypic_shape = ['OVAL', 'LOBULATED', 'ROUND', 'IRREGULAR', 'ARCHITECTURAL_DISTORTION']
        self.phenotypic_margin = ['ILL_DEFINED', 'MICROLOBULATED', 'CIRCUMSCRIBED', 'OBSCURED', 'SPICULATED']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        case_name = self.case_names[index]

        images = {view: [] for view in VIEWS.LIST}
        phenotypes = {view: [0, 0, 0] for view in VIEWS.LIST}
        phenotypes['case'] = [0, 0]
        labels = {view: [] for view in VIEWS.LIST}

        case_dir = os.path.join(self.root_dir, case_name)
        image_dir = os.path.join(case_dir, 'Images')
        annotation_dir = os.path.join(case_dir, 'Annotations')

        image_names = os.listdir(image_dir)
        annotation_names = os.listdir(annotation_dir)

        for image_name in image_names:

            image_type = image_name.split('.')[1]
            image_key = ''
            if image_type == 'LEFT_CC':
                image_key = VIEWS.L_CC
            elif image_type == 'LEFT_MLO':
                image_key = VIEWS.L_MLO
            elif image_type == 'RIGHT_CC':
                image_key = VIEWS.R_CC
            elif image_type == 'RIGHT_MLO':
                image_key = VIEWS.R_MLO

            image = torch.rand(1, 1, 1)
            if not self.only_phenotypic:
                image_path = os.path.join(image_dir, image_name)
                # 读取图像并进行一些处理
                image = Image.open(image_path)
                # image = image.resize((self.input_shape[1], self.input_shape[0]))
                image = np.array(image)

                # 读取目标框信息
                resized_annotation_name = image_name[0:-4] + '_resized.txt'
                resized_annotation_path = os.path.join(annotation_dir, resized_annotation_name)
                box = np.zeros((0, 0))
                if resized_annotation_name in annotation_names:
                    with open(resized_annotation_path) as f:
                        lines = f.readlines()
                        object_num = len(lines)
                        box = np.zeros((object_num, 5))
                        for index, line in enumerate(lines):
                            line = line.replace('\n', '')
                            num_list = line.split(' ')
                            box[index, 0] = int(num_list[0])
                            box[index, 1] = int(num_list[1])
                            box[index, 2] = int(num_list[2])
                            box[index, 3] = int(num_list[3])
                            box[index, 4] = int(num_list[4])

                # 对数据进行增强并resize
                if self.is_train and self.augment:
                    image, box = data_augment.RandomCrop()(np.copy(image), np.copy(box))
                    image, box = data_augment.RandomAffine()(np.copy(image), np.copy(box))

                image_type = image_name.split('.')[1]
                if image_type == 'RIGHT_CC' or image_type == 'RIGHT_MLO':
                    image = cv2.flip(image, 1)
                image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                image = torch.from_numpy(image.transpose((2, 0, 1)))
                image = image / 255.0

            # 读取标签和image层面表型信息
            label = 0
            annotation_name = image_name[:-4] + '.txt'
            annotation_path = os.path.join(annotation_dir, annotation_name)
            if annotation_name in annotation_names:
                with open(annotation_path) as f:
                    lines = f.readlines()
                # 从文件中读取标签信息
                for line in lines:
                    if 'ASSESSMENT' in line:
                        line = line.replace('\n', '')
                        # print(line.split(' '))
                        image_class = int(line.split(' ')[1]) - 1
                        if image_class > label:
                            # 取最高等级作为label
                            label = image_class
                    if 'SHAPE' in line and 'MARGINS' in line:
                        line_split = line.replace('\n', '').split(' ')
                        shape = line_split[3]
                        margins = line_split[5]
                        if shape in self.phenotypic_shape:
                            phenotypes[image_key][0] = self.phenotypic_shape.index(shape) + 1
                        else:
                            phenotypes[image_key][0] = -1
                        if margins in self.phenotypic_margin:
                            phenotypes[image_key][1] = self.phenotypic_margin.index(margins) + 1
                        else:
                            phenotypes[image_key][1] = -1
                    if 'SUBTLETY' in line:
                        line_split = line.replace('\n', '').split(' ')
                        subtlety = line_split[1]
                        phenotypes[image_key][2] = int(subtlety)

            # 添加到数据字典中
            images[image_key] = image
            labels[image_key] = label

        # 读取case层面表型信息
        annotation_name = case_name + '.ics.txt'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        if annotation_name in annotation_names:
            with open(annotation_path) as f:
                lines = f.readlines()
                for line in lines:
                    if 'AGE' in line:
                        age = line.replace('\n', '').split(' ')[1]
                        phenotypes['case'][0] = int(age)
                    if 'DENSITY' in line:
                        density = line.replace('\n', '').split(' ')[1]
                        phenotypes['case'][1] = int(density)

        labels['l_breast'] = self.label_max(labels[VIEWS.L_CC], labels[VIEWS.L_MLO])
        labels['r_breast'] = self.label_max(labels[VIEWS.R_CC], labels[VIEWS.R_MLO])
        labels['case'] = self.label_max(labels['l_breast'], labels['r_breast'])

        return images, phenotypes, labels

    def label_max(self, label_a, label_b):
        if label_a > label_b:
            label = label_a
        else:
            label = label_b
        return label


# 多视图半监督BIRADS分级时使用(有表型信息)
class DDSMMVDatasetForSemiSupervisedBIRADSClassificationWithPhenotypic(Dataset):
    def __init__(self, root_dir, input_shape, case_names, labeled_case_names, is_train=False, augment=True, only_phenotypic=False):
        super(DDSMMVDatasetForSemiSupervisedBIRADSClassificationWithPhenotypic, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.case_names = case_names
        self.labeled_case_names = labeled_case_names
        self.length = len(self.case_names)
        self.is_train = is_train
        self.augment = augment
        self.only_phenotypic = only_phenotypic
        self.phenotypic_shape = ['OVAL', 'LOBULATED', 'ROUND', 'IRREGULAR', 'ARCHITECTURAL_DISTORTION']
        self.phenotypic_margin = ['ILL_DEFINED', 'MICROLOBULATED', 'CIRCUMSCRIBED', 'OBSCURED', 'SPICULATED']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        case_name = self.case_names[index]

        images = {view: [] for view in VIEWS.LIST}
        phenotypes = {view: [0, 0, 0] for view in VIEWS.LIST}
        phenotypes['case'] = [0, 0]
        labels = {view: [] for view in VIEWS.LIST}

        case_dir = os.path.join(self.root_dir, case_name)
        image_dir = os.path.join(case_dir, 'Images')
        annotation_dir = os.path.join(case_dir, 'Annotations')

        image_names = os.listdir(image_dir)
        annotation_names = os.listdir(annotation_dir)

        for image_name in image_names:

            image_type = image_name.split('.')[1]
            image_key = ''
            if image_type == 'LEFT_CC':
                image_key = VIEWS.L_CC
            elif image_type == 'LEFT_MLO':
                image_key = VIEWS.L_MLO
            elif image_type == 'RIGHT_CC':
                image_key = VIEWS.R_CC
            elif image_type == 'RIGHT_MLO':
                image_key = VIEWS.R_MLO

            image = torch.rand(1, 1, 1)
            if not self.only_phenotypic:
                image_path = os.path.join(image_dir, image_name)
                # 读取图像并进行一些处理
                image = Image.open(image_path)
                # image = image.resize((self.input_shape[1], self.input_shape[0]))
                image = np.array(image)

                # 读取目标框信息
                resized_annotation_name = image_name[0:-4] + '_resized.txt'
                resized_annotation_path = os.path.join(annotation_dir, resized_annotation_name)
                box = np.zeros((0, 0))
                if resized_annotation_name in annotation_names:
                    with open(resized_annotation_path) as f:
                        lines = f.readlines()
                        object_num = len(lines)
                        box = np.zeros((object_num, 5))
                        for index, line in enumerate(lines):
                            line = line.replace('\n', '')
                            num_list = line.split(' ')
                            box[index, 0] = int(num_list[0])
                            box[index, 1] = int(num_list[1])
                            box[index, 2] = int(num_list[2])
                            box[index, 3] = int(num_list[3])
                            box[index, 4] = int(num_list[4])

                # 对数据进行增强并resize
                if self.is_train and self.augment:
                    image, box = data_augment.RandomCrop()(np.copy(image), np.copy(box))
                    image, box = data_augment.RandomAffine()(np.copy(image), np.copy(box))

                image_type = image_name.split('.')[1]
                if image_type == 'RIGHT_CC' or image_type == 'RIGHT_MLO':
                    image = cv2.flip(image, 1)
                image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                image = torch.from_numpy(image.transpose((2, 0, 1)))
                image = image / 255.0

            # 读取标签和image层面表型信息
            label = 0
            annotation_name = image_name[:-4] + '.txt'
            annotation_path = os.path.join(annotation_dir, annotation_name)
            if annotation_name in annotation_names:
                with open(annotation_path) as f:
                    lines = f.readlines()
                # 从文件中读取标签信息
                for line in lines:
                    if 'ASSESSMENT' in line:
                        line = line.replace('\n', '')
                        # print(line.split(' '))
                        image_class = int(line.split(' ')[1]) - 1
                        if image_class > label:
                            # 取最高等级作为label
                            label = image_class
                    if 'SHAPE' in line and 'MARGINS' in line:
                        line_split = line.replace('\n', '').split(' ')
                        shape = line_split[3]
                        margins = line_split[5]
                        if shape in self.phenotypic_shape:
                            phenotypes[image_key][0] = self.phenotypic_shape.index(shape) + 1
                        else:
                            phenotypes[image_key][0] = -1
                        if margins in self.phenotypic_margin:
                            phenotypes[image_key][1] = self.phenotypic_margin.index(margins) + 1
                        else:
                            phenotypes[image_key][1] = -1
                    if 'SUBTLETY' in line:
                        line_split = line.replace('\n', '').split(' ')
                        subtlety = line_split[1]
                        phenotypes[image_key][2] = int(subtlety)

            # 添加到数据字典中
            images[image_key] = image
            labels[image_key] = label

        # 读取case层面表型信息
        annotation_name = case_name + '.ics.txt'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        if annotation_name in annotation_names:
            with open(annotation_path) as f:
                lines = f.readlines()
                for line in lines:
                    if 'AGE' in line:
                        age = line.replace('\n', '').split(' ')[1]
                        phenotypes['case'][0] = int(age)
                    if 'DENSITY' in line:
                        density = line.replace('\n', '').split(' ')[1]
                        phenotypes['case'][1] = int(density)

        if case_name in self.labeled_case_names:
            labels['l_breast'] = self.label_max(labels[VIEWS.L_CC], labels[VIEWS.L_MLO])
            labels['r_breast'] = self.label_max(labels[VIEWS.R_CC], labels[VIEWS.R_MLO])
            labels['case'] = self.label_max(labels['l_breast'], labels['r_breast'])
        else:
            labels['l_breast'] = -1
            labels['r_breast'] = -1
            labels['case'] = -1

        return images, phenotypes, labels

    def label_max(self, label_a, label_b):
        if label_a > label_b:
            label = label_a
        else:
            label = label_b
        return label


# 多视图BIRADS分级时使用(有表型信息)
def ddsm_mv_dataset_for_birads_classification_with_phenotypic_collate(batch):
    images_batch = {view: [] for view in VIEWS.LIST}
    phenotypes_batch = {view: [] for view in VIEWS.LIST}
    phenotypes_batch['case'] = []
    labels_batch = {view: [] for view in VIEWS.LIST}
    labels_batch['l_breast'] = []
    labels_batch['r_breast'] = []
    labels_batch['case'] = []
    for images, phenotypes, labels in batch:
        for view in VIEWS.LIST:
            images_batch[view].append(images[view].unsqueeze(0))
            phenotypes_batch[view].append(torch.from_numpy(np.array(phenotypes[view])).unsqueeze(0))
        phenotypes_batch['case'].append(phenotypes['case'])
        labels_batch['l_breast'].append(labels['l_breast'])
        labels_batch['r_breast'].append(labels['r_breast'])
        labels_batch['case'].append(labels['case'])

    for view in VIEWS.LIST:
        images_batch[view] = torch.cat(images_batch[view])
        phenotypes_batch[view] = torch.cat(phenotypes_batch[view])
    phenotypes_batch['case'] = torch.from_numpy(np.array(phenotypes_batch['case']))
    labels_batch['l_breast'] = torch.from_numpy(np.array(labels_batch['l_breast']))
    labels_batch['r_breast'] = torch.from_numpy(np.array(labels_batch['r_breast']))
    labels_batch['case'] = torch.from_numpy(np.array(labels_batch['case']))

    return images_batch, phenotypes_batch, labels_batch

class LungDatasetForRegression(Dataset):
    def __init__(self, root_dir, input_shape, case_names, is_train=False, augment=True, only_phenotypic=False):
        super(LungDatasetForRegression, self).__init__()

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

        # Initialize phenotype features (if needed)
        phenotypes = [0, 0, 0, 0, 0, 0, 0, 0]

        # Paths for image and annotation
        image_dir = os.path.join(self.root_dir, 'JPEGImages')
        annotation_dir = os.path.join(self.root_dir, 'Annotations')
        image_path = os.path.join(image_dir, case_name + '.jpg')
        annotation_path = os.path.join(annotation_dir, case_name + '.xml')

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = torch.from_numpy(image.transpose((2, 0, 1)))  # Convert to CxHxW
        image = image / 255.0  # Normalize

        # Parse XML for phenotypic data
        collection = xml.dom.minidom.parse(annotation_path).documentElement
        objects = collection.getElementsByTagName("object")
        characteristic = objects[0].getElementsByTagName("characteristics")[0]

        # Convert phenotypic data
        for i, label in enumerate(['Age', 'Extent_of_Resection', 'VoxelVolume', 'VoxelNum', 'Elongation', 'Flatness', 'MajorAxisLength', 'MinorAxisLength']):
            value = characteristic.getElementsByTagName(label)[0].childNodes[0].data
            phenotypes[i] = float(value) if value != 'NA' else 0.0  # Handle 'NA' by setting default value

        # Convert target value
        survival_days_str = characteristic.getElementsByTagName('Survival_days')[0].childNodes[0].data

        # Extract numeric value from string, handle various cases
        match = re.search(r'\d+', survival_days_str)
        target = float(match.group()) if match else 0.0  # Default to 0.0 if no number found

        # Optionally convert to PyTorch tensors
        phenotypes = torch.tensor(phenotypes, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return image, phenotypes, target

# lung分类时使用(有表型信息)
class LungDatasetForClassificationWithPhenotypic(Dataset):
    def __init__(self, root_dir, input_shape, case_names, is_train=False, augment=True, only_phenotypic=False):
        super(LungDatasetForClassificationWithPhenotypic, self).__init__()

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

        phenotypes = [0, 0, 0, 0, 0, 0, 0, 0]

        image_dir = os.path.join(self.root_dir, 'JPEGImages')
        annotation_dir = os.path.join(self.root_dir, 'Annotations')
        image_path = os.path.join(image_dir, case_name + '.jpg')
        annotation_path = os.path.join(annotation_dir, case_name + '.xml')

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image / 255.0

        collection = xml.dom.minidom.parse(annotation_path).documentElement
        objects = collection.getElementsByTagName("object")
        characteristic = objects[0].getElementsByTagName("characteristics")[0]

        phenotypes[0] = int(characteristic.getElementsByTagName('Age')[0].childNodes[0].data)
        phenotypes[1] = int(characteristic.getElementsByTagName('Extent_of_Resection')[0].childNodes[0].data)
        phenotypes[2] = int(characteristic.getElementsByTagName('calcification')[0].childNodes[0].data)
        phenotypes[3] = int(characteristic.getElementsByTagName('sphericity')[0].childNodes[0].data)
        phenotypes[4] = int(characteristic.getElementsByTagName('margin')[0].childNodes[0].data)
        phenotypes[5] = int(characteristic.getElementsByTagName('lobulation')[0].childNodes[0].data)
        phenotypes[6] = int(characteristic.getElementsByTagName('spiculation')[0].childNodes[0].data)
        phenotypes[7] = int(characteristic.getElementsByTagName('texture')[0].childNodes[0].data)
        label = int(characteristic.getElementsByTagName('Survival_days')[0].childNodes[0].data) - 1

        return image, phenotypes, label


# lung半监督分类时使用(有表型信息)
class LungDatasetForSemiSupervisedClassificationWithPhenotypic(Dataset):
    def __init__(self, root_dir, input_shape, case_names, labeled_case_names, is_train=False, augment=True, only_phenotypic=False):
        super(LungDatasetForSemiSupervisedClassificationWithPhenotypic, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.case_names = case_names
        self.labeled_case_names = labeled_case_names
        self.length = len(self.case_names)
        self.is_train = is_train
        self.augment = augment
        self.only_phenotypic = only_phenotypic

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        case_name = self.case_names[index]

        phenotypes = [0, 0, 0, 0, 0, 0, 0, 0]

        image_dir = os.path.join(self.root_dir, 'JPEGImages')
        annotation_dir = os.path.join(self.root_dir, 'Annotations')
        image_path = os.path.join(image_dir, case_name + '.jpg')
        annotation_path = os.path.join(annotation_dir, case_name + '.xml')

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image / 255.0

        collection = xml.dom.minidom.parse(annotation_path).documentElement
        objects = collection.getElementsByTagName("object")
        characteristic = objects[0].getElementsByTagName("characteristics")[0]

        phenotypes[0] = int(characteristic.getElementsByTagName('subtlety')[0].childNodes[0].data)
        phenotypes[1] = int(characteristic.getElementsByTagName('internalStructure')[0].childNodes[0].data)
        phenotypes[2] = int(characteristic.getElementsByTagName('calcification')[0].childNodes[0].data)
        phenotypes[3] = int(characteristic.getElementsByTagName('sphericity')[0].childNodes[0].data)
        phenotypes[4] = int(characteristic.getElementsByTagName('margin')[0].childNodes[0].data)
        phenotypes[5] = int(characteristic.getElementsByTagName('lobulation')[0].childNodes[0].data)
        phenotypes[6] = int(characteristic.getElementsByTagName('spiculation')[0].childNodes[0].data)
        phenotypes[7] = int(characteristic.getElementsByTagName('texture')[0].childNodes[0].data)
        if case_name in self.labeled_case_names:
            label = int(characteristic.getElementsByTagName('malignancy')[0].childNodes[0].data) - 1
        else:
            label = -1

        return image, phenotypes, label


# lung分类时使用(有表型信息)
def lung_classification_with_phenotypic_collate(batch):
    images_batch = []
    phenotypes_batch = []
    labels_batch = []
    for images, phenotypes, labels in batch:
        images_batch.append(images.unsqueeze(0))
        phenotypes_batch.append(torch.from_numpy(np.array(phenotypes)).unsqueeze(0))
        labels_batch.append(labels)

    images_batch = torch.cat(images_batch)
    phenotypes_batch = torch.cat(phenotypes_batch)
    labels_batch = torch.from_numpy(np.array(labels_batch))

    return images_batch, phenotypes_batch, labels_batch
