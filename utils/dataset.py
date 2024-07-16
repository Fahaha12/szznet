import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import pandas as pd
import re

class MultiModalDataset(Dataset):
    def __init__(self, root_dir, input_shape, case_names, csv_file, is_train=False, augment=True, only_phenotypic=False):
        super(MultiModalDataset, self).__init__()

        self.root_dir = root_dir
        self.input_shape = input_shape
        self.case_names = case_names
        self.length = len(self.case_names)
        self.is_train = is_train
        self.augment = augment
        self.only_phenotypic = only_phenotypic

        # Load CSV file
        self.data_df = pd.read_csv(csv_file)
        self.data_df.set_index('Brats20ID', inplace=True)

    def __len__(self):
        return self.length

    def load_and_preprocess_image(self, file_path):
        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))  # Normalize
        img_data = np.expand_dims(img_data, axis=0)  # Add channel dimension
        img_data = np.array(img_data, dtype=np.float32)
        return img_data

    def __getitem__(self, index):
        case_name = self.case_names[index]

        # Initialize phenotype features (if needed)
        phenotypes = [0] * 8

        # Paths for images and segmentation
        case_dir = os.path.join(self.root_dir, case_name)
        t1_path = os.path.join(case_dir, case_name + '_t1.nii.gz')
        t1ce_path = os.path.join(case_dir, case_name + '_t1ce.nii.gz')
        t2_path = os.path.join(case_dir, case_name + '_t2.nii.gz')
        flair_path = os.path.join(case_dir, case_name + '_flair.nii.gz')
        seg_path = os.path.join(case_dir, case_name + '_seg.nii.gz')

        # Load and preprocess images
        t1 = self.load_and_preprocess_image(t1_path)
        t1ce = self.load_and_preprocess_image(t1ce_path)
        t2 = self.load_and_preprocess_image(t2_path)
        flair = self.load_and_preprocess_image(flair_path)
        segmentation = self.load_and_preprocess_image(seg_path)

        # Combine the modalities into a single tensor
        image = np.concatenate([t1, t1ce, t2, flair, segmentation], axis=0)  # Combine along channel dimension

        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32)

        # Optionally apply data augmentations here
        if self.augment and self.is_train:
            # Apply data augmentation
            image = self.apply_augmentations(image)

        # Retrieve phenotype data from CSV
        row = self.data_df.loc[case_name]
        phenotypes = [
            row.get('Age', 0.0),  # Age
            row.get('Extent_of_Resection', 0.0),  # Extent_of_Resection (can be encoded if needed)
            row.get('calcification', 0.0),
            row.get('sphericity', 0.0),
            row.get('margin', 0.0),
            row.get('lobulation', 0.0),
            row.get('spiculation', 0.0),
            row.get('texture', 0.0)
        ]

        # Convert target value
        survival_days = row.get('Survival_days', 0.0)

        # Convert phenotypes and target to PyTorch tensors
        phenotypes = torch.tensor(phenotypes, dtype=torch.float32)
        target = torch.tensor(float(survival_days), dtype=torch.float32)

        return image, phenotypes, target

    def apply_augmentations(self, image):
        # Define any augmentation transformations if needed
        # For example, random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
        return image