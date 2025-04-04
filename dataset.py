import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import config

class HRDataset(Dataset):
    def __init__(self, dataset, encoder, mode='train'):
        super().__init__()
        if mode == 'train':
            self.dataset_path = config.TRAIN_IMAGES_DIR
        if mode == 'valid':
            self.dataset_path = config.VALID_IMAGES_DIR
        if mode == 'test':
            self.dataset_path = config.TEST_IMAGES_DIR

        self.data = dataset
        self.encoder = encoder
        self.mode = mode

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image_file = self.data.iloc[index]['FILENAME']
        image = cv2.imread(os.path.join(self.dataset_path, image_file),
                           cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, config.IMAGE_SIZE,
                           interpolation=cv2.INTER_AREA)

        # Data augmentation (chỉ áp dụng cho tập train)
        if self.mode == 'train':
            # Xoay ngẫu nhiên (-10 đến 10 độ)
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D(
                (image.shape[1]/2, image.shape[0]/2), angle, 1)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            # Thay đổi độ sáng ngẫu nhiên
            brightness = np.random.uniform(0.8, 1.2)
            image = image * brightness
            image = np.clip(image, 0, 255).astype(np.uint8)
            # Thêm nhiễu Gaussian
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) / 255.

        image = np.expand_dims(image, 0)

        label = self.encoder(self.data.iloc[index]['IDENTITY'])
        label = torch.LongTensor(label)  # Sửa từ torch.IntTensor thành torch.LongTensor

        return torch.FloatTensor(image), label

