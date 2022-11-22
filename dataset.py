import os
import config
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


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

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image_file = self.data.iloc[index]['FILENAME']
        image = cv2.imread(os.path.join(self.dataset_path, image_file),
                           cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, config.IMAGE_SIZE,
                           interpolation=cv2.INTER_AREA)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) / 255.

        image = np.expand_dims(image, 0)

        label = self.data.iloc[index]['IDENTITY']
        label = self.encoder(label)

        image = torch.as_tensor(image, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int32)

        return image, label
