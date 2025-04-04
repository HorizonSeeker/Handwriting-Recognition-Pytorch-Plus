import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3  # Tăng từ 1e-4 lên 1e-3 để học nhanh hơn
TRAIN_IMAGES_DIR = 'dataset/train_v2/train'
VALID_IMAGES_DIR = 'dataset/validation_v2/validation'
TEST_IMAGES_DIR = 'dataset/test_v2/test'
TRAIN_CSV = 'dataset/written_name_train_v2.csv'
VALID_CSV = 'dataset/written_name_validation_v2.csv'
TEST_CSV = 'dataset/written_name_test_v2.csv'
MAX_LENGTH = 20
VOCAB_SIZE = 30
IMAGE_SIZE = [300, 100]