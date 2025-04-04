import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby  # Thêm import này

import config
import utils
from model import CRNN

# Đường dẫn
image_folder = "my_images"
model_path = "checkpoint.pth"  # Có thể thay bằng "best_checkpoint.pth" hoặc "trained_model.pth"

# Tiền xử lý hình ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    image = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) / 255.0
    image = np.expand_dims(image, 0)  # Thêm channel dimension
    image = torch.as_tensor(image, dtype=torch.float32)
    return image

# Load mô hình
device = config.DEVICE
input_size = 64
hidden_size = 128
output_size = config.VOCAB_SIZE + 1
num_layers = 2

model = CRNN(input_size, hidden_size, output_size, num_layers).to(device)

# Load từ checkpoint hoặc file mô hình
try:
    if "checkpoint" in model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from checkpoint: {model_path}")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from: {model_path}")
except Exception as e:
    print(f"Không thể load mô hình từ {model_path}: {e}")
    exit(1)

model.eval()

# Từ vựng để giải mã
int2char, _ = utils.get_vocabulary()

# Dự đoán trên từng hình ảnh
for image_name in os.listdir(image_folder):
    # Chỉ xử lý các file ảnh
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Bỏ qua file không phải ảnh: {image_name}")
        continue

    image_path = os.path.join(image_folder, image_name)
    try:
        image = preprocess_image(image_path)
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_name}: {e}")
        continue

    # Thêm batch dimension và chuyển lên device
    image = image.unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        y_pred = model(image)  # Shape: (batch, seq_len, vocab_size)
        y_pred = y_pred.permute(1, 0, 2)  # Shape: (seq_len, batch, vocab_size)
        _, max_index = torch.max(y_pred, dim=2)  # Lấy chỉ số có xác suất cao nhất
        raw_prediction = max_index[:, 0].cpu()  # Lấy batch đầu tiên

        # Loại bỏ các ký tự trùng lặp và blank (0)
        prediction = []
        for c, _ in groupby(raw_prediction.numpy()):  # Sửa từ utils.groupby thành groupby
            if c != 0:  # Bỏ qua blank
                prediction.append(c)
        prediction = torch.IntTensor(prediction)

        # Giải mã thành chuỗi
        predicted_text = utils.decode(prediction, int2char)

    # Hiển thị hình ảnh và kết quả
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap="gray")
    plt.title(f"Predicted Text: {predicted_text}")
    plt.axis("off")
    plt.show()
    print(f"Image: {image_name}, Predicted Text: {predicted_text}")
