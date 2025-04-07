import os
import json
import matplotlib.pyplot as plt

# Đường dẫn lưu biểu đồ
chart_folder = "./figures"  # Thay đổi thành thư mục "figures" theo ví dụ của cô
if not os.path.exists(chart_folder):
    os.makedirs(chart_folder)

# Đọc dữ liệu từ file
try:
    with open("training_history.json", "r") as f:
        history = json.load(f)
except FileNotFoundError:
    print("Không tìm thấy file 'training_history.json'. Vui lòng chạy huấn luyện trước.")
    exit(1)
except Exception as e:
    print(f"Lỗi khi đọc file 'training_history.json': {e}")
    exit(1)

try:
    with open("test_results.json", "r") as f:
        test_results = json.load(f)
except FileNotFoundError:
    print("Không tìm thấy file 'test_results.json'. Vui lòng chạy huấn luyện và đánh giá trước.")
    exit(1)
except Exception as e:
    print(f"Lỗi khi đọc file 'test_results.json': {e}")
    exit(1)

# Kiểm tra dữ liệu rỗng
if not history["train_loss"]:
    print("Dữ liệu huấn luyện rỗng. Vui lòng chạy huấn luyện trước.")
    exit(1)

# Dữ liệu
epochs = list(range(1, len(history["train_loss"]) + 1))
train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_accuracy = history["train_accuracy"]
val_accuracy = history["val_accuracy"]
test_accuracy = test_results["test_accuracy"]

# Vẽ và lưu biểu đồ Loss
plt.figure(figsize=(12, 5))
plt.plot(epochs, train_loss, "b*-", label="Training")
plt.plot(epochs, val_loss, "r*-", label="Validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss vs. No. of epochs")
plt.legend()
plt.grid(True)
loss_save_path = os.path.join(chart_folder, "loss_chart_iou.png")  # Tên file theo ví dụ của cô
plt.savefig(loss_save_path, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ Loss tại: {loss_save_path}")

# Vẽ và lưu biểu đồ Accuracy
plt.figure(figsize=(12, 5))
plt.plot(epochs, train_accuracy, "b*-", label="Training")
plt.plot(epochs, val_accuracy, "r*-", label="Validation")
plt.axhline(y=test_accuracy, color="g", linestyle="--", label=f"Test Accuracy ({test_accuracy:.4f})")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy vs. No. of epochs")
plt.legend()
plt.grid(True)
accuracy_save_path = os.path.join(chart_folder, "Accuracy.png")  # Tên file theo yêu cầu của cô
plt.savefig(accuracy_save_path, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ Accuracy tại: {accuracy_save_path}")

# Vẽ và lưu biểu đồ kết hợp (nếu cần)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, "b*-", label="Training")
plt.plot(epochs, val_loss, "r*-", label="Validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss vs. No. of epochs")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, "b*-", label="Training")
plt.plot(epochs, val_accuracy, "r*-", label="Validation")
plt.axhline(y=test_accuracy, color="g", linestyle="--", label=f"Test Accuracy ({test_accuracy:.4f})")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy vs. No. of epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
combined_save_path = os.path.join(chart_folder, "training_and_test_charts.png")
plt.savefig(combined_save_path, dpi=300)
plt.close()  # Thêm close để tránh chồng lấp
print(f"Đã lưu biểu đồ kết hợp tại: {combined_save_path}")