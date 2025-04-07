import torch
import torch.nn as nn
from model import CRNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(checkpoint_path, input_size=64, hidden_size=256, output_size=31, num_layers=2):
    model = CRNN(input_size=input_size, hidden_size=hidden_size,
                 output_size=output_size, num_layers=num_layers)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

def main():
    checkpoint_path = 'checkpoint.pth'
    try:
        model = load_model(checkpoint_path)
        print("Mô hình đã được load thành công!")
        print(f"Thiết bị đang dùng: {DEVICE}")

        # Dùng input (32, 96) thay vì (32, 128) để khớp với 1536
        dummy_input = torch.randn(1, 1, 32, 96).to(DEVICE)
        output = model(dummy_input)
        print("Kích thước đầu ra:", output.shape)
    
    except Exception as e:
        print(f"Không thể load mô hình từ {checkpoint_path}: {str(e)}")

if __name__ == "__main__":
    main()