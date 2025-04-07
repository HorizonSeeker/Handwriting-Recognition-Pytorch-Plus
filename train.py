import os
import torch
import numpy as np
import json
import time
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from itertools import groupby

import config
import utils
from dataset import HRDataset
from model import CRNN


def evaluate(loader, model, criterion, device, mode="Validation"):
    model.eval()
    loop = tqdm(loader, desc=f"Evaluating on {mode} set")
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loop):
            batch_size = inputs.shape[0]
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = model(inputs)
            y_pred = y_pred.permute(1, 0, 2)

            input_lengths = torch.IntTensor(batch_size).fill_(37).to(device)
            target_lengths = torch.IntTensor([len(t[t != 0]) for t in labels]).to(device)

            loss = criterion(y_pred, labels, input_lengths, target_lengths)
            total_loss += loss.item()

            _, max_index = torch.max(y_pred, dim=2)

            for i in range(batch_size):
                raw_prediction = list(max_index[:, i].cpu().numpy())
                prediction = torch.IntTensor(
                    [c for c, _ in groupby(raw_prediction) if c != 0])
                real = torch.IntTensor(
                    [c for c, _ in groupby(labels[i].cpu()) if c != 0])

                if len(prediction) == len(real) and torch.all(prediction.eq(real)):
                    correct += 1
                total += 1

            loop.set_postfix({
                "batch": f"{batch_idx+1}/{num_batches}",
                "loss": f"{total_loss/(batch_idx+1):.4f}",
                "accuracy": f"{correct/total:.4f}"
            })

    avg_loss = total_loss / num_batches
    accuracy = correct / total
    return avg_loss, accuracy, total


def train_one_epoch(loader, model, optimizer, criterion, device, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Training Epoch {epoch+1}/{config.NUM_EPOCHS}")
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(loader)

    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(loop):
        batch_size = inputs.shape[0]
        inputs = inputs.to(device)
        labels = labels.to(device)

        y_pred = model(inputs)
        y_pred = y_pred.permute(1, 0, 2)

        input_lengths = torch.IntTensor(batch_size).fill_(37).to(device)
        target_lengths = torch.IntTensor([len(t[t != 0]) for t in labels]).to(device)

        loss = criterion(y_pred, labels, input_lengths, target_lengths)
        total_loss += loss.item()

        _, max_index = torch.max(y_pred, dim=2)

        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].cpu().numpy())
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != 0])
            real = torch.IntTensor(
                [c for c, _ in groupby(labels[i].cpu()) if c != 0])

            if len(prediction) == len(real) and torch.all(prediction.eq(real)):
                correct += 1
            total += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix({
            "batch": f"{batch_idx+1}/{num_batches}",
            "loss": f"{total_loss/(batch_idx+1):.4f}",
            "accuracy": f"{correct/total:.4f}",
            "samples": total
        })

    end_time = time.time()
    epoch_time = end_time - start_time

    avg_loss = total_loss / num_batches
    accuracy = correct / total
    return avg_loss, accuracy, total, epoch_time


def save_checkpoint(model, optimizer, scheduler, epoch, history, filename="checkpoint.pth"):
    # Xóa tất cả checkpoint cũ (trừ best_checkpoint)
    for f in os.listdir('.'):
        if f.startswith('checkpoint') and f.endswith('.pth') and f != 'best_checkpoint.pth':
            os.remove(f)
    if os.path.exists(filename):
        os.remove(filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to '{filename}' at epoch {epoch+1}")


def save_best_checkpoint(model, optimizer, scheduler, epoch, history, val_loss, best_val_loss, filename="best_checkpoint.pth"):
    if val_loss < best_val_loss:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "val_loss": val_loss
        }
        torch.save(checkpoint, filename)
        print(f"Best checkpoint saved to '{filename}' at epoch {epoch+1} with validation loss {val_loss:.4f}")
        return val_loss
    return best_val_loss


def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth"):
    if os.path.exists(filename):
        try:
            checkpoint = torch.load(filename, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            history = checkpoint["history"]
            print(f"Loaded checkpoint from '{filename}', resuming from epoch {start_epoch}")
            return start_epoch, history
        except Exception as e:
            print(f"Không thể load checkpoint từ '{filename}' do lỗi: {e}")
            print("Bắt đầu huấn luyện từ epoch 1.")
            return 0, {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
    else:
        print(f"Không tìm thấy checkpoint tại '{filename}', bắt đầu huấn luyện từ epoch 1.")
        return 0, {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }


def main():
    train_data = utils.get_dataset(config.TRAIN_CSV)
    valid_data = utils.get_dataset(config.VALID_CSV)
    test_data = utils.get_dataset(config.TEST_CSV)

    train_dataset = HRDataset(train_data, utils.encode, mode='train')
    valid_dataset = HRDataset(valid_data, utils.encode, mode='valid')
    test_dataset = HRDataset(test_data, utils.encode, mode='test')

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    input_size = 64
    hidden_size = 256  # Tăng từ 128 lên 256
    output_size = config.VOCAB_SIZE + 1
    num_layers = 2

    model = CRNN(input_size, hidden_size, output_size, num_layers)
    model.to(config.DEVICE)

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    criterion.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6  # Tăng patience từ 2 lên 5
    )

    start_epoch, history = load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth")
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_val_accuracy = 0.0  # Theo dõi validation accuracy tốt nhất
    patience_lr_reset = 5  # Số epoch chờ trước khi reset learning rate
    lr_reset_counter = 0

    print("\n=== Training Configuration ===")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Number of Epochs: {config.NUM_EPOCHS}")
    print(f"Initial Learning Rate: {config.LEARNING_RATE}")
    print(f"Train Dataset Size: {len(train_dataset)} samples")
    print(f"Validation Dataset Size: {len(valid_dataset)} samples")
    print(f"Test Dataset Size: {len(test_dataset)} samples")
    print("=============================\n")

    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            print(f"\n=== Epoch {epoch+1}/{config.NUM_EPOCHS} ===")
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            train_loss, train_accuracy, train_samples, epoch_time = train_one_epoch(
                train_loader, model, optimizer, criterion, config.DEVICE, epoch)
            
            val_loss, val_accuracy, val_samples = evaluate(
                valid_loader, model, criterion, config.DEVICE, mode="Validation")

            print(f"\n--- Epoch {epoch+1} Summary ---")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Training Samples Processed: {train_samples}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Samples Processed: {val_samples}")
            print(f"Epoch Time: {epoch_time:.2f} seconds")
            print("---------------------------")

            history["train_loss"].append(float(train_loss))
            history["train_accuracy"].append(float(train_accuracy))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_accuracy))

            save_checkpoint(model, optimizer, scheduler, epoch, history, filename="checkpoint.pth")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = save_best_checkpoint(
                    model, optimizer, scheduler, epoch, history, val_loss, best_val_loss, filename="best_checkpoint.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Validation loss không cải thiện. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping tại epoch {epoch+1} do validation loss không cải thiện sau {patience} epoch.")
                    break

            # Reset learning rate nếu validation accuracy không cải thiện
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                lr_reset_counter = 0
            else:
                lr_reset_counter += 1
                if lr_reset_counter >= patience_lr_reset:
                    print(f"Reset learning rate về {config.LEARNING_RATE} do validation accuracy không cải thiện sau {patience_lr_reset} epoch.")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config.LEARNING_RATE
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
                    )
                    lr_reset_counter = 0

            with open("training_history.json", "w") as f:
                json.dump(history, f)
            print("Training history updated in 'training_history.json'")

            scheduler.step(val_loss)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print("Saving current state...")

        torch.save(model.state_dict(), "trained_model.pth")
        print("Model saved to 'trained_model.pth'")

        with open("training_history.json", "w") as f:
            json.dump(history, f)
        print("Training history saved to 'training_history.json'")

        print("\n=== Evaluating on Test Set ===")
        test_loss, test_accuracy, test_samples = evaluate(
            test_loader, model, criterion, config.DEVICE, mode="Test")
        
        print(f"\n--- Test Results ---")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Samples Processed: {test_samples}")
        print("--------------------")

        with open("test_results.json", "w") as f:
            json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}, f)
        print("Test results saved to 'test_results.json'")
        return

    torch.save(model.state_dict(), "trained_model.pth")
    print("\nModel saved to 'trained_model.pth'")

    with open("training_history.json", "w") as f:
        json.dump(history, f)
    print("Training history saved to 'training_history.json'")

    print("\n=== Evaluating on Test Set ===")
    test_loss, test_accuracy, test_samples = evaluate(
        test_loader, model, criterion, config.DEVICE, mode="Test")
    
    print(f"\n--- Test Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Samples Processed: {test_samples}")
    print("--------------------")

    with open("test_results.json", "w") as f:
        json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}, f)
    print("Test results saved to 'test_results.json'")


if __name__ == "__main__":
    main()