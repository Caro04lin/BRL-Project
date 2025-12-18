import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision.transforms as transforms

from Imports.dataloader_3 import HARDataSetTrain
from Imports.Models.fusion_tcn_3 import FusionModelTCN
from Imports.Models.MoViNet.config import _C as config

SAVE_DIR = "Pre Trained Model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# Data Loaders
# ===============================
def create_data_loaders(dataset, batch_size=16):
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    valid_size = int(0.15 * total_size)
    test_size = total_size - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    return train_loader, valid_loader, test_loader

# ===============================
# Evaluation
# ===============================
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for video_frames, imu_data, labels in loader:
            video_frames, imu_data, labels = video_frames.to(device), imu_data.to(device), labels.to(device)
            outputs = model(video_frames, imu_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())  # Save softmax outputs

    accuracy = 100 * total_correct / len(loader.dataset)
    return total_loss / len(loader), accuracy, np.array(all_preds), np.array(all_labels), np.array(all_scores)

# ===============================
# Plots
# ===============================
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Confusion_Matrix_9.pdf'))
    plt.show()

def plot_precision_recall_curve(all_labels, all_scores, class_names):
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(all_labels == i, all_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best', shadow=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Precision_Recall_Curve_9.pdf'))
    plt.show()

# ===============================
# Training
# ===============================
def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, device, num_epochs):
    model.to(device)

    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 4

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_train_correct, total_train_samples = 0, 0, 0
        running_loss = 0.0

        for video_frames, imu_data, labels in train_loader:
            video_frames, imu_data, labels = video_frames.to(device), imu_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(video_frames, imu_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        train_accuracy = 100 * total_train_correct / total_train_samples
        print(f"Epoch {epoch+1}/{num_epochs} | Train Accuracy: {train_accuracy:.2f}%")

        # Validation
        val_loss, val_acc, _, _, _ = evaluate_model(model, valid_loader, criterion, device)
        print(f"Validation | Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")


        scheduler.step()

        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        # Early stopping & best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "fusion_movinet_tcn_best_9.pt"))
            print(f"✅ New best model saved with val acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹ Early stopping triggered after {PATIENCE} epochs")
                break


    # Final evaluation

    test_loss, test_acc, all_preds, all_labels, all_scores = evaluate_model(model, test_loader, criterion, device)

    # Download test results to plot Precision-Recall curve => use plot_precision_recall.py
    import pickle
    test_results = {
        "labels": all_labels,
        "preds": all_preds,
        "scores": all_scores,
        "class_names": list(dataset.label_map.keys())
    }
    results_path = os.path.join(SAVE_DIR, "test_results_tcn_9.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(test_results, f)
    print(f"✅ Test results saved to {results_path}")


    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'fusion_movinet_tcn_final_9.pt'))
    print(f"Model saved as {os.path.join(SAVE_DIR, 'fusion_movinet_tcn_final_9.pt')}")

    print(f"Final Test | Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=list(dataset.label_map.keys())))

    cm = confusion_matrix(all_labels, all_preds)
    
    plot_confusion_matrix(cm, list(dataset.label_map.keys()))
    plot_precision_recall_curve(all_labels, all_scores, list(dataset.label_map.keys()))

    
# ===============================
# Main
# ===============================
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = HARDataSetTrain(
    root_dir="/home/epfstudent/Desktop/Caroline/Bristol Robotics Laboratory/AI Project/Dataset/Action Dataset", #Train on the laboratory computer
    transform=transform,
    sequence_length=10
)
    train_loader, valid_loader, test_loader = create_data_loaders(dataset)

    model = FusionModelTCN(
        movinet_config=config.MODEL.MoViNetA0,
        num_classes=len(dataset.label_map),
        tcn_input_size=27,
        tcn_hidden_size=512,
        tcn_num_layers=6,
        tcn_dropout=0.4,
        proj_imu_size=512,
        freeze_backbone=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, device, num_epochs=25)
