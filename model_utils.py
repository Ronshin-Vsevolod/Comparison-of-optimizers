import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from lion_pytorch import Lion
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_and_evaluate(optimizer_name, num_epochs, target_acc, train_loader, val_loader,
                       model_name, use_amp=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    if model_name == "simplecnn":
        model = SimpleCNN()
    elif model_name == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = torch.compile(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = None

    if optimizer_name == "adam":
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif optimizer_name == "sgd":
        lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif optimizer_name == "lion":
        lr = 1e-4
        optimizer = Lion(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=0.0,
            decoupled_weight_decay=False
        )

    else:
        raise ValueError("Unknown optimizer")

    print(f"Using {optimizer_name.upper()} with learning rate = {lr}")

    train_loss_list, train_acc_list = [], []
    epoch_times = []
    epoch_to_target_acc = None

    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if scheduler:
              scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        epoch_times.append(time.time() - start_time)

        if epoch_to_target_acc is None and train_acc >= target_acc:
            epoch_to_target_acc = epoch + 1

        print(f"[{optimizer_name.upper()}][Epoch {epoch+1}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_loss /= len(testloader)
    test_acc = 100 * correct / total

    print(f"\n[{optimizer_name.upper()}] FINAL TEST RESULTS - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    if epoch_to_target_acc:
        print(f"Epochs to reach {target_acc}% accuracy: {epoch_to_target_acc}")
    else:
        print(f"Did not reach {target_acc}% accuracy in {num_epochs} epochs")

    return {
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "epoch_times": epoch_times,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "epochs_to_target_acc": epoch_to_target_acc or num_epochs
    }