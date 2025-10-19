import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, Optional, List
from src.core.utils import calculate_grad_norm

# GradScalerType = Any if not hasattr(torch.cuda.amp, "GradScaler") else torch.cuda.amp.GradScaler
GradScalerType = Any


def train_and_evaluate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_name: str,
    num_epochs: int,
    target_acc: float,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Чистый цикл обучения и оценки.
    """
    criterion = nn.CrossEntropyLoss()

    train_loss_list, train_acc_list = [], []
    epoch_times: List[float] = []
    epoch_to_target_acc: Optional[int] = None
    grad_norms_per_epoch: List[float] = []

    scaler: Optional[GradScalerType] = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    # --- Epoch loop ---
    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        batch_grad_norms: List[float] = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                # Automatic Mixed Precision block
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()

                batch_grad_norms.append(calculate_grad_norm(model))

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                batch_grad_norms.append(calculate_grad_norm(model))

                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # --- End of epoch metrics ---
        if scheduler:
            scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Средняя норма градиента за эпоху
        epoch_grad_norm = (
            sum(batch_grad_norms) / len(batch_grad_norms) if batch_grad_norms else 0.0
        )
        grad_norms_per_epoch.append(epoch_grad_norm)

        epoch_times.append(time.time() - start_time)

        if epoch_to_target_acc is None and train_acc >= target_acc:
            epoch_to_target_acc = epoch + 1

        print(
            f"[{optimizer_name.upper()}][Epoch {epoch+1:02d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
        )

    # --- Evaluation on validation set (Test set) ---
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(val_loader)
    test_acc = 100 * correct / total

    print(
        f"\n[{optimizer_name.upper()}] FINAL TEST RESULTS - \
            Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%"
    )

    # Если не достигли целевой точности, возвращаем общее количество эпох
    epochs_to_target_acc_final = (
        epoch_to_target_acc if epoch_to_target_acc is not None else num_epochs
    )

    return {
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "epoch_times": epoch_times,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "epochs_to_target_acc": epochs_to_target_acc_final,
        "grad_norms": grad_norms_per_epoch,
    }
