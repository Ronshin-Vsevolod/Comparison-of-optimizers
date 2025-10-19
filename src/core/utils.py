import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def get_device(device_name: str) -> torch.device:
    """Возвращает подходящее устройство (GPU/CPU)."""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device


def calculate_grad_norm(model: nn.Module) -> float:
    """Рассчитывает L2 норму градиентов модели."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # param_norm = p.grad.data.norm(2) # Используем .data для mypy
            # total_norm += param_norm.item() ** 2
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


# --- Функции визуализации ---


def plot_metric(metric_name: str, results_dict: Dict[str, Any], title: str, filename: str) -> None:
    """Строит график для заданной метрики (loss/accuracy/grad_norm)."""
    plt.figure(figsize=(10, 6))
    for name, result in results_dict.items():
        if metric_name in result:
            plt.plot(result[metric_name], label=name)

    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()  # Закрыть фигуру


def summarize_result(
    name: str, results: Dict[str, Any], training_time: float, threshold: float = 70.0
) -> Dict[str, Any]:
    """Генерирует строку сводной таблицы."""
    last_losses = results["train_loss"][-5:]
    loss_std = np.std(last_losses)

    # Используем 'epochs_to_target_acc' из тренера
    epochs_to_acc = results["epochs_to_target_acc"]

    return {
        "Model (Opt)": name,
        "Time (s)": f"{training_time:.1f}",
        f"Epochs to {int(threshold)}%": (
            epochs_to_acc if epochs_to_acc != len(results["train_acc"]) else "-"
        ),
        "Final Acc (%)": f"{results['test_acc']:.2f}",
        "Final Loss": f"{results['test_loss']:.4f}",
        "Loss Std (Last 5)": f"{loss_std:.4f}",
    }
