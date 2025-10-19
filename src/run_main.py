import torch
import yaml
import time
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.core.utils import get_device, plot_metric, summarize_result
from src.data_loaders import get_cifar10_loaders
from src.models.arch import get_model
from src.optimization.optimizers import get_optimizer_and_scheduler
from src.core.trainer import train_and_evaluate


def load_config(path: str) -> Dict[str, Any]:
    """Загружает YAML файл конфигурации."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(
    exp_name: str,
    exp_config: Dict[str, Any],
    global_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], float]:
    """Запускает один эксперимент (обучение + оценка)."""

    model = get_model(exp_config["model_name"], device.type)

    num_epochs = global_config["num_epochs"]
    target_acc = global_config["target_acc"]

    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, config=exp_config, num_epochs=num_epochs
    )

    print(f"\n==== STARTING EXPERIMENT: {exp_name} ====")
    start_time = time.time()

    results = train_and_evaluate(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer_name=exp_config["optimizer_name"],
        num_epochs=num_epochs,
        target_acc=target_acc,
        device=device,
        use_amp=global_config["use_amp"],
    )

    elapsed = time.time() - start_time
    print(f"Total time for {exp_name}: {elapsed:.1f} seconds.")
    return results, elapsed


def main():
    # 1. Загрузка конфигураций
    global_config = load_config("configs/default.yaml")
    experiments_config = load_config("configs/experiments.yaml")

    # 2. Настройка окружения
    device = get_device(global_config["device"])
    torch.manual_seed(global_config["seed"])

    # 3. Загрузка данных
    print("\n==== LOADING DATA ====")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=global_config["batch_size"], num_workers=global_config["num_workers"]
    )

    # 4. Запуск всех экспериментов
    all_results: Dict[str, Any] = {}
    all_times: Dict[str, float] = {}

    for exp_name, exp_config in experiments_config.items():
        results, elapsed = run_experiment(
            exp_name=exp_name,
            exp_config=exp_config,
            global_config=global_config,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        all_results[exp_name] = results
        all_times[exp_name] = elapsed

    # 5. Визуализация и суммарные таблицы
    print("\n==== ANALYSIS AND PLOTTING ====")

    # Группировка результатов по архитектуре для графиков
    results_simplecnn = {k: v for k, v in all_results.items() if "simplecnn" in k}
    results_resnet = {k: v for k, v in all_results.items() if "resnet18" in k}

    # Графики SimpleCNN
    plot_metric(
        "train_loss",
        results_simplecnn,
        "Train Loss Comparison (SimpleCNN)",
        "simplecnn_loss.png",
    )
    plot_metric(
        "train_acc",
        results_simplecnn,
        "Train Accuracy Comparison (SimpleCNN)",
        "simplecnn_accuracy.png",
    )
    plot_metric(
        "grad_norms",
        results_simplecnn,
        "Gradient Norm per Epoch (SimpleCNN)",
        "simplecnn_grad_norm.png",
    )
    plot_metric(
        "epoch_times",
        results_simplecnn,
        "Epoch Time Comparison (SimpleCNN)",
        "simplecnn_times.png",
    )

    # Графики ResNet18
    plot_metric(
        "train_loss",
        results_resnet,
        "Train Loss Comparison (ResNet18)",
        "resnet18_loss.png",
    )
    plot_metric(
        "train_acc",
        results_resnet,
        "Train Accuracy Comparison (ResNet18)",
        "resnet18_accuracy.png",
    )
    plot_metric(
        "grad_norms",
        results_resnet,
        "Gradient Norm per Epoch (ResNet18)",
        "resnet18_grad_norm.png",
    )
    plot_metric(
        "epoch_times",
        results_resnet,
        "Epoch Time Comparison (ResNet18)",
        "resnet18_times.png",
    )

    # Сводная таблица
    summary_rows: List[Dict[str, Any]] = []
    for exp_name, results in all_results.items():
        summary_rows.append(
            summarize_result(
                name=exp_name,
                results=results,
                training_time=all_times[exp_name],
                threshold=global_config["target_acc"],
            )
        )

    df_summary = pd.DataFrame(summary_rows)
    print("\n==== FINAL SUMMARY TABLE ====")
    print(df_summary.to_markdown(index=False))


if __name__ == "__main__":
    # Для запуска проекта потребуется:
    # pip install torch torchvision lion-pytorch matplotlib pandas pyyaml omegaconf
    # python run_main.py
    main()
