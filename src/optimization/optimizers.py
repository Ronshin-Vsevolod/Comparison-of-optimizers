import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from lion_pytorch import Lion
from typing import Tuple, Dict, Any, Optional


def get_optimizer_and_scheduler(
    model: nn.Module, config: Dict[str, Any], num_epochs: int
) -> Tuple[optim.Optimizer, Optional[_LRScheduler]]:
    """Инициализирует оптимизатор и шедулер на основе конфигурации."""
    optimizer_name = config["optimizer_name"]
    lr = config["lr"]
    scheduler = None

    params = model.parameters()

    if optimizer_name == "adam":
        optimizer = optim.Adam(params, lr=lr)

    elif optimizer_name == "sgd":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    elif optimizer_name == "lion":
        betas = config.get("betas", (0.9, 0.99))
        weight_decay = config.get("weight_decay", 0.0)
        optimizer = Lion(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            decoupled_weight_decay=False,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"Using {optimizer_name.upper()} with LR={lr}")
    return optimizer, scheduler
