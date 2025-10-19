from typing import Tuple
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Загружает данные CIFAR-10 и возвращает DataLoader'ы."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


if __name__ == "__main__":
    # Пример использования
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, num_workers=2)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
