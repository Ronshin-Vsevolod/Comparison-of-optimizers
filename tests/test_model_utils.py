import pytest
from model_utils import train_and_evaluate

def test_train_and_evaluate_runs():
    import torchvision.transforms as transforms
    import torchvision
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    result = train_and_evaluate(
        optimizer_name="adam",
        num_epochs=1,
        target_acc=30,
        train_loader=trainloader,
        val_loader=testloader,
        model_name="simplecnn",
        use_amp=False
    )

    assert "train_loss" in result
    assert "test_acc" in result
    assert isinstance(result["test_acc"], float)
