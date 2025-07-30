# Optimizer Comparison on CIFAR-10: Adam, SGD, and Lion

This project explores the performance of three popular optimizers — **Adam**, **SGD**, and **Lion** — on image classification tasks using the CIFAR-10 dataset. Two model architectures were evaluated: a custom lightweight CNN (*SimpleCNN*) and a standard deep convolutional network (*ResNet18*).

## 🎯 Project Goal

To investigate and compare the training speed, convergence behavior, and final accuracy of Adam, SGD, and Lion when applied to different neural network architectures.

## 🧪 Experiment Setup

- **Dataset**: CIFAR-10 (50k training, 10k test images)
- **Architectures**:
  - `SimpleCNN`: A custom shallow CNN with ~1.25M parameters.
  - `ResNet18`: A standard residual network, used *without pretraining*.
- **Optimizers**:
  - Adam (`lr=0.001`)
  - SGD + momentum (`lr=0.05`)
  - Lion (`lr=1e-4`)
- **Training duration**: 30 epochs for each optimizer/architecture combination
- **Acceleration**:
  - Mixed precision training (`torch.amp`)
  - Graph-mode compilation (`torch.compile`)

## 📈 Key Results

| Model (Opt)          | Time (s) | Epochs to 70% | Final Accuracy | Final Loss | Loss Std (last 5) |
|----------------------|----------|----------------|----------------|------------|-------------------|
| SimpleCNN (Adam)     | 439.9    | 21             | 74.26%         | 0.7892     | 0.0161            |
| SimpleCNN (SGD)      | 378.6    | 8              | 74.91%         | 0.8662     | 0.0183            |
| SimpleCNN (Lion)     | 387.4    | 6              | 74.17%         | 0.9438     | 0.0124            |
| ResNet18 (Adam)      | 691.0    | 2              | 84.02%         | 0.8934     | 0.0047            |
| ResNet18 (SGD)       | 579.5    | 3              | 79.81%         | 1.0590     | 0.0000            |
| ResNet18 (Lion)      | 642.3    | 2              | 83.73%         | 0.9086     | 0.0015            |

## 🔍 Observations

**SGD** is the best choice if the goal is high final accuracy. It is especially effective on small models if there is time for training. It is sensitive to the learning rate, but with the right settings, it gives excellent results.
**Adam** is a good choice for complex models and situations where you need to quickly achieve acceptable quality. It is especially useful when the number of epochs is limited or when fast initial training is required.
**Lion** demonstrates good stability and fast convergence on simple architectures. In such conditions, it can outperform Adam in terms of speed and resource efficiency. However, its advantages are reduced on more complex models.
