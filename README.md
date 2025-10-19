# Comparison of Optimizers

This project compares deep learning optimizers using PyTorch and the Lion optimizer. It runs on a local machine with GPU (CUDA) or in Google Colab.

## Prerequisites

- **Python**: 3.12.3
- **Poetry**: 1.8.2
- **OS**: Linux (tested on Ubuntu with NVIDIA GPU and CUDA 12.8)
- **Optional for development**: `pre-commit`, `black`, `flake8`, `mypy`

## Setup Instructions (Local)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Comparison-of-optimizers
```

### 2. Install Poetry
```bash
pip install poetry==1.8.2
poetry --version
```

### 3. Set Up Virtual Environment
```bash
poetry env use python3.12
source $(poetry env info --path)/bin/activate
```

### 4. Install Core Dependencies
```bash
poetry install --no-root
```

### 5. Install Development Dependencies
```bash
pip install flake8>=7.0.0 mypy>=1.9.0 black>=24.3.0 pre-commit>=3.7.0
```

### 6. Verify Dependencies
```bash
pip list | grep -E "torch|torchvision|lion-pytorch|matplotlib|pandas|numpy|pyyaml|omegaconf|flake8|mypy|black|pre-commit"
```
Expected output:
```
black                    25.9.0 (or newer)
flake8                   7.3.0 (or newer)
lion-pytorch             0.1.0 (or newer)
matplotlib               3.8.0 (or newer)
mypy                     1.18.2 (or newer)
numpy                    1.26.0 (or newer)
omegaconf                2.3.0 (or newer)
pandas                   2.2.0 (or newer)
pre-commit               4.3.0 (or newer)
pyyaml                   6.0.1 (or newer)
torch                    2.9.0+cu128
torchvision              0.17.0 (or newer)
```

### 7. Verify GPU Support
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected output: `2.9.0+cu128 True`

### 8. Run Pre-Commit Checks (Optional)
For developers:
```bash
pre-commit run --all-files
```

## Running the Program (Local)
```bash
cd ~/Workplace/Comparison-of-optimizers
python3 -m src.run_main
```

## Setup and Running in Google Colab

### 1. Upload Project Files
- Upload `src/`, `configs/`, and `requirements.txt` to Google Colab.

### 2. Enable GPU
- Go to `Runtime > Change runtime type > Hardware accelerator > GPU`.

### 3. Install Dependencies
```python
!pip install -r requirements.txt
```

### 4. Verify GPU Support
```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

### 5. Run the Program
```python
!python3 -m src.run_main
```

## Development Tools
- Format code: `black src/`
- Lint code: `flake8 src/`
- Type check: `mypy src/`
- Run all checks: `pre-commit run --all-files`
