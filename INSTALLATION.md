# Installation Guide

This document provides comprehensive installation instructions for different package managers and environments.

## 🚀 Quick Start (Automated)

```bash
# Run the automated setup script
python setup_environment.py
```

## 📦 Installation Options

### Option 1: Mamba (Recommended)

**Best for**: Maximum compatibility, optimized package resolution

```bash
# Install mamba if you don't have it
conda install mamba -n base -c conda-forge

# Create environment from optimized config
mamba env create -f mamba-environment.yml

# Activate environment
mamba activate augmentation-agent-topic
```

### Option 2: Conda (Standard)

**Best for**: Standard conda users

```bash
# Create environment from standard config
conda env create -f environment.yml

# Activate environment
conda activate augmentation-agent-topic
```

### Option 3: Pure Pip

**Best for**: Virtual environments, containers, or pip-only workflows

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 4: Poetry (Advanced)

**Best for**: Dependency lock management

```bash
# Install poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies (requires pyproject.toml)
poetry install
poetry shell
```

## 🔧 GPU Support (Optional)

### CUDA Installation

If you want GPU acceleration for embeddings:

**With Mamba/Conda:**
```bash
# Already included in mamba-environment.yml
# Or install manually:
mamba install pytorch pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia
```

**With Pip:**
```bash
# Install CUDA-enabled PyTorch
pip install torch>=2.0.0+cu118 -f https://download.pytorch.org/whl/cu118
```

### Verify GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Mamba/Conda Package Conflicts

**Issue**: Warning about pip and conda package conflicts

**Solution**:
```bash
# Use the mamba-optimized configuration
mamba env create -f mamba-environment.yml

# Or create clean environment
mamba create -n augmentation-agent-topic python=3.10
mamba activate augmentation-agent-topic
mamba install -c conda-forge pandas numpy scikit-learn python-dotenv
pip install bertopic openai langchain
```

#### 2. PyTorch Installation Issues

**Issue**: PyTorch not installing correctly

**Solution**:
```bash
# Uninstall conflicting versions
pip uninstall torch torchvision torchaudio

# Reinstall from official source
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
# Or for CUDA:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 3. BERTopic Dependencies

**Issue**: UMAP or HDBSCAN compilation errors

**Solution**:
```bash
# Install build tools first
mamba install gcc_linux-64 gxx_linux-64  # Linux
# or
mamba install clang_osx-64 clangxx_osx-64  # macOS

# Then install packages
pip install umap-learn hdbscan
```

#### 4. OpenAI Package Conflicts

**Issue**: Multiple OpenAI package versions

**Solution**:
```bash
# Clean installation
pip uninstall openai
pip install openai>=1.0.0

# Verify version
python -c "import openai; print(openai.__version__)"
```

#### 5. Environment Variables Not Loading

**Issue**: `.env` file not being read

**Solution**:
```bash
# Verify file location
ls -la .env

# Check file contents
cat .env | grep -v "^#" | head -5

# Test loading
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY', 'NOT_FOUND')[:10])"
```

### Platform-Specific Issues

#### Windows

```bash
# If you encounter SSL issues
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <package>

# For long path issues
git config --system core.longpaths true
```

#### macOS

```bash
# If you encounter compiler issues
xcode-select --install

# For M1/M2 Macs, use conda-forge channel
mamba install -c conda-forge <package>
```

#### Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Or on RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

## 📊 Verify Installation

Run the verification script:

```bash
python -c "
import sys
print(f'Python: {sys.version}')

packages = [
    'pandas', 'numpy', 'sklearn', 'bertopic',
    'openai', 'transformers', 'torch', 'dotenv'
]

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: not installed')
"
```

Or use the built-in validation:

```bash
python src/main.py --validate-config
```

## 🔄 Environment Management

### Switching Between Environments

```bash
# List environments
mamba env list

# Remove old environment
mamba env remove -n augmentation-agent-topic

# Create new environment
mamba env create -f mamba-environment.yml

# Export current environment
mamba env export > my-environment.yml
```

### Updating Dependencies

```bash
# Update all packages
mamba update --all

# Update specific package
mamba update bertopic

# Update from requirements
pip install -r requirements.txt --upgrade
```

## 🐳 Docker Alternative

If you prefer containerized deployment:

```bash
# Build Docker image (requires Dockerfile)
docker build -t augmentation-agent-topic .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -e OPENAI_API_KEY=your-key-here \
  augmentation-agent-topic
```

## 📞 Getting Help

1. **Check logs**: Look in `logs/` directory for detailed error messages
2. **Validate config**: Run `python src/main.py --validate-config`
3. **Test setup**: Run `python setup_environment.py`
4. **Environment info**: Run `python src/main.py --show-config`

For persistent issues, please include:
- Operating system and version
- Python version
- Package manager used
- Complete error message
- Output of validation commands