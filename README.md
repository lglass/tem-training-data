# Getting started

## Install the Simulator
Get simulation software and install

```
wget https://gigenet.dl.sourceforge.net/project/tem-simulator/TEM-simulator_1.3.zip?viasf=1
```

(Alternatively visit [http://tem-simulator.sourceforge.net](http://tem-simulator.sourceforge.net))

The fftw3 location on Linux should be 

```
-I /usr/include -L /usr/lib/
```

## Python environment

Choose your preferred environment manager below to create the `protein-mixtures` environment and install the required dependencies.

### Option 1: uv

```bash
# Create the virtual environment
uv venv protein-mixtures

# Activate the environment
source protein-mixtures/bin/activate  # On Windows: protein-mixtures\Scripts\activate

# Install dependencies
uv pip install mrcfile starfile scipy cryosparc-tools python-dotenv tifffile torch torchvision pytorch-ignite
```

### Option 2: conda
```bash
conda create -n protein-mixtures python=3 -y
conda activate protein-mixtures

# Install dependencies
pip install mrcfile starfile scipy cryosparc-tools python-dotenv tifffile torch torchvision pytorch-ignite
```

