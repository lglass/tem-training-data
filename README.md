# Getting started

## Structure concentrations

Provide each structure with its TEM-simulator `particle_conc` value using
`NAME:CONCENTRATION` entries separated by commas:

```bash
python input_generator.py \
    --structures 7Q5Q:1200,7Q5R:1200,3JCT:400 \
    --output-dir data \
    --run-id example
```

Every structure must include a finite, non-negative concentration. Duplicate
structure names and bare names such as `7Q5Q` are rejected.

The parallel runner keeps these values alongside each structure:

```python
mixtures = [
    ("7Q5Q", 1200),
    ("7Q5R", 1200),
    ("3JCT", 400),
]
```

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

