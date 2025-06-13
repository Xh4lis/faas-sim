# Complete Installation Guide for faas-sim

## Step 1: Create and activate a Python 3.10 virtual environment

```bash
python3.10 -m venv venv10
source venv10/bin/activate
```

## Step 2: Upgrade pip to latest version

```bash
pip install --upgrade pip
```

## Step 3: Install pre-built binary packages

### Foundation packages (NumPy and SciPy)

```bash
pip install numpy==1.21.6 scipy==1.8.1
```

### Scientific packages

```bash
pip install pandas==1.3.5 matplotlib==3.5.1
```

### Remaining dependencies

```bash
pip install scikit-learn==1.0.2 joblib==0.17.0 tpot==0.11.5
pip install simpy==3.0.11 srds==0.1.0 tqdm requests networkx==2.6.3
pip install pyvis==0.1.9
```

## Step 4: Fix compatibility issue in ether

### Navigate to ether directory

```bash
cd /udd/msayah/Mahfoud/sim/ether
```

### Fix the Python 3.10 compatibility issue

```bash
sed -i 's/from collections import defaultdict, Iterable/from collections import defaultdict\nfrom collections.abc import Iterable/' /udd/msayah/Mahfoud/sim/ether/ether/cell.py
```

### Install ether locally without dependencies

```bash
pip install -e . --no-deps
```

## Step 5: Install skippy-core and faas-sim

### Install skippy-core

```bash
pip install edgerun-skippy-core>=0.1.1
```

### Install faas-sim

```bash
cd /udd/msayah/Mahfoud/sim/faas-sim
pip install -e . --no-deps
```

### Test the installation

```bash
python -m examples.basic.main
```
