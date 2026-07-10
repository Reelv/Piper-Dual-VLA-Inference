[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README_EN.md)

# ACT (Action Chunking Transformer) Deployment Guide

## Quick Start

### 1. Environment Setup

Install the dependencies required for ACT deployment:

```bash
cd policy/ACT/
pip install -r requirements.txt
```

### 2. Data Preparation and Training

#### Data Conversion

Convert the collected data to HDF5 format required by ACT model:

```bash
python scripts/convert2act_hdf5.py <input_data_path> <output_path>
```

**Examples:**
```bash
# Convert data from save/test/ directory to ACT format
python scripts/convert2act_hdf5.py ./save/test/ ~/RoboTwin/policy/ACT/data/

# Convert pick_place_cup task data
python scripts/convert2act_hdf5.py save/pick_place_cup /path/to/output
```

### 3. Real Robot Deployment

1. Copy the trained checkpoint to the following directory:
```
control_your_robot/policy/ACT/actckpt/
```

2. Modify the deployment script `example/deploy/piper_single_on_ACT.py`:

```python
# Modify the model path around line 120
model = MYACT("/path/your/policy/ACT/act_ckpt/act-pick_place_cup/100", "act-pick_place_cup")
```

**Parameter Description:**
- First parameter: Policy model folder path
- Second parameter: Corresponding task name

#### 3.3 Execute Deployment

Run the deployment script to start real robot execution:

```bash
python example/deploy/piper_single_on_ACT.py
```

## Notes

- Ensure the robotic arm is properly enabled and connected
- Check if the model path is correct
- It is recommended to verify the model performance in a test environment before deployment

