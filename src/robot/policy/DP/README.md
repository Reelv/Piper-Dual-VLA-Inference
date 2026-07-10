[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# DP (Diffusion Policy) Deployment Guide

## Quick Start

### 1. Environment Setup

Install the dependencies required for DP deployment:

```bash
cd policy/DP/
pip install -e .
```

**Optional Dependencies:**

```bash
# Install with training dependencies (wandb, tensorboard, etc.)
pip install -e .[training]

# Install with simulation dependencies
pip install -e .[simulation]

# Install with all optional dependencies
pip install -e .[all]
```

### 2. Data Preparation and Training

#### Data Conversion

Convert the collected data to zarr format required by DP model:

```bash
cd policy/DP/
python scripts/process_data.py <source_dir> <output_dir> <num_episodes>
```

**Example:**
```bash
# Convert 100 episodes from robot.data/test_data/ to zarr format
python process_data.py data/test_data/ processed_data/test_data-100.zarr/ 100
```

### 3. Real Robot Deployment

1. Copy the trained checkpoint to the following directory:
```
control_your_robot/policy/DP/checkpoints/
```

2. Modify the deployment script `example/deploy/piper_single_on_DP.py`:

```python
# Modify the model path around line 316
model = MYDP(model_path="policy/DP/checkpoints/feed_test_30-100-0/300.ckpt", task_name="feed_test_30", INFO="DEBUG")
```

**Parameter Description:**
- `model_path`: Path to the policy model checkpoint
- `task_name`: Corresponding task name
- `INFO`: Log level (DEBUG/INFO/ERROR)

#### 3.3 Execute Deployment

Run the deployment script to start real robot execution:

```bash
python example/deploy/piper_single_on_DP.py
```

## Notes

- Ensure the robotic arm is properly enabled and connected
- Check if the model path is correct
- It is recommended to verify the model performance in a test environment before deployment


