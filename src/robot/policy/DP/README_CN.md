[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# DP (Diffusion Policy) 模型部署指南

## 快速开始

### 1. 环境配置

安装 DP 部署所需的依赖环境:

```bash
cd policy/DP/
pip install -e .
```

### 2. 数据准备与训练

#### 数据转换

将采集的数据转换为 DP 模型所需的 zarr 格式:

```bash
cd policy/DP/
python scripts/process_data.py <source_dir> <output_dir> <num_episodes>
# 例子：python process_data.py data/test_data/ processed_data/test_data-100.zarr/ 100
```

### 3. 真机部署


1. 将训练好的 checkpoint 复制到以下目录:
```
control_your_robot/policy/DP/checkpoints/
```

2. 修改部署脚本 `example/deploy/piper_single_on_DP.py`:

```python
# 在第 316 行左右修改模型路径
 model = MYDP(model_path="policy/DP/checkpoints/feed_test_30-100-0/300.ckpt", task_name="feed_test_30", INFO="DEBUG")
```

**参数说明:**
- 第一个参数: Policy 模型的文件夹地址
- 第二个参数: 对应的任务名称

#### 3.3 执行部署

运行部署脚本启动真机执行:

```bash
python example/deploy/piper_single_on_DP.py
```

## 注意事项

- 确保机械臂已正确使能并连接
- 检查模型路径是否正确
- 部署前建议先在测试环境中验证模型效果