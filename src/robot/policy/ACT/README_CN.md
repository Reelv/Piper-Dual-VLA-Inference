[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# ACT (Action Chunking Transformer) 模型部署指南

## 快速开始

### 1. 环境配置

安装 ACT 部署所需的依赖环境:

```bash
cd policy/ACT/
pip install -r requirements.txt
```

### 2. 数据准备与训练

#### 数据转换

将采集的数据转换为 ACT 模型所需的 HDF5 格式:

```bash
python scripts/convert2act_hdf5.py <input_data_path> <output_path>
```

**示例:**
```bash
# 将 save/test/ 目录下的数据转换为 ACT 格式
python scripts/convert2act_hdf5.py ./save/test/ ~/RoboTwin/policy/ACT/data/

# 转换 pick_place_cup 任务数据
python scripts/convert2act_hdf5.py save/pick_place_cup /path/to/output
```

### 3. 真机部署


1. 将训练好的 checkpoint 复制到以下目录:
```
control_your_robot/policy/ACT/actckpt/
```

2. 修改部署脚本 `example/deploy/piper_single_on_ACT.py`:

```python
# 在第 120 行左右修改模型路径
model = MYACT("/path/your/policy/ACT/act_ckpt/act-pick_place_cup/100", "act-pick_place_cup")
```

**参数说明:**
- 第一个参数: Policy 模型的文件夹地址
- 第二个参数: 对应的任务名称

#### 3.3 执行部署

运行部署脚本启动真机执行:

```bash
python example/deploy/piper_single_on_ACT.py
```

## 注意事项

- 确保机械臂已正确使能并连接
- 检查模型路径是否正确
- 部署前建议先在测试环境中验证模型效果
