#!/bin/bash
# HDF5数据可视化工具启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <hdf5文件路径> [选项]"
    echo "示例: $0 data.hdf5 --save output.rrd"
    exit 1
fi

# 检查文件
if [ ! -f "$1" ] && [ ! -d "$1" ]; then
    echo "错误: 路径不存在: $1"
    exit 1
fi

# 配置渲染后端 (Vulkan优先)
if command -v vulkaninfo &> /dev/null && vulkaninfo --summary 2>&1 | grep -q "NVIDIA\|AMD\|Intel" 2>/dev/null; then
    export WGPU_BACKEND=vulkan
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
else
    export WGPU_BACKEND=gl
    export MESA_GL_VERSION_OVERRIDE=4.5
fi

# 运行可视化脚本
python "$SCRIPT_DIR/visual_hdf5_rerun.py" "$@"
