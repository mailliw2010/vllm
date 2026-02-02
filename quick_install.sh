#!/bin/bash
# 快速安装 vLLM - 国内优化版
# 适用于已激活 conda 环境的情况

set -e  # 遇到错误立即停止

echo "=========================================="
echo "vLLM 快速安装（国内加速）"
echo "=========================================="
echo ""

# 检查 conda 环境
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "❌ 错误: 未检测到激活的 conda 环境"
    echo "请先运行: conda activate vllm-dev"
    exit 1
fi

echo "✓ 当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 步骤1: 配置 pip 镜像源
echo "步骤 1/5: 配置 pip 清华镜像源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
echo "✓ 镜像源配置完成"
echo ""

# 步骤2: 升级基础工具
echo "步骤 2/5: 升级 pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ 基础工具升级完成"
echo ""

# 步骤3: 安装 PyTorch
echo "步骤 3/5: 安装 PyTorch（约需 2-5 分钟）..."
echo "注意: 将从 PyTorch 官方源下载，这是最快的方式"

# 检测 CUDA 版本
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "检测到 CUDA 版本: $CUDA_VERSION"
    echo ""
    
    # 根据 CUDA 版本选择合适的 PyTorch
    if [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l) -eq 1 ]]; then
        echo "安装支持 CUDA 11.8 的 PyTorch（最新稳定版）..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "安装支持 CUDA 11.7 的 PyTorch（最新稳定版）..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    fi
else
    echo "⚠️  未检测到 NVIDIA GPU"
    echo "安装 CPU 版本 PyTorch..."
    pip install torch torchvision torchaudio
fi

echo "✓ PyTorch 安装完成"
echo ""

# 验证 PyTorch 安装
echo "验证 PyTorch 安装..."
python -c "import torch; print(f'  PyTorch 版本: {torch.__version__}')"
python -c "import torch; print(f'  CUDA 可用: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'  GPU 数量: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'  GPU 型号: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# 步骤4: 安装其他依赖
echo "步骤 4/5: 安装 vLLM 依赖包..."
pip install ninja cmake  # 编译工具
echo "✓ 依赖包安装完成"
echo ""

# 步骤5: 安装 vLLM（可编辑模式）
echo "步骤 5/5: 安装 vLLM（可编辑模式，约需 5-10 分钟）..."
echo "这一步会编译 CUDA 扩展，请耐心等待..."
echo ""

# 显示详细安装过程
export VLLM_INSTALL_VERBOSE=1

# 从当前目录安装
pip install -e . --timeout 600 --retries 5

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""

# 验证安装
echo "验证 vLLM 安装..."
if python -c "import vllm" 2>/dev/null; then
    python -c "import vllm; print(f'  vLLM 版本: {vllm.__version__}')"
    echo "  导入测试: ✓"
    echo ""
    echo "🎉 vLLM 安装成功！"
    echo ""
    echo "快速测试命令："
    echo "  python -c \"from vllm import LLM, SamplingParams; print('OK')\""
    echo ""
    echo "开始学习："
    echo "  查看学习计划: cat VLLM_LLAMA_LEARNING_PLAN.md"
else
    echo "  ❌ vLLM 导入失败"
    echo ""
    echo "请检查错误信息并重试"
    exit 1
fi
