#!/bin/bash
# vLLM 国内加速安装脚本
# 适用于中国大陆网络环境

echo "=========================================="
echo "vLLM 国内加速安装脚本"
echo "=========================================="

# 方案1：使用清华大学镜像源（推荐）
echo ""
echo "方案1: 使用清华大学 PyPI 镜像源"
echo "------------------------------------------"
echo "临时使用（仅本次安装）："
echo "pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple"
echo ""
echo "永久配置（推荐）："
echo "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
echo ""

# 方案2：使用阿里云镜像源
echo "方案2: 使用阿里云 PyPI 镜像源"
echo "------------------------------------------"
echo "pip install -e . -i https://mirrors.aliyun.com/pypi/simple/"
echo ""

# 方案3：使用中科大镜像源
echo "方案3: 使用中科大 PyPI 镜像源"
echo "------------------------------------------"
echo "pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple/"
echo ""

# 方案4：使用豆瓣镜像源
echo "方案4: 使用豆瓣 PyPI 镜像源"
echo "------------------------------------------"
echo "pip install -e . -i https://pypi.douban.com/simple/"
echo ""

echo "=========================================="
echo "推荐安装步骤（清华源 + 并行下载）"
echo "=========================================="

# 检查是否在 conda 环境中
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  警告: 未检测到激活的 conda 环境"
    echo "请先运行: conda activate vllm-dev"
    exit 1
fi

echo "✓ 当前 conda 环境: $CONDA_DEFAULT_ENV"
echo ""

# 询问用户选择
read -p "是否现在配置 pip 镜像源并安装 vLLM？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "步骤 1/4: 配置 pip 镜像源..."
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
    
    echo ""
    echo "步骤 2/4: 升级 pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    echo ""
    echo "步骤 3/4: 预先安装大依赖包（PyTorch 等）..."
    echo "这一步可能需要较长时间，请耐心等待..."
    
    # 检查 CUDA 版本
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        echo "检测到 CUDA 版本: $CUDA_VERSION"
        
        if [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l) -eq 1 ]]; then
            echo "使用 CUDA 11.8 版本的 PyTorch (最新稳定版 2.7.1)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            echo "使用 CUDA 11.7 版本的 PyTorch (最新稳定版 2.7.1)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
        fi
    else
        echo "未检测到 NVIDIA GPU，安装 CPU 版本 PyTorch..."
        pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    
    echo ""
    echo "步骤 4/4: 安装 vLLM (可编辑模式)..."
    # 使用清华源安装，并增加超时时间
    pip install -e . \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --timeout 300 \
        --retries 5
    
    echo ""
    echo "=========================================="
    echo "✓ 安装完成！"
    echo "=========================================="
    echo ""
    echo "验证安装："
    echo "python -c 'import vllm; print(vllm.__version__)'"
    echo ""
else
    echo "取消安装。你可以手动运行以下命令："
    echo "pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple"
fi
