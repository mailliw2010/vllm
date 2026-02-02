# vLLM 国内安装加速指南

## 问题分析
在国内网络环境下，`pip install -e .` 安装 vLLM 慢的主要原因：
1. PyPI 官方源（pypi.org）在国内访问慢
2. PyTorch 等大型依赖包下载耗时长
3. GitHub 相关资源访问受限

## 解决方案

### 方案1：使用国内 PyPI 镜像源（推荐⭐）

#### 临时使用（仅本次安装）
```bash
# 清华大学镜像（推荐，更新最快）
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里云镜像
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/

# 中科大镜像
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple/

# 豆瓣镜像
pip install -e . -i https://pypi.douban.com/simple/
```

#### 永久配置（推荐⭐⭐⭐）
```bash
# 配置清华源为默认源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 查看配置
pip config list

# 之后所有 pip install 都会自动使用清华源
pip install -e .
```

### 方案2：分步安装大型依赖

vLLM 的主要耗时在于 PyTorch 等大型依赖，可以先单独安装：

```bash
# 1. 先安装 PyTorch（使用官方源，不指定版本会自动安装最新稳定版）
# 对于 CUDA 11.8（4090 推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者从清华源安装（如果上面的慢）
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 再安装 vLLM
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方案3：使用 Conda 安装 PyTorch

Conda 在国内也有镜像源，可以更快：

```bash
# 配置 conda 清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

# 使用 conda 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 再用 pip 安装 vLLM
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方案4：增加并行下载和超时设置

```bash
pip install -e . \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 300 \
    --retries 5 \
    --no-cache-dir
```

参数说明：
- `--timeout 300`: 超时时间设为300秒
- `--retries 5`: 失败后重试5次
- `--no-cache-dir`: 不使用缓存（节省空间）

### 方案5：离线安装（网络极差时）

如果网络实在太差，可以：

1. **下载依赖包到本地**：
```bash
# 在网络好的机器上
pip download -r requirements/build.txt -d ./packages -i https://pypi.tuna.tsinghua.edu.cn/simple

# 传输到目标机器后
pip install --no-index --find-links=./packages -e .
```

2. **使用预编译的 wheel**：
```bash
# 从清华源下载 vLLM wheel（如果有）
pip download vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 完整安装流程（国内优化版）

```bash
# 1. 激活环境
conda activate vllm-dev

# 2. 永久配置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 3. 升级基础工具
pip install --upgrade pip setuptools wheel

# 4. 先安装 PyTorch（最耗时的部分）
# 方式 A: 使用 PyTorch 官方源（通常更快，自动安装最新稳定版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 方式 B: 如果上面慢，用清华源
# pip install torch torchvision torchaudio

# 5. 安装 vLLM（可编辑模式）
cd /home/xcd/llm/vllm
pip install -e . --timeout 300

# 6. 验证安装
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## 镜像源速度对比

根据经验，在国内的速度排名（仅供参考）：

1. **清华大学镜像** ⭐⭐⭐⭐⭐ - 更新最及时，速度快
2. **阿里云镜像** ⭐⭐⭐⭐ - 稳定，企业级
3. **中科大镜像** ⭐⭐⭐⭐ - 教育网友好
4. **豆瓣镜像** ⭐⭐⭐ - 较早，但更新较慢

## 常见问题

### Q1: 配置镜像源后仍然很慢？
```bash
# 检查配置是否生效
pip config list

# 检查网络连接
ping pypi.tuna.tsinghua.edu.cn

# 尝试清除缓存
pip cache purge
```

### Q2: 提示 SSL 证书错误？
```bash
# 临时跳过 SSL 验证（不推荐）
pip install -e . --trusted-host pypi.tuna.tsinghua.edu.cn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者永久配置
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
```

### Q3: 某些包在镜像源找不到？
```bash
# 使用多个镜像源
pip install -e . \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.org/simple
```

### Q4: 编译 CUDA 扩展很慢？
```bash
# vLLM 需要编译 CUDA 扩展，这部分无法通过镜像加速
# 确保安装了正确的编译工具
conda install cmake ninja

# 查看编译进度
export VLLM_INSTALL_VERBOSE=1
pip install -e .
```

## 推荐配置文件

创建 `~/.pip/pip.conf`（Linux）或 `%APPDATA%\pip\pip.ini`（Windows）：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 300

[install]
retries = 5
```

## 估算时间

在国内网络环境下，使用镜像源后的预期安装时间：

- **不使用镜像源**: 30-60 分钟（经常失败）
- **使用镜像源**: 10-20 分钟
- **预先安装 PyTorch + 镜像源**: 5-10 分钟

其中编译 CUDA 扩展大约需要 3-5 分钟（取决于 CPU 性能）。

## 验证安装成功

```bash
# 检查 vLLM 版本
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# 检查 CUDA 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')"

# 快速测试
python -c "from vllm import LLM; print('vLLM import successful!')"
```

## 一键安装脚本

我已经为你准备了一键安装脚本 `install_china.sh`，直接运行：

```bash
chmod +x install_china.sh
./install_china.sh
```
