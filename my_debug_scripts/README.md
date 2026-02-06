# vLLM 学习调试脚本

本目录包含按学习阶段组织的调试脚本，用于系统学习 vLLM 框架。

## 📁 目录结构

```
my_debug_scripts/
├── README.md                          # 本文件
├── step1_basic_llama.py              # 阶段1: 基础推理示例
├── step2_paged_attention.py          # 阶段2: PagedAttention 机制（待创建）
├── step2_continuous_batching.py      # 阶段2: Continuous Batching（待创建）
├── step2_tensor_parallel.py          # 阶段2: 张量并行（待创建）
├── step2_kv_quantization.py          # 阶段2: KV Cache 量化（待创建）
└── step3_profiling.py                # 阶段3: 性能分析（待创建）
```

## 🚀 使用方法

### 方法 1: VSCode 调试（推荐）

1. 打开 VSCode
2. 按 `F5` 或点击左侧调试图标
3. 选择对应的调试配置（如 "🚀 Step1: 基础 LLaMA 推理"）
4. 开始调试

### 方法 2: 命令行运行

```bash
# 激活环境
source .venv/bin/activate

# 运行脚本
python my_debug_scripts/step1_basic_llama.py

# 或使用特定 GPU
CUDA_VISIBLE_DEVICES=0 python my_debug_scripts/step1_basic_llama.py
```

### 方法 3: 使用 ipdb 交互式调试

```bash
# 在脚本中添加断点
import ipdb; ipdb.set_trace()

# 运行
python my_debug_scripts/step1_basic_llama.py
```

## 🎯 学习路径

### 阶段 1: 基础入门（当前）

- [x] `step1_basic_llama.py` - 理解基本推理流程
- [ ] 跟踪 LLM 初始化
- [ ] 理解模型加载过程
- [ ] 追踪 token 生成流程

**关键断点位置：**
1. `vllm/__init__.py::LLM.__init__()`
2. `vllm/__init__.py::LLM.generate()`
3. `vllm/model_executor/models/llama.py::LlamaForCausalLM.forward()`

### 阶段 2: 核心技术（计划中）

- [ ] PagedAttention 机制
- [ ] Continuous Batching
- [ ] Tensor Parallelism
- [ ] KV Cache 量化

### 阶段 3: 高级优化（计划中）

- [ ] 性能分析
- [ ] 模型适配
- [ ] 自定义优化

## 🔍 调试技巧

### 常用环境变量

```bash
# 日志级别
export VLLM_LOGGING_LEVEL=DEBUG

# 函数调用追踪
export VLLM_TRACE_FUNCTION=1

# 同步 CUDA 调用（便于调试）
export CUDA_LAUNCH_BLOCKING=1

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0
```

### 常用调试命令

```python
# 在代码中插入断点
import ipdb; ipdb.set_trace()

# 或使用内置 pdb
import pdb; pdb.set_trace()

# 查看对象信息
dir(obj)        # 列出对象的所有属性
type(obj)       # 查看对象类型
vars(obj)       # 查看对象的 __dict__
help(obj)       # 查看帮助文档
```

### VSCode 调试快捷键

- `F5`: 启动/继续
- `F9`: 切换断点
- `F10`: 单步跳过
- `F11`: 单步进入
- `Shift+F11`: 单步跳出
- `Shift+F5`: 停止调试

## 📝 注意事项

1. **显存要求**：step1 使用 1B 模型，约需 1.5GB 显存
2. **网络要求**：首次运行会从 HuggingFace 下载模型
3. **环境要求**：确保已安装 vLLM 和 PyTorch CUDA 版本

## 🆘 常见问题

### Q0: VSCode 单步调试卡住（模型加载后不往下走）？

现象：从 `main()` 入口开始单步（F10/F11），在 `LLM(...)` 初始化完成后继续单步会“卡住”，看起来不再往下执行（有时光标停在 `logger.info(...)` 附近）。

原因：vLLM v1 默认会启用多进程引擎，主进程需要后台线程与子进程通信；而 VSCode/debugpy 单步时可能只恢复主线程/暂停通信线程，导致主进程与子进程互相等待，从而“卡住”（跟是否使用 vLLM 的 logger 关系不大）。

最终解决方法（本仓库已配置好）：
- 强制调试时使用单进程引擎：在 `.vscode/launch.json` 里设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，并在 `step1_basic_llama.py` 里做了兜底设置。
- 在 `.vscode/launch.json` 开启 `steppingResumesAllThreads: true`，减少“单步只跑主线程”的情况。
- 修改配置后需要 **完全重启** 调试会话（`Shift+F5` 后再 `F5`），确保不会残留旧的子进程。

调试建议：
- 不要从入口一路单步；在 `LLM.__init__()` / `LLM.generate()` 等关键位置打断点后用 `F5` 跳转更稳定。

### Q1: 模型下载慢？

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: 显存不足？

```python
# 调整 gpu_memory_utilization
llm = LLM(model=..., gpu_memory_utilization=0.3)
```

### Q3: 找不到模块？

```bash
# 确保已安装 vLLM
pip install -e .

# 或设置 PYTHONPATH
export PYTHONPATH=/home/xcd/llm/vllm:$PYTHONPATH
```

## 📚 参考资源

- [vLLM 官方文档](https://docs.vllm.ai)
- [学习计划文档](../VLLM_LLAMA_LEARNING_PLAN.md)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
