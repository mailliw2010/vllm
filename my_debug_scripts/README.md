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

### Q0: VSCode 单步调试卡在 `logger.info(...)` 不动？

现象：从 `main()` 入口开始单步（F10/F11），在模型加载后可能卡在某一行 `logger.info(...)`，看起来“无法执行下去”。

原因：vLLM 初始化会启动后台线程并打印日志；而 VSCode/debugpy 在“单步”时可能只恢复当前线程，其他线程保持暂停。若后台线程正好暂停在写日志（持有日志 Handler 的锁），当前线程再执行 `logger.info()` 会阻塞，从而卡住。

解决：
- 优先用 `F5`（Continue）跑到下一个断点，而不是从入口一直单步。
- 把断点打在你关心的 vLLM 函数上（如 `vllm/__init__.py::LLM.__init__()`、`LLM.generate()`），再 `F5` 跳过去。
- 如果已经卡住，尝试在 VSCode 的 `Threads` 视图里切换线程并继续/恢复全部线程。
- 已在 `step1_basic_llama.py` 增加调试保护：检测到调试器时，本脚本日志写入 `my_debug_scripts/.logs/step1_basic_llama.debug.log`，避免与 vLLM 的控制台日志抢同一把 stdout/stderr 锁。
- 已在 `step1_basic_llama.py` 增加调试保护：检测到调试器且未显式设置 `VLLM_ENABLE_V1_MULTIPROCESSING` 时，默认设为 `0`（单进程引擎），避免主进程打断点时通信线程暂停导致与子进程互相等待。
- 已在 `.vscode/launch.json` 增加 `steppingResumesAllThreads: true`（如果你的 VSCode/Python 插件版本支持，会明显改善多线程单步体验）。
- 已在 `.vscode/launch.json` 增加 `subProcess: true`（用于需要时把 vLLM 的子进程也纳入调试，避免“主进程暂停但子进程继续跑”）。

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
