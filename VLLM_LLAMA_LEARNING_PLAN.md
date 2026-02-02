# vLLM LLaMA推理学习计划

## 目标
以LLaMA模型为切入点，系统学习vLLM推理框架的核心技术，最终达到快速适配新大模型的能力。

---

## 一、环境配置建议

### 1.1 推荐环境方案：Conda虚拟环境（✅ 推荐）

**理由：**
- ✅ 便于断点调试和IDE集成
- ✅ 不影响宿主机其他Python环境
- ✅ 方便切换不同版本的PyTorch/CUDA
- ✅ 你已经在宿主机环境，不需要额外容器配置
- ✅ 4张4090D显卡直接访问，无需容器GPU映射

**环境搭建步骤：**

参考 [[https://docs.vllm.ai/en/latest/contributing/#license]]

```bash
# 1. It's recommended to use uv, a very fast Python environment manager, to create and manage Python environments.

## 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

## 2. create python3 env
uv venv --python 3.12 --seed
source .venv/bin/activate

## 3. I need to develop vLLM's Python and CUDA/C++ code,so:

## 需要先安装系统 Python 开发头文件
sudo apt-get update
sudo apt-get install python3.12-dev build-essential

uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129

## 有些依赖，如nvidia依赖直接从github上下载，因此最好开代理
export http_proxy=http://192.168.11.6:7890
export https_proxy=http://192.168.11.6:7890
uv pip install -e . --no-build-isolation


```

### 1.2 IDE配置建议

**VSCode配置（推荐）：**

```json
// .vscode/launch.json - 用于断点调试
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: vLLM Offline Inference",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/examples/offline_inference/basic/basic.py",
            "console": "integratedTerminal",
            "justMyCode": false,  // 允许调试vLLM内部代码
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",  // 开始时用单卡调试
                "VLLM_LOGGING_LEVEL": "DEBUG"
            }
        },
        {
            "name": "Python: vLLM LLaMA Inference",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/my_debug_scripts/llama_debug.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "VLLM_LOGGING_LEVEL": "DEBUG"
            }
        }
    ]
}
```

---

## 二、学习路线（3个阶段）

### 阶段1：基础入门 - 理解推理流程（1-2周）

#### 目标
掌握vLLM的基本架构和LLaMA模型的推理流程。

#### 1.1 第一个可运行示例

创建 `my_debug_scripts/step1_basic_llama.py`：

```python
from vllm import LLM, SamplingParams

# 使用小模型快速验证
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 或使用本地路径

# 创建LLM实例
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,  # 先用单卡
    gpu_memory_utilization=0.5,
    max_model_len=2048,
    trust_remote_code=True
)

# 简单推理
prompts = ["Hello, my name is", "The capital of China is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

**调试任务：**
1. 在 `vllm/__init__.py` 的 `LLM.__init__()` 设置断点
2. 跟踪模型加载过程：`vllm/model_executor/models/llama.py` 中的 `LlamaForCausalLM` 类
3. 理解关键组件初始化顺序

#### 1.2 核心代码跟踪路径

**推理流程关键文件：**

```
用户调用
  ↓
vllm/__init__.py::LLM.generate()
  ↓
vllm/v1/engine/llm_engine.py::LLMEngine
  ↓
vllm/v1/worker/gpu_worker.py::Worker
  ↓
vllm/model_executor/models/llama.py::LlamaForCausalLM
  ↓
vllm/attention/layer.py::Attention (PagedAttention核心)
```

**学习检查点：**
- [ ] 理解 `LLM` 类的初始化流程
- [ ] 找到模型权重加载位置（`model_executor/model_loader/`）
- [ ] 理解 `SamplingParams` 如何影响生成策略
- [ ] 追踪一个token的生成过程

---

### 阶段2：核心技术深入（3-4周）

#### 2.1 PagedAttention机制

**核心文件：**
- `vllm/attention/layer.py` - Attention层实现
- `csrc/attention/` - CUDA kernel实现
- `vllm/_custom_ops.py` - Python到C++的桥接

**调试脚本** `my_debug_scripts/step2_paged_attention.py`：

```python
from vllm import LLM, SamplingParams
import torch

# 开启详细日志
import os
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    block_size=16,  # PagedAttention的block大小
    max_num_seqs=8,  # 并发处理序列数
    enable_prefix_caching=True,  # 开启prefix caching
)

# 测试不同长度的序列
prompts = [
    "Write a short story: " * 10,  # 长prompt
    "Hello",  # 短prompt
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

**学习任务：**
1. **理解PagedAttention原理：**
   - 在 `vllm/attention/layer.py::Attention.forward()` 设置断点
   - 观察KV cache如何分块管理
   - 理解 `block_table` 的作用

2. **追踪CUDA Kernel：**
   - 查看 `csrc/attention/attention_kernels.cu`
   - 理解paged_attention_v1/v2的区别

3. **内存管理：**
   - 研究 `vllm/v1/core/scheduler.py` 中的调度逻辑
   - 理解block分配和回收机制

**检查点：**
- [ ] 画出PagedAttention的内存布局图
- [ ] 解释为什么PagedAttention能提高吞吐量
- [ ] 理解block_size参数的影响

#### 2.2 Continuous Batching（持续批处理）

**核心文件：**
- `vllm/v1/core/scheduler.py` - 调度器核心
- `vllm/v1/engine/llm_engine.py` - 请求处理

**调试脚本** `my_debug_scripts/step2_continuous_batching.py`：

```python
import asyncio
from vllm import AsyncLLM, SamplingParams

async def test_continuous_batching():
    llm = AsyncLLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        tensor_parallel_size=1,
        max_num_seqs=16,  # 增大并发数
    )
    
    # 模拟陆续到达的请求
    prompts = [f"Task {i}: " for i in range(20)]
    
    async def generate_with_delay(prompt, delay):
        await asyncio.sleep(delay)
        async for output in llm.generate(prompt, SamplingParams(max_tokens=50)):
            print(f"[{delay}s] {output.outputs[0].text[:50]}...")
    
    # 请求在不同时间到达
    tasks = [generate_with_delay(p, i*0.5) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)

asyncio.run(test_continuous_batching())
```

**学习任务：**
1. 在 `scheduler.py::schedule()` 设置断点
2. 观察新请求如何插入正在执行的batch
3. 理解prefill和decode阶段的区别

**检查点：**
- [ ] 解释prefill vs decode的区别
- [ ] 理解chunked prefill的作用
- [ ] 绘制continuous batching的时序图

#### 2.3 张量并行（Tensor Parallelism）

**核心文件：**
- `vllm/distributed/parallel_state.py` - 并行状态管理
- `vllm/model_executor/layers/linear.py` - 并行线性层
- `vllm/model_executor/models/llama.py` - LLaMA的并行实现

**调试脚本** `my_debug_scripts/step2_tensor_parallel.py`：

```python
from vllm import LLM, SamplingParams

# 使用2张GPU进行张量并行
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    tensor_parallel_size=2,  # 2张卡TP
    gpu_memory_utilization=0.9,
)

prompts = ["Explain tensor parallelism in deep learning:"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=200))

print(outputs[0].outputs[0].text)
```

**学习任务：**
1. **理解层切分：**
   - 在 `vllm/model_executor/layers/linear.py::ColumnParallelLinear` 设置断点
   - 观察权重如何按列切分
   - 理解all-reduce操作的时机

2. **追踪通信：**
   - 查看 `vllm/distributed/communication_op.py`
   - 理解什么时候需要GPU间通信

**检查点：**
- [ ] 画出LLaMA模型在TP=2时的切分方式
- [ ] 理解QKV projection为何用ColumnParallel
- [ ] 理解output projection为何用RowParallel

#### 2.4 KV Cache量化

**调试脚本** `my_debug_scripts/step2_kv_quantization.py`：

```python
from vllm import LLM, SamplingParams

# 对比FP16 vs INT8 KV cache
configs = [
    ("FP16 KV Cache", {"kv_cache_dtype": "auto"}),
    ("INT8 KV Cache", {"kv_cache_dtype": "int8"}),
]

for name, kwargs in configs:
    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        **kwargs
    )
    
    prompts = ["Write a story: "] * 10
    outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
    print(f"{name}: Generated {len(outputs)} sequences")
```

**学习任务：**
1. 在 `vllm/attention/layer.py` 中找到KV量化代码
2. 理解量化如何节省显存
3. 测试不同量化方式的精度影响

---

### 阶段3：模型适配实战（2-3周）

#### 3.1 从头适配一个LLaMA变体

**任务：为一个新的LLaMA变体添加支持（如Llama-3.3或自定义改进）**

**适配步骤模板：**

1. **创建模型文件** `vllm/model_executor/models/my_llama.py`：

```python
# 参考 vllm/model_executor/models/llama.py
from vllm.model_executor.models.llama import (
    LlamaForCausalLM,
    LlamaAttention,
    LlamaMLP,
)

class MyLlamaAttention(LlamaAttention):
    """自定义Attention修改"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加自定义初始化
    
    def forward(self, *args, **kwargs):
        # 添加自定义逻辑
        return super().forward(*args, **kwargs)

class MyLlamaForCausalLM(LlamaForCausalLM):
    """自定义LLaMA模型"""
    pass
```

2. **注册模型** - 修改 `vllm/model_executor/models/__init__.py`：

```python
"MyLlamaForCausalLM": ("my_llama", "MyLlamaForCausalLM"),
```

3. **添加配置支持** - 在 `vllm/transformers_utils/config.py` 中注册

4. **编写测试**：

```python
# tests/models/test_my_llama.py
import pytest
from vllm import LLM, SamplingParams

def test_my_llama_basic():
    llm = LLM(model="path/to/my-llama", trust_remote_code=True)
    outputs = llm.generate("Hello", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
```

**实战练习：**
- [ ] 适配一个带有GQA（Grouped Query Attention）的模型
- [ ] 添加自定义RoPE（Rotary Position Embedding）变体
- [ ] 支持不同的MLP激活函数

#### 3.2 理解权重加载机制

**核心文件：**
- `vllm/model_executor/model_loader/loader.py`
- `vllm/model_executor/model_loader/weight_utils.py`

**调试任务：**
1. 在 `loader.py::load_model()` 设置断点
2. 理解HuggingFace权重到vLLM的映射
3. 学习如何处理权重格式转换

#### 3.3 性能优化与Profiling

**Profiling脚本** `my_debug_scripts/step3_profiling.py`：

```python
import torch
from vllm import LLM, SamplingParams
from vllm.profiler import LayerwiseProfiler

# 启用profiling
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    tensor_parallel_size=2,
)

# 使用PyTorch Profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    outputs = llm.generate(
        ["Write a long story: "] * 10,
        SamplingParams(max_tokens=512)
    )

# 输出性能报告
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")  # 在chrome://tracing查看
```

**Nsys Profiling（NVIDIA工具）：**

```bash
# 使用Nsys进行详细的GPU分析
nsys profile -o llama_profile \
    --trace=cuda,nvtx,osrt \
    python my_debug_scripts/step1_basic_llama.py

# 查看报告
nsys-ui llama_profile.nsys-rep
```

---

## 三、关键技术清单（需要掌握）

### 3.1 核心概念
- [x] **PagedAttention**: 分块KV缓存管理
- [x] **Continuous Batching**: 动态批处理调度
- [x] **Tensor Parallelism**: 跨GPU模型切分
- [x] **Chunked Prefill**: 分块预填充
- [ ] **Prefix Caching**: 公共前缀复用
- [ ] **Speculative Decoding**: 推测解码加速

### 3.2 模型架构理解
- [ ] Transformer架构（Self-Attention, MLP）
- [ ] RoPE位置编码
- [ ] GQA（Grouped Query Attention）
- [ ] RMSNorm vs LayerNorm
- [ ] SwiGLU激活函数

### 3.3 系统优化
- [ ] CUDA Kernel编写基础
- [ ] FP8/INT8量化
- [ ] FlashAttention集成
- [ ] 内存池管理
- [ ] 通信优化（NCCL）

---

## 四、调试技巧与工具

### 4.1 调试命令

```bash
# 1. 单步调试
python -m pdb my_debug_scripts/step1_basic_llama.py

# 2. 使用ipdb（增强版pdb）
import ipdb; ipdb.set_trace()  # 在代码中插入

# 3. 查看GPU内存
nvitop  # 实时监控

# 4. 环境变量调试
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1  # 函数调用追踪
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA调用，便于调试

# 5. 只使用部分GPU
export CUDA_VISIBLE_DEVICES=0,1
```

### 4.2 常用断点位置

```python
# 推理入口
vllm/__init__.py::LLM.generate()

# 模型前向传播
vllm/model_executor/models/llama.py::LlamaForCausalLM.forward()

# Attention计算
vllm/attention/layer.py::Attention.forward()

# 调度器
vllm/v1/core/scheduler.py::Scheduler.schedule()

# 权重加载
vllm/model_executor/model_loader/loader.py::load_model()
```

### 4.3 日志分析

vLLM的日志包含大量调试信息：

```python
# 在代码中添加详细日志
from vllm.logger import init_logger
logger = init_logger(__name__)

logger.debug("Detailed debug info")
logger.info("Important info")
logger.warning("Warning message")
```

---

## 五、学习资源

### 5.1 必读论文
1. **vLLM原论文**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention"
3. **Tensor Parallelism**: "Megatron-LM: Training Multi-Billion Parameter Language Models"

### 5.2 代码导读顺序

```
第一周：
├── vllm/__init__.py                 # 入口API
├── examples/offline_inference/basic/basic.py  # 示例
└── vllm/model_executor/models/llama.py       # LLaMA实现

第二周：
├── vllm/attention/layer.py          # PagedAttention
├── vllm/v1/engine/llm_engine.py    # 引擎
└── vllm/v1/core/scheduler.py        # 调度器

第三周：
├── vllm/distributed/                # 分布式
├── vllm/model_executor/layers/      # 并行层
└── csrc/                            # CUDA kernels

第四周：
├── vllm/model_executor/model_loader/ # 权重加载
├── vllm/config/                     # 配置系统
└── tests/models/                    # 测试样例
```

### 5.3 社区资源
- **官方文档**: https://docs.vllm.ai
- **GitHub Issues**: 查看常见问题和解决方案
- **Discord/Slack**: 加入vLLM社区讨论

---

## 六、进度跟踪清单

### Week 1-2: 基础入门
- [ ] 环境搭建完成
- [ ] 运行第一个LLaMA推理示例
- [ ] 理解LLM类的初始化流程
- [ ] 追踪完整的token生成过程
- [ ] 阅读`llama.py`核心代码

### Week 3-4: PagedAttention
- [ ] 理解PagedAttention原理
- [ ] 调试KV cache管理
- [ ] 绘制内存布局图
- [ ] 分析block_size的影响

### Week 5-6: Continuous Batching
- [ ] 理解调度器工作原理
- [ ] 区分prefill和decode
- [ ] 测试并发请求处理
- [ ] 理解chunked prefill

### Week 7-8: 分布式推理
- [ ] 理解Tensor Parallelism
- [ ] 调试2卡TP推理
- [ ] 分析通信开销
- [ ] 测试4卡TP（利用你的4张4090）

### Week 9-10: 模型适配
- [ ] 阅读模型注册机制
- [ ] 适配一个LLaMA变体
- [ ] 添加自定义层
- [ ] 编写单元测试

### Week 11-12: 优化与调优
- [ ] 使用Profiler分析性能
- [ ] 理解CUDA kernel
- [ ] 测试不同量化方案
- [ ] 优化内存占用

---

## 七、实战项目建议

完成学习后，挑战以下项目来巩固技能：

1. **项目1**: 为Llama-3.3添加完整支持（如果vLLM还未支持）
2. **项目2**: 实现自定义的Attention变体（如Linear Attention）
3. **项目3**: 优化小batch推理的latency
4. **项目4**: 为特定硬件（如4090）优化kernel参数

---

## 八、常见问题速查

### Q1: 显存溢出 (OOM)
```python
# 降低gpu_memory_utilization
llm = LLM(model=..., gpu_memory_utilization=0.7)  # 默认0.9

# 减少max_num_seqs
llm = LLM(model=..., max_num_seqs=64)  # 降低并发

# 使用KV cache量化
llm = LLM(model=..., kv_cache_dtype="int8")
```

### Q2: 多卡不均衡
```bash
# 检查NCCL
export NCCL_DEBUG=INFO

# 使用nvlink topology
nvidia-smi topo -m
```

### Q3: 推理速度慢
```python
# 启用prefix caching
llm = LLM(model=..., enable_prefix_caching=True)

# 增大batch size
outputs = llm.generate(prompts * 10, ...)  # 更大batch

# 使用更大block_size
llm = LLM(model=..., block_size=32)  # 默认16
```

---

## 九、总结

这个学习计划的核心思路是：
1. **由浅入深**：从简单示例到复杂机制
2. **理论结合实践**：每个概念都有调试脚本验证
3. **以终为始**：目标是适配新模型，所以重点在架构理解

**学习节奏建议**：
- 每天2-3小时深度学习
- 一周完成一个阶段
- 边学边记录笔记和踩坑经验

**成功标准**：
- 能独立为新的LLaMA变体添加vLLM支持
- 理解vLLM的5大核心技术（PagedAttention, Continuous Batching, TP, Chunked Prefill, KV Quantization）
- 能分析和解决常见的性能/显存问题

祝学习顺利！有任何问题随时在代码中设断点深入探索。
