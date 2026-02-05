#!/usr/bin/env python3
"""
vLLM LLaMA 基础推理示例
功能：使用 LLaMA 小模型进行简单的文本生成
调试重点：理解 LLM 初始化和推理流程
"""

import os
import logging
import sys

# VSCode/debugpy 打断点时会暂停当前进程内的所有线程。
# vLLM v1 默认启用多进程引擎：主进程依赖后台线程与子进程通信。
# 若你在主进程打断点并单步，这些通信线程会被暂停，容易出现“卡住”的假象。
# 调试时默认切到单进程模式，避免子进程继续跑而主进程线程暂停造成的死锁。
if sys.gettrace() is not None and os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING") is None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

def _configure_script_logger() -> logging.Logger:
    """为当前脚本配置日志。

    说明：VSCode/debugpy 单步时，如果后台线程暂停在写 stdout/stderr，
    当前线程再写同一 stream 可能被 TextIO 的锁卡住。调试时把本脚本日志
    单独写入文件，避免与 vLLM 的控制台日志竞争同一把 I/O 锁。
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if sys.gettrace() is None:
        logging.basicConfig(level=logging.DEBUG)
        return logger

    # 调试器存在：脚本日志写文件，避免卡在 logger.info()
    from pathlib import Path

    log_dir = Path(__file__).resolve().parent / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "step1_basic_llama.debug.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _configure_script_logger()

def _avoid_vscode_step_deadlock_for_logging() -> None:
    """避免 VSCode 单步调试时，因多线程日志锁导致卡在 logger.info()。

    vLLM 初始化会启动后台线程并打印日志。VSCode/debugpy 在“单步”时
    可能只恢复当前线程，其他线程保持暂停；如果其他线程正持有日志 Handler 锁，
    当前线程调用 logger.info() 会阻塞，看起来像“卡在这一行”。
    """
    if sys.gettrace() is None:
        return
    try:
        logging.Handler.createLock = lambda self: None  # type: ignore[method-assign]
        root = logging.getLogger()
        for handler in root.handlers:
            handler.lock = None
    except Exception:
        # 仅为调试体验的兜底，不影响正常运行
        pass


_avoid_vscode_step_deadlock_for_logging()

def main():
    logger.info("=" * 60)
    logger.info("vLLM LLaMA 基础推理示例启动")
    logger.info("=" * 60)
    
    # 使用小模型快速验证
    # ===== 推荐用 LLaMA 架构的模型学习 vLLM =====
    # 环境变量 HF_HUB_OFFLINE=1 已在 launch.json 中配置
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # 方案2: 直接使用本地缓存路径（也可行）
    # model_name = os.path.expanduser(
    #     "~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    # )
    
    logger.info(f"正在加载模型: {model_name}")
    logger.info("提示：在下面这行设置断点，查看 LLM 初始化过程")
    
    # === 调试断点位置 1: LLM 初始化 ===
    # 建议在此处设置断点，进入 vllm/__init__.py::LLM.__init__()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,        # 单卡推理
        gpu_memory_utilization=0.3,    # 降低显存占用（其他进程在用GPU）
        max_model_len=2048,            # 最大序列长度
        trust_remote_code=True,        # 信任远程代码
        dtype="half",                  # 使用 FP16 精度
    )
    
    logger.info("模型加载完成！")
    
    # 准备测试 prompts
    prompts = [
        "Hello, my name is",
        "The capital of China is",
        "写一首关于人工智能的诗：",
    ]
    
    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,     # 温度：控制随机性，越高越随机
        top_p=0.95,         # 核采样：保留累积概率达到 top_p 的 tokens
        max_tokens=50,      # 最大生成 token 数
        # 其他可选参数：
        # top_k=40,         # Top-K 采样
        # repetition_penalty=1.1,  # 重复惩罚
    )
    
    logger.info(f"开始推理，共 {len(prompts)} 个 prompt")
    logger.info("提示：在下面这行设置断点，查看 generate() 流程")
    
    # === 调试断点位置 2: 生成过程 ===
    # 建议在此处设置断点，进入 vllm/__init__.py::LLM.generate()
    outputs = llm.generate(prompts, sampling_params)
    
    logger.info("\n" + "=" * 60)
    logger.info("推理结果：")
    logger.info("=" * 60)
    
    # 打印结果
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        
        print(f"\n[示例 {i}]")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Tokens: {tokens}")
        print("-" * 60)
    
    logger.info("\n推理完成！")
    
    # 显示性能统计（如果可用）
    if hasattr(llm, 'get_stats'):
        stats = llm.get_stats()
        logger.info(f"性能统计: {stats}")

if __name__ == "__main__":
    # === 调试提示 ===
    # 1. 在 VSCode 中按 F5 启动调试
    # 2. 或在终端运行: python my_debug_scripts/step1_basic_llama.py
    # 3. 关键断点位置：
    #    - LLM.__init__() : vllm/__init__.py
    #    - LLM.generate() : vllm/__init__.py
    #    - LlamaForCausalLM.forward() : vllm/model_executor/models/llama.py
    
    main()
