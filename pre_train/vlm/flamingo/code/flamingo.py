#!/usr/bin/env python3
"""
Flamingo 最小可运行示例（图像描述 / 视觉问答）。

一、Flamingo 原理（面向实现）
1) Flamingo 通过跨注意力层把视觉特征注入语言模型。
2) 图像由视觉编码器抽取表示，文本由语言模型编码。
3) 在多层语言网络中周期性插入视觉条件，完成图文联合生成。

新人阅读顺序（建议）
1) 先看 `build_default_args`：明确可调参数和默认值。
2) 再看 `main`：把握执行主链路（准备 -> 训练/推理 -> 导出）。
3) 最后看可视化导出函数（如 `export_artifacts`）理解输出文件。

二、代码框架（从入口到结果）
1) `build_default_args`：读取推理参数。
2) `ensure_layout_dirs`：创建统一目录结构。
3) `run_inference`：加载 Flamingo 风格模型并执行生成。
4) `export_artifacts`：导出 JSON 与可视化图。
5) `main`：串联完整流程。

用法：
  python code/flamingo.py --dry-run
  python code/flamingo.py --image /absolute/path/to/image.jpg --task vqa --question "图里有什么？"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

DEFAULT_OUTPUT_DIR = "output"


def build_default_args() -> argparse.Namespace:
    """解析命令行参数，返回 Flamingo 推理配置。"""
    parser = argparse.ArgumentParser(description="Run Flamingo-style inference and export artifacts.")
    parser.add_argument(
        "--model-id",
        default="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        help="Hugging Face 上的 Flamingo 模型名称。",
    )
    parser.add_argument(
        "--task",
        default="vqa",
        choices=["vqa", "caption"],
        help="推理任务：视觉问答或图像描述。",
    )
    parser.add_argument("--image", default=None, help="输入图片绝对路径。")
    parser.add_argument("--question", default="What is in this image?", help="VQA 问题文本。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅测试目录与导出流程，不加载模型。",
    )
    args = parser.parse_known_args([])[0]
    args.dry_run = True
    return args


def detect_device() -> torch.device:
    """选择运行设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 flamingo 目录解析，绝对路径原样使用。"""
    out = Path(output_dir)
    if not out.is_absolute():
        out = (base_dir / out).resolve()
    return out


def ensure_layout_dirs(module_dir: Path, output_arg: str) -> dict[str, Path]:
    """创建并返回标准目录布局：code/data/models/output/checkpoints。"""
    output_dir = resolve_output_dir(module_dir, output_arg)
    layout = {
        "code": module_dir / "code",
        "data": module_dir / "data",
        "models": module_dir / "models",
        "output": output_dir,
        "checkpoints": module_dir / "checkpoints",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _build_prompt(task: str, question: str) -> str:
    """构造 Flamingo 风格提示词。"""
    if task == "caption":
        user_msg = "Describe this image."
    else:
        user_msg = question.strip()
    return f"<image>Question: {user_msg} Answer:"


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """将 processor 输出中的 Tensor 移到目标设备。"""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_inference(
    model_id: str,
    task: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> dict[str, Any]:
    """执行 Flamingo 推理并返回结构化结果。"""
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Missing dependency `Pillow`: {exc}") from exc

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        raise RuntimeError(f"Missing dependency `transformers`: {exc}") from exc

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    prompt = _build_prompt(task=task, question=question)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    except Exception as exc:
        raise RuntimeError(
            "Processor failed to build multimodal inputs. "
            "Please confirm this model_id supports image+text input with AutoProcessor."
        ) from exc
    inputs = _move_to_device(inputs, device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    if hasattr(processor, "batch_decode"):
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    elif hasattr(processor, "tokenizer"):
        text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    else:
        text = str(generated_ids[0].tolist())

    return {
        "task": task,
        "model_id": model_id,
        "image_path": str(image_path),
        "prompt": prompt,
        "response": text,
        "device": device.type,
    }


def export_artifacts(result: dict[str, Any], output_dir: Path) -> Path:
    """导出 Flamingo 结果 JSON 与可视化图。"""
    metrics_dir = output_dir / "flamingo_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    image_path = result.get("image_path")
    if image_path and Path(image_path).exists():
        try:
            from PIL import Image
            import matplotlib.pyplot as plt

            image = Image.open(image_path).convert("RGB")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(result.get("response", ""), fontsize=10)
            fig.tight_layout()
            fig.savefig(metrics_dir / "preview.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    return metrics_dir


def main() -> None:
    """主流程入口：执行 Flamingo 推理并导出产物。"""
    print("=== Flamingo 主流程（学习版）===", flush=True)

    # 步骤 1：读取参数并创建目录结构。
    args = build_default_args()
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    # 步骤 2：保存运行配置，方便后续复现实验。
    config_path = layout["output"] / "flamingo_run_config.json"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    # 步骤 3：dry-run 分支，仅验证流程。
    if args.dry_run:
        result = {
            "task": args.task,
            "model_id": args.model_id,
            "image_path": args.image,
            "prompt": _build_prompt(args.task, args.question),
            "response": "DRY_RUN: 未执行模型推理。",
            "device": detect_device().type,
        }
        metrics_dir = export_artifacts(result=result, output_dir=layout["output"])
        print(f"Flamingo dry-run done. Artifacts exported to: {metrics_dir}")
        return

    # 步骤 4：真实推理前校验图片输入。
    if not args.image:
        raise ValueError("`--image` is required unless `--dry-run` is used.")

    # 步骤 5：执行真实推理并导出结果。
    device = detect_device()
    print(f"Runtime: device={device.type}")
    result = run_inference(
        model_id=args.model_id,
        task=args.task,
        image_path=Path(args.image).expanduser().resolve(),
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )
    metrics_dir = export_artifacts(result=result, output_dir=layout["output"])
    print(f"Flamingo done. Artifacts exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
