#!/usr/bin/env python3
"""
BLIP2 最小可运行示例（图像描述 / 简单视觉问答）。

一、BLIP2 原理（面向实现）
1) BLIP2 通过 Q-Former 连接视觉编码器与大语言模型。
2) 图像先被视觉编码器提取特征，再由 Q-Former压缩为查询表示。
3) 查询表示作为桥梁输入到语言模型，生成描述或回答。
4) 推理阶段无需端到端预训练，只需加载现成 BLIP2 权重即可完成多模态生成。

新人阅读顺序（建议）
1) 先看 `build_default_args`：明确可调参数和默认值。
2) 再看 `main`：把握执行主链路（准备 -> 训练/推理 -> 导出）。
3) 最后看可视化导出函数（如 `export_artifacts`）理解输出文件。

二、代码框架（从入口到结果）
1) `build_default_args`：读取推理参数。
2) `ensure_layout_dirs`：创建统一目录结构。
3) `run_inference`：加载 BLIP2 并执行生成（caption/vqa）。
4) `export_artifacts`：导出 JSON 与可视化图片。
5) `main`：串联完整流程。

用法：
  python code/blip2.py --image /absolute/path/to/image.jpg --task caption
  python code/blip2.py --image /absolute/path/to/image.jpg --task vqa --question "图里有什么？"
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
    """解析命令行参数，返回 BLIP2 推理配置。"""
    parser = argparse.ArgumentParser(description="Run BLIP2 inference and export artifacts.")
    parser.add_argument(
        "--model-id",
        default="Salesforce/blip2-opt-2.7b",
        help="Hugging Face 上的 BLIP2 模型名称。",
    )
    parser.add_argument(
        "--task",
        default="caption",
        choices=["caption", "vqa"],
        help="推理任务：图像描述或视觉问答。",
    )
    parser.add_argument("--image", default=None, help="输入图片绝对路径。")
    parser.add_argument("--question", default="What is in the image?", help="VQA 问题文本。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=3)
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
    """解析输出目录：相对路径按 blip2 目录解析，绝对路径原样使用。"""
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


def _get_prompt(task: str, question: str) -> str:
    """根据任务类型返回文本提示。"""
    if task == "caption":
        return "Question: what does the image describe? Answer:"
    return f"Question: {question.strip()} Answer:"


def run_inference(
    model_id: str,
    task: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> dict[str, Any]:
    """执行 BLIP2 推理并返回结果。"""
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Missing dependency `Pillow`: {exc}") from exc

    try:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
    except Exception as exc:
        raise RuntimeError(f"Missing dependency `transformers`: {exc}") from exc

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    processor = Blip2Processor.from_pretrained(model_id)

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    model.eval()

    prompt = _get_prompt(task=task, question=question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return {
        "task": task,
        "model_id": model_id,
        "image_path": str(image_path),
        "prompt": prompt,
        "response": text,
        "device": device.type,
    }


def export_artifacts(result: dict[str, Any], output_dir: Path) -> Path:
    """导出 BLIP2 结果 JSON 与可视化图。"""
    metrics_dir = output_dir / "blip2_metrics"
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
            # 可视化失败不影响核心结果导出。
            pass

    return metrics_dir


def main() -> None:
    """主流程入口：执行 BLIP2 推理并导出产物。"""
    print("=== BLIP2 主流程（学习版）===", flush=True)

    # 步骤 1：读取参数并创建目录结构。
    args = build_default_args()
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    # 步骤 2：保存本次运行配置（技术复盘时可直接引用）。
    config_path = layout["output"] / "blip2_run_config.json"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    # 步骤 3：dry-run 分支，仅验证流程与产物，不下载大模型。
    if args.dry_run:
        result = {
            "task": args.task,
            "model_id": args.model_id,
            "image_path": args.image,
            "prompt": _get_prompt(args.task, args.question),
            "response": "DRY_RUN: 未执行模型推理。",
            "device": detect_device().type,
        }
        metrics_dir = export_artifacts(result=result, output_dir=layout["output"])
        print(f"BLIP2 dry-run done. Artifacts exported to: {metrics_dir}")
        return

    # 步骤 4：真实推理前校验输入图片。
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
    print(f"BLIP2 done. Artifacts exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
