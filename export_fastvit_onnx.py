from __future__ import annotations

import pathlib
pathlib.WindowsPath = pathlib.PosixPath

import sys
from pathlib import Path


CHECKPOINT_PATH = Path("runs") / "fastvit_t12_dml_pretrained_full" / "best.pt"
OUTPUT_PATH = CHECKPOINT_PATH.with_name("fastvit_t12_pest.onnx")
MODEL_NAME = "fastvit_t12.apple_dist_in1k"
NUM_CLASSES = 15
IMAGE_SIZE = 224
DUMMY_INPUT_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def import_deps():
    try:
        import torch
        import timm
    except ImportError as error:
        missing = error.name or "a required dependency"
        raise RuntimeError(
            f"Missing dependency: {missing}\n"
            "Run this script in an environment with torch and timm installed."
        ) from error

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: onnx\n"
            "Install it in the export environment, for example:\n"
            "  pip install onnx\n"
            "Then rerun:\n"
            "  python export_fastvit_onnx.py"
        ) from error

    return torch, timm, onnx


def load_checkpoint(path: Path, torch) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a dictionary: {path}")
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint does not contain required key 'model': {path}")
    return checkpoint


def checkpoint_classes(checkpoint: dict) -> list[str]:
    classes = checkpoint.get("classes")
    if isinstance(classes, list) and classes:
        return [str(item) for item in classes]

    class_to_idx = checkpoint.get("class_to_idx")
    if isinstance(class_to_idx, dict) and class_to_idx:
        return [
            str(class_name)
            for class_name, _ in sorted(
                class_to_idx.items(),
                key=lambda item: int(item[1]),
            )
        ]

    return []


def write_classes_file(output_path: Path, classes: list[str]) -> None:
    classes_path = output_path.with_name("classes.txt")
    if classes_path.exists():
        return
    if not classes:
        raise RuntimeError("No classes found in checkpoint; cannot write classes.txt.")
    classes_path.write_text("\n".join(classes) + "\n", encoding="utf-8")


def main() -> int:
    torch, timm, onnx = import_deps()

    checkpoint = load_checkpoint(CHECKPOINT_PATH, torch)
    classes = checkpoint_classes(checkpoint)

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["model"])
    model.cpu()
    model.eval()

    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE, device=torch.device("cpu"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_classes_file(OUTPUT_PATH, classes)

    print("FastViT ONNX export")
    print(f"checkpoint path: {CHECKPOINT_PATH}")
    print(f"output path: {OUTPUT_PATH}")
    print(f"model name: {MODEL_NAME}")
    print(f"dummy input shape: {DUMMY_INPUT_SHAPE}")
    print(f"classes ({len(classes)}): {classes}")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(OUTPUT_PATH),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
    except Exception as error:
        raise RuntimeError(f"ONNX export failed: {error}") from error

    try:
        onnx_model = onnx.load(str(OUTPUT_PATH))
        onnx.checker.check_model(onnx_model)
    except Exception as error:
        raise RuntimeError(f"ONNX checker failed for {OUTPUT_PATH}: {error}") from error

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved ONNX model: {OUTPUT_PATH}")
    print(f"ONNX file size: {file_size_mb:.2f} MB")
    print(f"Classes file: {OUTPUT_PATH.with_name('classes.txt')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)