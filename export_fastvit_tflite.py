from __future__ import annotations

import importlib
import sys
from pathlib import Path


CHECKPOINT_PATH = Path("runs") / "fastvit_t12_dml_pretrained_full" / "best.pt"
DATA_ROOT = Path("dataset_prepared") / "fastvit_imagefolder"
OUTPUT_PATH = CHECKPOINT_PATH.with_name("fastvit_t12_pest.tflite")
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
            "Run this script in the same environment used for FastViT training, "
            "with torch and timm installed."
        ) from error

    for module_name in ("ai_edge_torch", "litert_torch"):
        try:
            converter = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(converter, "convert"):
            return torch, timm, converter

    raise RuntimeError(
        "LiteRT PyTorch conversion support is not installed.\n"
        "Install the Google AI Edge/LiteRT Torch converter, for example:\n"
        "  pip install ai-edge-torch\n"
        "Then rerun:\n"
        "  python export_fastvit_tflite.py"
    )


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

    classes_file = DATA_ROOT / "classes.txt"
    if classes_file.is_file():
        return [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return []


def write_classes_file(output_path: Path, classes: list[str]) -> None:
    classes_path = output_path.with_name("classes.txt")
    if classes_path.exists():
        return
    classes_path.write_text("\n".join(classes) + "\n", encoding="utf-8")


def print_verification(
    *,
    checkpoint_path: Path,
    output_path: Path,
    classes: list[str],
    model_name: str,
    dummy_input_shape: tuple[int, int, int, int],
) -> None:
    print("FastViT TFLite export")
    print(f"checkpoint path: {checkpoint_path}")
    print(f"output tflite path: {output_path}")
    print(f"classes ({len(classes)}): {classes}")
    print(f"model name: {model_name}")
    print(f"dummy input shape: {dummy_input_shape}")


def main() -> int:
    torch, timm, ai_edge_torch = import_deps()

    checkpoint = load_checkpoint(CHECKPOINT_PATH, torch)
    classes = checkpoint_classes(checkpoint)
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, but found {len(classes)} in the checkpoint/data files."
        )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_classes_file(OUTPUT_PATH, classes)

    print_verification(
        checkpoint_path=CHECKPOINT_PATH,
        output_path=OUTPUT_PATH,
        classes=classes,
        model_name=MODEL_NAME,
        dummy_input_shape=DUMMY_INPUT_SHAPE,
    )

    try:
        edge_model = ai_edge_torch.convert(model, (dummy_input,))
        edge_model.export(str(OUTPUT_PATH))
    except Exception as error:
        raise RuntimeError(
            "TFLite export failed. The converter is installed, but this model may use "
            "operators that are not supported by the current LiteRT Torch converter.\n"
            f"Original error: {error}"
        ) from error

    print(f"Saved TFLite model: {OUTPUT_PATH}")
    print(f"Saved classes file: {OUTPUT_PATH.with_name('classes.txt')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
