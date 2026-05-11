from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FastViT-T12 on a prepared ImageFolder dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset") / "fastvit_imagefolder",
        help="Prepared ImageFolder root with train/, validation/, and test/ folders.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--model",
        default="fastvit_t12.apple_dist_in1k",
        help="timm model name for FastViT-T12.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load ImageNet pretrained weights. This may download weights on first run.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        default="adamw",
    )
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "step", "none"),
        default="cosine",
    )
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "rocm", "dml"),
        help="Device to use. ROCm uses the cuda device API; DirectML uses privateuseone.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA mixed precision. Ignored on CPU.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile when available. Useful on the training machine, not required.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "fastvit_t12",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint saved by this script.",
    )
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Load only model weights from --resume and start a new optimizer schedule.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run validation/test only. Usually used together with --resume.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the test split after training or during --eval-only.",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Skip validation. Best checkpoint will not be updated.",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save periodic epoch checkpoints every N epochs. Always saves last.pt.",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit train batches per epoch. Use 1 for a quick smoke test.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation batches. Use 1 for a quick smoke test.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Limit test batches. Use 1 for a quick smoke test.",
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the final incomplete train batch.",
    )
    return parser.parse_args()


def import_training_deps():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        from torchvision.transforms import InterpolationMode
        import timm
    except ImportError as error:
        missing = error.name or "a training dependency"
        raise RuntimeError(
            f"Missing dependency: {missing}\n"
            "Install PyTorch/torchvision for your CPU, CUDA, ROCm, or DirectML setup, then install timm:\n"
            "  pip install -r requirements_fastvit.txt\n"
        ) from error

    return torch, nn, DataLoader, datasets, transforms, InterpolationMode, timm


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str, torch):
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")

    if requested == "rocm":
        if not torch.cuda.is_available():
            raise RuntimeError("--device rocm was requested, but torch cannot see a ROCm/HIP GPU.")
        if getattr(torch.version, "hip", None) is None:
            raise RuntimeError("--device rocm was requested, but torch.version.hip is None. This is not a ROCm PyTorch build.")
        return torch.device("cuda")

    if requested == "dml":
        try:
            import torch_directml
        except ImportError as error:
            raise RuntimeError(
                "--device dml was requested, but torch-directml is not installed. "
                "Run: pip install torch-directml"
            ) from error
        return torch_directml.device()

    # auto: prefer CUDA/ROCm if available, otherwise DirectML if installed, otherwise CPU
    if torch.cuda.is_available():
        selected = torch.device("cuda")
        print(f"--device auto selected: {selected}")
        return selected

    try:
        import torch_directml
        selected = torch_directml.device()
        print(f"--device auto selected: {selected} (DirectML)")
        return selected
    except Exception:
        selected = torch.device("cpu")
        print(f"--device auto selected: {selected} (CPU only)")
        return selected


def print_backend_info(torch, device, use_amp: bool) -> None:
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    if cuda_available and cuda_count > 0:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception as error:
            cuda_name = f"error: {error}"
    else:
        cuda_name = "n/a"

    try:
        import torch_directml  # noqa: F401
        dml_available = True
    except Exception:
        dml_available = False

    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    device_text = str(device)

    if device_text.startswith("privateuseone"):
        guess = "DirectML AMD/Windows GPU"
    elif cuda_available and hip is not None:
        guess = "ROCm/HIP AMD GPU"
    elif cuda_available and cuda is not None:
        guess = "CUDA NVIDIA GPU"
    else:
        guess = "CPU only"

    print("Backend info")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {cuda}")
    print(f"torch.version.hip: {hip}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.cuda.device_count(): {cuda_count}")
    print(f"torch.cuda.get_device_name(0): {cuda_name}")
    print(f"torch_directml imports successfully: {dml_available}")
    print(f"selected device: {device}")
    print(f"backend guess: {guess}")
    print(f"AMP enabled: {use_amp}")


def validate_data_root(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise RuntimeError(f"Data root does not exist: {args.data_root}")
    if not args.data_root.is_dir():
        raise RuntimeError(f"Data root is not a folder: {args.data_root}")

    train_dir = args.data_root / args.train_split
    val_dir = args.data_root / args.val_split
    test_dir = args.data_root / args.test_split

    if not args.eval_only and not train_dir.is_dir():
        raise RuntimeError(f"Training split folder not found: {train_dir}")
    if not args.no_val and not val_dir.is_dir():
        raise RuntimeError(f"Validation split folder not found: {val_dir}")
    if not test_dir.is_dir():
        raise RuntimeError(f"Test split folder not found: {test_dir}")


def print_dataset_sizes(args, train_dataset, val_dataset, test_dataset) -> None:
    def print_one(label: str, dataset) -> None:
        if dataset is None:
            print(f"{label}: skipped")
            return
        classes = getattr(dataset, "classes", [])
        print(f"{label}: {len(dataset)} images, {len(classes)} classes")

    print_one(args.train_split, train_dataset)
    print_one(args.val_split, val_dataset)
    print_one(args.test_split, test_dataset)


def build_transforms(args: argparse.Namespace, transforms, InterpolationMode):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.image_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    resize_size = math.ceil(args.image_size / 0.875)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def safe_pil_loader(path: str):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def make_safe_imagefolder_class(datasets):
    class SafeImageFolder(datasets.ImageFolder):
        def __init__(self, root, transform=None, target_transform=None):
            super().__init__(
                root,
                transform=transform,
                target_transform=target_transform,
                loader=safe_pil_loader,
            )
            self.bad_image_count = 0
            self.samples = self._filter_readable_samples(self.samples)
            self.imgs = self.samples

        def _filter_readable_samples(self, samples):
            readable = []
            for path, target in samples:
                try:
                    self.loader(path)
                except Exception:
                    print(f"WARNING: skipping unreadable image: {path}", flush=True)
                    self.bad_image_count += 1
                    continue
                readable.append((path, target))
            return readable

    return SafeImageFolder


def build_imagefolder_dataset(split_name: str, split_dir: Path, transform, datasets, required: bool):
    if not split_dir.is_dir():
        if required:
            raise RuntimeError(f"Split folder not found: {split_dir}")
        return None
    dataset_class = make_safe_imagefolder_class(datasets)
    dataset = dataset_class(split_dir, transform=transform)
    print(
        f"{split_name}: skipped {dataset.bad_image_count} unreadable images",
        flush=True,
    )
    if len(dataset) == 0 and required:
        raise RuntimeError(f"No images found in {split_dir}")
    return dataset


def build_loader(
    dataset,
    args: argparse.Namespace,
    DataLoader,
    torch,
    device,
    *,
    train: bool,
):
    if dataset is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    use_cuda = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=args.drop_last if train else False,
        generator=generator if train else None,
        persistent_workers=args.num_workers > 0,
    )


def build_dataloaders(args: argparse.Namespace, deps, torch, device):
    _, _, DataLoader, datasets, transforms, InterpolationMode, _ = deps
    train_transform, eval_transform = build_transforms(args, transforms, InterpolationMode)

    train_dataset = build_imagefolder_dataset(
        args.train_split,
        args.data_root / args.train_split,
        train_transform,
        datasets,
        required=not args.eval_only,
    )
    val_dataset = None
    if not args.no_val:
        val_dataset = build_imagefolder_dataset(
            args.val_split,
            args.data_root / args.val_split,
            eval_transform,
            datasets,
            required=not args.eval_only,
        )
    test_dataset = build_imagefolder_dataset(
        args.test_split,
        args.data_root / args.test_split,
        eval_transform,
        datasets,
        required=True,
    )

    reference_dataset = train_dataset or val_dataset or test_dataset
    if reference_dataset is None:
        raise RuntimeError("No usable dataset split was found.")

    for name, dataset in (("validation", val_dataset), ("test", test_dataset)):
        if dataset is not None and dataset.class_to_idx != reference_dataset.class_to_idx:
            raise RuntimeError(f"{name} class_to_idx does not match the training split.")

    train_loader = build_loader(train_dataset, args, DataLoader, torch, device, train=True)
    val_loader = build_loader(val_dataset, args, DataLoader, torch, device, train=False)
    test_loader = None
    if args.evaluate_test:
        test_loader = build_loader(test_dataset, args, DataLoader, torch, device, train=False)
    return reference_dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_model(args: argparse.Namespace, num_classes: int, timm):
    try:
        return timm.create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=num_classes,
        )
    except Exception as error:
        matches = timm.list_models("*fastvit_t12*")
        hint = ", ".join(matches[:10]) if matches else "no fastvit_t12 models found"
        raise RuntimeError(
            f"Could not create timm model {args.model!r}.\n"
            f"Available FastViT-T12-like models: {hint}"
        ) from error


def build_optimizer(args: argparse.Namespace, model, torch):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def build_scheduler(args: argparse.Namespace, optimizer, torch):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def amp_autocast(torch, use_amp: bool):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def create_grad_scaler(torch, use_amp: bool):
    if not use_amp:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler()


def accuracy_from_logits(logits, targets) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def train_one_epoch(
    *,
    epoch: int,
    args: argparse.Namespace,
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    torch,
    use_amp: bool,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    start_time = time.time()
    max_batches = args.max_train_batches or len(loader)

    for batch_index, (images, targets) in enumerate(loader, 1):
        if args.max_train_batches is not None and batch_index > args.max_train_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(torch, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            if scaler is None:
                raise RuntimeError("AMP is enabled, but no GradScaler was created.")
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size
        global_step += 1

        if args.log_interval > 0 and (
            batch_index == 1 or batch_index % args.log_interval == 0
        ):
            avg_loss = total_loss / max(1, total_seen)
            avg_acc = total_correct / max(1, total_seen)
            print(
                f"epoch {epoch:03d} train "
                f"[{batch_index}/{max_batches}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f} lr={current_lr(optimizer):.6g}",
                flush=True,
            )

    elapsed = time.time() - start_time
    return (
        {
            "loss": total_loss / max(1, total_seen),
            "acc": total_correct / max(1, total_seen),
            "samples": float(total_seen),
            "seconds": elapsed,
        },
        global_step,
    )


def evaluate(
    *,
    split_name: str,
    args: argparse.Namespace,
    model,
    loader,
    criterion,
    device,
    torch,
    use_amp: bool,
) -> dict[str, float]:
    if loader is None:
        return {"loss": float("nan"), "acc": float("nan"), "samples": 0.0, "seconds": 0.0}

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    start_time = time.time()
    max_batches = {
        "val": args.max_val_batches,
        "test": args.max_test_batches,
    }.get(split_name, None)

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, 1):
            if max_batches is not None and batch_index > max_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp_autocast(torch, use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_seen += batch_size

    elapsed = time.time() - start_time
    metrics = {
        "loss": total_loss / max(1, total_seen),
        "acc": total_correct / max(1, total_seen),
        "samples": float(total_seen),
        "seconds": elapsed,
    }
    print(
        f"{split_name} loss={metrics['loss']:.4f} "
        f"acc={metrics['acc']:.4f} samples={int(metrics['samples'])}",
        flush=True,
    )
    return metrics


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    model,
    optimizer,
    scheduler,
    scaler,
    dataset,
    best_val_acc: float,
    best_val_loss: float,
    metrics: dict[str, float],
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_name": args.model,
        "pretrained": args.pretrained,
        "num_classes": len(dataset.classes),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "args": vars(args),
    }


def save_checkpoint(path: Path, payload: dict, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    *,
    args: argparse.Namespace,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    torch,
) -> tuple[int, int, float, float]:
    if args.resume is None:
        return 1, 0, float("-inf"), float("inf")

    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model"])

    if not args.resume_model_only:
        if checkpoint.get("optimizer") is not None and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = 1 if args.resume_model_only else int(checkpoint.get("epoch", 0)) + 1
    global_step = 0 if args.resume_model_only else int(checkpoint.get("global_step", 0))
    best_val_acc = float(checkpoint.get("best_val_acc", float("-inf")))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    print(f"Loaded checkpoint: {args.resume}")
    print(f"Starting epoch: {start_epoch}")
    return start_epoch, global_step, best_val_acc, best_val_loss


def write_run_metadata(args: argparse.Namespace, dataset) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "class_to_idx.json").write_text(
        json.dumps(dataset.class_to_idx, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "classes.txt").write_text(
        "\n".join(dataset.classes) + "\n",
        encoding="utf-8",
    )


def append_metrics_row(args: argparse.Namespace, row: dict[str, float | int | str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.csv"
    write_header = not log_path.exists()
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "train_samples",
        "val_loss",
        "val_acc",
        "val_samples",
        "seconds",
    ]
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def maybe_compile_model(args: argparse.Namespace, model, torch):
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this PyTorch version; continuing uncompiled.")
        return model
    print("Compiling model with torch.compile...")
    return torch.compile(model)


def main() -> int:
    args = parse_args()
    validate_data_root(args)
    deps = import_training_deps()
    torch, nn, _, _, _, _, timm = deps
    seed_everything(args.seed, torch)

    if args.epochs < 1 and not args.eval_only:
        raise RuntimeError("--epochs must be >= 1 for training.")

    device = resolve_device(args.device, torch)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("--amp requested, but CUDA is not active; running without AMP.")
    print_backend_info(torch, device, use_amp)

    (
        dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(args, deps, torch, device)
    model = create_model(args, num_classes=len(dataset.classes), timm=timm)
    model.to(device)
    model = maybe_compile_model(args, model, torch)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model, torch)
    scheduler = build_scheduler(args, optimizer, torch)
    scaler = create_grad_scaler(torch, use_amp)

    start_epoch, global_step, best_val_acc, best_val_loss = load_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        torch=torch,
    )

    if not args.no_save:
        write_run_metadata(args, dataset)

    print("FastViT-T12 training configuration")
    print(f"data_root: {args.data_root}")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"amp: {use_amp}")
    print(f"num_classes: {len(dataset.classes)}")
    print_dataset_sizes(args, train_dataset, val_dataset, test_dataset)
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"output_dir: {args.output_dir}")

    if args.eval_only:
        if val_loader is not None:
            evaluate(
                split_name="val",
                args=args,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                torch=torch,
                use_amp=use_amp,
            )
        if test_loader is not None:
            evaluate(
                split_name="test",
                args=args,
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                torch=torch,
                use_amp=use_amp,
            )
        return 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_metrics, global_step = train_one_epoch(
            epoch=epoch,
            args=args,
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            torch=torch,
            use_amp=use_amp,
            global_step=global_step,
        )

        val_metrics = {"loss": float("nan"), "acc": float("nan"), "samples": 0.0, "seconds": 0.0}
        if val_loader is not None:
            val_metrics = evaluate(
                split_name="val",
                args=args,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                torch=torch,
                use_amp=use_amp,
            )

        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.time() - epoch_start
        print(
            f"epoch {epoch:03d} done "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )

        is_best = False
        if val_loader is not None:
            val_acc = float(val_metrics["acc"])
            val_loss = float(val_metrics["loss"])
            is_best = val_acc > best_val_acc or (
                math.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
            )
            if is_best:
                best_val_acc = val_acc
                best_val_loss = val_loss

        metrics = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "train_samples": float(train_metrics["samples"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_samples": float(val_metrics["samples"]),
            "lr": current_lr(optimizer),
            "seconds": epoch_seconds,
        }

        if not args.no_save:
            append_metrics_row(
                args,
                {
                    "epoch": epoch,
                    "lr": metrics["lr"],
                    "train_loss": metrics["train_loss"],
                    "train_acc": metrics["train_acc"],
                    "train_samples": int(metrics["train_samples"]),
                    "val_loss": metrics["val_loss"],
                    "val_acc": metrics["val_acc"],
                    "val_samples": int(metrics["val_samples"]),
                    "seconds": epoch_seconds,
                },
            )
            payload = checkpoint_payload(
                args=args,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                dataset=dataset,
                best_val_acc=best_val_acc,
                best_val_loss=best_val_loss,
                metrics=metrics,
            )
            save_checkpoint(args.output_dir / "last.pt", payload, torch)
            if is_best:
                save_checkpoint(args.output_dir / "best.pt", payload, torch)
                print(f"saved new best checkpoint: {args.output_dir / 'best.pt'}")
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", payload, torch)

    if args.evaluate_test and test_loader is not None:
        print("Final test evaluation")
        evaluate(
            split_name="test",
            args=args,
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            torch=torch,
            use_amp=use_amp,
        )

    print("Training complete.")
    if not args.no_save:
        print(f"last checkpoint: {args.output_dir / 'last.pt'}")
        if (args.output_dir / "best.pt").exists():
            print(f"best checkpoint: {args.output_dir / 'best.pt'}")
        print(f"metrics log: {args.output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
