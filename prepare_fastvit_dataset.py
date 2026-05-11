from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an initial train/val/test class-folder image dataset into "
            "flat split folders with manifests, plus an Apple FastViT/ImageFolder view."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Folder containing train/, val/, and test/ class subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Where to write the prepared dataset. Defaults to input_root and converts "
            "in place by renaming the original train/val/test folders."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed used for per-class redistribution.",
    )
    parser.add_argument(
        "--no-imagefolder",
        action="store_true",
        help="Skip fastvit_imagefolder/, the ImageFolder-compatible view.",
    )
    parser.add_argument(
        "--force-empty-output",
        action="store_true",
        help=(
            "Only for a separate --output-root: delete that output folder's existing "
            "contents before writing."
        ),
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text) or "class"


def clean_stem(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9._()-]+", "_", text).strip("_")
    return re.sub(r"_+", "_", text) or "image"


def require_initial_split(split_dir: Path) -> None:
    if not split_dir.is_dir():
        raise RuntimeError(f"Missing split folder: {split_dir}")
    class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    direct_images = [path for path in split_dir.iterdir() if is_image(path)]
    if not class_dirs:
        raise RuntimeError(f"{split_dir} has no class subfolders.")
    if direct_images:
        raise RuntimeError(
            f"{split_dir} already contains direct images. Expected the initial "
            "class-folder layout, for example train/<class_name>/image.jpg."
        )


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


def hardlink_or_copy(source: str, target: str) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def check_output_conflicts(output_root: Path, in_place: bool) -> None:
    output_files = [
        "train.txt",
        "val.txt",
        "test.txt",
        "train_with_names.tsv",
        "val_with_names.tsv",
        "test_with_names.tsv",
        "classes.txt",
        "classes_with_indices.tsv",
        "class_to_idx.json",
        "idx_to_class.json",
        "split_summary.csv",
        "fastvit_dataset_metadata.json",
        "README_FASTVIT.md",
    ]
    conflicts = [output_root / name for name in output_files if (output_root / name).exists()]
    if (output_root / "fastvit_imagefolder").exists():
        conflicts.append(output_root / "fastvit_imagefolder")
    if not in_place:
        conflicts.extend([output_root / split for split in SPLITS if (output_root / split).exists()])
    if conflicts:
        conflict_text = "\n".join(f"  - {path}" for path in conflicts)
        raise RuntimeError(
            "Output already contains prepared-dataset files. Use a clean output folder.\n"
            f"{conflict_text}"
        )


def prepare_source_folders(
    input_root: Path,
    output_root: Path,
    in_place: bool,
    timestamp: str,
) -> dict[str, Path]:
    source_dirs: dict[str, Path] = {}
    for split in SPLITS:
        source_split_dir = input_root / split
        backup_name = f"original_{split}_classfolders_{timestamp}"
        backup_dir = next_available_path(output_root / backup_name)

        if in_place:
            source_split_dir.rename(backup_dir)
        else:
            shutil.copytree(source_split_dir, backup_dir, copy_function=hardlink_or_copy)

        source_dirs[split] = backup_dir
    return source_dirs


def collect_images(source_dirs: dict[str, Path]) -> dict[str, list[dict[str, str]]]:
    images_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source_split, source_dir in source_dirs.items():
        for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            class_name = class_dir.name
            for image_path in sorted(path for path in class_dir.rglob("*") if is_image(path)):
                images_by_class[class_name].append(
                    {
                        "source_split": source_split,
                        "source_path": str(image_path),
                        "source_relative_path": (
                            Path(source_split) / class_name / image_path.relative_to(class_dir)
                        ).as_posix(),
                    }
                )
    if not images_by_class:
        raise RuntimeError("No images found in the source train/val/test folders.")
    return images_by_class


def split_counts(total: int) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "val": 0, "test": 0}

    val_count = round(total * RATIOS["val"])
    test_count = round(total * RATIOS["test"])
    if total >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)

    while val_count + test_count >= total and total > 1:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    return {
        "train": total - val_count - test_count,
        "val": val_count,
        "test": test_count,
    }


def unique_output_name(
    used_names: set[str],
    class_index: int,
    class_name: str,
    source_path: Path,
) -> str:
    base = f"{class_index:03d}_{slugify(class_name)}__{clean_stem(source_path.stem)}"
    suffix = source_path.suffix.lower()
    candidate = f"{base}{suffix}"
    index = 2
    while candidate in used_names:
        candidate = f"{base}_{index}{suffix}"
        index += 1
    used_names.add(candidate)
    return candidate


def redistribute_and_copy(
    output_root: Path,
    images_by_class: dict[str, list[dict[str, str]]],
    seed: int,
) -> tuple[list[dict[str, str | int]], list[str]]:
    class_names = sorted(images_by_class)
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}

    for split in SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=False)

    used_names = {split: set() for split in SPLITS}
    records: list[dict[str, str | int]] = []

    for class_name in class_names:
        class_index = class_to_idx[class_name]
        items = list(images_by_class[class_name])
        rng = random.Random(f"{seed}:{class_name}")
        rng.shuffle(items)

        counts = split_counts(len(items))
        train_end = counts["train"]
        val_end = counts["train"] + counts["val"]

        for item_index, item in enumerate(items):
            if item_index < train_end:
                split = "train"
            elif item_index < val_end:
                split = "val"
            else:
                split = "test"

            source_path = Path(str(item["source_path"]))
            filename = unique_output_name(
                used_names[split],
                class_index,
                class_name,
                source_path,
            )
            target_path = output_root / split / filename
            shutil.copy2(source_path, target_path)

            records.append(
                {
                    "split": split,
                    "filename": filename,
                    "relative_path": f"{split}/{filename}",
                    "class_index": class_index,
                    "class_name": class_name,
                    "source_split": item["source_split"],
                    "source_relative_path": item["source_relative_path"],
                }
            )

    return records, class_names


def write_manifests(output_root: Path, records: list[dict[str, str | int]]) -> None:
    by_split: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for record in records:
        by_split[str(record["split"])].append(record)

    for split in SPLITS:
        split_records = sorted(
            by_split[split],
            key=lambda record: (int(record["class_index"]), str(record["filename"])),
        )

        root_lines = [
            f"{record['relative_path']} {record['class_index']}"
            for record in split_records
        ]
        local_lines = [
            f"{record['filename']} {record['class_index']}"
            for record in split_records
        ]

        (output_root / f"{split}.txt").write_text(
            "\n".join(root_lines) + "\n",
            encoding="utf-8",
        )
        (output_root / split / f"{split}.txt").write_text(
            "\n".join(local_lines) + "\n",
            encoding="utf-8",
        )

        header = [
            "relative_path",
            "class_index",
            "class_name",
            "source_split",
            "source_relative_path",
        ]
        root_tsv = output_root / f"{split}_with_names.tsv"
        local_tsv = output_root / split / f"{split}_with_names.tsv"

        with root_tsv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            for record in split_records:
                writer.writerow(
                    [
                        record["relative_path"],
                        record["class_index"],
                        record["class_name"],
                        record["source_split"],
                        record["source_relative_path"],
                    ]
                )

        with local_tsv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            for record in split_records:
                writer.writerow(
                    [
                        record["filename"],
                        record["class_index"],
                        record["class_name"],
                        record["source_split"],
                        record["source_relative_path"],
                    ]
                )


def write_class_files(output_root: Path, class_names: list[str]) -> None:
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    idx_to_class = {str(index): class_name for class_name, index in class_to_idx.items()}

    (output_root / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )
    (output_root / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "idx_to_class.json").write_text(
        json.dumps(idx_to_class, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (output_root / "classes_with_indices.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["class_index", "class_name"])
        for index, class_name in enumerate(class_names):
            writer.writerow([index, class_name])


def write_summary(output_root: Path, records: list[dict[str, str | int]], class_names: list[str]) -> dict[str, int]:
    totals_by_split = {split: 0 for split in SPLITS}
    counts_by_class = {
        class_name: {"train": 0, "val": 0, "test": 0, "total": 0}
        for class_name in class_names
    }

    for record in records:
        split = str(record["split"])
        class_name = str(record["class_name"])
        totals_by_split[split] += 1
        counts_by_class[class_name][split] += 1
        counts_by_class[class_name]["total"] += 1

    with (output_root / "split_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "class_index",
            "class_name",
            "train",
            "val",
            "test",
            "total",
            "train_ratio",
            "val_ratio",
            "test_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, class_name in enumerate(class_names):
            row = counts_by_class[class_name]
            total = row["total"]
            writer.writerow(
                {
                    "class_index": index,
                    "class_name": class_name,
                    "train": row["train"],
                    "val": row["val"],
                    "test": row["test"],
                    "total": total,
                    "train_ratio": f"{row['train'] / total:.4f}" if total else "0.0000",
                    "val_ratio": f"{row['val'] / total:.4f}" if total else "0.0000",
                    "test_ratio": f"{row['test'] / total:.4f}" if total else "0.0000",
                }
            )

    return totals_by_split


def create_imagefolder_view(output_root: Path, records: list[dict[str, str | int]]) -> dict[str, int | str]:
    view_root = output_root / "fastvit_imagefolder"
    view_root.mkdir(exist_ok=False)
    counts = {"hardlink": 0, "copy": 0}

    for record in records:
        split = str(record["split"])
        imagefolder_split = "validation" if split == "val" else split
        class_name = str(record["class_name"])
        source_path = output_root / str(record["relative_path"])
        target_dir = view_root / imagefolder_split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / str(record["filename"])
        try:
            os.link(source_path, target_path)
            counts["hardlink"] += 1
        except OSError:
            shutil.copy2(source_path, target_path)
            counts["copy"] += 1

    return {
        "path": view_root.name,
        "method": "hardlink" if counts["copy"] == 0 else "mixed",
        "hardlinks": counts["hardlink"],
        "copies": counts["copy"],
    }


def write_metadata(
    output_root: Path,
    source_dirs: dict[str, Path],
    class_names: list[str],
    totals_by_split: dict[str, int],
    seed: int,
    imagefolder_info: dict[str, int | str] | None,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(output_root),
        "seed": seed,
        "ratios": RATIOS,
        "num_classes": len(class_names),
        "num_images": sum(totals_by_split.values()),
        "splits": totals_by_split,
        "source_folders": {split: path.name for split, path in source_dirs.items()},
        "flat_root_manifest_format": "relative/path/to/image class_index",
        "flat_split_manifest_format": "image_filename class_index",
        "class_index_order": "alphabetical by original folder name, matching torchvision ImageFolder",
        "imagefolder_view": imagefolder_info,
    }
    (output_root / "fastvit_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_readme(
    output_root: Path,
    source_dirs: dict[str, Path],
    totals_by_split: dict[str, int],
    class_count: int,
    imagefolder_info: dict[str, int | str] | None,
) -> None:
    source_lines = "\n".join(
        f"- `{split}` -> `{path.name}`" for split, path in source_dirs.items()
    )
    split_lines = "\n".join(f"- `{split}`: {totals_by_split[split]}" for split in SPLITS)
    imagefolder_section = "Not created."
    if imagefolder_info:
        imagefolder_section = f"""Created at `{imagefolder_info['path']}`.

The unmodified Apple `ml-fastvit` training script expects ImageNet-style folders:

```text
fastvit_imagefolder/
  train/<class name>/*.jpg
  validation/<class name>/*.jpg
  test/<class name>/*.jpg
```

Example data argument for Apple FastViT:

```powershell
python train.py "{(output_root / 'fastvit_imagefolder').as_posix()}" --model fastvit_t8 --input-size 3 256 256
```

Files were created using `{imagefolder_info['method']}` mode (`{imagefolder_info['hardlinks']}` hardlinks, `{imagefolder_info['copies']}` copies).
"""

    readme = f"""# FastViT Dataset Preparation

This dataset was prepared by `prepare_fastvit_dataset.py`.

## Source Preservation

The original class-folder splits were preserved:

{source_lines}

## New Flat Splits

The current `train/`, `val/`, and `test/` folders contain images directly in each split folder. The split was rebuilt stratified by class from all source images with an 80/10/10 target ratio.

{split_lines}

Total images: {sum(totals_by_split.values())}
Classes: {class_count}

Each split has a local manifest:

```text
train/train.txt
val/val.txt
test/test.txt
```

Local manifest format:

```text
image_filename class_index
```

Root-level manifests are also available:

```text
train.txt
val.txt
test.txt
```

Root manifest format:

```text
split/image_filename class_index
```

The `*_with_names.tsv` files include `class_name` and original source path columns for inspection.

## Class Mapping

- `classes.txt`: one class name per line. The line number is the class index.
- `classes_with_indices.tsv`: class index plus class name.
- `class_to_idx.json` and `idx_to_class.json`: machine-readable mappings.
- `split_summary.csv`: per-class split counts and ratios.
- `fastvit_dataset_metadata.json`: preparation metadata.

Class indices are alphabetical by original folder name, matching torchvision `ImageFolder` ordering.

## ImageFolder View

{imagefolder_section}
"""
    (output_root / "README_FASTVIT.md").write_text(readme, encoding="utf-8")


def verify_outputs(output_root: Path, totals_by_split: dict[str, int]) -> None:
    class_to_idx = json.loads((output_root / "class_to_idx.json").read_text(encoding="utf-8"))
    valid_labels = set(class_to_idx.values())

    for split, expected_count in totals_by_split.items():
        split_dir = output_root / split
        image_count = sum(1 for path in split_dir.iterdir() if is_image(path))
        if image_count != expected_count:
            raise RuntimeError(f"{split} has {image_count} images, expected {expected_count}.")

        for manifest in (output_root / f"{split}.txt", split_dir / f"{split}.txt"):
            lines = manifest.read_text(encoding="utf-8").splitlines()
            if len(lines) != expected_count:
                raise RuntimeError(f"{manifest} has {len(lines)} rows, expected {expected_count}.")
            for line_number, line in enumerate(lines, 1):
                image_ref, label_text = line.rsplit(" ", 1)
                label = int(label_text)
                if label not in valid_labels:
                    raise RuntimeError(f"Invalid label in {manifest}:{line_number}: {label}")
                image_path = output_root / image_ref if manifest.parent == output_root else split_dir / image_ref
                if not image_path.is_file():
                    raise RuntimeError(f"Missing image in {manifest}:{line_number}: {image_ref}")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root).resolve()
    in_place = same_path(input_root, output_root)

    if not input_root.is_dir():
        raise RuntimeError(f"Input folder does not exist: {input_root}")
    for split in SPLITS:
        require_initial_split(input_root / split)

    if in_place:
        check_output_conflicts(output_root, in_place=True)
    else:
        if output_root.exists() and any(output_root.iterdir()):
            if not args.force_empty_output:
                raise RuntimeError(
                    f"Output folder is not empty: {output_root}\n"
                    "Use --force-empty-output only when you intentionally want it cleared."
                )
            clear_directory_contents(output_root)
        else:
            output_root.mkdir(parents=True, exist_ok=True)
        check_output_conflicts(output_root, in_place=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_dirs = prepare_source_folders(input_root, output_root, in_place, timestamp)
    images_by_class = collect_images(source_dirs)
    records, class_names = redistribute_and_copy(output_root, images_by_class, args.seed)
    write_manifests(output_root, records)
    write_class_files(output_root, class_names)
    totals_by_split = write_summary(output_root, records, class_names)
    imagefolder_info = None
    if not args.no_imagefolder:
        imagefolder_info = create_imagefolder_view(output_root, records)
    write_metadata(output_root, source_dirs, class_names, totals_by_split, args.seed, imagefolder_info)
    write_readme(output_root, source_dirs, totals_by_split, len(class_names), imagefolder_info)
    verify_outputs(output_root, totals_by_split)

    print(f"Prepared dataset: {output_root}")
    print(f"Classes: {len(class_names)}")
    print(f"Images: {sum(totals_by_split.values())}")
    for split in SPLITS:
        print(f"{split}: {totals_by_split[split]}")
    if imagefolder_info:
        print(f"ImageFolder view: {output_root / 'fastvit_imagefolder'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
