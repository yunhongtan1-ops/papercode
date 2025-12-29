import os
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_small_object_label(label_path: Path, area_thr: float = 0.005, target_class: int | None = None) -> bool:
    """
    判断一个 label 文件中是否存在“小目标”实例：
    YOLO 格式：cls x y w h（均为归一化）
    bbox 面积占比 = w*h
    - area_thr: 0.005 表示 0.5%
    - target_class: 若只筛选特定类别（如污水排口），填类别 id；None 表示不区分类别
    """
    if not label_path.exists():
        return False

    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    except UnicodeDecodeError:
        # 少数情况下标签可能不是 utf-8
        lines = label_path.read_text(encoding="gbk", errors="ignore").strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        if target_class is not None and cls != target_class:
            continue

        if (w * h) < area_thr:
            return True

    return False


def copy_pair(img_path: Path, label_path: Path, out_img_dir: Path, out_label_dir: Path, mode: str = "copy"):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    dst_img = out_img_dir / img_path.name
    dst_lbl = out_label_dir / label_path.name

    if mode == "move":
        shutil.move(str(img_path), str(dst_img))
        shutil.move(str(label_path), str(dst_lbl))
    else:
        shutil.copy2(str(img_path), str(dst_img))
        shutil.copy2(str(label_path), str(dst_lbl))


def main(
    source_root: str,
    out_root: str | None = None,
    area_thr: float = 0.005,      # 0.5%
    target_class: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    mode: str = "copy",           # copy / move
):
    """
    source_root 目录结构期望：
      source_root/
        images/
        labels/

    输出 out_root：
      out_root/
        images/train|val|test
        labels/train|val|test
    """
    source_root = Path(source_root)
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels 目录不存在: {labels_dir}")

    if out_root is None:
        out_root = str(source_root.parent / f"{source_root.name}_Small_Split")
    out_root = Path(out_root)

    # 收集所有图像
    img_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    img_files.sort()

    missing_labels = []
    empty_labels = 0
    kept = []

    for img_path in img_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue

        # 空标注文件：不可能包含 small-object
        try:
            content = label_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = label_path.read_text(encoding="gbk", errors="ignore").strip()
        if content == "":
            empty_labels += 1
            continue

        if is_small_object_label(label_path, area_thr=area_thr, target_class=target_class):
            kept.append((img_path, label_path))

    # 统计
    print("==== Scan Summary ====")
    print(f"Total images found: {len(img_files)}")
    print(f"Missing label files: {len(missing_labels)}")
    print(f"Empty label files: {empty_labels}")
    print(f"Small-object kept: {len(kept)} (area_thr={area_thr}, target_class={target_class})")

    if len(kept) == 0:
        raise RuntimeError("没有筛选出任何 small-object 样本，请检查阈值 area_thr 或类别 target_class 是否设置正确。")

    # 划分
    random.seed(seed)
    random.shuffle(kept)

    n = len(kept)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_set = kept[:n_train]
    val_set = kept[n_train:n_train + n_val]
    test_set = kept[n_train + n_val:]

    print("==== Split Summary ====")
    print(f"Seed: {seed}")
    print(f"Train/Val/Test = {n_train}/{n_val}/{n_test} (ratio {train_ratio}:{val_ratio}:{1-train_ratio-val_ratio})")

    # 输出目录
    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        out_img_dir = out_root / "images" / split_name
        out_lbl_dir = out_root / "labels" / split_name
        for img_path, label_path in split_data:
            copy_pair(img_path, label_path, out_img_dir, out_lbl_dir, mode=mode)

    print(f"Done. Output saved to: {out_root}")
    if missing_labels:
        print("Example missing labels (first 10):", missing_labels[:10])