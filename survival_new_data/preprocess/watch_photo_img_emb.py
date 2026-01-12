from __future__ import annotations

"""
边下载图片边向量化（photo 级别）：持续扫描 image_dir，发现某张图片已落盘就立刻编码并追加写入 parquet。

适用场景：
- 你正在跑 scripts/download_openrice_images.py 在 openrice_images/ 下持续落盘图片；
- 同时启动本脚本，让 GPU/CPU 空闲时就把已下载的图片先向量化。

示例（另开一个终端并行运行）：
  python -m survival_new_data.preprocess.watch_photo_img_emb ^
    --image-dir openrice_images ^
    --manifest openrice_images/selected_reviews.csv ^
    --out-dir artifacts ^
    --max-images-per-review 1 ^
    --device auto
"""

import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _normalize_img_entry(entry: str) -> str:
    entry = (entry or "").strip()
    if "|" in entry:
        return entry.split("|", 1)[0].strip()
    return entry


def _extract_image_plan(review_imgsrc: str, review_photo_id: str, *, max_images: int) -> List[Tuple[str, str]]:
    raw_urls = [
        _normalize_img_entry(part)
        for part in (review_imgsrc or "").split("\n")
        if part and part.strip()
    ]
    raw_ids = [part.strip() for part in (review_photo_id or "").split(";") if part and part.strip()]
    if max_images <= 0:
        limit = min(len(raw_urls), len(raw_ids))
    else:
        limit = min(int(max_images), len(raw_urls), len(raw_ids))
    return [(raw_urls[i], raw_ids[i]) for i in range(limit)]


def _image_path_from_url(out_dir: Path, photo_id: str, img_url: str) -> Path:
    ext = os.path.splitext(str(img_url).split("?", 1)[0])[1].lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
    return out_dir / f"{photo_id}{ext}"


def _find_existing_image(out_dir: Path, photo_id: str, img_url: str) -> Optional[Path]:
    p = _image_path_from_url(out_dir, photo_id, img_url)
    if p.exists():
        return p
    for ext in IMAGE_EXTS:
        alt = out_dir / f"{photo_id}{ext}"
        if alt.exists():
            return alt
    return None


@dataclass(frozen=True)
class PhotoItem:
    photo_id: str
    review_id: str
    path: Path


def _iter_photos_from_manifest(
    manifest_path: Path,
    *,
    image_dir: Path,
    max_images_per_review: int,
) -> Iterable[PhotoItem]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review_id = str(row.get("review_id", "")).strip()
            if not review_id:
                continue
            plan = _extract_image_plan(
                str(row.get("review_imgsrc", "") or ""),
                str(row.get("review_photo_id", "") or ""),
                max_images=max_images_per_review,
            )
            for url, photo_id in plan:
                photo_id = str(photo_id).strip()
                if not photo_id:
                    continue
                p = _find_existing_image(image_dir, photo_id, url)
                if p is None:
                    continue
                yield PhotoItem(photo_id=photo_id, review_id=review_id, path=p)


def _load_resnet18_encoder(*, device: torch.device) -> Tuple[nn.Module, object, int]:
    try:  # torchvision>=0.13
        from torchvision.models import ResNet18_Weights  # type: ignore

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        preprocess = weights.transforms()
    except Exception:  # pragma: no cover
        model = resnet18(pretrained=True)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    img_dim = 512
    return model, preprocess, img_dim


def _encode_images(
    model: nn.Module,
    preprocess,
    paths: Sequence[Path],
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[Path], Optional[np.ndarray]]:
    if not paths:
        return [], None
    images = []
    kept_paths: List[Path] = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(preprocess(img))
            kept_paths.append(p)
        except Exception:
            continue
    if not images:
        return [], None

    x = torch.stack(images, dim=0).to(device)
    with torch.no_grad():
        y = model(x).detach().float().cpu().numpy()
    return kept_paths, y.astype(np.float32, copy=False)


def _encode_one_image(
    model: nn.Module,
    preprocess,
    path: Path,
    *,
    device: torch.device,
) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x).detach().float().cpu().numpy()
        return y[0].astype(np.float32, copy=False)
    except Exception:
        return None


def _load_existing_photo_ids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    try:
        dataset = ds.dataset(str(out_path), format="parquet")
        table = dataset.to_table(columns=["photo_id"])
        df = table.to_pandas()
        if "photo_id" not in df.columns:
            return set()
        return set(df["photo_id"].astype(str).tolist())
    except Exception:
        return set()


def _append_rows(
    *,
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
    photo_ids: Sequence[str],
    review_ids: Sequence[str],
    embs: np.ndarray,
) -> pq.ParquetWriter:
    if len(photo_ids) != len(review_ids) or len(photo_ids) != int(embs.shape[0]):
        raise ValueError("Mismatched batch lengths")
    data: Dict[str, object] = {"photo_id": list(map(str, photo_ids)), "review_id": list(map(str, review_ids))}
    for i in range(int(embs.shape[1])):
        data[f"img_emb_{i}"] = embs[:, i].astype(np.float32, copy=False)
    table = pa.Table.from_pydict(data)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    return writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch image_dir and incrementally write photo-level embeddings.")
    p.add_argument("--image-dir", type=str, default="openrice_images")
    p.add_argument("--manifest", type=str, default="openrice_images/selected_reviews.csv")
    p.add_argument("--out", type=str, default="", help="输出 parquet 文件路径（优先级高于 --out-dir）")
    p.add_argument("--out-dir", type=str, default="", help="输出目录（会写入 <out_dir>/photo_img_emb.parquet；若传了 --out 则忽略）")
    p.add_argument("--max-images-per-review", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64, help="每次编码多少张图片")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--poll-seconds", type=float, default=5.0, help="No new embeddings -> sleep this many seconds.")
    p.add_argument("--idle-exit-seconds", type=float, default=600.0, help="Exit if no progress for this long.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    image_dir = Path(args.image_dir)
    manifest = Path(args.manifest)
    if args.out:
        out_path = Path(args.out)
    elif args.out_dir:
        out_path = Path(args.out_dir) / "photo_img_emb.parquet"
    else:
        out_path = Path("survival_hsk") / "data" / "photo_img_emb.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    LOGGER.info("device=%s", device)

    model, preprocess, img_dim = _load_resnet18_encoder(device=device)
    LOGGER.info("encoder=resnet18 img_dim=%d", img_dim)

    processed: Set[str] = _load_existing_photo_ids(out_path)
    if processed:
        LOGGER.info("resume: loaded %d existing photo_id from %s", len(processed), out_path)

    writer: Optional[pq.ParquetWriter] = None
    last_progress = time.time()
    total_written = 0
    errors = 0

    try:
        while True:
            wrote_this_round = 0

            batch_paths: List[Path] = []
            batch_photo_ids: List[str] = []
            batch_review_ids: List[str] = []

            def flush_batch() -> None:
                nonlocal writer, wrote_this_round, total_written, last_progress, errors
                if not batch_paths:
                    return
                for pid, rid, p in zip(batch_photo_ids, batch_review_ids, batch_paths):
                    LOGGER.info("encoding photo_id=%s review_id=%s path=%s", pid, rid, p)
                kept_paths, embs = _encode_images(
                    model,
                    preprocess,
                    batch_paths,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                if embs is None or not kept_paths:
                    errors += 1
                    batch_paths.clear()
                    batch_photo_ids.clear()
                    batch_review_ids.clear()
                    return
                # 因为 _encode_images 会跳过坏图，严格对齐 kept_paths
                keep_set = set(kept_paths)
                kept_photo_ids = [pid for pid, p in zip(batch_photo_ids, batch_paths) if p in keep_set]
                kept_review_ids = [rid for rid, p in zip(batch_review_ids, batch_paths) if p in keep_set]
                if len(kept_photo_ids) != int(embs.shape[0]):
                    # 极端情况下顺序不一致，回退为逐张编码写入（避免错配）
                    errors += 1
                    for pid, rid, p in zip(batch_photo_ids, batch_review_ids, batch_paths):
                        if pid in processed:
                            continue
                        LOGGER.info("encoding(single) photo_id=%s review_id=%s path=%s", pid, rid, p)
                        one = _encode_one_image(model, preprocess, p, device=device)
                        if one is None:
                            continue
                        writer = _append_rows(
                            writer=writer,
                            out_path=out_path,
                            photo_ids=[pid],
                            review_ids=[rid],
                            embs=one.reshape(1, -1),
                        )
                        processed.add(pid)
                        LOGGER.info("wrote photo_id=%s", pid)
                        wrote_this_round += 1
                        total_written += 1
                        last_progress = time.time()
                else:
                    writer = _append_rows(
                        writer=writer,
                        out_path=out_path,
                        photo_ids=kept_photo_ids,
                        review_ids=kept_review_ids,
                        embs=embs,
                    )
                    for pid in kept_photo_ids:
                        processed.add(pid)
                        LOGGER.info("wrote photo_id=%s", pid)
                    wrote_this_round += len(kept_photo_ids)
                    total_written += len(kept_photo_ids)
                    last_progress = time.time()
                batch_paths.clear()
                batch_photo_ids.clear()
                batch_review_ids.clear()

            for item in _iter_photos_from_manifest(
                manifest,
                image_dir=image_dir,
                max_images_per_review=int(args.max_images_per_review),
            ):
                if item.photo_id in processed:
                    continue
                batch_paths.append(item.path)
                batch_photo_ids.append(item.photo_id)
                batch_review_ids.append(item.review_id)
                if len(batch_paths) >= int(args.batch_size):
                    flush_batch()

            flush_batch()

            if wrote_this_round > 0:
                LOGGER.info("wrote %d new photo embeddings (total_written=%d)", wrote_this_round, total_written)
                continue

            idle = time.time() - last_progress
            if idle >= float(args.idle_exit_seconds):
                LOGGER.info("no progress for %.1fs; exiting. out=%s", idle, out_path)
                break

            time.sleep(max(0.2, float(args.poll_seconds)))

    finally:
        if writer is not None:
            writer.close()

    LOGGER.info("DONE out=%s total_written=%d errors=%d", out_path, total_written, errors)


if __name__ == "__main__":
    main()
