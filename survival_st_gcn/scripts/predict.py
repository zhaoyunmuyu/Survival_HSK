"""预测脚本

用法示例：
  - 使用训练好的 best 模型对 test 集做预测，并保存到 CSV：
    python -m survival.scripts.predict --checkpoint checkpoints/best_model.pt \
        --cache-dir preprocessed_data_cache --split test --batch-size 64 --out predictions_test.csv

说明：
  - 会从缓存中载入 Dataset 和年度图（若图缺失将使用零图兜底）；
  - 为避免 Windows 多进程问题，本脚本在推理时强制使用 num_workers=0；
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from survival_st_gcn.data.cache import get_dataloaders_with_cache
from survival_st_gcn.training.loop import _move_batch_to_device  # reuse util
from survival_st_gcn.models.survival import RestaurantSurvivalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with trained survival model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--cache-dir", type=str, default="preprocessed_data_cache", help="Cache directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV path")
    return parser.parse_args()


@torch.no_grad()
def run_prediction() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders: Dict[str, object] = get_dataloaders_with_cache(batch_size=args.batch_size, cache_dir=args.cache_dir, use_cache=True)
    yearly_graphs = loaders.get("yearly_graphs")
    if not yearly_graphs:
        raise RuntimeError("Yearly graphs not loaded. Ensure data files are accessible.")
    total_rests = yearly_graphs[0].total_rests

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RestaurantSurvivalModel(total_rests=total_rests, hidden_dim=512).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = {
        "train": loaders["train_loader"].dataset,
        "val": loaders["val_loader"].dataset,
        "test": loaders["test_loader"].dataset,
    }[args.split]
    # 推理时，强制 num_workers=0，避免 Windows 共享内存问题
    inf_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    all_probs = []
    all_labels = []
    all_ids = []
    for batch in inf_loader:
        _move_batch_to_device(batch, device)
        outputs = model(batch, yearly_graphs, device)
        probs = torch.sigmoid(outputs).squeeze(-1).detach().cpu().numpy()
        labels = (1.0 - batch["is_open"]).detach().cpu().numpy().astype(np.float32)
        rest_ids = batch["restaurant_id"].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
        all_ids.append(rest_ids)

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels).reshape(-1)
    ids_arr = np.concatenate(all_ids).reshape(-1)

    df = pd.DataFrame({
        "probability": probs_arr,
        "true_label": labels_arr,
        "restaurant_id": ids_arr,
    })
    df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out} (rows={len(df)})")


if __name__ == "__main__":
    run_prediction()

