# `survival_hsk` 代码结构

本目录用于承载基于 OpenRice 数据的预处理与建模代码，尽量与历史包解耦。

## 目录说明

- `survival_hsk/preprocess/`：数据预处理脚本（生成 `survival_hsk/data/` 中间结果）
- `survival_hsk/data/`：预处理产物落地目录（Parquet/CSV 等）

## 兼容说明（避免导入/命令踩坑）

- 源码实际位于 `survival_new_data/`；仓库同时提供 `survival_hsk/` 包作为兼容入口，因此文档中的 `python -m survival_hsk...` 可直接运行。
