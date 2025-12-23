# Survival_HSK

本仓库主要包含 3 组代码：
- `survival_st_gcn/`：主模型（图结构 + 时序）
- `survival_kd/`：知识蒸馏（Teacher/Student，不使用图）
- `survival_new_data/`：OpenRice 预处理 + 面向 `survival_hsk/data` 的蒸馏/KD 流水线

为避免导入/命令不一致导致“踩坑”，仓库提供兼容入口：
- `survival.*` 兼容映射到 `survival_st_gcn.*`
- `survival_hsk.*` 兼容映射到 `survival_new_data.*`
