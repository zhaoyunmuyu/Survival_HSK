# survival_hsk 数据说明

本目录存放的是在不修改 `openrice/` 原始数据前提下，由脚本生成的**中间处理结果**，供建模与分析使用。

目前包含的数据文件如下：

---

## 1. `restaurant_base.parquet` / `restaurant_base.csv`

> 来源：`openrice/时间标签/整合餐厅信息_rating_survive_fillna_refreshed.csv`  
> 生成脚本：`survival_new_data/preprocess/build_restaurant_base.py`

### 1.1 粒度与行数

- 粒度：每行对应一个餐厅（`restaurant_id` 唯一）。  
- 行数：当前为 `67762` 家餐厅。

### 1.2 字段结构概览

绝大部分字段直接继承自原始整合表，仅做轻量清洗；少数为新增派生字段。

- **标识与基础信息**  
  `restaurant_id`, `restaurant_name`, `restaurant_location`, `restaurant_region`,  
  `restaurant_region_num`, `Latitude_new`, `Longitude_new`, `restaurant_phone`,  
  `seat_num`, `note`, `link` 等。

- **评分与人气相关**  
  - 原始字段：`restaurant_rating`, `restaurant_rating_dist`, `restaurant_rating_num`,  
    `restaurant_save_num`, `restaurant_photo_num`, `good`, `ok`, `bad`。  
  - 按年份累计的环境/卫生评价：  
    `review_environment_cum_2001/2006/2011/2016/2021/2024`  
    `review_hygiene_cum_2001/2006/2011/2016/2021/2024`。

- **价格与营业时间**  
  - 价格：`restaurant_cost_range`。  
  - 原始营业时间文本：`open_time`。  
  - 派生营业时间结构特征：  
    `open_time_周中_天数`, `open_time_周末_天数`,  
    `open_time_周中_平均上班时间`, `open_time_周末_平均上班时间`。

- **支付方式（0/1 one-hot）**  
  如：`現金_payment`, `PayMe_payment`, `Visa_payment`, `Master_payment`,  
  `AlipayHK_payment`, `支付寶_payment`, `八達通_payment`, `銀聯_payment`,  
  `微信支付_payment`, `Apple Pay_payment`, `Google Pay_payment`,  
  `FPS_payment`, `PayPal_payment`, `MPay_payment`, `澳門通_payment` 等。

- **菜系与餐厅类型（0/1 one-hot）**  
  - 菜系：`粵菜 (廣東)`, `川菜 (四川)`, `日本菜`, `韓國菜`, `泰國菜`,  
    `意大利菜`, `法國菜`, `美國菜`, `葡國菜`, `中東菜`, `地中海菜`,  
    `墨西哥菜`, `巴西菜`, `多國菜` 等。  
  - 业态/类型：`茶餐廳/冰室`, `火鍋`, `燒烤`, `串燒`, `酒樓`, `中菜館`,  
    `甜品/糖水`, `咖啡店`, `樓上cafe`, `快餐店`, `Food Court`, `自助餐`,  
    `親子餐廳`, `主題餐廳`, `酒吧`, `美食車`, `外賣店`, `無肉餐單`,  
    `素食`, `動物友善餐廳`, `清真認證食品` 等（包含部分 `.1` 结尾的细分类字段）。

- **文本字段**  
  `restaurant_intro`（餐厅介绍）、`note`（备注）、`open_time`（原始营业时间描述）。

- **评论时间窗口（基于历史评论推断）**  
  `review_earliest_date`, `review_latest_date`,  
  `review_earliest_year`, `review_latest_year`。

- **生存标签与生存时间（建模核心）**  
  - `is_open`：当前是否仍在营业，0=已关闭，1=仍营业。  
  - `operation_early`, `operation_latest`：推断的最早/最晚营业日期。  
  - `operation_early_year`, `operation_latest_year`：对应年份。  
  - `data_source`：生存时间信息的来源标记。

- **新增派生字段**（由 `build_restaurant_base.py` 生成）  
  - `restaurant_rating_value`：  
    由 `restaurant_rating` 解析得到的浮点评分，例如 `"4星"` → `4.0`，`"4.5星"` → `4.5`。  
  - `operation_years`：  
    `operation_latest_year - operation_early_year`，粗略表示餐厅运营年数。  
  - `operation_early_year_missing`, `operation_latest_year_missing`：  
    相应年份是否缺失的指示列（缺失=1，非缺失=0）。

### 1.3 预处理步骤概述

在 `build_restaurant_base.py` 中，主要处理流程：

1. 读取 `openrice/时间标签/整合餐厅信息_rating_survive_fillna_refreshed.csv`。  
2. 删除索引型冗余列 `Unnamed: 0`（如果存在）。  
3. 将 `is_open` 规范为可空整数类型 `Int64`，方便后续建模。  
4. 解析文本评分 `restaurant_rating` 为数值列 `restaurant_rating_value`。  
5. 基于 `operation_early_year` 和 `operation_latest_year` 构造：  
   `operation_years`、`operation_early_year_missing`、`operation_latest_year_missing`。  
6. 将结果保存为 `restaurant_base.parquet` 和 `restaurant_base.csv`。

---

## 2. `reviews_agg_by_restaurant.parquet` / `reviews_agg_by_restaurant.csv`

> 来源：`openrice/整合餐厅评论信息.csv`  
> 生成脚本：`survival_new_data/preprocess/build_reviews_agg.py`

### 2.1 粒度与行数

- 粒度：每行对应一个餐厅（`restaurant_id` 唯一）。  
- 行数：取决于实际存在评论的餐厅数量（示例小样本约为 1500 左右，全量运行后会更多）。

### 2.2 字段结构

核心字段（评论统计特征）：

- `restaurant_id`：餐厅 ID。  
- 评论数量与评分：  
  - `n_reviews`：评论总数。  
  - `n_reviews_with_rating`：有口味评分（`review_taste`）的评论数量。  
  - `n_reviews_with_photo`：带图片的评论数量。  
  - `rating_mean`：口味评分平均值。  
  - `rating_std`：口味评分标准差。  
  - `rating_low_ratio`：低分评论占比（评分 ≤ 3 的比例）。  
  - `rating_high_ratio`：高分评论占比（评分 ≥ 4.5 的比例）。

- 文本长度与评论时间：  
  - `review_length_max`：单条评论最大长度（来自 `text_length`）。  
  - `review_length_mean`：评论长度平均值。  
  - `review_length_std`：评论长度标准差。  
  - `first_review_date`：最早评论日期。  
  - `last_review_date`：最近评论日期。  
  - `review_span_days`：最近评论日期与最早评论日期之间的天数。

- emoji / 情绪相关统计：  
  - `emoji_num_mean`：每条评论平均 emoji 数量（来自 `text_emoji_num`）。  
  - `pos_emoji_ratio`：正向 emoji 占比（`text_pos_emoji_num` / emoji 总数）。  
  - `neu_emoji_ratio`：中性 emoji 占比。  
  - `neg_emoji_ratio`：负向 emoji 占比。

> 注意：本表 **不再包含 TF-IDF/SVD 生成的文本向量字段**；  
> 评论文本的语义信息将在单独的 BERT 向量化脚本中处理（得到评论级或餐厅级 BERT 向量）。

### 2.3 预处理步骤概述

在 `build_reviews_agg.py` 中，主要处理流程：

1. **分块读取整合评论表**  
   - 路径：`openrice/整合餐厅评论信息.csv`。  
   - 只读取必要列：  
     `restaurant_id`, `review_id`, `review_date`,  
     `review_taste`, `text_length`,  
     `text_emoji_num`, `text_pos_emoji_num`, `text_neu_emoji_num`, `text_neg_emoji_num`,  
     `review_photo_num`。  
   - 使用 `chunksize` 分块处理，避免一次性读入全部数据。

2. **按餐厅聚合统计特征**  
   - 对每一块数据，逐行更新对应 `restaurant_id` 下的聚合统计：  
     - 评论数量：总数 / 有评分评论数 / 有图片评论数；  
     - 评分求和 / 平方和，用于后续计算均值和标准差；  
     - 评论长度求和 / 平方和 / 最大值；  
     - emoji 总数与正/中/负 emoji 求和；  
     - 最早 / 最近评论日期。

3. **计算派生指标**  
   - 基于上述聚合结果计算：  
     - `rating_mean`, `rating_std`, `rating_low_ratio`, `rating_high_ratio`；  
     - `review_length_mean`, `review_length_std`, `review_length_max`；  
     - `emoji_num_mean`, `pos_emoji_ratio`, `neu_emoji_ratio`, `neg_emoji_ratio`；  
     - `review_span_days`（评论时间跨度，单位天）。  
   - 将所有餐厅的聚合结果整理为一张 `DataFrame`。

4. **保存结果**  
   - 将聚合表保存为：  
     - `survival_hsk/data/reviews_agg_by_restaurant.parquet`  
     - `survival_hsk/data/reviews_agg_by_restaurant.csv`

---

## 3. `reviews_clean.parquet` / `reviews_clean.csv`

> 来源：`openrice/整合餐厅评论信息.csv`  
> 生成脚本：`survival_new_data/preprocess/build_reviews_clean.py`

### 3.1 粒度与行数

- 粒度：每行对应一条评论（`review_id` 唯一）。  
- 行数：与原始评论表一致（示例运行中使用 `--max-rows 200000`，形状为 `200000 × 35`；全量运行时会包含所有评论）。

### 3.2 字段结构与预处理

- 字段：与原始 `整合餐厅评论信息.csv` 基本一致，仅做轻量处理：  
  - 删除自动生成的索引列 `Unnamed: 0`（如果存在）；  
  - 将 `review_date` 解析为日期并统一格式化为 `YYYY-MM-DD` 字符串；  
  - 其他列（如 `review_id`, `restaurant_id`, `review_text`, `review_taste`, `review_photo_num` 等）保持原始含义不变。

- 作用：  
  - 作为「每条评论详细信息」的清洗版快照，便于：  
    - 与 `review_bert_emb.parquet`（评论级 BERT 向量，后续生成）按 `review_id` 关联；  
    - 做更细粒度的分析、调试或可视化（例如查某类评论的原文）。

- 生成方式（简述）：  
  - 分块读取 `openrice/整合餐厅评论信息.csv`；  
  - 对每个块删除 `Unnamed: 0`，规范 `review_date`；  
  - 流式写出到 `reviews_clean.parquet`（通过 ParquetWriter 追加）和 `reviews_clean.csv`（首块写表头，后续追加）。

---

## 4. 计划中的其他中间表

以下表尚未生成，作为后续扩展预留：

- **商家回复聚合特征表**  
  - 预计文件名：`replies_agg_by_restaurant.parquet` / `.csv`。  
  - 将包含回复率、回复延迟、回复文本特征等。

- **宏观经济特征与餐厅关联表**  
  - 预计文件名：`macro_by_restaurant.parquet` / `.csv`。  
  - 将通过餐厅所在地区与年份，将 62 维宏观特征拼接到餐厅粒度。

- **图片特征聚合表**  
  - 预计文件名：`image_features_by_restaurant.parquet` / `.csv`。  
  - 将基于评论/餐厅图片抽取视觉 embedding，并在餐厅层面聚合。

每新增一个数据文件，会在本 Markdown 中补充对应的小节，说明其来源、字段含义和预处理细节。
