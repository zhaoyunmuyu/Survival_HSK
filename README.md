# Survival_HSK

本仓库主要包含 3 组代码：
- `survival_st_gcn/`：主模型（图结构 + 时序）
- `survival_kd/`：知识蒸馏（Teacher/Student，不使用图）
- `survival_new_data/`：OpenRice 预处理 + 面向 `survival_hsk/data` 的蒸馏/KD 流水线

## 词频统计（final_review_data）

`final_review_data/` 下包含 3 类评论文件（每类各有 1/2 两次筛选版本）。可以用下面脚本对不同文件/类别做词频统计：

```bash
python scripts/word_frequency.py --input-dir final_review_data --text-cols review_text --out-dir artifacts/word_freq
```

- 输出：`artifacts/word_freq/by_file/*_top.csv`（按文件）与 `artifacts/word_freq/by_group/*_top.csv`（按类别聚合）
- 分词：默认 `--tokenizer auto`（若安装了 `jieba` 会优先用；否则使用中文 2-gram 回退）
- 可选：`pip install jieba` 后用 `--tokenizer jieba` 得到更接近“词”的统计结果

## TF-IDF（final_review_data）

如果需要 TF-IDF（基于每条评论作为“文档”计算 df/idf，并输出按文件/类别聚合的 top-N token）：

```bash
python scripts/tfidf.py --input-dir final_review_data --text-cols review_text --out-dir artifacts/tfidf
```

- 输出：`artifacts/tfidf/by_file/*_top.csv`、`artifacts/tfidf/by_group/*_top.csv`、`artifacts/tfidf/by_group/tfidf_by_group_long.csv`
- 过滤阈值：`--min-tf`（总词频）与 `--min-df`（文档频次）

## 主题模型（final_review_data）

可以在抽样的评论文档上跑主题模型（更适合先做探索性分析）：

```bash
python scripts/topic_model.py --input-dir final_review_data --text-cols review_text --method nmf --n-topics 10 --max-docs 8000 --out-dir artifacts/topics
```

- `--method nmf`：基于 TF-IDF 的 NMF，速度快、效果稳定（推荐先用）
- `--method lda`：经典 LDA（基于词袋计数）
- 输出：`artifacts/topics/{method}/by_file/*_topics.csv` 与 `artifacts/topics/{method}/by_group/*_topics.csv`

## BERTopic（final_review_data）

BERTopic 基于句向量聚类 + c-TF-IDF，适合抽样做“语义主题”探索：

```bash
python scripts/bertopic_model.py --input-dir final_review_data --text-cols review_text --tokenizer char --char-ngram 2 --max-docs 5000 --n-topics 12 --out-dir artifacts/topics/bertopic
```

- 输出：`artifacts/topics/bertopic/by_file/*_topics.csv`（主题词）与 `artifacts/topics/bertopic/by_file/*_info.csv`（主题概览）
- 依赖：需要额外安装 `bertopic` / `sentence-transformers` / `umap-learn`
