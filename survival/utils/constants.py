"""全局常量定义

说明：
- 该文件集中管理项目中会被多处引用的常量，避免到处硬编码；
- 如果需要统一调参（例如评论序列长度、特征维度等），只需在此处修改；
- 仅包含常量，不包含任何业务逻辑。
"""

from typing import Dict, List

GLOBAL_SEED: int = 42  # 全局随机种子（用于 Python/NumPy/PyTorch）

MAX_REVIEWS_PER_RESTAURANT: int = 128  # 每家餐厅最多纳入的评论条数（序列长度）
TEXT_VECTOR_DIM: int = 300             # 单条评论的文本向量维度
IMAGE_VECTOR_DIM: int = 512            # 单条评论的图片向量维度（已抽取的图像特征）
REVIEW_FEATURE_DIM: int = 8            # 单条评论的结构化特征维度（时间段、评分等）
MAX_REVIEW_PHOTOS: int = 6             # 单条评论最多聚合的图片特征数量（求均值）

REGION_ORDER: List[str] = [  # 区域编码固定顺序，确保映射稳定
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
]
REGION_MAPPING: Dict[str, int] = {code: idx + 1 for idx, code in enumerate(REGION_ORDER)}  # 区域编码 -> 序号（从1开始）
