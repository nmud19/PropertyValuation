from typing import List

TRAIN_TEST_DIR: str = (
    "/Users/nimud/Downloads/house-prices-advanced-regression-techniques/train.csv"
)
ALPHA: float = 0.95
QUANTILES: List[float] = [1 - ALPHA, 0.5, ALPHA]
TRAIN_SIZE: float = 0.8
