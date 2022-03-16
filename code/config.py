from typing import List

TRAIN_TEST_DIR:str = '../input/house-prices-advanced-regression-techniques/train.csv'
ALPHA:float = .95
QUANTILES:List[float] = [1-ALPHA, 0.5, ALPHA]
TRAIN_SIZE:float = 0.8