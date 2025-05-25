from . import dump, model_selection
from .builtin_datasets import get_dataset_dir
from .dataset import Dataset
from .reader import Reader
from .trainset import Trainset

# 将 prediction_algorithms 的导入移到最后
from .prediction_algorithms import (
    BaselineOnly,
    NMF,
    NormalPredictor,
    Prediction,
    PredictionImpossible,
    SlopeOne,
    SVD,
    SVDpp,
)

__all__ = [
    "NormalPredictor",
    "BaselineOnly",
    "SVD",
    "SVDpp",
    "NMF",
    "SlopeOne",
    "PredictionImpossible",
    "Prediction",
    "Dataset",
    "Reader",
    "Trainset",
    "dump",
    "get_dataset_dir",
    "model_selection",
]

__version__ = "1.1.4"
