from .utils import dump
from .utils.builtin_datasets import get_dataset_dir
from .core.dataset import Dataset
from .utils.reader import Reader
from .core.trainset import Trainset
from . import model_selection
from .utils import accuracy

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
    "accuracy",
]

__version__ = "1.1.4"
