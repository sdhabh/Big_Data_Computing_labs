from . import dump, model_selection
from .builtin_datasets import get_dataset_dir

from .dataset import Dataset

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
from .reader import Reader
from .trainset import Trainset

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
