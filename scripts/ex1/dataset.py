"""Provides an abstract base class for datasets."""
from abc import ABC, abstractmethod
import pandas as pd


class DatasetABC(ABC):
    """Abstract base class for datasets.
    All datasets should inherit from this class and implement these
    methods. This helps with comparability between datasets.
    """

    @abstractmethod
    def __init__(self, classifier: str, hyper_parameters: dict, seed: int):
        """Initialize the dataset with the classifier to be used."""
        ...

    @property
    @abstractmethod
    def raw(self) -> pd.DataFrame:
        """Return the raw data."""
        ...

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Return the preprocessed data."""
        ...

    @property
    @abstractmethod
    def X(self) -> pd.DataFrame:
        """Return the features (preprocessed data)."""
        ...

    @property
    @abstractmethod
    def y(self) -> pd.Series:
        """Return the target (preprocessed data)."""
        ...

    @abstractmethod
    def pipeline(self):
        """Returns a pipeline with preprocessing and classifier."""

