import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from scripts.ex1.dataset import DatasetABC


class Cancer(DatasetABC):

    def __init__(self, classifier: str, hyper_parameters: dict, test: bool = False, seed=42):
        self.classifier = classifier
        self.test = test
        self._raw = self._load_raw()
        self._data = self._clean(self._raw)
        self.seed = seed
        self.hyper_parameters = hyper_parameters

    @property
    def raw(self):
        return self._raw

    @property
    def data(self):
        return self._data

    @property
    def X(self) -> pd.DataFrame:
        """Return the features (preprocessed data)."""
        if self.test:
            return self._data
        else:
            return self._data.drop(['class', 'ID'], axis=1)

    @property
    def y(self) -> pd.Series:
        """Return the target (preprocessed data)."""
        if self.test:
            raise ValueError("y is not available for test data.")
        return self._data['class']

    @property
    def pipeline(self):
        """Returns a pipeline with preprocessing and classifier."""
        if self.classifier == "dt":
            classifier = DecisionTreeClassifier(random_state=self.seed, **self.hyper_parameters)
        elif self.classifier == "nb":
            classifier = GaussianNB()
            print("ignoring hyperparameters for gaussian naive bayes")
        elif self.classifier == "mlp":
            classifier = MLPClassifier(random_state=self.seed, **self.hyper_parameters)
        else:
            raise ValueError(f"Classifier {self.classifier} not supported.")
        preprocessor = CancerPreprocessor(self.classifier)
        return make_pipeline(
            preprocessor,
            classifier)

    def _load_raw(self) -> pd.DataFrame:
        """Load the dataset."""
        path = os.path.abspath("../../data/breast-cancer-diagnostic.shuf.tes.csv") if self.test \
            else os.path.abspath("../../data/breast-cancer-diagnostic.shuf.lrn.csv")
        with open(path, 'r') as f:
            df = pd.read_csv(f)
        return df

    @staticmethod
    def _clean(data: pd.DataFrame):
        df = data.copy()
        skewed = [' areaMean', ' compactnessMean', ' concavityMean', ' concavePointsMean',
                  ' radiusStdErr', ' perimeterStdErr', ' areaStdErr', ' compactnessStdErr',
                  ' concavityStdErr', ' symmetryStdErr', ' fractalDimensionStdErr', ' areaWorst',
                  ' compactnessWorst', ' concavityWorst']
        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
        df[skewed] = log_transformer.transform(df[skewed])

        return df


class CancerPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for the MembershipWoes dataset."""

    def __init__(self, classifier: str):
        self.classifier = classifier
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        df = X.copy()
        # scaling
        df = self.scaler.transform(df)

        return df
