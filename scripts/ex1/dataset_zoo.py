import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

from scripts.ex1.dataset import DatasetABC


class Zoo(DatasetABC):

    def __init__(self, classifier: str, hyper_parameters: dict, seed=42):
        self.classifier = classifier
        self._raw = fetch_ucirepo(id=111).data.original
        self._preprocessed = self._clean(self._raw)
        self.seed = seed
        self.hyper_parameters = hyper_parameters

    @property
    def raw(self):
        return self._raw

    @property
    def data(self):
        return self._preprocessed

    @property
    def X(self):
        return self._preprocessed.drop("type", axis=1)

    @property
    def y(self):
        return self._preprocessed["type"]

    @property
    def pipeline(self):
        """Returns a pipeline with preprocessing and classifier."""
        if self.classifier == "dt":
            classifier = DecisionTreeClassifier(random_state=self.seed, **self.hyper_parameters)
        elif self.classifier == "nb":
            classifier = CategoricalNB(**self.hyper_parameters)
        elif self.classifier == "mlp":
            classifier = MLPClassifier(random_state=self.seed, **self.hyper_parameters)
        else:
            raise ValueError(f"Classifier {self.classifier} not supported.")
        preprocessor = ZooPreprocessor(self.classifier)
        return make_pipeline(preprocessor, classifier)

    @staticmethod
    def _clean(X: pd.DataFrame):
        df = X.copy()
        df.drop('animal_name', axis='columns', inplace=True)

        return df


class ZooPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for the MembershipWoes dataset."""

    def __init__(self, classifier: str):
        self.classifier = classifier
        self.categorical = ['legs']
        self.one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
        if self.classifier == 'nb':
            self.ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).set_output(transform='pandas')

    def fit(self, X, y=None):
        if self.classifier == 'nb':
            self.ordinal.fit(X)
        else:
            self.one_hot.fit(X[self.categorical])
        return self

    def transform(self, X):
        df = X.copy()

        if self.classifier == 'nb':
            df = self.ordinal.transform(df).astype(int)
            df += 1
        else:
            # encode categorical features one-hot
            df = df.join(self.one_hot.transform(df[self.categorical]).astype(int))
            df.drop(self.categorical, axis=1, inplace=True)

        return df

