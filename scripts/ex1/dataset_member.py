import openml
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

from scripts.ex1.dataset import DatasetABC


class Member(DatasetABC):

    def __init__(self, classifier: str, hyper_parameters: dict, seed=42):
        self.classifier = classifier
        dataset = openml.datasets.get_dataset(44225)  # Download directly from OpenML
        self._raw, _, _, _ = dataset.get_data(dataset_format="dataframe")
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
    def X(self):
        return self._data.drop("MEMBERSHIP_STATUS", axis=1)

    @property
    def y(self):
        return self._data["MEMBERSHIP_STATUS"]

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
        preprocessor = MemberPreprocessor(self.classifier)
        return make_pipeline(preprocessor, classifier)

    @staticmethod
    def _clean(data: pd.DataFrame):
        df = data.copy()
        # drop irrelevant features
        relevant_features = [
            'MEMBERSHIP_TERM_YEARS', 'ANNUAL_FEES', 'MEMBER_MARITAL_STATUS', 'MEMBER_GENDER',
            'MEMBER_ANNUAL_INCOME', 'MEMBER_AGE_AT_ISSUE', 'ADDITIONAL_MEMBERS', 'PAYMENT_MODE',
            'MEMBERSHIP_STATUS']
        df = df[relevant_features]
        # fill missing values with 0
        strings = ['MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'PAYMENT_MODE']
        df[strings] = df[strings].fillna('unknown')
        for string in strings:
            df[string] = df[string].str.encode('utf-8').apply(lambda x: None if x is None else abs(hash(x)) % 1000)

        for i in ['ANNUAL_FEES', 'MEMBER_ANNUAL_INCOME']:
            q1 = df[i].quantile(0.25)
            q3 = df[i].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Exclude outliers
            df = df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]

        return df


class MemberPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for the MembershipWoes dataset."""

    def __init__(self, classifier: str):
        self.classifier = classifier
        self.categorical = ['MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'PAYMENT_MODE']
        self.continouos = ['ANNUAL_FEES', 'MEMBER_ANNUAL_INCOME']
        self.one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median').set_output(transform='pandas')
        if self.classifier == 'nb':
            self.ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).set_output(transform='pandas')
            self.kbins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile').set_output(transform='pandas')

    def fit(self, X, y=None):
        self.one_hot.fit(X[self.categorical])
        self.scaler.fit(X[self.continouos])
        self.imputer.fit(X)
        if self.classifier == 'nb':
            c = X.copy()
            c.update(self.imputer.transform(X)[self.continouos])
            self.kbins.fit(c[self.continouos])
            c.update(self.kbins.transform(c[self.continouos]))
            self.ordinal.fit(c)
        return self

    def transform(self, X):
        df = X.copy()

        # fill missing values
        df = self.imputer.transform(df)

        if self.classifier == 'nb':
            df.update(self.kbins.transform(df[self.continouos]))
            df = self.ordinal.transform(df).astype(int)
            df += 1
        else:
            # encode categorical features one-hot
            df = df.join(self.one_hot.transform(df[self.categorical]))
            df.drop(self.categorical, axis=1, inplace=True)
            # scaling
            df[self.continouos] = self.scaler.transform(df[self.continouos])


        return df

