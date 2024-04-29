"""Provides the LoanDataset class."""
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from scripts.ex1.dataset import DatasetABC
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, KBinsDiscretizer, OneHotEncoder
import pandas as pd
from scripts.ex1.encoding import BinaryEncoder


class Loan(DatasetABC):
    """Loan dataset."""

    def __init__(self, classifier: str, hyper_parameters: dict, seed=42, test: bool = False, ):
        """Initialize the dataset with the classifier to be used."""
        self.classifier = classifier
        self.hyper_parameters = hyper_parameters
        self.test = test
        self._raw = self._load_raw()
        self._data = self._clean(self._raw)
        self.seed = seed

    @property
    def raw(self) -> pd.DataFrame:
        """Return the raw data."""
        return self._raw

    @property
    def data(self) -> pd.DataFrame:
        """Return the cleansed data."""
        return self._data

    @property
    def X(self) -> pd.DataFrame:
        """Return the features (preprocessed data)."""
        if self.test:
            return self._data
        else:
            return self._data.drop(['grade', 'ID'], axis=1)

    @property
    def y(self) -> pd.Series:
        """Return the target (preprocessed data)."""
        if self.test:
            raise ValueError("y is not available for test data.")
        return self._data["grade"]

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
        preprocessor = LoanPreprocessor(self.classifier)
        return make_pipeline(preprocessor, classifier)

    def _load_raw(self) -> pd.DataFrame:
        """Load the dataset."""
        path = os.path.abspath("../../data/loan-10k.tes.csv") if self.test \
            else os.path.abspath("../../data/loan-10k.lrn.csv")
        with open(path, 'r') as f:
            df = pd.read_csv(f)
        return df

    @staticmethod
    def _clean(raw_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        drop = [
            # "ID",  # ID is not a feature, need it for test submission to kaggle
            "pymnt_plan",  # totally unbalanced (5 vs 9995)
            "out_prncp_inv",  # highly correlated with out_prncp
            "total_pymnt_inv",  # highly correlated with total_pymnt
            "total_rec_late_fee",  # unbalanced
            "recoveries",  # unbalanced
            "collection_recovery_fee",  # highly correlated with recoveries
            "collections_12_mths_ex_med",  # totally unbalanced (9833 vs rest)
            "policy_code",  # no information
            "application_type",  # totally unbalanced (95% vs 5%)
            "acc_now_delinq",  # totally unbalanced (9963 vs rest)
            "tot_coll_amt",  # unbalanced
            "total_rev_hi_lim",  # unbalanced
            "avg_cur_bal",  # unbalanced
            "bc_open_to_buy",  # unbalanced
            "chargeoff_within_12_mths",  # totally unbalanced (9929 vs rest)
            "delinq_amnt",  # totally unbalanced (9997 vs rest)
            "mo_sin_rcnt_rev_tl_op",  # unbalanced
            "mo_sin_rcnt_tl",  # unbalanced
            "mort_acc",  # unbalanced
            "mths_since_recent_bc",  # unbalanced
            "num_accts_ever_120_pd",  # unbalanced
            "num_tl_120dpd_2m",  # totally unbalanced (9992 vs 8)
            "num_tl_30dpd",  # totally unbalanced (9977 vs 23)
            "num_tl_op_past_12m",  # unbalanced
            "tax_liens",  # unbalanced
            "hardship_flag",  # totally unbalanced (9994 vs 6)
            "disbursement_method",  # totally unbalanced (96 vs 4)
            "debt_settlement_flag",  # totally unbalanced (9820 vs 180)
            "revol_bal",  # unbalanced
            "num_tl_90g_dpd_24m",  # unbalanced
        ]
        # outliers are dealt with automatically by QuantileTransformer

        combine_to_date = [
            ("issue_d_year", "issue_d_month"),
            ("earliest_cr_line_year", "earliest_cr_line_month"),
            ("last_pymnt_d_year", "last_pymnt_d_month"),
            ("last_credit_pull_d_year", "last_credit_pull_d_month"),
        ]
        df = raw_data.copy()

        # drop features
        df.drop(drop, axis=1, inplace=True)

        # boolean features
        df["longterm"] = df["term"].map({" 36 months": False, " 60 months": True})
        df.drop("term", axis=1, inplace=True)

        # convert to date
        for col1, col2 in combine_to_date:
            date = pd.DataFrame(dtype=int)
            date["year"], date["month"], date["day"] = df[col1].astype(int), df[col2].astype(int)+1, 1
            df[col1[:-4]+"date"] = pd.to_datetime(date).astype(int)  # convert to unix timestamp
            df.drop(col1, axis=1, inplace=True)
            df.drop(col2, axis=1, inplace=True)

        # convert to ordinal
        df["emp_length"] = OrdinalEncoder(
            categories=[["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]]
        ).fit_transform(df[["emp_length"]])

        return df


class LoanPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess the data."""
    def __init__(self, classifier: str):
        self.classifier = classifier
        if self.classifier == "dt":
            self.bin = BinaryEncoder()
        elif self.classifier == "nb":
            self.kbins = KBinsDiscretizer(
                n_bins=18, encode="ordinal", strategy="quantile"  # 18 bins seem to give good results
            )  # results in some bins being removed bc they are too small, but that's ok for naive bayes anyway.
            self.ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        elif self.classifier == 'mlp':
            self.one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
            self.scaler = QuantileTransformer().set_output(transform='pandas')
        self.categorical = [
            "home_ownership",
            "verification_status",
            "addr_state",
            "initial_list_status",
            "loan_status",
            "purpose",
        ]
        self.drop_nb = [
            "out_prncp",
            "funded_amnt",
            "funded_amnt_inv",
            "installment",
            "delinq_2yrs",
            "num_sats",
            "bc_util",
            "pub_rec_bankruptcies",
            "num_op_rev_tl",
            "num_rev_accts",
            "num_rev_tl_bal_gt_0",
        ]

    def fit(self, X, y=None):
        if self.classifier == "dt":
            self.bin.fit(X[self.categorical])
        elif self.classifier == "nb":
            df = X.drop(self.drop_nb, axis=1)
            self.kbins.fit(df.select_dtypes(exclude=[object, bool]))
            self.ord.fit(df)
        elif self.classifier == 'mlp':
            self.one_hot.fit(X[self.categorical])
            self.scaler.fit(X.drop(self.categorical, axis=1))
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.classifier == "dt":
            df = self._pp_dt(df)
        elif self.classifier == "nb":
            df = self._pp_nb(df)
        elif self.classifier == "mlp":
            df = self._pp_mlp(df)
        return df

    def _pp_dt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for decision tree."""
        # Binary encode categorical features
        bin_enc = self.bin.transform(df[self.categorical])
        # bin_enc = pd.DataFrame(OrdinalEncoder().fit_transform(df[categorical]), columns=categorical, dtype=int)
        # bin_enc = pd.get_dummies(df[categorical])
        df.drop(self.categorical, axis=1, inplace=True)
        df = pd.concat([df, bin_enc], axis=1)
        return df

    def _pp_nb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for naive bayes."""
        # these are highly correlated with other features
        # naive bayes assumes independence, so we drop them
        df.drop(self.drop_nb, axis=1, inplace=True)

        # transform float features to int using their 10-percentiles
        float_columns = df.select_dtypes(exclude=[object, bool]).columns

        df[float_columns] = self.kbins.transform(df[float_columns])

        # automatic ordinal transform for every string feature, bc categorical nb does not take ordering into account
        df = pd.DataFrame(self.ord.fit_transform(df), columns=df.columns, dtype=int)
        return df

    def _pp_mlp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for multilayer perceptron."""
        # One-Hot encode categorical features
        one_hot_enc = pd.DataFrame(
            self.one_hot.transform(df[self.categorical]),
            index=df.index,
            columns=self.one_hot.get_feature_names_out(self.categorical)
        )
        df.drop(self.categorical, axis=1, inplace=True)
        df.update(self.scaler.transform(df))
        df = df.join(one_hot_enc, how="outer")
        return df

    def plot(self):
        """Plot the data after preprocessing."""
        pass

