import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class BinaryEncoder:
    """This encoder encodes a categorical feature into multiple binary features.
    It produces less features than OneHotEncoder, but without information loss.
    Unknown values will be encoded as 0. Encoding of known values starts at 1.
    """

    def __init__(self, categories='auto'):
        """Initialize the encoder."""
        self.ordinal = OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1).set_output(transform='pandas')
        self.index = None

    def fit(self, X: pd.DataFrame):
        """Fit the encoder."""
        self.ordinal.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        trans_ord = self.ordinal.transform(X)
        trans_ord += 1  # make unknown values 0
        result = pd.DataFrame(index=trans_ord.index)
        for name, col in trans_ord.items():
            max_digits = int(col.max()).bit_length()
            col_lists = col.astype(int).map('{:b}'.format).str.rjust(max_digits, '0').map(list).tolist()
            new_df = pd.DataFrame(col_lists, index=trans_ord.index,  columns=[f"{name}_2^{i-1}" for i in range(max_digits, 0, -1)])
            result = result.join(new_df, how="outer")
        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        self.fit(X)
        return self.transform(X)
