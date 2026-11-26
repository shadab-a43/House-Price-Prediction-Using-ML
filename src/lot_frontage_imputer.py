
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class LotFrontageImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        X = X.copy()

        if "Lot Frontage" not in X.columns:
            raise ValueError("Lot Frontage column is missing.")

        if "Neighborhood" not in X.columns:
            raise ValueError("Neighborhood column is missing.")

        # Learn neighborhood medians from training data only.
        self.neighborhood_medians_ = (
            X.groupby("Neighborhood")["Lot Frontage"]
            .median()
        )

        # Learn global median from training data only.
        self.global_median_ = X["Lot Frontage"].median()

        return self

    def transform(self, X):

        check_is_fitted(
            self,
            ["neighborhood_medians_", "global_median_"]
        )

        X = X.copy()

        if "Lot Frontage" not in X.columns:
            raise ValueError("Lot Frontage column is missing.")

        if "Neighborhood" not in X.columns:
            raise ValueError("Neighborhood column is missing.")

        # Find the corresponding neighborhood median.
        neighborhood_values = X["Neighborhood"].map(
            self.neighborhood_medians_
        )

        # Fill missing values using neighborhood median.
        # If unavailable, use training global median.
        X["Lot Frontage"] = (
            X["Lot Frontage"]
            .fillna(neighborhood_values)
            .fillna(self.global_median_)
        )

        return X
