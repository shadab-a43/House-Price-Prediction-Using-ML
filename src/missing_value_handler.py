
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueHandler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # This transformer does not learn statistical values.
        return self

    def transform(self, X):
        X = X.copy()

        # ------------------------------------------------
        # 1. GARAGE
        # ------------------------------------------------

        garage_categorical = [
            "Garage Type",
            "Garage Finish",
            "Garage Qual",
            "Garage Cond"
        ]

        if (
            "Garage Cars" in X.columns
            and "Garage Area" in X.columns
        ):

            no_garage = (
                X["Garage Cars"].eq(0)
                & X["Garage Area"].eq(0)
            )

            for column in garage_categorical:

                if column in X.columns:

                    X.loc[
                        no_garage & X[column].isna(),
                        column
                    ] = "None"

                    X.loc[
                        ~no_garage & X[column].isna(),
                        column
                    ] = "Unknown"

        # ------------------------------------------------
        # 2. BASEMENT
        # ------------------------------------------------

        basement_categorical = [
            "Bsmt Qual",
            "Bsmt Cond",
            "Bsmt Exposure",
            "BsmtFin Type 1",
            "BsmtFin Type 2"
        ]

        if "Total Bsmt SF" in X.columns:

            no_basement = X["Total Bsmt SF"].eq(0)

            for column in basement_categorical:

                if column in X.columns:

                    X.loc[
                        no_basement & X[column].isna(),
                        column
                    ] = "None"

                    X.loc[
                        ~no_basement & X[column].isna(),
                        column
                    ] = "Unknown"

        # ------------------------------------------------
        # 3. MASONRY VENEER
        # ------------------------------------------------

        if (
            "Mas Vnr Type" in X.columns
            and "Mas Vnr Area" in X.columns
        ):

            missing_type = X["Mas Vnr Type"].isna()
            no_masonry = X["Mas Vnr Area"].eq(0)

            X.loc[
                missing_type & no_masonry,
                "Mas Vnr Type"
            ] = "None"

            X.loc[
                missing_type & ~no_masonry,
                "Mas Vnr Type"
            ] = "Unknown"

        # ------------------------------------------------
        # 4. POOL QUALITY
        # ------------------------------------------------

        if (
            "Pool QC" in X.columns
            and "Pool Area" in X.columns
        ):

            no_pool = X["Pool Area"].eq(0)

            X.loc[
                X["Pool QC"].isna() & no_pool,
                "Pool QC"
            ] = "None"

            X.loc[
                X["Pool QC"].isna() & ~no_pool,
                "Pool QC"
            ] = "Unknown"

        # ------------------------------------------------
        # 5. FIREPLACE QUALITY
        # ------------------------------------------------

        if (
            "Fireplace Qu" in X.columns
            and "Fireplaces" in X.columns
        ):

            no_fireplace = X["Fireplaces"].eq(0)

            X.loc[
                X["Fireplace Qu"].isna() & no_fireplace,
                "Fireplace Qu"
            ] = "None"

            X.loc[
                X["Fireplace Qu"].isna() & ~no_fireplace,
                "Fireplace Qu"
            ] = "Unknown"

        # ------------------------------------------------
        # 6. OTHER ABSENCE-BASED CATEGORIES
        # ------------------------------------------------

        absence_categories = [
            "Alley",
            "Fence",
            "Misc Feature"
        ]

        for column in absence_categories:

            if column in X.columns:
                X[column] = X[column].fillna("None")

        return X
