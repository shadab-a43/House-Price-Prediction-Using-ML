
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.feature_engineering import FeatureEngineeringTransformer
from src.missing_value_handler import MissingValueHandler
from src.lot_frontage_imputer import LotFrontageImputer


def create_preprocessing_pipeline(X_reference):

    # Determine feature types after feature engineering
    # and special missing-value handling.
    feature_sample = FeatureEngineeringTransformer().transform(
        X_reference
    )

    feature_sample = MissingValueHandler().transform(
        feature_sample
    )

    numerical_features = feature_sample.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = feature_sample.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine numerical and categorical pipelines
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    # Complete preprocessing pipeline
    preprocessing_pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineeringTransformer()
            ),
            (
                "missing_value_handler",
                MissingValueHandler()
            ),
            (
                "lot_frontage_imputer",
                LotFrontageImputer()
            ),
            (
                "column_transformer",
                column_transformer
            )
        ]
    )

    return preprocessing_pipeline
