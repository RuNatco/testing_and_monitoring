import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    columns = list(needed_columns) if needed_columns is not None else list(FEATURE_COLUMNS)
    unknown_columns = sorted(set(columns) - set(FEATURE_COLUMNS))
    if unknown_columns:
        raise ValueError(f'Unknown feature columns requested: {unknown_columns}')

    row = [getattr(req, column.replace('.', '_')) for column in columns]
    return pd.DataFrame([row], columns=columns)


def get_missing_features(frame: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for column in frame.columns:
        value = frame.iloc[0][column]
        if pd.isna(value):
            missing.append(column)
    return missing
