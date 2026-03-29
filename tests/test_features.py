import pytest

from ml_service.features import FEATURE_COLUMNS, to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_uses_all_feature_columns_by_default(sample_payload):
    request = PredictRequest(**sample_payload)

    frame = to_dataframe(request)

    assert frame.columns.tolist() == FEATURE_COLUMNS
    assert frame.iloc[0]['education.num'] == sample_payload['education.num']
    assert frame.iloc[0]['native.country'] == sample_payload['native.country']


def test_to_dataframe_keeps_only_requested_columns(sample_payload):
    request = PredictRequest(**sample_payload)

    frame = to_dataframe(request, needed_columns=['capital.gain', 'age'])

    assert frame.columns.tolist() == ['capital.gain', 'age']
    assert frame.iloc[0].tolist() == [sample_payload['capital.gain'], sample_payload['age']]


def test_to_dataframe_raises_for_unknown_columns(sample_payload):
    request = PredictRequest(**sample_payload)

    with pytest.raises(ValueError, match='Unknown feature columns requested'):
        to_dataframe(request, needed_columns=['age', 'unknown_feature'])
