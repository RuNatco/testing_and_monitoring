import pytest

from ml_service.model import ModelLoadError
from tests.fakes import FakeSklearnModel


@pytest.fixture
def sample_payload() -> dict:
    return {
        'age': 39,
        'workclass': 'State-gov',
        'fnlwgt': 77516,
        'education': 'Bachelors',
        'education.num': 13,
        'marital.status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital.gain': 2174,
        'capital.loss': 0,
        'hours.per.week': 40,
        'native.country': 'United-States',
    }


@pytest.fixture
def fake_model() -> FakeSklearnModel:
    return FakeSklearnModel()


@pytest.fixture
def invalid_run_error() -> ModelLoadError:
    return ModelLoadError('Failed to load model for run_id "missing-run"')
