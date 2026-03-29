import pytest

from ml_service.model import Model, ModelLoadError, ModelNotLoadedError


def test_model_set_updates_model_and_run_id(monkeypatch, fake_model):
    monkeypatch.setattr('ml_service.model.load_model', lambda run_id: fake_model)
    container = Model()

    container.set('run-123')

    state = container.get()
    assert state.model is fake_model
    assert state.run_id == 'run-123'
    assert container.features == ['age', 'capital.gain']


def test_model_set_wraps_load_errors(monkeypatch):
    def raise_load_error(run_id):
        raise RuntimeError('mlflow is unavailable')

    monkeypatch.setattr('ml_service.model.load_model', raise_load_error)
    container = Model()

    with pytest.raises(ModelLoadError, match='Failed to load model for run_id "bad-run"'):
        container.set('bad-run')


def test_model_features_raise_when_model_not_loaded():
    container = Model()

    with pytest.raises(ModelNotLoadedError, match='Model is not loaded yet'):
        _ = container.features
