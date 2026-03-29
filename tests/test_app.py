from fastapi.testclient import TestClient

from ml_service.app import _create_app, create_app
from ml_service.model import ModelLoadError

from tests.fakes import FakeDriftTracker, FakeModelContainer, FakeSklearnModel


def _build_client(monkeypatch, model_container):
    monkeypatch.setattr('ml_service.app.MODEL', model_container)
    monkeypatch.setattr('ml_service.app.DRIFT_TRACKER', FakeDriftTracker())
    return TestClient(_create_app(load_on_startup=False))


def test_health_returns_ok_when_model_loaded(monkeypatch, fake_model):
    client = _build_client(
        monkeypatch,
        FakeModelContainer(model=fake_model, run_id='run-123'),
    )

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'run_id': 'run-123'}


def test_health_returns_503_when_model_not_loaded(monkeypatch):
    client = _build_client(monkeypatch, FakeModelContainer(model=None, run_id=None))

    response = client.get('/health')

    assert response.status_code == 503
    assert response.json() == {'status': 'unavailable', 'run_id': None}


def test_metrics_endpoint_returns_prometheus_payload(monkeypatch):
    client = _build_client(monkeypatch, FakeModelContainer(model=None, run_id=None))

    response = client.get('/metrics')

    assert response.status_code == 200
    assert 'ml_service_http_requests_total' in response.text


def test_predict_returns_prediction_and_uses_model_features(monkeypatch, sample_payload):
    model = FakeSklearnModel(probability=0.73, feature_names=['capital.gain', 'age'])
    client = _build_client(
        monkeypatch,
        FakeModelContainer(model=model, run_id='run-123'),
    )

    response = client.post('/predict', json=sample_payload)

    assert response.status_code == 200
    assert response.json() == {'prediction': 1, 'probability': 0.73}
    assert model.seen_frames[0].columns.tolist() == ['capital.gain', 'age']
    assert model.seen_frames[0].iloc[0].tolist() == [sample_payload['capital.gain'], sample_payload['age']]


def test_predict_returns_422_when_required_feature_is_missing(monkeypatch, sample_payload):
    model = FakeSklearnModel(probability=0.73, feature_names=['capital.gain', 'age'])
    client = _build_client(
        monkeypatch,
        FakeModelContainer(model=model, run_id='run-123'),
    )
    sample_payload.pop('age')

    response = client.post('/predict', json=sample_payload)

    assert response.status_code == 422
    assert response.json() == {
        'detail': {
            'message': 'Required features are missing',
            'missing_features': ['age'],
        }
    }


def test_predict_returns_503_when_model_not_loaded(monkeypatch, sample_payload):
    client = _build_client(monkeypatch, FakeModelContainer(model=None, run_id=None))

    response = client.post('/predict', json=sample_payload)

    assert response.status_code == 503
    assert response.json() == {'detail': 'Model is not loaded yet'}


def test_update_model_switches_active_model(monkeypatch):
    container = FakeModelContainer(
        model=None,
        run_id=None,
        next_model=FakeSklearnModel(probability=0.42, feature_names=['age']),
    )
    client = _build_client(monkeypatch, container)

    response = client.post('/updateModel', json={'run_id': 'new-run'})

    assert response.status_code == 200
    assert response.json() == {'run_id': 'new-run'}
    assert container.run_id == 'new-run'
    assert container.model is not None


def test_update_model_returns_400_for_invalid_run_id(monkeypatch):
    container = FakeModelContainer(
        model=None,
        run_id=None,
        set_error=ModelLoadError('Failed to load model for run_id "missing-run"'),
    )
    client = _build_client(monkeypatch, container)

    response = client.post('/updateModel', json={'run_id': 'missing-run'})

    assert response.status_code == 400
    assert response.json() == {'detail': 'Failed to load model for run_id "missing-run"'}


def test_update_model_rejects_blank_run_id(monkeypatch):
    client = _build_client(monkeypatch, FakeModelContainer())

    response = client.post('/updateModel', json={'run_id': '   '})

    assert response.status_code == 422


def test_drift_status_returns_tracker_state(monkeypatch):
    tracker = FakeDriftTracker()
    tracker.reset(run_id='run-123', feature_names=['age'])
    monkeypatch.setattr('ml_service.app.MODEL', FakeModelContainer())
    monkeypatch.setattr('ml_service.app.DRIFT_TRACKER', tracker)
    client = TestClient(_create_app(load_on_startup=False))

    response = client.get('/drift/status')

    assert response.status_code == 200
    assert response.json()['run_id'] == 'run-123'


def test_drift_report_returns_409_while_warming_up(monkeypatch):
    client = _build_client(monkeypatch, FakeModelContainer())

    response = client.get('/drift/report')

    assert response.status_code == 409


def test_service_startup_loads_model_and_serves_prediction(monkeypatch, sample_payload):
    monkeypatch.setattr('ml_service.app.configure_mlflow', lambda: None)
    monkeypatch.setattr('ml_service.app.config.default_run_id', lambda: 'startup-run')
    monkeypatch.setattr(
        'ml_service.app.MODEL',
        FakeModelContainer(
            model=None,
            run_id=None,
            next_model=FakeSklearnModel(probability=0.61, feature_names=['capital.gain', 'age']),
        ),
    )
    monkeypatch.setattr('ml_service.app.DRIFT_TRACKER', FakeDriftTracker())

    with TestClient(create_app()) as client:
        response = client.post('/predict', json=sample_payload)

    assert response.status_code == 200
    assert response.json() == {'prediction': 1, 'probability': 0.61}
