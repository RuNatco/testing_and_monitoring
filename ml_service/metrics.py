import logging
from threading import Lock

import graphyte
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from ml_service import config

LOGGER = logging.getLogger(__name__)

HTTP_REQUESTS_TOTAL = Counter(
    'ml_service_http_requests_total',
    'Total number of HTTP requests handled by the service',
    ['method', 'path', 'status_code'],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'ml_service_http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'path'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREPROCESSING_DURATION_SECONDS = Histogram(
    'ml_service_preprocessing_duration_seconds',
    'Preprocessing latency in seconds',
    ['run_id'],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
INFERENCE_DURATION_SECONDS = Histogram(
    'ml_service_inference_duration_seconds',
    'Model inference latency in seconds',
    ['run_id'],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTIONS_TOTAL = Counter(
    'ml_service_predictions_total',
    'Total number of predictions grouped by class and active model',
    ['prediction', 'run_id'],
)
PREDICTION_PROBABILITY = Histogram(
    'ml_service_prediction_probability',
    'Distribution of prediction probabilities',
    ['run_id'],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_LOADS_TOTAL = Counter(
    'ml_service_model_loads_total',
    'Total number of model load attempts',
    ['result', 'run_id'],
)
MODEL_LOADED = Gauge(
    'ml_service_model_loaded',
    'Whether a model is currently loaded',
)
MODEL_INFO = Gauge(
    'ml_service_model_info',
    'Information about the currently active model',
    ['run_id'],
)
MODEL_TYPE_INFO = Gauge(
    'ml_service_model_type_info',
    'Type of the currently active model',
    ['run_id', 'model_type'],
)
MODEL_REQUIRED_FEATURE_INFO = Gauge(
    'ml_service_model_required_feature_info',
    'Feature required by the currently active model',
    ['run_id', 'feature'],
)
FEATURE_OBSERVATIONS_TOTAL = Counter(
    'ml_service_feature_observations_total',
    'Observed feature values grouped by feature name',
    ['run_id', 'feature'],
)
FEATURE_MISSING_TOTAL = Counter(
    'ml_service_feature_missing_total',
    'Missing required features grouped by feature name',
    ['run_id', 'feature'],
)
FEATURE_NUMERIC_VALUE = Histogram(
    'ml_service_feature_numeric_value',
    'Distribution of numeric feature values',
    ['run_id', 'feature'],
    buckets=(0, 1, 5, 10, 20, 30, 40, 50, 100, 500, 1000, 5000, 10000, 50000, 100000),
)
FEATURE_CATEGORICAL_TOTAL = Counter(
    'ml_service_feature_categorical_total',
    'Observed categorical feature values',
    ['run_id', 'feature', 'value'],
)
DATA_DRIFT_READY = Gauge(
    'ml_service_data_drift_ready',
    'Whether enough data exists to build a drift report',
)
DATA_DRIFT_DETECTED = Gauge(
    'ml_service_data_drift_detected',
    'Whether evidently detected dataset drift',
)
DATA_DRIFT_SHARE = Gauge(
    'ml_service_data_drift_share',
    'Share of drifting columns in the latest evidently report',
)
DATA_DRIFT_REFERENCE_ROWS = Gauge(
    'ml_service_data_drift_reference_rows',
    'Number of rows in the evidently reference window',
)
DATA_DRIFT_CURRENT_ROWS = Gauge(
    'ml_service_data_drift_current_rows',
    'Number of rows in the evidently current window',
)
DRIFT_STATUS_UPDATES_TOTAL = Counter(
    'ml_service_drift_status_updates_total',
    'Total number of drift status updates',
    ['result'],
)


def _sanitize_path(path: str) -> str:
    clean = path.strip('/').replace('-', '_').replace('/', '.')
    return clean or 'root'


class MetricsRecorder:
    def __init__(self) -> None:
        self._graphite_initialized = False
        self._graphite_lock = Lock()
        self._model_lock = Lock()
        self._active_run_id: str | None = None
        self._active_model_type: str | None = None
        self._active_features: list[str] = []

    def _graphite_enabled(self) -> bool:
        return bool(config.graphite_host())

    def _init_graphite(self) -> None:
        if not self._graphite_enabled() or self._graphite_initialized:
            return

        with self._graphite_lock:
            if self._graphite_initialized:
                return
            try:
                graphyte.init(
                    config.graphite_host(),
                    port=config.graphite_port(),
                    prefix=config.graphite_prefix(),
                    timeout=config.graphite_timeout_seconds(),
                )
            except Exception:
                LOGGER.exception('Failed to initialize Graphite client')
                return
            self._graphite_initialized = True

    def _send_graphite(self, metric_name: str, value: float) -> None:
        if not self._graphite_enabled():
            return

        self._init_graphite()
        if not self._graphite_initialized:
            return

        try:
            graphyte.send(metric_name, value)
        except Exception:
            LOGGER.warning('Failed to send Graphite metric %s', metric_name, exc_info=True)

    def record_http_request(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        method_label = method.upper()
        path_label = path
        status_label = str(status_code)

        HTTP_REQUESTS_TOTAL.labels(
            method=method_label,
            path=path_label,
            status_code=status_label,
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method_label,
            path=path_label,
        ).observe(duration_seconds)

        graphite_path = _sanitize_path(path)
        graphite_method = method.lower()
        self._send_graphite(
            f'http.requests.{graphite_path}.{graphite_method}.status_{status_code}',
            1,
        )
        self._send_graphite(
            f'http.latency_ms.{graphite_path}.{graphite_method}',
            duration_seconds * 1000.0,
        )

    def record_preprocessing_duration(self, run_id: str | None, duration_seconds: float) -> None:
        run_id_label = run_id or 'unknown'
        PREPROCESSING_DURATION_SECONDS.labels(run_id=run_id_label).observe(duration_seconds)
        self._send_graphite('preprocessing.duration_ms', duration_seconds * 1000.0)

    def record_inference_duration(self, run_id: str | None, duration_seconds: float) -> None:
        run_id_label = run_id or 'unknown'
        INFERENCE_DURATION_SECONDS.labels(run_id=run_id_label).observe(duration_seconds)
        self._send_graphite('inference.duration_ms', duration_seconds * 1000.0)

    def record_prediction(self, prediction: int, probability: float, run_id: str | None) -> None:
        run_id_label = run_id or 'unknown'

        PREDICTIONS_TOTAL.labels(
            prediction=str(prediction),
            run_id=run_id_label,
        ).inc()
        PREDICTION_PROBABILITY.labels(run_id=run_id_label).observe(probability)

        self._send_graphite(f'predictions.class_{prediction}', 1)
        self._send_graphite('predictions.probability', probability)

    def record_feature_values(self, run_id: str | None, row: dict) -> None:
        run_id_label = run_id or 'unknown'

        for feature, value in row.items():
            if value is None:
                continue

            FEATURE_OBSERVATIONS_TOTAL.labels(run_id=run_id_label, feature=feature).inc()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                FEATURE_NUMERIC_VALUE.labels(run_id=run_id_label, feature=feature).observe(float(value))
                self._send_graphite(f'features.numeric.{feature}', float(value))
            else:
                safe_value = str(value).replace(' ', '_')[:100]
                FEATURE_CATEGORICAL_TOTAL.labels(
                    run_id=run_id_label,
                    feature=feature,
                    value=safe_value,
                ).inc()
                self._send_graphite(f'features.categorical.{feature}.{safe_value}', 1)

    def record_missing_features(self, run_id: str | None, missing_features: list[str]) -> None:
        run_id_label = run_id or 'unknown'
        for feature in missing_features:
            FEATURE_MISSING_TOTAL.labels(run_id=run_id_label, feature=feature).inc()
            self._send_graphite(f'features.missing.{feature}', 1)

    def record_model_load(self, run_id: str, success: bool) -> None:
        result = 'success' if success else 'error'
        MODEL_LOADS_TOTAL.labels(result=result, run_id=run_id).inc()
        self._send_graphite(f'model.load.{result}', 1)

        if success:
            self.set_model_state(run_id=run_id, loaded=True)

    def set_model_state(self, run_id: str | None, loaded: bool) -> None:
        MODEL_LOADED.set(1 if loaded else 0)
        self._send_graphite('model.loaded', 1 if loaded else 0)

        with self._model_lock:
            if self._active_run_id is not None:
                try:
                    MODEL_INFO.remove(self._active_run_id)
                except KeyError:
                    pass
                self._active_run_id = None

            if loaded and run_id:
                MODEL_INFO.labels(run_id=run_id).set(1)
                self._active_run_id = run_id

    def set_model_metadata(self, run_id: str, model_type: str, features: list[str]) -> None:
        with self._model_lock:
            if self._active_run_id and self._active_model_type:
                try:
                    MODEL_TYPE_INFO.remove(self._active_run_id, self._active_model_type)
                except KeyError:
                    pass
            if self._active_run_id:
                for feature in self._active_features:
                    try:
                        MODEL_REQUIRED_FEATURE_INFO.remove(self._active_run_id, feature)
                    except KeyError:
                        pass

            MODEL_TYPE_INFO.labels(run_id=run_id, model_type=model_type).set(1)
            for feature in features:
                MODEL_REQUIRED_FEATURE_INFO.labels(run_id=run_id, feature=feature).set(1)

            self._active_run_id = run_id
            self._active_model_type = model_type
            self._active_features = list(features)

        self._send_graphite(f'model.type.{model_type}', 1)

    def record_drift_status(self, summary: dict) -> None:
        ready = bool(summary.get('ready', False))
        dataset_drift = bool(summary.get('dataset_drift', False))
        drift_share = float(summary.get('drift_share', 0.0))
        reference_size = int(summary.get('reference_size', 0))
        current_size = int(summary.get('current_size', 0))

        DATA_DRIFT_READY.set(1 if ready else 0)
        DATA_DRIFT_DETECTED.set(1 if dataset_drift else 0)
        DATA_DRIFT_SHARE.set(drift_share)
        DATA_DRIFT_REFERENCE_ROWS.set(reference_size)
        DATA_DRIFT_CURRENT_ROWS.set(current_size)

        result = 'ready' if ready else 'warming_up'
        DRIFT_STATUS_UPDATES_TOTAL.labels(result=result).inc()

        self._send_graphite('drift.ready', 1 if ready else 0)
        self._send_graphite('drift.detected', 1 if dataset_drift else 0)
        self._send_graphite('drift.share', drift_share)
        self._send_graphite('drift.reference_rows', reference_size)
        self._send_graphite('drift.current_rows', current_size)

    @staticmethod
    def render_prometheus_metrics() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST


METRICS = MetricsRecorder()
