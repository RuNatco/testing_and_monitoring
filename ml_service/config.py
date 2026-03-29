import os

MODEL_ARTIFACT_PATH = 'model'


def tracking_uri() -> str:
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        raise RuntimeError('Please set MLFLOW_TRACKING_URI')
    return tracking_uri


def default_run_id() -> str:
    """
    Returns model URI for startup.
    """

    default_run_id = os.getenv('DEFAULT_RUN_ID')
    if not default_run_id:
        raise RuntimeError('Set DEFAULT_RUN_ID to load model on startup')
    return default_run_id


def graphite_host() -> str | None:
    return os.getenv('GRAPHITE_HOST')


def graphite_port() -> int:
    return int(os.getenv('GRAPHITE_PORT', '2003'))


def graphite_prefix() -> str:
    return os.getenv('GRAPHITE_PREFIX', 'ml_service')


def graphite_timeout_seconds() -> float:
    return float(os.getenv('GRAPHITE_TIMEOUT_SECONDS', '1.0'))


def drift_reference_size() -> int:
    return int(os.getenv('DRIFT_REFERENCE_SIZE', '50'))


def drift_current_size() -> int:
    return int(os.getenv('DRIFT_CURRENT_SIZE', '50'))


def drift_report_dir() -> str:
    return os.getenv('DRIFT_REPORT_DIR', 'artifacts/evidently')


def evidently_url() -> str | None:
    return os.getenv('EVIDENTLY_URL')


def evidently_project_id() -> str | None:
    return os.getenv('EVIDENTLY_PROJECT_ID')
