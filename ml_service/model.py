import threading
from typing import Any, NamedTuple

from ml_service.mlflow_utils import load_model


class ModelError(RuntimeError):
    """Base exception for model lifecycle failures."""


class ModelNotLoadedError(ModelError):
    """Raised when the service tries to use a missing model."""


class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded from MLflow."""


class ModelData(NamedTuple):
    model: Any | None
    run_id: str | None


class Model:
    """
    Thread-safe container for the currently active model.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData(model=None, run_id=None)

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set(self, run_id: str) -> None:
        try:
            model = load_model(run_id=run_id)
        except Exception as exc:
            raise ModelLoadError(f'Failed to load model for run_id "{run_id}"') from exc
        with self.lock:
            self.data = ModelData(model=model, run_id=run_id)

    @property
    def features(self) -> list[str]:
        state = self.get()
        if state.model is None:
            raise ModelNotLoadedError('Model is not loaded yet')

        feature_names = getattr(state.model, 'feature_names_in_', None)
        if feature_names is None:
            raise ModelLoadError('Loaded model does not expose feature names')

        return list(feature_names)
