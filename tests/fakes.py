from types import SimpleNamespace

import numpy as np


class FakeSklearnModel:
    def __init__(self, probability: float = 0.8, feature_names: list[str] | None = None) -> None:
        self.probability = probability
        self.feature_names_in_ = np.array(feature_names or ['age', 'capital.gain'])
        self.seen_frames = []

    def predict_proba(self, df):
        self.seen_frames.append(df.copy())
        return np.array([[1 - self.probability, self.probability]])


class FakeModelContainer:
    def __init__(
        self,
        model=None,
        run_id: str | None = None,
        set_error: Exception | None = None,
        next_model=None,
    ) -> None:
        self.model = model
        self.run_id = run_id
        self.set_error = set_error
        self.next_model = next_model or model

    def get(self):
        return SimpleNamespace(model=self.model, run_id=self.run_id)

    def set(self, run_id: str) -> None:
        if self.set_error is not None:
            raise self.set_error
        self.run_id = run_id
        self.model = self.next_model


class FakeDriftTracker:
    def __init__(self) -> None:
        self._summary = {
            'run_id': None,
            'feature_names': [],
            'reference_size': 0,
            'current_size': 0,
            'ready': False,
            'windows_ready': False,
            'dataset_drift': False,
            'drift_share': 0.0,
            'drifted_columns': 0,
            'total_columns': 0,
            'report_path': None,
            'report_json_path': None,
            'report_error': None,
        }

    def reset(self, run_id: str, feature_names: list[str]) -> dict:
        self._summary.update(
            {
                'run_id': run_id,
                'feature_names': list(feature_names),
                'reference_size': 0,
                'current_size': 0,
                'ready': False,
                'windows_ready': False,
                'dataset_drift': False,
                'drift_share': 0.0,
                'drifted_columns': 0,
                'total_columns': len(feature_names),
                'report_path': None,
                'report_json_path': None,
                'report_error': None,
            }
        )
        return dict(self._summary)

    def record(self, frame, run_id: str | None) -> dict:
        self._summary.update(
            {
                'run_id': run_id,
                'feature_names': frame.columns.tolist(),
                'reference_size': self._summary['reference_size'] + 1,
                'current_size': self._summary['current_size'],
            }
        )
        return dict(self._summary)

    def status(self) -> dict:
        return dict(self._summary)

    def latest_report(self) -> dict:
        if not self._summary['ready']:
            raise ValueError('Not enough data to build drift report yet')
        return dict(self._summary)

    def latest_report_html(self) -> str:
        if not self._summary['ready']:
            raise ValueError('Not enough data to build drift report yet')
        return '<html><body>drift</body></html>'
