import json
from collections import deque
from copy import deepcopy
from html import escape
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

from ml_service import config


def _find_first(payload: Any, key: str) -> Any | None:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_first(value, key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_first(item, key)
            if found is not None:
                return found
    return None


def _extract_drift_summary(payload: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    metrics = payload.get('metrics', [])
    for metric in metrics:
        config = metric.get('config', {})
        metric_type = config.get('type', '')
        if metric_type.endswith('DriftedColumnsCount'):
            value = metric.get('value', {})
            drifted_columns = int(float(value.get('count', 0)))
            share = float(value.get('share', 0.0))
            threshold = float(config.get('drift_share', 0.5))

            total_columns = len(feature_names)
            if share > 0 and drifted_columns > 0:
                total_columns = max(total_columns, int(round(drifted_columns / share)))

            return {
                'dataset_drift': share >= threshold,
                'drift_share': share,
                'drifted_columns': drifted_columns,
                'total_columns': total_columns,
            }

    dataset_drift = bool(
        _find_first(payload, 'dataset_drift')
        or _find_first(payload, 'drift_detected')
        or False
    )
    drift_share = float(
        _find_first(payload, 'share_of_drifted_columns')
        or _find_first(payload, 'drift_share')
        or 0.0
    )
    drifted_columns = int(
        _find_first(payload, 'number_of_drifted_columns')
        or _find_first(payload, 'drifted_columns_count')
        or 0
    )
    total_columns = int(
        _find_first(payload, 'number_of_columns')
        or _find_first(payload, 'total_columns')
        or len(feature_names)
    )

    return {
        'dataset_drift': dataset_drift,
        'drift_share': drift_share,
        'drifted_columns': drifted_columns,
        'total_columns': total_columns,
    }


class DriftTracker:
    def __init__(
        self,
        reference_size: int | None = None,
        current_size: int | None = None,
        report_dir: str | None = None,
    ) -> None:
        self.reference_size = reference_size or config.drift_reference_size()
        self.current_size = current_size or config.drift_current_size()
        self.report_dir = Path(report_dir or config.drift_report_dir())
        self._lock = RLock()
        self._run_id: str | None = None
        self._feature_names: list[str] = []
        self._reference_rows: deque[dict[str, Any]] = deque(maxlen=self.reference_size)
        self._current_rows: deque[dict[str, Any]] = deque(maxlen=self.current_size)
        self._last_summary = self._empty_summary()

    @staticmethod
    def _empty_summary() -> dict[str, Any]:
        return {
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
            'workspace_url': config.evidently_url(),
            'workspace_project_id': config.evidently_project_id(),
            'workspace_synced': False,
            'workspace_error': None,
        }

    def reset(self, run_id: str, feature_names: list[str]) -> dict[str, Any]:
        with self._lock:
            self._run_id = run_id
            self._feature_names = list(feature_names)
            self._reference_rows = deque(maxlen=self.reference_size)
            self._current_rows = deque(maxlen=self.current_size)
            self._last_summary = self._empty_summary()
            self._last_summary.update(
                {
                    'run_id': run_id,
                    'feature_names': list(feature_names),
                }
            )
            return deepcopy(self._last_summary)

    def record(self, frame: pd.DataFrame, run_id: str | None) -> dict[str, Any]:
        row = frame.iloc[0].to_dict()
        with self._lock:
            if run_id != self._run_id:
                self.reset(run_id=run_id or 'unknown', feature_names=frame.columns.tolist())

            if len(self._reference_rows) < self.reference_size:
                self._reference_rows.append(row)
            else:
                self._current_rows.append(row)

            ready = (
                len(self._reference_rows) == self.reference_size
                and len(self._current_rows) == self.current_size
            )
            self._last_summary.update(
                {
                    'run_id': self._run_id,
                    'feature_names': list(self._feature_names),
                    'reference_size': len(self._reference_rows),
                    'current_size': len(self._current_rows),
                    'windows_ready': ready,
                    'ready': False,
                    'report_path': None,
                    'report_json_path': None,
                    'report_error': None,
                    'workspace_synced': False,
                    'workspace_error': None,
                }
            )

            if ready:
                try:
                    self._update_report()
                except Exception as exc:
                    self._last_summary.update(
                        {
                            'ready': False,
                            'report_path': None,
                            'report_json_path': None,
                            'report_error': str(exc),
                            'workspace_synced': False,
                        }
                    )

            return deepcopy(self._last_summary)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._last_summary)

    def latest_report(self) -> dict[str, Any]:
        with self._lock:
            if not self._last_summary['ready'] or not self._last_summary.get('report_json_path'):
                raise ValueError('Not enough data to build drift report yet')
            return deepcopy(self._last_summary)

    def latest_report_html(self) -> str:
        with self._lock:
            report_path = self._last_summary.get('report_path')
            if not report_path:
                raise ValueError('Not enough data to build drift report yet')
            return Path(report_path).read_text(encoding='utf-8')

    def _update_report(self) -> None:
        reference_df = pd.DataFrame(list(self._reference_rows), columns=self._feature_names)
        current_df = pd.DataFrame(list(self._current_rows), columns=self._feature_names)

        report = Report([DataDriftPreset()])
        evaluation = report.run(current_data=current_df, reference_data=reference_df)
        report_result = evaluation or report

        self.report_dir.mkdir(parents=True, exist_ok=True)
        html_path = self.report_dir / 'latest_drift_report.html'
        json_path = self.report_dir / 'latest_drift_report.json'
        payload = self._report_as_dict(report_result)
        self._write_report_json(report_result, json_path, payload)
        self._write_report_html(report_result, html_path, payload)
        summary = _extract_drift_summary(payload, self._feature_names)
        workspace_synced = False
        workspace_error = None

        if self._workspace_enabled():
            try:
                self._upload_to_workspace(report_result)
                workspace_synced = True
            except Exception as exc:
                workspace_error = str(exc)

        self._last_summary.update(
            {
                'ready': True,
                'dataset_drift': summary['dataset_drift'],
                'drift_share': summary['drift_share'],
                'drifted_columns': summary['drifted_columns'],
                'total_columns': summary['total_columns'],
                'report_path': str(html_path),
                'report_json_path': str(json_path),
                'report_error': None,
                'workspace_url': config.evidently_url(),
                'workspace_project_id': config.evidently_project_id(),
                'workspace_synced': workspace_synced,
                'workspace_error': workspace_error,
            }
        )

    @staticmethod
    def _workspace_enabled() -> bool:
        return bool(config.evidently_url() and config.evidently_project_id())

    @staticmethod
    def _upload_to_workspace(report_result: Any) -> None:
        workspace = RemoteWorkspace(config.evidently_url())
        workspace.add_run(config.evidently_project_id(), report_result)

    @staticmethod
    def _report_as_dict(report: Any) -> dict[str, Any]:
        if callable(getattr(report, 'as_dict', None)):
            return report.as_dict()
        if callable(getattr(report, 'dict', None)):
            return report.dict()
        if callable(getattr(report, 'json', None)):
            return json.loads(report.json())
        raise AttributeError('Evidently report does not support JSON export')

    @staticmethod
    def _write_report_json(report: Any, json_path: Path, payload: dict[str, Any]) -> None:
        if callable(getattr(report, 'save_json', None)):
            report.save_json(str(json_path))
            return

        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    @staticmethod
    def _write_report_html(report: Any, html_path: Path, payload: dict[str, Any]) -> None:
        if callable(getattr(report, 'save_html', None)):
            report.save_html(str(html_path))
            return

        if callable(getattr(report, 'html', None)):
            html_content = report.html()
            html_path.write_text(html_content, encoding='utf-8')
            return

        # Fallback for Evidently versions without HTML export helpers.
        pretty_payload = escape(json.dumps(payload, ensure_ascii=False, indent=2))
        html_path.write_text(
            (
                '<html><body>'
                '<h1>Evidently Drift Report</h1>'
                '<p>Interactive HTML export is unavailable in this Evidently version. '
                'Showing JSON payload instead.</p>'
                f'<pre>{pretty_payload}</pre>'
                '</body></html>'
            ),
            encoding='utf-8',
        )


DRIFT_TRACKER = DriftTracker()
