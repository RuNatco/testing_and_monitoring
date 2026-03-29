import json

import pandas as pd

from ml_service.drift import DriftTracker


class FakeReport:
    def __init__(self, metrics):
        self.metrics = metrics
        self.runs = []

    def run(self, current_data, reference_data):
        self.runs.append((current_data.copy(), reference_data.copy()))
        return self

    def save_html(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            file.write('<html><body>report</body></html>')

    def save_json(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.as_dict(), file)

    def as_dict(self):
        return {
            'metrics': [
                {
                    'metric_name': 'DriftedColumnsCount(drift_share=0.5)',
                    'config': {
                        'type': 'evidently:metric_v2:DriftedColumnsCount',
                        'drift_share': 0.5,
                    },
                    'value': {
                        'count': 1.0,
                        'share': 0.5,
                    },
                }
            ],
            'tests': [],
        }


class FakeReportWithoutSaveHtml(FakeReport):
    save_html = None


class BrokenReport(FakeReport):
    def save_json(self, path: str) -> None:
        raise RuntimeError('cannot save report')


class FakeWorkspace:
    runs = []

    def __init__(self, url: str):
        self.url = url

    def add_run(self, project_id: str, result):
        self.__class__.runs.append((self.url, project_id, result))


class BrokenWorkspace(FakeWorkspace):
    def add_run(self, project_id: str, result):
        raise RuntimeError('workspace is unavailable')


def test_drift_tracker_builds_report_when_enough_data(monkeypatch, tmp_path):
    monkeypatch.setattr('ml_service.drift.Report', FakeReport)
    monkeypatch.setattr('ml_service.drift.DataDriftPreset', lambda: 'preset')
    monkeypatch.setattr('ml_service.drift.config.evidently_url', lambda: None)
    monkeypatch.setattr('ml_service.drift.config.evidently_project_id', lambda: None)
    tracker = DriftTracker(reference_size=2, current_size=2, report_dir=str(tmp_path))
    tracker.reset(run_id='run-123', feature_names=['age', 'capital.gain'])

    tracker.record(pd.DataFrame([[30, 0]], columns=['age', 'capital.gain']), run_id='run-123')
    tracker.record(pd.DataFrame([[31, 10]], columns=['age', 'capital.gain']), run_id='run-123')
    tracker.record(pd.DataFrame([[60, 1000]], columns=['age', 'capital.gain']), run_id='run-123')
    summary = tracker.record(pd.DataFrame([[62, 1100]], columns=['age', 'capital.gain']), run_id='run-123')

    assert summary['ready'] is True
    assert summary['dataset_drift'] is True
    assert summary['drift_share'] == 0.5
    assert summary['drifted_columns'] == 1
    assert summary['total_columns'] == 2
    assert tmp_path.joinpath('latest_drift_report.html').exists()
    assert tmp_path.joinpath('latest_drift_report.json').exists()


def test_drift_tracker_falls_back_when_save_html_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr('ml_service.drift.Report', FakeReportWithoutSaveHtml)
    monkeypatch.setattr('ml_service.drift.DataDriftPreset', lambda: 'preset')
    monkeypatch.setattr('ml_service.drift.config.evidently_url', lambda: None)
    monkeypatch.setattr('ml_service.drift.config.evidently_project_id', lambda: None)
    tracker = DriftTracker(reference_size=1, current_size=1, report_dir=str(tmp_path))
    tracker.reset(run_id='run-123', feature_names=['age'])

    tracker.record(pd.DataFrame([[30]], columns=['age']), run_id='run-123')
    summary = tracker.record(pd.DataFrame([[60]], columns=['age']), run_id='run-123')

    assert summary['report_path'] is not None
    assert summary['report_json_path'] is not None
    assert 'Interactive HTML export is unavailable' in tmp_path.joinpath('latest_drift_report.html').read_text(
        encoding='utf-8'
    )


def test_drift_tracker_keeps_status_when_report_generation_fails(monkeypatch, tmp_path):
    monkeypatch.setattr('ml_service.drift.Report', BrokenReport)
    monkeypatch.setattr('ml_service.drift.DataDriftPreset', lambda: 'preset')
    monkeypatch.setattr('ml_service.drift.config.evidently_url', lambda: None)
    monkeypatch.setattr('ml_service.drift.config.evidently_project_id', lambda: None)
    tracker = DriftTracker(reference_size=1, current_size=1, report_dir=str(tmp_path))
    tracker.reset(run_id='run-123', feature_names=['age'])

    tracker.record(pd.DataFrame([[30]], columns=['age']), run_id='run-123')
    summary = tracker.record(pd.DataFrame([[60]], columns=['age']), run_id='run-123')

    assert summary['windows_ready'] is True
    assert summary['ready'] is False
    assert summary['report_path'] is None
    assert summary['report_error'] == 'cannot save report'


def test_drift_tracker_uploads_report_to_remote_workspace(monkeypatch, tmp_path):
    FakeWorkspace.runs = []
    monkeypatch.setattr('ml_service.drift.Report', FakeReport)
    monkeypatch.setattr('ml_service.drift.DataDriftPreset', lambda: 'preset')
    monkeypatch.setattr('ml_service.drift.RemoteWorkspace', FakeWorkspace)
    monkeypatch.setattr('ml_service.drift.config.evidently_url', lambda: 'http://158.160.2.37:8000/')
    monkeypatch.setattr('ml_service.drift.config.evidently_project_id', lambda: '019d061f-cc08-7b5e-b932-d792a1f258e2')
    tracker = DriftTracker(reference_size=1, current_size=1, report_dir=str(tmp_path))
    tracker.reset(run_id='run-123', feature_names=['age'])

    tracker.record(pd.DataFrame([[30]], columns=['age']), run_id='run-123')
    summary = tracker.record(pd.DataFrame([[60]], columns=['age']), run_id='run-123')

    assert summary['workspace_synced'] is True
    assert summary['workspace_error'] is None
    assert FakeWorkspace.runs
    assert FakeWorkspace.runs[0][0] == 'http://158.160.2.37:8000/'
    assert FakeWorkspace.runs[0][1] == '019d061f-cc08-7b5e-b932-d792a1f258e2'


def test_drift_tracker_keeps_local_report_when_workspace_sync_fails(monkeypatch, tmp_path):
    monkeypatch.setattr('ml_service.drift.Report', FakeReport)
    monkeypatch.setattr('ml_service.drift.DataDriftPreset', lambda: 'preset')
    monkeypatch.setattr('ml_service.drift.RemoteWorkspace', BrokenWorkspace)
    monkeypatch.setattr('ml_service.drift.config.evidently_url', lambda: 'http://158.160.2.37:8000/')
    monkeypatch.setattr('ml_service.drift.config.evidently_project_id', lambda: '019d061f-cc08-7b5e-b932-d792a1f258e2')
    tracker = DriftTracker(reference_size=1, current_size=1, report_dir=str(tmp_path))
    tracker.reset(run_id='run-123', feature_names=['age'])

    tracker.record(pd.DataFrame([[30]], columns=['age']), run_id='run-123')
    summary = tracker.record(pd.DataFrame([[60]], columns=['age']), run_id='run-123')

    assert summary['ready'] is True
    assert summary['workspace_synced'] is False
    assert summary['workspace_error'] == 'workspace is unavailable'
    assert summary['report_path'] is not None
