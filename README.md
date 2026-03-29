# MLflow + FastAPI service

Сервис на FastAPI, который:
- при старте приложения загружает ML‑модель из MLflow
- имеет хэндлер `POST /predict`, который принимает на вход признаки
- имеет хэндлер `POST /updateModel`, который принимает `run_id` и подменяет текущую модель на модель из этого run

## Переменные окружения

- **`MLFLOW_TRACKING_URI`**: URI MLflow Tracking Server (например, `http://158.160.2.37:5000/`)
- **`DEFAULT_RUN_ID`**: модель загрузится из запуска с этим ID
- **`EVIDENTLY_URL`**: URL удаленного Evidently Workspace для публикации drift-report
- **`EVIDENTLY_PROJECT_ID`**: идентификатор проекта в Evidently Workspace

## Запуск

Через docker compose:

```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=<your_run_id>
export EVIDENTLY_URL=http://158.160.2.37:8000/
export EVIDENTLY_PROJECT_ID=<your project id>
docker compose up --build
```

Сервис будет доступен на `http://<ip>:1488`.

## Мониторинг

После запуска вместе с сервисом поднимаются:

- `Prometheus` на `http://<ip>:9090`
- `Grafana` на `http://<ip>:3000`
- `Graphite` на `http://<ip>:8081`
- `Alertmanager` на `http://<ip>:9093`


Для Grafana используются учетные данные, выданные отдельно. Для локальной Grafana задаются через переменные окружения `GF_SECURITY_ADMIN_USER` и `GF_SECURITY_ADMIN_PASSWORD` в `docker-compose.yaml`.

### Что мониторится

Сервис экспортирует:

- HTTP-метрики по endpoint-ам: количество запросов, ошибки, latency
- метрики предобработки и инференса: latency, перцентили, распределение классов и вероятностей
- метрики входных данных: пропуски обязательных фич, наблюдения по фичам, статистика по числовым и категориальным значениям
- метрики жизненного цикла модели: активный `run_id`, тип модели, необходимые фичи, успешные и неуспешные загрузки модели
- метрики drift: готовность отчета, доля drifted columns, размеры окон, статус синхронизации с `RemoteWorkspace`

Prometheus-метрики доступны через endpoint:

```bash
curl http://<ip>:1488/metrics
```

### Дашборды

Локальная Grafana автоматически подхватывает следующие дашборды:

- `ML Service Overview` для метрик из Prometheus
- `ML Service Operations` для percentiles, resource usage, feature statistics и model metadata
- `ML Service Graphite` для метрик из Graphite
- `ML Service Drift` для метрик data drift из evidently

## Алертинг

Prometheus загружает правила из `prometheus/alerts.yml` и отправляет события в `Alertmanager`.

Настроены алерты на:

- недоступность сервиса
- высокий уровень `5xx`
- высокую latency у `/predict`
- обнаруженный data drift
- высокое потребление памяти
- пропущенные обязательные фичи
- перекос распределения классов предсказаний

Проверить активные алерты можно в:

- `Prometheus Alerts`: `http://<ip>:9090/alerts`
- `Alertmanager`: `http://<ip>:9093`

## Evidently

Сервис собирает live-запросы и строит drift report:

- первые `DRIFT_REFERENCE_SIZE` запросов после загрузки модели используются как reference window
- следующие `DRIFT_CURRENT_SIZE` запросов попадают в current window
- когда данных достаточно, `evidently` вычисляет data drift по фичам, предсказаниям и вероятностям модели
- локально сохраняются HTML/JSON-отчеты в `artifacts/evidently`
- если заданы `EVIDENTLY_URL` и `EVIDENTLY_PROJECT_ID`, готовый отчёт дополнительно отправляется в `RemoteWorkspace`

Полезные endpoints:

```bash
curl http://<ip>:1488/drift/status
curl http://<ip>:1488/drift/report
curl http://<ip>:1488/drift/report/html
```

Пример проверки remote sync:

```bash
curl http://<ip>:1488/drift/status
```

В ответе будут поля:

- `workspace_url`
- `workspace_project_id`
- `workspace_synced`
- `workspace_error`

