# Отчёт

## 1. Повышение надёжности сервиса

В сервис добавлена обработка следующих ситуаций:

- модель ещё не загружена: `GET /health` возвращает `503`, `POST /predict` возвращает `503`
- в `POST /updateModel` передан пустой `run_id`: возвращается `422`
- в `POST /updateModel` передан невалидный `run_id`: ошибка загрузки модели преобразуется в `400`
- в `POST /predict` пришёл невалидный JSON или поля неверного типа: ошибка валидируется `Pydantic` и возвращается `422`
- в `POST /predict` не переданы обязательные для текущей модели фичи: возвращается `422` с перечислением `missing_features`
- модель загружена некорректно и не содержит `feature_names_in_`: возвращается `500` ошибка
- ошибка инференса модели возвращается как `500 Prediction failed`
- ошибка генерации drift report через `evidently` не ломает основной inference-path, сервис продолжает отвечать на `/predict`, а причина сохраняется в `report_error` внутри `/drift/status`
- при смене модели метаданные активной модели, списка её фич и drift-окна сбрасываются и пересоздаются

## 2. Добавленные тесты

### Unit-тесты

- `tests/test_features.py`
  - проверка `to_dataframe()` с полным набором фич
  - проверка выбора только нужных фич
  - проверка ошибки для неизвестной колонки

- `tests/test_model.py`
  - успешная загрузка модели и обновление `run_id`
  - оборачивание ошибок загрузки в `ModelLoadError`
  - ошибка при доступе к фичам, если модель не загружена

- `tests/test_drift.py`
  - построение drift report при достаточном числе событий
  - fallback для версий `evidently` без `save_html()`
  - корректное состояние tracker-а при ошибке генерации отчёта
- успешная отправка drift report в `RemoteWorkspace`
- мягкая деградация при недоступном `RemoteWorkspace`

### API-тесты

- `tests/test_app.py`
  - `GET /health` для загруженной и незагруженной модели
  - `GET /metrics`
  - `POST /predict` для успешного инференса
  - `POST /predict` для незагруженной модели
  - `POST /predict` для missing required features
  - `POST /updateModel` для успешного обновления модели
  - `POST /updateModel` для невалидного `run_id`
  - `POST /updateModel` для пустого `run_id`
  - `GET /drift/status`
  - `GET /drift/report` в режиме warming up

### Интеграционный smoke-тест

- `tests/test_app.py::test_service_startup_loads_model_and_serves_prediction`
  - проверяет запуск приложения с lifespan/startup
  - проверяет, что сервис после старта отдаёт предсказание

## 3. Логируемые метрики

В качестве основного хранилища метрик используется `Prometheus`. Сервис отправляет дублирующий поток метрик в `Graphite`, но dashboards и alerting для сдачи строятся вокруг `Prometheus default`.

### Технические метрики сервиса

- `ml_service_http_requests_total` — количество запросов
- `ml_service_http_request_duration_seconds` — latency HTTP-запросов
- `process_resident_memory_bytes`, `process_virtual_memory_bytes`, `process_cpu_seconds_total` — потребление ресурсов процесса

### Метрики входных данных

- `ml_service_preprocessing_duration_seconds` — время предобработки
- `ml_service_feature_observations_total` — частота наблюдения фич
- `ml_service_feature_missing_total` — частота отсутствующих обязательных фич
- `ml_service_feature_numeric_value` — распределение числовых фич
- `ml_service_feature_categorical_total` — частоты категориальных значений

### Метрики модели

- `ml_service_inference_duration_seconds` — время инференса
- `ml_service_predictions_total` — распределение классов
- `ml_service_prediction_probability` — распределение вероятностей
- `ml_service_model_loads_total` — успешные и неуспешные загрузки модели
- `ml_service_model_loaded` — флаг наличия загруженной модели
- `ml_service_model_info` — активный `run_id`
- `ml_service_model_type_info` — тип модели в проде
- `ml_service_model_required_feature_info` — фичи, которые нужны текущей модели

### Метрики data drift

- `ml_service_data_drift_ready`
- `ml_service_data_drift_detected`
- `ml_service_data_drift_share`
- `ml_service_data_drift_reference_rows`
- `ml_service_data_drift_current_rows`
- `ml_service_drift_status_updates_total`

Дополнительно через `GET /drift/status` публикуется служебная информация о построении и удалённой публикации отчёта:

- `workspace_url`
- `workspace_project_id`
- `workspace_synced`
- `workspace_error`

### Перцентили

Для временных метрик в Grafana построены запросы для перцентилей:

- `p75`
- `p90`
- `p95`
- `p99`
- `p99.9`

Они используются для:

- latency сервиса
- времени предобработки
- времени инференса

## 4. Алертинг

Алертинг настроен во внешней Grafana через `Prometheus default` и `Telegram contact point`.

Создан Telegram-бот для отправки уведомлений в отдельный групповой чат с алертами. 

Во внешней Grafana созданы alert rules на следующие события:

- недоступность сервиса
- высокий процент `5xx`
- высокая latency инференса / запроса предсказания
- обнаруженный data drift
- высокое потребление памяти
- запросы с пропущенными обязательными фичами
- сильный перекос распределения классов предсказаний
- аномально высокая концентрация вероятностей около `1`


Эти правила покрывают как технические проблемы сервиса, так и проблемы, связанные с поведением модели и качеством входящих данных.

## 5. Дашборды Grafana

Для локальной разработки dashboards поднимаются автоматически через provisioning в контейнере Grafana. 

Импортированы следующие dashboards в Grafana:

- `grafana/dashboards/ml_service_overview.json`
- `grafana/dashboards/ml_service_operations.json`
- `grafana/dashboards/ml_service_drift.json`

На dashboards визуализируются следующие блоки метрик:

- технические метрики сервиса: RPS, error ratio, request latency, resource usage
- метрики данных: preprocessing latency, feature observation rate, missing features, drift window sizes
- метрики модели: inference latency, distribution of predictions, distribution of probabilities
- метрики информации о модели: текущий `run_id`, тип модели, необходимые фичи, события обновления модели


## 6. Evidently

В сервисе реализован мониторинг drift-а:

- входных фич
- предсказаний модели
- вероятностей модели

Подход:

- сервис накапливает события в памяти
- первые `DRIFT_REFERENCE_SIZE` событий образуют reference window
- следующие `DRIFT_CURRENT_SIZE` событий образуют current window
- после заполнения окон строится отчёт `evidently`
- отчёт сохраняется в `artifacts/evidently/latest_drift_report.html` и `artifacts/evidently/latest_drift_report.json`
- в drift dataset включаются не только входные фичи, но и `prediction` и `probability`
- при наличии `EVIDENTLY_URL` и `EVIDENTLY_PROJECT_ID` тот же отчёт дополнительно публикуется в `RemoteWorkspace`
- если удалённая публикация не удалась, локальные HTML/JSON-артефакты всё равно сохраняются, а причина ошибки отражается в `workspace_error`
- состояние доступно через:
  - `GET /drift/status`
  - `GET /drift/report`
  - `GET /drift/report/html`

