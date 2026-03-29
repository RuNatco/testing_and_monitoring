from typing import Any
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse
import pandas as pd

from ml_service import config
from ml_service.drift import DRIFT_TRACKER
from ml_service.features import get_missing_features, to_dataframe
from ml_service.metrics import METRICS
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model, ModelLoadError
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


MODEL = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    configure_mlflow()
    run_id = config.default_run_id()
    try:
        MODEL.set(run_id=run_id)
    except ModelLoadError:
        METRICS.record_model_load(run_id=run_id, success=False)
        METRICS.set_model_state(run_id=None, loaded=False)
        raise

    METRICS.record_model_load(run_id=run_id, success=True)
    loaded_model = MODEL.get().model
    feature_names = _get_model_features(loaded_model)
    METRICS.set_model_metadata(
        run_id=run_id,
        model_type=_get_model_type(loaded_model),
        features=feature_names,
    )
    METRICS.record_drift_status(
        DRIFT_TRACKER.reset(run_id=run_id, feature_names=_get_drift_columns(feature_names))
    )
    yield
    # add any teardown logic here if needed


def _get_model_features(model: Any) -> list[str]:
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Loaded model does not expose feature names',
        )
    return list(feature_names)


def _get_model_type(model: Any) -> str:
    steps = getattr(model, 'steps', None)
    if steps:
        return steps[-1][1].__class__.__name__
    return model.__class__.__name__


def _get_drift_columns(feature_names: list[str]) -> list[str]:
    return list(feature_names) + ['prediction', 'probability']


def _create_app(load_on_startup: bool = True) -> FastAPI:
    app = FastAPI(
        title='MLflow FastAPI service',
        version='1.0.0',
        lifespan=lifespan if load_on_startup else None,
    )

    @app.middleware('http')
    async def instrument_requests(request, call_next):
        started_at = perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            if request.url.path != '/metrics':
                METRICS.record_http_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    duration_seconds=perf_counter() - started_at,
                )

    @app.get('/health')
    def health(response: Response) -> dict[str, Any]:
        model_state = MODEL.get()
        if model_state.model is None:
            METRICS.set_model_state(run_id=None, loaded=False)
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {'status': 'unavailable', 'run_id': None}

        METRICS.set_model_state(run_id=model_state.run_id, loaded=True)
        return {'status': 'ok', 'run_id': model_state.run_id}

    @app.get('/metrics')
    def metrics() -> PlainTextResponse:
        payload, content_type = METRICS.render_prometheus_metrics()
        return PlainTextResponse(content=payload, media_type=content_type)

    @app.get('/drift/status')
    def drift_status() -> dict[str, Any]:
        return DRIFT_TRACKER.status()

    @app.get('/drift/report')
    def drift_report() -> dict[str, Any]:
        try:
            return DRIFT_TRACKER.latest_report()
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.get('/drift/report/html')
    def drift_report_html() -> HTMLResponse:
        try:
            return HTMLResponse(content=DRIFT_TRACKER.latest_report_html())
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model_state = MODEL.get()
        model = model_state.model
        if model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        feature_names = _get_model_features(model)
        preprocessing_started_at = perf_counter()
        df = to_dataframe(request, needed_columns=feature_names)
        METRICS.record_preprocessing_duration(
            run_id=model_state.run_id,
            duration_seconds=perf_counter() - preprocessing_started_at,
        )
        METRICS.record_feature_values(run_id=model_state.run_id, row=df.iloc[0].to_dict())

        missing_features = get_missing_features(df)
        if missing_features:
            METRICS.record_missing_features(run_id=model_state.run_id, missing_features=missing_features)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    'message': 'Required features are missing',
                    'missing_features': missing_features,
                },
            )

        inference_started_at = perf_counter()
        try:
            probability = float(model.predict_proba(df)[0][1])
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Prediction failed',
            ) from exc
        METRICS.record_inference_duration(
            run_id=model_state.run_id,
            duration_seconds=perf_counter() - inference_started_at,
        )
        prediction = int(probability >= 0.5)
        METRICS.record_prediction(
            prediction=prediction,
            probability=probability,
            run_id=model_state.run_id,
        )
        drift_frame = pd.DataFrame(df.copy())
        drift_frame['prediction'] = prediction
        drift_frame['probability'] = probability
        METRICS.record_drift_status(DRIFT_TRACKER.record(drift_frame, run_id=model_state.run_id))

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        try:
            MODEL.set(run_id=run_id)
        except ModelLoadError as exc:
            METRICS.record_model_load(run_id=run_id, success=False)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        METRICS.record_model_load(run_id=run_id, success=True)
        loaded_model = MODEL.get().model
        feature_names = _get_model_features(loaded_model)
        METRICS.set_model_metadata(
            run_id=run_id,
            model_type=_get_model_type(loaded_model),
            features=feature_names,
        )
        METRICS.record_drift_status(
            DRIFT_TRACKER.reset(run_id=run_id, feature_names=_get_drift_columns(feature_names))
        )
        return UpdateModelResponse(run_id=run_id)

    return app


def create_app() -> FastAPI:
    return _create_app(load_on_startup=True)


app = create_app()
