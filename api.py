"""ConvertIQ FastAPI service. Run: uvicorn api:app --host 127.0.0.1 --port 8000."""

from contextlib import asynccontextmanager
import logging
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from src.config import ARTIFACTS
from src.models.inference import get_predictor, Predictor
from src.models.schemas import SessionInput, BatchRequest, PredictionResponse, ExplanationResponse
from src.data.schema import DataValidationError
from src.monitoring.drift import monitoring_history, record_rejection, rejection_summary

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("convertiq.api")


@asynccontextmanager
async def lifespan(app):
    try:
        get_predictor()
        LOG.info("ConvertIQ model ready")
    except FileNotFoundError:
        LOG.warning("No model found; prediction routes return 503 until training completes.")
    yield


app = FastAPI(
    title="ConvertIQ",
    version="1.0.0",
    lifespan=lifespan,
    description="Session-level conversion propensity. No early-session or causal guarantees.",
)


def predictor() -> Predictor:
    try:
        return get_predictor()
    except (FileNotFoundError, OSError, ValueError) as exc:
        LOG.exception("Model loading failed")
        raise HTTPException(503, "Model unavailable. Train or restore a valid ConvertIQ artifact.") from exc


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    try:
        directory = get_predictor().run_dir.parents[1]
    except (FileNotFoundError, OSError, ValueError):
        directory = ARTIFACTS
    record_rejection(request.url.path, exc.errors(), directory)
    errors = [{"type": e["type"], "loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


@app.exception_handler(DataValidationError)
async def validation_error(request: Request, exc: DataValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    LOG.exception("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal error. Check the server logs."})


@app.get("/health")
def health():
    try:
        model = get_predictor()
        return {"status": "ok", "model_ready": True, "model_version": model.metadata["version"]}
    except (FileNotFoundError, OSError, ValueError):
        return JSONResponse(status_code=503, content={"status": "not_ready", "model_ready": False})


@app.get("/model/info")
def model_info(model: Predictor = Depends(predictor)):
    return {k: v for k, v in model.metadata.items() if k not in ["test_metrics", "validation_metrics"]}


@app.get("/model/metrics")
def model_metrics(model: Predictor = Depends(predictor)):
    return {
        "model_version": model.metadata["version"],
        "test": model.metadata["test_metrics"],
        "validation": model.metadata["validation_metrics"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(session: SessionInput, model: Predictor = Depends(predictor)):
    return model.predict(pd.DataFrame([session.model_dump()]), record=True)[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchRequest, model: Predictor = Depends(predictor)):
    return model.predict(pd.DataFrame([session.model_dump() for session in request.sessions]), record=True)


@app.post("/explain", response_model=ExplanationResponse)
def explain(session: SessionInput, model: Predictor = Depends(predictor)):
    return model.explain(pd.DataFrame([session.model_dump()]))


@app.get("/monitoring/status")
def monitoring_status():
    try:
        directory = get_predictor().run_dir.parents[1]
    except (FileNotFoundError, OSError, ValueError):
        directory = ARTIFACTS
    history = monitoring_history(directory, limit=20)
    return {
        "rejected_requests": rejection_summary(directory),
        "latest": history[0] if history else None,
        "recent_batches": history,
        "note": "Single-session drift is not assessed; minimum batch size is configurable.",
    }
