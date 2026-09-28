import time
import uuid

from fastapi import FastAPI, HTTPException, Request

from src.serving.predictor import CTRPredictor, PredictionRequest, PredictionResponse
from src.serving.model_loader import ModelLoader
from src.monitoring.metrics_collector import MetricsCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Sponsored Ads CTR Prediction API", version="1.0.0")

model_loader = ModelLoader()
predictor: CTRPredictor | None = None
metrics = MetricsCollector()
start_time = time.time()


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


@app.get("/health")
async def health():
    info = model_loader.health_check()
    return {
        "status": "healthy" if info["model_loaded"] else "degraded",
        "uptime_seconds": round(time.time() - start_time, 1),
        "model": info,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    try:
        resp = predictor.predict(request)
        metrics.record_prediction_latency(resp.latency_ms)
        metrics.record_prediction_request(resp.model_version, "success")
        return resp
    except Exception as e:
        metrics.record_prediction_request("unknown", "error")
        raise HTTPException(500, str(e))


@app.post("/batch_predict", response_model=list[PredictionResponse])
async def batch_predict(requests: list[PredictionRequest]):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    if len(requests) > 100:
        raise HTTPException(400, "Max batch size is 100")
    return predictor.batch_predict(requests)


@app.get("/metrics")
async def get_metrics():
    return metrics.get_summary()


@app.get("/model/info")
async def model_info():
    return model_loader.health_check()


@app.post("/model/reload")
async def reload_model(model_path: str):
    global predictor
    try:
        model_loader.hot_reload(model_path)
        predictor = CTRPredictor(model=model_loader.get_model())
        return {"status": "reloaded", "path": model_path}
    except Exception as e:
        raise HTTPException(500, str(e))
