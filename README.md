# Sponsored Ads CTR Prediction System

A production-grade, end-to-end machine learning system for predicting click-through rates on sponsored advertisements. Demonstrates scalable ML pipelines, real-time low-latency model serving, feature engineering, A/B testing, model monitoring, and operational excellence.

---

## Architecture

### Online Serving Path

```
                                +-----------------+
                                |   Ad Request    |
                                | (user, ad, query|
                                |   position,     |
                                |   device)       |
                                +--------+--------+
                                         |
                         +---------------v----------------+
                         |        Feature Store           |
                         |       (Redis / Memory)         |
                         +---+-------------------+--------+
                             |                   |
                    +--------v--------+  +-------v---------+
                    | Feature Pipeline|  | A/B Experiment   |
                    |                 |  |    Manager       |
                    | - User features |  |                  |
                    |   (hist. CTR,   |  | - Hash-based     |
                    |    impressions) |  |   assignment     |
                    | - Ad features   |  | - Traffic split  |
                    |   (quality,     |  | - Statistical    |
                    |    relevance)   |  |   significance   |
                    | - Cross features|  |                  |
                    |   (user-ad      |  |                  |
                    |    affinity)    |  |                  |
                    +--------+--------+  +-------+----------+
                             |                   |
                             +-------+-----------+
                                     |
                            +--------v---------+
                            |  CTR Predictor   |
                            |  (LightGBM /     |
                            |   DeepFM)        |
                            +--------+---------+
                                     |
                            +--------v---------+
                            |   Monitoring     |
                            |                  |
                            | - Latency p99    |
                            | - Drift (PSI)    |
                            | - Alerts         |
                            | - Prometheus     |
                            +------------------+
```

### Offline Training Path

```
  +-------------------+      +-------------------+      +-------------------+
  |   Data Pipeline   | ---> |     Training      | ---> |  Model Registry   |
  |                   |      |                   |      |                   |
  | - Ingestion       |      | - LightGBM        |      | - Versioning      |
  | - Preprocessing   |      | - DeepFM          |      | - Hot reload      |
  | - Validation      |      | - Hypertuning     |      | - Promotion       |
  | - Temporal split  |      | - Early stopping  |      | - A/B deployment  |
  +-------------------+      +-------------------+      +-------------------+
```

### Request Flow (Sequence)

```
  Client            API Gateway        Feature Store       Model            Monitoring
    |                   |                   |                 |                  |
    |--- POST /predict->|                   |                 |                  |
    |                   |-- get features -->|                 |                  |
    |                   |<-- features ------|                 |                  |
    |                   |-- predict ------->|---------------->|                  |
    |                   |<-- ctr_proba -----|-----------------|                  |
    |                   |-- record -------->|---------------->|----------------->|
    |<-- response ------|                   |                 |                  |
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Scalable ML Pipeline** | Training with LightGBM and DeepFM, experiment tracking, hyperparameter tuning, temporal splits |
| **Real-Time Serving** | FastAPI prediction API with batch support, model hot-reload, <5ms p99 latency target |
| **Feature Store** | Online (Redis) and offline (Parquet) stores with TTL and versioning |
| **A/B Testing** | Deterministic hash-based traffic splitting, chi-squared significance testing, sample size calculation |
| **Monitoring** | Latency percentiles (p50/p95/p99), PSI-based drift detection, concept drift, alerting |
| **Operational Excellence** | Health checks, Prometheus metrics, structured JSON logging, Docker deployment, CI/CD |

---

## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| ML Models          | LightGBM, PyTorch (DeepFM)         |
| Serving            | FastAPI, Uvicorn                    |
| Feature Store      | Redis (online), Parquet (offline)   |
| Monitoring         | Prometheus metrics, PSI drift       |
| Data Processing    | Pandas, NumPy                       |
| Experiment Track   | Custom JSON-based tracker           |
| Containerization   | Docker, Docker Compose              |
| CI/CD              | GitHub Actions                      |
| Testing            | pytest                              |

---

## Project Structure

```
sponsored-ads-ctr-prediction/
|
+-- README.md
+-- requirements.txt
+-- setup.py
+-- Dockerfile
+-- docker-compose.yml
+-- .gitignore
|
+-- .github/workflows/
|   +-- ci.yml                    # GitHub Actions CI pipeline
|
+-- configs/
|   +-- default.yaml              # Default configuration
|
+-- scripts/
|   +-- generate_data.py          # Generate 100K synthetic impressions
|   +-- train_model.py            # End-to-end training pipeline
|   +-- run_server.py             # Launch prediction server
|
+-- src/
|   +-- utils/
|   |   +-- logger.py             # Structured JSON logging
|   |   +-- config.py             # Pydantic-based config management
|   |
|   +-- data_pipeline/
|   |   +-- ingestion.py          # Click-stream generation, batch loading
|   |   +-- preprocessing.py      # Cleaning, validation, text processing
|   |
|   +-- feature_store/
|   |   +-- feature_engineering.py # User, ad, query, cross features
|   |   +-- feature_store.py      # Online (Redis) + Offline (Parquet) store
|   |
|   +-- training/
|   |   +-- dataset.py            # Temporal splits, feature prep, validation
|   |   +-- models.py             # LightGBM, DeepFM, Logistic baseline
|   |   +-- trainer.py            # Training loop, experiment tracking, tuning
|   |   +-- evaluation.py         # AUC, calibration, lift charts, bias analysis
|   |
|   +-- serving/
|   |   +-- predictor.py          # Core prediction engine
|   |   +-- api.py                # FastAPI endpoints
|   |   +-- model_loader.py       # Model registry, hot-reload
|   |
|   +-- monitoring/
|   |   +-- metrics_collector.py  # Latency p50/p95/p99, request counts
|   |   +-- drift_detector.py     # PSI drift, concept drift, alerts
|   |
|   +-- ab_testing/
|       +-- experiment.py         # A/B experiments, statistical analysis
|
+-- tests/
    +-- test_feature_engineering.py
    +-- test_models.py
    +-- test_serving.py
    +-- test_ab_testing.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic training data

```bash
python scripts/generate_data.py --num_samples 100000 --output_dir data/
```

This generates realistic ad impressions with:
- Position bias (position 1 ~ 5% CTR, position 5 ~ 1% CTR)
- User-ad category affinity
- Time-of-day and device effects
- User segment value tiers

### 3. Train models

```bash
python scripts/train_model.py --data_dir data/ --output_dir artifacts/
```

Trains LightGBM (primary) and Logistic Regression (baseline), evaluates on test set, saves model artifacts and experiment logs.

### 4. Start the prediction server

```bash
python scripts/run_server.py --model_path artifacts/model.joblib --port 8000
```

### 5. Make predictions

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_000001",
    "ad_id": "a_000010",
    "query": "wireless headphones",
    "position": 1,
    "device_type": "mobile"
  }'
```

**Batch prediction:**
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[
    {"user_id":"u_000001","ad_id":"a_000010","query":"wireless headphones","position":1,"device_type":"mobile"},
    {"user_id":"u_000002","ad_id":"a_000020","query":"running shoes","position":2,"device_type":"desktop"}
  ]'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Metrics:**
```bash
curl http://localhost:8000/metrics
```

---

## Docker

```bash
# Build and run with Redis
docker-compose up --build

# Or standalone
docker build -t ctr-prediction .
docker run -p 8000:8000 ctr-prediction
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single CTR prediction |
| `POST` | `/batch_predict` | Batch predictions (max 100) |
| `GET` | `/health` | Health check with model status and uptime |
| `GET` | `/metrics` | Latency percentiles, request counts, error rates |
| `GET` | `/model/info` | Current model version and load time |
| `POST` | `/model/reload` | Hot-reload model without downtime |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| LightGBM as primary model | Best speed/accuracy tradeoff for tabular CTR data at scale |
| Temporal train/test split | Prevents data leakage from future information |
| Hash-based A/B assignment | Deterministic, stateless, no DB lookup needed |
| PSI for drift detection | Industry standard for production ML monitoring |
| Feature store with TTL | Stale features are worse than missing features |
| Model hot-reload with locks | Zero-downtime deployments for model updates |
| Singleton MetricsCollector | Thread-safe, single source of truth across workers |

---

## License

MIT
