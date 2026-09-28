# Sponsored Ads CTR Prediction System

A production-grade, end-to-end machine learning system for predicting click-through rates on sponsored advertisements. Demonstrates scalable ML pipelines, real-time low-latency model serving, feature engineering, A/B testing, model monitoring, and operational excellence.

## Architecture

+------------------+ | Ad Request | | (user, ad, query)| +--------+---------+ | +--------v---------+ | Feature Store | | (Redis/Memory) | +--------+---------+ | +------------------+------------------+ | | +--------v---------+ +----------v----------+ | Feature Pipeline | | A/B Experiment | | - User features | | Manager | | - Ad features | | - Hash assignment | | - Cross features | | - Traffic splitting | +--------+---------+ +----------+----------+ | | +------------------+------------------+ | +--------v---------+ | CTR Predictor | | (LightGBM / | | DeepFM) | +--------+---------+ | +--------v---------+ | Monitoring | | - Latency p99 | | - Drift (PSI) | | - Alerts | | - Prometheus | +------------------+

OFFLINE: +------------------+ +------------------+ +------------------+ | Data Pipeline |---->| Training |---->| Model Registry | | - Ingestion | | - LightGBM | | - Versioning | | - Preprocessing | | - DeepFM | | - Hot reload | | - Validation | | - Hypertuning | | - Promotion | +------------------+ +------------------+ +------------------+


## Key Features

- **Scalable ML Pipeline** -- Training with LightGBM and DeepFM, experiment tracking, hyperparameter tuning, temporal train/test splits
- **Real-Time Serving** -- FastAPI prediction API with batch support, model hot-reload, <5ms p99 latency target
- **Feature Store** -- Online (Redis) and offline (Parquet) stores with TTL and versioning
- **A/B Testing** -- Deterministic hash-based traffic splitting, chi-squared significance testing, sample size calculation
- **Monitoring** -- Latency percentiles (p50/p95/p99), PSI-based drift detection, concept drift detection, alerting
- **Operational Excellence** -- Health checks, Prometheus metrics, structured JSON logging, Docker deployment, CI/CD

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

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Generate synthetic data (100K impressions)
python scripts/generate_data.py --num_samples 100000 --output_dir data/

# Train model
python scripts/train_model.py --data_dir data/ --output_dir artifacts/

# Start prediction server
python scripts/run_server.py --model_path artifacts/model.joblib --port 8000

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u_000001","ad_id":"a_000010","query":"wireless headphones","position":1,"device_type":"mobile"}'
