import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start CTR prediction server")
    parser.add_argument("--model_path", type=str, default="artifacts/model.joblib")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Load model and initialize predictor before starting server
    from src.serving.model_loader import ModelLoader
    from src.serving.predictor import CTRPredictor
    from src.feature_store.feature_store import OnlineFeatureStore
    import src.serving.api as api_module

    loader = ModelLoader()
    model = loader.load_model(args.model_path)
    feature_store = OnlineFeatureStore(fallback_to_memory=True)

    api_module.model_loader = loader
    api_module.predictor = CTRPredictor(
        model=model, feature_store=feature_store
    )

    print(f"Starting server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run(
        "src.serving.api:app",
        host=args.host, port=args.port, workers=args.workers,
    )


if __name__ == "__main__":
    main()
