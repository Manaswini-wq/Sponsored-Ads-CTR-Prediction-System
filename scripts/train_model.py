import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline.ingestion import BatchDataLoader
from src.training.dataset import prepare_features, DataValidator
from src.training.models import GradientBoostedCTRModel, LogisticBaselineModel
from src.training.trainer import Trainer, ExperimentTracker
from src.training.evaluation import ModelEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CTR prediction model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    train_df = BatchDataLoader.load(os.path.join(args.data_dir, "train.csv"))
    val_df = BatchDataLoader.load(os.path.join(args.data_dir, "val.csv"))
    test_df = BatchDataLoader.load(os.path.join(args.data_dir, "test.csv"))

    print("Preparing features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    # Validate data
    balance = DataValidator.check_class_balance(y_train)
    print(f"Training set positive rate: {balance['positive_rate']:.4f}")

    # Train LightGBM (primary model)
    print("\n--- Training LightGBM ---")
    trainer = Trainer(checkpoint_dir=os.path.join(args.output_dir, "checkpoints"))
    lgbm = GradientBoostedCTRModel(learning_rate=0.05, num_trees=500, max_depth=7)
    lgbm_metrics = trainer.train(lgbm, X_train, y_train, X_val, y_val)

    # Train Logistic Regression (baseline)
    print("\n--- Training Logistic Baseline ---")
    baseline = LogisticBaselineModel()
    baseline.fit(X_train, y_train)

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_metrics = ModelEvaluator.generate_report(
        lgbm, X_test, y_test, os.path.join(args.output_dir, "evaluation")
    )
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save final model
    model_path = os.path.join(args.output_dir, "model.joblib")
    lgbm.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Log experiment
    tracker = ExperimentTracker()
    tracker.log_experiment(
        name="lgbm_ctr_v1",
        params={"learning_rate": 0.05, "num_trees": 500, "max_depth": 7},
        metrics=test_metrics,
        artifacts={"model_path": model_path},
    )
    print("Experiment logged.")


if __name__ == "__main__":
    main()
