"""Train and save all configured models outside the Streamlit app."""

from __future__ import annotations

from src.data_processing import DATASET_CONFIG, ensure_datasets
from src.model_training import model_artifact_path, save_model, train_model
from src.ui import MODEL_NAMES


def main() -> None:
    datasets = ensure_datasets()
    for dataset_name, config in DATASET_CONFIG.items():
        for model_name in MODEL_NAMES:
            result = train_model(
                datasets[dataset_name],
                config["features"],
                config["target"],
                model_name,
            )
            artifact = model_artifact_path(dataset_name, model_name)
            save_model(result, artifact)
            print(
                f"Saved {model_name} for {dataset_name}: {artifact} "
                f"(accuracy={result['metrics']['accuracy']:.3f})"
            )


if __name__ == "__main__":
    main()
