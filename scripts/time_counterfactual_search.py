import time
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import DATASET_CONFIG
from src.dice_explainer import generate_counterfactuals
from src.model_training import load_model, model_artifact_path


parser = argparse.ArgumentParser()
parser.add_argument("--method", default="random")
args = parser.parse_args()

dataset_name = "Legal risk (synthetic)"
model_name = "Random Forest"
record = {
    "age": 27,
    "prior_offenses": 4,
    "offense_category": "traffic",
    "employment_status": "employed",
    "education_level": "secondary",
    "program_completion": "no",
    "substance_treatment": "not_completed",
}

model_result = load_model(model_artifact_path(dataset_name, model_name))
config = DATASET_CONFIG[dataset_name]
started = time.perf_counter()
result = generate_counterfactuals(
    model_result,
    record,
    total_cfs=3,
    method=args.method,
    immutable_features=config["immutable_features"],
    permitted_range=config["permitted_range"],
    constraint_mode="full",
)
elapsed = time.perf_counter() - started

print(f"elapsed_seconds={elapsed:.3f}")
print(f"returned_rows={len(result)}")
print(result.to_string(index=False) if not result.empty else "NONE")
