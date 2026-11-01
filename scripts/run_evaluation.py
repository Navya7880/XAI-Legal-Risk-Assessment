import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.runner import run_evaluation
from src.evaluation.robustness import run_robustness

if __name__ == "__main__":
    run_evaluation()
    run_robustness()
