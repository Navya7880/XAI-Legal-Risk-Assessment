.PHONY: evaluate test

evaluate:
	python scripts/run_evaluation.py

test:
	python -m pytest -q
