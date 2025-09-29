.PHONY: tb train test lint clean

train:
	python train.py

tb:
	./scripts/tensorboard.sh

test:
	pytest -q

test-metrics:
	pytest tests/test_eval_metrics.py -v

test-telemetry:
	pytest tests/test_telemetry_*.py -v

lint:
	flake8 src/

clean:
	rm -rf runs/
	rm -rf artifacts/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete