.PHONY: tb train test

train:
	python train.py

tb:
	./scripts/tensorboard.sh

test:
	pytest -q
