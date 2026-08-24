.PHONY: install prepare train evaluate test lint check api docker

install:
	pip install -e ".[dev]"

prepare:
	python scripts/prepare_dataset.py --data-root data/raw/Otoscopic_Data

train:
	python scripts/train.py --config configs/train.yaml

evaluate:
	python scripts/evaluate.py --config configs/train.yaml --checkpoint artifacts/models/best.pt

test:
	pytest -q

lint:
	ruff check .

check: lint test

api:
	MODEL_PATH=artifacts/models/best.pt uvicorn otovision.api:app --reload --port 8000

docker:
	docker build -t otovision-mlops:latest .
