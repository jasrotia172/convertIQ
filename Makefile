PYTHON ?= .venv/bin/python
export MPLCONFIGDIR := /tmp/convertiq-matplotlib
.PHONY: setup download eda train dashboard api test lint mlflow notebook
setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt
download:
	$(PYTHON) -m src.data.download
eda:
	$(PYTHON) -m src.data.analytics
train:
	$(PYTHON) -m src.models.train
dashboard:
	$(PYTHON) -m streamlit run app.py --server.address 127.0.0.1
api:
	$(PYTHON) -m uvicorn api:app --host 127.0.0.1 --port 8000
test:
	$(PYTHON) -m pytest -q
lint:
	.venv/bin/ruff check .
mlflow:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
notebook:
	$(PYTHON) scripts/execute_notebook.py
