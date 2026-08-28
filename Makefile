.PHONY: all fetch analysis test clean

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

all: analysis test

fetch:
	$(PY) src/fetch_data.py

analysis:
	$(PY) src/run_analysis.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf analysis/charts/*.png data/processed/results.json
	find . -name __pycache__ -type d -exec rm -rf {} +
