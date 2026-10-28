.PHONY: test lint format clean docker-up docker-down eval install

test:
	PYTHONPATH=src:. pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

eval:
	PYTHONPATH=src:. python scripts/run_e2e_eval.py

install:
	pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info src/*.egg-info

docker-up:
	docker compose up -d

docker-down:
	docker compose down
