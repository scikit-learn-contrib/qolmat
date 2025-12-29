# Install dependencies
install:
	uv sync --all-extras

# Code quality checks
check-coverage:
	uv run pytest --cov-branch --cov=qolmat/ --cov-report=xml tests/

check-quality:
	uv run ruff check qolmat/ tests/

check-security:
	uv run bandit --recursive --configfile=pyproject.toml qolmat/

check-tests:
	uv run pytest tests/

check-types:
	uv run mypy qolmat/ tests/

checkers: check-coverage check-types

# Formatting
format:
	uv run ruff format qolmat/ tests/
	uv run ruff check --fix qolmat/ tests/

# Cleaning
clean:
	rm -rf .mypy_cache .pytest_cache .coverage* .ruff_cache
	rm -rf **__pycache__
	uv run make clean -C docs

# Documentation
doc:
	uv run make html -C docs

doctest:
	uv run pytest --doctest-modules --pyargs qolmat

# Development helpers
lock:
	uv lock

.PHONY: install check-coverage check-quality check-security check-tests check-types checkers format clean doc doctest lock
