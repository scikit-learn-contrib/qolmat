
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

clean:
	rm -rf .mypy_cache .pytest_cache .coverage*
	rm -rf **__pycache__
	make clean -C docs

coverage:
	uv run pytest --cov-branch --cov=qolmat --cov-report=xml tests

doc:
	make html -C docs

doctest:
	uv run pytest --doctest-modules --pyargs qolmat
