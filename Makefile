
check-coverage:
	poetry run pytest --cov-branch --cov=qolmat/ --cov-report=xml tests/

check-poetry:
	poetry check --lock

check-quality:
	poetry run ruff check qolmat/ tests/

check-security:
	poetry run bandit --recursive --configfile=pyproject.toml qolmat/

check-tests:
	poetry run pytest tests/

check-types:
	poetry run mypy qolmat/ tests/

checkers: check-coverage check-types

clean:
	rm -rf .mypy_cache .pytest_cache .coverage*
	rm -rf **__pycache__
	make clean -C docs

coverage:
	poetry run pytest --cov-branch --cov=qolmat --cov-report=xml tests

doc:
	make html -C docs

doctest:
	poetry run pytest --doctest-modules --pyargs qolmat
