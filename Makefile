coverage:
	pytest --cov-branch --cov=qolmat --cov-report=xml tests

doctest:
	pytest --doctest-modules --pyargs qolmat

doc:
	make html -C docs

clean:
	rm -rf .mypy_cache .pytest_cache .coverage*
	rm -rf **__pycache__
	make clean -C docs
