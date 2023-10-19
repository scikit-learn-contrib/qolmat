coverage:
	pytest --cov-branch --cov=qolmat --cov-report=xml

doctest:
	pytest --doctest-modules --pyargs qolmat

clean:
	rm -rf .mypy_cache .pytest_cache .coverage*
	rm -rf **__pycache__
