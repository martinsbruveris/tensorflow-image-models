check_dirs := tfimm

# This target runs checks on all files
quality:
	poetry run black --check $(check_dirs)
	poetry run isort --check-only $(check_dirs)
	poetry run flake8 $(check_dirs)

# This target runs checks on all files and potentially modifies some of them
style:
	poetry run black $(check_dirs)
	poetry run isort $(check_dirs)

# Run tests for the library
# test:
# 	python -m pytest -n auto --dist=loadfile -s -v ./tests/
