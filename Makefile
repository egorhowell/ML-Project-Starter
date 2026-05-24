.PHONY: install install-dev lint format type-check test check clean run dashboard

# Install runtime dependencies only (used in production / CI deploys).
install:
	poetry install --without dev

# Install everything, including dev tools (linters, type checker, pytest).
# Use this on your local machine when developing.
install-dev:
	poetry install

# Check code style without making changes. Fails if anything would be reformatted.
lint:
	poetry run ruff check src tests
	poetry run black --check src tests

# Auto-fix style issues and format the code.
format:
	poetry run ruff check --fix src tests
	poetry run black src tests

# Static type checking. Catches type errors before runtime.
type-check:
	poetry run mypy src

# Run the test suite.
test:
	poetry run pytest

# One command to format, lint, type-check, and test. Run this before pushing.
check: format lint type-check test

# Clean up Python and tool caches. Useful if something gets into a weird state.
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type f -name ".coverage" -delete

# Run the main pipeline once: extract → preprocess → predict → save.
run:
	poetry run python -m src.main

# Launch the Streamlit dashboard locally on http://localhost:8501
dashboard:
	poetry run streamlit run src/streamlit_app.py
