PYTHON=python

lint:
black --check .
isort --check .
ruff .

format:
black .
isort .

mypy:
mypy --strict services webapp

test:
pytest -q

run-ui:
gunicorn -k uvicorn.workers.UvicornWorker webapp.app:app
