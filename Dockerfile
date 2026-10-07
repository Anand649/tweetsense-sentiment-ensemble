# Serving image: Flask dashboard + API over the exported models in models/.
# Build after running the benchmark so the models are baked in, or mount
# them at runtime:  docker run -v $PWD/models:/app/models -p 5000:5000 ...
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[postgres]" gunicorn

COPY app ./app
COPY configs ./configs
COPY models ./models
COPY reports ./reports

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/health')" || exit 1
CMD ["sh", "-c", "gunicorn --chdir app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 60 'app:create_app()'"]
