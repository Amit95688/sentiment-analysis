# syntax=docker/dockerfile:1.4
FROM apache/airflow:2.8.1

USER root

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Set workdir
WORKDIR /opt/airflow

# Copy requirements
COPY requirements.txt ./

# Copy wheels if present
COPY wheels/ ./wheels/

# Install torch from wheel if available, else from PyPI
RUN set -e; \
    if [ -d "wheels" ] && ls wheels/torch*.whl 1> /dev/null 2>&1; then \
        pip install wheels/torch*.whl; \
    else \
        pip install torch; \
    fi

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy DAGs
COPY dags/ /opt/airflow/dags/

# Copy any additional source code if needed
COPY src/ /opt/airflow/src/

# Set Airflow environment variables (optional, can be set in docker-compose.yml)
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Entrypoint and CMD are inherited from base image
