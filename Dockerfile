FROM apache/airflow:2.8.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
COPY wheels/ /tmp/wheels/
# Install torch from local wheel if available
RUN if ls /tmp/wheels/torch*.whl 1> /dev/null 2>&1; then pip install --no-cache-dir /tmp/wheels/torch*.whl; fi
RUN pip install --no-cache-dir -r /requirements.txt
