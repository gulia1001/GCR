FROM apache/airflow:2.8.1
USER airflow
COPY requirements_docker.txt /requirements_docker.txt
RUN pip install --no-cache-dir -r /requirements_docker.txt