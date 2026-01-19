from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

POSTGRES_CONN_ID = "postgres_default"
REVIEWS_FILE = "/opt/airflow/artifacts/train.csv"
TMP_DIR = "/tmp/sentiment_etl"

default_args = {"owner": "airflow"}

with DAG(
    dag_id="etl_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["nlp", "sentiment"],
):

    @task
    def init_db():
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        hook.run("""
            CREATE TABLE IF NOT EXISTS review_sentiment (
                id SERIAL PRIMARY KEY,
                review TEXT,
                sentiment INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    @task
    def extract():
        os.makedirs(TMP_DIR, exist_ok=True)
        df = pd.read_csv(REVIEWS_FILE)
        path = f"{TMP_DIR}/reviews.json"
        df["review"].to_json(path, orient="values")
        return path

    @task
    def transform(path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ).to(device)

        reviews = json.load(open(path))
        results = []

        with torch.no_grad():
            for review in reviews:
                inputs = tokenizer(
                    review,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(device)
                sentiment = model(**inputs).logits.argmax(dim=1).item() + 1
                results.append((review, sentiment))

        out = f"{TMP_DIR}/results.json"
        json.dump(results, open(out, "w"))
        return out

    @task
    def load(path: str):
        records = json.load(open(path))
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        hook.run("TRUNCATE TABLE review_sentiment;")
        hook.insert_rows(
            "review_sentiment",
            records,
            target_fields=["review", "sentiment"],
        )

    init_db() >> load(transform(extract()))
