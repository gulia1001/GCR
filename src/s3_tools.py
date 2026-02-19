import boto3
import os
import io
import pandas as pd
import logging

def get_s3_client():
    # .strip() убирает пробелы и невидимые символы \r \n
    access_key = os.getenv('AWS_ACCESS_KEY_ID').strip()
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY').strip()
    token = os.getenv('AWS_SESSION_TOKEN').strip()
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1').strip()
    
    return boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=token,
        region_name=region
    )

def upload_df_parquet(df, key):
    """Сохраняет DataFrame как Parquet (быстрее и легче CSV) в S3"""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    s3 = get_s3_client()
    bucket = os.getenv('BUCKET_NAME')
    s3.upload_fileobj(buffer, bucket, key)
    logging.info(f"Uploaded {key} to S3")

def upload_model(model_buffer, key):
    """Сохраняет бинарный файл модели в S3"""
    s3 = get_s3_client()
    bucket = os.getenv('BUCKET_NAME')
    s3.upload_fileobj(model_buffer, bucket, key)
    logging.info(f"Uploaded model {key}")