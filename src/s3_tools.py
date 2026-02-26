import boto3
import os
import io
import pandas as pd
import logging

def get_s3_client():
    access_key = os.getenv('AWS_ACCESS_KEY_ID', '').strip()
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '').strip()
    token = os.getenv('AWS_SESSION_TOKEN')
    token = token.strip() if token else None # Токен может быть None
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1').strip()
    
    return boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=token,
        region_name=region
    )

def get_bucket_name():
    """Вспомогательная функция для получения имени бакета"""
    return os.getenv('BUCKET_NAME')

def upload_df_parquet(df, key):
    """Сохраняет DataFrame как Parquet (быстрее и легче CSV) в S3"""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    s3 = get_s3_client()
    bucket = get_bucket_name()
    s3.upload_fileobj(buffer, bucket, key)
    logging.info(f"✅ Uploaded Parquet to s3://{bucket}/{key}")

def upload_df_csv(df, key):
    """Сохраняет DataFrame как CSV в S3 (добавлено для extract_noaa)"""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3 = get_s3_client()
    bucket = get_bucket_name()
    # Для строк (StringIO) используем put_object, а не upload_fileobj
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    logging.info(f"✅ Uploaded CSV to s3://{bucket}/{key}")

def upload_model(model_buffer, key):
    """Сохраняет бинарный файл модели в S3"""
    s3 = get_s3_client()
    bucket = get_bucket_name()
    s3.upload_fileobj(model_buffer, bucket, key)
    logging.info(f"✅ Uploaded model to s3://{bucket}/{key}")

def upload_to_s3(data_string, key):
    """Загружает готовую строку (например, сырой JSON или CSV) в S3"""
    s3 = get_s3_client()
    bucket = get_bucket_name()
    s3.put_object(Bucket=bucket, Key=key, Body=data_string)
    logging.info(f"✅ Успешно загружено в S3: s3://{bucket}/{key}")

def read_csv_from_s3(key, **kwargs):
    """Читает CSV из S3 через boto3 и возвращает DataFrame"""
    s3 = get_s3_client()
    bucket = get_bucket_name()
    
    logging.info(f"Чтение CSV из s3://{bucket}/{key}...")
    response = s3.get_object(Bucket=bucket, Key=key)
    file_content = response['Body'].read().decode('utf-8') 
    return pd.read_csv(io.StringIO(file_content), **kwargs)
