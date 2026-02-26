from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow')
from src.etl_logic import run_etl
from src.ml_logic import train_risk_model
from src.s3_tools import upload_df_parquet, upload_model

# Убран импорт mock_extract_local_files
from src.gulia_part import extract_all_noaa, extract_aviation, transform_and_load_processed

def etl_for_map():
    df = run_etl('/opt/airflow/data/emdat.csv')
    upload_df_parquet(df, 'data/processed_risk_data.parquet')

def ml_task(model_name):
    df = run_etl('/opt/airflow/data/emdat.csv') 
    
    model_buf, cols_buf = train_risk_model(df, model_type=model_name)
    
    upload_model(model_buf, f'models/{model_name}_classifier.joblib')
    upload_model(cols_buf, f'models/{model_name}_columns.joblib')


with DAG('risk_analysis_v1', start_date=datetime(2023, 1, 1), schedule_interval=None, catchup=False) as dag:
    
    # --- ВЕТКА 1: Старый процесс (Обучение ML) ---
    t1 = PythonOperator(task_id='prep_map', python_callable=etl_for_map)
    
    t2 = PythonOperator(task_id='train_random_forest', python_callable=ml_task, op_kwargs={'model_name': 'rf'})
    t3 = PythonOperator(task_id='train_gradient_boosting', python_callable=ml_task, op_kwargs={'model_name': 'gb'})
    t4 = PythonOperator(task_id='train_logistic_regression', python_callable=ml_task, op_kwargs={'model_name': 'lr'})
    
    t1 >> [t2, t3, t4]

    # --- ВЕТКА 2: Новый процесс (Глобальные данные ETL) ---
    t_extract_noaa = PythonOperator(task_id='extract_noaa_all', python_callable=extract_all_noaa)
    t_extract_avia = PythonOperator(task_id='extract_aviation', python_callable=extract_aviation)
    t_transform_load = PythonOperator(task_id='transform_and_load_to_processed', python_callable=transform_and_load_processed)
    
    # Настраиваем зависимости (Убрали t_check_files)
    [t_extract_noaa, t_extract_avia] >> t_transform_load
