from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow')
from src.etl_logic import run_etl
from src.ml_logic import train_risk_model
from src.s3_tools import upload_df_parquet, upload_model

def etl_task():
    df = run_etl('/opt/airflow/data/emdat.csv')
    upload_df_parquet(df, 'data/processed_risk_data.parquet')

def ml_task(model_name):
    df = run_etl('/opt/airflow/data/emdat.csv') 
    
    # Передаем тип модели в функцию
    model_buf, cols_buf = train_risk_model(df, model_type=model_name)
    
    # Сохраняем модели под разными именами
    upload_model(model_buf, f'models/{model_name}_classifier.joblib')
    upload_model(cols_buf, f'models/{model_name}_columns.joblib')

with DAG('risk_analysis_v1', start_date=datetime(2023, 1, 1), schedule_interval=None, catchup=False) as dag:
    
    t1 = PythonOperator(task_id='etl_process', python_callable=etl_task)
    
    # Три параллельные задачи для разных моделей
    t2 = PythonOperator(task_id='train_random_forest', python_callable=ml_task, op_kwargs={'model_name': 'rf'})
    t3 = PythonOperator(task_id='train_gradient_boosting', python_callable=ml_task, op_kwargs={'model_name': 'gb'})
    t4 = PythonOperator(task_id='train_logistic_regression', python_callable=ml_task, op_kwargs={'model_name': 'lr'})
    
    # t1 выполнится первым, затем t2, t3 и t4 запустятся параллельно
    t1 >> [t2, t3, t4]