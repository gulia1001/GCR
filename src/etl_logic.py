import pandas as pd
import requests
import logging

def get_gdp_data():
    """Качает ВВП стран через API World Bank"""
    url = "http://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
    params = {"format": "json", "date": "2000:2024", "per_page": 15000}
    
    try:
        r = requests.get(url, params=params)
        data = r.json()[1]
        
        # Преобразуем JSON в DataFrame
        records = [{'ISO': x['countryiso3code'], 'Year': int(x['date']), 'GDP': x['value']} for x in data]
        return pd.DataFrame(records)
    except Exception as e:
        logging.error(f"API Error: {e}")
        return pd.DataFrame(columns=['ISO', 'Year', 'GDP'])

def run_etl(csv_path):
    # 1. Читаем CSV. 
    # skiprows=[1] удаляет строку с метаданными (#date +occurred...)
    logging.info("Reading local CSV...")
    df = pd.read_csv(csv_path, skiprows=[1])
    
    # 2. Чистим колонки (удаляем лишние пробелы в именах)
    df.columns = df.columns.str.strip()
    
    # Приводим типы данных. Ошибки превращаем в NaN (coerce) и заменяем на 0
    num_cols = ['Total Affected', 'Total Deaths', 'Total Damage (USD, adjusted)']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 3. Добавляем данные о ВВП
    logging.info("Fetching World Bank Data...")
    gdp_df = get_gdp_data()
    
    # 4. Merge (Left Join)
    final_df = pd.merge(df, gdp_df, on=['ISO', 'Year'], how='left')
    
    # Заполняем пропуски ВВП средним значением (чтобы модель не падала)
    final_df['GDP'] = final_df['GDP'].fillna(final_df['GDP'].mean())
    
    logging.info(f"ETL Complete. Rows: {len(final_df)}")
    return final_df