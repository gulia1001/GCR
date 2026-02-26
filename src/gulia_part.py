import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging

from src.s3_tools import (
    upload_df_parquet, 
    upload_df_csv, 
    read_csv_from_s3, 
)

HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/145.0.0.0 Safari/537.36'
}

URLS = {
    'earthquake': 'https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/earthquakes',
    'tsunami': 'https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events',
    'volcano': 'https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/volcanoes?order=year%3Aasc'
}

# --- EXTRACT ---
def extract_noaa_generic(endpoint, s3_file_name):
    base_url = URLS[endpoint.lower()]
    all_items = []
    page = 1
    
    logging.info(f"Начинаем выгрузку NOAA: {endpoint}...")
    while True:
        response = requests.get(base_url, params={'page': str(page)}, headers=HEADERS)
        if response.status_code != 200:
            break
        data = response.json()
        items = data.get('items', [])
        if not items:
            break
        all_items.extend(items)
        page += 1
        time.sleep(0.5)

    df = pd.DataFrame(all_items)
    upload_df_csv(df, f"raw/noaa/{s3_file_name}.csv")

def extract_all_noaa(**kwargs):
    extract_noaa_generic('earthquake', 'earthquakes_raw')
    extract_noaa_generic('tsunami', 'tsunamis_raw') 
    extract_noaa_generic('volcano', 'volcanoes_raw')

def extract_aviation(**kwargs):
    url = 'https://aviation-safety.net/database/year/2024/1'
    logging.info(f"Парсинг Aviation Safety: {url}...")
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    table_headers = [th.get_text(strip=True) for th in soup.find_all("th")]
    table_headers = [h if h != "\xa0" else f"extra_{i}" for i, h in enumerate(table_headers)]

    rows = []
    for tr in soup.find_all("tr", class_="list"):
        cells = tr.find_all("td")
        row = []
        for td in cells:
            link = td.find("a")
            if link:
                row.append(link.get_text(strip=True))
            else:
                img = td.find("img")
                row.append(img["alt"] if img and img.get("alt") else td.get_text(strip=True))
        while len(row) < len(table_headers):
            row.append("")
        rows.append(row)

    df = pd.DataFrame(rows, columns=table_headers)
    upload_df_csv(df, "raw/aircraft/aviation_safety_raw.csv")

# --- TRANSFORM ---
def transform_noaa(raw_df, event_type):
    df = raw_df.copy()
    cols_to_keep = ['year', 'month', 'day', 'country', 'deaths', 'damageMillionsDollars', 'housesDestroyed']
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']].fillna(1), errors='coerce')

    return pd.DataFrame({
        'Date': df['Date'],
        'Year': pd.to_numeric(df.get('year'), errors='coerce'),
        'Country': df.get('country', 'Unknown').astype(str).str.title(),
        'Event_Type': event_type,
        'Fatalities': pd.to_numeric(df.get('deaths', 0), errors='coerce').fillna(0),
        'Damage (mill dollars)': pd.to_numeric(df.get('damageMillionsDollars', 0), errors='coerce').fillna(0),
        'Houses Destroyed': pd.to_numeric(df.get('housesDestroyed', 0), errors='coerce').fillna(0),
        'Source': 'NOAA'
    }).dropna(subset=['Year'])

def transform_emdat(raw_df):
    df = raw_df.copy()
    
    df.columns = df.columns.str.strip()
    
    year_col = 'Start Year' if 'Start Year' in df.columns else 'Year'
    damage_col = "Total Damage ('000 US$)" if "Total Damage ('000 US$)" in df.columns else 'Total Damage'
    
    year_s = df.get(year_col, pd.Series(1970, index=df.index))
    month_s = df.get('Start Month', pd.Series(1, index=df.index))
    day_s = df.get('Start Day', pd.Series(1, index=df.index))
    
    df['Date'] = pd.to_datetime(
        pd.DataFrame({'year': year_s, 'month': month_s, 'day': day_s}).fillna(1), 
        errors='coerce'
    )
    

    fatalities = pd.to_numeric(df.get('Total Deaths', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    damage = pd.to_numeric(df.get(damage_col, pd.Series(0, index=df.index)), errors='coerce').fillna(0) / 1000
    houses = pd.to_numeric(df.get('Total Destroyed', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    
    # 3. Формируем финальный датафрейм
    return pd.DataFrame({
        'Date': df['Date'],
        'Year': pd.to_numeric(df.get(year_col), errors='coerce'),
        'Country': df.get('Country', pd.Series('Unknown', index=df.index)).astype(str),
        'Event_Type': df.get('Disaster Type', pd.Series('Disaster', index=df.index)).astype(str),
        'Fatalities': fatalities,
        'Damage (mill dollars)': damage,
        'Houses Destroyed': houses,
        'Source': 'EM-DAT'
    }).dropna(subset=['Year'])

def transform_gtd(raw_df):
    logging.info("Трансформация данных GTD...")
    df = raw_df.copy()
    
    month_series = df.get('imonth', pd.Series(1, index=df.index))
    day_series = df.get('iday', pd.Series(1, index=df.index))
    prop_series = df.get('propvalue', pd.Series(0, index=df.index))
    
    df['month_clean'] = month_series.apply(lambda x: int(x) if pd.notnull(x) and x > 0 else 1)
    df['day_clean'] = day_series.apply(lambda x: int(x) if pd.notnull(x) and x > 0 else 1)
    
    year_str = pd.to_numeric(df.get('iyear', 1970), errors='coerce').fillna(1970).astype(int).astype(str)
    
    df['Date'] = pd.to_datetime(
        year_str + '-' + df['month_clean'].astype(str) + '-' + df['day_clean'].astype(str), 
        errors='coerce'
    )
    
    damage = pd.to_numeric(prop_series, errors='coerce').fillna(0)
    damage = damage.apply(lambda x: x / 1_000_000 if x > 0 else 0.0)
    
    return pd.DataFrame({
        'Date': df['Date'],
        'Year': pd.to_numeric(df.get('iyear'), errors='coerce'),
        'Country': df.get('country_txt', 'Unknown').astype(str),
        'Event_Type': df.get('attacktype1_txt', 'Terrorism').astype(str),
        'Fatalities': pd.to_numeric(df.get('nkill', 0), errors='coerce').fillna(0),
        'Damage (mill dollars)': damage,
        'Houses Destroyed': 0.0,
        'Source': 'GTD'
    }).dropna(subset=['Year'])

def transform_aviation(raw_df):
    logging.info("Трансформация данных Aviation...")
    df = raw_df.copy()
    
    df.columns = df.columns.str.strip().str.lower()
    
    date_series = df.get('date', pd.Series(pd.NaT, index=df.index))
    df['Date'] = pd.to_datetime(date_series, errors='coerce')
    
    
    fat_col = 'fat.' if 'fat.' in df.columns else ('fatalities' if 'fatalities' in df.columns else None)
    
    if fat_col:
        fatalities = pd.to_numeric(df[fat_col].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0)
    else:
        fatalities = pd.Series(0, index=df.index)
        
    return pd.DataFrame({
        'Date': df['Date'],
        'Year': df['Date'].dt.year,
        'Country': df.get('location', pd.Series('Unknown', index=df.index)).astype(str),
        'Event_Type': pd.Series('Aviation Accident', index=df.index).astype(str),
        'Fatalities': fatalities,
        'Damage (mill dollars)': 0.0,
        'Houses Destroyed': 0.0,
        'Source': 'Aviation Safety Network'
    }).dropna(subset=['Year'])

# --- LOAD ---
def transform_and_load_processed(**kwargs):
    logging.info("Чтение сырых данных...")
    
    try:
        logging.info("Поиск EM-DAT в S3...")
        raw_emdat = read_csv_from_s3("raw/emdat.csv", skiprows=6)
    except Exception as e:
        logging.warning(f"Файл EM-DAT в S3 не найден. Читаем локально: {e}")
        raw_emdat = pd.read_csv("/opt/airflow/data/emdat.csv", skiprows=6)
    print("ТАК ЗДЕСЬ ВРОДЕ ВСЕ НОРМ ЕСЛИ МЕНЯ НЕТУ ТЕБЕ ПИЗДЕС")
    try:
        logging.info("Поиск GTD в S3...")
        raw_gtd = read_csv_from_s3("raw/gtd.csv")
    except Exception as e:
        logging.warning(f"Файл GTD в S3 не найден. Читаем локально: {e}")
        raw_gtd = pd.read_csv("/opt/airflow/data/gtd.csv")

    raw_eq = read_csv_from_s3("raw/noaa/earthquakes_raw.csv")
    raw_ts = read_csv_from_s3("raw/noaa/tsunamis_raw.csv")
    raw_volc = read_csv_from_s3("raw/noaa/volcanoes_raw.csv")
    raw_avia = read_csv_from_s3("raw/aircraft/aviation_safety_raw.csv")

    # Трансформация
    clean_emdat = transform_emdat(raw_emdat)
    clean_gtd = transform_gtd(raw_gtd)
    clean_eq = transform_noaa(raw_eq, 'Earthquake')
    clean_ts = transform_noaa(raw_ts, 'Tsunami')
    clean_volc = transform_noaa(raw_volc, 'Volcano')
    clean_avia = transform_aviation(raw_avia)

    # Объединение
    final_df = pd.concat([clean_eq, clean_ts, clean_volc, clean_avia, clean_emdat, clean_gtd], ignore_index=True)
    
    final_df['Year'] = final_df['Year'].astype(int)
    final_df['Fatalities'] = final_df['Fatalities'].astype(int)
    final_df.drop_duplicates(inplace=True)
    
    # Загрузка финального файла в S3
    upload_df_parquet(final_df, "processed/global_risks_clean.parquet")
    logging.info(f"ETL завершен! Собрано записей: {len(final_df)}.")