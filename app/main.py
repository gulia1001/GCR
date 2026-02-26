import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import joblib
import os
import io

st.set_page_config(layout="wide", page_title="SDU Risk Monitor")

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID').strip(),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY').strip(),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN').strip(),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1').strip()
    )

@st.cache_data
def load_data():
    s3 = get_s3_client()
    bucket = os.getenv('BUCKET_NAME')
    obj = s3.get_object(Bucket=bucket, Key='data/processed_risk_data.parquet')
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))

@st.cache_resource
def load_model_resources(model_type):
    """Теперь загружает конкретную модель по выбору пользователя"""
    s3 = get_s3_client()
    bucket = os.getenv('BUCKET_NAME')
    
    with io.BytesIO() as f:
        s3.download_fileobj(bucket, f'models/{model_type}_classifier.joblib', f)
        f.seek(0)
        model = joblib.load(f)
        
    with io.BytesIO() as f:
        s3.download_fileobj(bucket, f'models/{model_type}_columns.joblib', f)
        f.seek(0)
        cols = joblib.load(f)
        
    return model, cols

st.title("🌍 Global Catastrophic Risk Monitor")
st.markdown("Платформа для анализа природных катастроф и предсказания рисков.")

try:
    df = load_data()
    
    st.sidebar.header("Фильтры")
    years = st.sidebar.slider("Период", int(df['Start Year'].min()), int(df['Start Year'].max()), (2010, 2026))
    types = st.sidebar.multiselect("Тип катастрофы", df['Disaster Type'].unique())
    
    mask = (df['Start Year'].between(years[0], years[1]))
    if types:
        mask = mask & (df['Disaster Type'].isin(types))
    filtered_df = df[mask]
    
    tab1, tab2 = st.tabs(["📊 Аналитика", "🤖 AI Прогноз"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего событий", len(filtered_df))
        c2.metric("Пострадало людей", f"{filtered_df['Total Affected'].sum():,.0f}")
        # 1. Сначала считаем сумму и переводим в миллиарды
        total_damage_b = filtered_df["Total Damage, Adjusted ('000 US$)"].sum() / 1e9
        
        # 2. Затем красиво выводим в метрику (никакого конфликта кавычек!)
        c3.metric("Ущерб (USD)", f"${total_damage_b:.1f} B")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Динамика по годам")
            ev_per_year = filtered_df.groupby('Start Year')['Total Events'].sum().reset_index()
            fig = px.bar(ev_per_year, x='Start Year', y='Total Events')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            st.subheader("Карта ущерба")
            fig_map = px.choropleth(
                filtered_df, locations="ISO", color="Total Damage ('000 US$)",
                hover_name="Country", color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.header("ML Risk Prediction")
        
        # ВЫБОР МОДЕЛИ
        model_names = {
            "Случайный Лес (Random Forest)": "rf",
            "Градиентный Бустинг (Gradient Boosting)": "gb",
            "Логистическая Регрессия (Logistic Regression)": "lr"
        }
        selected_model_ui = st.selectbox("Выберите AI-модель для расчета", list(model_names.keys()))
        selected_model_key = model_names[selected_model_ui]
        
        c1, c2 = st.columns(2)
        sim_gdp = c1.number_input("ВВП Страны (USD)", value=10000000000)
        sim_type = c2.selectbox("Тип угрозы", df['Disaster Type'].unique())
        
        if st.button("Рассчитать риск"):
            model, model_cols = load_model_resources(selected_model_key)
            
            input_data = pd.DataFrame(0, index=[0], columns=model_cols)
            input_data['GDP'] = sim_gdp
            input_data['Start Year'] = 2026
            
            type_col = f"Type_{sim_type}"
            if type_col in input_data.columns:
                input_data[type_col] = 1
                
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"⚠️ ВЫСОКИЙ РИСК! ({selected_model_ui}). Вероятность: {prob:.1%}")
            else:
                st.success(f"✅ Низкий риск ({selected_model_ui}). Вероятность: {prob:.1%}")

except Exception as e:
    st.warning("Данные загружаются или Airflow Pipeline еще не завершен.")
    st.error(f"Error details: {e}")