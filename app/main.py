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
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', '').strip(),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', '').strip(),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN', '').strip() if os.getenv('AWS_SESSION_TOKEN') else None,
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1').strip()
    )

@st.cache_data(ttl=3600)
def load_new_data():
    """Загрузка новых объединенных данных (с новыми колонками)"""
    s3 = get_s3_client()
    bucket = os.getenv('BUCKET_NAME')
    obj = s3.get_object(Bucket=bucket, Key='data/processed_risk_data.parquet')
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))

@st.cache_data(ttl=3600)
def load_old_data():
    """Загрузка старых данных EM-DAT (ВАЖНО: укажите правильный ключ к старому файлу в S3)"""
    s3 = get_s3_client()
    bucket = os.getenv('BUCKET_NAME')
    # Замените на имя вашего старого файла, если он назывался иначе!
    obj = s3.get_object(Bucket=bucket, Key='data/emdat_old_data.parquet') 
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))

@st.cache_resource
def load_model_resources(model_type):
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

# ================= НАВИГАЦИЯ =================
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите версию дашборда:", 
    ["🌍 Версия 2.0 (Все источники)", "📜 Версия 1.0 (Только EM-DAT)"]
)
st.sidebar.markdown("---")


# ================= СТРАНИЦА 1: НОВАЯ ВЕРСИЯ =================
if page == "🌍 Версия 2.0 (Все источники)":
    st.title("🌍 Global Catastrophic Risk Monitor (v2.0)")
    st.markdown("Объединенные данные: Природные катастрофы, Терроризм, Авиация.")

    try:
        df = load_new_data()
        
        st.sidebar.header("Фильтры v2.0")
        min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
        years = st.sidebar.slider("Период", min_year, max_year, (2000, max_year))
        
        sources = st.sidebar.multiselect("Источник данных", df['Source'].unique(), default=df['Source'].unique())
        
        mask = (df['Year'].between(years[0], years[1]))
        if sources:
            mask = mask & (df['Source'].isin(sources))
        temp_df = df[mask]
        
        types = st.sidebar.multiselect("Тип события", temp_df['Event_Type'].unique())
        if types:
            mask = mask & (df['Event_Type'].isin(types))
            
        filtered_df = df[mask]
        
        tab1, tab2 = st.tabs(["📊 Аналитика", "🤖 AI Прогноз"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Всего событий", f"{len(filtered_df):,}")
            c2.metric("Погибло (Fatalities)", f"{filtered_df['Fatalities'].sum():,.0f}")
            total_damage_b = filtered_df["Damage (mill dollars)"].sum() / 1000 
            c3.metric("Ущерб (USD)", f"${total_damage_b:.1f} B")
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Динамика по годам")
                ev_per_year = filtered_df.groupby(['Year', 'Source']).size().reset_index(name='Количество')
                fig = px.bar(ev_per_year, x='Year', y='Количество', color='Source')
                st.plotly_chart(fig, use_container_width=True)
                
            with col_chart2:
                st.subheader("Карта событий (Погибшие)")
                country_stats = filtered_df.groupby('Country')['Fatalities'].sum().reset_index()
                fig_map = px.choropleth(
                    country_stats, locationmode='country names', locations="Country", 
                    color="Fatalities", hover_name="Country", color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_map, use_container_width=True)
                
            st.subheader("Последние события")
            st.dataframe(filtered_df.sort_values(by='Date', ascending=False).head(100), use_container_width=True)

        with tab2:
            st.header("ML Risk Prediction")
            st.info("⚠️ Модели пока настроены на старую структуру данных. Требуется переобучение.")

    except Exception as e:
        st.error(f"Ошибка загрузки новых данных: {e}")


# ================= СТРАНИЦА 2: СТАРАЯ ВЕРСИЯ =================
elif page == "📜 Версия 1.0 (Только EM-DAT)":
    st.title("🌍 SDU Risk Monitor (Classic)")
    st.markdown("Архивная версия дашборда на основе данных EM-DAT.")

    try:
        old_df = load_old_data()
        
        st.sidebar.header("Фильтры v1.0")
        years_old = st.sidebar.slider("Период", int(old_df['Start Year'].min()), int(old_df['Start Year'].max()), (2010, 2026))
        types_old = st.sidebar.multiselect("Тип катастрофы", old_df['Disaster Type'].unique())
        
        mask_old = (old_df['Start Year'].between(years_old[0], years_old[1]))
        if types_old:
            mask_old = mask_old & (old_df['Disaster Type'].isin(types_old))
        filtered_old_df = old_df[mask_old]
        
        tab1_old, tab2_old = st.tabs(["📊 Аналитика", "🤖 AI Прогноз"])
        
        with tab1_old:
            c1, c2, c3 = st.columns(3)
            c1.metric("Всего событий", len(filtered_old_df))
            c2.metric("Пострадало людей", f"{filtered_old_df['Total Affected'].sum():,.0f}")
            total_damage_old_b = filtered_old_df["Total Damage, Adjusted ('000 US$)"].sum() / 1e9
            c3.metric("Ущерб (USD)", f"${total_damage_old_b:.1f} B")
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Динамика по годам")
                # Считаем количество строк (событий) в каждом году
                ev_per_year_old = filtered_old_df.groupby('Start Year').size().reset_index(name='Total Events')
                
                fig_old = px.bar(ev_per_year_old, x='Start Year', y='Total Events')
                st.plotly_chart(fig_old, use_container_width=True)
                
            with col_chart2:
                st.subheader("Карта ущерба")
                fig_map_old = px.choropleth(
                    filtered_old_df, locations="ISO", color="Total Damage ('000 US$)",
                    hover_name="Country", color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_map_old, use_container_width=True)

        with tab2_old:
            st.header("ML Risk Prediction")
            model_names = {
                "Случайный Лес (Random Forest)": "rf",
                "Градиентный Бустинг (Gradient Boosting)": "gb",
                "Логистическая Регрессия (Logistic Regression)": "lr"
            }
            selected_model_ui = st.selectbox("Выберите AI-модель", list(model_names.keys()))
            selected_model_key = model_names[selected_model_ui]
            
            c1, c2 = st.columns(2)
            sim_gdp = c1.number_input("ВВП Страны (USD)", value=10000000000)
            sim_type = c2.selectbox("Тип угрозы", old_df['Disaster Type'].unique())
            
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
        st.warning("Не удалось загрузить старые данные. Проверьте ключ файла в S3 (data/emdat_old_data.parquet).")
        st.error(f"Детали ошибки: {e}")