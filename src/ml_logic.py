import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import io
import logging

def train_risk_model(df, model_type='rf'):
    """
    Обучает выбранную модель и вычисляет её точность.
    model_type: 'rf' (Random Forest), 'gb' (Gradient Boosting), 'lr' (Logistic Regression)
    """
    logging.info(f"Начинаем подготовку данных для модели: {model_type}")
    
    df['Is_High_Risk'] = ((df['Total Deaths'] > 100) | (df['Total Affected'] > 1000000)).astype(int)
    features = ['Start Year', 'GDP']
    
    if 'Disaster Type' in df.columns:
        df_encoded = pd.get_dummies(df, columns=['Disaster Type'], prefix='Type')
    else:
        df_encoded = df.copy()
        
    feature_cols = [c for c in df_encoded.columns if c.startswith('Type_') or c in features]
    
    X = df_encoded[feature_cols].fillna(0)
    y = df_encoded['Is_High_Risk']
    
    # Разбиваем данные для проверки точности (80% обучение, 20% тест)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Выбор модели
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Неизвестный тип модели")

    # Обучение
    logging.info("Обучение модели...")
    model.fit(X_train, y_train)
    
    # Проверка точности
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # ВЫВОД В ЛОГИ AIRFLOW
    logging.info(f"==========================================")
    logging.info(f"РЕЗУЛЬТАТЫ МОДЕЛИ {model_type.upper()}:")
    logging.info(f"Точность (Accuracy): {accuracy * 100:.2f}%")
    logging.info(f"==========================================")
    
    # Сохранение в буфер
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    
    cols_buffer = io.BytesIO()
    joblib.dump(feature_cols, cols_buffer)
    cols_buffer.seek(0)
    
    return buffer, cols_buffer