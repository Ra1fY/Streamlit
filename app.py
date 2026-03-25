import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Импорт пользовательских модулей
from data_generator import generate_realistic_temperature_data, seasonal_temperatures
from data_analysis import (
    calculate_rolling_stats,
    detect_anomalies,
    calculate_seasonal_stats,
    calculate_trend,
    get_city_summary_stats,
    check_temperature_normal
)
from weather_api import WeatherAPI, compare_sync_vs_async
from parallel_analysis import compare_parallel_vs_sequential

# Конфигурация страницы
st.set_page_config(
    page_title="Система мониторинга температуры",
    page_icon="🌡️",
    layout="wide"
)

# Загрузка CSS стилей
def load_css():
    """Загрузка CSS стилей"""
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Инициализация
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'api_key' not in st.session_state:
    # Загружаем API ключ из .env файла
    st.session_state.api_key = os.getenv("OPENWEATHER_API_KEY", "")

def generate_sample_data():
    """Генерирует пример данных для демонстрации"""
    with st.spinner("Генерация примера данных..."):
        # Используем функцию из data_generator
        df = generate_realistic_temperature_data(list(seasonal_temperatures.keys()))
        return df

def load_data(uploaded_file):
    """Загружает и предобрабатывает данные о температуре"""
    try:
        df = pd.read_csv(uploaded_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Проверка на наличие необходимых колонок
        required_columns = ['city', 'timestamp', 'temperature']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Файл должен содержать колонки: {', '.join(required_columns)}")
            return None
        
        # Если колонка season отсутствует
        if 'season' not in df.columns:
            month_to_season = {
                12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "autumn", 10: "autumn", 11: "autumn"
            }
            df['season'] = df['timestamp'].dt.month.map(lambda x: month_to_season[x])
        
        # Вычисление скользящих статистики для всех городов
        results = []
        for city in df['city'].unique():
            city_data = df[df['city'] == city].copy()
            city_data = calculate_rolling_stats(city_data)
            city_data = detect_anomalies(city_data)
            results.append(city_data)
        
        df = pd.concat(results, ignore_index=True)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None

def create_temperature_time_series(df, city):
    """Создает интерактивный график временного ряда с выделением аномалий"""
    city_data = df[df['city'] == city].copy()
    
    # Проверяем наличие необходимых колонок
    if 'rolling_mean' not in city_data.columns or 'rolling_std' not in city_data.columns:
        # Если нет, вычисляем их
        city_data = calculate_rolling_stats(city_data)
        city_data = detect_anomalies(city_data)
    
    fig = go.Figure()
    
    # Линия температуры
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['temperature'],
        mode='lines',
        name='Температура',
        line=dict(color='#2c3e50', width=2),
        hovertemplate='Дата: %{x}<br>Температура: %{y:.1f}°C<br>%{text}<extra></extra>',
        text=city_data['season']
    ))
    
    # Скользящее среднее
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['rolling_mean'],
        mode='lines',
        name='Скользящее среднее (30 дней)',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        hovertemplate='Дата: %{x}<br>Скользящее среднее: %{y:.1f}°C<extra></extra>'
    ))
    
    # Доверительный интервал
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['rolling_mean'] + 2 * city_data['rolling_std'],
        mode='lines',
        name='Верхняя граница (+2σ)',
        line=dict(color='rgba(231, 76, 60, 0.3)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['rolling_mean'] - 2 * city_data['rolling_std'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.1)',
        line=dict(color='rgba(231, 76, 60, 0.3)', width=0),
        name='Нормальный диапазон'
    ))
    
    # Точки аномалий
    anomalies = city_data[city_data['is_anomaly']]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['temperature'],
            mode='markers',
            name='Аномалии',
            marker=dict(color='#ff6b6b', size=8, symbol='circle'),
            hovertemplate='Дата: %{x}<br>Температура: %{y:.1f}°C<br>⚠️ Обнаружена аномалия!<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Временной ряд температуры для {city}',
        xaxis_title='Дата',
        yaxis_title='Температура (°C)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_seasonal_profile(df, city):
    """Создает визуализацию сезонного профиля"""
    seasonal_stats = calculate_seasonal_stats(df, city)
    
    seasons_ru = {
        'winter': 'Зима',
        'spring': 'Весна', 
        'summer': 'Лето',
        'autumn': 'Осень'
    }
    
    seasons = ['winter', 'spring', 'summer', 'autumn']
    means = [seasonal_stats[s]['mean'] for s in seasons]
    stds = [seasonal_stats[s]['std'] for s in seasons]
    seasons_ru_list = [seasons_ru[s] for s in seasons]
    
    fig = go.Figure()
    
    # Столбчатая диаграмма с планками погрешностей
    fig.add_trace(go.Bar(
        x=seasons_ru_list,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
        text=[f"{m:.1f}°C" for m in means],
        textposition='outside',
        hovertemplate='Сезон: %{x}<br>Средняя: %{y:.1f}°C<br>Стд. отклонение: ±%{customdata:.1f}°C<extra></extra>',
        customdata=stds
    ))
    
    fig.update_layout(
        title=f'Сезонный профиль температуры для {city}',
        xaxis_title='Сезон',
        yaxis_title='Температура (°C)',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """Основное приложение Streamlit"""
    
    # Заголовок
    st.markdown("""
        <div class="main-header">
            <h1>🌡️ Система мониторинга климата</h1>
            <p>Анализ температуры и мониторинг в реальном времени</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.header("📊 Управление")
        
        # Загрузка данных
        st.subheader("1. Загрузка данных")
        
        # Кнопка для генерации примера данных
        if st.button("📁 Сгенерировать пример данных"):
            df = generate_sample_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                # Сохраняем временно в файл для демонстрации
                df.to_csv('temp_generated_data.csv', index=False)
                st.success(f"✅ Сгенерированы данные для {len(df['city'].unique())} городов")
                st.info("Файл 'temp_generated_data.csv' создан и загружен в приложение")
        
        uploaded_file = st.file_uploader(
            "Или загрузите свой CSV файл",
            type=['csv'],
            help="Загрузите CSV файл с историческими данными о температуре"
        )
        
        if uploaded_file is not None:
            with st.spinner("Загрузка данных..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Загружены данные для {len(df['city'].unique())} городов")
        
        # Выбор города (только если данные загружены)
        if st.session_state.data_loaded and st.session_state.df is not None:
            st.subheader("2. Выбор города")
            cities = sorted(st.session_state.df['city'].unique())
            selected_city = st.selectbox("Выберите город", cities)
            
            # Фильтр по дате
            st.subheader("3. Диапазон дат")
            min_date = st.session_state.df['timestamp'].min().date()
            max_date = st.session_state.df['timestamp'].max().date()
            date_range = st.date_input(
                "Выберите диапазон дат",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # API конфигурация
        st.subheader("4. API OpenWeatherMap")
        api_key = st.text_input(
            "API ключ",
            type="password",
            value=st.session_state.api_key,
            help="API ключ автоматически загружается из .env файла. Можете ввести другой для переопределения."
        )

        # Обновляем session state если пользователь ввел новый ключ
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # Тестирование производительности (только если данные загружены)
        if st.session_state.data_loaded and st.session_state.df is not None:
            st.subheader("5. Тестирование производительности")
            
            # Тест параллельной обработки
            if st.button("🚀 Сравнить параллельную и последовательную обработку"):
                with st.spinner("Сравнение производительности..."):
                    comparison = compare_parallel_vs_sequential(st.session_state.df)
                    st.success("Анализ завершен!")
                    
                    # Вывод комментария о результатах
                    st.markdown("""
                    **О параллельной обработке:**
                    
                    Параллельная обработка демонстрирует значительное ускорение за счет 
                    использования всех доступных ядер процессора. При анализе данных для 
                    нескольких городов, независимые вычисления могут выполняться одновременно,
                    что сокращает общее время выполнения.
                    """)
                    
                    st.metric("Ускорение", f"{comparison['speedup']:.2f}x")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Последовательная обработка", f"{comparison['sequential_time']:.2f} сек")
                    with col2:
                        st.metric("Параллельная обработка", f"{comparison['parallel_time']:.2f} сек")
                    
                    st.info(f"Проанализировано {comparison['cities_analyzed']} городов, {comparison['records_processed']:,} записей")
        
        # Сравнение API подходов
        if api_key:
            st.subheader("6. Тест API производительности")
            test_cities = ["London", "Paris", "Tokyo", "New York"]
            if st.button("🌐 Сравнить синхронные и асинхронные API запросы"):
                with st.spinner("Тестирование API..."):
                    results = compare_sync_vs_async(api_key, test_cities, num_calls=3)
                    st.success("Тест завершен!")
                    
                    # Вывод комментария о сравнении подходов
                    st.markdown("""
                    **О сравнении подходов:**
                    
                    **Синхронные запросы** выполняются последовательно - каждый следующий запрос
                    ожидает завершения предыдущего. Это просто в реализации, но неэффективно при
                    большом количестве запросов.
                    
                    **Асинхронные запросы** выполняются параллельно, позволяя отправлять несколько
                    запросов одновременно и обрабатывать ответы по мере их поступления. Это значительно
                    ускоряет работу при получении данных для нескольких городов.
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Синхронные (среднее)", f"{results['sync_avg']:.2f} сек")
                    with col2:
                        st.metric("Асинхронные (среднее)", f"{results['async_avg']:.2f} сек")
                    st.metric("Ускорение", f"{results['speedup']:.2f}x")
    
    # Основной контент
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("👈 Загрузите файл с данными о температуре или сгенерируйте пример данных для начала работы")
        
        # Пример формата данных
        with st.expander("📋 Требования к формату данных"):
            st.write("Ваш CSV файл должен содержать следующие колонки:")
            st.code("""
                city: Название города
                timestamp: Дата (формат YYYY-MM-DD)
                temperature: Температура в Цельсиях
                season: Сезон (опционально, будет определен автоматически)
            """)
            st.write("Пример данных:")
            example_df = pd.DataFrame({
                'city': ['New York', 'New York', 'London'],
                'timestamp': ['2023-01-01', '2023-01-02', '2023-01-01'],
                'temperature': [0.5, 1.2, 5.8],
                'season': ['winter', 'winter', 'winter']
            })
            st.dataframe(example_df)
    
    else:
        # Фильтр по дате
        if 'date_range' in locals() and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = st.session_state.df[
                (st.session_state.df['timestamp'].dt.date >= start_date) &
                (st.session_state.df['timestamp'].dt.date <= end_date)
            ]
        else:
            filtered_df = st.session_state.df
        
        # Фильтруем для выбранного города
        if 'selected_city' in locals():
            city_data = filtered_df[filtered_df['city'] == selected_city].copy()
        else:
            city_data = filtered_df[filtered_df['city'] == filtered_df['city'].unique()[0]].copy()
            selected_city = filtered_df['city'].unique()[0]
        
        if len(city_data) == 0:
            st.warning(f"Нет данных для {selected_city} в выбранном диапазоне дат")
            return
        
        # Сводная статистика
        st.subheader(f"📈 Анализ температуры: {selected_city}")
        
        # Метрики
        summary_stats = get_city_summary_stats(filtered_df, selected_city)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Средняя температура", f"{summary_stats['mean']:.1f}°C")
        with col2:
            st.metric("Диапазон температур", f"{summary_stats['min']:.1f}°C - {summary_stats['max']:.1f}°C")
        with col3:
            st.metric("Стандартное отклонение", f"{summary_stats['std']:.1f}°C")
        with col4:
            anomaly_count = city_data['is_anomaly'].sum() if 'is_anomaly' in city_data.columns else 0
            anomaly_pct = (anomaly_count / len(city_data)) * 100 if len(city_data) > 0 else 0
            st.metric("Обнаружено аномалий", f"{anomaly_count} ({anomaly_pct:.1f}%)")
        
        # Анализ тренда
        slope, intercept = calculate_trend(filtered_df, selected_city)
        trend_direction = "потепление" if slope > 0 else "похолодание"
        trend_strength = abs(slope) * 365  # Годовое изменение
        
        with st.expander("📊 Анализ долгосрочного тренда"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Направление тренда:** {trend_direction.upper()}")
                st.write(f"**Годовое изменение:** {trend_strength:.2f}°C/год")
                total_change = trend_strength * (city_data['timestamp'].max().year - city_data['timestamp'].min().year)
                st.write(f"**Общее изменение за период:** {total_change:.2f}°C")
            with col2:
                # Визуализация тренда
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=city_data['timestamp'],
                    y=city_data['temperature'],
                    mode='markers',
                    name='Данные',
                    marker=dict(size=2, opacity=0.5)
                ))
                trend_line = slope * (city_data['timestamp'] - city_data['timestamp'].min()).dt.days + intercept
                fig_trend.add_trace(go.Scatter(
                    x=city_data['timestamp'],
                    y=trend_line,
                    mode='lines',
                    name='Линия тренда',
                    line=dict(color='red', width=2)
                ))
                fig_trend.update_layout(
                    title="Тренд температуры",
                    xaxis_title="Дата",
                    yaxis_title="Температура (°C)",
                    height=300
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # Временной ряд
        st.subheader("📉 Временной ряд температуры")
        fig_ts = create_temperature_time_series(city_data, selected_city)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Сезонный профиль
        st.subheader("🌸 Сезонный профиль")
        fig_seasonal = create_seasonal_profile(filtered_df, selected_city)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Мониторинг текущей температуры
        if api_key:
            st.subheader("🌍 Мониторинг температуры в реальном времени")
            
            # Текущая температура
            weather_api = WeatherAPI(api_key)
            
            with st.spinner(f"Получение текущей температуры для {selected_city}..."):
                temp, error = weather_api.get_current_temperature_sync(selected_city)
            
            if error:
                st.error(f"Ошибка получения температуры: {error}")
                if "401" in error or "Invalid API key" in error:
                    st.warning("Проверьте API ключ OpenWeatherMap")
            else:
                # Текущий сезон
                current_month = datetime.now().month
                month_to_season_ru = {
                    12: "winter", 1: "winter", 2: "winter",
                    3: "spring", 4: "spring", 5: "spring",
                    6: "summer", 7: "summer", 8: "summer",
                    9: "autumn", 10: "autumn", 11: "autumn"
                }
                current_season = month_to_season_ru[current_month]
                
                # Нормальность температуры
                seasonal_stats = calculate_seasonal_stats(filtered_df, selected_city)
                is_normal, message = check_temperature_normal(temp, seasonal_stats, current_season)
                
                # Вывод результатов
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Текущая температура", f"{temp:.1f}°C")
                
                with col2:
                    season_ru = {
                        'winter': 'Зима',
                        'spring': 'Весна',
                        'summer': 'Лето',
                        'autumn': 'Осень'
                    }
                    st.metric("Текущий сезон", season_ru[current_season])
                
                with col3:
                    if is_normal:
                        st.markdown('<div class="normal-badge">✅ НОРМАЛЬНО</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-badge">⚠️ АНОМАЛИЯ</div>', unsafe_allow_html=True)
                
                st.info(message)
                
                # Сравнение с историческими данными
                historical_mean = seasonal_stats[current_season]['mean']
                historical_std = seasonal_stats[current_season]['std']
                
                # График-индикатор
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=temp,
                    title={'text': "Текущая vs Историческая температура"},
                    gauge={
                        'axis': {'range': [historical_mean - 3*historical_std, historical_mean + 3*historical_std]},
                        'bar': {'color': "#ff6b6b" if not is_normal else "#51cf66"},
                        'steps': [
                            {'range': [historical_mean - 2*historical_std, historical_mean + 2*historical_std], 'color': "lightgray"},
                            {'range': [historical_mean - historical_std, historical_mean + historical_std], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': historical_mean
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        else:
            st.info("🔑 Введите API ключ OpenWeatherMap в боковой панели для мониторинга текущей погоды")
        
        # Дополнительная статистика
        with st.expander("📋 Детальная статистика"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Распределение температуры**")
                fig_hist = px.histogram(
                    city_data,
                    x='temperature',
                    nbins=50,
                    title="Гистограмма распределения температуры",
                    labels={'temperature': 'Температура (°C)'},
                    color_discrete_sequence=['#3498db']
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.write("**Сезонная статистика**")
                seasons_ru = {'winter': 'Зима', 'spring': 'Весна', 'summer': 'Лето', 'autumn': 'Осень'}
                seasonal_stats = calculate_seasonal_stats(filtered_df, selected_city)
                seasonal_df = pd.DataFrame(seasonal_stats).T
                seasonal_df.index = [seasons_ru[idx] for idx in seasonal_df.index]
                seasonal_df.columns = ['Средняя', 'Стд. отклонение', 'Мин.', 'Макс.', 'Количество']
                st.dataframe(seasonal_df.style.format("{:.2f}"))
        
        # Footer
        st.markdown("---")
        st.markdown(
            "📊 **Система мониторинга климата** | Анализ данных с обнаружением аномалий | Мониторинг погоды в реальном времени"
        )

if __name__ == "__main__":
    main()