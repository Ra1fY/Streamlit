import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

def calculate_rolling_stats(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Вычисляет скользящее среднее и стандартное отклонение
    
    Args:
        df: DataFrame с температурными данными
        window: Размер окна для скользящего среднего (дней)
    
    Returns:
        DataFrame с добавленными колонками rolling_mean и rolling_std
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values('timestamp')
    df_copy['rolling_mean'] = df_copy['temperature'].rolling(window=window, center=True).mean()
    df_copy['rolling_std'] = df_copy['temperature'].rolling(window=window, center=True).std()
    return df_copy

def detect_anomalies(df: pd.DataFrame, sigma: float = 2.0) -> pd.DataFrame:
    """
    Обнаруживает аномалии на основе скользящих статистик
    
    Args:
        df: DataFrame с колонками rolling_mean и rolling_std
        sigma: Количество стандартных отклонений для определения аномалий
    
    Returns:
        DataFrame с добавленной колонкой is_anomaly
    """
    df_copy = df.copy()
    df_copy['is_anomaly'] = (
        (df_copy['temperature'] > df_copy['rolling_mean'] + sigma * df_copy['rolling_std']) |
        (df_copy['temperature'] < df_copy['rolling_mean'] - sigma * df_copy['rolling_std'])
    )
    return df_copy

def calculate_seasonal_stats(df: pd.DataFrame, city: str) -> Dict:
    """
    Рассчитывает сезонную статистику для конкретного города
    
    Args:
        df: DataFrame с температурными данными
        city: Название города
    
    Returns:
        Словарь со статистикой по сезонам
    """
    city_data = df[df['city'] == city].copy()
    seasonal_stats = {}
    
    for season in ['winter', 'spring', 'summer', 'autumn']:
        season_data = city_data[city_data['season'] == season]['temperature']
        seasonal_stats[season] = {
            'mean': season_data.mean(),
            'std': season_data.std(),
            'min': season_data.min(),
            'max': season_data.max(),
            'count': len(season_data)
        }
    
    return seasonal_stats

def calculate_trend(df: pd.DataFrame, city: str) -> Tuple[float, float]:
    """
    Вычисляет долгосрочный тренд температуры с помощью линейной регрессии
    
    Args:
        df: DataFrame с температурными данными
        city: Название города
    
    Returns:
        Кортеж (наклон, пересечение) линии тренда
    """
    city_data = df[df['city'] == city].copy()
    city_data = city_data.sort_values('timestamp')
    
    # Конвертируем timestamp в числовые значения (дни с начала)
    days_since_start = (city_data['timestamp'] - city_data['timestamp'].min()).dt.days
    
    # Вычисляем линейную регрессию
    slope, intercept = np.polyfit(days_since_start, city_data['temperature'], 1)
    
    return slope, intercept

def get_city_summary_stats(df: pd.DataFrame, city: str) -> Dict:
    """
    Получает сводную статистику для города
    
    Args:
        df: DataFrame с температурными данными
        city: Название города
    
    Returns:
        Словарь со сводной статистикой
    """
    city_data = df[df['city'] == city]['temperature']
    
    return {
        'mean': city_data.mean(),
        'std': city_data.std(),
        'min': city_data.min(),
        'max': city_data.max(),
        'median': city_data.median(),
        'q1': city_data.quantile(0.25),
        'q3': city_data.quantile(0.75)
    }

def check_temperature_normal(current_temp: float, season_stats: Dict, season: str) -> Tuple[bool, str]:
    """
    Проверяет, является ли текущая температура нормальной для сезона
    
    Args:
        current_temp: Текущая температура
        season_stats: Статистика по сезонам
        season: Текущий сезон
    
    Returns:
        Кортеж (нормальна ли температура, сообщение)
    """
    if season not in season_stats:
        return False, f"Нет исторических данных для сезона {season}"
    
    mean = season_stats[season]['mean']
    std = season_stats[season]['std']
    
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    
    if lower_bound <= current_temp <= upper_bound:
        return True, f"Нормальная температура для {season} (нормальный диапазон: {lower_bound:.1f}°C до {upper_bound:.1f}°C)"
    elif current_temp < lower_bound:
        return False, f"Ниже нормы для {season} (нормальный диапазон: {lower_bound:.1f}°C до {upper_bound:.1f}°C)"
    else:
        return False, f"Выше нормы для {season} (нормальный диапазон: {lower_bound:.1f}°C до {upper_bound:.1f}°C)"