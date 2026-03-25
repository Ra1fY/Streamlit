import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time
from typing import Dict, List, Tuple

def analyze_city_parallel(city: str, df: pd.DataFrame) -> Dict:
    """
    Анализирует данные для одного города
    
    Args:
        city: Название города
        df: DataFrame с данными
    
    Returns:
        Словарь с результатами анализа для города
    """
    city_data = df[df['city'] == city].copy()
    city_data = city_data.sort_values('timestamp')
    
    # Вычисляем скользящие статистики
    city_data['rolling_mean'] = city_data['temperature'].rolling(window=30, center=True).mean()
    city_data['rolling_std'] = city_data['temperature'].rolling(window=30, center=True).std()
    
    # Обнаруживаем аномалии
    sigma = 2.0
    city_data['is_anomaly'] = (
        (city_data['temperature'] > city_data['rolling_mean'] + sigma * city_data['rolling_std']) |
        (city_data['temperature'] < city_data['rolling_mean'] - sigma * city_data['rolling_std'])
    )
    
    # Вычисляем сезонную статистику
    seasonal_stats = {}
    for season in ['winter', 'spring', 'summer', 'autumn']:
        season_data = city_data[city_data['season'] == season]['temperature']
        seasonal_stats[season] = {
            'mean': season_data.mean(),
            'std': season_data.std(),
            'min': season_data.min(),
            'max': season_data.max()
        }
    
    return {
        'city': city,
        'anomaly_count': city_data['is_anomaly'].sum(),
        'anomaly_percentage': (city_data['is_anomaly'].sum() / len(city_data)) * 100,
        'mean_temp': city_data['temperature'].mean(),
        'seasonal_stats': seasonal_stats
    }

def analyze_all_cities_parallel(df: pd.DataFrame, n_jobs: int = -1) -> Tuple[List[Dict], float]:
    """
    Анализирует все города параллельно
    
    Комментарий о параллельной обработке:
    Параллельная обработка особенно эффективна, когда у нас много независимых задач,
    которые можно выполнять одновременно. В нашем случае анализ каждого города
    не зависит от других, что позволяет обрабатывать их параллельно.
    
    Использование joblib с n_jobs=-1 задействует все доступные ядра процессора,
    что значительно ускоряет обработку при большом количестве городов.
    
    Args:
        df: DataFrame с данными
        n_jobs: Количество параллельных процессов (-1 для всех ядер)
    
    Returns:
        Кортеж (список результатов, время выполнения)
    """
    cities = df['city'].unique()
    
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze_city_parallel)(city, df) for city in cities
    )
    execution_time = time.time() - start_time
    
    return results, execution_time

def analyze_all_cities_sequential(df: pd.DataFrame) -> Tuple[List[Dict], float]:
    """
    Анализирует все города последовательно
    
    Args:
        df: DataFrame с данными
    
    Returns:
        Кортеж (список результатов, время выполнения)
    """
    cities = df['city'].unique()
    results = []
    
    start_time = time.time()
    for city in cities:
        results.append(analyze_city_parallel(city, df))
    execution_time = time.time() - start_time
    
    return results, execution_time

def compare_parallel_vs_sequential(df: pd.DataFrame) -> Dict:
    """
    Сравнивает производительность параллельной и последовательной обработки
    
    Комментарий о сравнении:
    При анализе 15 городов с 10 годами данных (≈54,750 записей),
    параллельная обработка демонстрирует значительное ускорение.
    На многоядерных системах ускорение может достигать 3-8 раз,
    в зависимости от количества доступных ядер и накладных расходов на
    создание процессов.
    
    Args:
        df: DataFrame с данными
    
    Returns:
        Словарь с результатами сравнения
    """
    # Последовательная обработка
    seq_results, seq_time = analyze_all_cities_sequential(df)
    
    # Параллельная обработка
    par_results, par_time = analyze_all_cities_parallel(df)
    
    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': seq_time / par_time,
        'cities_analyzed': len(df['city'].unique()),
        'records_processed': len(df)
    }