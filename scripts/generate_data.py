#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据生成脚本
用于生成时间序列测试数据
"""

import os
import sys
import json
import yaml
import time
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 自定义模块
from src.utils.helpers import setup_logger, DEFAULT_LOGGER
from src.config.config_manager import ConfigManager

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        参数命名空间
    """
    parser = argparse.ArgumentParser(description='时间序列数据生成脚本')
    
    # 基本设置
    parser.add_argument('--output', '-o',
                      type=str,
                      default='./data/generated_data.csv',
                      help='输出文件路径')
    
    parser.add_argument('--rows', '-n',
                      type=int,
                      default=1000,
                      help='生成数据行数')
    
    parser.add_argument('--start-date',
                      type=str,
                      default='2020-01-01',
                      help='开始日期 (YYYY-MM-DD格式)')
    
    parser.add_argument('--freq',
                      type=str,
                      default='D',
                      choices=['D', 'B', 'H', 'T', 'S'],
                      help='时间频率: D(日), B(工作日), H(小时), T(分钟), S(秒)')
    
    # 数据模式
    parser.add_argument('--pattern', '-p',
                      choices=['trend', 'seasonal', 'cyclical', 'random', 'all'],
                      default='all',
                      help='数据模式')
    
    parser.add_argument('--target-type',
                      choices=['regression', 'classification'],
                      default='regression',
                      help='目标变量类型')
    
    # 特征设置
    parser.add_argument('--numeric-features',
                      type=int,
                      default=10,
                      help='数值特征数量')
    
    parser.add_argument('--categorical-features',
                      type=int,
                      default=5,
                      help='分类特征数量')
    
    parser.add_argument('--datetime-features',
                      type=bool,
                      default=True,
                      help='是否生成时间特征')
    
    # 噪声和异常
    parser.add_argument('--noise-level',
                      type=float,
                      default=0.1,
                      help='噪声水平 (0-1之间)')
    
    parser.add_argument('--anomaly-rate',
                      type=float,
                      default=0.01,
                      help='异常值比例')
    
    # 其他选项
    parser.add_argument('--config', '-c',
                      type=str,
                      help='配置文件路径')
    
    parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help='随机种子')
    
    parser.add_argument('--format',
                      choices=['csv', 'json', 'parquet'],
                      default='csv',
                      help='输出文件格式')
    
    parser.add_argument('--compress',
                      action='store_true',
                      help='是否压缩输出文件')
    
    parser.add_argument('--verbose',
                      action='store_true',
                      help='详细输出')
    
    return parser.parse_args()


def generate_base_time_series(n: int, 
                              start_date: str, 
                              freq: str = 'D') -> pd.DatetimeIndex:
    """
    生成基础时间序列索引
    
    Args:
        n: 数据点数
        start_date: 开始日期
        freq: 时间频率
        
    Returns:
        时间序列索引
    """
    start = pd.to_datetime(start_date)
    return pd.date_range(start=start, periods=n, freq=freq)


def generate_trend(n: int, 
                  slope: float = 0.1, 
                  intercept: float = 0.0) -> np.ndarray:
    """
    生成趋势成分
    
    Args:
        n: 数据点数
        slope: 趋势斜率
        intercept: 截距
        
    Returns:
        趋势数组
    """
    return np.arange(n) * slope + intercept


def generate_seasonal(n: int, 
                     period: int = 7, 
                     amplitude: float = 1.0, 
                     phase: float = 0.0) -> np.ndarray:
    """
    生成季节性成分
    
    Args:
        n: 数据点数
        period: 季节周期
        amplitude: 振幅
        phase: 相位
        
    Returns:
        季节性数组
    """
    t = np.arange(n)
    return amplitude * np.sin(2 * np.pi * t / period + phase)


def generate_cyclical(n: int, 
                     periods: List[Tuple[float, float]]) -> np.ndarray:
    """
    生成周期性成分（多个周期的叠加）
    
    Args:
        n: 数据点数
        periods: [(周期长度, 振幅)] 的列表
        
    Returns:
        周期性数组
    """
    result = np.zeros(n)
    t = np.arange(n)
    
    for period, amplitude in periods:
        result += amplitude * np.sin(2 * np.pi * t / period)
    
    return result


def generate_noise(n: int, 
                  level: float = 0.1) -> np.ndarray:
    """
    生成噪声成分
    
    Args:
        n: 数据点数
        level: 噪声水平
        
    Returns:
        噪声数组
    """
    return np.random.normal(0, level, n)


def generate_random_walk(n: int, 
                        step_size: float = 0.1, 
                        start: float = 0.0) -> np.ndarray:
    """
    生成随机游走成分
    
    Args:
        n: 数据点数
        step_size: 步长
        start: 起始值
        
    Returns:
        随机游走数组
    """
    steps = np.random.normal(0, step_size, n)
    return start + np.cumsum(steps)


def add_anomalies(data: np.ndarray, 
                  rate: float = 0.01, 
                  scale: float = 3.0) -> np.ndarray:
    """
    添加异常值
    
    Args:
        data: 原始数据
        rate: 异常值比例
        scale: 异常值幅度倍数
        
    Returns:
        添加异常值后的数据
    """
    n = len(data)
    num_anomalies = int(n * rate)
    
    # 随机选择异常值位置
    anomaly_indices = np.random.choice(n, num_anomalies, replace=False)
    
    # 计算数据标准差
    std_dev = np.std(data)
    
    # 添加异常值（大幅正向或负向偏离）
    result = data.copy()
    directions = np.random.choice([-1, 1], num_anomalies)
    
    for i, idx in enumerate(anomaly_indices):
        # 异常值大小为标准差的倍数
        anomaly_size = scale * std_dev * directions[i]
        result[idx] += anomaly_size
    
    return result


def generate_numeric_features(n: int, 
                             num_features: int, 
                             target: Optional[np.ndarray] = None,
                             correlation_strength: float = 0.5) -> np.ndarray:
    """
    生成数值特征
    
    Args:
        n: 数据点数
        num_features: 特征数量
        target: 目标变量（用于创建相关特征）
        correlation_strength: 与目标变量的相关强度
        
    Returns:
        数值特征矩阵 (n x num_features)
    """
    features = np.random.randn(n, num_features)
    
    # 如果提供了目标变量，创建一些相关特征
    if target is not None:
        # 选择一部分特征与目标变量相关
        num_correlated = max(1, num_features // 3)
        
        for i in range(num_correlated):
            # 添加与目标变量的相关性
            features[:, i] = correlation_strength * target + np.sqrt(1 - correlation_strength**2) * features[:, i]
    
    return features


def generate_categorical_features(n: int, 
                                 num_features: int,
                                 categories_per_feature: int = 5) -> np.ndarray:
    """
    生成分类特征
    
    Args:
        n: 数据点数
        num_features: 特征数量
        categories_per_feature: 每个特征的类别数
        
    Returns:
        分类特征矩阵 (n x num_features)
    """
    features = np.random.randint(0, categories_per_feature, size=(n, num_features))
    return features


def generate_datetime_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    从时间索引生成时间特征
    
    Args:
        index: 时间索引
        
    Returns:
        时间特征DataFrame
    """
    df = pd.DataFrame(index=index)
    
    # 提取各种时间特征
    df['year'] = index.year
    df['month'] = index.month
    df['day'] = index.day
    df['dayofyear'] = index.dayofyear
    df['dayofweek'] = index.dayofweek
    df['weekofyear'] = index.isocalendar().week
    df['quarter'] = index.quarter
    df['is_month_start'] = index.is_month_start.astype(int)
    df['is_month_end'] = index.is_month_end.astype(int)
    df['is_quarter_start'] = index.is_quarter_start.astype(int)
    df['is_quarter_end'] = index.is_quarter_end.astype(int)
    df['is_year_start'] = index.is_year_start.astype(int)
    df['is_year_end'] = index.is_year_end.astype(int)
    
    # 添加节假日特征（简化版本）
    # 这里只是示例，实际应用中可能需要更复杂的节假日逻辑
    df['is_weekend'] = (index.dayofweek >= 5).astype(int)
    
    # 如果是小时频率，添加小时相关特征
    if index.freq in ['H', 'T', 'S']:
        df['hour'] = index.hour
        df['minute'] = index.minute
        df['is_business_hours'] = ((index.hour >= 9) & (index.hour < 17)).astype(int)
    
    return df


def generate_target(data: np.ndarray, 
                    target_type: str = 'regression',
                    num_classes: int = 2) -> np.ndarray:
    """
    生成目标变量
    
    Args:
        data: 基础数据
        target_type: 目标类型 ('regression' 或 'classification')
        num_classes: 分类问题的类别数
        
    Returns:
        目标变量数组
    """
    if target_type == 'regression':
        return data
    elif target_type == 'classification':
        # 基于数据分位数创建类别
        quantiles = np.linspace(0, 1, num_classes + 1)[1:-1]
        thresholds = np.quantile(data, quantiles)
        
        # 将数据分到不同类别
        classes = np.zeros_like(data, dtype=int)
        for i, threshold in enumerate(thresholds):
            classes[data > threshold] = i + 1
        
        return classes
    else:
        raise ValueError(f"不支持的目标类型: {target_type}")


def create_time_series_pattern(n: int, 
                               pattern: str,
                               noise_level: float = 0.1,
                               anomaly_rate: float = 0.01) -> np.ndarray:
    """
    根据指定模式创建时间序列
    
    Args:
        n: 数据点数
        pattern: 数据模式
        noise_level: 噪声水平
        anomaly_rate: 异常值比例
        
    Returns:
        时间序列数组
    """
    result = np.zeros(n)
    
    if pattern in ['trend', 'all']:
        # 添加趋势（线性增长）
        result += generate_trend(n, slope=0.02)
    
    if pattern in ['seasonal', 'all']:
        # 添加季节性（年、月、周周期）
        # 假设n对应的时间跨度足够大，例如以天为单位，1000天约等于3年
        # 年度周期
        yearly_period = 365
        if n > yearly_period:
            result += generate_seasonal(n, period=yearly_period, amplitude=2.0)
        
        # 月度周期
        monthly_period = 30
        if n > monthly_period:
            result += generate_seasonal(n, period=monthly_period, amplitude=1.0)
        
        # 周度周期
        weekly_period = 7
        result += generate_seasonal(n, period=weekly_period, amplitude=0.5)
    
    if pattern in ['cyclical', 'all']:
        # 添加长周期波动（多个周期的叠加）
        # 这里使用一些任意的周期和振幅
        periods = [(50, 1.0), (100, 0.5), (200, 0.3)]
        result += generate_cyclical(n, periods)
    
    if pattern == 'random':
        # 纯随机数据
        result = generate_random_walk(n, step_size=0.1)
    
    # 添加噪声
    result += generate_noise(n, level=noise_level)
    
    # 添加异常值
    if anomaly_rate > 0:
        result = add_anomalies(result, rate=anomaly_rate)
    
    return result


def generate_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """
    生成完整的数据集
    
    Args:
        args: 命令行参数
        
    Returns:
        生成的数据集
    """
    logger.info(f"生成 {args.rows} 行数据，模式: {args.pattern}")
    
    # 生成时间索引
    index = generate_base_time_series(args.rows, args.start_date, args.freq)
    
    # 生成基础时间序列
    base_series = create_time_series_pattern(
        args.rows,
        args.pattern,
        args.noise_level,
        args.anomaly_rate
    )
    
    # 生成目标变量
    target = generate_target(base_series, args.target_type)
    
    # 生成特征
    # 数值特征
    numeric_features = generate_numeric_features(
        args.rows,
        args.numeric_features,
        target=target,
        correlation_strength=0.5
    )
    
    # 分类特征
    categorical_features = generate_categorical_features(
        args.rows,
        args.categorical_features
    )
    
    # 创建数据集
    df = pd.DataFrame(index=index)
    
    # 添加日期列
    df['date'] = index
    
    # 添加目标列
    df['target'] = target
    
    # 添加数值特征
    for i in range(args.numeric_features):
        df[f'numeric_feature_{i}'] = numeric_features[:, i]
    
    # 添加分类特征
    for i in range(args.categorical_features):
        df[f'cat_feature_{i}'] = categorical_features[:, i]
    
    # 添加时间特征（如果启用）
    if args.datetime_features:
        time_features = generate_datetime_features(index)
        df = pd.concat([df, time_features], axis=1)
    
    # 随机打乱特征列的顺序（除了日期和目标列）
    non_key_columns = [col for col in df.columns if col not in ['date', 'target']]
    np.random.shuffle(non_key_columns)
    
    # 重新排列列，使日期和目标列在前
    df = df[['date', 'target'] + non_key_columns]
    
    logger.info(f"数据集生成完成，形状: {df.shape}")
    return df


def save_dataset(df: pd.DataFrame, 
                 output_path: str, 
                 format: str = 'csv',
                 compress: bool = False) -> None:
    """
    保存数据集到文件
    
    Args:
        df: 数据集
        output_path: 输出路径
        format: 文件格式
        compress: 是否压缩
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"保存数据到: {output_path}")
    
    # 根据格式保存
    if format == 'csv':
        if compress:
            # 使用gzip压缩
            if not output_path.endswith('.gz'):
                output_path += '.gz'
            df.to_csv(output_path, index=False, compression='gzip')
        else:
            df.to_csv(output_path, index=False)
    
    elif format == 'json':
        if compress:
            # 使用gzip压缩
            import gzip
            if not output_path.endswith('.gz'):
                output_path += '.gz'
            with gzip.open(output_path, 'wt') as f:
                df.to_json(f, orient='records', date_format='iso')
        else:
            df.to_json(output_path, orient='records', date_format='iso')
    
    elif format == 'parquet':
        if compress:
            # 默认可行压缩
            df.to_parquet(output_path, compression='gzip')
        else:
            df.to_parquet(output_path)
    
    else:
        raise ValueError(f"不支持的文件格式: {format}")
    
    logger.info(f"数据保存完成")


def generate_multivariate_dataset(args: argparse.Namespace,
                                 num_series: int = 3,
                                 correlation_matrix: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    生成多变量时间序列数据集
    
    Args:
        args: 命令行参数
        num_series: 时间序列数量
        correlation_matrix: 相关矩阵
        
    Returns:
        多变量数据集
    """
    logger.info(f"生成包含 {num_series} 个相关时间序列的多变量数据集")
    
    # 生成时间索引
    index = generate_base_time_series(args.rows, args.start_date, args.freq)
    
    # 如果没有提供相关矩阵，创建一个随机的
    if correlation_matrix is None:
        # 生成半正定矩阵
        random_matrix = np.random.randn(num_series, num_series)
        correlation_matrix = np.dot(random_matrix, random_matrix.T)
        # 归一化对角线为1
        diag = np.diag(correlation_matrix)
        correlation_matrix = correlation_matrix / np.sqrt(np.outer(diag, diag))
    
    # 生成基础时间序列
    base_series = np.random.multivariate_normal(
        mean=np.zeros(num_series),
        cov=correlation_matrix,
        size=args.rows
    )
    
    # 添加各种模式和噪声到每个序列
    for i in range(num_series):
        # 添加不同的季节性模式
        seasonal = generate_seasonal(
            args.rows,
            period=7 + i * 3,  # 每个序列有略微不同的周期
            amplitude=1.0 - i * 0.1
        )
        
        # 添加不同的趋势
        trend = generate_trend(args.rows, slope=0.01 * (i + 1))
        
        # 添加噪声
        noise = generate_noise(args.rows, level=args.noise_level)
        
        # 组合所有成分
        base_series[:, i] = base_series[:, i] + seasonal + trend + noise
    
    # 创建数据集
    df = pd.DataFrame(index=index)
    
    # 添加日期列
    df['date'] = index
    
    # 添加时间序列列
    for i in range(num_series):
        df[f'timeseries_{i}'] = base_series[:, i]
    
    # 将第一个时间序列设为目标变量
    df['target'] = df.pop('timeseries_0')
    
    # 生成其他特征（与单变量情况类似）
    # 数值特征
    numeric_features = generate_numeric_features(
        args.rows,
        args.numeric_features,
        target=df['target'].values,
        correlation_strength=0.5
    )
    
    # 分类特征
    categorical_features = generate_categorical_features(
        args.rows,
        args.categorical_features
    )
    
    # 添加数值特征
    for i in range(args.numeric_features):
        df[f'numeric_feature_{i}'] = numeric_features[:, i]
    
    # 添加分类特征
    for i in range(args.categorical_features):
        df[f'cat_feature_{i}'] = categorical_features[:, i]
    
    # 添加时间特征（如果启用）
    if args.datetime_features:
        time_features = generate_datetime_features(index)
        df = pd.concat([df, time_features], axis=1)
    
    logger.info(f"多变量数据集生成完成，形状: {df.shape}")
    return df


def generate_config_file(config_path: str) -> None:
    """
    生成默认配置文件
    
    Args:
        config_path: 配置文件路径
    """
    default_config = {
        'data_generation': {
            'base': {
                'rows': 1000,
                'start_date': '2020-01-01',
                'freq': 'D',
                'seed': 42
            },
            'patterns': {
                'trend': True,
                'seasonal': True,
                'cyclical': True,
                'random': False
            },
            'noise': {
                'level': 0.1,
                'anomaly_rate': 0.01
            },
            'features': {
                'numeric': 10,
                'categorical': 5,
                'datetime': True,
                'correlation_strength': 0.5
            },
            'target': {
                'type': 'regression',
                'num_classes': 2
            }
        },
        'multivariate': {
            'enabled': False,
            'num_series': 3
        },
        'output': {
            'format': 'csv',
            'compress': False
        }
    }
    
    # 确保目录存在
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logger.info(f"默认配置文件已保存到: {config_path}")


def generate_metrics_report(df: pd.DataFrame, output_path: str) -> None:
    """
    生成数据集统计报告
    
    Args:
        df: 数据集
        output_path: 输出路径
    """
    report = [
        "# 数据集统计报告",
        f"生成时间: {datetime.now().isoformat()}",
        f"数据集大小: {df.shape[0]} 行 × {df.shape[1]} 列",
        "\n## 数据类型分布",
    ]
    
    # 数据类型统计
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        report.append(f"- {dtype}: {count} 列")
    
    # 基本统计信息
    report.append("\n## 数值特征统计摘要")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().T
        report.append(stats.to_markdown())
    else:
        report.append("没有数值特征")
    
    # 缺失值统计
    report.append("\n## 缺失值统计")
    missing_stats = df.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0]
    
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            percent = (count / len(df)) * 100
            report.append(f"- {col}: {count} ({percent:.2f}%)")
    else:
        report.append("没有缺失值")
    
    # 保存报告
    report_dir = os.path.join(os.path.dirname(output_path), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, 'dataset_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"数据集统计报告已保存到: {report_file}")


def main() -> None:
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        import logging as lg
        lg.getLogger().setLevel(lg.DEBUG)
    
    try:
        # 设置随机种子
        np.random.seed(args.seed)
        
        # 如果提供了配置文件，加载它
        if args.config:
            if os.path.exists(args.config):
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # 更新参数
                data_config = config.get('data_generation', {})
                base_config = data_config.get('base', {})
                
                if 'rows' in base_config:
                    args.rows = base_config['rows']
                if 'start_date' in base_config:
                    args.start_date = base_config['start_date']
                if 'freq' in base_config:
                    args.freq = base_config['freq']
                if 'seed' in base_config:
                    args.seed = base_config['seed']
                    np.random.seed(args.seed)
                
                # 其他配置...
                logger.info(f"从配置文件加载设置: {args.config}")
            else:
                # 配置文件不存在，生成默认配置
                logger.warning(f"配置文件不存在，生成默认配置: {args.config}")
                generate_config_file(args.config)
        
        # 生成数据集
        if hasattr(args, 'multivariate') and args.multivariate:
            # 生成多变量数据集
            df = generate_multivariate_dataset(args, num_series=args.num_series)
        else:
            # 生成单变量数据集
            df = generate_dataset(args)
        
        # 保存数据集
        save_dataset(df, args.output, args.format, args.compress)
        
        # 生成统计报告
        generate_metrics_report(df, args.output)
        
        logger.info("数据生成流程完成")
        
        # 显示数据集预览
        print("\n数据集预览:")
        print(df.head())
        print(f"\n数据集形状: {df.shape}")
        
    except Exception as e:
        logger.error(f"数据生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()