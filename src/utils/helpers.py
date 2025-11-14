import os
import sys
import json
import logging
import time
import functools
import hashlib
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import warnings


# ============================ 日志管理 ============================

def setup_logger(name: str, 
                 log_file: Optional[str] = None, 
                 level: int = logging.INFO,
                 console_level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志名称
        log_file: 日志文件路径（可选）
        level: 文件日志级别
        console_level: 控制台日志级别
        
    Returns:
        配置好的日志记录器
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置最低级别
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建文件handler
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# 项目默认日志
DEFAULT_LOGGER = setup_logger(
    name="m_timeseries_model",
    log_file=os.path.join("logs", "app.log"),
    level=logging.INFO,
    console_level=logging.INFO
)


# ============================ 缓存装饰器 ============================

def _get_cache_key(func: Callable, *args: Any, **kwargs: Any) -> str:
    """
    生成缓存键
    
    Args:
        func: 函数对象
        args: 位置参数
        kwargs: 关键字参数
        
    Returns:
        缓存键字符串
    """
    # 序列化函数和参数
    key_parts = [
        func.__module__,
        func.__name__
    ]
    
    # 序列化参数
    for arg in args:
        try:
            if isinstance(arg, pd.DataFrame):
                # 对DataFrame使用其哈希值
                key_parts.append(hashlib.md5(pickle.dumps(arg)).hexdigest())
            elif hasattr(arg, '__repr__'):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(id(arg)))
        except Exception:
            key_parts.append(str(id(arg)))
    
    # 序列化关键字参数（排序以确保一致性）
    for k in sorted(kwargs.keys()):
        try:
            key_parts.append(f"{k}:{kwargs[k]}")
        except Exception:
            key_parts.append(f"{k}:{id(kwargs[k])}")
    
    # 生成MD5哈希值
    return hashlib.md5(''.join(key_parts).encode()).hexdigest()


def cache_result(cache_dir: str = ".cache", 
                 max_age: int = 3600,  # 缓存有效期（秒）
                 use_memory: bool = True):
    """
    结果缓存装饰器
    
    Args:
        cache_dir: 缓存文件存储目录
        max_age: 缓存有效期（秒）
        use_memory: 是否使用内存缓存
        
    Returns:
        装饰后的函数
    """
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    # 内存缓存字典
    memory_cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 生成缓存键
            cache_key = _get_cache_key(func, *args, **kwargs)
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # 检查内存缓存
            if use_memory:
                if cache_key in memory_cache:
                    cached_data, timestamp = memory_cache[cache_key]
                    if time.time() - timestamp < max_age:
                        DEFAULT_LOGGER.debug(f"从内存缓存返回 {func.__name__}")
                        return cached_data
            
            # 检查文件缓存
            if os.path.exists(cache_file):
                # 检查缓存是否过期
                cache_mtime = os.path.getmtime(cache_file)
                if time.time() - cache_mtime < max_age:
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        # 更新内存缓存
                        if use_memory:
                            memory_cache[cache_key] = (data, time.time())
                        
                        DEFAULT_LOGGER.debug(f"从文件缓存返回 {func.__name__}")
                        return data
                    except Exception as e:
                        DEFAULT_LOGGER.warning(f"读取缓存失败: {e}")
            
            # 执行函数
            DEFAULT_LOGGER.debug(f"执行函数 {func.__name__}")
            result = func(*args, **kwargs)
            
            # 保存到文件缓存
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                DEFAULT_LOGGER.warning(f"保存缓存失败: {e}")
            
            # 保存到内存缓存
            if use_memory:
                memory_cache[cache_key] = (result, time.time())
                
                # 限制内存缓存大小（最多100个条目）
                if len(memory_cache) > 100:
                    oldest_key = next(iter(memory_cache.keys()))
                    del memory_cache[oldest_key]
            
            return result
        
        # 添加清除缓存的方法
        def clear_cache():
            nonlocal memory_cache
            memory_cache = {}
            
            # 清除相关的文件缓存
            for filename in os.listdir(cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(cache_dir, filename))
            
            DEFAULT_LOGGER.info(f"已清除 {func.__name__} 的缓存")
        
        wrapper.clear_cache = clear_cache
        return wrapper
    
    return decorator


# ============================ 性能监控 ============================

def timer(func: Callable) -> Callable:
    """
    计时装饰器
    
    Args:
        func: 需要计时的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        DEFAULT_LOGGER.info(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        
        return result
    
    return wrapper


class PerformanceTracker:
    """
    性能跟踪器上下文管理器
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        """
        初始化性能跟踪器
        
        Args:
            name: 跟踪名称
            log_level: 日志级别
        """
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self) -> 'PerformanceTracker':
        self.start_time = time.time()
        if self.log_level <= logging.INFO:
            DEFAULT_LOGGER.info(f"开始执行 {self.name}")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        execution_time = time.time() - self.start_time
        message = f"{self.name} 执行完成，耗时: {execution_time:.4f} 秒"
        
        if exc_type is not None:
            DEFAULT_LOGGER.error(f"{self.name} 执行出错: {exc_val}")
        else:
            if self.log_level <= logging.INFO:
                DEFAULT_LOGGER.info(message)
        
        return False  # 不抑制异常


# ============================ 数据处理辅助函数 ============================

def ensure_datetime_index(df: pd.DataFrame, 
                          date_column: Optional[str] = None) -> pd.DataFrame:
    """
    确保DataFrame有DatetimeIndex
    
    Args:
        df: 输入DataFrame
        date_column: 日期列名称（如果索引不是日期）
        
    Returns:
        带有DatetimeIndex的DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_column is None:
            # 尝试自动检测日期列
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_column = col
                    break
            
            # 如果没有找到，尝试将索引转换为日期
            if date_column is None:
                try:
                    df.index = pd.to_datetime(df.index)
                    return df
                except Exception:
                    raise ValueError("无法自动检测日期列，请明确指定date_column")
        
        # 将指定列设置为索引
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    
    return df


def resample_time_series(df: pd.DataFrame, 
                         freq: str = 'D',
                         method: str = 'ffill') -> pd.DataFrame:
    """
    重采样时间序列数据
    
    Args:
        df: 输入DataFrame（必须有DatetimeIndex）
        freq: 重采样频率
        method: 填充方法 ('ffill', 'bfill', 'mean', 'sum', 'last', 'first')
        
    Returns:
        重采样后的DataFrame
    """
    # 确保索引是日期时间类型
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame必须有DatetimeIndex")
    
    # 重采样方法映射
    resample_methods = {
        'ffill': df.resample(freq).ffill(),
        'bfill': df.resample(freq).bfill(),
        'mean': df.resample(freq).mean(),
        'sum': df.resample(freq).sum(),
        'last': df.resample(freq).last(),
        'first': df.resample(freq).first()
    }
    
    if method not in resample_methods:
        raise ValueError(f"不支持的重采样方法: {method}")
    
    return resample_methods[method]


def normalize_data(df: pd.DataFrame, 
                   columns: Optional[List[str]] = None,
                   method: str = 'zscore') -> pd.DataFrame:
    """
    标准化数据
    
    Args:
        df: 输入DataFrame
        columns: 需要标准化的列（默认所有数值列）
        method: 标准化方法 ('zscore', 'minmax', 'robust')
        
    Returns:
        标准化后的DataFrame
    """
    # 复制DataFrame以避免修改原数据
    normalized_df = df.copy()
    
    # 如果没有指定列，选择所有数值列
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 应用标准化
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            if method == 'zscore':
                # Z-Score标准化
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            
            elif method == 'minmax':
                # Min-Max标准化
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            
            elif method == 'robust':
                # 稳健标准化（使用中位数和四分位距）
                median_val = normalized_df[col].median()
                iqr = normalized_df[col].quantile(0.75) - normalized_df[col].quantile(0.25)
                if iqr != 0:
                    normalized_df[col] = (normalized_df[col] - median_val) / iqr
            
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
    
    return normalized_df


def fill_missing_values(df: pd.DataFrame, 
                        columns: Optional[List[str]] = None,
                        method: str = 'ffill',
                        limit: Optional[int] = None) -> pd.DataFrame:
    """
    填充缺失值
    
    Args:
        df: 输入DataFrame
        columns: 需要填充的列（默认所有列）
        method: 填充方法 ('ffill', 'bfill', 'mean', 'median', 'zero')
        limit: 填充限制（仅适用于ffill和bfill）
        
    Returns:
        填充后的DataFrame
    """
    # 复制DataFrame以避免修改原数据
    filled_df = df.copy()
    
    # 如果没有指定列，使用所有列
    if columns is None:
        columns = filled_df.columns.tolist()
    
    # 应用填充
    for col in columns:
        if col in filled_df.columns:
            if method == 'ffill':
                filled_df[col] = filled_df[col].ffill(limit=limit)
            elif method == 'bfill':
                filled_df[col] = filled_df[col].bfill(limit=limit)
            elif method == 'mean' and pd.api.types.is_numeric_dtype(filled_df[col]):
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
            elif method == 'median' and pd.api.types.is_numeric_dtype(filled_df[col]):
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
            elif method == 'zero' and pd.api.types.is_numeric_dtype(filled_df[col]):
                filled_df[col] = filled_df[col].fillna(0)
            else:
                raise ValueError(f"不支持的填充方法: {method}，或列类型不匹配")
    
    return filled_df


def remove_outliers(df: pd.DataFrame, 
                    columns: Optional[List[str]] = None,
                    method: str = 'zscore',
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    移除异常值
    
    Args:
        df: 输入DataFrame
        columns: 需要处理的列（默认所有数值列）
        method: 检测方法 ('zscore', 'iqr')
        threshold: 阈值
        
    Returns:
        移除异常值后的DataFrame
    """
    # 复制DataFrame
    cleaned_df = df.copy()
    
    # 如果没有指定列，选择所有数值列
    if columns is None:
        columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 生成掩码
    mask = pd.Series([True] * len(cleaned_df), index=cleaned_df.index)
    
    # 检测异常值
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            if method == 'zscore':
                # 使用Z-Score方法
                z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                mask = mask & (z_scores <= threshold)
            
            elif method == 'iqr':
                # 使用IQR方法
                q1 = cleaned_df[col].quantile(0.25)
                q3 = cleaned_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                mask = mask & (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
            
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
    
    # 应用掩码
    result = cleaned_df[mask]
    
    # 记录移除的行数
    removed_rows = len(cleaned_df) - len(result)
    if removed_rows > 0:
        DEFAULT_LOGGER.info(f"移除了 {removed_rows} 行异常值 ({removed_rows/len(cleaned_df)*100:.2f}%)")
    
    return result


# ============================ 文件操作辅助函数 ============================

def ensure_directory(path: str):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_project_root() -> str:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    # 从当前文件路径推断项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设项目结构为: project_root/src/utils/helpers.py
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return project_root


def load_config(config_file: str = "config/config.json") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    # 如果是相对路径，相对于项目根目录
    if not os.path.isabs(config_file):
        config_file = os.path.join(get_project_root(), config_file)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    """
    # 确保目录存在
    ensure_directory(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(file_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        扩展名（不含点号）
    """
    return os.path.splitext(file_path)[1][1:].lower()


def find_files(directory: str, 
               pattern: Optional[str] = None,
               recursive: bool = True) -> List[str]:
    """
    查找目录中的文件
    
    Args:
        directory: 要搜索的目录
        pattern: 文件名模式（正则表达式）
        recursive: 是否递归搜索子目录
        
    Returns:
        文件路径列表
    """
    results = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if pattern is None or re.match(pattern, file):
                    results.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if pattern is None or re.match(pattern, file):
                    results.append(file_path)
    
    return results


# ============================ 可视化辅助函数 ============================

def plot_time_series(df: pd.DataFrame, 
                     columns: Optional[List[str]] = None,
                     title: str = "时间序列图",
                     xlabel: str = "时间",
                     ylabel: str = "值",
                     figsize: Tuple[int, int] = (12, 6),
                     save_path: Optional[str] = None,
                     show: bool = True):
    """
    绘制时间序列图
    
    Args:
        df: 输入DataFrame（必须有DatetimeIndex）
        columns: 需要绘制的列
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        figsize: 图表大小
        save_path: 保存图片的路径（可选）
        show: 是否显示图表
    """
    # 确保索引是日期时间类型
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame必须有DatetimeIndex")
    
    # 选择要绘制的列
    if columns is None:
        columns = df.columns
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制每一列
    for col in columns:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        DEFAULT_LOGGER.info(f"图表已保存至: {save_path}")
    
    # 显示图表
    if show:
        plt.show()


def plot_heatmap(df: pd.DataFrame, 
                title: str = "相关性热力图",
                figsize: Tuple[int, int] = (10, 8),
                cmap: str = "coolwarm",
                save_path: Optional[str] = None,
                show: bool = True):
    """
    绘制热力图
    
    Args:
        df: 输入DataFrame
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
        save_path: 保存图片的路径（可选）
        show: 是否显示图表
    """
    import seaborn as sns
    
    # 计算相关系数矩阵
    corr = df.corr()
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(corr, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
    
    # 设置图表属性
    plt.title(title)
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        DEFAULT_LOGGER.info(f"热力图已保存至: {save_path}")
    
    # 显示图表
    if show:
        plt.show()


# ============================ 统计分析辅助函数 ============================

def calculate_drawdown(returns: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    计算回撤
    
    Args:
        returns: 收益率序列
        
    Returns:
        回撤序列
    """
    # 计算累计收益
    cumulative_returns = (1 + returns).cumprod()
    
    # 计算累计最大值
    running_max = cumulative_returns.cummax()
    
    # 计算回撤
    drawdown = (cumulative_returns / running_max) - 1
    
    return drawdown


def calculate_sharpe_ratio(returns: Union[pd.Series, pd.DataFrame], 
                          risk_free_rate: float = 0.0,
                          annualization_factor: float = 252) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        annualization_factor: 年化因子（日度数据通常为252）
        
    Returns:
        夏普比率
    """
    # 计算超额收益
    excess_returns = returns - (risk_free_rate / annualization_factor)
    
    # 计算夏普比率
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(annualization_factor)
    
    return sharpe_ratio


def calculate_sortino_ratio(returns: Union[pd.Series, pd.DataFrame], 
                           risk_free_rate: float = 0.0,
                           annualization_factor: float = 252,
                           target_return: float = 0.0) -> float:
    """
    计算索提诺比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        annualization_factor: 年化因子
        target_return: 目标收益率（日度）
        
    Returns:
        索提诺比率
    """
    # 计算超额收益
    excess_returns = returns - (risk_free_rate / annualization_factor)
    
    # 计算下行风险（仅考虑负收益）
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt((downside_returns ** 2).mean())
    
    # 计算索提诺比率
    sortino_ratio = (excess_returns.mean() - target_return) / downside_deviation * np.sqrt(annualization_factor)
    
    return sortino_ratio


def calculate_max_drawdown(returns: Union[pd.Series, pd.DataFrame]) -> float:
    """
    计算最大回撤
    
    Args:
        returns: 收益率序列
        
    Returns:
        最大回撤
    """
    drawdown = calculate_drawdown(returns)
    return drawdown.min()


def calculate_calmar_ratio(returns: Union[pd.Series, pd.DataFrame], 
                          risk_free_rate: float = 0.0,
                          annualization_factor: float = 252) -> float:
    """
    计算卡尔马比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        annualization_factor: 年化因子
        
    Returns:
        卡尔马比率
    """
    # 计算年化收益率
    annual_return = returns.mean() * annualization_factor
    
    # 计算最大回撤
    max_drawdown = abs(calculate_max_drawdown(returns))
    
    # 计算卡尔马比率
    if max_drawdown > 0:
        calmar_ratio = (annual_return - risk_free_rate) / max_drawdown
    else:
        calmar_ratio = float('inf')
    
    return calmar_ratio


# ============================ 其他辅助函数 ============================

def suppress_warnings():
    """
    抑制所有警告
    """
    warnings.filterwarnings('ignore')


def validate_parameters(params: Dict[str, Any], 
                        required_params: List[str],
                        param_types: Optional[Dict[str, Any]] = None) -> bool:
    """
    验证参数
    
    Args:
        params: 参数字典
        required_params: 必需参数列表
        param_types: 参数类型字典
        
    Returns:
        是否验证通过
    """
    # 检查必需参数
    for param in required_params:
        if param not in params:
            raise ValueError(f"缺少必需参数: {param}")
    
    # 检查参数类型
    if param_types:
        for param, expected_type in param_types.items():
            if param in params and not isinstance(params[param], expected_type):
                raise TypeError(f"参数 {param} 应该是 {expected_type.__name__} 类型，得到 {type(params[param]).__name__}")
    
    return True


def format_number(value: float, 
                  decimals: int = 2,
                  thousands_separator: str = ',',
                  decimal_separator: str = '.') -> str:
    """
    格式化数字
    
    Args:
        value: 要格式化的数值
        decimals: 小数位数
        thousands_separator: 千位分隔符
        decimal_separator: 小数分隔符
        
    Returns:
        格式化后的字符串
    """
    # 格式化为指定位数的小数
    formatted = f"{value:,.{decimals}f}"
    
    # 替换分隔符
    if thousands_separator != ',':
        formatted = formatted.replace(',', thousands_separator)
    
    if decimal_separator != '.':
        formatted = formatted.replace('.', decimal_separator)
    
    return formatted


def format_percentage(value: float, 
                      decimals: int = 2,
                      include_symbol: bool = True) -> str:
    """
    格式化百分比
    
    Args:
        value: 要格式化的数值
        decimals: 小数位数
        include_symbol: 是否包含百分号
        
    Returns:
        格式化后的百分比字符串
    """
    # 转换为百分比并格式化
    formatted = f"{value * 100:.{decimals}f}"
    
    # 添加百分号
    if include_symbol:
        formatted += '%'
    
    return formatted
