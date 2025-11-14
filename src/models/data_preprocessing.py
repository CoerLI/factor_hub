import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit
from src.utils.helpers import setup_logger, DEFAULT_LOGGER

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class DataPreprocessor:
    """
    数据预处理器
    
    负责时间序列数据的清洗、特征工程和预处理
    """
    
    def __init__(self):
        """
        初始化数据预处理器
        """
        self.scalers = {}
        self.feature_selector = None
        self.feature_names = None
        self.feature_importance = None
    
    def clean_data(self, 
                   data: pd.DataFrame,
                   drop_na: bool = True,
                   fill_na_method: Optional[str] = None,
                   fill_na_value: Optional[float] = None,
                   remove_duplicates: bool = True,
                   datetime_column: Optional[str] = 'timestamp') -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            data: 原始数据
            drop_na: 是否删除NaN值
            fill_na_method: 填充NaN的方法 ('ffill', 'bfill', 'mean', 'median')
            fill_na_value: 填充NaN的固定值
            remove_duplicates: 是否删除重复行
            datetime_column: 日期时间列名
            
        Returns:
            清洗后的数据
        """
        logger.info(f"开始数据清洗，数据形状: {data.shape}")
        
        # 复制数据以避免修改原数据
        cleaned_data = data.copy()
        
        # 处理日期时间列
        if datetime_column in cleaned_data.columns:
            cleaned_data[datetime_column] = pd.to_datetime(cleaned_data[datetime_column])
            # 设置日期时间为索引（如果尚未设置）
            if not isinstance(cleaned_data.index, pd.DatetimeIndex):
                cleaned_data.set_index(datetime_column, inplace=True)
        
        # 删除重复行
        if remove_duplicates:
            initial_len = len(cleaned_data)
            cleaned_data.drop_duplicates(inplace=True)
            logger.info(f"删除了 {initial_len - len(cleaned_data)} 行重复数据")
        
        # 处理NaN值
        if drop_na:
            initial_len = len(cleaned_data)
            cleaned_data.dropna(inplace=True)
            logger.info(f"删除了 {initial_len - len(cleaned_data)} 行包含NaN的数据")
        elif fill_na_method:
            if fill_na_method == 'ffill':
                cleaned_data.fillna(method='ffill', inplace=True)
            elif fill_na_method == 'bfill':
                cleaned_data.fillna(method='bfill', inplace=True)
            elif fill_na_method == 'mean':
                cleaned_data.fillna(cleaned_data.mean(), inplace=True)
            elif fill_na_method == 'median':
                cleaned_data.fillna(cleaned_data.median(), inplace=True)
            logger.info(f"使用 {fill_na_method} 方法填充NaN值")
        elif fill_na_value is not None:
            cleaned_data.fillna(fill_na_value, inplace=True)
            logger.info(f"使用固定值 {fill_na_value} 填充NaN值")
        
        # 排序（基于索引或日期时间列）
        if isinstance(cleaned_data.index, pd.DatetimeIndex):
            cleaned_data.sort_index(inplace=True)
        elif datetime_column in cleaned_data.columns:
            cleaned_data.sort_values(by=datetime_column, inplace=True)
        
        logger.info(f"数据清洗完成，清洗后数据形状: {cleaned_data.shape}")
        
        return cleaned_data
    
    def handle_outliers(self, 
                       data: pd.DataFrame,
                       columns: List[str],
                       method: str = 'clip',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            data: 数据
            columns: 要处理异常值的列名列表
            method: 处理方法 ('clip', 'drop', 'replace')
            threshold: 异常值阈值（Z-score方法使用）
            
        Returns:
            处理异常值后的数据
        """
        logger.info(f"开始处理异常值，列: {columns}, 方法: {method}")
        
        # 复制数据
        processed_data = data.copy()
        
        for column in columns:
            if column not in processed_data.columns:
                logger.warning(f"列 {column} 不存在于数据中，跳过")
                continue
            
            # 计算Z-score
            mean = processed_data[column].mean()
            std = processed_data[column].std()
            z_scores = (processed_data[column] - mean) / std
            
            # 确定异常值
            outliers = abs(z_scores) > threshold
            outlier_count = outliers.sum()
            
            if outlier_count == 0:
                logger.info(f"列 {column} 未检测到异常值")
                continue
            
            logger.info(f"列 {column} 检测到 {outlier_count} 个异常值")
            
            # 根据方法处理异常值
            if method == 'clip':
                # 截断到阈值范围内
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                processed_data[column] = processed_data[column].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"列 {column}: 异常值已截断到 [{lower_bound:.4f}, {upper_bound:.4f}]")
                
            elif method == 'drop':
                # 删除异常值
                initial_len = len(processed_data)
                processed_data = processed_data[~outliers]
                logger.info(f"列 {column}: 删除了 {initial_len - len(processed_data)} 行异常值")
                
            elif method == 'replace':
                # 替换为均值或中位数
                # 这里使用中位数更稳健
                median_val = processed_data[column].median()
                processed_data.loc[outliers, column] = median_val
                logger.info(f"列 {column}: 异常值已替换为中位数 {median_val:.4f}")
        
        return processed_data
    
    def create_lag_features(self, 
                           data: pd.DataFrame,
                           columns: List[str],
                           lags: List[int],
                           prefix: str = 'lag_') -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            data: 数据
            columns: 要创建滞后特征的列名列表
            lags: 滞后步数列表
            prefix: 滞后特征列名前缀
            
        Returns:
            添加滞后特征后的数据
        """
        logger.info(f"开始创建滞后特征，列: {columns}, 滞后步数: {lags}")
        
        # 复制数据
        lagged_data = data.copy()
        
        for column in columns:
            if column not in lagged_data.columns:
                logger.warning(f"列 {column} 不存在于数据中，跳过")
                continue
            
            for lag in lags:
                new_col_name = f"{prefix}{column}_{lag}"
                lagged_data[new_col_name] = lagged_data[column].shift(lag)
                logger.info(f"创建滞后特征: {new_col_name}")
        
        # 删除由于滞后操作产生的NaN值
        lagged_data.dropna(inplace=True)
        logger.info(f"滞后特征创建完成，数据形状: {lagged_data.shape}")
        
        return lagged_data
    
    def create_rolling_features(self, 
                               data: pd.DataFrame,
                               columns: List[str],
                               windows: List[int],
                               functions: List[str] = ['mean', 'std', 'min', 'max'],
                               prefix: str = 'rolling_') -> pd.DataFrame:
        """
        创建滚动窗口特征
        
        Args:
            data: 数据
            columns: 要创建滚动特征的列名列表
            windows: 窗口大小列表
            functions: 聚合函数列表
            prefix: 滚动特征列名前缀
            
        Returns:
            添加滚动窗口特征后的数据
        """
        logger.info(f"开始创建滚动窗口特征，列: {columns}, 窗口大小: {windows}, 函数: {functions}")
        
        # 复制数据
        rolling_data = data.copy()
        
        # 支持的聚合函数映射
        agg_funcs = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'sum': np.sum,
            'median': np.median,
            'skew': pd.Series.skew,
            'kurt': pd.Series.kurtosis
        }
        
        for column in columns:
            if column not in rolling_data.columns:
                logger.warning(f"列 {column} 不存在于数据中，跳过")
                continue
            
            for window in windows:
                for func_name in functions:
                    if func_name not in agg_funcs:
                        logger.warning(f"不支持的聚合函数: {func_name}，跳过")
                        continue
                    
                    # 计算滚动特征
                    new_col_name = f"{prefix}{column}_{func_name}_{window}"
                    rolling_data[new_col_name] = rolling_data[column].rolling(window=window).agg(agg_funcs[func_name])
                    logger.info(f"创建滚动特征: {new_col_name}")
        
        # 删除由于滚动操作产生的NaN值
        rolling_data.dropna(inplace=True)
        logger.info(f"滚动窗口特征创建完成，数据形状: {rolling_data.shape}")
        
        return rolling_data
    
    def create_time_features(self, 
                            data: pd.DataFrame,
                            datetime_column: Optional[str] = None,
                            include_weekday: bool = True,
                            include_month: bool = True,
                            include_quarter: bool = True,
                            include_year: bool = False,
                            include_hour: bool = False,
                            include_minute: bool = False,
                            encode_cyclical: bool = True) -> pd.DataFrame:
        """
        创建时间相关特征
        
        Args:
            data: 数据
            datetime_column: 日期时间列名（如果索引不是DatetimeIndex）
            include_weekday: 是否包含星期
            include_month: 是否包含月份
            include_quarter: 是否包含季度
            include_year: 是否包含年份
            include_hour: 是否包含小时
            include_minute: 是否包含分钟
            encode_cyclical: 是否将循环特征编码为sin和cos
            
        Returns:
            添加时间特征后的数据
        """
        logger.info("开始创建时间特征")
        
        # 复制数据
        time_features_data = data.copy()
        
        # 确保有DatetimeIndex
        if isinstance(time_features_data.index, pd.DatetimeIndex):
            dt_series = time_features_data.index
        elif datetime_column and datetime_column in time_features_data.columns:
            dt_series = pd.to_datetime(time_features_data[datetime_column])
        else:
            raise ValueError("数据必须有DatetimeIndex或指定的日期时间列")
        
        # 创建时间特征
        features_to_add = {}
        
        if include_weekday:
            features_to_add['weekday'] = dt_series.weekday
        
        if include_month:
            features_to_add['month'] = dt_series.month
        
        if include_quarter:
            features_to_add['quarter'] = dt_series.quarter
        
        if include_year:
            features_to_add['year'] = dt_series.year
        
        if include_hour:
            features_to_add['hour'] = dt_series.hour
        
        if include_minute:
            features_to_add['minute'] = dt_series.minute
        
        # 添加特征到数据中
        for feature_name, feature_values in features_to_add.items():
            time_features_data[feature_name] = feature_values
            logger.info(f"添加时间特征: {feature_name}")
            
            # 编码循环特征
            if encode_cyclical:
                if feature_name == 'weekday':
                    max_val = 7
                elif feature_name == 'month':
                    max_val = 12
                elif feature_name == 'hour':
                    max_val = 24
                elif feature_name == 'minute':
                    max_val = 60
                elif feature_name == 'quarter':
                    max_val = 4
                else:
                    continue  # 非循环特征，不需要编码
                
                # 正弦和余弦编码
                time_features_data[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature_values / max_val)
                time_features_data[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature_values / max_val)
                logger.info(f"添加循环编码特征: {feature_name}_sin, {feature_name}_cos")
        
        logger.info(f"时间特征创建完成，数据形状: {time_features_data.shape}")
        
        return time_features_data
    
    def normalize_features(self, 
                          data: pd.DataFrame,
                          columns: List[str],
                          method: str = 'standard',
                          fit: bool = True) -> pd.DataFrame:
        """
        标准化/归一化特征
        
        Args:
            data: 数据
            columns: 要标准化的列名列表
            method: 标准化方法 ('standard', 'minmax', 'robust')
            fit: 是否拟合标准化器（在测试集上应设为False）
            
        Returns:
            标准化后的数据
        """
        logger.info(f"开始标准化特征，方法: {method}, 列: {columns}")
        
        # 复制数据
        normalized_data = data.copy()
        
        for column in columns:
            if column not in normalized_data.columns:
                logger.warning(f"列 {column} 不存在于数据中，跳过")
                continue
            
            # 检查是否需要创建新的标准化器
            if fit:
                if method == 'standard':
                    self.scalers[column] = StandardScaler()
                elif method == 'minmax':
                    self.scalers[column] = MinMaxScaler()
                elif method == 'robust':
                    self.scalers[column] = RobustScaler()
                else:
                    raise ValueError(f"不支持的标准化方法: {method}")
                
                # 拟合标准化器并转换
                normalized_data[column] = self.scalers[column].fit_transform(normalized_data[[column]]).flatten()
                logger.info(f"拟合并标准化列: {column}")
            else:
                # 使用已拟合的标准化器
                if column not in self.scalers:
                    logger.warning(f"列 {column} 没有拟合的标准化器，跳过")
                    continue
                
                normalized_data[column] = self.scalers[column].transform(normalized_data[[column]]).flatten()
                logger.info(f"使用已有标准化器转换列: {column}")
        
        return normalized_data
    
    def select_features(self, 
                       X: pd.DataFrame,
                       y: pd.Series,
                       k: int = 10,
                       method: str = 'kbest') -> Tuple[pd.DataFrame, List[str]]:
        """
        特征选择
        
        Args:
            X: 特征数据
            y: 目标变量
            k: 要选择的特征数量
            method: 特征选择方法 ('kbest')
            
        Returns:
            (选择特征后的数据, 选择的特征名称列表)
        """
        logger.info(f"开始特征选择，方法: {method}, 选择特征数量: {k}")
        
        # 限制选择的特征数量不超过总特征数
        k = min(k, X.shape[1])
        
        if method == 'kbest':
            # 使用SelectKBest方法
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # 获取选择的特征索引和名称
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_features = X.columns[selected_indices].tolist()
            
            # 获取特征重要性得分
            scores = self.feature_selector.scores_
            self.feature_importance = {}
            for i, feature in enumerate(X.columns):
                self.feature_importance[feature] = scores[i]
            
            logger.info(f"特征选择完成，选择的特征: {selected_features}")
            
            # 返回选择特征后的数据
            return X[selected_features], selected_features
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
    
    def split_time_series(self, 
                         data: pd.DataFrame,
                         target_column: str,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                      pd.Series, pd.Series, pd.Series]:
        """
        时间序列数据分割
        
        Args:
            data: 完整数据
            target_column: 目标列名
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱数据（时间序列通常不打乱）
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"开始分割时间序列数据，训练集比例: {train_ratio}, 验证集比例: {val_ratio}, 测试集比例: {test_ratio}")
        
        # 确保比例总和为1
        ratios_sum = train_ratio + val_ratio + test_ratio
        if not np.isclose(ratios_sum, 1.0):
            logger.warning(f"比例总和不为1 ({ratios_sum})，将进行归一化")
            train_ratio /= ratios_sum
            val_ratio /= ratios_sum
            test_ratio /= ratios_sum
        
        # 分离特征和目标
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 分割数据
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # 不打乱的情况（时间序列默认情况）
        if not shuffle:
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            
            X_val = X.iloc[train_size:train_size + val_size]
            y_val = y.iloc[train_size:train_size + val_size]
            
            X_test = X.iloc[train_size + val_size:]
            y_test = y.iloc[train_size + val_size:]
        else:
            # 打乱的情况
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            X = X.iloc[indices]
            y = y.iloc[indices]
            
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            
            X_val = X.iloc[train_size:train_size + val_size]
            y_val = y.iloc[train_size:train_size + val_size]
            
            X_test = X.iloc[train_size + val_size:]
            y_test = y.iloc[train_size + val_size:]
        
        logger.info(f"数据分割完成：")
        logger.info(f"  - 训练集: {len(X_train)} 样本")
        logger.info(f"  - 验证集: {len(X_val)} 样本")
        logger.info(f"  - 测试集: {len(X_test)} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_sequences(self, 
                        X: pd.DataFrame,
                        y: pd.Series,
                        sequence_length: int,
                        forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建用于序列模型（如LSTM）的输入序列
        
        Args:
            X: 特征数据
            y: 目标变量
            sequence_length: 输入序列长度
            forecast_horizon: 预测步长
            
        Returns:
            (X_sequences, y_sequences)
        """
        logger.info(f"开始创建序列数据，序列长度: {sequence_length}, 预测步长: {forecast_horizon}")
        
        X_sequences = []
        y_sequences = []
        
        # 转换为numpy数组
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # 创建序列
        for i in range(len(X_array) - sequence_length - forecast_horizon + 1):
            # 输入序列
            X_seq = X_array[i:i + sequence_length]
            
            # 目标序列
            y_seq = y_array[i + sequence_length:i + sequence_length + forecast_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        # 转换为numpy数组
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"序列数据创建完成，形状: X={X_sequences.shape}, y={y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def apply_feature_transform(self, 
                              data: pd.DataFrame,
                              transform_funcs: Dict[str, Callable]) -> pd.DataFrame:
        """
        应用自定义特征变换函数
        
        Args:
            data: 数据
            transform_funcs: 变换函数字典 {新列名: 变换函数}
            
        Returns:
            添加变换特征后的数据
        """
        logger.info(f"开始应用自定义特征变换，函数数量: {len(transform_funcs)}")
        
        # 复制数据
        transformed_data = data.copy()
        
        for new_col_name, transform_func in transform_funcs.items():
            try:
                transformed_data[new_col_name] = transform_func(transformed_data)
                logger.info(f"应用特征变换: {new_col_name}")
            except Exception as e:
                logger.error(f"应用特征变换 {new_col_name} 失败: {str(e)}")
                continue
        
        logger.info(f"特征变换完成，数据形状: {transformed_data.shape}")
        
        return transformed_data
    
    def remove_correlated_features(self, 
                                  X: pd.DataFrame,
                                  threshold: float = 0.9) -> Tuple[pd.DataFrame, List[str]]:
        """
        删除高度相关的特征
        
        Args:
            X: 特征数据
            threshold: 相关系数阈值
            
        Returns:
            (删除相关特征后的数据, 保留的特征名称列表)
        """
        logger.info(f"开始删除相关特征，相关系数阈值: {threshold}")
        
        # 计算相关矩阵
        corr_matrix = X.corr().abs()
        
        # 选择上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找到相关系数大于阈值的列
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        logger.info(f"删除了 {len(to_drop)} 个高度相关的特征: {to_drop}")
        
        # 删除相关特征
        X_filtered = X.drop(columns=to_drop)
        
        return X_filtered, X_filtered.columns.tolist()
    
    def save_preprocessing_state(self, path: str) -> None:
        """
        保存预处理状态（用于后续转换新数据）
        
        Args:
            path: 保存路径
        """
        import pickle
        
        state = {
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"预处理状态已保存到: {path}")
    
    def load_preprocessing_state(self, path: str) -> None:
        """
        加载预处理状态
        
        Args:
            path: 加载路径
        """
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.scalers = state.get('scalers', {})
        self.feature_selector = state.get('feature_selector', None)
        self.feature_names = state.get('feature_names', None)
        self.feature_importance = state.get('feature_importance', None)
        
        logger.info(f"预处理状态已从: {path} 加载")


class TimeSeriesDataLoader:
    """
    时间序列数据加载器
    
    负责从不同来源加载时间序列数据并进行初步处理
    """
    
    @staticmethod
    def load_from_csv(file_path: str, 
                     datetime_column: str = 'timestamp',
                     index_column: Optional[str] = None,
                     parse_dates: bool = True,
                     **kwargs) -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        Args:
            file_path: CSV文件路径
            datetime_column: 日期时间列名
            index_column: 索引列名
            parse_dates: 是否解析日期
            **kwargs: 传递给pd.read_csv的其他参数
            
        Returns:
            加载的数据
        """
        logger.info(f"从CSV文件加载数据: {file_path}")
        
        # 设置参数
        if parse_dates and datetime_column:
            kwargs['parse_dates'] = [datetime_column]
        
        if index_column:
            kwargs['index_col'] = index_column
        elif parse_dates and datetime_column:
            kwargs['index_col'] = datetime_column
        
        # 加载数据
        data = pd.read_csv(file_path, **kwargs)
        
        logger.info(f"数据加载完成，形状: {data.shape}")
        
        return data
    
    @staticmethod
    def load_from_excel(file_path: str, 
                       sheet_name: str = 0,
                       datetime_column: str = 'timestamp',
                       index_column: Optional[str] = None,
                       parse_dates: bool = True,
                       **kwargs) -> pd.DataFrame:
        """
        从Excel文件加载数据
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称
            datetime_column: 日期时间列名
            index_column: 索引列名
            parse_dates: 是否解析日期
            **kwargs: 传递给pd.read_excel的其他参数
            
        Returns:
            加载的数据
        """
        logger.info(f"从Excel文件加载数据: {file_path}, 工作表: {sheet_name}")
        
        # 设置参数
        if parse_dates and datetime_column:
            kwargs['parse_dates'] = [datetime_column]
        
        # 加载数据
        data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        
        # 设置索引
        if index_column:
            data.set_index(index_column, inplace=True)
        elif parse_dates and datetime_column in data.columns:
            data.set_index(datetime_column, inplace=True)
        
        logger.info(f"数据加载完成，形状: {data.shape}")
        
        return data
    
    @staticmethod
    def load_from_multiple_files(file_paths: List[str], 
                               datetime_column: str = 'timestamp',
                               how: str = 'concat',
                               **kwargs) -> pd.DataFrame:
        """
        从多个文件加载数据并合并
        
        Args:
            file_paths: 文件路径列表
            datetime_column: 日期时间列名
            how: 合并方式 ('concat', 'merge')
            **kwargs: 传递给加载函数的其他参数
            
        Returns:
            合并后的数据
        """
        logger.info(f"从多个文件加载数据，文件数量: {len(file_paths)}")
        
        # 加载所有文件
        data_frames = []
        for file_path in file_paths:
            if file_path.endswith('.csv'):
                df = TimeSeriesDataLoader.load_from_csv(
                    file_path, datetime_column=datetime_column, **kwargs
                )
            elif file_path.endswith(('.xlsx', '.xls')):
                df = TimeSeriesDataLoader.load_from_excel(
                    file_path, datetime_column=datetime_column, **kwargs
                )
            else:
                logger.warning(f"不支持的文件格式: {file_path}，跳过")
                continue
            
            data_frames.append(df)
        
        # 合并数据
        if how == 'concat':
            merged_data = pd.concat(data_frames, axis=0)
        elif how == 'merge':
            # 使用merge合并（需要有共同的列）
            if not data_frames:
                return pd.DataFrame()
                
            merged_data = data_frames[0]
            for df in data_frames[1:]:
                merged_data = pd.merge(merged_data, df, on=datetime_column, how='outer')
        else:
            raise ValueError(f"不支持的合并方式: {how}")
        
        # 排序
        if isinstance(merged_data.index, pd.DatetimeIndex):
            merged_data.sort_index(inplace=True)
        elif datetime_column in merged_data.columns:
            merged_data.sort_values(by=datetime_column, inplace=True)
        
        logger.info(f"数据合并完成，形状: {merged_data.shape}")
        
        return merged_data
    
    @staticmethod
    def resample_time_series(data: pd.DataFrame,
                           rule: str,
                           agg_dict: Optional[Dict[str, str]] = None,
                           **kwargs) -> pd.DataFrame:
        """
        重采样时间序列数据
        
        Args:
            data: 时间序列数据（需要有DatetimeIndex）
            rule: 重采样规则 ('D', 'W', 'M', 'Q', 'Y' 等)
            agg_dict: 聚合函数字典 {column: agg_func}
            **kwargs: 传递给resample的其他参数
            
        Returns:
            重采样后的数据
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据必须有DatetimeIndex才能重采样")
        
        logger.info(f"重采样时间序列数据，规则: {rule}")
        
        # 执行重采样
        if agg_dict:
            resampled_data = data.resample(rule, **kwargs).agg(agg_dict)
        else:
            # 默认使用均值聚合
            resampled_data = data.resample(rule, **kwargs).mean()
        
        logger.info(f"重采样完成，形状: {resampled_data.shape}")
        
        return resampled_data


def create_feature_pipeline(config: Dict) -> List[Callable]:
    """
    根据配置创建特征工程流水线
    
    Args:
        config: 特征工程配置
        
    Returns:
        处理函数列表
    """
    pipeline = []
    preprocessor = DataPreprocessor()
    
    # 根据配置创建处理函数
    if 'clean_data' in config:
        clean_config = config['clean_data']
        pipeline.append(lambda data: preprocessor.clean_data(data, **clean_config))
    
    if 'handle_outliers' in config:
        outlier_config = config['handle_outliers']
        pipeline.append(lambda data: preprocessor.handle_outliers(data, **outlier_config))
    
    if 'create_lag_features' in config:
        lag_config = config['create_lag_features']
        pipeline.append(lambda data: preprocessor.create_lag_features(data, **lag_config))
    
    if 'create_rolling_features' in config:
        rolling_config = config['create_rolling_features']
        pipeline.append(lambda data: preprocessor.create_rolling_features(data, **rolling_config))
    
    if 'create_time_features' in config:
        time_config = config['create_time_features']
        pipeline.append(lambda data: preprocessor.create_time_features(data, **time_config))
    
    return pipeline, preprocessor


def apply_feature_pipeline(data: pd.DataFrame, 
                           pipeline: List[Callable]) -> pd.DataFrame:
    """
    应用特征工程流水线
    
    Args:
        data: 数据
        pipeline: 处理函数列表
        
    Returns:
        处理后的数据
    """
    processed_data = data.copy()
    
    for process_func in pipeline:
        processed_data = process_func(processed_data)
    
    return processed_data