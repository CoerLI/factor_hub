import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union


class DataLoader:
    """
    数据加载器类，负责从CSV或Parquet文件加载金融时间序列数据
    并提供数据预处理和转换功能
    """
    
    def __init__(self, data_dir: str = "index_data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据文件所在目录
        """
        self.data_dir = data_dir
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_csv_data(self, file_name: str, **kwargs) -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        Args:
            file_name: 文件名或完整路径
            **kwargs: 传递给pandas.read_csv的其他参数
            
        Returns:
            加载的DataFrame数据
        """
        # 检查是否提供了完整路径
        file_path = file_name if os.path.isabs(file_name) else os.path.join(self.data_dir, file_name)
        
        # 检查缓存
        if file_path in self.data_cache:
            return self.data_cache[file_path].copy()
        
        # 默认参数设置
        default_kwargs = {
            'parse_dates': ['kline_time'] if 'kline_time' in pd.read_csv(file_path, nrows=0).columns else False,
            'index_col': 'kline_time' if 'kline_time' in pd.read_csv(file_path, nrows=0).columns else None
        }
        default_kwargs.update(kwargs)
        
        # 加载数据
        data = pd.read_csv(file_path, **default_kwargs)
        
        # 缓存数据
        self.data_cache[file_path] = data.copy()
        
        return data
    
    def load_parquet_data(self, file_name: str, **kwargs) -> pd.DataFrame:
        """
        从Parquet文件加载数据
        
        Args:
            file_name: 文件名或完整路径
            **kwargs: 传递给pandas.read_parquet的其他参数
            
        Returns:
            加载的DataFrame数据
        """
        # 检查是否提供了完整路径
        file_path = file_name if os.path.isabs(file_name) else os.path.join(self.data_dir, file_name)
        
        # 检查缓存
        if file_path in self.data_cache:
            return self.data_cache[file_path].copy()
        
        # 加载数据
        data = pd.read_parquet(file_path, **kwargs)
        
        # 缓存数据
        self.data_cache[file_path] = data.copy()
        
        return data
    
    def load_data(self, file_name: str, **kwargs) -> pd.DataFrame:
        """
        根据文件扩展名自动选择加载方法
        
        Args:
            file_name: 文件名
            **kwargs: 传递给具体加载方法的参数
            
        Returns:
            加载的DataFrame数据
        """
        if file_name.endswith('.csv'):
            return self.load_csv_data(file_name, **kwargs)
        elif file_name.endswith('.parquet'):
            return self.load_parquet_data(file_name, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_name}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        # 创建副本以避免修改原始数据
        processed_data = data.copy()
        
        # 处理缺失值
        if processed_data.isnull().values.any():
            # 对于价格数据，使用前向填充
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in processed_data.columns:
                    processed_data[col] = processed_data[col].ffill()
            
            # 对于成交量数据，使用0填充
            volume_columns = ['volume', 'amount']
            for col in volume_columns:
                if col in processed_data.columns:
                    processed_data[col] = processed_data[col].fillna(0)
        
        return processed_data
    
    def convert_to_parquet(self, csv_file: str, parquet_file: str = None, **kwargs) -> str:
        """
        将CSV文件转换为Parquet格式
        
        Args:
            csv_file: CSV文件路径
            parquet_file: 输出的Parquet文件路径，默认为替换CSV扩展名为parquet
            **kwargs: 传递给to_parquet的参数
            
        Returns:
            生成的Parquet文件路径
        """
        # 加载CSV数据
        data = self.load_csv_data(csv_file)
        
        # 如果未指定输出文件名，则自动生成
        if parquet_file is None:
            csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(self.data_dir, csv_file)
            parquet_file = os.path.splitext(csv_path)[0] + '.parquet'
        
        # 保存为Parquet格式
        data.to_parquet(parquet_file, **kwargs)
        
        return parquet_file
    
    def load_multiple_files(self, file_pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
        """
        批量加载多个文件
        
        Args:
            file_pattern: 文件匹配模式，如"*.csv"或"*_day.csv"
            
        Returns:
            文件名到DataFrame的映射字典
        """
        import glob
        
        # 获取匹配的文件列表
        search_path = os.path.join(self.data_dir, file_pattern)
        files = glob.glob(search_path)
        
        # 加载所有匹配的文件
        result = {}
        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                data = self.load_data(file_path)
                result[file_name] = data
            except Exception as e:
                print(f"加载文件 {file_name} 时出错: {e}")
        
        return result
    
    def clear_cache(self):
        """
        清除数据缓存
        """
        self.data_cache.clear()
    
    def get_available_files(self, extension: str = ".csv") -> List[str]:
        """
        获取指定目录下所有可用的数据文件
        
        Args:
            extension: 文件扩展名，如".csv"或".parquet"
            
        Returns:
            文件列表
        """
        import glob
        
        search_path = os.path.join(self.data_dir, f"*{extension}")
        return [os.path.basename(file) for file in glob.glob(search_path)]
