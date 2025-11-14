import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime


class FactorBase(ABC):
    """
    因子基类，所有具体因子都需要继承此类并实现抽象方法
    
    设计理念：
    1. 标准化接口：提供统一的因子计算、验证和缓存机制
    2. 参数化配置：支持灵活的参数设置和调优
    3. 缓存机制：避免重复计算，提高性能
    4. 可扩展性：便于添加新的因子类型
    5. 验证支持：内置基本的因子验证功能
    """
    
    def __init__(self, factor_name: str, params: Optional[Dict] = None):
        """
        初始化因子
        
        Args:
            factor_name: 因子名称，用于标识和缓存
            params: 因子计算所需的参数
        """
        self.factor_name = factor_name
        self.params = params or {}
        self.cache = {}
        self._factor_type = "base"
        self._description = "基础因子类"
        
    @property
    def factor_type(self) -> str:
        """获取因子类型"""
        return self._factor_type
    
    @property
    def description(self) -> str:
        """获取因子描述"""
        return self._description
    
    @abstractmethod
    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        抽象方法：计算因子值
        
        Args:
            data: 输入数据，通常是OHLCV格式的DataFrame
            
        Returns:
            包含因子值的DataFrame
        """
        pass
    
    def compute(self, data: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        计算因子值的公共接口，包含缓存机制
        
        Args:
            data: 输入数据
            use_cache: 是否使用缓存
            
        Returns:
            包含因子值的DataFrame
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(data)
        
        # 检查缓存
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # 执行计算
        result = self._compute(data)
        
        # 缓存结果
        if use_cache:
            self.cache[cache_key] = result.copy()
        
        return result
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """
        生成缓存键
        
        Args:
            data: 输入数据
            
        Returns:
            缓存键字符串
        """
        # 使用数据的哈希值、因子名和参数生成缓存键
        # 注意：对于大数据集，这可能会影响性能，可以改进为更简单的键生成策略
        data_hash = str(hash(data.shape))
        if not data.empty:
            # 使用数据的时间范围作为部分缓存键
            if isinstance(data.index, pd.DatetimeIndex):
                time_range = f"{data.index.min()}_{data.index.max()}"
            else:
                time_range = f"{data.index.min()}_{data.index.max()}"
        else:
            time_range = "empty"
        
        params_str = "_".join([f"{k}={v}" for k, v in sorted(self.params.items())])
        return f"{self.factor_name}_{time_range}_{params_str}_{data_hash}"
    
    def validate(self, factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证因子计算结果
        
        Args:
            factor_data: 因子计算结果
            
        Returns:
            验证结果字典
        """
        validation_results = {
            "factor_name": self.factor_name,
            "timestamp": datetime.now().isoformat(),
            "has_nan": factor_data.isnull().values.any(),
            "nan_count": factor_data.isnull().sum().sum(),
            "data_shape": factor_data.shape,
            "stats": {}
        }
        
        # 计算基本统计量
        for col in factor_data.columns:
            if np.issubdtype(factor_data[col].dtype, np.number):
                validation_results["stats"][col] = {
                    "mean": factor_data[col].mean(),
                    "std": factor_data[col].std(),
                    "min": factor_data[col].min(),
                    "max": factor_data[col].max(),
                    "skew": factor_data[col].skew(),
                    "kurtosis": factor_data[col].kurtosis()
                }
        
        return validation_results
    
    def clear_cache(self):
        """
        清除因子缓存
        """
        self.cache.clear()
    
    def get_params(self) -> Dict:
        """
        获取因子参数
        
        Returns:
            参数字典
        """
        return self.params.copy()
    
    def set_params(self, params: Dict):
        """
        更新因子参数
        
        Args:
            params: 新的参数字典
        """
        self.params.update(params)
        # 更新参数后清除缓存
        self.clear_cache()
    
    def __str__(self) -> str:
        """
        返回因子的字符串表示
        """
        return f"{self.factor_name} (Type: {self.factor_type}, Params: {self.params})"
    
    def __repr__(self) -> str:
        """
        返回因子的详细表示
        """
        return f"{self.__class__.__name__}(name='{self.factor_name}', params={self.params})"


class AlphaFactor(FactorBase):
    """
    Alpha因子基类，专注于预测资产价格变动的因子
    """
    
    def __init__(self, factor_name: str, params: Optional[Dict] = None):
        super().__init__(factor_name, params)
        self._factor_type = "alpha"
        self._description = "Alpha因子，用于预测资产价格变动"


class RiskFactor(FactorBase):
    """
    风险因子基类，专注于衡量市场风险的因子
    """
    
    def __init__(self, factor_name: str, params: Optional[Dict] = None):
        super().__init__(factor_name, params)
        self._factor_type = "risk"
        self._description = "风险因子，用于衡量市场风险"


class FactorContainer:
    """
    因子容器类，用于管理多个因子
    
    功能：
    1. 注册和管理多个因子实例
    2. 批量计算因子值
    3. 合并多个因子的计算结果
    4. 统一的缓存管理
    """
    
    def __init__(self):
        """
        初始化因子容器
        """
        self.factors: Dict[str, FactorBase] = {}
        self.global_cache = {}
    
    def add_factor(self, factor: FactorBase):
        """
        添加因子到容器
        
        Args:
            factor: FactorBase的实例
        """
        if factor.factor_name in self.factors:
            print(f"警告：因子 {factor.factor_name} 已存在，将被覆盖")
        self.factors[factor.factor_name] = factor
    
    def remove_factor(self, factor_name: str):
        """
        从容器中移除因子
        
        Args:
            factor_name: 要移除的因子名称
        """
        if factor_name in self.factors:
            del self.factors[factor_name]
        else:
            print(f"警告：因子 {factor_name} 不存在")
    
    def compute_all(self, data: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        计算所有因子的值
        
        Args:
            data: 输入数据
            use_cache: 是否使用缓存
            
        Returns:
            合并后的因子值DataFrame
        """
        results = {}
        
        for factor_name, factor in self.factors.items():
            factor_result = factor.compute(data, use_cache)
            
            # 确保结果是DataFrame
            if isinstance(factor_result, pd.Series):
                factor_result = factor_result.to_frame(name=factor_name)
            
            # 确保索引对齐
            if not factor_result.index.equals(data.index):
                # 尝试重新索引到原始数据的索引
                factor_result = factor_result.reindex(data.index)
            
            results[factor_name] = factor_result
        
        # 合并所有因子结果
        if not results:
            return pd.DataFrame(index=data.index)
        
        # 使用join而不是concat，以确保索引对齐
        merged_result = pd.DataFrame(index=data.index)
        for factor_name, result_df in results.items():
            # 为避免列名冲突，为每个因子的列添加前缀
            result_df_renamed = result_df.add_prefix(f"{factor_name}_")
            merged_result = merged_result.join(result_df_renamed)
        
        return merged_result
    
    def get_factor(self, factor_name: str) -> Optional[FactorBase]:
        """
        获取指定名称的因子
        
        Args:
            factor_name: 因子名称
            
        Returns:
            FactorBase实例或None
        """
        return self.factors.get(factor_name)
    
    def get_all_factors(self) -> List[FactorBase]:
        """
        获取所有因子实例
        
        Returns:
            FactorBase实例列表
        """
        return list(self.factors.values())
    
    def validate_all(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        验证所有因子
        
        Args:
            data: 输入数据
            
        Returns:
            每个因子的验证结果
        """
        validation_results = {}
        
        for factor_name, factor in self.factors.items():
            factor_data = factor.compute(data)
            validation_results[factor_name] = factor.validate(factor_data)
        
        return validation_results
    
    def clear_all_caches(self):
        """
        清除所有因子的缓存
        """
        for factor in self.factors.values():
            factor.clear_cache()
        self.global_cache.clear()
    
    def __len__(self) -> int:
        """
        返回容器中因子的数量
        """
        return len(self.factors)
    
    def __contains__(self, factor_name: str) -> bool:
        """
        检查容器中是否包含指定名称的因子
        """
        return factor_name in self.factors
    
    def __str__(self) -> str:
        """
        返回容器的字符串表示
        """
        return f"FactorContainer with {len(self)} factors: {list(self.factors.keys())}"
