import pandas as pd
import numpy as np
import json
import inspect
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import hashlib
import pickle

from src.factors.factor_base import FactorBase, FactorContainer, FactorResult
from src.utils.helpers import setup_logger, DEFAULT_LOGGER


class FactorMetadataCollector:
    """
    因子元数据收集器，负责详细收集和序列化因子信息
    
    功能特点：
    1. 收集因子的完整元数据，包括基本信息、参数配置、计算性能等
    2. 分析因子计算结果的统计特征
    3. 生成因子依赖关系图
    4. 支持元数据的序列化和反序列化
    5. 提供因子版本控制信息
    6. 计算因子的唯一性标识
    """
    
    def __init__(self):
        """初始化因子元数据收集器"""
        self.logger = setup_logger(
            name="factor_metadata_collector",
            log_file=os.path.join("logs", "factor_metadata.log"),
            level=DEFAULT_LOGGER.level
        )
    
    def collect_metadata(self, 
                        factor: FactorBase,
                        data: Optional[pd.DataFrame] = None,
                        compute_result: Optional[pd.DataFrame] = None,
                        computation_time: Optional[float] = None,
                        dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        收集因子的完整元数据
        
        Args:
            factor: 因子实例
            data: 输入数据（可选）
            compute_result: 计算结果（可选）
            computation_time: 计算时间（秒，可选）
            dependencies: 因子依赖列表（可选）
            
        Returns:
            完整的因子元数据字典
        """
        # 基本信息
        basic_info = self._collect_basic_info(factor)
        
        # 参数信息
        params_info = self._collect_params_info(factor)
        
        # 类信息
        class_info = self._collect_class_info(factor)
        
        # 性能信息
        performance_info = self._collect_performance_info(factor, computation_time)
        
        # 因子计算结果统计（如果提供）
        result_stats = {}
        if compute_result is not None:
            result_stats = self._analyze_factor_result(compute_result)
        
        # 数据特征（如果提供）
        data_features = {}
        if data is not None:
            data_features = self._analyze_input_data(data)
        
        # 依赖关系
        dependency_info = self._collect_dependency_info(dependencies)
        
        # 版本信息
        version_info = self._collect_version_info()
        
        # 唯一标识
        unique_id = self._generate_unique_id(factor, params_info["parameters"])
        
        # 合并所有信息
        metadata = {
            "unique_id": unique_id,
            "basic_info": basic_info,
            "params_info": params_info,
            "class_info": class_info,
            "performance_info": performance_info,
            "result_stats": result_stats,
            "data_features": data_features,
            "dependency_info": dependency_info,
            "version_info": version_info,
            "collected_at": datetime.now().isoformat()
        }
        
        return metadata
    
    def _collect_basic_info(self, factor: FactorBase) -> Dict[str, Any]:
        """
        收集因子的基本信息
        
        Args:
            factor: 因子实例
            
        Returns:
            基本信息字典
        """
        return {
            "factor_name": factor.factor_name,
            "factor_type": factor.factor_type,
            "description": factor.description,
            "created_at": getattr(factor, "created_at", datetime.now().isoformat()),
            "is_custom": not factor.__class__.__module__.startswith("src.factors"),
            "cache_enabled": hasattr(factor, "cache") and bool(factor.cache)
        }
    
    def _collect_params_info(self, factor: FactorBase) -> Dict[str, Any]:
        """
        收集因子的参数信息
        
        Args:
            factor: 因子实例
            
        Returns:
            参数信息字典
        """
        # 获取参数
        params = getattr(factor, "params", {})
        
        # 分析参数类型和范围
        param_details = {}
        for param_name, param_value in params.items():
            param_details[param_name] = {
                "value": param_value,
                "type": type(param_value).__name__,
                "is_default": self._is_default_param(factor, param_name, param_value)
            }
            
            # 添加数值参数的范围信息（如果适用）
            if isinstance(param_value, (int, float)):
                param_details[param_name]["range"] = self._get_param_range(factor, param_name)
        
        return {
            "parameters": params,
            "param_count": len(params),
            "param_details": param_details,
            "params_hash": self._hash_params(params)
        }
    
    def _is_default_param(self, factor: FactorBase, param_name: str, param_value: Any) -> bool:
        """
        判断参数是否为默认值
        
        Args:
            factor: 因子实例
            param_name: 参数名
            param_value: 参数值
            
        Returns:
            是否为默认值
        """
        # 尝试获取类的默认参数
        try:
            sig = inspect.signature(factor.__init__)
            if param_name in sig.parameters:
                default = sig.parameters[param_name].default
                if default is not inspect.Parameter.empty:
                    return param_value == default
        except Exception as e:
            self.logger.warning(f"获取默认参数信息时出错: {e}")
        
        return False
    
    def _get_param_range(self, factor: FactorBase, param_name: str) -> Optional[Dict[str, Any]]:
        """
        获取参数的合理范围（如果有定义）
        
        Args:
            factor: 因子实例
            param_name: 参数名
            
        Returns:
            参数范围字典
        """
        # 检查是否有参数范围定义
        if hasattr(factor, "PARAM_RANGES") and param_name in factor.PARAM_RANGES:
            return factor.PARAM_RANGES[param_name]
        
        # 基于常见因子参数的经验范围
        common_ranges = {
            "window": {"min": 1, "max": 365, "step": 1},
            "period": {"min": 1, "max": 365, "step": 1},
            "lag": {"min": 1, "max": 100, "step": 1},
            "threshold": {"min": 0.0, "max": 1.0, "step": 0.01},
            "multiplier": {"min": 0.1, "max": 5.0, "step": 0.1},
            "decay": {"min": 0.01, "max": 1.0, "step": 0.01}
        }
        
        return common_ranges.get(param_name)
    
    def _hash_params(self, params: Dict) -> str:
        """
        计算参数的哈希值
        
        Args:
            params: 参数字典
            
        Returns:
            哈希值字符串
        """
        params_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _collect_class_info(self, factor: FactorBase) -> Dict[str, Any]:
        """
        收集因子类的信息
        
        Args:
            factor: 因子实例
            
        Returns:
            类信息字典
        """
        factor_class = factor.__class__
        
        # 获取继承层次
        inheritance_chain = [cls.__name__ for cls in inspect.getmro(factor_class)]
        
        # 获取类定义位置
        module_name = factor_class.__module__
        class_name = factor_class.__name__
        
        # 获取类文档
        docstring = inspect.getdoc(factor_class) or ""
        
        # 获取类属性
        class_attrs = {}
        for attr_name in dir(factor_class):
            if not attr_name.startswith("_") and not callable(getattr(factor_class, attr_name)):
                try:
                    class_attrs[attr_name] = str(getattr(factor_class, attr_name))
                except Exception:
                    class_attrs[attr_name] = "<unserializable>"
        
        return {
            "class_name": class_name,
            "module_name": module_name,
            "full_class_name": f"{module_name}.{class_name}",
            "inheritance_chain": inheritance_chain,
            "docstring": docstring,
            "class_attributes": class_attrs,
            "source_file": inspect.getfile(factor_class) if hasattr(inspect, 'getfile') else "unknown"
        }
    
    def _collect_performance_info(self, 
                                 factor: FactorBase,
                                 computation_time: Optional[float] = None) -> Dict[str, Any]:
        """
        收集因子的性能信息
        
        Args:
            factor: 因子实例
            computation_time: 计算时间（秒）
            
        Returns:
            性能信息字典
        """
        # 获取缓存信息
        cache_info = {
            "cache_size": len(factor.cache) if hasattr(factor, "cache") else 0,
            "cache_memory_usage": self._estimate_cache_memory(factor) if hasattr(factor, "cache") else 0
        }
        
        return {
            "computation_time": computation_time,
            "last_computed": getattr(factor, "last_computed", None),
            "computation_count": getattr(factor, "computation_count", 0),
            "cache_info": cache_info,
            "estimated_complexity": self._estimate_complexity(factor)
        }
    
    def _estimate_cache_memory(self, factor: FactorBase) -> float:
        """
        估算缓存的内存使用量
        
        Args:
            factor: 因子实例
            
        Returns:
            估算的内存使用量（MB）
        """
        try:
            # 使用pickle来估算对象大小
            total_size = 0
            for key, value in factor.cache.items():
                total_size += len(pickle.dumps((key, value)))
            
            # 转换为MB
            return total_size / (1024 * 1024)
        except Exception as e:
            self.logger.warning(f"估算缓存内存使用量时出错: {e}")
            return 0
    
    def _estimate_complexity(self, factor: FactorBase) -> str:
        """
        估算因子计算的复杂度
        
        Args:
            factor: 因子实例
            
        Returns:
            复杂度描述（low, medium, high）
        """
        # 根据因子类型和参数估算复杂度
        if hasattr(factor, "factor_type"):
            # 基础统计类因子通常复杂度较低
            if factor.factor_type in ["moving_average", "rsi", "bollinger_bands"]:
                return "low"
            # 高级因子复杂度较高
            elif factor.factor_type in ["machine_learning", "wavelet", "fractal"]:
                return "high"
        
        # 根据参数数量和类型进一步判断
        params = getattr(factor, "params", {})
        if len(params) > 5:
            return "high"
        
        return "medium"
    
    def _analyze_factor_result(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        分析因子计算结果的统计特征
        
        Args:
            result: 因子计算结果DataFrame
            
        Returns:
            统计特征字典
        """
        # 基本统计信息
        stats = {
            "shape": {"rows": len(result), "columns": len(result.columns)},
            "columns": list(result.columns),
            "index_type": str(type(result.index)),
            "has_nulls": result.isnull().any().any(),
            "null_percentage": result.isnull().mean().mean() * 100
        }
        
        # 数值统计（针对每一列）
        numeric_stats = {}
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                col_stats = result[col].describe().to_dict()
                # 添加额外的统计量
                col_stats["kurtosis"] = result[col].kurtosis()
                col_stats["skew"] = result[col].skew()
                col_stats["autocorr_1"] = result[col].autocorr(lag=1)
                col_stats["autocorr_5"] = result[col].autocorr(lag=5)
                numeric_stats[col] = col_stats
        
        # 分布特征
        distribution = self._analyze_distribution(result)
        
        # 时间序列特征（如果索引是时间序列）
        time_series_features = {}
        if isinstance(result.index, pd.DatetimeIndex):
            time_series_features = self._analyze_time_series(result)
        
        return {
            "basic_stats": stats,
            "numeric_stats": numeric_stats,
            "distribution": distribution,
            "time_series_features": time_series_features,
            "data_hash": self._hash_dataframe(result)
        }
    
    def _analyze_distribution(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据分布特征
        
        Args:
            result: 因子计算结果DataFrame
            
        Returns:
            分布特征字典
        """
        distribution = {}
        
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                # 计算分位数
                quantiles = result[col].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
                
                # 检测异常值（基于IQR）
                Q1 = quantiles[0.25]
                Q3 = quantiles[0.75]
                IQR = Q3 - Q1
                outliers = result[col][(result[col] < Q1 - 1.5 * IQR) | (result[col] > Q3 + 1.5 * IQR)]
                
                distribution[col] = {
                    "quantiles": quantiles,
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(result)) * 100 if len(result) > 0 else 0,
                    "is_normal": self._test_normality(result[col])
                }
        
        return distribution
    
    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """
        简单的正态性检验
        
        Args:
            series: 数值序列
            
        Returns:
            正态性检验结果
        """
        # 移除NaN值
        series_clean = series.dropna()
        if len(series_clean) < 20:
            return {"tested": False, "reason": "样本量不足"}
        
        # 计算偏度和峰度（正态分布的偏度接近0，峰度接近3）
        skew = series_clean.skew()
        kurtosis = series_clean.kurtosis()
        
        # 简单判断：偏度和峰度是否接近0和3
        is_approximately_normal = abs(skew) < 1.0 and abs(kurtosis) < 3.0
        
        return {
            "tested": True,
            "skew": skew,
            "kurtosis": kurtosis,
            "is_approximately_normal": is_approximately_normal
        }
    
    def _analyze_time_series(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        分析时间序列特征
        
        Args:
            result: 因子计算结果DataFrame（索引为DatetimeIndex）
            
        Returns:
            时间序列特征字典
        """
        # 时间范围
        time_range = {
            "start": result.index.min().isoformat(),
            "end": result.index.max().isoformat(),
            "duration_days": (result.index.max() - result.index.min()).days,
            "observation_count": len(result)
        }
        
        # 频率检测
        freq_info = self._detect_frequency(result.index)
        
        # 缺失时间点检测
        missing_info = self._detect_missing_timestamps(result.index, freq_info.get("inferred_freq"))
        
        return {
            "time_range": time_range,
            "frequency": freq_info,
            "missing_data": missing_info
        }
    
    def _detect_frequency(self, index: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        检测时间序列频率
        
        Args:
            index: 时间索引
            
        Returns:
            频率信息字典
        """
        try:
            # 尝试推断频率
            inferred_freq = pd.infer_freq(index)
            
            # 计算主要时间间隔
            if len(index) > 1:
                deltas = index[1:] - index[:-1]
                main_delta = deltas.mode()[0]
                
                return {
                    "inferred_freq": inferred_freq,
                    "main_interval": str(main_delta),
                    "interval_seconds": main_delta.total_seconds(),
                    "has_regular_interval": len(set(deltas)) == 1
                }
        except Exception as e:
            self.logger.warning(f"检测频率时出错: {e}")
        
        return {"inferred_freq": None, "has_regular_interval": False}
    
    def _detect_missing_timestamps(self, 
                                  index: pd.DatetimeIndex,
                                  freq: Optional[str]) -> Dict[str, Any]:
        """
        检测缺失的时间戳
        
        Args:
            index: 时间索引
            freq: 推断的频率
            
        Returns:
            缺失信息字典
        """
        try:
            if freq and len(index) > 1:
                # 生成完整的时间范围
                full_range = pd.date_range(start=index.min(), end=index.max(), freq=freq)
                
                # 找出缺失的时间点
                missing = full_range.difference(index)
                
                return {
                    "expected_count": len(full_range),
                    "actual_count": len(index),
                    "missing_count": len(missing),
                    "missing_percentage": (len(missing) / len(full_range)) * 100 if len(full_range) > 0 else 0,
                    "missing_timestamps": [ts.isoformat() for ts in missing[:10]]  # 只返回前10个
                }
        except Exception as e:
            self.logger.warning(f"检测缺失时间戳时出错: {e}")
        
        return {"missing_count": 0, "missing_percentage": 0}
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """
        计算DataFrame的哈希值
        
        Args:
            df: 输入DataFrame
            
        Returns:
            哈希值字符串
        """
        # 使用数据的形状、索引信息和采样值计算哈希
        shape_str = str(df.shape)
        index_info = f"{df.index.min()}_{df.index.max()}" if len(df) > 0 else "empty"
        
        # 采样数据进行哈希
        sample_size = min(100, len(df))
        if sample_size > 0:
            sample = df.sample(sample_size).reset_index()
            sample_str = sample.to_json(orient="records")
        else:
            sample_str = ""
        
        combined = f"{shape_str}:{index_info}:{sample_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _analyze_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析输入数据的特征
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            数据特征字典
        """
        # 基本信息
        data_info = {
            "shape": {"rows": len(data), "columns": len(data.columns)},
            "columns": list(data.columns),
            "data_types": {col: str(data[col].dtype) for col in data.columns},
            "index_type": str(type(data.index)),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # 数据质量
        quality = {
            "null_percentage": data.isnull().mean().mean() * 100,
            "duplicate_rows": data.duplicated().sum(),
            "has_ohlcv": self._has_ohlcv_columns(data)
        }
        
        # 时间序列特征（如果适用）
        time_info = {}
        if isinstance(data.index, pd.DatetimeIndex):
            time_info = {
                "time_range": {
                    "start": data.index.min().isoformat(),
                    "end": data.index.max().isoformat(),
                    "duration_days": (data.index.max() - data.index.min()).days
                }
            }
        
        return {
            "data_info": data_info,
            "data_quality": quality,
            "time_info": time_info,
            "data_hash": self._hash_dataframe(data)
        }
    
    def _has_ohlcv_columns(self, data: pd.DataFrame) -> bool:
        """
        检查数据是否包含OHLCV列
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            是否包含OHLCV列
        """
        ohlcv_columns = {'open', 'high', 'low', 'close', 'volume'}
        lower_columns = {col.lower() for col in data.columns}
        
        # 检查是否至少包含OHLC中的三个列和成交量
        price_columns = {'open', 'high', 'low', 'close'}
        price_match = len(price_columns.intersection(lower_columns)) >= 3
        volume_match = 'volume' in lower_columns or 'vol' in lower_columns
        
        return price_match and volume_match
    
    def _collect_dependency_info(self, dependencies: Optional[List[str]]) -> Dict[str, Any]:
        """
        收集依赖关系信息
        
        Args:
            dependencies: 依赖列表
            
        Returns:
            依赖信息字典
        """
        return {
            "dependencies": dependencies or [],
            "dependency_count": len(dependencies) if dependencies else 0
        }
    
    def _collect_version_info(self) -> Dict[str, Any]:
        """
        收集版本信息
        
        Returns:
            版本信息字典
        """
        return {
            "schema_version": "1.0",
            "collector_version": "1.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    
    def _generate_unique_id(self, 
                           factor: FactorBase,
                           params: Dict[str, Any]) -> str:
        """
        生成因子的唯一标识
        
        Args:
            factor: 因子实例
            params: 参数字典
            
        Returns:
            唯一标识字符串
        """
        # 结合因子类、名称和参数生成唯一ID
        class_name = factor.__class__.__name__
        factor_name = factor.factor_name
        params_str = json.dumps(params, sort_keys=True, default=str)
        
        combined = f"{class_name}:{factor_name}:{params_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def serialize_metadata(self, metadata: Dict[str, Any], format: str = "json") -> Union[str, bytes]:
        """
        序列化元数据
        
        Args:
            metadata: 元数据字典
            format: 序列化格式（json或pickle）
            
        Returns:
            序列化后的数据
        """
        if format == "json":
            # JSON序列化（处理numpy类型）
            def default_encoder(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                raise TypeError(f"无法序列化类型 {type(obj).__name__}")
            
            return json.dumps(metadata, ensure_ascii=False, indent=2, default=default_encoder)
        elif format == "pickle":
            # Pickle序列化（保留所有Python对象）
            return pickle.dumps(metadata)
        else:
            raise ValueError(f"不支持的序列化格式: {format}")
    
    def deserialize_metadata(self, serialized_data: Union[str, bytes], format: str = "json") -> Dict[str, Any]:
        """
        反序列化元数据
        
        Args:
            serialized_data: 序列化后的数据
            format: 序列化格式（json或pickle）
            
        Returns:
            元数据字典
        """
        if format == "json":
            return json.loads(serialized_data)
        elif format == "pickle":
            return pickle.loads(serialized_data)
        else:
            raise ValueError(f"不支持的反序列化格式: {format}")
    
    def compare_metadata(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两个因子的元数据
        
        Args:
            metadata1: 第一个元数据
            metadata2: 第二个元数据
            
        Returns:
            差异字典
        """
        differences = {
            "basic_info": self._compare_dicts(metadata1.get("basic_info", {}), metadata2.get("basic_info", {})),
            "params_info": self._compare_dicts(metadata1.get("params_info", {}), metadata2.get("params_info", {})),
            "performance_info": self._compare_dicts(metadata1.get("performance_info", {}), metadata2.get("performance_info", {})),
            "has_differences": False
        }
        
        # 检查是否有差异
        differences["has_differences"] = any(len(diff) > 0 for diff in differences.values() if isinstance(diff, dict))
        
        return differences
    
    def _compare_dicts(self, dict1: Dict, dict2: Dict) -> Dict[str, Dict[str, Any]]:
        """
        比较两个字典的差异
        
        Args:
            dict1: 第一个字典
            dict2: 第二个字典
            
        Returns:
            差异字典
        """
        differences = {}
        
        # 检查dict1中有但dict2中没有的键
        for key in dict1:
            if key not in dict2:
                differences[key] = {"only_in_first": dict1[key]}
        
        # 检查dict2中有但dict1中没有的键
        for key in dict2:
            if key not in dict1:
                differences[key] = {"only_in_second": dict2[key]}
        
        # 检查共同键的值差异
        for key in set(dict1.keys()) & set(dict2.keys()):
            val1 = dict1[key]
            val2 = dict2[key]
            
            # 递归比较嵌套字典
            if isinstance(val1, dict) and isinstance(val2, dict):
                nested_diff = self._compare_dicts(val1, val2)
                if nested_diff:
                    differences[key] = {"nested_differences": nested_diff}
            # 对于数值类型，允许一定的误差
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if not np.isclose(val1, val2, rtol=1e-5):
                    differences[key] = {"first": val1, "second": val2, "difference": val2 - val1}
            # 其他类型直接比较
            elif val1 != val2:
                differences[key] = {"first": val1, "second": val2}
        
        return differences
    
    def generate_summary(self, metadata: Dict[str, Any]) -> str:
        """
        生成元数据摘要
        
        Args:
            metadata: 元数据字典
            
        Returns:
            摘要字符串
        """
        basic_info = metadata.get("basic_info", {})
        params_info = metadata.get("params_info", {})
        result_stats = metadata.get("result_stats", {})
        
        summary = f"因子摘要：\n"
        summary += f"- 名称: {basic_info.get('factor_name', '未知')}\n"
        summary += f"- 类型: {basic_info.get('factor_type', '未知')}\n"
        summary += f"- 类名: {metadata.get('class_info', {}).get('full_class_name', '未知')}\n"
        summary += f"- 参数数量: {params_info.get('param_count', 0)}\n"
        
        if result_stats:
            basic_stats = result_stats.get('basic_stats', {})
            summary += f"- 结果形状: {basic_stats.get('shape', {}).get('rows', 0)}行 × {basic_stats.get('shape', {}).get('columns', 0)}列\n"
            summary += f"- 空值比例: {basic_stats.get('null_percentage', 0):.2f}%\n"
        
        performance = metadata.get('performance_info', {})
        if performance.get('computation_time'):
            summary += f"- 计算时间: {performance['computation_time']:.4f}秒\n"
        
        dependencies = metadata.get('dependency_info', {})
        summary += f"- 依赖数量: {dependencies.get('dependency_count', 0)}\n"
        
        return summary


# 添加必要的导入
import sys
import os
