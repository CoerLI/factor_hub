import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime


class ModelBase(ABC):
    """
    模型基类，为所有机器学习模型提供统一接口
    
    设计理念：
    1. 标准化接口：提供一致的模型训练、预测和评估方法
    2. 模型序列化：支持模型的保存和加载
    3. 参数调优：内置超参数优化支持
    4. 性能监控：提供模型性能评估和分析功能
    5. 可扩展性：便于集成不同类型的机器学习模型
    """
    
    def __init__(self, model_name: str, params: Optional[Dict] = None):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
            params: 模型参数
        """
        self.model_name = model_name
        self.params = params or {}
        self._model = None
        self.is_trained = False
        self.feature_names = None
        self._model_type = "base"
        self._description = "基础模型类"
        self._training_time = None
    
    @property
    def model_type(self) -> str:
        """获取模型类型"""
        return self._model_type
    
    @property
    def description(self) -> str:
        """获取模型描述"""
        return self._description
    
    @abstractmethod
    def _build_model(self) -> Any:
        """
        抽象方法：构建具体的模型实例
        
        Returns:
            模型实例
        """
        pass
    
    @abstractmethod
    def _train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        抽象方法：训练模型
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            训练后的模型
        """
        pass
    
    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        抽象方法：模型预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
        """
        训练模型的公共接口
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 额外的训练参数
            
        Returns:
            训练后的模型
        """
        # 记录特征名称
        self.feature_names = X.columns.tolist()
        
        # 记录训练开始时间
        start_time = datetime.now()
        
        # 构建并训练模型
        self._model = self._build_model()
        self._model = self._train(X, y, **kwargs)
        
        # 记录训练结束时间
        end_time = datetime.now()
        self._training_time = (end_time - start_time).total_seconds()
        
        # 更新训练状态
        self.is_trained = True
        
        return self._model
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        模型预测的公共接口
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果Series
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 确保特征列与训练时一致
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"缺少必要的特征: {missing_features}")
            
            # 确保特征顺序一致
            X = X[self.feature_names]
        
        # 执行预测
        predictions = self._predict(X)
        
        # 返回与输入索引对应的Series
        return pd.Series(predictions, index=X.index)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征数据
            y: 真实目标值
            
        Returns:
            评估指标字典
        """
        # 进行预测
        y_pred = self.predict(X)
        
        # 计算评估指标
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred)
        }
        
        return metrics
    
    def save(self, file_path: str):
        """
        保存模型
        
        Args:
            file_path: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 创建保存目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存模型状态
        model_state = {
            "model_name": self.model_name,
            "params": self.params,
            "model": self._model,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "model_type": self._model_type,
            "training_time": self._training_time,
            "save_time": datetime.now().isoformat()
        }
        
        joblib.dump(model_state, file_path)
    
    def load(self, file_path: str):
        """
        加载模型
        
        Args:
            file_path: 加载路径
        """
        # 加载模型状态
        model_state = joblib.load(file_path)
        
        # 恢复模型属性
        self.model_name = model_state["model_name"]
        self.params = model_state["params"]
        self._model = model_state["model"]
        self.is_trained = model_state["is_trained"]
        self.feature_names = model_state["feature_names"]
        self._model_type = model_state["model_type"]
        self._training_time = model_state.get("training_time")
    
    def get_params(self) -> Dict:
        """
        获取模型参数
        
        Returns:
            参数字典
        """
        return self.params.copy()
    
    def set_params(self, params: Dict):
        """
        更新模型参数
        
        Args:
            params: 新的参数字典
        """
        self.params.update(params)
        # 更新参数后需要重新训练模型
        self.is_trained = False
        self._model = None
    
    def feature_importance(self) -> Optional[pd.Series]:
        """
        获取特征重要性（如果模型支持）
        
        Returns:
            特征重要性Series，或None（如果模型不支持）
        """
        return None
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        获取训练信息
        
        Returns:
            训练信息字典
        """
        return {
            "model_name": self.model_name,
            "model_type": self._model_type,
            "is_trained": self.is_trained,
            "training_time_seconds": self._training_time,
            "feature_count": len(self.feature_names) if self.feature_names else 0
        }
    
    def __str__(self) -> str:
        """
        返回模型的字符串表示
        """
        status = "已训练" if self.is_trained else "未训练"
        return f"{self.model_name} ({self.model_type}, {status}, 参数: {self.params})"
    
    def __repr__(self) -> str:
        """
        返回模型的详细表示
        """
        return f"{self.__class__.__name__}(name='{self.model_name}', params={self.params})"


class RegressionModel(ModelBase):
    """
    回归模型基类
    """
    
    def __init__(self, model_name: str, params: Optional[Dict] = None):
        super().__init__(model_name, params)
        self._model_type = "regression"
        self._description = "回归模型，用于预测连续值"


class ClassificationModel(ModelBase):
    """
    分类模型基类
    """
    
    def __init__(self, model_name: str, params: Optional[Dict] = None):
        super().__init__(model_name, params)
        self._model_type = "classification"
        self._description = "分类模型，用于预测离散类别"
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            类别概率DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 确保特征列与训练时一致
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"缺少必要的特征: {missing_features}")
            
            # 确保特征顺序一致
            X = X[self.feature_names]
        
        # 执行概率预测
        probabilities = self._predict_proba(X)
        
        # 返回与输入索引对应的DataFrame
        return pd.DataFrame(probabilities, index=X.index)
    
    @abstractmethod
    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        抽象方法：预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            类别概率数组
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估分类模型性能
        
        Args:
            X: 特征数据
            y: 真实目标值
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 进行预测
        y_pred = self.predict(X)
        
        # 计算评估指标
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted'),
            "recall": recall_score(y, y_pred, average='weighted'),
            "f1": f1_score(y, y_pred, average='weighted')
        }
        
        return metrics
