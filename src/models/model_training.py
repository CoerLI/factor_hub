import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import logging
import joblib
import json
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tqdm import tqdm

# 深度学习相关导入（可选）
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Bidirectional, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam, RMSprop
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow未安装，无法使用深度学习模型")
    DEEP_LEARNING_AVAILABLE = False

from src.utils.helpers import setup_logger, DEFAULT_LOGGER, timeit, ensure_directory
from src.utils.cache_manager import CacheManager

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class ModelTrainer:
    """
    模型训练器
    
    负责训练、评估和管理各种时间序列预测模型
    """
    
    def __init__(self, 
                 cache_manager: Optional[CacheManager] = None,
                 models_dir: str = './models'):
        """
        初始化模型训练器
        
        Args:
            cache_manager: 缓存管理器
            models_dir: 模型保存目录
        """
        self.cache_manager = cache_manager or CacheManager()
        self.models_dir = models_dir
        self.model = None
        self.model_type = None
        self.best_params = None
        self.train_history = {}
        self.ensure_models_directory()
    
    def ensure_models_directory(self):
        """
        确保模型保存目录存在
        """
        ensure_directory(self.models_dir)
    
    def create_model(self, 
                    model_type: str,
                    params: Optional[Dict] = None) -> Any:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型
            params: 模型参数
            
        Returns:
            模型实例
        """
        logger.info(f"创建模型: {model_type}")
        self.model_type = model_type
        
        if params is None:
            params = {}
        
        # 创建不同类型的模型
        if model_type == 'linear_regression':
            model = LinearRegression(**params)
        elif model_type == 'ridge':
            model = Ridge(**params)
        elif model_type == 'lasso':
            model = Lasso(**params)
        elif model_type == 'elastic_net':
            model = ElasticNet(**params)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(**params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**params)
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(**params)
        elif model_type == 'lightgbm':
            model = lgb.LGBMRegressor(**params)
        elif model_type == 'catboost':
            model = cb.CatBoostRegressor(**params, verbose=False)
        elif model_type == 'svr':
            model = SVR(**params)
        elif model_type == 'knn':
            model = KNeighborsRegressor(**params)
        elif model_type.startswith('lstm') or model_type.startswith('gru') or model_type.startswith('cnn'):
            if not DEEP_LEARNING_AVAILABLE:
                raise ImportError("TensorFlow未安装，无法创建深度学习模型")
            model = self._create_dl_model(model_type, **params)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.model = model
        return model
    
    def _create_dl_model(self, 
                        model_type: str,
                        input_shape: Tuple = None,
                        units: List[int] = None,
                        dropout_rate: float = 0.2,
                        learning_rate: float = 0.001,
                        **kwargs) -> tf.keras.Model:
        """
        创建深度学习模型
        
        Args:
            model_type: 模型类型
            input_shape: 输入形状
            units: 各层单元数
            dropout_rate: Dropout比率
            learning_rate: 学习率
            
        Returns:
            Keras模型
        """
        if input_shape is None:
            raise ValueError("深度学习模型需要指定input_shape")
        
        if units is None:
            units = [64, 32]
        
        model = Sequential()
        
        # 添加输入层
        model.add(Input(shape=input_shape))
        
        # 根据模型类型构建不同的网络结构
        if model_type == 'lstm':
            # 基本LSTM
            for i, unit in enumerate(units):
                if i == len(units) - 1:
                    model.add(LSTM(unit, **kwargs))
                else:
                    model.add(LSTM(unit, return_sequences=True, **kwargs))
                model.add(Dropout(dropout_rate))
        
        elif model_type == 'bidirectional_lstm':
            # 双向LSTM
            for i, unit in enumerate(units):
                if i == len(units) - 1:
                    model.add(Bidirectional(LSTM(unit), **kwargs))
                else:
                    model.add(Bidirectional(LSTM(unit, return_sequences=True), **kwargs))
                model.add(Dropout(dropout_rate))
        
        elif model_type == 'gru':
            # 基本GRU
            for i, unit in enumerate(units):
                if i == len(units) - 1:
                    model.add(GRU(unit, **kwargs))
                else:
                    model.add(GRU(unit, return_sequences=True, **kwargs))
                model.add(Dropout(dropout_rate))
        
        elif model_type == 'cnn_lstm':
            # CNN-LSTM混合模型
            # CNN部分
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            # LSTM部分
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(32))
        
        elif model_type == 'cnn':
            # CNN模型
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            for unit in units:
                model.add(Dense(unit, activation='relu'))
                model.add(Dropout(dropout_rate))
        
        else:
            # 简单的全连接网络
            for unit in units:
                model.add(Dense(unit, activation='relu'))
                model.add(Dropout(dropout_rate))
        
        # 输出层
        model.add(Dense(1))
        
        # 编译模型
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    @timeit
    def train(self, 
             X_train: Union[pd.DataFrame, np.ndarray],
             y_train: Union[pd.Series, np.ndarray],
             X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
             y_val: Optional[Union[pd.Series, np.ndarray]] = None,
             model: Optional[Any] = None,
             model_type: Optional[str] = None,
             params: Optional[Dict] = None,
             **kwargs) -> Any:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            model: 预创建的模型实例（可选）
            model_type: 模型类型（如果未提供模型实例）
            params: 模型参数
            **kwargs: 额外的训练参数
            
        Returns:
            训练好的模型
        """
        # 如果没有提供模型，创建新模型
        if model is None:
            if model_type is None:
                raise ValueError("必须提供model_type或model实例")
            self.create_model(model_type, params)
        else:
            self.model = model
            self.model_type = model_type or type(model).__name__
        
        logger.info(f"开始训练模型: {self.model_type}")
        
        # 准备训练数据
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_array = y_train.values.ravel() if isinstance(y_train, pd.Series) else y_train.ravel()
        
        val_data = None
        if X_val is not None and y_val is not None:
            X_val_array = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_array = y_val.values.ravel() if isinstance(y_val, pd.Series) else y_val.ravel()
            val_data = (X_val_array, y_val_array)
        
        # 区分深度学习模型和传统机器学习模型的训练
        if hasattr(self.model, 'fit'):
            # 对于深度学习模型的特殊处理
            if DEEP_LEARNING_AVAILABLE and isinstance(self.model, tf.keras.Model):
                # 准备回调
                callbacks = []
                
                # 早停回调
                if kwargs.get('early_stopping', True):
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=kwargs.get('patience', 10),
                        restore_best_weights=True
                    )
                    callbacks.append(early_stopping)
                
                # 模型检查点
                if kwargs.get('save_checkpoint', False):
                    checkpoint_path = os.path.join(
                        self.models_dir,
                        f"{self.model_type}_checkpoint.h5"
                    )
                    model_checkpoint = ModelCheckpoint(
                        checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        mode='min'
                    )
                    callbacks.append(model_checkpoint)
                
                # 学习率调整
                if kwargs.get('reduce_lr', False):
                    reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                    callbacks.append(reduce_lr)
                
                # 训练模型
                history = self.model.fit(
                    X_train_array,
                    y_train_array,
                    validation_data=val_data,
                    epochs=kwargs.get('epochs', 100),
                    batch_size=kwargs.get('batch_size', 32),
                    callbacks=callbacks,
                    verbose=kwargs.get('verbose', 1)
                )
                
                # 保存训练历史
                self.train_history = history.history
            else:
                # 传统机器学习模型
                if val_data is not None:
                    # 对于支持验证集的模型
                    if 'eval_set' in self._get_fit_params():
                        # XGBoost, LightGBM, CatBoost等
                        eval_set = [(X_train_array, y_train_array), (val_data[0], val_data[1])]
                        self.model.fit(
                            X_train_array,
                            y_train_array,
                            eval_set=eval_set,
                            **self._filter_kwargs(kwargs)
                        )
                    else:
                        # 其他模型直接训练
                        self.model.fit(X_train_array, y_train_array)
                else:
                    self.model.fit(X_train_array, y_train_array)
        
        logger.info(f"模型训练完成: {self.model_type}")
        
        return self.model
    
    def _get_fit_params(self) -> List[str]:
        """
        获取模型fit方法支持的参数列表
        
        Returns:
            参数名称列表
        """
        # 这里简单实现，实际项目中可以使用inspect模块获取
        if isinstance(self.model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            return ['eval_set', 'eval_metric', 'early_stopping_rounds', 'verbose']
        elif isinstance(self.model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
            return ['eval_set', 'eval_metric', 'early_stopping_rounds', 'verbose']
        elif isinstance(self.model, cb.CatBoostRegressor):
            return ['eval_set', 'early_stopping_rounds', 'verbose']
        return []
    
    def _filter_kwargs(self, kwargs: Dict) -> Dict:
        """
        过滤kwargs，只保留模型支持的参数
        
        Args:
            kwargs: 原始参数
            
        Returns:
            过滤后的参数
        """
        supported_params = self._get_fit_params()
        return {k: v for k, v in kwargs.items() if k in supported_params}
    
    def evaluate(self, 
                X_test: Union[pd.DataFrame, np.ndarray],
                y_test: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        logger.info(f"评估模型: {self.model_type}")
        
        # 准备测试数据
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test_array = y_test.values.ravel() if isinstance(y_test, pd.Series) else y_test.ravel()
        
        # 预测
        y_pred = self.predict(X_test_array)
        
        # 计算评估指标
        metrics = {
            'mse': mean_squared_error(y_test_array, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_array, y_pred)),
            'mae': mean_absolute_error(y_test_array, y_pred),
            'r2': r2_score(y_test_array, y_pred)
        }
        
        # 计算MAPE (避免除零)
        non_zero_mask = y_test_array != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((y_test_array[non_zero_mask] - y_pred[non_zero_mask]) / y_test_array[non_zero_mask])) * 100
            metrics['mape'] = mape
        
        # 计算Directional Accuracy
        y_test_diff = y_test_array[1:] - y_test_array[:-1]
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        directional_accuracy = np.mean((y_test_diff * y_pred_diff) > 0) * 100
        metrics['directional_accuracy'] = directional_accuracy
        
        logger.info(f"模型评估完成: {metrics}")
        
        return metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 准备输入数据
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # 预测
        try:
            if DEEP_LEARNING_AVAILABLE and isinstance(self.model, tf.keras.Model):
                y_pred = self.model.predict(X_array).flatten()
            else:
                y_pred = self.model.predict(X_array)
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise
        
        return y_pred
    
    def hyperparameter_tuning(self,
                            X_train: Union[pd.DataFrame, np.ndarray],
                            y_train: Union[pd.Series, np.ndarray],
                            param_grid: Dict[str, List],
                            model_type: str,
                            cv: int = 5,
                            scoring: str = 'neg_mean_squared_error',
                            n_jobs: int = -1,
                            **kwargs) -> Dict[str, Any]:
        """
        超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            param_grid: 参数网格
            model_type: 模型类型
            cv: 交叉验证折数
            scoring: 评分指标
            n_jobs: 并行作业数
            
        Returns:
            调优结果
        """
        logger.info(f"开始超参数调优: {model_type}")
        
        # 对于时间序列数据，使用TimeSeriesSplit
        if kwargs.get('time_series_split', True):
            cv_strategy = TimeSeriesSplit(n_splits=cv)
        else:
            cv_strategy = cv
        
        # 创建基础模型
        base_model = self.create_model(model_type, {})
        
        # 对于深度学习模型，使用简单的网格搜索（这里简化处理）
        if DEEP_LEARNING_AVAILABLE and isinstance(base_model, tf.keras.Model):
            # 这里可以实现更复杂的超参数搜索策略
            logger.warning("深度学习模型的超参数调优未完全实现")
            best_params = param_grid
        else:
            # 使用GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=2
            )
            
            # 执行网格搜索
            grid_search.fit(X_train, y_train)
            
            # 获取最佳参数
            best_params = grid_search.best_params_
            logger.info(f"最佳参数: {best_params}")
            logger.info(f"最佳分数: {grid_search.best_score_}")
        
        self.best_params = best_params
        
        # 使用最佳参数创建新模型
        self.model = self.create_model(model_type, best_params)
        
        return {
            'best_params': best_params,
            'model': self.model
        }
    
    def save_model(self, 
                  filename: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> str:
        """
        保存模型
        
        Args:
            filename: 文件名
            metadata: 模型元数据
            
        Returns:
            保存路径
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.model_type}_{timestamp}"
        
        # 确保目录存在
        self.ensure_models_directory()
        
        # 保存模型
        model_path = os.path.join(self.models_dir, f"{filename}.joblib")
        
        try:
            # 对于不同类型的模型使用不同的保存方法
            if DEEP_LEARNING_AVAILABLE and isinstance(self.model, tf.keras.Model):
                # 保存Keras模型
                model_path = os.path.join(self.models_dir, f"{filename}.h5")
                self.model.save(model_path)
            else:
                # 保存其他模型
                joblib.dump(self.model, model_path)
            
            logger.info(f"模型已保存到: {model_path}")
            
            # 保存元数据
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'model_type': self.model_type,
                'saved_at': datetime.now().isoformat(),
                'best_params': self.best_params
            })
            
            metadata_path = os.path.join(self.models_dir, f"{filename}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"模型元数据已保存到: {metadata_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise
    
    def load_model(self, 
                  model_path: str,
                  load_metadata: bool = True) -> Any:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            load_metadata: 是否加载元数据
            
        Returns:
            加载的模型
        """
        logger.info(f"从 {model_path} 加载模型")
        
        try:
            # 根据文件扩展名确定加载方法
            if model_path.endswith('.h5') and DEEP_LEARNING_AVAILABLE:
                # 加载Keras模型
                self.model = tf.keras.models.load_model(model_path)
                self.model_type = 'keras_model'
            else:
                # 加载其他模型
                self.model = joblib.load(model_path)
                # 尝试推断模型类型
                self.model_type = type(self.model).__name__
            
            logger.info(f"模型加载成功: {self.model_type}")
            
            # 加载元数据
            if load_metadata:
                metadata_path = model_path.replace('.joblib', '_metadata.json').replace('.h5', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    self.best_params = metadata.get('best_params')
                    if 'model_type' in metadata:
                        self.model_type = metadata['model_type']
                    logger.info(f"模型元数据已加载")
            
            return self.model
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def get_feature_importance(self,
                             feature_names: List[str],
                             top_n: int = 10) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            top_n: 显示前N个特征
            
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        importance_scores = None
        
        # 尝试从不同模型中获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest, XGBoost, LightGBM, CatBoost等
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 线性模型
            importance_scores = np.abs(self.model.coef_)
        else:
            logger.warning("当前模型不支持特征重要性计算")
            return pd.DataFrame()
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # 排序并返回前N个
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制训练历史（仅适用于深度学习模型）
        
        Args:
            figsize: 图表大小
            
        Returns:
            matplotlib Figure对象
        """
        if not self.train_history:
            logger.warning("没有训练历史可绘制")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 绘制损失
        if 'loss' in self.train_history:
            axes[0].plot(self.train_history['loss'], label='Training Loss')
            if 'val_loss' in self.train_history:
                axes[0].plot(self.train_history['val_loss'], label='Validation Loss')
            axes[0].set_title('Model Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # 绘制其他指标
        metric_names = [k for k in self.train_history.keys() if k not in ['loss', 'val_loss']]
        if metric_names:
            metric = metric_names[0]
            axes[1].plot(self.train_history[metric], label=f'Training {metric}')
            val_metric = f'val_{metric}'
            if val_metric in self.train_history:
                axes[1].plot(self.train_history[val_metric], label=f'Validation {metric}')
            axes[1].set_title(f'Model {metric}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self,
                        X_test: Union[pd.DataFrame, np.ndarray],
                        y_test: Union[pd.Series, np.ndarray],
                        figsize: Tuple[int, int] = (14, 7),
                        plot_full: bool = False,
                        plot_residuals: bool = True) -> plt.Figure:
        """
        绘制预测结果
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            figsize: 图表大小
            plot_full: 是否绘制完整序列
            plot_residuals: 是否绘制残差
            
        Returns:
            matplotlib Figure对象
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 准备数据
        if isinstance(y_test, pd.Series):
            index = y_test.index
        else:
            index = range(len(y_test))
        
        # 创建子图
        if plot_residuals:
            fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        
        # 绘制预测vs实际
        axes[0].plot(index, y_test, 'b-', label='实际值')
        axes[0].plot(index, y_pred, 'r--', label='预测值')
        axes[0].set_title(f'模型预测结果: {self.model_type}')
        axes[0].set_xlabel('时间/索引')
        axes[0].set_ylabel('值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 如果是DataFrame索引，格式化日期
        if isinstance(index, pd.DatetimeIndex):
            fig.autofmt_xdate()
        
        # 绘制残差
        if plot_residuals:
            residuals = y_test - y_pred
            axes[1].plot(index, residuals, 'g-')
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_title('预测残差')
            axes[1].set_xlabel('时间/索引')
            axes[1].set_ylabel('残差')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class EnsembleModel:
    """
    集成模型
    
    组合多个模型的预测结果
    """
    
    def __init__(self):
        """
        初始化集成模型
        """
        self.models = []
        self.weights = []
        self.model_names = []
    
    def add_model(self, 
                 model: Any,
                 name: str,
                 weight: float = 1.0) -> None:
        """
        添加模型到集成
        
        Args:
            model: 模型实例
            name: 模型名称
            weight: 模型权重
        """
        self.models.append(model)
        self.weights.append(weight)
        self.model_names.append(name)
        logger.info(f"添加模型到集成: {name}, 权重: {weight}")
    
    def set_weights(self, weights: List[float]) -> None:
        """
        设置模型权重
        
        Args:
            weights: 权重列表
        """
        if len(weights) != len(self.models):
            raise ValueError("权重数量必须与模型数量相同")
        
        self.weights = weights
        logger.info(f"更新模型权重: {weights}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        集成预测
        
        Args:
            X: 输入特征
            
        Returns:
            集成预测结果
        """
        if not self.models:
            raise ValueError("集成模型为空")
        
        # 获取所有模型的预测
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)
            else:
                raise ValueError("模型必须有predict方法")
        
        # 加权平均
        weights = np.array(self.weights) / np.sum(self.weights)  # 归一化权重
        weighted_predictions = np.zeros_like(predictions[0], dtype=float)
        
        for i, pred in enumerate(predictions):
            weighted_predictions += weights[i] * pred
        
        return weighted_predictions
    
    def evaluate(self,
                X_test: Union[pd.DataFrame, np.ndarray],
                y_test: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        评估集成模型
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估指标
        """
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"集成模型评估结果: {metrics}")
        
        return metrics


def get_best_model(X_train: Union[pd.DataFrame, np.ndarray],
                  y_train: Union[pd.Series, np.ndarray],
                  X_val: Union[pd.DataFrame, np.ndarray],
                  y_val: Union[pd.Series, np.ndarray],
                  model_types: List[str],
                  model_params: Optional[Dict[str, Dict]] = None,
                  metric: str = 'rmse') -> Tuple[Any, str, Dict[str, float]]:
    """
    比较多个模型并返回最佳模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征
        y_val: 验证目标
        model_types: 要比较的模型类型列表
        model_params: 模型参数字典 {model_type: params}
        metric: 评估指标 ('rmse', 'mae', 'r2')
        
    Returns:
        (最佳模型, 最佳模型类型, 各模型性能指标)
    """
    logger.info(f"比较模型: {model_types}")
    
    if model_params is None:
        model_params = {}
    
    trainer = ModelTrainer()
    results = {}
    best_score = float('inf') if metric in ['rmse', 'mae'] else -float('inf')
    best_model = None
    best_model_type = None
    
    # 比较每个模型
    for model_type in model_types:
        try:
            logger.info(f"训练模型: {model_type}")
            
            # 获取模型参数
            params = model_params.get(model_type, {})
            
            # 训练模型
            model = trainer.create_model(model_type, params)
            model = trainer.train(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            
            # 评估模型
            metrics = trainer.evaluate(X_val, y_val)
            results[model_type] = metrics
            
            # 更新最佳模型
            current_score = metrics[metric]
            if (metric in ['rmse', 'mae'] and current_score < best_score) or \
               (metric == 'r2' and current_score > best_score):
                best_score = current_score
                best_model = model
                best_model_type = model_type
            
            logger.info(f"模型 {model_type} 评估结果: {metrics[metric]}")
            
        except Exception as e:
            logger.error(f"训练模型 {model_type} 失败: {str(e)}")
            continue
    
    logger.info(f"最佳模型: {best_model_type}, 得分: {best_score}")
    
    return best_model, best_model_type, results


def cross_validate_model(model_type: str,
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         params: Optional[Dict] = None,
                         cv: int = 5,
                         scoring: str = 'neg_mean_squared_error',
                         time_series_split: bool = True,
                         **kwargs) -> Dict[str, float]:
    """
    交叉验证模型
    
    Args:
        model_type: 模型类型
        X: 特征数据
        y: 目标变量
        params: 模型参数
        cv: 交叉验证折数
        scoring: 评分指标
        time_series_split: 是否使用时间序列交叉验证
        
    Returns:
        交叉验证结果
    """
    from sklearn.model_selection import cross_validate
    
    logger.info(f"交叉验证模型: {model_type}")
    
    # 创建模型
    trainer = ModelTrainer()
    model = trainer.create_model(model_type, params)
    
    # 准备交叉验证策略
    if time_series_split:
        cv_strategy = TimeSeriesSplit(n_splits=cv)
    else:
        cv_strategy = cv
    
    # 执行交叉验证
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=True,
        **kwargs
    )
    
    # 计算平均分数
    results = {
        'mean_train_score': np.mean(cv_results['train_score']),
        'mean_test_score': np.mean(cv_results['test_score']),
        'std_test_score': np.std(cv_results['test_score']),
        'fit_time': np.mean(cv_results['fit_time']),
        'score_time': np.mean(cv_results['score_time'])
    }
    
    logger.info(f"交叉验证结果: {results}")
    
    return results


def save_predictions(y_pred: np.ndarray,
                    index: Union[pd.Index, List],
                    filename: str = 'predictions.csv',
                    output_dir: str = './predictions') -> str:
    """
    保存预测结果
    
    Args:
        y_pred: 预测结果
        index: 索引
        filename: 文件名
        output_dir: 输出目录
        
    Returns:
        保存路径
    """
    # 确保目录存在
    ensure_directory(output_dir)
    
    # 创建DataFrame
    pred_df = pd.DataFrame({
        'prediction': y_pred
    }, index=index)
    
    # 保存到CSV
    save_path = os.path.join(output_dir, filename)
    pred_df.to_csv(save_path)
    
    logger.info(f"预测结果已保存到: {save_path}")
    
    return save_path