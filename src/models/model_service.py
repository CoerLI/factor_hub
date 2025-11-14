import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# 数据处理
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 自定义模块
from src.utils.helpers import setup_logger, DEFAULT_LOGGER, timeit, ensure_directory, load_config
from src.utils.cache_manager import CacheManager
from src.models.data_preprocessing import DataPreprocessor, TimeSeriesDataLoader
from src.models.model_training import ModelTrainer, EnsembleModel, get_best_model
from src.models.model_deployment import ModelDeployment, ModelDeploymentPipeline, create_api_server_script

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class TimeSeriesModelService:
    """
    时间序列模型服务
    
    整合数据预处理、模型训练、评估和部署的完整服务
    """
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 data_dir: str = './data',
                 models_dir: str = './models',
                 deploy_dir: str = './deployments',
                 cache_dir: str = './cache'):
        """
        初始化时间序列模型服务
        
        Args:
            config_path: 配置文件路径
            data_dir: 数据目录
            models_dir: 模型目录
            deploy_dir: 部署目录
            cache_dir: 缓存目录
        """
        # 加载配置
        self.config = load_config(config_path) if config_path else {}
        
        # 目录设置
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.deploy_dir = deploy_dir
        self.cache_dir = cache_dir
        
        # 确保目录存在
        self._ensure_directories()
        
        # 初始化组件
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        self.preprocessor = DataPreprocessor(cache_manager=self.cache_manager)
        self.loader = TimeSeriesDataLoader(data_dir=data_dir, cache_manager=self.cache_manager)
        self.trainer = ModelTrainer(cache_manager=self.cache_manager, models_dir=models_dir)
        self.deployment = ModelDeployment(
            models_dir=models_dir,
            deploy_dir=deploy_dir,
            cache_manager=self.cache_manager
        )
        self.deployment_pipeline = ModelDeploymentPipeline(
            models_dir=models_dir,
            deploy_dir=deploy_dir,
            config=self.config
        )
        
        # 服务状态
        self.is_running = False
        self.active_pipelines = {}
        self.service_stats = {
            'started_at': datetime.now().isoformat(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        logger.info("时间序列模型服务初始化完成")
    
    def _ensure_directories(self):
        """
        确保所有必要的目录存在
        """
        for directory in [self.data_dir, self.models_dir, self.deploy_dir, self.cache_dir]:
            ensure_directory(directory)
    
    @timeit
    def train_pipeline(self,
                      data_source: Union[str, pd.DataFrame],
                      target_column: str,
                      model_config: Dict,
                      preprocessing_config: Optional[Dict] = None,
                      feature_config: Optional[Dict] = None,
                      split_config: Optional[Dict] = None,
                      auto_deploy: bool = False,
                      pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整的训练流水线
        
        Args:
            data_source: 数据源（文件路径或DataFrame）
            target_column: 目标列名
            model_config: 模型配置
            preprocessing_config: 预处理配置
            feature_config: 特征工程配置
            split_config: 数据分割配置
            auto_deploy: 是否自动部署
            pipeline_id: 流水线ID
            
        Returns:
            流水线结果
        """
        # 生成流水线ID
        if pipeline_id is None:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"开始执行训练流水线: {pipeline_id}")
        
        # 更新服务统计
        self.service_stats['total_requests'] += 1
        
        # 初始化结果
        results = {
            'pipeline_id': pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'steps': [],
            'status': 'in_progress'
        }
        
        # 记录活跃流水线
        self.active_pipelines[pipeline_id] = results
        
        try:
            # 1. 加载数据
            logger.info(f"[{pipeline_id}] 加载数据")
            if isinstance(data_source, str):
                df = self.loader.load_data(data_source)
            else:
                df = data_source.copy()
            
            results['data_loading'] = {
                'shape': df.shape,
                'status': 'completed'
            }
            results['steps'].append('data_loading')
            
            # 2. 数据预处理
            logger.info(f"[{pipeline_id}] 数据预处理")
            if preprocessing_config is None:
                preprocessing_config = {}
            
            processed_df = self.preprocessor.preprocess(
                df,
                target_column=target_column,
                **preprocessing_config
            )
            
            results['preprocessing'] = {
                'shape': processed_df.shape,
                'config': preprocessing_config,
                'status': 'completed'
            }
            results['steps'].append('preprocessing')
            
            # 3. 特征工程
            logger.info(f"[{pipeline_id}] 特征工程")
            if feature_config is None:
                feature_config = {}
            
            # 创建特征
            X, y = self.preprocessor.create_features(
                processed_df,
                target_column=target_column,
                **feature_config
            )
            
            results['feature_engineering'] = {
                'features_count': X.shape[1],
                'samples_count': X.shape[0],
                'config': feature_config,
                'status': 'completed'
            }
            results['steps'].append('feature_engineering')
            
            # 4. 数据分割
            logger.info(f"[{pipeline_id}] 数据分割")
            if split_config is None:
                split_config = {
                    'test_size': 0.2,
                    'valid_size': 0.1,
                    'shuffle': False
                }
            
            # 分割数据
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
                X, y, **split_config
            )
            
            results['data_splitting'] = {
                'train_size': X_train.shape[0],
                'val_size': X_val.shape[0],
                'test_size': X_test.shape[0],
                'config': split_config,
                'status': 'completed'
            }
            results['steps'].append('data_splitting')
            
            # 5. 特征缩放/标准化
            logger.info(f"[{pipeline_id}] 特征缩放")
            X_train_scaled, X_val_scaled, X_test_scaled = self.preprocessor.scale_features(
                X_train, X_val, X_test
            )
            
            results['feature_scaling'] = {
                'status': 'completed'
            }
            results['steps'].append('feature_scaling')
            
            # 6. 模型训练
            logger.info(f"[{pipeline_id}] 模型训练")
            model_type = model_config.get('type', 'xgboost')
            model_params = model_config.get('params', {})
            model_name = model_config.get('name', f"{model_type}_model")
            
            # 检查是否需要超参数调优
            perform_hyperparam_tuning = model_config.get('hyperparam_tuning', False)
            param_grid = model_config.get('param_grid', {})
            
            # 训练模型
            if perform_hyperparam_tuning and param_grid:
                # 超参数调优
                tuning_results = self.trainer.hyperparameter_tuning(
                    X_train_scaled, y_train,
                    param_grid=param_grid,
                    model_type=model_type
                )
                best_params = tuning_results['best_params']
                model = tuning_results['model']
            else:
                # 直接训练
                model = self.trainer.create_model(model_type, model_params)
                model = self.trainer.train(
                    model=model,
                    X_train=X_train_scaled,
                    y_train=y_train,
                    X_val=X_val_scaled,
                    y_val=y_val
                )
                best_params = model_params
            
            # 评估模型
            train_metrics = self.trainer.evaluate(X_train_scaled, y_train)
            val_metrics = self.trainer.evaluate(X_val_scaled, y_val)
            test_metrics = self.trainer.evaluate(X_test_scaled, y_test)
            
            results['model_training'] = {
                'model_type': model_type,
                'model_name': model_name,
                'best_params': best_params,
                'metrics': {
                    'train': train_metrics,
                    'validation': val_metrics,
                    'test': test_metrics
                },
                'status': 'completed'
            }
            results['steps'].append('model_training')
            
            # 7. 保存模型
            logger.info(f"[{pipeline_id}] 保存模型")
            model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_metadata = {
                'model_type': model_type,
                'parameters': best_params,
                'training_metrics': test_metrics,
                'preprocessing_config': preprocessing_config,
                'feature_config': feature_config,
                'training_date': datetime.now().isoformat()
            }
            
            model_path = self.trainer.save_model(
                filename=f"{model_name}_v{model_version}",
                metadata=model_metadata
            )
            
            results['model_saving'] = {
                'model_path': model_path,
                'version': model_version,
                'status': 'completed'
            }
            results['steps'].append('model_saving')
            
            # 8. 部署模型（可选）
            if auto_deploy:
                logger.info(f"[{pipeline_id}] 部署模型")
                deployment_info = self.deployment.deploy_model(
                    model_path=model_path,
                    model_name=model_name,
                    version=model_version,
                    metadata=model_metadata
                )
                
                results['deployment'] = deployment_info
                results['steps'].append('deployment')
            
            # 更新状态
            results['status'] = 'success'
            self.service_stats['successful_requests'] += 1
            
            logger.info(f"[{pipeline_id}] 训练流水线执行成功")
            
        except Exception as e:
            logger.error(f"[{pipeline_id}] 训练流水线执行失败: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            self.service_stats['failed_requests'] += 1
        
        # 更新活跃流水线
        if pipeline_id in self.active_pipelines:
            del self.active_pipelines[pipeline_id]
        
        # 保存结果到日志
        self._log_pipeline_results(results)
        
        return results
    
    def _log_pipeline_results(self, results: Dict):
        """
        记录流水线结果
        
        Args:
            results: 流水线结果
        """
        # 保存到日志文件
        logs_dir = os.path.join(self.models_dir, 'logs')
        ensure_directory(logs_dir)
        
        log_file = os.path.join(logs_dir, f"pipeline_{results['pipeline_id']}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"流水线结果已保存到: {log_file}")
    
    @timeit
    def predict_with_model(self,
                          model_name: str,
                          version: str,
                          data: Union[pd.DataFrame, List[Dict]],
                          target_column: Optional[str] = None,
                          apply_preprocessing: bool = True,
                          preprocessing_config: Optional[Dict] = None) -> np.ndarray:
        """
        使用已部署的模型进行预测
        
        Args:
            model_name: 模型名称
            version: 模型版本
            data: 输入数据
            target_column: 目标列名（用于预处理）
            apply_preprocessing: 是否应用预处理
            preprocessing_config: 预处理配置
            
        Returns:
            预测结果
        """
        logger.info(f"使用模型 {model_name} v{version} 进行预测")
        
        # 更新服务统计
        self.service_stats['total_requests'] += 1
        
        try:
            # 准备数据
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # 应用预处理
            if apply_preprocessing:
                if preprocessing_config is None:
                    preprocessing_config = {}
                
                # 加载模型元数据以获取预处理配置
                model_key = f"{model_name}_v{version}"
                if model_key in self.deployment.active_models:
                    model_metadata = self.deployment.active_models[model_key]['metadata']
                    saved_preprocessing_config = model_metadata.get('preprocessing_config', {})
                    saved_feature_config = model_metadata.get('feature_config', {})
                    
                    # 合并配置
                    preprocessing_config = {**saved_preprocessing_config, **preprocessing_config}
                    
                    # 预处理
                    processed_df = self.preprocessor.preprocess(
                        df,
                        target_column=target_column,
                        **preprocessing_config
                    )
                    
                    # 创建特征
                    X, _ = self.preprocessor.create_features(
                        processed_df,
                        target_column=target_column,
                        **saved_feature_config
                    )
                    
                    # 缩放特征
                    # 注意：需要使用与训练时相同的缩放器
                    # 这里简化处理，实际应用中需要加载保存的缩放器
                    X_scaled = self.preprocessor.scale_features(X)
                    
                    # 预测
                    predictions = self.deployment.predict_with_model(
                        model_name,
                        version,
                        X_scaled
                    )
                else:
                    # 模型未加载，直接使用原始数据格式
                    predictions = self.deployment.predict_with_model(
                        model_name,
                        version,
                        data
                    )
            else:
                # 不进行预处理，直接预测
                predictions = self.deployment.predict_with_model(
                    model_name,
                    version,
                    data
                )
            
            # 更新统计
            self.service_stats['successful_requests'] += 1
            
            return predictions
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            self.service_stats['failed_requests'] += 1
            raise
    
    def start_model_service(self,
                           host: str = "0.0.0.0",
                           port: int = 8000,
                           enable_monitoring: bool = True,
                           monitoring_port: int = 8001) -> threading.Thread:
        """
        启动模型服务API
        
        Args:
            host: 主机地址
            port: 端口
            enable_monitoring: 是否启用监控
            monitoring_port: 监控端口
            
        Returns:
            服务线程
        """
        logger.info(f"启动模型服务API在 http://{host}:{port}")
        
        # 创建API服务器脚本
        api_script_path = create_api_server_script()
        
        # 启动部署服务
        server_thread = self.deployment.start_api_server(
            host=host,
            port=port,
            enable_monitoring=enable_monitoring,
            monitoring_port=monitoring_port
        )
        
        # 更新服务状态
        self.is_running = True
        
        return server_thread
    
    def stop_model_service(self):
        """
        停止模型服务API
        """
        if self.is_running:
            logger.info("停止模型服务API")
            self.deployment.stop_api_server()
            self.is_running = False
        else:
            logger.warning("模型服务未运行")
    
    def compare_models(self,
                      X_train: Union[pd.DataFrame, np.ndarray],
                      y_train: Union[pd.Series, np.ndarray],
                      X_val: Union[pd.DataFrame, np.ndarray],
                      y_val: Union[pd.Series, np.ndarray],
                      model_configs: List[Dict],
                      metric: str = 'rmse') -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            model_configs: 模型配置列表
            metric: 评估指标
            
        Returns:
            比较结果
        """
        logger.info(f"比较 {len(model_configs)} 个模型")
        
        comparisons = []
        best_score = float('inf') if metric in ['rmse', 'mae'] else -float('inf')
        best_model = None
        best_config = None
        
        for config in model_configs:
            model_type = config.get('type', 'xgboost')
            model_params = config.get('params', {})
            model_name = config.get('name', f"{model_type}_model")
            
            try:
                # 训练模型
                model = self.trainer.create_model(model_type, model_params)
                model = self.trainer.train(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val
                )
                
                # 评估模型
                metrics = self.trainer.evaluate(X_val, y_val)
                
                # 记录结果
                comparison = {
                    'model_type': model_type,
                    'model_name': model_name,
                    'params': model_params,
                    'metrics': metrics
                }
                comparisons.append(comparison)
                
                # 更新最佳模型
                current_score = metrics[metric]
                if (metric in ['rmse', 'mae'] and current_score < best_score) or \
                   (metric == 'r2' and current_score > best_score):
                    best_score = current_score
                    best_model = model
                    best_config = config
                
            except Exception as e:
                logger.error(f"模型 {model_type} 比较失败: {str(e)}")
                comparisons.append({
                    'model_type': model_type,
                    'model_name': model_name,
                    'error': str(e),
                    'metrics': {}
                })
        
        results = {
            'comparisons': comparisons,
            'best_model': {
                'type': best_config.get('type') if best_config else None,
                'name': best_config.get('name') if best_config else None,
                'score': best_score
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"模型比较完成，最佳模型: {results['best_model']['name']}")
        
        return results
    
    def create_ensemble(self,
                       model_configs: List[Dict],
                       weights: Optional[List[float]] = None) -> EnsembleModel:
        """
        创建集成模型
        
        Args:
            model_configs: 模型配置列表
            weights: 模型权重列表
            
        Returns:
            集成模型
        """
        logger.info(f"创建集成模型，包含 {len(model_configs)} 个子模型")
        
        ensemble = EnsembleModel()
        
        # 如果未提供权重，使用均匀权重
        if weights is None:
            weights = [1.0 / len(model_configs)] * len(model_configs)
        
        # 添加每个模型
        for i, config in enumerate(model_configs):
            model_type = config.get('type', 'xgboost')
            model_params = config.get('params', {})
            model_name = config.get('name', f"{model_type}_model_{i}")
            weight = weights[i]
            
            # 创建模型
            model = self.trainer.create_model(model_type, model_params)
            
            # 添加到集成
            ensemble.add_model(model, model_name, weight)
        
        logger.info("集成模型创建完成")
        
        return ensemble
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        获取服务状态
        
        Returns:
            服务状态信息
        """
        status = {
            'is_running': self.is_running,
            'uptime': (datetime.now() - datetime.fromisoformat(self.service_stats['started_at'])).total_seconds(),
            'active_pipelines': len(self.active_pipelines),
            'active_models': list(self.deployment.active_models.keys()),
            'statistics': self.service_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def monitor_models(self,
                      interval: int = 3600,
                      alert_thresholds: Optional[Dict[str, float]] = None) -> threading.Thread:
        """
        启动模型监控
        
        Args:
            interval: 监控间隔（秒）
            alert_thresholds: 告警阈值
            
        Returns:
            监控线程
        """
        logger.info(f"启动模型监控，间隔 {interval} 秒")
        
        def monitor_loop():
            while self.is_running:
                try:
                    self._check_model_performance(alert_thresholds)
                    self._cleanup_old_models()
                    self._generate_monitoring_report()
                except Exception as e:
                    logger.error(f"监控循环错误: {str(e)}")
                
                # 等待下一次检查
                time.sleep(interval)
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread
    
    def _check_model_performance(self, alert_thresholds: Optional[Dict[str, float]] = None):
        """
        检查模型性能
        
        Args:
            alert_thresholds: 告警阈值
        """
        # 这里实现模型性能检查逻辑
        # 例如检查模型漂移、预测延迟等
        pass
    
    def _cleanup_old_models(self):
        """
        清理旧模型
        """
        # 这里实现模型清理逻辑
        # 例如删除未使用的旧版本模型
        pass
    
    def _generate_monitoring_report(self):
        """
        生成监控报告
        """
        # 生成模型监控报告
        # 可以保存到日志或发送邮件
        pass
    
    def save_service_config(self, output_path: str = './config/service_config.json'):
        """
        保存服务配置
        
        Args:
            output_path: 输出路径
            
        Returns:
            配置路径
        """
        config = {
            'data_dir': self.data_dir,
            'models_dir': self.models_dir,
            'deploy_dir': self.deploy_dir,
            'cache_dir': self.cache_dir,
            'service_stats': self.service_stats,
            'created_at': datetime.now().isoformat()
        }
        
        # 确保目录存在
        ensure_directory(os.path.dirname(output_path))
        
        # 保存配置
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"服务配置已保存到: {output_path}")
        
        return output_path
    
    def generate_model_summary(self,
                              model_name: str,
                              version: str,
                              output_dir: str = './reports') -> str:
        """
        生成模型摘要报告
        
        Args:
            model_name: 模型名称
            version: 模型版本
            output_dir: 输出目录
            
        Returns:
            报告路径
        """
        # 确保目录存在
        ensure_directory(output_dir)
        
        # 获取模型信息
        model_key = f"{model_name}_v{version}"
        if model_key not in self.deployment.active_models:
            raise ValueError(f"模型未加载: {model_key}")
        
        model_info = self.deployment.active_models[model_key]
        metadata = model_info['metadata']
        
        # 创建摘要内容
        summary = {
            'report_title': '模型摘要报告',
            'generation_time': datetime.now().isoformat(),
            'model_info': {
                'name': model_name,
                'version': version,
                'type': metadata.get('model_type'),
                'parameters': metadata.get('parameters', {})
            },
            'performance_metrics': metadata.get('training_metrics', {}),
            'preprocessing_info': metadata.get('preprocessing_config', {}),
            'feature_info': metadata.get('feature_config', {}),
            'deployment_info': {
                'deployed_at': metadata.get('deployment_time'),
                'path': metadata.get('deployment_path')
            }
        }
        
        # 保存摘要
        report_filename = f"model_summary_{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"模型摘要报告已生成: {report_path}")
        
        return report_path


class ModelServiceClient:
    """
    模型服务客户端
    
    用于与模型服务API交互的客户端
    """
    
    def __init__(self,
                 base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: API基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = self._create_session()
        
        logger.info(f"初始化模型服务客户端，连接到: {base_url}")
    
    def _create_session(self):
        """
        创建HTTP会话
        
        Returns:
            HTTP会话
        """
        import requests
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        return session
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态
        """
        endpoint = f"{self.base_url}/health"
        response = self.session.get(endpoint)
        response.raise_for_status()
        
        return response.json()
    
    def list_models(self) -> List[Dict[str, str]]:
        """
        列出可用模型
        
        Returns:
            模型列表
        """
        endpoint = f"{self.base_url}/models"
        response = self.session.get(endpoint)
        response.raise_for_status()
        
        return response.json().get('models', [])
    
    def predict(self,
               model_name: str,
               version: str,
               data: List[Dict]) -> List[float]:
        """
        发送预测请求
        
        Args:
            model_name: 模型名称
            version: 模型版本
            data: 输入数据
            
        Returns:
            预测结果
        """
        endpoint = f"{self.base_url}/predict"
        payload = {
            'model_name': model_name,
            'version': version,
            'data': data
        }
        
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        
        return response.json().get('predictions', [])
    
    def batch_predict(self,
                     model_name: str,
                     version: str,
                     datasets: List[List[Dict]]) -> List[List[float]]:
        """
        批量预测
        
        Args:
            model_name: 模型名称
            version: 模型版本
            datasets: 数据集列表
            
        Returns:
            批量预测结果
        """
        endpoint = f"{self.base_url}/batch_predict"
        payload = {
            'model_name': model_name,
            'version': version,
            'datasets': datasets
        }
        
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        
        return response.json().get('batch_predictions', [])
    
    def explain(self,
               model_name: str,
               version: str,
               data: List[Dict]) -> Dict[str, Any]:
        """
        获取预测解释
        
        Args:
            model_name: 模型名称
            version: 模型版本
            data: 输入数据
            
        Returns:
            解释结果
        """
        endpoint = f"{self.base_url}/explain"
        payload = {
            'model_name': model_name,
            'version': version,
            'data': data
        }
        
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_metrics(self) -> str:
        """
        获取监控指标
        
        Returns:
            指标文本
        """
        endpoint = f"{self.base_url}/metrics"
        response = self.session.get(endpoint)
        response.raise_for_status()
        
        return response.text


def create_model_service(config_path: Optional[str] = None) -> TimeSeriesModelService:
    """
    创建模型服务实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        模型服务实例
    """
    logger.info("创建时间序列模型服务实例")
    
    service = TimeSeriesModelService(config_path)
    
    # 记录服务启动
    logger.info("模型服务实例创建完成")
    
    return service


def run_model_service(config_path: Optional[str] = None,
                      host: str = "0.0.0.0",
                      port: int = 8000):
    """
    运行模型服务
    
    Args:
        config_path: 配置文件路径
        host: 主机地址
        port: 端口
    """
    # 创建服务实例
    service = create_model_service(config_path)
    
    try:
        # 启动服务
        service.start_model_service(host=host, port=port)
        
        logger.info(f"模型服务已启动，API地址: http://{host}:{port}")
        logger.info(f"健康检查地址: http://{host}:{port}/health")
        logger.info(f"模型列表地址: http://{host}:{port}/models")
        
        # 保持服务运行
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("正在停止模型服务...")
        service.stop_model_service()
        logger.info("模型服务已停止")
    
    except Exception as e:
        logger.error(f"模型服务异常: {str(e)}")
        service.stop_model_service()
        raise


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="时间序列模型服务")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    
    args = parser.parse_args()
    
    run_model_service(
        config_path=args.config,
        host=args.host,
        port=args.port
    )