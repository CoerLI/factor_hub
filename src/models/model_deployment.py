import os
import json
import pickle
import joblib
import logging
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Web服务相关
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import flask
import gunicorn
import requests
import uvicorn

# 监控相关
from prometheus_client import Counter, Histogram, Summary, Gauge, start_http_server

# 任务队列相关
try:
    import celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# 数据处理
import numpy as np
import pandas as pd

# 自定义工具
from src.utils.helpers import setup_logger, DEFAULT_LOGGER, timeit, ensure_directory, load_config
from src.utils.cache_manager import CacheManager
from src.models.model_training import ModelTrainer

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class ModelDeployment:
    """
    模型部署管理
    
    负责模型的序列化、部署、监控和服务化
    """
    
    def __init__(self,
                 config_path: str = None,
                 models_dir: str = './models',
                 deploy_dir: str = './deployments',
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化模型部署管理器
        
        Args:
            config_path: 配置文件路径
            models_dir: 模型存储目录
            deploy_dir: 部署目录
            cache_manager: 缓存管理器
        """
        self.config = load_config(config_path) if config_path else {}
        self.models_dir = models_dir
        self.deploy_dir = deploy_dir
        self.cache_manager = cache_manager or CacheManager()
        self.active_models = {}
        self.model_metrics = {}
        self.api_server = None
        self.monitoring_server = None
        
        # 确保目录存在
        ensure_directory(self.models_dir)
        ensure_directory(self.deploy_dir)
        
        # 初始化监控指标
        self._init_metrics()
    
    def _init_metrics(self):
        """
        初始化监控指标
        """
        # 请求计数
        self.request_counter = Counter(
            'model_requests_total',
            'Total number of model requests',
            ['model_name', 'endpoint']
        )
        
        # 请求延迟
        self.request_latency = Histogram(
            'model_request_latency_seconds',
            'Request latency in seconds',
            ['model_name', 'endpoint']
        )
        
        # 预测分布
        self.prediction_histogram = Histogram(
            'model_prediction_values',
            'Distribution of prediction values',
            ['model_name', 'endpoint'],
            buckets=(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        )
        
        # 错误计数
        self.error_counter = Counter(
            'model_errors_total',
            'Total number of model errors',
            ['model_name', 'endpoint', 'error_type']
        )
        
        # 模型加载状态
        self.model_loaded = Gauge(
            'model_loaded',
            'Model loaded status',
            ['model_name', 'version']
        )
        
        # 缓存命中率
        self.cache_hit_ratio = Gauge(
            'model_cache_hit_ratio',
            'Cache hit ratio',
            ['model_name']
        )
    
    @timeit
    def deploy_model(self,
                    model_path: str,
                    model_name: str,
                    version: str = None,
                    metadata: Optional[Dict] = None,
                    overwrite: bool = False) -> Dict[str, str]:
        """
        部署模型
        
        Args:
            model_path: 模型文件路径
            model_name: 模型名称
            version: 模型版本
            metadata: 模型元数据
            overwrite: 是否覆盖已有部署
            
        Returns:
            部署信息
        """
        # 验证模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 生成版本号
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 部署路径
        deployment_path = os.path.join(self.deploy_dir, f"{model_name}_v{version}")
        
        # 检查是否已存在
        if os.path.exists(deployment_path) and not overwrite:
            raise ValueError(f"模型部署已存在: {deployment_path}")
        
        # 创建部署目录
        ensure_directory(deployment_path)
        
        # 复制模型文件
        import shutil
        model_filename = os.path.basename(model_path)
        target_model_path = os.path.join(deployment_path, model_filename)
        shutil.copy2(model_path, target_model_path)
        
        # 复制元数据文件
        metadata_path = model_path.replace('.joblib', '_metadata.json').replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            target_metadata_path = os.path.join(deployment_path, os.path.basename(metadata_path))
            shutil.copy2(metadata_path, target_metadata_path)
        
        # 更新或创建元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'version': version,
            'deployment_time': datetime.now().isoformat(),
            'model_path': target_model_path,
            'original_path': model_path
        })
        
        # 保存部署信息
        deployment_info_path = os.path.join(deployment_path, 'deployment_info.json')
        with open(deployment_info_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # 加载模型到内存
        self.load_model_to_memory(model_name, version)
        
        logger.info(f"模型部署成功: {model_name} v{version}, 路径: {deployment_path}")
        
        return {
            'model_name': model_name,
            'version': version,
            'deployment_path': deployment_path,
            'model_path': target_model_path,
            'metadata_path': deployment_info_path
        }
    
    def load_model_to_memory(self,
                            model_name: str,
                            version: str) -> Any:
        """
        将模型加载到内存
        
        Args:
            model_name: 模型名称
            version: 模型版本
            
        Returns:
            加载的模型
        """
        # 检查部署是否存在
        deployment_path = os.path.join(self.deploy_dir, f"{model_name}_v{version}")
        if not os.path.exists(deployment_path):
            raise ValueError(f"部署不存在: {deployment_path}")
        
        # 加载部署信息
        deployment_info_path = os.path.join(deployment_path, 'deployment_info.json')
        if not os.path.exists(deployment_info_path):
            raise FileNotFoundError(f"部署信息不存在: {deployment_info_path}")
        
        with open(deployment_info_path, 'r', encoding='utf-8') as f:
            deployment_info = json.load(f)
        
        # 加载模型
        model_path = deployment_info['model_path']
        trainer = ModelTrainer()
        model = trainer.load_model(model_path)
        
        # 存储模型
        model_key = f"{model_name}_v{version}"
        self.active_models[model_key] = {
            'model': model,
            'metadata': deployment_info,
            'trainer': trainer,
            'loaded_time': datetime.now().isoformat()
        }
        
        # 更新监控指标
        self.model_loaded.labels(model_name=model_name, version=version).set(1)
        
        logger.info(f"模型加载到内存: {model_name} v{version}")
        
        return model
    
    def unload_model_from_memory(self,
                               model_name: str,
                               version: str) -> bool:
        """
        从内存卸载模型
        
        Args:
            model_name: 模型名称
            version: 模型版本
            
        Returns:
            是否成功卸载
        """
        model_key = f"{model_name}_v{version}"
        
        if model_key in self.active_models:
            del self.active_models[model_key]
            
            # 更新监控指标
            self.model_loaded.labels(model_name=model_name, version=version).set(0)
            
            logger.info(f"模型从内存卸载: {model_name} v{version}")
            return True
        else:
            logger.warning(f"模型未加载在内存中: {model_name} v{version}")
            return False
    
    def predict_with_model(self,
                          model_name: str,
                          version: str,
                          X: Union[pd.DataFrame, np.ndarray, List[Dict]]) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            model_name: 模型名称
            version: 模型版本
            X: 输入数据
            
        Returns:
            预测结果
        """
        model_key = f"{model_name}_v{version}"
        
        # 检查模型是否加载
        if model_key not in self.active_models:
            logger.warning(f"模型未加载，尝试加载: {model_name} v{version}")
            try:
                self.load_model_to_memory(model_name, version)
            except Exception as e:
                raise ValueError(f"无法加载模型: {str(e)}")
        
        # 获取模型
        model_info = self.active_models[model_key]
        model = model_info['model']
        trainer = model_info['trainer']
        
        # 记录请求计数
        self.request_counter.labels(model_name=model_name, endpoint='predict').inc()
        
        # 记录请求延迟
        start_time = time.time()
        
        try:
            # 处理不同格式的输入
            if isinstance(X, list) and all(isinstance(item, dict) for item in X):
                # 列表字典格式，转换为DataFrame
                X_df = pd.DataFrame(X)
                y_pred = trainer.predict(X_df)
            else:
                # 直接预测
                y_pred = trainer.predict(X)
            
            # 记录预测分布
            for pred in y_pred:
                self.prediction_histogram.labels(model_name=model_name, endpoint='predict').observe(float(pred))
            
            # 记录缓存命中率
            if hasattr(self.cache_manager, 'get_hit_ratio'):
                hit_ratio = self.cache_manager.get_hit_ratio()
                self.cache_hit_ratio.labels(model_name=model_name).set(hit_ratio)
            
            return y_pred
            
        except Exception as e:
            # 记录错误
            error_type = type(e).__name__
            self.error_counter.labels(model_name=model_name, endpoint='predict', error_type=error_type).inc()
            logger.error(f"预测失败: {str(e)}")
            raise
        finally:
            # 记录延迟
            latency = time.time() - start_time
            self.request_latency.labels(model_name=model_name, endpoint='predict').observe(latency)
    
    def create_fastapi_app(self,
                          models_dir: Optional[str] = None,
                          enable_monitoring: bool = True,
                          **kwargs) -> FastAPI:
        """
        创建FastAPI应用
        
        Args:
            models_dir: 模型目录
            enable_monitoring: 是否启用监控
            
        Returns:
            FastAPI应用实例
        """
        app = FastAPI(
            title="时间序列预测模型API",
            description="用于预测时间序列数据的RESTful API",
            version="1.0.0"
        )
        
        # 保存对self的引用
        app.state.deployment = self
        
        # 定义请求模型
        class PredictionRequest(BaseModel):
            model_name: str
            version: str
            data: List[Dict]
            
        class BatchPredictionRequest(BaseModel):
            model_name: str
            version: str
            datasets: List[List[Dict]]
        
        # 健康检查端点
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_models": list(self.active_models.keys())
            }
        
        # 模型列表端点
        @app.get("/models")
        async def list_models():
            models = []
            for key, info in self.active_models.items():
                models.append({
                    "model_name": info['metadata']['model_name'],
                    "version": info['metadata']['version'],
                    "loaded_time": info['loaded_time']
                })
            return {"models": models}
        
        # 单个预测端点
        @app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                predictions = self.predict_with_model(
                    request.model_name,
                    request.version,
                    request.data
                )
                return {
                    "model_name": request.model_name,
                    "version": request.version,
                    "predictions": predictions.tolist(),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # 批量预测端点
        @app.post("/batch_predict")
        async def batch_predict(request: BatchPredictionRequest):
            try:
                results = []
                for dataset in request.datasets:
                    predictions = self.predict_with_model(
                        request.model_name,
                        request.version,
                        dataset
                    )
                    results.append(predictions.tolist())
                
                return {
                    "model_name": request.model_name,
                    "version": request.version,
                    "batch_predictions": results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # 解释性端点
        @app.post("/explain")
        async def explain_prediction(request: PredictionRequest):
            try:
                model_key = f"{request.model_name}_v{request.version}"
                if model_key not in self.active_models:
                    raise ValueError("模型未加载")
                
                model_info = self.active_models[model_key]
                trainer = model_info['trainer']
                
                # 转换数据
                X_df = pd.DataFrame(request.data)
                
                # 获取特征重要性
                feature_importance = trainer.get_feature_importance(X_df.columns.tolist())
                
                # 预测
                predictions = trainer.predict(X_df)
                
                return {
                    "model_name": request.model_name,
                    "version": request.version,
                    "predictions": predictions.tolist(),
                    "feature_importance": feature_importance.to_dict(orient='records'),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # 监控端点
        @app.get("/metrics")
        async def get_metrics():
            return Response(
                content=generate_metrics_text(),
                media_type="text/plain"
            )
        
        # 启动监控服务器
        if enable_monitoring:
            monitoring_port = kwargs.get('monitoring_port', 8001)
            threading.Thread(
                target=self._start_monitoring_server,
                args=(monitoring_port,),
                daemon=True
            ).start()
        
        logger.info("FastAPI应用创建成功")
        return app
    
    def _start_monitoring_server(self, port: int = 8001):
        """
        启动监控服务器
        
        Args:
            port: 监控服务器端口
        """
        try:
            start_http_server(port)
            logger.info(f"监控服务器启动在端口 {port}")
        except Exception as e:
            logger.error(f"监控服务器启动失败: {str(e)}")
    
    def start_api_server(self,
                        host: str = "0.0.0.0",
                        port: int = 8000,
                        models_dir: Optional[str] = None,
                        enable_monitoring: bool = True,
                        **kwargs):
        """
        启动API服务器
        
        Args:
            host: 主机地址
            port: 端口
            models_dir: 模型目录
            enable_monitoring: 是否启用监控
        """
        # 创建FastAPI应用
        app = self.create_fastapi_app(
            models_dir=models_dir,
            enable_monitoring=enable_monitoring,
            **kwargs
        )
        
        # 保存服务器配置
        self.api_config = {
            'host': host,
            'port': port,
            'models_dir': models_dir,
            'enable_monitoring': enable_monitoring
        }
        
        logger.info(f"启动API服务器在 http://{host}:{port}")
        
        # 启动服务器（异步）
        def run_server():
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info"
            )
        
        # 在新线程中运行服务器
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 保存服务器引用
        self.api_server = server_thread
        
        # 等待服务器启动
        time.sleep(2)
        
        logger.info(f"API服务器已启动，健康检查地址: http://{host}:{port}/health")
        
        return server_thread
    
    def stop_api_server(self):
        """
        停止API服务器
        """
        if self.api_server and self.api_server.is_alive():
            # FastAPI/uvicorn 服务器需要特殊处理来优雅停止
            # 这里简化处理
            logger.info("停止API服务器...")
            # 在实际应用中，可能需要更复杂的停止逻辑
            self.api_server = None
            logger.info("API服务器已停止")
    
    def update_model_version(self,
                           model_name: str,
                           new_version: str,
                           old_version: str = None,
                           rollback: bool = False) -> Dict[str, str]:
        """
        更新模型版本（蓝绿部署）
        
        Args:
            model_name: 模型名称
            new_version: 新版本
            old_version: 旧版本（可选）
            rollback: 是否回滚
            
        Returns:
            部署信息
        """
        # 如果未指定旧版本，尝试查找当前版本
        if old_version is None:
            for key in self.active_models:
                if key.startswith(f"{model_name}_v"):
                    old_version = key.split('_v')[-1]
                    break
        
        logger.info(f"{'回滚' if rollback else '更新'}模型版本: {model_name} v{old_version} -> v{new_version}")
        
        # 加载新版本
        try:
            new_model = self.load_model_to_memory(model_name, new_version)
            
            # 如果成功，卸载旧版本
            if old_version and old_version != new_version:
                self.unload_model_from_memory(model_name, old_version)
            
            logger.info(f"模型版本{'回滚' if rollback else '更新'}成功")
            
            return {
                'model_name': model_name,
                'new_version': new_version,
                'old_version': old_version,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"模型版本{'回滚' if rollback else '更新'}失败: {str(e)}")
            raise
    
    def generate_deployment_report(self,
                                 model_name: str,
                                 version: str,
                                 output_dir: str = './reports') -> str:
        """
        生成部署报告
        
        Args:
            model_name: 模型名称
            version: 模型版本
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        # 确保目录存在
        ensure_directory(output_dir)
        
        # 获取部署信息
        model_key = f"{model_name}_v{version}"
        if model_key not in self.active_models:
            raise ValueError(f"模型未加载: {model_key}")
        
        model_info = self.active_models[model_key]
        metadata = model_info['metadata']
        
        # 创建报告内容
        report = {
            "report_title": "模型部署报告",
            "generation_time": datetime.now().isoformat(),
            "model_info": {
                "model_name": model_name,
                "version": version,
                "deployment_time": metadata.get('deployment_time'),
                "deployment_path": metadata.get('deployment_path')
            },
            "performance_metrics": {
                "requests_total": self.request_counter.labels(
                    model_name=model_name, 
                    endpoint='predict'
                )._value.get(),
                "errors_total": self.error_counter.labels(
                    model_name=model_name,
                    endpoint='predict',
                    error_type='all'
                )._value.get(),
                # 这里可以添加更多指标
            },
            "deployment_details": metadata
        }
        
        # 保存报告
        report_filename = f"deployment_report_{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"部署报告生成成功: {report_path}")
        
        return report_path


class ModelDeploymentPipeline:
    """
    模型部署流水线
    
    自动化模型训练、验证、部署流程
    """
    
    def __init__(self,
                 models_dir: str = './models',
                 deploy_dir: str = './deployments',
                 config: Optional[Dict] = None):
        """
        初始化部署流水线
        
        Args:
            models_dir: 模型目录
            deploy_dir: 部署目录
            config: 配置字典
        """
        self.models_dir = models_dir
        self.deploy_dir = deploy_dir
        self.config = config or {}
        self.trainer = ModelTrainer(models_dir=models_dir)
        self.deployment = ModelDeployment(
            models_dir=models_dir,
            deploy_dir=deploy_dir
        )
        self.pipeline_history = []
    
    @timeit
    def run_pipeline(self,
                    X_train: Union[pd.DataFrame, np.ndarray],
                    y_train: Union[pd.Series, np.ndarray],
                    X_val: Union[pd.DataFrame, np.ndarray],
                    y_val: Union[pd.Series, np.ndarray],
                    model_type: str,
                    model_params: Dict,
                    model_name: str,
                    perform_hyperparam_tuning: bool = False,
                    param_grid: Optional[Dict] = None,
                    auto_deploy: bool = True,
                    evaluate_after_deploy: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """
        运行完整部署流水线
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            model_type: 模型类型
            model_params: 模型参数
            model_name: 模型名称
            perform_hyperparam_tuning: 是否执行超参数调优
            param_grid: 参数网格
            auto_deploy: 是否自动部署
            evaluate_after_deploy: 部署后是否评估
            
        Returns:
            流水线结果
        """
        pipeline_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"开始运行部署流水线 #{pipeline_id}")
        
        results = {
            'pipeline_id': pipeline_id,
            'model_name': model_name,
            'model_type': model_type,
            'steps': []
        }
        
        try:
            # 1. 超参数调优
            if perform_hyperparam_tuning and param_grid:
                logger.info("执行超参数调优")
                tuning_results = self.trainer.hyperparameter_tuning(
                    X_train, y_train,
                    param_grid=param_grid,
                    model_type=model_type,
                    **kwargs
                )
                best_params = tuning_results['best_params']
                results['hyperparam_tuning'] = {
                    'best_params': best_params,
                    'status': 'completed'
                }
                results['steps'].append('hyperparam_tuning')
            else:
                best_params = model_params
            
            # 2. 训练模型
            logger.info("训练模型")
            model = self.trainer.train(
                X_train, y_train,
                X_val=X_val,
                y_val=y_val,
                model_type=model_type,
                params=best_params,
                **kwargs
            )
            
            # 3. 评估模型
            logger.info("评估模型")
            metrics = self.trainer.evaluate(X_val, y_val)
            results['training'] = {
                'metrics': metrics,
                'status': 'completed'
            }
            results['steps'].append('training')
            
            # 4. 保存模型
            logger.info("保存模型")
            model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_metadata = {
                'model_type': model_type,
                'parameters': best_params,
                'training_metrics': metrics,
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
            
            # 5. 部署模型
            if auto_deploy:
                logger.info("部署模型")
                deployment_info = self.deployment.deploy_model(
                    model_path=model_path,
                    model_name=model_name,
                    version=model_version,
                    metadata=model_metadata
                )
                results['deployment'] = deployment_info
                results['steps'].append('deployment')
                
                # 6. 部署后评估
                if evaluate_after_deploy:
                    logger.info("部署后评估")
                    deployed_metrics = self.deployment.predict_with_model(
                        model_name=model_name,
                        version=model_version,
                        X=X_val
                    )
                    results['post_deploy_evaluation'] = {
                        'status': 'completed'
                    }
                    results['steps'].append('post_deploy_evaluation')
            
            # 7. 生成报告
            if auto_deploy:
                report_path = self.deployment.generate_deployment_report(
                    model_name=model_name,
                    version=model_version
                )
                results['report'] = {
                    'report_path': report_path
                }
                results['steps'].append('report_generation')
            
            results['status'] = 'success'
            logger.info(f"部署流水线 #{pipeline_id} 成功完成")
            
        except Exception as e:
            logger.error(f"部署流水线 #{pipeline_id} 失败: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        # 保存历史记录
        self.pipeline_history.append(results)
        
        return results
    
    def validate_deployment(self,
                          model_name: str,
                          version: str,
                          X_test: Union[pd.DataFrame, np.ndarray],
                          y_test: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        验证部署的模型
        
        Args:
            model_name: 模型名称
            version: 模型版本
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            验证指标
        """
        logger.info(f"验证部署的模型: {model_name} v{version}")
        
        # 使用部署的模型进行预测
        predictions = self.deployment.predict_with_model(
            model_name=model_name,
            version=version,
            X=X_test
        )
        
        # 计算评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        logger.info(f"模型验证结果: {metrics}")
        
        return metrics


def generate_metrics_text():
    """
    生成Prometheus格式的指标文本
    
    Returns:
        指标文本
    """
    # 这里简化实现，实际应用中可以使用prometheus_client的功能
    from prometheus_client import generate_latest
    return generate_latest().decode('utf-8')


def create_dockerfile(model_name: str,
                      version: str,
                      output_dir: str = './docker',
                      base_image: str = 'python:3.9-slim',
                      app_port: int = 8000,
                      monitoring_port: int = 8001) -> str:
    """
    创建Dockerfile用于容器化部署
    
    Args:
        model_name: 模型名称
        version: 模型版本
        output_dir: 输出目录
        base_image: 基础镜像
        app_port: 应用端口
        monitoring_port: 监控端口
        
    Returns:
        Dockerfile路径
    """
    # 确保目录存在
    ensure_directory(output_dir)
    
    # Dockerfile内容
    dockerfile_content = f'''
FROM {base_image}

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p models deployments

# 复制模型和部署文件
COPY models/{model_name}_v{version}* models/

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE {app_port}
EXPOSE {monitoring_port}

# 设置环境变量
ENV MODEL_NAME={model_name}
ENV MODEL_VERSION={version}
ENV PORT={app_port}
ENV MONITORING_PORT={monitoring_port}

# 启动命令
CMD ["python", "-m", "src.api.server"]
'''
    
    # 保存Dockerfile
    dockerfile_path = os.path.join(output_dir, 'Dockerfile')
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    
    logger.info(f"Dockerfile创建成功: {dockerfile_path}")
    
    # 创建requirements.txt示例
    requirements_path = os.path.join(output_dir, 'requirements.txt')
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write('''fastapi>=0.70.0
uvicorn>=0.15.0
pydantic>=1.8.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
joblib>=1.0.0
prometheus_client>=0.12.0
''')
    
    return dockerfile_path


def create_helm_chart(model_name: str,
                      version: str,
                      output_dir: str = './helm',
                      chart_version: str = '1.0.0',
                      replica_count: int = 2,
                      resources: Optional[Dict] = None) -> str:
    """
    创建Helm Chart用于Kubernetes部署
    
    Args:
        model_name: 模型名称
        version: 模型版本
        output_dir: 输出目录
        chart_version: Chart版本
        replica_count: 副本数量
        resources: 资源限制
        
    Returns:
        Helm Chart目录路径
    """
    if resources is None:
        resources = {
            'requests': {'cpu': '100m', 'memory': '256Mi'},
            'limits': {'cpu': '500m', 'memory': '512Mi'}
        }
    
    # 确保目录存在
    chart_dir = os.path.join(output_dir, f"{model_name}-model")
    ensure_directory(chart_dir)
    ensure_directory(os.path.join(chart_dir, 'templates'))
    
    # 创建Chart.yaml
    chart_yaml = f'''
apiVersion: v2
name: {model_name}-model
description: A Helm chart for deploying {model_name} ML model
version: {chart_version}
appVersion: "{version}"
'''
    
    with open(os.path.join(chart_dir, 'Chart.yaml'), 'w', encoding='utf-8') as f:
        f.write(chart_yaml)
    
    # 创建values.yaml
    values_yaml = f'''
replicaCount: {replica_count}

image:
  repository: {model_name}-model
  pullPolicy: IfNotPresent
  tag: "v{version}"

nameOverride: ""
fullnameOverride: ""

model:
  name: {model_name}
  version: {version}

ports:
  api:
    containerPort: 8000
    protocol: TCP
  monitoring:
    containerPort: 8001
    protocol: TCP

resources:
  requests:
    cpu: {resources['requests']['cpu']}
    memory: {resources['requests']['memory']}
  limits:
    cpu: {resources['limits']['cpu']}
    memory: {resources['limits']['memory']}
'''
    
    with open(os.path.join(chart_dir, 'values.yaml'), 'w', encoding='utf-8') as f:
        f.write(values_yaml)
    
    # 创建deployment template
    deployment_template = f'''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
  labels:
    app: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: {{ .Release.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          env:
            - name: MODEL_NAME
              value: "{{ .Values.model.name }}"
            - name: MODEL_VERSION
              value: "{{ .Values.model.version }}"
            - name: PORT
              value: "{{ .Values.ports.api.containerPort }}"
            - name: MONITORING_PORT
              value: "{{ .Values.ports.monitoring.containerPort }}"
          ports:
            - name: api
              containerPort: {{ .Values.ports.api.containerPort }}
              protocol: {{ .Values.ports.api.protocol }}
            - name: monitoring
              containerPort: {{ .Values.ports.monitoring.containerPort }}
              protocol: {{ .Values.ports.monitoring.protocol }}
          resources:
            requests:
              cpu: {{ .Values.resources.requests.cpu }}
              memory: {{ .Values.resources.requests.memory }}
            limits:
              cpu: {{ .Values.resources.limits.cpu }}
              memory: {{ .Values.resources.limits.memory }}
'''
    
    with open(os.path.join(chart_dir, 'templates', 'deployment.yaml'), 'w', encoding='utf-8') as f:
        f.write(deployment_template)
    
    # 创建service template
    service_template = f'''
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}
  labels:
    app: {{ .Release.Name }}
spec:
  type: ClusterIP
  ports:
    - port: {{ .Values.ports.api.containerPort }}
      targetPort: {{ .Values.ports.api.containerPort }}
      protocol: {{ .Values.ports.api.protocol }}
      name: api
    - port: {{ .Values.ports.monitoring.containerPort }}
      targetPort: {{ .Values.ports.monitoring.containerPort }}
      protocol: {{ .Values.ports.monitoring.protocol }}
      name: monitoring
  selector:
    app: {{ .Release.Name }}
'''
    
    with open(os.path.join(chart_dir, 'templates', 'service.yaml'), 'w', encoding='utf-8') as f:
        f.write(service_template)
    
    logger.info(f"Helm Chart创建成功: {chart_dir}")
    
    return chart_dir


def create_api_server_script(output_path: str = './src/api/server.py'):
    """
    创建API服务器启动脚本
    
    Args:
        output_path: 输出路径
        
    Returns:
        脚本路径
    """
    # 确保目录存在
    ensure_directory(os.path.dirname(output_path))
    
    # 服务器脚本内容
    server_script = '''
import os
import sys
import logging
from src.models.model_deployment import ModelDeployment

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取环境变量
MODEL_NAME = os.environ.get('MODEL_NAME', 'default_model')
MODEL_VERSION = os.environ.get('MODEL_VERSION', 'latest')
PORT = int(os.environ.get('PORT', 8000))
MONITORING_PORT = int(os.environ.get('MONITORING_PORT', 8001))

# 初始化部署管理器
deployment = ModelDeployment(
    models_dir='./models',
    deploy_dir='./deployments'
)

# 加载模型
try:
    deployment.load_model_to_memory(MODEL_NAME, MODEL_VERSION)
    logger.info(f"模型 {MODEL_NAME} v{MODEL_VERSION} 已加载")
except Exception as e:
    logger.error(f"加载模型失败: {str(e)}")
    sys.exit(1)

# 启动API服务器
logger.info(f"启动API服务器在端口 {PORT}")
deployment.start_api_server(
    host='0.0.0.0',
    port=PORT,
    enable_monitoring=True,
    monitoring_port=MONITORING_PORT
)

# 保持进程运行
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("正在关闭服务器...")
    deployment.stop_api_server()
    logger.info("服务器已关闭")
'''
    
    # 保存脚本
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(server_script)
    
    logger.info(f"API服务器脚本创建成功: {output_path}")
    
    return output_path


def create_inference_client():
    """
    创建推理客户端
    
    Returns:
        客户端类
    """
    class InferenceClient:
        """
        推理API客户端
        """
        
        def __init__(self,
                    base_url: str = "http://localhost:8000",
                    model_name: str = "default_model",
                    model_version: str = "latest"):
            """
            初始化客户端
            
            Args:
                base_url: API基础URL
                model_name: 模型名称
                model_version: 模型版本
            """
            self.base_url = base_url.rstrip('/')
            self.model_name = model_name
            self.model_version = model_version
            self.session = requests.Session()
        
        def predict(self,
                   data: List[Dict],
                   model_name: Optional[str] = None,
                   model_version: Optional[str] = None) -> List[float]:
            """
            发送预测请求
            
            Args:
                data: 输入数据
                model_name: 模型名称（可选）
                model_version: 模型版本（可选）
                
            Returns:
                预测结果
            """
            endpoint = f"{self.base_url}/predict"
            payload = {
                "model_name": model_name or self.model_name,
                "version": model_version or self.model_version,
                "data": data
            }
            
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            
            return response.json().get("predictions", [])
        
        def batch_predict(self,
                         datasets: List[List[Dict]],
                         model_name: Optional[str] = None,
                         model_version: Optional[str] = None) -> List[List[float]]:
            """
            批量预测
            
            Args:
                datasets: 数据集列表
                model_name: 模型名称（可选）
                model_version: 模型版本（可选）
                
            Returns:
                批量预测结果
            """
            endpoint = f"{self.base_url}/batch_predict"
            payload = {
                "model_name": model_name or self.model_name,
                "version": model_version or self.model_version,
                "datasets": datasets
            }
            
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            
            return response.json().get("batch_predictions", [])
        
        def explain(self,
                   data: List[Dict],
                   model_name: Optional[str] = None,
                   model_version: Optional[str] = None) -> Dict[str, Any]:
            """
            获取预测解释
            
            Args:
                data: 输入数据
                model_name: 模型名称（可选）
                model_version: 模型版本（可选）
                
            Returns:
                解释结果
            """
            endpoint = f"{self.base_url}/explain"
            payload = {
                "model_name": model_name or self.model_name,
                "version": model_version or self.model_version,
                "data": data
            }
            
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            
            return response.json()
        
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
            
            return response.json().get("models", [])
    
    return InferenceClient