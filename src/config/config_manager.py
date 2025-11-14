import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# 自定义模块
from src.utils.helpers import setup_logger, DEFAULT_LOGGER

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class ConfigManager:
    """
    配置管理器
    
    负责加载、验证、合并和提供配置信息
    """
    
    # 默认配置文件路径
    DEFAULT_CONFIG_PATHS = [
        './config/config.json',
        './config/config.yaml',
        './config/config.yml'
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = {}
        self.config_path = config_path
        self.loaded_paths = []
        
        # 加载配置
        self.load_config(config_path)
        
        # 验证配置
        self.validate_config()
        
        logger.info("配置管理器初始化完成")
    
    def load_config(self, config_path: Optional[str] = None):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        # 如果提供了配置路径，尝试加载
        if config_path:
            if os.path.exists(config_path):
                self._load_single_config(config_path)
            else:
                logger.warning(f"配置文件不存在: {config_path}")
                # 尝试使用默认配置
                self._load_default_configs()
        else:
            # 使用默认配置
            self._load_default_configs()
        
        # 应用环境变量覆盖
        self._apply_environment_overrides()
    
    def _load_single_config(self, config_path: str):
        """
        加载单个配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            file_ext = os.path.splitext(config_path)[1].lower()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if file_ext in ['.json']:
                    config = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {file_ext}")
                    return
            
            # 合并配置
            self._merge_config(config)
            self.loaded_paths.append(config_path)
            
            logger.info(f"配置文件加载成功: {config_path}")
            
            # 检查是否有包含其他配置文件的指令
            if 'include_configs' in config:
                for include_path in config['include_configs']:
                    abs_include_path = self._resolve_path(include_path, os.path.dirname(config_path))
                    if os.path.exists(abs_include_path) and abs_include_path not in self.loaded_paths:
                        self._load_single_config(abs_include_path)
        
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {str(e)}")
    
    def _load_default_configs(self):
        """
        加载默认配置文件
        """
        for default_path in self.DEFAULT_CONFIG_PATHS:
            if os.path.exists(default_path):
                self._load_single_config(default_path)
    
    def _resolve_path(self, path: str, base_dir: str) -> str:
        """
        解析路径
        
        Args:
            path: 路径
            base_dir: 基础目录
            
        Returns:
            解析后的绝对路径
        """
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(base_dir, path))
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """
        合并配置
        
        Args:
            new_config: 新配置
        """
        # 递归合并配置字典
        def deep_merge(a: Dict, b: Dict) -> Dict:
            result = a.copy()
            
            for key, value in b.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        self.config = deep_merge(self.config, new_config)
    
    def _apply_environment_overrides(self):
        """
        应用环境变量覆盖配置
        """
        # 环境变量格式: TIMESERIES__SECTION__KEY=value
        prefix = 'TIMESERIES__'
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀，将双下划线替换为单下划线
                config_key = key[len(prefix):].replace('__', '.')
                
                # 设置配置值
                self._set_nested_config_value(config_key, value)
                
                logger.debug(f"环境变量覆盖配置: {config_key}")
    
    def _set_nested_config_value(self, key_path: str, value: str):
        """
        设置嵌套配置值
        
        Args:
            key_path: 键路径，使用点分隔
            value: 值
        """
        parts = key_path.split('.')
        current = self.config
        
        # 遍历键路径，创建必要的嵌套字典
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                logger.warning(f"配置路径冲突: {'.'.join(parts[:i+1])}")
                return
            current = current[part]
        
        # 设置最终值
        # 尝试转换值的类型
        final_key = parts[-1]
        try:
            # 尝试转换为数字
            if '.' in value:
                current[final_key] = float(value)
            else:
                current[final_key] = int(value)
        except ValueError:
            # 尝试转换为布尔值
            if value.lower() == 'true':
                current[final_key] = True
            elif value.lower() == 'false':
                current[final_key] = False
            # 尝试转换为None
            elif value.lower() == 'none':
                current[final_key] = None
            # 保持字符串
            else:
                current[final_key] = value
    
    def validate_config(self):
        """
        验证配置的有效性
        """
        # 这里实现配置验证逻辑
        # 可以添加必要的配置项检查
        
        # 确保必要的配置节存在
        required_sections = ['data', 'models', 'backtest', 'logging']
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"缺少配置节: {section}")
                # 设置默认配置
                self.config[section] = self._get_default_section_config(section)
    
    def _get_default_section_config(self, section: str) -> Dict[str, Any]:
        """
        获取默认配置节
        
        Args:
            section: 配置节名称
            
        Returns:
            默认配置
        """
        default_configs = {
            'data': {
                'data_dir': './data',
                'cache_dir': './cache',
                'file_extension': 'csv',
                'date_column': 'date',
                'timezone': 'UTC'
            },
            'models': {
                'models_dir': './models',
                'deploy_dir': './deployments',
                'max_models': 10,
                'model_retention_days': 30
            },
            'backtest': {
                'initial_capital': 1000000.0,
                'commission': 0.001,
                'slippage': 0.0005,
                'risk_free_rate': 0.02,
                'benchmark_symbol': 'SPY'
            },
            'logging': {
                'level': 'INFO',
                'log_dir': './logs',
                'rotation': 'daily',
                'max_bytes': 10485760,
                'backup_count': 7
            }
        }
        
        return default_configs.get(section, {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，可以使用点分隔的路径
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，可以使用点分隔的路径
            value: 配置值
        """
        parts = key.split('.')
        current = self.config
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                logger.warning(f"配置路径冲突: {'.'.join(parts[:i+1])}")
                return
            current = current[part]
        
        current[parts[-1]] = value
        
        logger.debug(f"配置已更新: {key}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section: 配置节名称
            
        Returns:
            配置节字典
        """
        return self.get(section, {})
    
    def update_section(self, section: str, config: Dict[str, Any]):
        """
        更新配置节
        
        Args:
            section: 配置节名称
            config: 新配置
        """
        if section not in self.config:
            self.config[section] = {}
        
        # 合并配置
        self.config[section].update(config)
        
        logger.debug(f"配置节已更新: {section}")
    
    def get_available_sections(self) -> List[str]:
        """
        获取所有可用的配置节
        
        Returns:
            配置节列表
        """
        return list(self.config.keys())
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        获取完整配置
        
        Returns:
            完整配置字典
        """
        return self.config.copy()
    
    def save_config(self, output_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        if output_path is None:
            if self.config_path:
                output_path = self.config_path
            else:
                output_path = './config/config.json'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 根据文件扩展名选择格式
        file_ext = os.path.splitext(output_path)[1].lower()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if file_ext in ['.json']:
                    json.dump(self.config, f, indent=2, ensure_ascii=False, default=str)
                elif file_ext in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"不支持的文件格式: {file_ext}")
            
            logger.info(f"配置已保存到: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            raise
    
    def generate_config_template(self, output_path: str = './config/config_template.yaml') -> str:
        """
        生成配置模板
        
        Args:
            output_path: 输出路径
            
        Returns:
            生成的模板文件路径
        """
        # 生成完整的配置模板
        template = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            
            # 数据配置
            'data': {
                'data_dir': './data',
                'cache_dir': './cache',
                'file_extension': 'csv',
                'date_column': 'date',
                'timezone': 'UTC',
                'default_fields': ['open', 'high', 'low', 'close', 'volume']
            },
            
            # 模型配置
            'models': {
                'models_dir': './models',
                'deploy_dir': './deployments',
                'max_models': 10,
                'model_retention_days': 30,
                'default_model_type': 'xgboost',
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                },
                'deep_learning': {
                    'input_size': None,
                    'hidden_sizes': [64, 32],
                    'dropout_rates': [0.2, 0.2],
                    'batch_size': 32,
                    'epochs': 100,
                    'learning_rate': 0.001
                }
            },
            
            # 特征工程配置
            'features': {
                'use_technical_indicators': True,
                'use_fundamental_data': False,
                'use_sentiment_data': False,
                'window_sizes': [5, 10, 20, 50],
                'technical_indicators': {
                    'moving_averages': True,
                    'rsi': True,
                    'macd': True,
                    'bollinger_bands': True,
                    'volume_indicators': True
                },
                'feature_selection': {
                    'enabled': True,
                    'method': 'importance',
                    'top_n': 20
                }
            },
            
            # 回测配置
            'backtest': {
                'initial_capital': 1000000.0,
                'commission': 0.001,
                'slippage': 0.0005,
                'risk_free_rate': 0.02,
                'benchmark_symbol': 'SPY',
                'rebalance_frequency': 'daily',
                'position_sizing': 'equal_weight',
                'max_position_size': 0.1,
                'min_position_size': 0.01
            },
            
            # 因子配置
            'factors': {
                'factor_library_path': './src/factors',
                'max_factor_depth': 5,
                'cache_expiry_seconds': 3600,
                'parallel_computation': True,
                'max_workers': 4
            },
            
            # API服务配置
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'workers': 4,
                'cors_origins': ['*'],
                'timeout': 300,
                'max_request_size': 10485760
            },
            
            # 监控配置
            'monitoring': {
                'enabled': True,
                'port': 8001,
                'metrics_path': '/metrics',
                'log_level': 'INFO',
                'alerting': {
                    'enabled': False,
                    'email': '',
                    'slack_webhook': ''
                }
            },
            
            # 日志配置
            'logging': {
                'level': 'INFO',
                'log_dir': './logs',
                'rotation': 'daily',
                'max_bytes': 10485760,
                'backup_count': 7,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            
            # 部署配置
            'deployment': {
                'docker': {
                    'enabled': True,
                    'image_name': 'timeseries-model-service',
                    'tag': 'latest',
                    'dockerfile_path': './Dockerfile'
                },
                'kubernetes': {
                    'enabled': False,
                    'namespace': 'timeseries',
                    'replicas': 2,
                    'resources': {
                        'cpu': '1',
                        'memory': '1Gi'
                    }
                }
            },
            
            # 包含其他配置文件
            'include_configs': []
        }
        
        # 保存模板
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        file_ext = os.path.splitext(output_path)[1].lower()
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if file_ext in ['.json']:
                    json.dump(template, f, indent=2, ensure_ascii=False, default=str)
                elif file_ext in ['.yaml', '.yml']:
                    yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"不支持的文件格式: {file_ext}")
            
            logger.info(f"配置模板已生成: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"生成配置模板失败: {str(e)}")
            raise
    
    def validate_model_config(self, model_config: Dict[str, Any]) -> bool:
        """
        验证模型配置
        
        Args:
            model_config: 模型配置
            
        Returns:
            是否有效
        """
        # 必要的字段检查
        required_fields = ['type', 'params']
        for field in required_fields:
            if field not in model_config:
                logger.error(f"模型配置缺少必要字段: {field}")
                return False
        
        # 检查模型类型是否支持
        supported_types = ['xgboost', 'lightgbm', 'random_forest', 'deep_learning', 'linear', 'ridge', 'lasso']
        model_type = model_config['type']
        if model_type not in supported_types:
            logger.warning(f"可能不支持的模型类型: {model_type}")
        
        return True
    
    def validate_backtest_config(self, backtest_config: Dict[str, Any]) -> bool:
        """
        验证回测配置
        
        Args:
            backtest_config: 回测配置
            
        Returns:
            是否有效
        """
        # 必要的字段检查
        required_fields = ['initial_capital', 'commission']
        for field in required_fields:
            if field not in backtest_config:
                logger.error(f"回测配置缺少必要字段: {field}")
                return False
        
        # 验证数值范围
        if backtest_config['initial_capital'] <= 0:
            logger.error("初始资金必须大于0")
            return False
        
        if backtest_config['commission'] < 0:
            logger.error("佣金必须大于等于0")
            return False
        
        return True
    
    def get_environment_variables_map(self) -> Dict[str, str]:
        """
        获取配置到环境变量的映射
        
        Returns:
            环境变量映射字典
        """
        env_vars = {}
        
        # 递归构建环境变量映射
        def build_env_vars(prefix: str, config_dict: Dict[str, Any]):
            for key, value in config_dict.items():
                current_prefix = f"{prefix}__{key}" if prefix else key
                
                if isinstance(value, dict):
                    build_env_vars(current_prefix, value)
                else:
                    # 转换为字符串
                    env_key = f"TIMESERIES__{current_prefix}"
                    env_vars[env_key] = str(value)
        
        build_env_vars("", self.config)
        
        return env_vars
    
    def generate_docker_env_file(self, output_path: str = './.env.docker') -> str:
        """
        生成Docker环境变量文件
        
        Args:
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        # 获取环境变量映射
        env_vars = self.get_environment_variables_map()
        
        # 写入.env文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Docker环境变量 - 自动生成\n")
            f.write(f"# 生成时间: {datetime.now().isoformat()}\n\n")
            
            for key, value in env_vars.items():
                # 对特殊字符进行转义
                escaped_value = value.replace('"', '\\"').replace('\n', '\\n')
                f.write(f"{key}="{escaped_value}"\n")
        
        logger.info(f"Docker环境变量文件已生成: {output_path}")
        
        return output_path
    
    def __str__(self) -> str:
        """
        字符串表示
        """
        return json.dumps(self.config, indent=2, ensure_ascii=False, default=str)
    
    def __repr__(self) -> str:
        """
        调试表示
        """
        return f"ConfigManager(config_path={self.config_path}, loaded_paths={self.loaded_paths})"


# 全局配置管理器实例
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    elif config_path and config_path != _config_manager.config_path:
        # 如果提供了不同的配置路径，重新加载
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置（便捷函数）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_manager = get_config_manager(config_path)
    return config_manager.get_full_config()


def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取配置值（便捷函数）
    
    Args:
        key: 配置键
        default: 默认值
        
    Returns:
        配置值
    """
    config_manager = get_config_manager()
    return config_manager.get(key, default)


def generate_default_config(output_path: str = './config/config.yaml') -> str:
    """
    生成默认配置文件
    
    Args:
        output_path: 输出路径
        
    Returns:
        生成的配置文件路径
    """
    config_manager = ConfigManager()
    return config_manager.generate_config_template(output_path)


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="配置管理工具")
    parser.add_argument("--generate", action="store_true", help="生成配置模板")
    parser.add_argument("--validate", type=str, help="验证配置文件")
    parser.add_argument("--output", type=str, default="./config/config.yaml", help="输出文件路径")
    parser.add_argument("--env-file", type=str, help="生成环境变量文件")
    
    args = parser.parse_args()
    
    if args.generate:
        # 生成配置模板
        config_manager = ConfigManager()
        config_manager.generate_config_template(args.output)
        print(f"配置模板已生成: {args.output}")
    
    elif args.validate:
        # 验证配置文件
        config_manager = ConfigManager(args.validate)
        print("配置验证通过！")
        print(f"加载的配置文件: {config_manager.loaded_paths}")
        
        # 打印配置摘要
        print("\n配置摘要:")
        for section in config_manager.get_available_sections():
            print(f"  - {section}")
    
    elif args.env_file:
        # 生成环境变量文件
        config_manager = ConfigManager()
        config_manager.generate_docker_env_file(args.env_file)
        print(f"环境变量文件已生成: {args.env_file}")
    
    else:
        # 显示帮助
        parser.print_help()