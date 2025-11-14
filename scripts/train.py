#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练与评估脚本
用于自动化模型训练、超参数调优和评估流程
"""

import os
import sys
import json
import yaml
import time
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 自定义模块
from src.utils.helpers import setup_logger, DEFAULT_LOGGER
from src.config.config_manager import ConfigManager
from src.models.data_preprocessing import DataPreprocessor, TimeSeriesDataLoader
from src.models.model_training import ModelTrainer, EnsembleModel

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        参数命名空间
    """
    parser = argparse.ArgumentParser(description='模型训练与评估脚本')
    
    # 数据相关
    parser.add_argument('--data-path', '-d',
                      type=str,
                      default='./data/training_data.csv',
                      help='训练数据文件路径')
    
    parser.add_argument('--test-data-path',
                      type=str,
                      help='测试数据文件路径（可选）')
    
    parser.add_argument('--target-column', '-t',
                      type=str,
                      default='target',
                      help='目标变量列名')
    
    parser.add_argument('--date-column',
                      type=str,
                      default='date',
                      help='日期列名')
    
    parser.add_argument('--test-size',
                      type=float,
                      default=0.2,
                      help='测试集比例')
    
    parser.add_argument('--validation-size',
                      type=float,
                      default=0.1,
                      help='验证集比例')
    
    # 模型相关
    parser.add_argument('--model-type', '-m',
                      choices=['linear', 'ridge', 'lasso', 'elasticnet', 'random_forest',
                               'gradient_boosting', 'xgboost', 'lightgbm', 'catboost',
                               'mlp', 'lstm', 'gru', 'transformer', 'cnn', 'ensemble'],
                      default='xgboost',
                      help='模型类型')
    
    parser.add_argument('--model-config',
                      type=str,
                      help='模型配置文件路径')
    
    parser.add_argument('--hyperopt',
                      action='store_true',
                      help='启用超参数优化')
    
    parser.add_argument('--n-trials',
                      type=int,
                      default=50,
                      help='超参数优化迭代次数')
    
    # 训练相关
    parser.add_argument('--n-folds',
                      type=int,
                      default=5,
                      help='交叉验证折数')
    
    parser.add_argument('--max-epochs',
                      type=int,
                      default=100,
                      help='最大训练轮数')
    
    parser.add_argument('--patience',
                      type=int,
                      default=10,
                      help='早停耐心值')
    
    parser.add_argument('--batch-size',
                      type=int,
                      default=64,
                      help='批次大小')
    
    parser.add_argument('--learning-rate',
                      type=float,
                      default=0.001,
                      help='学习率')
    
    parser.add_argument('--random-state',
                      type=int,
                      default=42,
                      help='随机种子')
    
    # 特征工程
    parser.add_argument('--feature-config',
                      type=str,
                      help='特征工程配置文件路径')
    
    parser.add_argument('--features-to-select',
                      nargs='+',
                      help='选择的特征列表')
    
    parser.add_argument('--feature-importance-threshold',
                      type=float,
                      help='特征重要性阈值')
    
    # 保存和输出
    parser.add_argument('--output-dir', '-o',
                      type=str,
                      default='./models',
                      help='模型输出目录')
    
    parser.add_argument('--model-name',
                      type=str,
                      help='模型保存名称')
    
    parser.add_argument('--save-plots',
                      action='store_true',
                      help='保存可视化图表')
    
    parser.add_argument('--save-report',
                      action='store_true',
                      help='保存评估报告')
    
    # 其他选项
    parser.add_argument('--config', '-c',
                      type=str,
                      default='./config/config.yaml',
                      help='全局配置文件路径')
    
    parser.add_argument('--ensemble-models',
                      nargs='+',
                      help='集成学习使用的模型列表')
    
    parser.add_argument('--verbose',
                      action='store_true',
                      help='详细输出')
    
    parser.add_argument('--resume',
                      action='store_true',
                      help='从检查点恢复训练')
    
    return parser.parse_args()


def load_and_preprocess_data(args: argparse.Namespace,
                           config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """
    加载和预处理数据
    
    Args:
        args: 命令行参数
        config: 配置字典
        
    Returns:
        训练集、验证集、测试集的特征和目标变量，以及数据预处理器
    """
    logger.info(f"加载数据: {args.data_path}")
    
    # 加载数据
    loader = TimeSeriesDataLoader()
    df = loader.load_data(args.data_path)
    
    # 检查目标列和日期列
    if args.target_column not in df.columns:
        raise ValueError(f"目标列 '{args.target_column}' 不存在于数据中")
    
    if args.date_column in df.columns:
        # 解析日期
        df[args.date_column] = pd.to_datetime(df[args.date_column])
        # 按日期排序
        df = df.sort_values(args.date_column)
    
    # 数据预处理
    preprocessor_config = config.get('data_preprocessing', {})
    
    # 合并命令行参数和配置文件中的预处理设置
    if args.feature_config:
        with open(args.feature_config, 'r') as f:
            feature_config = yaml.safe_load(f)
            preprocessor_config.update(feature_config)
    
    # 创建预处理器
    preprocessor = DataPreprocessor(config=preprocessor_config)
    
    # 特征工程和数据分割
    logger.info("执行特征工程和数据分割...")
    
    # 如果有测试数据路径，使用它作为测试集
    if args.test_data_path:
        test_df = loader.load_data(args.test_data_path)
        
        # 分割训练和验证数据
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(
            df,
            target_column=args.target_column,
            test_size=args.validation_size,
            random_state=args.random_state,
            date_column=args.date_column
        )
        
        # 对测试数据进行转换
        X_test, y_test = preprocessor.transform(test_df, target_column=args.target_column)
    else:
        # 自动分割训练、验证、测试数据
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform_and_split(
            df,
            target_column=args.target_column,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.random_state,
            date_column=args.date_column
        )
    
    # 特征选择
    if args.features_to_select:
        logger.info(f"选择特征: {args.features_to_select}")
        # 确保选择的特征存在
        selected_features_indices = []
        for feature in args.features_to_select:
            if feature in preprocessor.feature_names_:
                idx = preprocessor.feature_names_.index(feature)
                selected_features_indices.append(idx)
            else:
                logger.warning(f"特征 '{feature}' 不存在，跳过")
        
        if selected_features_indices:
            X_train = X_train[:, selected_features_indices]
            X_val = X_val[:, selected_features_indices]
            X_test = X_test[:, selected_features_indices]
            # 更新特征名称
            preprocessor.feature_names_ = [preprocessor.feature_names_[i] for i in selected_features_indices]
    
    logger.info(f"训练集大小: {X_train.shape}")
    logger.info(f"验证集大小: {X_val.shape}")
    logger.info(f"测试集大小: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


def create_model_trainer(args: argparse.Namespace,
                        config: Dict[str, Any]) -> ModelTrainer:
    """
    创建模型训练器
    
    Args:
        args: 命令行参数
        config: 配置字典
        
    Returns:
        模型训练器实例
    """
    # 获取模型配置
    model_config = config.get('models', {}).get(args.model_type, {})
    
    # 合并命令行参数和配置文件中的模型设置
    model_params = {
        'model_type': args.model_type,
        'random_state': args.random_state,
        'n_folds': args.n_folds,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    }
    
    # 更新模型参数
    model_params.update(model_config)
    
    # 如果提供了模型配置文件，加载它
    if args.model_config:
        with open(args.model_config, 'r') as f:
            custom_model_config = yaml.safe_load(f)
            model_params.update(custom_model_config)
    
    # 启用超参数优化
    if args.hyperopt:
        model_params['hyperopt'] = True
        model_params['n_trials'] = args.n_trials
    
    # 特殊处理集成学习
    if args.model_type == 'ensemble' and args.ensemble_models:
        model_params['base_models'] = args.ensemble_models
    
    logger.info(f"创建模型训练器: {args.model_type}")
    logger.debug(f"模型参数: {model_params}")
    
    # 创建模型训练器
    trainer = ModelTrainer(**model_params)
    
    return trainer


def train_model(trainer: ModelTrainer,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                args: argparse.Namespace) -> Any:
    """
    训练模型
    
    Args:
        trainer: 模型训练器
        X_train: 训练集特征
        y_train: 训练集目标
        X_val: 验证集特征
        y_val: 验证集目标
        args: 命令行参数
        
    Returns:
        训练好的模型
    """
    logger.info("开始模型训练...")
    
    # 训练模型
    if args.resume:
        # 尝试从检查点恢复训练
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints')
        if os.path.exists(checkpoint_path):
            logger.info(f"从检查点恢复训练: {checkpoint_path}")
            model = trainer.resume_training(
                X_train, y_train,
                X_val, y_val,
                checkpoint_dir=checkpoint_path
            )
        else:
            logger.warning("未找到检查点，从头开始训练")
            model = trainer.train(X_train, y_train, X_val, y_val)
    else:
        # 从头开始训练
        model = trainer.train(X_train, y_train, X_val, y_val)
    
    logger.info("模型训练完成")
    
    # 打印模型信息
    if hasattr(trainer, 'best_params_') and trainer.best_params_:
        logger.info(f"最佳超参数: {trainer.best_params_}")
    
    return model


def evaluate_model(model: Any,
                   trainer: ModelTrainer,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   args: argparse.Namespace,
                   output_dir: str) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 训练好的模型
        trainer: 模型训练器
        X_test: 测试集特征
        y_test: 测试集目标
        args: 命令行参数
        output_dir: 输出目录
        
    Returns:
        评估指标
    """
    logger.info("评估模型性能...")
    
    # 评估模型
    metrics = trainer.evaluate(model, X_test, y_test)
    
    # 打印评估结果
    logger.info("模型评估结果:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.6f}")
    
    # 保存评估结果
    metrics_file = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"评估结果已保存到: {metrics_file}")
    
    # 可视化（如果启用）
    if args.save_plots:
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        logger.info(f"生成可视化图表，保存到: {plots_dir}")
        
        # 生成各种可视化
        trainer.plot_predictions(model, X_test, y_test, plots_dir)
        trainer.plot_learning_curves(plots_dir)
        
        if hasattr(trainer, 'plot_feature_importance'):
            try:
                trainer.plot_feature_importance(model, plots_dir)
            except Exception as e:
                logger.warning(f"生成特征重要性图失败: {e}")
        
        if hasattr(trainer, 'plot_residuals'):
            try:
                trainer.plot_residuals(model, X_test, y_test, plots_dir)
            except Exception as e:
                logger.warning(f"生成残差图失败: {e}")
    
    # 保存评估报告（如果启用）
    if args.save_report:
        report_file = os.path.join(output_dir, 'evaluation_report.md')
        trainer.generate_evaluation_report(model, X_test, y_test, report_file)
        logger.info(f"评估报告已保存到: {report_file}")
    
    return metrics


def save_model(model: Any,
               trainer: ModelTrainer,
               preprocessor: DataPreprocessor,
               args: argparse.Namespace,
               metrics: Dict[str, float]) -> str:
    """
    保存模型和相关组件
    
    Args:
        model: 训练好的模型
        trainer: 模型训练器
        preprocessor: 数据预处理器
        args: 命令行参数
        metrics: 评估指标
        
    Returns:
        模型保存目录
    """
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.model_name:
        model_dir = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}")
    else:
        model_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"保存模型到: {model_dir}")
    
    # 保存模型
    model_file = os.path.join(model_dir, 'model.pkl')
    trainer.save_model(model, model_file)
    
    # 保存预处理器
    preprocessor_file = os.path.join(model_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_file)
    
    # 保存模型元数据
    metadata = {
        'model_type': args.model_type,
        'timestamp': timestamp,
        'metrics': metrics,
        'features': preprocessor.feature_names_ if hasattr(preprocessor, 'feature_names_') else [],
        'params': trainer.get_params()
    }
    
    metadata_file = os.path.join(model_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 更新最新模型链接
    latest_link = os.path.join(args.output_dir, 'latest_model')
    try:
        if os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            elif os.path.isdir(latest_link):
                import shutil
                shutil.rmtree(latest_link)
        
        # 创建符号链接或复制目录
        if hasattr(os, 'symlink') and not os.name == 'nt':  # Windows不支持符号链接
            os.symlink(model_dir, latest_link)
        else:
            import shutil
            shutil.copytree(model_dir, latest_link, dirs_exist_ok=True)
            
        logger.info(f"更新最新模型链接: {latest_link} -> {model_dir}")
    except Exception as e:
        logger.warning(f"创建最新模型链接失败: {e}")
    
    return model_dir


def perform_feature_selection(trainer: ModelTrainer,
                              model: Any,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_val: np.ndarray,
                              y_val: np.ndarray,
                              preprocessor: DataPreprocessor,
                              threshold: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    基于特征重要性执行特征选择
    
    Args:
        trainer: 模型训练器
        model: 训练好的模型
        X_train: 训练集特征
        y_train: 训练集目标
        X_val: 验证集特征
        y_val: 验证集目标
        preprocessor: 数据预处理器
        threshold: 特征重要性阈值
        
    Returns:
        选择后的训练集和验证集特征，以及选择的特征名称
    """
    if not hasattr(trainer, 'get_feature_importance'):
        logger.warning("当前模型不支持特征重要性计算，跳过特征选择")
        return X_train, X_val, preprocessor.feature_names_
    
    logger.info(f"基于特征重要性进行特征选择，阈值: {threshold}")
    
    # 获取特征重要性
    importance = trainer.get_feature_importance(model)
    
    # 选择重要性高于阈值的特征
    selected_indices = [i for i, imp in enumerate(importance) if imp >= threshold]
    
    if not selected_indices:
        logger.warning("没有特征的重要性高于阈值，使用所有特征")
        return X_train, X_val, preprocessor.feature_names_
    
    # 筛选特征
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    
    # 更新特征名称
    selected_features = [preprocessor.feature_names_[i] for i in selected_indices]
    
    logger.info(f"选择了 {len(selected_features)} 个特征")
    logger.info(f"选择的特征: {selected_features}")
    
    return X_train_selected, X_val_selected, selected_features


def train_ensemble_model(args: argparse.Namespace,
                         config: Dict[str, Any],
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> EnsembleModel:
    """
    训练集成模型
    
    Args:
        args: 命令行参数
        config: 配置字典
        X_train: 训练集特征
        y_train: 训练集目标
        X_val: 验证集特征
        y_val: 验证集目标
        
    Returns:
        训练好的集成模型
    """
    logger.info("训练集成模型...")
    
    # 确定基础模型
    base_models = args.ensemble_models
    if not base_models:
        # 使用默认模型
        base_models = config.get('models', {}).get('ensemble', {}).get('base_models', 
                                                                     ['random_forest', 'xgboost', 'lightgbm'])
    
    logger.info(f"使用基础模型: {base_models}")
    
    # 创建集成模型
    ensemble = EnsembleModel(base_models=base_models, random_state=args.random_state)
    
    # 训练集成模型
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    logger.info("集成模型训练完成")
    
    # 保存各基础模型
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints', 'ensemble_models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for name, model in ensemble.base_models.items():
        model_path = os.path.join(checkpoint_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"保存基础模型: {model_path}")
    
    return ensemble


def generate_training_report(args: argparse.Namespace,
                            metrics: Dict[str, float],
                            model_dir: str) -> None:
    """
    生成训练报告
    
    Args:
        args: 命令行参数
        metrics: 评估指标
        model_dir: 模型保存目录
    """
    report = [
        "# 模型训练报告",
        f"生成时间: {datetime.now().isoformat()}",
        f"模型类型: {args.model_type}",
        f"训练数据集: {args.data_path}",
        f"目标变量: {args.target_column}",
        "\n## 评估指标",
    ]
    
    for metric_name, metric_value in metrics.items():
        report.append(f"- **{metric_name}**: {metric_value:.6f}")
    
    report.append("\n## 训练参数")
    report.append(f"- **随机种子**: {args.random_state}")
    report.append(f"- **交叉验证折数**: {args.n_folds}")
    report.append(f"- **最大轮数**: {args.max_epochs}")
    report.append(f"- **早停耐心值**: {args.patience}")
    report.append(f"- **学习率**: {args.learning_rate}")
    
    if args.hyperopt:
        report.append(f"- **超参数优化**: 启用 ({args.n_trials} 轮)")
    
    report.append(f"\n## 模型位置")
    report.append(f"模型文件保存在: {model_dir}")
    
    # 保存报告
    report_file = os.path.join(model_dir, 'training_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"训练报告已保存到: {report_file}")
    
    # 打印报告
    print("\n" + "=" * 80)
    print('\n'.join(report))
    print("=" * 80)


def main() -> None:
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        import logging as lg
        lg.getLogger().setLevel(lg.DEBUG)
    
    try:
        # 加载配置
        config_manager = ConfigManager(args.config)
        config = config_manager.get_full_config()
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 加载和预处理数据
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_preprocess_data(args, config)
        
        # 训练模型
        model = None
        trainer = None
        
        if args.model_type == 'ensemble':
            # 训练集成模型
            model = train_ensemble_model(args, config, X_train, y_train, X_val, y_val)
            # 创建一个临时训练器用于评估
            trainer = ModelTrainer(model_type='ensemble', random_state=args.random_state)
        else:
            # 创建和训练单一模型
            trainer = create_model_trainer(args, config)
            model = train_model(trainer, X_train, y_train, X_val, y_val, args)
        
        # 特征重要性筛选（如果启用）
        if args.feature_importance_threshold and not args.model_type == 'ensemble':
            X_train, X_val, selected_features = perform_feature_selection(
                trainer, model, X_train, y_train, X_val, y_val, preprocessor, args.feature_importance_threshold
            )
            # 更新测试集特征
            X_test = X_test[:, [preprocessor.feature_names_.index(f) for f in selected_features]]
            # 更新预处理器的特征名称
            preprocessor.feature_names_ = selected_features
            # 重新训练模型
            logger.info("使用选择的特征重新训练模型...")
            model = train_model(trainer, X_train, y_train, X_val, y_val, args)
        
        # 评估模型
        metrics = evaluate_model(model, trainer, X_test, y_test, args, args.output_dir)
        
        # 保存模型
        model_dir = save_model(model, trainer, preprocessor, args, metrics)
        
        # 生成训练报告
        generate_training_report(args, metrics, model_dir)
        
        logger.info("训练流程完成")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()