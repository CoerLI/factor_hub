# 使用指南

本文档提供了时间序列预测与交易因子分析框架的详细使用说明，包括数据处理、因子计算、回测和模型训练等核心功能的使用方法。

## 目录

- [数据处理模块](#数据处理模块)
- [因子库模块](#因子库模块)
- [回测系统模块](#回测系统模块)
- [机器学习模型模块](#机器学习模型模块)
- [配置管理](#配置管理)
- [高级用法](#高级用法)

## 数据处理模块

### 数据加载

```python
from src.data.data_loader import DataLoader

# 从CSV文件加载数据
data_loader = DataLoader()
data = data_loader.load_from_csv('data/sample_data.csv', date_column='date')

# 从数据库加载数据
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'timeseries_db',
    'user': 'user',
    'password': 'password'
}
data = data_loader.load_from_database('stock_prices', db_config, date_column='date')

# 从API加载数据
api_config = {
    'api_key': 'your_api_key',
    'endpoint': 'https://api.example.com/stocks'
}
data = data_loader.load_from_api(api_config, symbol='AAPL', start_date='2020-01-01', end_date='2022-12-31')
```

### 数据预处理

```python
from src.data.data_preprocessor import DataPreprocessor

# 创建数据预处理器
preprocessor = DataPreprocessor()

# 处理缺失值
data = preprocessor.handle_missing_values(data, method='ffill')

# 处理异常值
data = preprocessor.handle_outliers(data, columns=['open', 'high', 'low', 'close', 'volume'], method='iqr')

# 重采样数据（转换频率）
data_daily = preprocessor.resample(data, freq='D', agg_dict={'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})

# 创建滞后特征
data_with_lags = preprocessor.create_lag_features(data, columns=['close'], lags=[1, 5, 10, 20])

# 创建滚动窗口特征
data_with_rolling = preprocessor.create_rolling_features(
    data, 
    columns=['close'], 
    windows=[20, 50], 
    functions=['mean', 'std', 'min', 'max']
)

# 创建时间特征
data_with_time_features = preprocessor.create_time_features(data)

# 标准化数据
data_scaled, scaler = preprocessor.scale_features(
    data, 
    columns=['open', 'high', 'low', 'close', 'volume'], 
    method='minmax'
)
```

## 因子库模块

### 因子计算

```python
from src.factors.factor_calculator import FactorCalculator

# 创建因子计算器
calculator = FactorCalculator()

# 计算技术指标因子
data = calculator.calculate_moving_average(data, window=20, column='close', output_column='ma20')
data = calculator.calculate_rsi(data, window=14, column='close', output_column='rsi14')
data = calculator.calculate_macd(data, fast=12, slow=26, signal=9, column='close')
data = calculator.calculate_bollinger_bands(data, window=20, num_std=2, column='close')

# 计算统计因子
data = calculator.calculate_volatility(data, window=20, column='close', output_column='volatility20')
data = calculator.calculate_skewness(data, window=20, column='close', output_column='skewness20')
data = calculator.calculate_kurtosis(data, window=20, column='close', output_column='kurtosis20')

# 计算自定义因子
def custom_factor(series, param1=1, param2=2):
    return series.rolling(param1).mean() * param2

data = calculator.calculate_custom_factor(data, column='close', custom_func=custom_factor, output_column='custom_factor', param1=5, param2=3)
```

### 因子分析

```python
from src.factors.factor_analysis import FactorAnalyzer

# 创建因子分析器
analyzer = FactorAnalyzer()

# 分析因子表现
performance = analyzer.analyze_performance(
    data, 
    factor_column='factor', 
    price_column='close', 
    holding_period=5
)

# 因子IC分析
ic_results = analyzer.calculate_ic(data, factor_column='factor', price_column='close', lookback=1)

# 因子相关性分析
corr_matrix = analyzer.calculate_correlation(data, factor_columns=['factor1', 'factor2', 'factor3'])

# 因子分层分析
tier_results = analyzer.analyze_factor_tiers(
    data, 
    factor_column='factor', 
    price_column='close', 
    num_tiers=5, 
    holding_period=5
)

# 生成因子分析报告
analyzer.generate_report(
    data, 
    factor_columns=['factor1', 'factor2'], 
    price_column='close', 
    output_file='factor_analysis_report.html'
)
```

### 因子验证

```python
from src.factors.factor_analysis import FactorValidator

# 创建因子验证器
validator = FactorValidator()

# 进行统计显著性检验
significance_results = validator.test_statistical_significance(
    data, 
    factor_column='factor', 
    price_column='close'
)

# 进行稳健性检验
robustness_results = validator.test_robustness(
    data, 
    factor_column='factor', 
    price_column='close', 
    test_splits=5
)

# 进行样本外测试
out_of_sample_results = validator.test_out_of_sample(
    data, 
    factor_column='factor', 
    price_column='close', 
    test_ratio=0.3
)
```

## 回测系统模块

### 事件驱动回测

```python
from src.backtest.event_driven_backtest import EventDrivenBacktest
from src.backtest.strategy_implementations import MovingAverageCrossStrategy

# 创建回测引擎
backtest = EventDrivenBacktest(
    data=data, 
    initial_capital=100000, 
    commission=0.001,
    slippage=0.0005
)

# 创建并添加策略
strategy = MovingAverageCrossStrategy(
    data=data, 
    short_window=50, 
    long_window=200,
    symbol='AAPL'
)
backtest.add_strategy(strategy)

# 运行回测
backtest.run()

# 获取结果
results = backtest.get_results()

# 生成绩效报告
backtest.generate_report(output_file='backtest_report.html')
```

### 策略实现

```python
from src.backtest.strategy_base import StrategyBase
import numpy as np

class MyCustomStrategy(StrategyBase):
    def __init__(self, data, lookback_period=20, threshold=0.01):
        super().__init__(data)
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signals(self):
        # 计算信号
        self.data['returns'] = self.data['close'].pct_change()
        self.data['signal'] = 0
        
        # 简单动量策略
        self.data['momentum'] = self.data['close'] / self.data['close'].shift(self.lookback_period) - 1
        self.data.loc[self.data['momentum'] > self.threshold, 'signal'] = 1
        self.data.loc[self.data['momentum'] < -self.threshold, 'signal'] = -1
        
        return self.data['signal']

# 使用自定义策略
custom_strategy = MyCustomStrategy(data, lookback_period=20, threshold=0.01)
backtest.add_strategy(custom_strategy)
```

### 参数优化

```python
from src.backtest.backtest_runner import BacktestRunner

# 创建回测执行器
runner = BacktestRunner()

# 定义参数网格
param_grid = {
    'short_window': [20, 30, 50],
    'long_window': [100, 150, 200]
}

# 运行参数优化
optimization_results = runner.optimize_parameters(
    strategy_class=MovingAverageCrossStrategy,
    data=data,
    param_grid=param_grid,
    metric='sharpe_ratio',
    maximize=True
)

# 获取最佳参数
best_params = optimization_results['best_params']
print(f"最佳参数: {best_params}")
print(f"最佳指标值: {optimization_results['best_score']}")
```

## 机器学习模型模块

### 数据预处理

```python
from src.models.data_preprocessing import TimeSeriesDataLoader

# 创建时间序列数据加载器
data_loader = TimeSeriesDataLoader()

# 准备训练数据
X_train, X_test, y_train, y_test = data_loader.prepare_data(
    data=data,
    target_column='close',
    feature_columns=['open', 'high', 'low', 'volume', 'ma20', 'rsi14'],
    test_size=0.2,
    shuffle=False
)

# 创建特征工程流水线
pipeline = data_loader.create_preprocessing_pipeline(
    steps=[
        ('imputer', 'mean'),
        ('scaler', 'standard'),
        ('feature_selection', {'method': 'select_k_best', 'k': 5})
    ]
)

# 应用特征工程
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

### 模型训练与评估

```python
from src.models.model_training import ModelTrainer

# 创建模型训练器
trainer = ModelTrainer()

# 训练模型
model = trainer.train_model(
    model_type='xgboost',
    X_train=X_train_processed,
    y_train=y_train,
    params={'max_depth': 6, 'n_estimators': 100}
)

# 评估模型
metrics = trainer.evaluate_model(
    model=model,
    X_test=X_test_processed,
    y_test=y_test,
    metrics=['mse', 'mae', 'r2', 'mape']
)
print(f"模型评估指标: {metrics}")

# 超参数调优
best_params = trainer.hyperparameter_tuning(
    model_type='xgboost',
    X_train=X_train_processed,
    y_train=y_train,
    param_grid={
        'max_depth': [3, 6, 9],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    cv=5,
    scoring='neg_mean_squared_error'
)

# 模型集成
ensemble_model = trainer.create_ensemble(
    model_types=['xgboost', 'random_forest', 'linear_regression'],
    X_train=X_train_processed,
    y_train=y_train,
    ensemble_method='weighted',
    weights=[0.4, 0.3, 0.3]
)

# 保存模型
trainer.save_model(model, 'models/trained_model.pkl')
```

### 模型预测

```python
import numpy as np
from src.models.model_training import ModelTrainer

# 加载模型
trainer = ModelTrainer()
model = trainer.load_model('models/trained_model.pkl')

# 单样本预测
new_data = np.array([[100.5, 101.2, 99.8, 1000000, 100.0, 55.0]])
prediction = trainer.predict(model, new_data)

# 批量预测
batch_data = np.array([
    [100.5, 101.2, 99.8, 1000000, 100.0, 55.0],
    [101.0, 101.5, 100.2, 950000, 100.5, 58.0]
])
predictions = trainer.predict(model, batch_data)

# 特征重要性分析
feature_importance = trainer.get_feature_importance(model, feature_names=['open', 'high', 'low', 'volume', 'ma20', 'rsi14'])
print(f"特征重要性: {feature_importance}")
```

## 配置管理

```python
from src.config.config_manager import ConfigManager

# 创建配置管理器
config_manager = ConfigManager()

# 加载配置
config = config_manager.load_config('config/config.yaml')

# 获取特定配置
model_config = config_manager.get_config('model')
data_config = config_manager.get_config('data')

# 更新配置
config_manager.update_config('model.max_depth', 10)

# 保存配置
config_manager.save_config('config/updated_config.yaml')

# 验证配置
is_valid = config_manager.validate_config()
if not is_valid:
    print("配置验证失败")
```

## 高级用法

### 自定义因子和策略组合

```python
from src.factors.factor_calculator import FactorCalculator
from src.backtest.strategy_base import FactorBasedStrategy

# 创建因子计算器
calculator = FactorCalculator()

# 计算多个因子
data = calculator.calculate_moving_average(data, window=20, column='close', output_column='ma20')
data = calculator.calculate_rsi(data, window=14, column='close', output_column='rsi14')
data = calculator.calculate_macd(data, fast=12, slow=26, signal=9, column='close')
data['composite_factor'] = 0.5 * data['ma20'] / data['close'] + 0.3 * data['rsi14'] / 100 + 0.2 * data['macd']

# 创建基于组合因子的策略
factor_strategy = FactorBasedStrategy(
    data=data,
    factor_columns=['composite_factor'],
    weights=[1.0],
    lookback=252,  # 使用过去一年的数据进行标准化
    top_n=3,       # 选择因子值最高的3个资产
    rebalance_frequency='M'  # 每月重新平衡
)
```

### 多资产回测

```python
from src.backtest.event_driven_backtest import EventDrivenBacktest
from src.backtest.strategy_base import MultiAssetStrategy
import pandas as pd

# 假设我们有多个资产的数据
asset_data = {
    'AAPL': aapl_data,
    'MSFT': msft_data,
    'GOOGL': googl_data
}

# 创建多资产策略
multi_strategy = MultiAssetStrategy(
    asset_data=asset_data,
    allocation_method='equal_weight'
)

# 创建回测引擎
backtest = EventDrivenBacktest(
    data=pd.concat(asset_data.values()),
    initial_capital=100000,
    commission=0.001,
    multi_asset=True
)

backtest.add_strategy(multi_strategy)
backtest.run()
```

### 集成模型服务

```python
from src.models.model_service import TimeSeriesModelService

# 创建模型服务
service = TimeSeriesModelService()

# 加载数据
service.load_data('data/sample_data.csv')

# 预处理数据
service.preprocess_data(target_column='close', feature_columns=['open', 'high', 'low', 'volume'])

# 训练多个模型
models = {
    'xgboost': {'model_type': 'xgboost', 'params': {'max_depth': 6, 'n_estimators': 100}},
    'rf': {'model_type': 'random_forest', 'params': {'n_estimators': 100}},
    'lr': {'model_type': 'linear_regression', 'params': {}}
}

service.train_models(models)

# 比较模型
comparison = service.compare_models(metrics=['mse', 'mae', 'r2'])
print(f"模型比较: {comparison}")

# 部署最佳模型
service.deploy_best_model('models/best_model.pkl')

# 启动预测服务
service.start_prediction_service(port=8000)
```

### 自动化工作流

```python
import yaml
from src.config.config_manager import ConfigManager
from src.data.data_loader import DataLoader
from src.factors.factor_calculator import FactorCalculator
from src.backtest.backtest_runner import BacktestRunner
from src.models.model_training import ModelTrainer

# 加载配置
config_manager = ConfigManager()
config = config_manager.load_config('config/config.yaml')

# 1. 数据加载
data_loader = DataLoader()
data = data_loader.load_from_csv(config['data']['path'])

# 2. 计算因子
calculator = FactorCalculator()
for factor_config in config['factors']['to_calculate']:
    factor_type = factor_config['type']
    params = factor_config['params']
    calculator.calculate_factor(data, factor_type=factor_type, **params)

# 3. 运行回测
runner = BacktestRunner()
backtest_results = runner.run_backtest(
    strategy_class=config['backtest']['strategy'],
    data=data,
    params=config['backtest']['params']
)

# 4. 训练预测模型
trainer = ModelTrainer()
model = trainer.train_model(
    model_type=config['model']['type'],
    X_train=X_train,
    y_train=y_train,
    params=config['model']['params']
)

# 5. 保存结果
trainer.save_model(model, config['model']['save_path'])
runner.save_results(backtest_results, config['backtest']['results_path'])
```