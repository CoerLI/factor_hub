import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings

from src.factors.factor_base import AlphaFactor, RiskFactor
from src.utils.helpers import setup_logger, DEFAULT_LOGGER

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class CointegrationFactor(AlphaFactor):
    """
    协整因子
    
    检测两个时间序列之间的协整关系，用于统计套利策略
    """
    
    def __init__(self, 
                 pair_names: Tuple[str, str],
                 window: int = 252,
                 test_method: str = 'engle_granger',
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化协整因子
        
        Args:
            pair_names: 配对的两个资产名称
            window: 协整测试窗口大小
            test_method: 协整测试方法，目前支持'engle_granger'
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"Cointegration_{'_'.join(pair_names)}_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.pair_names = pair_names
        self.window = window
        self.test_method = test_method.lower()
        self.price_col = price_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
        
        if self.test_method not in ['engle_granger']:
            raise ValueError("目前仅支持 'engle_granger' 协整测试方法")
    
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算协整因子
        
        Args:
            data: 字典，包含两个资产的时间序列数据
            
        Returns:
            包含协整统计量和交易信号的DataFrame
        """
        # 验证数据
        asset1, asset2 = self.pair_names
        if asset1 not in data or asset2 not in data:
            raise ValueError(f"数据中缺少 {asset1} 或 {asset2}")
        
        for asset in [asset1, asset2]:
            if self.price_col not in data[asset].columns:
                raise ValueError(f"{asset} 数据中缺少 {self.price_col} 列")
        
        # 对齐两个时间序列
        df1 = data[asset1].copy()
        df2 = data[asset2].copy()
        
        # 确保时间索引一致
        aligned_index = df1.index.intersection(df2.index)
        df1 = df1.loc[aligned_index]
        df2 = df2.loc[aligned_index]
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=aligned_index)
        result[f"{asset1}_{self.price_col}"] = df1[self.price_col]
        result[f"{asset2}_{self.price_col}"] = df2[self.price_col]
        
        # 计算滚动协整统计量
        for i in range(self.window, len(result) + 1):
            # 获取窗口内数据
            window_data1 = df1[self.price_col].iloc[i - self.window:i]
            window_data2 = df2[self.price_col].iloc[i - self.window:i]
            
            try:
                # 执行简单线性回归：y = beta * x + alpha
                x = sm.add_constant(window_data1)
                model = sm.OLS(window_data2, x).fit()
                beta = model.params[1]  # 斜率
                alpha = model.params[0]  # 截距
                
                # 计算残差
                residual = window_data2 - (alpha + beta * window_data1)
                
                # 对残差进行ADF检验
                adf_result = sm.tsa.adfuller(residual, autolag='AIC')
                adf_stat = adf_result[0]  # ADF统计量
                p_value = adf_result[1]  # p值
                
                # 计算Z-score标准化的残差
                residual_mean = residual.mean()
                residual_std = residual.std()
                z_score = (residual.iloc[-1] - residual_mean) / (residual_std + 1e-10)
                
                # 存储结果
                current_idx = result.index[i - 1]
                result.loc[current_idx, f"{self.name}_beta"] = beta
                result.loc[current_idx, f"{self.name}_alpha"] = alpha
                result.loc[current_idx, f"{self.name}_adf"] = adf_stat
                result.loc[current_idx, f"{self.name}_pvalue"] = p_value
                result.loc[current_idx, f"{self.name}_zscore"] = z_score
                
                # 生成交易信号：z_score > 2表示高估，z_score < -2表示低估
                # 信号为1表示买入价差，-1表示卖出价差
                if z_score > 2:
                    result.loc[current_idx, f"{self.name}_signal"] = -1
                elif z_score < -2:
                    result.loc[current_idx, f"{self.name}_signal"] = 1
                else:
                    result.loc[current_idx, f"{self.name}_signal"] = 0
                
                # 标准化因子值（使用z_score）
                result.loc[current_idx, f"{self.name}_norm"] = z_score
                
            except Exception as e:
                # 如果计算失败，填充NaN
                current_idx = result.index[i - 1]
                result.loc[current_idx, f"{self.name}_beta"] = np.nan
                result.loc[current_idx, f"{self.name}_alpha"] = np.nan
                result.loc[current_idx, f"{self.name}_adf"] = np.nan
                result.loc[current_idx, f"{self.name}_pvalue"] = np.nan
                result.loc[current_idx, f"{self.name}_zscore"] = np.nan
                result.loc[current_idx, f"{self.name}_signal"] = 0
                result.loc[current_idx, f"{self.name}_norm"] = 0
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "pair_names": self.pair_names,
            "window": self.window,
            "test_method": self.test_method,
            "price_col": self.price_col
        })
        return config


class MarketMicrostructureFactor(AlphaFactor):
    """
    市场微结构因子
    
    计算买卖价差、交易量分布等市场微观结构特征
    """
    
    def __init__(self, 
                 window: int = 20,
                 volume_col: str = 'volume',
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化市场微结构因子
        
        Args:
            window: 计算周期
            volume_col: 使用的成交量列名
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"Microstructure_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.volume_col = volume_col
        self.price_col = price_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场微结构因子
        
        Args:
            data: 输入数据，必须包含开盘价、最高价、最低价、收盘价和成交量
            
        Returns:
            包含市场微结构因子的DataFrame
        """
        # 验证数据
        required_cols = [self.price_col, self.volume_col, 'high', 'low', 'open']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据中缺少 {col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col, self.volume_col]].copy()
        
        # 1. 计算买卖价差的替代指标（使用价格波动）
        result[f"{self.name}_spread"] = (data['high'] - data['low']) / data['close']
        
        # 2. 价格影响系数（价格变化与成交量的关系）
        price_change = data[self.price_col].pct_change()
        volume_change = data[self.volume_col].pct_change()
        
        # 计算两者的滚动相关系数
        combined = pd.DataFrame({
            'price_change': price_change,
            'volume_change': volume_change
        })
        result[f"{self.name}_impact"] = combined.rolling(window=self.window).corr().unstack()
            .loc[:, ('price_change', 'volume_change')]
        
        # 3. 价格反转概率（基于前一天的价格变化）
        prev_return = price_change.shift(1)
        current_return = price_change
        # 反转指示符：前一天上涨，今天下跌，或者前一天下跌，今天上涨
        reversal = ((prev_return > 0) & (current_return < 0)) | ((prev_return < 0) & (current_return > 0))
        result[f"{self.name}_reversal"] = reversal.rolling(window=self.window).mean()
        
        # 4. 价格连续性（连续上涨或下跌的平均长度）
        direction = np.sign(price_change)
        # 计算连续相同方向的周期数
        run_length = []
        current_run = 1
        for i in range(1, len(direction)):
            if direction.iloc[i] == direction.iloc[i-1] and direction.iloc[i] != 0:
                current_run += 1
            else:
                if direction.iloc[i-1] != 0:  # 避免记录零变化
                    run_length.append(current_run)
                current_run = 1
        
        # 计算滚动平均连续长度
        run_length_series = pd.Series([np.nan] * (len(data) - len(run_length)) + run_length, 
                                     index=data.index[len(data) - len(run_length):])
        result[f"{self.name}_run_length"] = run_length_series.reindex(data.index).fillna(0)
        result[f"{self.name}_avg_run"] = result[f"{self.name}_run_length"].rolling(window=self.window).mean()
        
        # 5. 成交量分布偏度（衡量成交量分布的不对称性）
        result[f"{self.name}_volume_skew"] = data[self.volume_col].rolling(window=self.window).skew()
        
        # 6. 标准化的流动性指标
        # 价格波动与成交量的比率
        volatility = data[self.price_col].rolling(window=self.window).std()
        avg_volume = data[self.volume_col].rolling(window=self.window).mean()
        result[f"{self.name}_liquidity"] = volatility / (avg_volume + 1e-10)
        
        # 标准化因子值
        liq_mean = result[f"{self.name}_liquidity"].rolling(window=self.window).mean()
        liq_std = result[f"{self.name}_liquidity"].rolling(window=self.window).std()
        result[f"{self.name}_norm"] = (result[f"{self.name}_liquidity"] - liq_mean) / (liq_std + 1e-10)
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "window": self.window,
            "volume_col": self.volume_col,
            "price_col": self.price_col
        })
        return config


class FundamentalFactor(AlphaFactor):
    """
    基本面因子
    
    结合财务数据和市场数据计算基本面相关指标
    """
    
    def __init__(self, 
                 factor_type: str = 'pe_ratio',
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化基本面因子
        
        Args:
            factor_type: 因子类型，支持'pe_ratio'(市盈率), 'pb_ratio'(市净率), 'ps_ratio'(市销率), 'dividend_yield'(股息率)
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"Fundamental_{factor_type.upper()}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.factor_type = factor_type.lower()
        self.price_col = price_col
        
        # 验证参数
        valid_types = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'dividend_yield']
        if self.factor_type not in valid_types:
            raise ValueError(f"factor_type 必须是 {', '.join(valid_types)} 之一")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本面因子
        
        Args:
            data: 输入数据，必须包含价格列和相应的财务数据列
            
        Returns:
            包含基本面因子的DataFrame
        """
        # 验证必要的价格列
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        try:
            if self.factor_type == 'pe_ratio':
                # 市盈率 = 股价 / 每股收益
                if 'earnings_per_share' not in data.columns:
                    raise ValueError("数据中缺少 'earnings_per_share' 列")
                
                # 计算市盈率
                pe = data[self.price_col] / data['earnings_per_share']
                result[self.name] = pe
                
                # 计算市盈率与行业平均水平的偏离度
                if 'industry_pe_avg' in data.columns:
                    result[f"{self.name}_rel"] = pe / data['industry_pe_avg']
                
            elif self.factor_type == 'pb_ratio':
                # 市净率 = 股价 / 每股净资产
                if 'book_value_per_share' not in data.columns:
                    raise ValueError("数据中缺少 'book_value_per_share' 列")
                
                pb = data[self.price_col] / data['book_value_per_share']
                result[self.name] = pb
                
                # 计算市净率与行业平均水平的偏离度
                if 'industry_pb_avg' in data.columns:
                    result[f"{self.name}_rel"] = pb / data['industry_pb_avg']
                
            elif self.factor_type == 'ps_ratio':
                # 市销率 = 股价 / 每股销售额
                if 'sales_per_share' not in data.columns:
                    raise ValueError("数据中缺少 'sales_per_share' 列")
                
                ps = data[self.price_col] / data['sales_per_share']
                result[self.name] = ps
                
                # 计算市销率与行业平均水平的偏离度
                if 'industry_ps_avg' in data.columns:
                    result[f"{self.name}_rel"] = ps / data['industry_ps_avg']
                
            elif self.factor_type == 'dividend_yield':
                # 股息率 = 每股股息 / 股价
                if 'dividend_per_share' not in data.columns:
                    raise ValueError("数据中缺少 'dividend_per_share' 列")
                
                dividend_yield = data['dividend_per_share'] / data[self.price_col]
                result[self.name] = dividend_yield * 100  # 转换为百分比
                
                # 计算股息率与行业平均水平的偏离度
                if 'industry_dividend_yield_avg' in data.columns:
                    result[f"{self.name}_rel"] = dividend_yield / (data['industry_dividend_yield_avg'] / 100)
            
            # 对因子值进行Z-score标准化
            factor_mean = result[self.name].rolling(window=252).mean()  # 使用一年的数据
            factor_std = result[self.name].rolling(window=252).std()
            result[f"{self.name}_norm"] = (result[self.name] - factor_mean) / (factor_std + 1e-10)
            
            # 对于估值类因子（PE, PB, PS），反转标准化值，因为低估值通常被视为正向信号
            if self.factor_type in ['pe_ratio', 'pb_ratio', 'ps_ratio']:
                result[f"{self.name}_norm"] = -result[f"{self.name}_norm"]
                
        except Exception as e:
            logging.error(f"计算基本面因子时出错: {e}")
            result[f"{self.name}_norm"] = np.nan
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "factor_type": self.factor_type,
            "price_col": self.price_col
        })
        return config


class MachineLearningFactor(AlphaFactor):
    """
    机器学习因子
    
    使用机器学习模型预测价格走势，生成信号
    """
    
    def __init__(self, 
                 model: Callable,
                 features: List[str],
                 lookback: int = 20,
                 horizon: int = 5,
                 name: str = None,
                 **kwargs):
        """
        初始化机器学习因子
        
        Args:
            model: 机器学习模型，需要实现fit和predict方法
            features: 用于预测的特征列表
            lookback: 特征构建的回看期
            horizon: 预测的时间窗口
            name: 因子名称
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        if name is None:
            name = f"MLFactor_{model.__class__.__name__}_{lookback}_{horizon}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.model = model
        self.features = features
        self.lookback = lookback
        self.horizon = horizon
        
        # 验证参数
        if self.lookback <= 0:
            raise ValueError("lookback 必须大于0")
        
        if self.horizon <= 0:
            raise ValueError("horizon 必须大于0")
            
        if not hasattr(self.model, 'fit') or not hasattr(self.model, 'predict'):
            raise ValueError("model 必须实现 fit 和 predict 方法")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算机器学习因子
        
        Args:
            data: 输入数据，必须包含所有指定的特征
            
        Returns:
            包含机器学习预测结果的DataFrame
        """
        # 验证数据
        for feature in self.features:
            if feature not in data.columns:
                raise ValueError(f"数据中缺少特征 {feature}")
        
        if 'close' not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        # 创建结果DataFrame
        result = data[['close']].copy()
        
        # 准备特征和标签
        X = data[self.features].values
        
        # 计算未来收益率作为标签
        future_returns = data['close'].shift(-self.horizon) / data['close'] - 1
        y = future_returns.values
        
        # 初始化预测结果数组
        predictions = np.full(len(data), np.nan)
        
        try:
            # 滚动训练和预测
            for i in range(self.lookback, len(data) - self.horizon):
                # 使用lookback窗口的数据进行训练
                X_train = X[i - self.lookback:i]
                y_train = y[i - self.lookback:i]
                
                # 确保没有NaN值
                mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
                if mask.sum() < 10:  # 确保有足够的数据进行训练
                    continue
                    
                X_train = X_train[mask]
                y_train = y_train[mask]
                
                # 训练模型
                self.model.fit(X_train, y_train)
                
                # 预测当前时间点
                X_test = X[i].reshape(1, -1)
                pred = self.model.predict(X_test)[0]
                predictions[i] = pred
            
            # 将预测结果添加到DataFrame
            result[f"{self.name}_prediction"] = predictions
            
            # 生成交易信号（根据预测的收益率符号）
            result[f"{self.name}_signal"] = np.sign(predictions)
            
            # 对预测结果进行Z-score标准化
            pred_mean = pd.Series(predictions).rolling(window=self.lookback).mean()
            pred_std = pd.Series(predictions).rolling(window=self.lookback).std()
            result[f"{self.name}_norm"] = (pd.Series(predictions) - pred_mean) / (pred_std + 1e-10)
            
        except Exception as e:
            logging.error(f"计算机器学习因子时出错: {e}")
            result[f"{self.name}_norm"] = np.nan
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "model_type": self.model.__class__.__name__,
            "features": self.features,
            "lookback": self.lookback,
            "horizon": self.horizon
        })
        return config


class PCABasedFactor(AlphaFactor):
    """
    基于主成分分析(PCA)的因子
    
    使用PCA方法从多个因子中提取主要成分，减少特征维度并捕捉共同变化
    """
    
    def __init__(self, 
                 n_components: int = 3,
                 features: List[str] = None,
                 window: int = 252,
                 **kwargs):
        """
        初始化PCA因子
        
        Args:
            n_components: 要提取的主成分数量
            features: 用于PCA的特征列表，如果为None则使用所有可用特征
            window: 滚动PCA计算的窗口大小
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"PCA_{n_components}_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.n_components = n_components
        self.features = features
        self.window = window
        
        # 验证参数
        if self.n_components <= 0:
            raise ValueError("n_components 必须大于0")
        
        if self.window <= 0:
            raise ValueError("window 必须大于0")
        
        # 初始化PCA模型
        self.pca = PCA(n_components=self.n_components)
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算PCA因子
        
        Args:
            data: 输入数据，包含用于PCA的所有特征
            
        Returns:
            包含PCA主成分的DataFrame
        """
        # 确定要使用的特征
        if self.features is None:
            # 排除日期和价格等原始数据列
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'date', 'time']
            self.features = [col for col in data.columns if col.lower() not in exclude_cols]
        
        # 验证数据
        for feature in self.features:
            if feature not in data.columns:
                raise ValueError(f"数据中缺少特征 {feature}")
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=data.index)
        
        try:
            # 滚动PCA计算
            for i in range(self.window, len(data) + 1):
                # 获取窗口数据
                window_data = data[self.features].iloc[i - self.window:i].copy()
                
                # 处理缺失值
                window_data = window_data.fillna(window_data.mean())
                
                # 标准化数据
                window_data_std = (window_data - window_data.mean()) / (window_data.std() + 1e-10)
                
                # 执行PCA
                components = self.pca.fit_transform(window_data_std)
                
                # 获取解释方差比例
                explained_variance = self.pca.explained_variance_ratio_
                
                # 存储当前时间点的主成分值
                current_idx = data.index[i-1]
                for j in range(self.n_components):
                    result.loc[current_idx, f"{self.name}_PC{j+1}"] = components[-1, j]  # 取最后一个样本的结果
                    result.loc[current_idx, f"{self.name}_PC{j+1}_explained"] = explained_variance[j]
                
                # 计算综合得分（加权主成分）
                weights = explained_variance / explained_variance.sum()
                composite_score = np.dot(components[-1], weights)
                result.loc[current_idx, f"{self.name}_composite"] = composite_score
            
            # 标准化综合得分
            comp_mean = result[f"{self.name}_composite"].rolling(window=self.window).mean()
            comp_std = result[f"{self.name}_composite"].rolling(window=self.window).std()
            result[f"{self.name}_norm"] = (result[f"{self.name}_composite"] - comp_mean) / (comp_std + 1e-10)
            
        except Exception as e:
            logging.error(f"计算PCA因子时出错: {e}")
            result[f"{self.name}_norm"] = np.nan
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "n_components": self.n_components,
            "features": self.features,
            "window": self.window
        })
        return config


class GARCHVolatility(AlphaFactor):
    """
    GARCH波动率因子
    
    使用GARCH模型预测金融时间序列的波动率
    """
    
    def __init__(self, 
                 p: int = 1,
                 q: int = 1,
                 window: int = 252,
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化GARCH波动率因子
        
        Args:
            p: GARCH模型的ARCH项阶数
            q: GARCH模型的GARCH项阶数
            window: 模型估计的滚动窗口大小
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"GARCH_{p}_{q}_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.p = p
        self.q = q
        self.window = window
        self.price_col = price_col
        
        # 验证参数
        if self.p <= 0 or self.q <= 0:
            raise ValueError("p 和 q 必须大于0")
        
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算GARCH波动率因子
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含GARCH波动率预测的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        try:
            # 计算收益率
            returns = data[self.price_col].pct_change().dropna()
            
            # 初始化波动率预测数组
            garch_vol = np.full(len(data), np.nan)
            
            # 滚动GARCH估计和预测
            for i in range(self.window, len(returns)):
                # 获取窗口数据
                window_returns = returns.iloc[i - self.window:i].values
                
                # 使用简单的GARCH(1,1)模型实现
                # 这里使用简化版本，实际应用中可以使用arch库的GARCH模型
                try:
                    # 简化版GARCH(1,1)计算
                    # 初始化参数（可以使用更复杂的优化方法）
                    omega = np.var(window_returns) * 0.1  # 截距
                    alpha = 0.1  # ARCH项系数
                    beta = 0.85  # GARCH项系数
                    
                    # 计算条件方差
                    cond_var = np.zeros_like(window_returns)
                    cond_var[0] = np.var(window_returns)
                    
                    for t in range(1, len(window_returns)):
                        cond_var[t] = omega + alpha * window_returns[t-1]**2 + beta * cond_var[t-1]
                    
                    # 预测下一期波动率
                    predicted_vol = np.sqrt(omega + alpha * window_returns[-1]**2 + beta * cond_var[-1])
                    
                    # 存储结果
                    garch_vol[i] = predicted_vol
                    
                except Exception:
                    # 如果计算失败，使用NaN
                    continue
            
            # 将波动率预测结果添加到DataFrame
            result[f"{self.name}_vol"] = garch_vol
            
            # 计算波动率变化
            result[f"{self.name}_vol_change"] = result[f"{self.name}_vol"].pct_change()
            
            # 标准化波动率值
            vol_mean = result[f"{self.name}_vol"].rolling(window=self.window).mean()
            vol_std = result[f"{self.name}_vol"].rolling(window=self.window).std()
            result[f"{self.name}_norm"] = (result[f"{self.name}_vol"] - vol_mean) / (vol_std + 1e-10)
            
        except Exception as e:
            logging.error(f"计算GARCH波动率因子时出错: {e}")
            result[f"{self.name}_norm"] = np.nan
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "p": self.p,
            "q": self.q,
            "window": self.window,
            "price_col": self.price_col
        })
        return config


class SentimentFactor(AlphaFactor):
    """
    情绪因子
    
    基于市场情绪数据计算情绪指标，如波动率指数、恐慌与贪婪指数等
    """
    
    def __init__(self, 
                 sentiment_source: str = 'vix',
                 window: int = 20,
                 **kwargs):
        """
        初始化情绪因子
        
        Args:
            sentiment_source: 情绪数据源，支持'vix'(波动率指数)或'custom'(自定义情绪数据)
            window: 计算周期
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"Sentiment_{sentiment_source.upper()}_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.sentiment_source = sentiment_source.lower()
        self.window = window
        
        # 验证参数
        if self.sentiment_source not in ['vix', 'custom']:
            raise ValueError("sentiment_source 必须是 'vix' 或 'custom'")
        
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算情绪因子
        
        Args:
            data: 输入数据，必须包含情绪数据列
            
        Returns:
            包含情绪指标的DataFrame
        """
        # 确定情绪数据列名
        if self.sentiment_source == 'vix':
            sentiment_col = 'vix' if 'vix' in data.columns else 'volatility_index'
        else:
            sentiment_col = 'sentiment' if 'sentiment' in data.columns else 'sentiment_index'
        
        # 验证数据
        if sentiment_col not in data.columns:
            raise ValueError(f"数据中缺少情绪数据列 {sentiment_col}")
        
        # 创建结果DataFrame
        result = data[[sentiment_col]].copy()
        
        # 获取情绪数据
        sentiment_data = data[sentiment_col]
        
        # 1. 原始情绪值
        result[f"{self.name}_raw"] = sentiment_data
        
        # 2. 情绪值的移动平均
        result[f"{self.name}_ma"] = sentiment_data.rolling(window=self.window).mean()
        
        # 3. 情绪值的变化率
        result[f"{self.name}_roc"] = sentiment_data.pct_change()
        
        # 4. 情绪值与移动平均的偏离度
        result[f"{self.name}_deviation"] = (sentiment_data - result[f"{self.name}_ma"]) / result[f"{self.name}_ma"]
        
        # 5. 标准化的情绪值（Z-score）
        sentiment_mean = sentiment_data.rolling(window=self.window).mean()
        sentiment_std = sentiment_data.rolling(window=self.window).std()
        result[f"{self.name}_norm"] = (sentiment_data - sentiment_mean) / (sentiment_std + 1e-10)
        
        # 对于VIX等恐慌指数，反转标准化值，因为高VIX通常意味着市场恐慌（负面信号）
        if self.sentiment_source == 'vix':
            result[f"{self.name}_norm"] = -result[f"{self.name}_norm"]
            
        # 6. 极端情绪检测（超过2个标准差）
        extreme_threshold = 2.0
        result[f"{self.name}_extreme"] = (abs(result[f"{self.name}_norm"]) > extreme_threshold).astype(int)
        
        # 7. 情绪趋势
        result[f"{self.name}_trend"] = result[f"{self.name}_ma"].diff(self.window)
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "sentiment_source": self.sentiment_source,
            "window": self.window
        })
        return config


class TimeSeriesMomentum(AlphaFactor):
    """
    时间序列动量因子
    
    基于资产过去的表现预测未来表现，通常使用多时间尺度
    """
    
    def __init__(self, 
                 lookback_windows: List[int] = [1, 3, 6, 12],
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化时间序列动量因子
        
        Args:
            lookback_windows: 回看窗口列表（单位：月）
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        windows_str = '_'.join([str(w) for w in lookback_windows])
        name = f"TSMomentum_{windows_str}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.lookback_windows = lookback_windows
        self.price_col = price_col
        
        # 验证参数
        for window in self.lookback_windows:
            if window <= 0:
                raise ValueError("所有回看窗口必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算时间序列动量因子
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含时间序列动量指标的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 对于每个回看窗口，计算动量
        momentum_values = []
        
        for window in self.lookback_windows:
            # 将月转换为交易日（假设每月21个交易日）
            trading_days = window * 21
            
            # 计算动量（当前价格减去window期前的价格，除以window期前的价格）
            mom = (data[self.price_col] - data[self.price_col].shift(trading_days)) / \
                  data[self.price_col].shift(trading_days)
            
            # 添加到结果和动量值列表
            result[f"{self.name}_{window}m"] = mom
            momentum_values.append(mom)
        
        # 计算多时间尺度动量的平均值
        mom_avg = pd.concat(momentum_values, axis=1).mean(axis=1)
        result[f"{self.name}_avg"] = mom_avg
        
        # 标准化平均动量
        mom_mean = mom_avg.rolling(window=252).mean()  # 使用一年的数据
        mom_std = mom_avg.rolling(window=252).std()
        result[f"{self.name}_norm"] = (mom_avg - mom_mean) / (mom_std + 1e-10)
        
        # 计算动量信号（基于标准化值的符号）
        result[f"{self.name}_signal"] = np.sign(result[f"{self.name}_norm"])
        
        # 计算动量排名（相对历史表现）
        result[f"{self.name}_rank"] = result[f"{self.name}_avg"].rolling(window=252).apply(
            lambda x: stats.rankdata(x)[-1] / len(x) if len(x) > 0 else np.nan
        )
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "lookback_windows": self.lookback_windows,
            "price_col": self.price_col
        })
        return config


# 创建高级因子的便捷函数
def create_cointegration_factors(pair_list: List[Tuple[str, str]] = [('asset1', 'asset2')]) -> List[CointegrationFactor]:
    """
    创建一组协整因子
    
    Args:
        pair_list: 资产对列表
        
    Returns:
        协整因子列表
    """
    factors = []
    for pair in pair_list:
        factors.append(CointegrationFactor(pair_names=pair, window=252))
    return factors


def create_ml_factor(model: Callable,
                     features: List[str],
                     lookback: int = 20,
                     horizon: int = 5,
                     name: str = None) -> MachineLearningFactor:
    """
    创建机器学习因子
    
    Args:
        model: 机器学习模型
        features: 特征列表
        lookback: 回看期
        horizon: 预测窗口
        name: 因子名称
        
    Returns:
        机器学习因子
    """
    return MachineLearningFactor(
        model=model,
        features=features,
        lookback=lookback,
        horizon=horizon,
        name=name
    )


def create_pca_factor(n_components: int = 3,
                     features: List[str] = None,
                     window: int = 252) -> PCABasedFactor:
    """
    创建PCA因子
    
    Args:
        n_components: 主成分数量
        features: 特征列表
        window: 滚动窗口
        
    Returns:
        PCA因子
    """
    return PCABasedFactor(
        n_components=n_components,
        features=features,
        window=window
    )


def create_advanced_factors_set() -> List[AlphaFactor]:
    """
    创建一组高级因子集合
    
    Returns:
        高级因子列表
    """
    factors = []
    
    # 添加市场微结构因子
    factors.append(MarketMicrostructureFactor(window=20))
    
    # 添加基本面因子
    factors.append(FundamentalFactor(factor_type='pe_ratio'))
    factors.append(FundamentalFactor(factor_type='pb_ratio'))
    factors.append(FundamentalFactor(factor_type='dividend_yield'))
    
    # 添加波动率因子
    factors.append(GARCHVolatility(p=1, q=1, window=252))
    
    # 添加情绪因子
    factors.append(SentimentFactor(sentiment_source='vix', window=20))
    
    # 添加时间序列动量因子
    factors.append(TimeSeriesMomentum(lookback_windows=[1, 3, 6]))
    
    return factors
