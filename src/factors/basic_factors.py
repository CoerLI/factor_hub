import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
import os

from src.factors.factor_base import AlphaFactor, RiskFactor
from src.utils.helpers import setup_logger, DEFAULT_LOGGER


class MovingAverage(AlphaFactor):
    """
    移动平均线因子
    
    计算指定周期的简单移动平均线(SMA)或指数移动平均线(EMA)
    """
    
    def __init__(self, 
                 window: int = 20,
                 ma_type: str = 'sma',
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化移动平均线因子
        
        Args:
            window: 移动窗口大小
            ma_type: 移动平均类型，'sma' 或 'ema'
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"{ma_type.upper()}_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.ma_type = ma_type.lower()
        self.price_col = price_col
        
        # 验证参数
        if self.ma_type not in ['sma', 'ema']:
            raise ValueError("ma_type 必须是 'sma' 或 'ema'")
        
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算移动平均线
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含移动平均线值的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 计算移动平均线
        if self.ma_type == 'sma':
            # 简单移动平均线
            result[self.name] = data[self.price_col].rolling(window=self.window).mean()
        else:  # ema
            # 指数移动平均线
            result[self.name] = data[self.price_col].ewm(span=self.window, adjust=False).mean()
        
        # 计算与价格的偏离度（标准化因子值）
        result[f"{self.name}_norm"] = (data[self.price_col] - result[self.name]) / data[self.price_col]
        
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
            "ma_type": self.ma_type,
            "price_col": self.price_col
        })
        return config


class RSI(AlphaFactor):
    """
    相对强弱指标因子
    
    衡量价格变动的速度和幅度，用于识别超买超卖情况
    """
    
    def __init__(self, 
                 window: int = 14,
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化RSI因子
        
        Args:
            window: 计算周期
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"RSI_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.price_col = price_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含RSI值的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 计算价格变化
        delta = data[self.price_col].diff()
        
        # 分离涨跌
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        
        # 计算RS和RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 将RSI值添加到结果中
        result[self.name] = rsi
        
        # 标准化RSI值（将0-100映射到-1到1）
        result[f"{self.name}_norm"] = (rsi - 50) / 50
        
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
            "price_col": self.price_col
        })
        return config


class BollingerBands(AlphaFactor):
    """
    布林带因子
    
    由中轨（移动平均线）和上下轨（中轨±标准差倍数）组成
    """
    
    def __init__(self, 
                 window: int = 20,
                 num_std: float = 2.0,
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化布林带因子
        
        Args:
            window: 移动窗口大小
            num_std: 标准差倍数
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"BB_{window}_{num_std}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.num_std = num_std
        self.price_col = price_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
        
        if self.num_std <= 0:
            raise ValueError("num_std 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带指标
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含布林带指标的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 计算中轨（SMA）
        middle_band = data[self.price_col].rolling(window=self.window).mean()
        result[f"{self.name}_middle"] = middle_band
        
        # 计算标准差
        std = data[self.price_col].rolling(window=self.window).std()
        
        # 计算上轨和下轨
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        result[f"{self.name}_upper"] = upper_band
        result[f"{self.name}_lower"] = lower_band
        
        # 计算布林带宽度
        bb_width = (upper_band - lower_band) / middle_band
        result[f"{self.name}_width"] = bb_width
        
        # 计算价格在布林带中的位置（0-100）
        bb_position = 100 * (data[self.price_col] - lower_band) / (upper_band - lower_band)
        result[f"{self.name}_position"] = bb_position
        
        # 标准化的位置（-1到1）
        result[f"{self.name}_norm"] = (bb_position - 50) / 50
        
        # 计算突破信号：上突破=1，下突破=-1，否则=0
        upper_break = (data[self.price_col] > upper_band).astype(int)
        lower_break = (data[self.price_col] < lower_band).astype(int)
        break_signal = upper_break - lower_break
        result[f"{self.name}_signal"] = break_signal
        
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
            "num_std": self.num_std,
            "price_col": self.price_col
        })
        return config


class MACD(AlphaFactor):
    """
    MACD（移动平均线收敛/发散）因子
    
    由快线、慢线、信号线和柱状图组成
    """
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化MACD因子
        
        Args:
            fast_period: 快线EMA周期
            slow_period: 慢线EMA周期
            signal_period: 信号线EMA周期
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"MACD_{fast_period}_{slow_period}_{signal_period}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_col = price_col
        
        # 验证参数
        if self.fast_period <= 0 or self.slow_period <= 0 or self.signal_period <= 0:
            raise ValueError("所有周期参数必须大于0")
        
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period 必须小于 slow_period")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含MACD指标的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 计算快线和慢线EMA
        fast_ema = data[self.price_col].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.price_col].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算MACD线（快线减慢线）
        macd_line = fast_ema - slow_ema
        result[f"{self.name}_line"] = macd_line
        
        # 计算信号线（MACD线的EMA）
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        result[f"{self.name}_signal"] = signal_line
        
        # 计算柱状图（MACD线减信号线）
        histogram = macd_line - signal_line
        result[f"{self.name}_histogram"] = histogram
        
        # 计算标准化的MACD值（使用Z-score标准化）
        hist_mean = histogram.rolling(window=20).mean()
        hist_std = histogram.rolling(window=20).std()
        result[f"{self.name}_norm"] = (histogram - hist_mean) / hist_std
        
        # 计算交叉信号：金叉=1，死叉=-1，否则=0
        # 当前柱状图为正，前一柱状图为负 -> 金叉
        # 当前柱状图为负，前一柱状图为正 -> 死叉
        prev_histogram = histogram.shift(1)
        cross_signal = ((histogram > 0) & (prev_histogram < 0)).astype(int) - \
                      ((histogram < 0) & (prev_histogram > 0)).astype(int)
        result[f"{self.name}_cross"] = cross_signal
        
        return result
    
    def get_config(self) -> Dict[str, any]:
        """
        获取因子配置
        
        Returns:
            配置字典
        """
        config = super().get_config()
        config.update({
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "price_col": self.price_col
        })
        return config


class VolumeFactors(AlphaFactor):
    """
    成交量相关因子
    
    计算成交量的各种统计特征和变化
    """
    
    def __init__(self, 
                 window: int = 20,
                 volume_col: str = 'volume',
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化成交量因子
        
        Args:
            window: 移动窗口大小
            volume_col: 使用的成交量列名
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"VolumeStats_{window}"
        
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
        计算成交量相关因子
        
        Args:
            data: 输入数据，必须包含指定的成交量和价格列
            
        Returns:
            包含成交量因子的DataFrame
        """
        # 验证数据
        for col in [self.volume_col, self.price_col]:
            if col not in data.columns:
                raise ValueError(f"数据中缺少 {col} 列")
        
        # 创建结果DataFrame
        result = data[[self.volume_col, self.price_col]].copy()
        
        # 1. 成交量变化率
        result[f"{self.name}_change"] = data[self.volume_col].pct_change()
        
        # 2. 成交量的移动平均
        result[f"{self.name}_ma"] = data[self.volume_col].rolling(window=self.window).mean()
        
        # 3. 成交量的标准化值（相对于移动平均）
        volume_ma = data[self.volume_col].rolling(window=self.window).mean()
        result[f"{self.name}_norm"] = (data[self.volume_col] - volume_ma) / volume_ma
        
        # 4. 成交量的标准差
        result[f"{self.name}_std"] = data[self.volume_col].rolling(window=self.window).std()
        
        # 5. 价格上涨/下跌时的成交量对比
        price_change = data[self.price_col].diff()
        up_volume = data[self.volume_col].where(price_change > 0, 0)
        down_volume = data[self.volume_col].where(price_change < 0, 0)
        
        # 计算涨跌成交量的移动平均
        avg_up_volume = up_volume.rolling(window=self.window).mean()
        avg_down_volume = down_volume.rolling(window=self.window).mean()
        
        # 成交量比率（VROC）
        result[f"{self.name}_ratio"] = avg_up_volume / (avg_down_volume + 1e-10)  # 避免除以零
        
        # 6. 成交量与价格的相关系数
        result[f"{self.name}_corr"] = data[[self.price_col, self.volume_col]]
            .rolling(window=self.window)
            .corr()
            .unstack()
            .loc[:, (self.price_col, self.volume_col)]
        
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


class Momentum(AlphaFactor):
    """
    动量因子
    
    衡量价格变化的速度和方向
    """
    
    def __init__(self, 
                 window: int = 12,
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化动量因子
        
        Args:
            window: 动量计算周期
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"Momentum_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.price_col = price_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量因子
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含动量因子的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 1. 简单动量（当前价格减去window期前的价格）
        result[f"{self.name}_raw"] = data[self.price_col] - data[self.price_col].shift(self.window)
        
        # 2. 动量百分比（相对于window期前价格的变化率）
        result[self.name] = (data[self.price_col] - data[self.price_col].shift(self.window)) / \
                          data[self.price_col].shift(self.window)
        
        # 3. 标准化动量（使用Z-score）
        mom_mean = result[self.name].rolling(window=self.window).mean()
        mom_std = result[self.name].rolling(window=self.window).std()
        result[f"{self.name}_norm"] = (result[self.name] - mom_mean) / (mom_std + 1e-10)  # 避免除以零
        
        # 4. 动量变化率
        result[f"{self.name}_roc"] = result[self.name].pct_change()
        
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
            "price_col": self.price_col
        })
        return config


class Volatility(AlphaFactor):
    """
    波动率因子
    
    计算价格的波动率特征
    """
    
    def __init__(self, 
                 window: int = 20,
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化波动率因子
        
        Args:
            window: 波动率计算周期
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"Volatility_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.price_col = price_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率因子
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含波动率因子的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 计算收益率
        returns = data[self.price_col].pct_change()
        
        # 1. 历史波动率（日收益率标准差）
        result[f"{self.name}_std"] = returns.rolling(window=self.window).std()
        
        # 2. 年化波动率
        annualized_vol = returns.rolling(window=self.window).std() * np.sqrt(252)  # 假设252个交易日
        result[f"{self.name}_annualized"] = annualized_vol
        
        # 3. 相对波动率（相对于更长周期的波动率）
        long_window = self.window * 2
        long_vol = returns.rolling(window=long_window).std()
        result[f"{self.name}_relative"] = result[f"{self.name}_std"] / (long_vol + 1e-10)  # 避免除以零
        
        # 4. 波动率变化率
        result[f"{self.name}_roc"] = result[f"{self.name}_std"].pct_change()
        
        # 5. 收益率分布偏度（衡量分布的不对称性）
        result[f"{self.name}_skew"] = returns.rolling(window=self.window).skew()
        
        # 6. 收益率分布峰度（衡量分布的尾部厚重程度）
        result[f"{self.name}_kurtosis"] = returns.rolling(window=self.window).kurtosis()
        
        # 标准化的波动率值（Z-score）
        vol_mean = result[f"{self.name}_std"].rolling(window=self.window).mean()
        vol_std = result[f"{self.name}_std"].rolling(window=self.window).std()
        result[f"{self.name}_norm"] = (result[f"{self.name}_std"] - vol_mean) / (vol_std + 1e-10)
        
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
            "price_col": self.price_col
        })
        return config


class TrendStrength(AlphaFactor):
    """
    趋势强度因子
    
    衡量价格趋势的强度
    """
    
    def __init__(self, 
                 window: int = 20,
                 price_col: str = 'close',
                 **kwargs):
        """
        初始化趋势强度因子
        
        Args:
            window: 计算周期
            price_col: 使用的价格列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"TrendStrength_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.price_col = price_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势强度因子
        
        Args:
            data: 输入数据，必须包含指定的价格列
            
        Returns:
            包含趋势强度因子的DataFrame
        """
        # 验证数据
        if self.price_col not in data.columns:
            raise ValueError(f"数据中缺少 {self.price_col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col]].copy()
        
        # 1. 线性回归趋势强度
        def calculate_trend_strength(prices):
            # 生成x轴（时间）
            x = np.arange(len(prices))
            # 计算线性回归
            coefficients = np.polyfit(x, prices, 1)
            # 计算R²值作为趋势强度
            y_pred = np.polyval(coefficients, x)
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))  # 避免除以零
            # 趋势方向（系数的符号）
            direction = np.sign(coefficients[0])
            # 返回带方向的趋势强度
            return direction * r_squared
        
        # 应用滚动计算
        result[f"{self.name}_rsquared"] = data[self.price_col].rolling(window=self.window)
            .apply(calculate_trend_strength)
        
        # 2. ADX（平均趋向指标）简化版
        # 计算+DM和-DM
        delta_high = data['high'].diff()
        delta_low = -data['low'].diff()
        
        plus_dm = delta_high.where((delta_high > delta_low) & (delta_high > 0), 0)
        minus_dm = delta_low.where((delta_low > delta_high) & (delta_low > 0), 0)
        
        # 计算真实波动幅度(TR)
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # 计算+DI和-DI
        atr = tr.rolling(window=self.window).mean()
        plus_di = 100 * plus_dm.rolling(window=self.window).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=self.window).mean() / atr
        
        # 计算DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # 计算ADX（DX的移动平均）
        result[f"{self.name}_adx"] = dx.rolling(window=self.window).mean()
        
        # 3. 标准化的趋势强度
        result[f"{self.name}_norm"] = result[f"{self.name}_rsquared"]
        
        # 4. 趋势方向信号
        result[f"{self.name}_signal"] = np.sign(result[f"{self.name}_rsquared"])
        
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
            "price_col": self.price_col
        })
        return config


class VolumeWeightedAveragePrice(AlphaFactor):
    """
    成交量加权平均价格因子
    
    考虑成交量权重的价格平均值，更能反映市场真实价格
    """
    
    def __init__(self, 
                 window: int = 20,
                 price_col: str = 'close',
                 volume_col: str = 'volume',
                 **kwargs):
        """
        初始化VWAP因子
        
        Args:
            window: 计算周期
            price_col: 使用的价格列名
            volume_col: 使用的成交量列名
            **kwargs: 传递给父类的参数
        """
        # 设置因子名称
        name = f"VWAP_{window}"
        
        # 调用父类初始化
        super().__init__(name=name, **kwargs)
        
        # 因子参数
        self.window = window
        self.price_col = price_col
        self.volume_col = volume_col
        
        # 验证参数
        if self.window <= 0:
            raise ValueError("window 必须大于0")
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算VWAP因子
        
        Args:
            data: 输入数据，必须包含指定的价格和成交量列
            
        Returns:
            包含VWAP因子的DataFrame
        """
        # 验证数据
        for col in [self.price_col, self.volume_col]:
            if col not in data.columns:
                raise ValueError(f"数据中缺少 {col} 列")
        
        # 创建结果DataFrame
        result = data[[self.price_col, self.volume_col]].copy()
        
        # 计算价格*成交量的累计和
        price_volume = data[self.price_col] * data[self.volume_col]
        cum_price_volume = price_volume.rolling(window=self.window).sum()
        
        # 计算成交量的累计和
        cum_volume = data[self.volume_col].rolling(window=self.window).sum()
        
        # 计算VWAP
        result[self.name] = cum_price_volume / cum_volume
        
        # 计算价格与VWAP的偏离度
        result[f"{self.name}_deviation"] = (data[self.price_col] - result[self.name]) / result[self.name]
        
        # 标准化的偏离度
        dev_mean = result[f"{self.name}_deviation"].rolling(window=self.window).mean()
        dev_std = result[f"{self.name}_deviation"].rolling(window=self.window).std()
        result[f"{self.name}_norm"] = (result[f"{self.name}_deviation"] - dev_mean) / (dev_std + 1e-10)
        
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
            "price_col": self.price_col,
            "volume_col": self.volume_col
        })
        return config


# 创建常用因子的便捷函数
def create_ma_factors(window_sizes: List[int] = [5, 10, 20, 60, 120, 250]) -> List[MovingAverage]:
    """
    创建一组常用移动平均线因子
    
    Args:
        window_sizes: 移动窗口大小列表
        
    Returns:
        移动平均线因子列表
    """
    factors = []
    for window in window_sizes:
        # 添加SMA
        factors.append(MovingAverage(window=window, ma_type='sma'))
        # 添加EMA
        factors.append(MovingAverage(window=window, ma_type='ema'))
    return factors


def create_rsi_factors(window_sizes: List[int] = [6, 14, 21]) -> List[RSI]:
    """
    创建一组常用RSI因子
    
    Args:
        window_sizes: 计算周期列表
        
    Returns:
        RSI因子列表
    """
    return [RSI(window=window) for window in window_sizes]


def create_bollinger_bands_factors(window_sizes: List[int] = [20], 
                                 num_std_list: List[float] = [1.0, 2.0, 3.0]) -> List[BollingerBands]:
    """
    创建一组布林带因子
    
    Args:
        window_sizes: 移动窗口大小列表
        num_std_list: 标准差倍数列表
        
    Returns:
        布林带因子列表
    """
    factors = []
    for window in window_sizes:
        for num_std in num_std_list:
            factors.append(BollingerBands(window=window, num_std=num_std))
    return factors


def create_macd_factors(params_list: List[Tuple[int, int, int]] = [(12, 26, 9), (5, 35, 5)]) -> List[MACD]:
    """
    创建一组MACD因子
    
    Args:
        params_list: 参数元组列表，每个元组包含(fast_period, slow_period, signal_period)
        
    Returns:
        MACD因子列表
    """
    return [MACD(*params) for params in params_list]


def create_volume_factors(window_sizes: List[int] = [10, 20, 60]) -> List[VolumeFactors]:
    """
    创建一组成交量因子
    
    Args:
        window_sizes: 计算周期列表
        
    Returns:
        成交量因子列表
    """
    return [VolumeFactors(window=window) for window in window_sizes]


def create_momentum_factors(window_sizes: List[int] = [3, 6, 12, 24]) -> List[Momentum]:
    """
    创建一组动量因子
    
    Args:
        window_sizes: 计算周期列表
        
    Returns:
        动量因子列表
    """
    return [Momentum(window=window) for window in window_sizes]


def create_volatility_factors(window_sizes: List[int] = [10, 20, 30, 60]) -> List[Volatility]:
    """
    创建一组波动率因子
    
    Args:
        window_sizes: 计算周期列表
        
    Returns:
        波动率因子列表
    """
    return [Volatility(window=window) for window in window_sizes]


def create_basic_factors_set() -> List[AlphaFactor]:
    """
    创建一组基础因子集合
    
    Returns:
        基础因子列表
    """
    factors = []
    
    # 添加移动平均线因子
    factors.extend(create_ma_factors([5, 10, 20, 60]))
    
    # 添加RSI因子
    factors.extend(create_rsi_factors([6, 14, 21]))
    
    # 添加布林带因子
    factors.extend(create_bollinger_bands_factors([20], [1.0, 2.0]))
    
    # 添加MACD因子
    factors.extend(create_macd_factors([(12, 26, 9)]))
    
    # 添加成交量因子
    factors.extend(create_volume_factors([20]))
    
    # 添加动量因子
    factors.extend(create_momentum_factors([3, 12]))
    
    # 添加波动率因子
    factors.extend(create_volatility_factors([20]))
    
    # 添加趋势强度和VWAP因子
    factors.append(TrendStrength(window=20))
    factors.append(VolumeWeightedAveragePrice(window=20))
    
    return factors
