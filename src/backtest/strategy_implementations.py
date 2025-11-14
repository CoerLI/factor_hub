import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timedelta

from src.backtest.strategy_base import (
    StrategyBase, SingleAssetStrategy, MultiAssetStrategy, FactorBasedStrategy,
    Order, Position, Portfolio
)
from src.factors.factor_base import FactorContainer
from src.utils.helpers import setup_logger, DEFAULT_LOGGER, timeit
from src.factors.basic_factors import (
    MovingAverage, RSI, BollingerBands, MACD, VolumeFactors,
    Momentum, Volatility, TrendStrength, VolumeWeightedAveragePrice
)

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class MovingAverageCrossStrategy(SingleAssetStrategy):
    """
    移动平均线交叉策略
    
    当短期移动平均线上穿长期移动平均线时买入，下穿时卖出
    """
    
    def __init__(self, 
                 symbol: str,
                 short_window: int = 5,
                 long_window: int = 20,
                 lookback: int = 100,
                 name: str = 'MovingAverageCross'):
        """
        初始化移动平均线交叉策略
        
        Args:
            symbol: 交易标的
            short_window: 短期移动平均线窗口
            long_window: 长期移动平均线窗口
            lookback: 回测需要的历史数据长度
            name: 策略名称
        """
        super().__init__(symbols=[symbol], lookback=lookback, name=name)
        
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        
        # 状态变量
        self.position = 0  # 当前持仓状态: -1(做空), 0(空仓), 1(做多)
        self.last_signals = None
        self.signal_history = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文对象
        """
        super().initialize(context)
        logger.info(f"初始化{self.name}策略: 标的={self.symbol}, 短期窗口={self.short_window}, 长期窗口={self.long_window}")
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成订单
        
        Args:
            data: 数据字典 {symbol: dataframe}
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取数据
        df = data.get(self.symbol)
        if df is None or len(df) < self.long_window:
            return orders
        
        # 计算信号
        signals = self._calculate_signals(df)
        
        # 生成订单
        if signals is not None and self.last_signals is not None:
            orders = self._generate_orders(signals, df, timestamp)
        
        self.last_signals = signals
        
        # 记录信号
        if signals is not None:
            self.signal_history.append({
                'timestamp': timestamp,
                'signal': signals[self.symbol],
                'short_ma': signals['short_ma'],
                'long_ma': signals['long_ma']
            })
        
        return orders
    
    def _calculate_signals(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        计算交易信号
        
        Args:
            df: 价格数据
            
        Returns:
            信号字典
        """
        # 复制数据以避免修改原始数据
        df = df.copy()
        
        try:
            # 计算移动平均线
            df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
            df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
            
            # 计算信号
            df['signal'] = 0.0
            df['signal'][self.long_window:] = np.where(
                df['short_ma'][self.long_window:] > df['long_ma'][self.long_window:], 
                1.0, 0.0
            )
            
            # 计算持仓变化
            df['position'] = df['signal'].diff()
            
            # 获取最新信号
            latest = df.iloc[-1]
            
            # 当前信号
            current_signal = 1.0 if latest['short_ma'] > latest['long_ma'] else 0.0
            
            return {
                self.symbol: current_signal,
                'short_ma': latest['short_ma'],
                'long_ma': latest['long_ma'],
                'position_change': latest['position']
            }
            
        except Exception as e:
            logger.error(f"计算信号出错: {str(e)}")
            return None
    
    def _generate_orders(self, signals: Dict, df: pd.DataFrame, timestamp: pd.Timestamp) -> List[Order]:
        """
        根据信号生成订单
        
        Args:
            signals: 信号字典
            df: 价格数据
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取最新价格
        latest_price = df['close'].iloc[-1]
        
        # 获取投资组合信息
        portfolio = self.context.get('portfolio')
        if not portfolio:
            return orders
        
        # 计算头寸大小 (使用总资产的固定比例)
        equity = portfolio.get_total_equity()
        position_size = int(equity * 0.1 / latest_price)  # 10% 的资金
        
        # 避免零头寸
        if position_size <= 0:
            position_size = 1
        
        # 获取当前持仓
        current_position = portfolio.get_position(self.symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # 根据信号变化生成订单
        if signals.get('position_change') == 1.0 and current_quantity <= 0:
            # 金叉：买入
            quantity = position_size - current_quantity  # 如果有空仓，先平仓再买入
            order = Order(
                symbol=self.symbol,
                action=Order.BUY if current_quantity >= 0 else Order.COVER,
                quantity=quantity,
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 1
            logger.info(f"[{timestamp}] 金叉信号: 买入 {self.symbol}, 数量: {quantity}, 价格: {latest_price}")
            
        elif signals.get('position_change') == -1.0 and current_quantity >= 0:
            # 死叉：卖出
            order = Order(
                symbol=self.symbol,
                action=Order.SELL,
                quantity=abs(current_quantity),
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 0
            logger.info(f"[{timestamp}] 死叉信号: 卖出 {self.symbol}, 数量: {abs(current_quantity)}, 价格: {latest_price}")
        
        return orders
    
    def on_order_filled(self, order: Order, timestamp: pd.Timestamp) -> None:
        """
        订单成交回调
        
        Args:
            order: 订单
            timestamp: 时间戳
        """
        logger.info(f"[{timestamp}] 订单成交: {order.action} {order.symbol} {order.quantity} @ {order.fill_price}")
    
    def on_end_of_day(self, timestamp: pd.Timestamp) -> None:
        """
        日终处理
        
        Args:
            timestamp: 时间戳
        """
        # 可以在这里添加日终清理、记录等操作
        pass
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略绩效摘要
        
        Returns:
            绩效摘要字典
        """
        return {
            'strategy': self.name,
            'symbol': self.symbol,
            'short_window': self.short_window,
            'long_window': self.long_window,
            'total_signals': len(self.signal_history)
        }


class RSIStrategy(SingleAssetStrategy):
    """
    RSI策略
    
    使用相对强弱指数进行超买超卖判断
    """
    
    def __init__(self, 
                 symbol: str,
                 period: int = 14,
                 overbought: float = 70.0,
                 oversold: float = 30.0,
                 lookback: int = 100,
                 name: str = 'RSIStrategy'):
        """
        初始化RSI策略
        
        Args:
            symbol: 交易标的
            period: RSI计算周期
            overbought: 超买阈值
            oversold: 超卖阈值
            lookback: 回测需要的历史数据长度
            name: 策略名称
        """
        super().__init__(symbols=[symbol], lookback=lookback, name=name)
        
        self.symbol = symbol
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        # 状态变量
        self.position = 0  # 当前持仓状态: -1(做空), 0(空仓), 1(做多)
        self.last_rsi = None
        self.signal_history = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文对象
        """
        super().initialize(context)
        logger.info(f"初始化{self.name}策略: 标的={self.symbol}, 周期={self.period}, 超买={self.overbought}, 超卖={self.oversold}")
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成订单
        
        Args:
            data: 数据字典 {symbol: dataframe}
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取数据
        df = data.get(self.symbol)
        if df is None or len(df) < self.period + 1:
            return orders
        
        # 计算RSI
        rsi = self._calculate_rsi(df)
        
        # 生成订单
        if rsi is not None:
            orders = self._generate_orders(rsi, df, timestamp)
        
        self.last_rsi = rsi
        
        # 记录信号
        if rsi is not None:
            self.signal_history.append({
                'timestamp': timestamp,
                'rsi': rsi
            })
        
        return orders
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Optional[float]:
        """
        计算RSI
        
        Args:
            df: 价格数据
            
        Returns:
            RSI值
        """
        try:
            # 计算价格变化
            delta = df['close'].diff()
            
            # 分离涨跌
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            
            # 计算RS和RSI
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
            
        except Exception as e:
            logger.error(f"计算RSI出错: {str(e)}")
            return None
    
    def _generate_orders(self, rsi: float, df: pd.DataFrame, timestamp: pd.Timestamp) -> List[Order]:
        """
        根据RSI生成订单
        
        Args:
            rsi: RSI值
            df: 价格数据
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取最新价格
        latest_price = df['close'].iloc[-1]
        
        # 获取投资组合信息
        portfolio = self.context.get('portfolio')
        if not portfolio:
            return orders
        
        # 计算头寸大小 (使用总资产的固定比例)
        equity = portfolio.get_total_equity()
        position_size = int(equity * 0.1 / latest_price)  # 10% 的资金
        
        # 避免零头寸
        if position_size <= 0:
            position_size = 1
        
        # 获取当前持仓
        current_position = portfolio.get_position(self.symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # 根据RSI生成信号
        # 超卖买入
        if rsi <= self.oversold and self.position <= 0:
            quantity = position_size - current_quantity
            order = Order(
                symbol=self.symbol,
                action=Order.BUY if current_quantity >= 0 else Order.COVER,
                quantity=quantity,
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 1
            logger.info(f"[{timestamp}] RSI超卖: 买入 {self.symbol}, 数量: {quantity}, 价格: {latest_price}, RSI: {rsi:.2f}")
        
        # 超买卖出
        elif rsi >= self.overbought and self.position >= 0:
            order = Order(
                symbol=self.symbol,
                action=Order.SELL,
                quantity=abs(current_quantity),
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 0
            logger.info(f"[{timestamp}] RSI超买: 卖出 {self.symbol}, 数量: {abs(current_quantity)}, 价格: {latest_price}, RSI: {rsi:.2f}")
        
        # RSI回到中性区域，平仓
        elif (self.last_rsi is not None and 
              ((self.last_rsi < self.oversold and rsi > self.oversold and self.position < 0) or
               (self.last_rsi > self.overbought and rsi < self.overbought and self.position > 0))):
            if current_quantity != 0:
                order = Order(
                    symbol=self.symbol,
                    action=Order.COVER if current_quantity < 0 else Order.SELL,
                    quantity=abs(current_quantity),
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
                self.position = 0
                logger.info(f"[{timestamp}] RSI回到中性: 平仓 {self.symbol}, 数量: {abs(current_quantity)}, RSI: {rsi:.2f}")
        
        return orders
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略绩效摘要
        
        Returns:
            绩效摘要字典
        """
        return {
            'strategy': self.name,
            'symbol': self.symbol,
            'period': self.period,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'total_signals': len(self.signal_history)
        }


class BollingerBandsStrategy(SingleAssetStrategy):
    """
    布林带策略
    
    在价格触及下轨时买入，触及上轨时卖出
    """
    
    def __init__(self, 
                 symbol: str,
                 window: int = 20,
                 num_std: float = 2.0,
                 lookback: int = 100,
                 name: str = 'BollingerBandsStrategy'):
        """
        初始化布林带策略
        
        Args:
            symbol: 交易标的
            window: 移动平均窗口
            num_std: 标准差倍数
            lookback: 回测需要的历史数据长度
            name: 策略名称
        """
        super().__init__(symbols=[symbol], lookback=lookback, name=name)
        
        self.symbol = symbol
        self.window = window
        self.num_std = num_std
        
        # 状态变量
        self.position = 0  # 当前持仓状态: -1(做空), 0(空仓), 1(做多)
        self.bbands_history = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文对象
        """
        super().initialize(context)
        logger.info(f"初始化{self.name}策略: 标的={self.symbol}, 窗口={self.window}, 标准差倍数={self.num_std}")
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成订单
        
        Args:
            data: 数据字典 {symbol: dataframe}
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取数据
        df = data.get(self.symbol)
        if df is None or len(df) < self.window:
            return orders
        
        # 计算布林带
        bbands = self._calculate_bollinger_bands(df)
        
        # 生成订单
        if bbands is not None:
            orders = self._generate_orders(bbands, df, timestamp)
        
        # 记录数据
        if bbands is not None:
            self.bbands_history.append({
                'timestamp': timestamp,
                **bbands
            })
        
        return orders
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        计算布林带
        
        Args:
            df: 价格数据
            
        Returns:
            布林带数据字典
        """
        try:
            # 计算移动平均和标准差
            middle_band = df['close'].rolling(window=self.window).mean()
            std = df['close'].rolling(window=self.window).std()
            
            # 计算上轨和下轨
            upper_band = middle_band + (std * self.num_std)
            lower_band = middle_band - (std * self.num_std)
            
            # 计算带宽百分比
            bandwidth = (upper_band - lower_band) / middle_band
            
            # 获取最新值
            latest_idx = -1
            
            return {
                'middle_band': middle_band.iloc[latest_idx],
                'upper_band': upper_band.iloc[latest_idx],
                'lower_band': lower_band.iloc[latest_idx],
                'bandwidth': bandwidth.iloc[latest_idx]
            }
            
        except Exception as e:
            logger.error(f"计算布林带出错: {str(e)}")
            return None
    
    def _generate_orders(self, bbands: Dict[str, float], df: pd.DataFrame, timestamp: pd.Timestamp) -> List[Order]:
        """
        根据布林带生成订单
        
        Args:
            bbands: 布林带数据
            df: 价格数据
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取最新价格
        latest_price = df['close'].iloc[-1]
        
        # 获取投资组合信息
        portfolio = self.context.get('portfolio')
        if not portfolio:
            return orders
        
        # 计算头寸大小 (使用总资产的固定比例)
        equity = portfolio.get_total_equity()
        position_size = int(equity * 0.1 / latest_price)  # 10% 的资金
        
        # 避免零头寸
        if position_size <= 0:
            position_size = 1
        
        # 获取当前持仓
        current_position = portfolio.get_position(self.symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # 价格触及下轨买入
        if latest_price <= bbands['lower_band'] and self.position <= 0:
            quantity = position_size - current_quantity
            order = Order(
                symbol=self.symbol,
                action=Order.BUY if current_quantity >= 0 else Order.COVER,
                quantity=quantity,
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 1
            logger.info(f"[{timestamp}] 触及下轨: 买入 {self.symbol}, 数量: {quantity}, 价格: {latest_price}")
        
        # 价格触及上轨卖出
        elif latest_price >= bbands['upper_band'] and self.position >= 0:
            order = Order(
                symbol=self.symbol,
                action=Order.SELL,
                quantity=abs(current_quantity),
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 0
            logger.info(f"[{timestamp}] 触及上轨: 卖出 {self.symbol}, 数量: {abs(current_quantity)}, 价格: {latest_price}")
        
        # 价格回归中轨，平仓
        elif ((self.position > 0 and latest_price <= bbands['middle_band']) or
              (self.position < 0 and latest_price >= bbands['middle_band'])):
            if current_quantity != 0:
                order = Order(
                    symbol=self.symbol,
                    action=Order.COVER if current_quantity < 0 else Order.SELL,
                    quantity=abs(current_quantity),
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
                self.position = 0
                logger.info(f"[{timestamp}] 回归中轨: 平仓 {self.symbol}, 数量: {abs(current_quantity)}")
        
        return orders
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略绩效摘要
        
        Returns:
            绩效摘要字典
        """
        return {
            'strategy': self.name,
            'symbol': self.symbol,
            'window': self.window,
            'num_std': self.num_std,
            'total_signals': len(self.bbands_history)
        }


class MACDStrategy(SingleAssetStrategy):
    """
    MACD策略
    
    使用MACD指标进行趋势跟踪
    """
    
    def __init__(self, 
                 symbol: str,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 lookback: int = 100,
                 name: str = 'MACDStrategy'):
        """
        初始化MACD策略
        
        Args:
            symbol: 交易标的
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            lookback: 回测需要的历史数据长度
            name: 策略名称
        """
        super().__init__(symbols=[symbol], lookback=lookback, name=name)
        
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # 状态变量
        self.position = 0  # 当前持仓状态: -1(做空), 0(空仓), 1(做多)
        self.last_macd = None
        self.macd_history = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文对象
        """
        super().initialize(context)
        logger.info(f"初始化{self.name}策略: 标的={self.symbol}, 快线={self.fast_period}, 慢线={self.slow_period}, 信号={self.signal_period}")
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成订单
        
        Args:
            data: 数据字典 {symbol: dataframe}
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取数据
        df = data.get(self.symbol)
        if df is None or len(df) < self.slow_period + self.signal_period:
            return orders
        
        # 计算MACD
        macd_data = self._calculate_macd(df)
        
        # 生成订单
        if macd_data is not None:
            orders = self._generate_orders(macd_data, df, timestamp)
        
        self.last_macd = macd_data
        
        # 记录数据
        if macd_data is not None:
            self.macd_history.append({
                'timestamp': timestamp,
                **macd_data
            })
        
        return orders
    
    def _calculate_macd(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        计算MACD
        
        Args:
            df: 价格数据
            
        Returns:
            MACD数据字典
        """
        try:
            # 计算指数移动平均线
            ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()
            
            # 计算MACD线
            macd_line = ema_fast - ema_slow
            
            # 计算信号线
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            
            # 计算柱状图
            histogram = macd_line - signal_line
            
            # 获取最新值
            latest_idx = -1
            
            return {
                'macd_line': macd_line.iloc[latest_idx],
                'signal_line': signal_line.iloc[latest_idx],
                'histogram': histogram.iloc[latest_idx]
            }
            
        except Exception as e:
            logger.error(f"计算MACD出错: {str(e)}")
            return None
    
    def _generate_orders(self, macd_data: Dict[str, float], df: pd.DataFrame, timestamp: pd.Timestamp) -> List[Order]:
        """
        根据MACD生成订单
        
        Args:
            macd_data: MACD数据
            df: 价格数据
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取最新价格
        latest_price = df['close'].iloc[-1]
        
        # 获取投资组合信息
        portfolio = self.context.get('portfolio')
        if not portfolio:
            return orders
        
        # 计算头寸大小 (使用总资产的固定比例)
        equity = portfolio.get_total_equity()
        position_size = int(equity * 0.1 / latest_price)  # 10% 的资金
        
        # 避免零头寸
        if position_size <= 0:
            position_size = 1
        
        # 获取当前持仓
        current_position = portfolio.get_position(self.symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # 金叉：MACD线上穿信号线
        if self.last_macd is not None and \
           macd_data['histogram'] > 0 and self.last_macd['histogram'] <= 0 and \
           self.position <= 0:
            quantity = position_size - current_quantity
            order = Order(
                symbol=self.symbol,
                action=Order.BUY if current_quantity >= 0 else Order.COVER,
                quantity=quantity,
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 1
            logger.info(f"[{timestamp}] MACD金叉: 买入 {self.symbol}, 数量: {quantity}, 价格: {latest_price}")
        
        # 死叉：MACD线下穿信号线
        elif self.last_macd is not None and \
             macd_data['histogram'] < 0 and self.last_macd['histogram'] >= 0 and \
             self.position >= 0:
            order = Order(
                symbol=self.symbol,
                action=Order.SELL,
                quantity=abs(current_quantity),
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 0
            logger.info(f"[{timestamp}] MACD死叉: 卖出 {self.symbol}, 数量: {abs(current_quantity)}, 价格: {latest_price}")
        
        return orders
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略绩效摘要
        
        Returns:
            绩效摘要字典
        """
        return {
            'strategy': self.name,
            'symbol': self.symbol,
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'total_signals': len(self.macd_history)
        }


class PortfolioRebalancingStrategy(MultiAssetStrategy):
    """
    投资组合再平衡策略
    
    根据预设权重定期调整投资组合
    """
    
    def __init__(self, 
                 symbols: List[str],
                 target_weights: Optional[Dict[str, float]] = None,
                 rebalance_frequency: str = 'daily',  # 'daily', 'weekly', 'monthly'
                 lookback: int = 30,
                 name: str = 'PortfolioRebalancing'):
        """
        初始化投资组合再平衡策略
        
        Args:
            symbols: 交易标的列表
            target_weights: 目标权重字典 {symbol: weight}
            rebalance_frequency: 再平衡频率
            lookback: 回测需要的历史数据长度
            name: 策略名称
        """
        super().__init__(symbols=symbols, lookback=lookback, name=name)
        
        # 设置默认权重（等权重）
        if target_weights is None:
            n_symbols = len(symbols)
            self.target_weights = {symbol: 1.0 / n_symbols for symbol in symbols}
        else:
            self.target_weights = target_weights
            
        # 归一化权重
        total_weight = sum(self.target_weights.values())
        if total_weight > 0:
            self.target_weights = {k: v / total_weight for k, v in self.target_weights.items()}
        
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance_date = None
        
        # 记录再平衡历史
        self.rebalance_history = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文对象
        """
        super().initialize(context)
        logger.info(f"初始化{self.name}策略: 标的数={len(self.symbols)}, 再平衡频率={self.rebalance_frequency}")
        logger.info(f"目标权重: {self.target_weights}")
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成订单
        
        Args:
            data: 数据字典 {symbol: dataframe}
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 检查是否需要再平衡
        if not self._should_rebalance(timestamp):
            return orders
        
        # 获取投资组合信息
        portfolio = self.context.get('portfolio')
        if not portfolio:
            return orders
        
        # 计算再平衡订单
        orders = self._calculate_rebalance_orders(data, portfolio, timestamp)
        
        # 记录再平衡
        if orders:
            self.last_rebalance_date = timestamp.date()
            self.rebalance_history.append({
                'timestamp': timestamp,
                'orders_count': len(orders)
            })
            logger.info(f"[{timestamp}] 执行投资组合再平衡: {len(orders)} 个订单")
        
        return orders
    
    def _should_rebalance(self, timestamp: pd.Timestamp) -> bool:
        """
        检查是否需要再平衡
        
        Args:
            timestamp: 时间戳
            
        Returns:
            是否需要再平衡
        """
        # 如果是第一次运行，执行再平衡
        if self.last_rebalance_date is None:
            return True
        
        current_date = timestamp.date()
        
        # 根据频率判断
        if self.rebalance_frequency == 'daily':
            return current_date > self.last_rebalance_date
        
        elif self.rebalance_frequency == 'weekly':
            # 每个周一再平衡
            return (current_date.weekday() == 0 and 
                    current_date > self.last_rebalance_date)
        
        elif self.rebalance_frequency == 'monthly':
            # 每月第一个交易日再平衡
            return (current_date.month != self.last_rebalance_date.month and
                    current_date.year != self.last_rebalance_date.year)
        
        return False
    
    def _calculate_rebalance_orders(self, 
                                   data: Dict[str, pd.DataFrame],
                                   portfolio: Portfolio,
                                   timestamp: pd.Timestamp) -> List[Order]:
        """
        计算再平衡订单
        
        Args:
            data: 数据字典
            portfolio: 投资组合
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取总权益
        total_equity = portfolio.get_total_equity()
        
        # 计算当前价格
        current_prices = {}
        for symbol, df in data.items():
            if not df.empty:
                current_prices[symbol] = df['close'].iloc[-1]
        
        # 计算目标持仓价值和当前持仓价值
        target_values = {}
        current_values = {}
        current_positions = {}
        
        for symbol, weight in self.target_weights.items():
            # 目标持仓价值
            target_values[symbol] = total_equity * weight
            
            # 当前持仓
            position = portfolio.get_position(symbol)
            if position:
                current_positions[symbol] = position.quantity
                if symbol in current_prices:
                    current_values[symbol] = position.quantity * current_prices[symbol]
            else:
                current_positions[symbol] = 0
                current_values[symbol] = 0.0
        
        # 计算需要调整的数量
        for symbol, target_value in target_values.items():
            if symbol not in current_prices:
                continue
            
            current_value = current_values[symbol]
            price = current_prices[symbol]
            
            # 计算目标数量和当前数量
            target_quantity = int(target_value / price)
            current_quantity = current_positions[symbol]
            
            # 计算需要调整的数量
            quantity_diff = target_quantity - current_quantity
            
            # 避免微小调整
            if abs(quantity_diff) <= 0:
                continue
            
            # 创建订单
            if quantity_diff > 0:
                # 买入
                order = Order(
                    symbol=symbol,
                    action=Order.BUY,
                    quantity=quantity_diff,
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
                logger.debug(f"[{timestamp}] 买入 {symbol}: {quantity_diff} 股, 目标权重: {self.target_weights[symbol]:.2%}")
            else:
                # 卖出
                order = Order(
                    symbol=symbol,
                    action=Order.SELL,
                    quantity=abs(quantity_diff),
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
                logger.debug(f"[{timestamp}] 卖出 {symbol}: {abs(quantity_diff)} 股, 目标权重: {self.target_weights[symbol]:.2%}")
        
        return orders
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略绩效摘要
        
        Returns:
            绩效摘要字典
        """
        return {
            'strategy': self.name,
            'symbols_count': len(self.symbols),
            'rebalance_frequency': self.rebalance_frequency,
            'total_rebalances': len(self.rebalance_history),
            'target_weights': self.target_weights
        }


class FactorRotationStrategy(FactorBasedStrategy):
    """
    因子轮动策略
    
    根据多个因子的表现轮动配置投资组合
    """
    
    def __init__(self, 
                 symbols: List[str],
                 factor_container: FactorContainer,
                 lookback: int = 60,
                 top_n: int = 5,
                 rebalance_frequency: str = 'weekly',
                 name: str = 'FactorRotation'):
        """
        初始化因子轮动策略
        
        Args:
            symbols: 交易标的列表
            factor_container: 因子容器
            lookback: 回测需要的历史数据长度
            top_n: 选择排名前N的标的
            rebalance_frequency: 再平衡频率
            name: 策略名称
        """
        super().__init__(symbols=symbols, factor_container=factor_container, lookback=lookback, name=name)
        
        self.top_n = top_n
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance_date = None
        
        # 记录轮动历史
        self.rotation_history = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文对象
        """
        super().initialize(context)
        logger.info(f"初始化{self.name}策略: 标的数={len(self.symbols)}, 因子数={len(self.factor_container.factors)}, 再平衡频率={self.rebalance_frequency}")
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成订单
        
        Args:
            data: 数据字典 {symbol: dataframe}
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 先更新因子值
        self._update_factors(data, timestamp)
        
        # 检查是否需要再平衡
        if not self._should_rebalance(timestamp):
            return orders
        
        # 生成信号
        signals = self._generate_signals(timestamp)
        
        # 根据信号生成订单
        if signals:
            orders = self.signals_to_orders(signals, data, self.context['portfolio'], timestamp)
            
            # 记录轮动
            self.last_rebalance_date = timestamp.date()
            self.rotation_history.append({
                'timestamp': timestamp,
                'selected_symbols': [s for s, sig in signals.items() if sig > 0],
                'signals': signals
            })
            logger.info(f"[{timestamp}] 执行因子轮动: 选择了 {len(orders) // 2} 个标的")
        
        return orders
    
    def _update_factors(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> None:
        """
        更新因子值
        
        Args:
            data: 数据字典
            timestamp: 时间戳
        """
        # 更新因子容器中的数据
        for symbol, df in data.items():
            self.factor_container.update_data(symbol, df)
        
        # 计算所有因子
        self.factor_container.calculate_factors(timestamp)
    
    def _should_rebalance(self, timestamp: pd.Timestamp) -> bool:
        """
        检查是否需要再平衡
        
        Args:
            timestamp: 时间戳
            
        Returns:
            是否需要再平衡
        """
        # 如果是第一次运行，执行再平衡
        if self.last_rebalance_date is None:
            return True
        
        current_date = timestamp.date()
        
        # 根据频率判断
        if self.rebalance_frequency == 'daily':
            return current_date > self.last_rebalance_date
        
        elif self.rebalance_frequency == 'weekly':
            # 每个周一再平衡
            return (current_date.weekday() == 0 and 
                    current_date > self.last_rebalance_date)
        
        elif self.rebalance_frequency == 'monthly':
            # 每月第一个交易日再平衡
            return (current_date.month != self.last_rebalance_date.month and
                    current_date.year != self.last_rebalance_date.year)
        
        return False
    
    def _generate_signals(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        生成交易信号
        
        Args:
            timestamp: 时间戳
            
        Returns:
            信号字典 {symbol: score}
        """
        # 获取所有因子值
        factor_values = self.factor_container.get_factor_values(timestamp)
        
        if not factor_values:
            return {}
        
        # 计算每个标的的综合得分
        scores = {}
        
        for symbol in self.symbols:
            if symbol not in factor_values:
                continue
            
            # 计算所有因子的平均分
            factor_scores = []
            for factor_name, values in factor_values[symbol].items():
                # 使用最新的因子值
                if values and len(values) > 0:
                    factor_scores.append(values[-1])
            
            if factor_scores:
                scores[symbol] = sum(factor_scores) / len(factor_scores)
        
        # 按得分排序，选择前N个
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, score in sorted_symbols[:self.top_n]]
        
        # 生成信号
        signals = {symbol: 1.0 if symbol in top_symbols else 0.0 for symbol in self.symbols}
        
        return signals
    
    def signals_to_orders(self, 
                         signals: Dict[str, float],
                         data: Dict[str, pd.DataFrame],
                         portfolio: Portfolio,
                         timestamp: pd.Timestamp) -> List[Order]:
        """
        将信号转换为订单
        
        Args:
            signals: 信号字典
            data: 数据字典
            portfolio: 投资组合
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取当前价格
        current_prices = {}
        for symbol, df in data.items():
            if not df.empty:
                current_prices[symbol] = df['close'].iloc[-1]
        
        # 获取总权益
        total_equity = portfolio.get_total_equity()
        
        # 计算选中的标的数量
        selected_count = sum(1 for s in signals.values() if s > 0)
        
        if selected_count == 0:
            # 如果没有选中的标的，清空所有持仓
            for position in portfolio.get_active_positions().values():
                order = Order(
                    symbol=position.symbol,
                    action=Order.COVER if position.quantity < 0 else Order.SELL,
                    quantity=abs(position.quantity),
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
            return orders
        
        # 每个选中标的的目标权重
        target_weight_per_symbol = 1.0 / selected_count
        
        # 处理每个标的
        for symbol, signal in signals.items():
            if symbol not in current_prices:
                continue
            
            price = current_prices[symbol]
            
            # 计算目标数量
            if signal > 0:
                # 买入信号
                target_value = total_equity * target_weight_per_symbol
                target_quantity = int(target_value / price)
            else:
                # 卖出信号
                target_quantity = 0
            
            # 获取当前持仓
            position = portfolio.get_position(symbol)
            current_quantity = position.quantity if position else 0
            
            # 计算需要调整的数量
            quantity_diff = target_quantity - current_quantity
            
            # 避免微小调整
            if abs(quantity_diff) <= 0:
                continue
            
            # 创建订单
            if quantity_diff > 0:
                # 买入
                order = Order(
                    symbol=symbol,
                    action=Order.BUY,
                    quantity=quantity_diff,
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
            else:
                # 卖出
                order = Order(
                    symbol=symbol,
                    action=Order.SELL,
                    quantity=abs(quantity_diff),
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
        
        return orders
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略绩效摘要
        
        Returns:
            绩效摘要字典
        """
        return {
            'strategy': self.name,
            'symbols_count': len(self.symbols),
            'factors_count': len(self.factor_container.factors),
            'top_n': self.top_n,
            'rebalance_frequency': self.rebalance_frequency,
            'total_rotations': len(self.rotation_history)
        }


class MeanReversionStrategy(SingleAssetStrategy):
    """
    均值回归策略
    
    当价格偏离移动平均线过多时，预期价格会回归均值
    """
    
    def __init__(self, 
                 symbol: str,
                 window: int = 20,
                 z_score_threshold: float = 1.5,
                 lookback: int = 100,
                 name: str = 'MeanReversion'):
        """
        初始化均值回归策略
        
        Args:
            symbol: 交易标的
            window: 移动平均窗口
            z_score_threshold: Z-score阈值
            lookback: 回测需要的历史数据长度
            name: 策略名称
        """
        super().__init__(symbols=[symbol], lookback=lookback, name=name)
        
        self.symbol = symbol
        self.window = window
        self.z_score_threshold = z_score_threshold
        
        # 状态变量
        self.position = 0  # 当前持仓状态: -1(做空), 0(空仓), 1(做多)
        self.last_z_score = None
        self.z_score_history = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文对象
        """
        super().initialize(context)
        logger.info(f"初始化{self.name}策略: 标的={self.symbol}, 窗口={self.window}, Z-score阈值={self.z_score_threshold}")
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成订单
        
        Args:
            data: 数据字典 {symbol: dataframe}
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取数据
        df = data.get(self.symbol)
        if df is None or len(df) < self.window:
            return orders
        
        # 计算Z-score
        z_score = self._calculate_z_score(df)
        
        # 生成订单
        if z_score is not None:
            orders = self._generate_orders(z_score, df, timestamp)
        
        self.last_z_score = z_score
        
        # 记录Z-score
        if z_score is not None:
            self.z_score_history.append({
                'timestamp': timestamp,
                'z_score': z_score
            })
        
        return orders
    
    def _calculate_z_score(self, df: pd.DataFrame) -> Optional[float]:
        """
        计算价格相对于移动平均线的Z-score
        
        Args:
            df: 价格数据
            
        Returns:
            Z-score值
        """
        try:
            # 计算移动平均和标准差
            mean = df['close'].rolling(window=self.window).mean()
            std = df['close'].rolling(window=self.window).std()
            
            # 计算Z-score
            z_score = (df['close'] - mean) / std
            
            return z_score.iloc[-1]
            
        except Exception as e:
            logger.error(f"计算Z-score出错: {str(e)}")
            return None
    
    def _generate_orders(self, z_score: float, df: pd.DataFrame, timestamp: pd.Timestamp) -> List[Order]:
        """
        根据Z-score生成订单
        
        Args:
            z_score: Z-score值
            df: 价格数据
            timestamp: 时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 获取最新价格
        latest_price = df['close'].iloc[-1]
        
        # 获取投资组合信息
        portfolio = self.context.get('portfolio')
        if not portfolio:
            return orders
        
        # 计算头寸大小 (使用总资产的固定比例)
        equity = portfolio.get_total_equity()
        position_size = int(equity * 0.1 / latest_price)  # 10% 的资金
        
        # 避免零头寸
        if position_size <= 0:
            position_size = 1
        
        # 获取当前持仓
        current_position = portfolio.get_position(self.symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # Z-score 过高，价格过高，卖出
        if z_score > self.z_score_threshold and self.position >= 0:
            order = Order(
                symbol=self.symbol,
                action=Order.SELL,
                quantity=abs(current_quantity),
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 0
            logger.info(f"[{timestamp}] Z-score过高: 卖出 {self.symbol}, 数量: {abs(current_quantity)}, 价格: {latest_price}, Z-score: {z_score:.2f}")
        
        # Z-score 过低，价格过低，买入
        elif z_score < -self.z_score_threshold and self.position <= 0:
            quantity = position_size - current_quantity
            order = Order(
                symbol=self.symbol,
                action=Order.BUY if current_quantity >= 0 else Order.COVER,
                quantity=quantity,
                order_type=Order.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
            self.position = 1
            logger.info(f"[{timestamp}] Z-score过低: 买入 {self.symbol}, 数量: {quantity}, 价格: {latest_price}, Z-score: {z_score:.2f}")
        
        # Z-score回归到接近零，平仓获利
        elif abs(z_score) < 0.5 and self.position != 0:
            if current_quantity != 0:
                order = Order(
                    symbol=self.symbol,
                    action=Order.COVER if current_quantity < 0 else Order.SELL,
                    quantity=abs(current_quantity),
                    order_type=Order.MARKET,
                    timestamp=timestamp
                )
                orders.append(order)
                self.position = 0
                logger.info(f"[{timestamp}] Z-score回归: 平仓 {self.symbol}, 数量: {abs(current_quantity)}, Z-score: {z_score:.2f}")
        
        return orders
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略绩效摘要
        
        Returns:
            绩效摘要字典
        """
        return {
            'strategy': self.name,
            'symbol': self.symbol,
            'window': self.window,
            'z_score_threshold': self.z_score_threshold,
            'total_signals': len(self.z_score_history)
        }


# 策略工厂函数
def create_strategy(strategy_type: str, **kwargs) -> StrategyBase:
    """
    创建策略实例的工厂函数
    
    Args:
        strategy_type: 策略类型
        **kwargs: 策略参数
        
    Returns:
        策略实例
    """
    strategy_map = {
        'moving_average_cross': MovingAverageCrossStrategy,
        'rsi': RSIStrategy,
        'bollinger_bands': BollingerBandsStrategy,
        'macd': MACDStrategy,
        'portfolio_rebalancing': PortfolioRebalancingStrategy,
        'factor_rotation': FactorRotationStrategy,
        'mean_reversion': MeanReversionStrategy
    }
    
    if strategy_type.lower() not in strategy_map:
        raise ValueError(f"不支持的策略类型: {strategy_type}")
    
    return strategy_map[strategy_type.lower()](**kwargs)


def create_default_strategy(symbol: str, strategy_type: str = 'moving_average_cross') -> StrategyBase:
    """
    创建默认配置的策略实例
    
    Args:
        symbol: 交易标的
        strategy_type: 策略类型
        
    Returns:
        策略实例
    """
    if strategy_type.lower() == 'moving_average_cross':
        return MovingAverageCrossStrategy(symbol)
    elif strategy_type.lower() == 'rsi':
        return RSIStrategy(symbol)
    elif strategy_type.lower() == 'bollinger_bands':
        return BollingerBandsStrategy(symbol)
    elif strategy_type.lower() == 'macd':
        return MACDStrategy(symbol)
    elif strategy_type.lower() == 'mean_reversion':
        return MeanReversionStrategy(symbol)
    else:
        raise ValueError(f"不支持的单资产策略类型: {strategy_type}")