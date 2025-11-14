import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import copy

from src.utils.helpers import setup_logger, DEFAULT_LOGGER
from src.factors.factor_base import FactorContainer
from src.data.data_loader import DataLoader
from src.utils.cache_manager import CacheManager

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class Order:
    """
    订单类
    
    表示交易订单，包含订单的所有必要信息
    """
    
    # 订单类型
    MARKET = 'market'  # 市价单
    LIMIT = 'limit'    # 限价单
    STOP = 'stop'      # 止损单
    STOP_LIMIT = 'stop_limit'  # 止损限价单
    
    # 订单动作
    BUY = 'buy'        # 买入
    SELL = 'sell'      # 卖出
    SHORT = 'short'    # 做空
    COVER = 'cover'    # 平空
    
    # 订单状态
    PENDING = 'pending'        # 待执行
    FILLED = 'filled'          # 已成交
    PARTIALLY_FILLED = 'partially_filled'  # 部分成交
    CANCELLED = 'cancelled'    # 已取消
    REJECTED = 'rejected'      # 已拒绝
    EXPIRED = 'expired'        # 已过期
    
    def __init__(self, 
                 symbol: str,
                 action: str,
                 quantity: float,
                 order_type: str = MARKET,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 tif: str = 'day',  # 有效期: day, gtc(Good Till Cancel), ioc(Immediate or Cancel)
                 order_id: Optional[str] = None,
                 timestamp: Optional[pd.Timestamp] = None,
                 strategy_id: Optional[str] = None,
                 reason: Optional[str] = None):
        """
        初始化订单
        
        Args:
            symbol: 交易标的
            action: 交易动作 (buy, sell, short, cover)
            quantity: 交易数量
            order_type: 订单类型 (market, limit, stop, stop_limit)
            price: 价格 (限价单和止损限价单需要)
            stop_price: 止损价格 (止损单和止损限价单需要)
            tif: 有效期
            order_id: 订单ID (如果不提供会自动生成)
            timestamp: 下单时间
            strategy_id: 策略ID
            reason: 下单原因
        """
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.tif = tif
        
        # 自动生成订单ID
        if order_id is None:
            import uuid
            self.order_id = str(uuid.uuid4())
        else:
            self.order_id = order_id
        
        self.timestamp = timestamp or pd.Timestamp.now()
        self.strategy_id = strategy_id
        self.reason = reason
        
        # 订单状态
        self.status = Order.PENDING
        self.filled_quantity = 0.0
        self.filled_price = None
        self.fill_timestamp = None
        
        # 手续费
        self.commission = 0.0
        
        # 订单历史
        self.order_history = [(self.timestamp, self.status)]
    
    def update_status(self, status: str, timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        更新订单状态
        
        Args:
            status: 新状态
            timestamp: 时间戳
        """
        self.status = status
        ts = timestamp or pd.Timestamp.now()
        self.order_history.append((ts, status))
        
        if status == Order.FILLED:
            self.fill_timestamp = ts
    
    def fill(self, 
             quantity: float,
             price: float,
             commission: float = 0.0,
             timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        部分成交订单
        
        Args:
            quantity: 成交数量
            price: 成交价格
            commission: 手续费
            timestamp: 成交时间
        """
        self.filled_quantity += quantity
        self.commission += commission
        self.filled_price = price  # 最后一次成交价格
        
        ts = timestamp or pd.Timestamp.now()
        
        # 更新状态
        if self.filled_quantity >= self.quantity:
            # 完全成交
            self.filled_quantity = self.quantity  # 确保数量精确
            self.update_status(Order.FILLED, ts)
        else:
            # 部分成交
            self.update_status(Order.PARTIALLY_FILLED, ts)
    
    def cancel(self, timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        取消订单
        
        Args:
            timestamp: 取消时间
        """
        if self.status in [Order.PENDING, Order.PARTIALLY_FILLED]:
            self.update_status(Order.CANCELLED, timestamp)
    
    def to_dict(self) -> Dict:
        """
        将订单转换为字典
        
        Returns:
            订单信息字典
        """
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'price': self.price,
            'stop_price': self.stop_price,
            'tif': self.tif,
            'status': self.status,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, pd.Timestamp) else self.timestamp,
            'fill_timestamp': self.fill_timestamp.isoformat() if isinstance(self.fill_timestamp, pd.Timestamp) else self.fill_timestamp,
            'strategy_id': self.strategy_id,
            'reason': self.reason
        }
    
    def __repr__(self) -> str:
        """
        订单的字符串表示
        """
        return (f"Order(id={self.order_id}, symbol={self.symbol}, action={self.action}, "
                f"quantity={self.quantity}, type={self.order_type}, status={self.status})")


class Position:
    """
    持仓类
    
    表示某个交易标的的当前持仓状态
    """
    
    def __init__(self, 
                 symbol: str,
                 init_quantity: float = 0.0,
                 init_price: float = 0.0,
                 init_timestamp: Optional[pd.Timestamp] = None):
        """
        初始化持仓
        
        Args:
            symbol: 交易标的
            init_quantity: 初始数量
            init_price: 初始价格
            init_timestamp: 初始时间戳
        """
        self.symbol = symbol
        
        # 持仓数量
        self.quantity = init_quantity
        
        # 持仓均价（按成本计算）
        if init_quantity != 0:
            self.avg_price = init_price
        else:
            self.avg_price = 0.0
        
        # 持仓市值
        self.market_value = 0.0
        
        # 浮动盈亏
        self.unrealized_pnl = 0.0
        
        # 已实现盈亏
        self.realized_pnl = 0.0
        
        # 交易历史
        self.trade_history = []
        
        # 入场时间
        self.entry_timestamp = init_timestamp or pd.Timestamp.now()
        
        # 上次更新时间
        self.last_update_timestamp = init_timestamp or pd.Timestamp.now()
        
        # 交易成本（手续费）
        self.commission = 0.0
    
    def update_trade(self, 
                    quantity: float,
                    price: float,
                    commission: float = 0.0,
                    timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        更新交易记录和持仓信息
        
        Args:
            quantity: 交易数量（正数为买入/加仓，负数为卖出/减仓）
            price: 交易价格
            commission: 手续费
            timestamp: 交易时间
        """
        ts = timestamp or pd.Timestamp.now()
        
        # 记录交易
        self.trade_history.append({
            'timestamp': ts,
            'quantity': quantity,
            'price': price,
            'commission': commission
        })
        
        # 计算已实现盈亏
        if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            # 平仓操作，计算已实现盈亏
            closing_quantity = min(abs(quantity), abs(self.quantity))
            if self.quantity > 0:  # 多头平仓
                realized_pnl = (price - self.avg_price) * closing_quantity
            else:  # 空头平仓
                realized_pnl = (self.avg_price - price) * closing_quantity
                
            self.realized_pnl += realized_pnl
        
        # 更新持仓成本和均价
        if quantity != 0:
            old_value = self.quantity * self.avg_price
            new_value = abs(quantity) * price
            
            # 如果方向相同，合并持仓
            if (self.quantity >= 0 and quantity > 0) or (self.quantity <= 0 and quantity < 0):
                if self.quantity + quantity != 0:
                    self.avg_price = (old_value + new_value) / abs(self.quantity + quantity)
            else:
                # 如果方向相反，先平掉旧持仓，再计算新均价
                if abs(quantity) > abs(self.quantity):
                    # 完全平仓后还有剩余
                    remaining_quantity = quantity + self.quantity
                    self.avg_price = price
                elif abs(quantity) == abs(self.quantity):
                    # 完全平仓
                    self.avg_price = 0.0
            
            # 更新持仓数量
            self.quantity += quantity
        
        # 更新手续费
        self.commission += commission
        
        # 更新时间戳
        self.last_update_timestamp = ts
        
        # 如果持仓变为0，重置入场时间
        if self.quantity == 0:
            self.entry_timestamp = None
        elif self.entry_timestamp is None:
            # 如果新建仓，设置入场时间
            self.entry_timestamp = ts
    
    def update_market_value(self, 
                          current_price: float,
                          timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        更新市值和浮动盈亏
        
        Args:
            current_price: 当前价格
            timestamp: 时间戳
        """
        ts = timestamp or pd.Timestamp.now()
        
        # 计算市值
        self.market_value = abs(self.quantity) * current_price
        
        # 计算浮动盈亏
        if self.quantity > 0:  # 多头
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        elif self.quantity < 0:  # 空头
            self.unrealized_pnl = (self.avg_price - current_price) * abs(self.quantity)
        else:
            self.unrealized_pnl = 0.0
        
        # 更新时间戳
        self.last_update_timestamp = ts
    
    def get_total_pnl(self) -> float:
        """
        获取总盈亏（已实现+浮动）
        
        Returns:
            总盈亏金额
        """
        return self.realized_pnl + self.unrealized_pnl
    
    def get_holding_period(self, timestamp: Optional[pd.Timestamp] = None) -> Optional[pd.Timedelta]:
        """
        获取持仓时间
        
        Args:
            timestamp: 当前时间
            
        Returns:
            持仓时间
        """
        if self.entry_timestamp is None:
            return None
        
        ts = timestamp or pd.Timestamp.now()
        return ts - self.entry_timestamp
    
    def to_dict(self) -> Dict:
        """
        将持仓转换为字典
        
        Returns:
            持仓信息字典
        """
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.get_total_pnl(),
            'commission': self.commission,
            'entry_timestamp': self.entry_timestamp.isoformat() if isinstance(self.entry_timestamp, pd.Timestamp) else self.entry_timestamp,
            'last_update_timestamp': self.last_update_timestamp.isoformat() if isinstance(self.last_update_timestamp, pd.Timestamp) else self.last_update_timestamp,
            'trade_count': len(self.trade_history)
        }
    
    def __repr__(self) -> str:
        """
        持仓的字符串表示
        """
        return (f"Position(symbol={self.symbol}, quantity={self.quantity}, "
                f"avg_price={self.avg_price}, market_value={self.market_value}, "
                f"unrealized_pnl={self.unrealized_pnl})")


class Portfolio:
    """
    投资组合类
    
    管理多个交易标的的持仓，处理资金分配和风险控制
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000.0,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化投资组合
        
        Args:
            initial_capital: 初始资金
            cache_manager: 缓存管理器
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.transaction_history = []
        self.order_history = []
        self.equity_history = []
        self.last_update_timestamp = None
        
        self.cache_manager = cache_manager
    
    def add_position(self, 
                    symbol: str,
                    quantity: float,
                    price: float,
                    commission: float = 0.0,
                    timestamp: Optional[pd.Timestamp] = None) -> Position:
        """
        添加或更新持仓
        
        Args:
            symbol: 交易标的
            quantity: 交易数量
            price: 交易价格
            commission: 手续费
            timestamp: 交易时间
            
        Returns:
            更新后的持仓对象
        """
        ts = timestamp or pd.Timestamp.now()
        
        # 如果持仓不存在，创建新持仓
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0.0, 0.0, ts)
        
        # 更新持仓
        position = self.positions[symbol]
        position.update_trade(quantity, price, commission, ts)
        
        # 记录交易历史
        self.transaction_history.append({
            'timestamp': ts,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'cash_change': -quantity * price - commission
        })
        
        # 更新现金
        self.cash -= quantity * price + commission
        
        # 更新时间戳
        self.last_update_timestamp = ts
        
        # 更新权益历史
        self._update_equity_history(ts)
        
        return position
    
    def close_position(self, 
                      symbol: str,
                      price: float,
                      commission: float = 0.0,
                      timestamp: Optional[pd.Timestamp] = None) -> Optional[Position]:
        """
        平仓
        
        Args:
            symbol: 交易标的
            price: 平仓价格
            commission: 手续费
            timestamp: 平仓时间
            
        Returns:
            平仓后的持仓对象（应该为None）
        """
        if symbol not in self.positions:
            logger.warning(f"尝试平掉不存在的持仓: {symbol}")
            return None
        
        position = self.positions[symbol]
        if position.quantity == 0:
            logger.warning(f"持仓 {symbol} 数量为0，无需平仓")
            return position
        
        # 平掉全部仓位
        close_quantity = -position.quantity
        self.add_position(symbol, close_quantity, price, commission, timestamp)
        
        # 如果持仓数量为0，从positions中移除
        if position.quantity == 0:
            del self.positions[symbol]
            return None
        
        return position
    
    def update_market_values(self, 
                           prices: Dict[str, float],
                           timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        更新所有持仓的市值
        
        Args:
            prices: 价格字典 {symbol: price}
            timestamp: 时间戳
        """
        ts = timestamp or pd.Timestamp.now()
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_market_value(prices[symbol], ts)
            else:
                logger.warning(f"缺少持仓 {symbol} 的市场价格")
        
        # 更新权益历史
        self._update_equity_history(ts)
    
    def _update_equity_history(self, timestamp: pd.Timestamp) -> None:
        """
        更新权益历史
        
        Args:
            timestamp: 时间戳
        """
        equity = self.get_total_equity()
        self.equity_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'market_value': self.get_total_market_value(),
            'equity': equity,
            'unrealized_pnl': self.get_total_unrealized_pnl(),
            'realized_pnl': self.get_total_realized_pnl()
        })
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取特定标的的持仓
        
        Args:
            symbol: 交易标的
            
        Returns:
            持仓对象，如果不存在则返回None
        """
        return self.positions.get(symbol)
    
    def get_active_positions(self) -> Dict[str, Position]:
        """
        获取所有非零持仓
        
        Returns:
            非零持仓字典
        """
        return {symbol: pos for symbol, pos in self.positions.items() if pos.quantity != 0}
    
    def get_total_market_value(self) -> float:
        """
        获取总市值
        
        Returns:
            总市值
        """
        return sum(position.market_value for position in self.positions.values())
    
    def get_total_equity(self) -> float:
        """
        获取总资产（现金+市值）
        
        Returns:
            总资产
        """
        return self.cash + self.get_total_market_value()
    
    def get_total_unrealized_pnl(self) -> float:
        """
        获取总浮动盈亏
        
        Returns:
            总浮动盈亏
        """
        return sum(position.unrealized_pnl for position in self.positions.values())
    
    def get_total_realized_pnl(self) -> float:
        """
        获取总已实现盈亏
        
        Returns:
            总已实现盈亏
        """
        return sum(position.realized_pnl for position in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """
        获取总盈亏
        
        Returns:
            总盈亏
        """
        return self.get_total_realized_pnl() + self.get_total_unrealized_pnl()
    
    def get_equity_series(self) -> pd.Series:
        """
        获取权益时间序列
        
        Returns:
            权益时间序列
        """
        if not self.equity_history:
            return pd.Series()
        
        df = pd.DataFrame(self.equity_history)
        df.set_index('timestamp', inplace=True)
        return df['equity']
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        计算基本绩效指标
        
        Returns:
            绩效指标字典
        """
        equity_series = self.get_equity_series()
        if len(equity_series) < 2:
            return {}
        
        # 计算收益率
        returns = equity_series.pct_change().dropna()
        
        # 总收益率
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        
        # 年化收益率
        days = (equity_series.index[-1] - equity_series.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 波动率
        annual_volatility = returns.std() * np.sqrt(252)  # 假设252个交易日
        
        # 最大回撤
        roll_max = equity_series.cummax()
        drawdown = (equity_series - roll_max) / roll_max
        max_drawdown = drawdown.min()
        
        # 夏普比率 (假设无风险利率为0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.transaction_history),
            'current_equity': equity_series.iloc[-1],
            'initial_capital': self.initial_capital
        }
    
    def to_dict(self) -> Dict:
        """
        将投资组合转换为字典
        
        Returns:
            投资组合信息字典
        """
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'total_market_value': self.get_total_market_value(),
            'total_equity': self.get_total_equity(),
            'total_unrealized_pnl': self.get_total_unrealized_pnl(),
            'total_realized_pnl': self.get_total_realized_pnl(),
            'position_count': len(self.positions),
            'active_position_count': len(self.get_active_positions()),
            'transaction_count': len(self.transaction_history),
            'last_update_timestamp': self.last_update_timestamp.isoformat() if isinstance(self.last_update_timestamp, pd.Timestamp) else self.last_update_timestamp,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
        }
    
    def __repr__(self) -> str:
        """
        投资组合的字符串表示
        """
        return (f"Portfolio(cash={self.cash:.2f}, market_value={self.get_total_market_value():.2f}, "
                f"equity={self.get_total_equity():.2f}, positions={len(self.positions)})")


class StrategyBase:
    """
    策略基类
    
    所有交易策略都应该继承这个基类，并实现相应的方法
    """
    
    def __init__(self, 
                 name: str,
                 params: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            params: 策略参数
            cache_manager: 缓存管理器
        """
        self.name = name
        self.params = params or {}
        self.cache_manager = cache_manager
        
        # 初始化数据和状态
        self.data = {}
        self.signals = {}
        self.orders = []
        self.positions = {}
        self.trade_history = []
        
        # 策略状态
        self.is_initialized = False
        self.current_timestamp = None
        
        # 性能统计
        self.performance = {}
    
    def initialize(self, context: Dict) -> None:
        """
        策略初始化
        
        在回测开始前调用，用于设置初始状态、加载数据等
        
        Args:
            context: 上下文信息，包含数据、因子等
        """
        self.is_initialized = True
        logger.info(f"策略 {self.name} 已初始化")
    
    def on_data(self, data: Dict, timestamp: pd.Timestamp) -> List[Order]:
        """
        处理数据并生成交易信号
        
        在每个时间点调用，接收当前数据并返回交易订单
        
        Args:
            data: 当前时间点的数据
            timestamp: 当前时间戳
            
        Returns:
            订单列表
        """
        self.current_timestamp = timestamp
        self.data[timestamp] = data
        return []
    
    def generate_signals(self, data: Dict, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        生成交易信号
        
        Args:
            data: 当前数据
            timestamp: 当前时间戳
            
        Returns:
            信号字典 {symbol: signal_value}
        """
        # 子类应该实现此方法
        raise NotImplementedError("子类必须实现 generate_signals 方法")
    
    def signals_to_orders(self, 
                         signals: Dict[str, float],
                         data: Dict,
                         portfolio: Portfolio,
                         timestamp: pd.Timestamp) -> List[Order]:
        """
        将信号转换为订单
        
        Args:
            signals: 信号字典
            data: 当前数据
            portfolio: 当前投资组合
            timestamp: 当前时间戳
            
        Returns:
            订单列表
        """
        orders = []
        
        # 子类应该实现此方法
        return orders
    
    def on_order_filled(self, order: Order, timestamp: pd.Timestamp) -> None:
        """
        订单成交回调
        
        Args:
            order: 已成交的订单
            timestamp: 成交时间
        """
        # 记录交易历史
        self.trade_history.append({
            'timestamp': timestamp,
            'order': order.to_dict()
        })
        
        # 更新持仓
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        
        # 更新持仓数量
        if order.action in [Order.BUY, Order.COVER]:
            self.positions[symbol] += order.filled_quantity
        elif order.action in [Order.SELL, Order.SHORT]:
            self.positions[symbol] -= order.filled_quantity
    
    def on_order_cancelled(self, order: Order, timestamp: pd.Timestamp) -> None:
        """
        订单取消回调
        
        Args:
            order: 已取消的订单
            timestamp: 取消时间
        """
        pass
    
    def on_order_rejected(self, order: Order, timestamp: pd.Timestamp, reason: str) -> None:
        """
        订单拒绝回调
        
        Args:
            order: 已拒绝的订单
            timestamp: 拒绝时间
            reason: 拒绝原因
        """
        pass
    
    def on_end_of_day(self, timestamp: pd.Timestamp) -> None:
        """
        每日结束时的回调
        
        Args:
            timestamp: 日期时间
        """
        pass
    
    def on_end_of_backtest(self) -> None:
        """
        回测结束时的回调
        
        用于清理资源、生成报告等
        """
        logger.info(f"策略 {self.name} 回测结束")
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略性能摘要
        
        Returns:
            性能摘要字典
        """
        return self.performance
    
    def load_data(self, 
                 data_loader: DataLoader,
                 symbols: List[str],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 **kwargs) -> None:
        """
        加载策略所需数据
        
        Args:
            data_loader: 数据加载器
            symbols: 交易标的列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
        """
        pass
    
    def set_params(self, params: Dict) -> None:
        """
        设置策略参数
        
        Args:
            params: 参数字典
        """
        self.params.update(params)
    
    def reset(self) -> None:
        """
        重置策略状态
        """
        self.signals = {}
        self.orders = []
        self.trade_history = []
        self.positions = {}
        self.is_initialized = False
        self.current_timestamp = None
        logger.info(f"策略 {self.name} 已重置")
    
    def to_dict(self) -> Dict:
        """
        将策略转换为字典
        
        Returns:
            策略信息字典
        """
        return {
            'name': self.name,
            'params': self.params,
            'is_initialized': self.is_initialized,
            'positions': self.positions,
            'trade_count': len(self.trade_history),
            'performance': self.performance
        }
    
    def __repr__(self) -> str:
        """
        策略的字符串表示
        """
        return f"Strategy(name={self.name}, initialized={self.is_initialized})")


class SingleAssetStrategy(StrategyBase):
    """
    单资产策略基类
    
    用于交易单个资产的策略
    """
    
    def __init__(self, 
                 name: str,
                 symbol: str,
                 params: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化单资产策略
        
        Args:
            name: 策略名称
            symbol: 交易标的
            params: 策略参数
            cache_manager: 缓存管理器
        """
        super().__init__(name, params, cache_manager)
        self.symbol = symbol
    
    def generate_signals(self, data: Dict, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        生成单资产交易信号
        
        Args:
            data: 当前数据
            timestamp: 当前时间戳
            
        Returns:
            信号字典
        """
        # 默认实现为0信号
        return {self.symbol: 0.0}
    
    def get_position(self) -> float:
        """
        获取当前持仓
        
        Returns:
            持仓数量
        """
        return self.positions.get(self.symbol, 0.0)
    
    def __repr__(self) -> str:
        """
        单资产策略的字符串表示
        """
        return (f"SingleAssetStrategy(name={self.name}, symbol={self.symbol}, "
                f"position={self.get_position()})")


class MultiAssetStrategy(StrategyBase):
    """
    多资产策略基类
    
    用于交易多个资产的策略
    """
    
    def __init__(self, 
                 name: str,
                 symbols: List[str],
                 params: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化多资产策略
        
        Args:
            name: 策略名称
            symbols: 交易标的列表
            params: 策略参数
            cache_manager: 缓存管理器
        """
        super().__init__(name, params, cache_manager)
        self.symbols = symbols
    
    def generate_signals(self, data: Dict, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        生成多资产交易信号
        
        Args:
            data: 当前数据
            timestamp: 当前时间戳
            
        Returns:
            信号字典
        """
        # 默认实现为所有资产0信号
        return {symbol: 0.0 for symbol in self.symbols}
    
    def get_active_positions(self) -> Dict[str, float]:
        """
        获取所有非零持仓
        
        Returns:
            非零持仓字典
        """
        return {symbol: pos for symbol, pos in self.positions.items() if pos != 0}
    
    def __repr__(self) -> str:
        """
        多资产策略的字符串表示
        """
        active_positions = len(self.get_active_positions())
        return (f"MultiAssetStrategy(name={self.name}, symbols={len(self.symbols)}, "
                f"active_positions={active_positions})")


class FactorBasedStrategy(StrategyBase):
    """
    因子驱动策略基类
    
    使用因子信号进行交易决策的策略
    """
    
    def __init__(self, 
                 name: str,
                 factor_container: FactorContainer,
                 params: Optional[Dict] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化因子驱动策略
        
        Args:
            name: 策略名称
            factor_container: 因子容器
            params: 策略参数
            cache_manager: 缓存管理器
        """
        super().__init__(name, params, cache_manager)
        self.factor_container = factor_container
        self.symbols = []
    
    def initialize(self, context: Dict) -> None:
        """
        初始化策略
        
        Args:
            context: 上下文信息
        """
        super().initialize(context)
        
        # 获取因子容器中的所有标的
        if hasattr(self.factor_container, 'symbols'):
            self.symbols = self.factor_container.symbols
    
    def generate_signals(self, data: Dict, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        基于因子生成交易信号
        
        Args:
            data: 当前数据
            timestamp: 当前时间戳
            
        Returns:
            信号字典
        """
        signals = {}
        
        # 获取当前时间点的因子值
        factor_values = self.factor_container.get_factor_values_at(timestamp)
        
        if not factor_values:
            return signals
        
        # 默认简单地将因子值作为信号
        # 子类应该实现更复杂的信号生成逻辑
        for symbol, factors in factor_values.items():
            # 可以根据需要组合多个因子
            # 这里简单地取第一个因子的值作为示例
            if factors:
                signals[symbol] = next(iter(factors.values()))
        
        return signals
    
    def update_factors(self, timestamp: pd.Timestamp) -> None:
        """
        更新因子值
        
        Args:
            timestamp: 当前时间戳
        """
        if hasattr(self.factor_container, 'update'):
            self.factor_container.update(timestamp)
    
    def __repr__(self) -> str:
        """
        因子驱动策略的字符串表示
        """
        factor_count = len(self.factor_container.factors) if hasattr(self.factor_container, 'factors') else 0
        return (f"FactorBasedStrategy(name={self.name}, factors={factor_count}, "
                f"symbols={len(self.symbols)})")


# 便捷函数
def create_order(symbol: str,
                action: str,
                quantity: float,
                order_type: str = Order.MARKET,
                price: Optional[float] = None,
                reason: Optional[str] = None,
                timestamp: Optional[pd.Timestamp] = None) -> Order:
    """
    创建订单的便捷函数
    
    Args:
        symbol: 交易标的
        action: 交易动作
        quantity: 交易数量
        order_type: 订单类型
        price: 价格
        reason: 下单原因
        timestamp: 时间戳
        
    Returns:
        订单对象
    """
    return Order(
        symbol=symbol,
        action=action,
        quantity=quantity,
        order_type=order_type,
        price=price,
        reason=reason,
        timestamp=timestamp
    )


def calculate_position_size(equity: float,
                           risk_per_trade: float,
                           entry_price: float,
                           stop_loss_price: float,
                           max_position_size: Optional[float] = None) -> float:
    """
    计算头寸大小
    
    Args:
        equity: 总资产
        risk_per_trade: 每笔交易的风险比例
        entry_price: 入场价格
        stop_loss_price: 止损价格
        max_position_size: 最大头寸比例（可选）
        
    Returns:
        头寸大小
    """
    # 计算风险金额
    risk_amount = equity * risk_per_trade
    
    # 计算每单位的风险
    if stop_loss_price > 0:
        risk_per_unit = abs(entry_price - stop_loss_price)
    else:
        # 如果没有设置止损，使用固定比例
        risk_per_unit = entry_price * 0.01  # 默认1%
    
    # 计算头寸大小
    if risk_per_unit > 0:
        position_size = risk_amount / risk_per_unit
    else:
        position_size = 0
    
    # 应用最大头寸限制
    if max_position_size is not None:
        max_size = equity * max_position_size / entry_price
        position_size = min(position_size, max_size)
    
    return position_size


def normalize_signals(signals: Dict[str, float],
                      method: str = 'zscore') -> Dict[str, float]:
    """
    标准化信号
    
    Args:
        signals: 原始信号
        method: 标准化方法 ('zscore', 'minmax', 'rank')
        
    Returns:
        标准化后的信号
    """
    if not signals:
        return signals
    
    values = np.array(list(signals.values()))
    symbols = list(signals.keys())
    
    if method == 'zscore':
        # Z-Score标准化
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val > 0:
            normalized = (values - mean_val) / std_val
        else:
            normalized = np.zeros_like(values)
    
    elif method == 'minmax':
        # 最小-最大标准化
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)
    
    elif method == 'rank':
        # 排名标准化
        ranks = np.argsort(np.argsort(values))
        normalized = (ranks - ranks.mean()) / ranks.std() if ranks.std() > 0 else np.zeros_like(ranks)
    
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    return {symbol: value for symbol, value in zip(symbols, normalized)}


def filter_signals(signals: Dict[str, float],
                   threshold: float = 0.0) -> Dict[str, float]:
    """
    根据阈值过滤信号
    
    Args:
        signals: 信号字典
        threshold: 信号阈值
        
    Returns:
        过滤后的信号
    """
    return {symbol: value for symbol, value in signals.items() if abs(value) >= threshold}


def rank_assets(signals: Dict[str, float],
                top_n: Optional[int] = None,
                bottom_n: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    根据信号对资产进行排名
    
    Args:
        signals: 信号字典
        top_n: 选择前N个
        bottom_n: 选择后N个
        
    Returns:
        (top_assets, bottom_assets) 元组
    """
    # 按信号值排序
    sorted_assets = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    
    # 选择前N个和后N个
    top_assets = [asset for asset, _ in sorted_assets[:top_n]] if top_n else []
    
    # 反转排序以获取底部资产
    bottom_assets = [asset for asset, _ in sorted_assets[-bottom_n:]][::-1] if bottom_n else []
    
    return top_assets, bottom_assets