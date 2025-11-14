import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import copy
import time
import threading
from datetime import datetime, timedelta

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.strategy_base import StrategyBase, Order, Portfolio
from src.factors.factor_base import FactorContainer
from src.data.data_loader import DataLoader
from src.utils.helpers import setup_logger, DEFAULT_LOGGER, timeit, log_performance
from src.utils.cache_manager import CacheManager

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class Event:
    """
    事件基类
    
    所有事件的基础类
    """
    
    # 事件类型
    MARKET_DATA = 'market_data'  # 市场数据事件
    SIGNAL = 'signal'            # 信号事件
    ORDER = 'order'              # 订单事件
    FILL = 'fill'                # 成交事件
    ORDER_CANCEL = 'order_cancel'  # 订单取消事件
    ORDER_REJECT = 'order_reject'  # 订单拒绝事件
    END_OF_DAY = 'end_of_day'    # 日终事件
    END_OF_BACKTEST = 'end_of_backtest'  # 回测结束事件
    
    def __init__(self, 
                 event_type: str,
                 timestamp: Optional[pd.Timestamp] = None,
                 **kwargs):
        """
        初始化事件
        
        Args:
            event_type: 事件类型
            timestamp: 事件时间戳
            **kwargs: 其他事件数据
        """
        self.event_type = event_type
        self.timestamp = timestamp or pd.Timestamp.now()
        self.data = kwargs
    
    def __repr__(self) -> str:
        """
        事件的字符串表示
        """
        return f"Event(type={self.event_type}, time={self.timestamp})"


class MarketDataEvent(Event):
    """
    市场数据事件
    
    包含特定时间点的市场数据
    """
    
    def __init__(self, 
                 timestamp: pd.Timestamp,
                 data: Dict[str, pd.DataFrame],
                 symbols: Optional[List[str]] = None):
        """
        初始化市场数据事件
        
        Args:
            timestamp: 数据时间戳
            data: 市场数据 {symbol: data_frame}
            symbols: 交易标的列表
        """
        super().__init__(Event.MARKET_DATA, timestamp)
        self.data = data
        self.symbols = symbols or list(data.keys())
    
    def get_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取特定标的的数据
        
        Args:
            symbol: 交易标的
            
        Returns:
            数据帧，如果不存在则返回None
        """
        return self.data.get(symbol)


class SignalEvent(Event):
    """
    信号事件
    
    包含策略生成的交易信号
    """
    
    def __init__(self, 
                 timestamp: pd.Timestamp,
                 strategy_id: str,
                 signals: Dict[str, float],
                 reason: Optional[str] = None):
        """
        初始化信号事件
        
        Args:
            timestamp: 信号时间戳
            strategy_id: 策略ID
            signals: 信号字典 {symbol: signal_value}
            reason: 信号生成原因
        """
        super().__init__(Event.SIGNAL, timestamp)
        self.strategy_id = strategy_id
        self.signals = signals
        self.reason = reason


class OrderEvent(Event):
    """
    订单事件
    
    包含要发送到执行系统的订单
    """
    
    def __init__(self, 
                 timestamp: pd.Timestamp,
                 order: Order,
                 strategy_id: Optional[str] = None):
        """
        初始化订单事件
        
        Args:
            timestamp: 订单时间戳
            order: 订单对象
            strategy_id: 策略ID
        """
        super().__init__(Event.ORDER, timestamp)
        self.order = order
        self.strategy_id = strategy_id


class FillEvent(Event):
    """
    成交事件
    
    包含订单的成交信息
    """
    
    def __init__(self, 
                 timestamp: pd.Timestamp,
                 order: Order,
                 fill_price: float,
                 fill_quantity: float,
                 commission: float = 0.0,
                 strategy_id: Optional[str] = None):
        """
        初始化成交事件
        
        Args:
            timestamp: 成交时间戳
            order: 已成交的订单
            fill_price: 成交价格
            fill_quantity: 成交数量
            commission: 手续费
            strategy_id: 策略ID
        """
        super().__init__(Event.FILL, timestamp)
        self.order = order
        self.fill_price = fill_price
        self.fill_quantity = fill_quantity
        self.commission = commission
        self.strategy_id = strategy_id


class OrderCancelEvent(Event):
    """
    订单取消事件
    
    包含已取消的订单信息
    """
    
    def __init__(self, 
                 timestamp: pd.Timestamp,
                 order: Order,
                 reason: Optional[str] = None,
                 strategy_id: Optional[str] = None):
        """
        初始化订单取消事件
        
        Args:
            timestamp: 取消时间戳
            order: 已取消的订单
            reason: 取消原因
            strategy_id: 策略ID
        """
        super().__init__(Event.ORDER_CANCEL, timestamp)
        self.order = order
        self.reason = reason
        self.strategy_id = strategy_id


class OrderRejectEvent(Event):
    """
    订单拒绝事件
    
    包含被拒绝的订单信息
    """
    
    def __init__(self, 
                 timestamp: pd.Timestamp,
                 order: Order,
                 reason: str,
                 strategy_id: Optional[str] = None):
        """
        初始化订单拒绝事件
        
        Args:
            timestamp: 拒绝时间戳
            order: 被拒绝的订单
            reason: 拒绝原因
            strategy_id: 策略ID
        """
        super().__init__(Event.ORDER_REJECT, timestamp)
        self.order = order
        self.reason = reason
        self.strategy_id = strategy_id


class EndOfDayEvent(Event):
    """
    日终事件
    
    表示一个交易日的结束
    """
    
    def __init__(self, timestamp: pd.Timestamp):
        """
        初始化日终事件
        
        Args:
            timestamp: 日终时间戳
        """
        super().__init__(Event.END_OF_DAY, timestamp)


class EndOfBacktestEvent(Event):
    """
    回测结束事件
    
    表示整个回测过程的结束
    """
    
    def __init__(self, timestamp: pd.Timestamp):
        """
        初始化回测结束事件
        
        Args:
            timestamp: 结束时间戳
        """
        super().__init__(Event.END_OF_BACKTEST, timestamp)


class EventQueue:
    """
    事件队列
    
    管理事件的FIFO队列
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        初始化事件队列
        
        Args:
            max_size: 队列最大大小（可选）
        """
        self.events = []
        self.max_size = max_size
        self.lock = threading.RLock()  # 使用可重入锁保证线程安全
    
    def put(self, event: Event) -> None:
        """
        向队列中添加事件
        
        Args:
            event: 要添加的事件
        """
        with self.lock:
            # 按时间戳排序插入
            # 找到合适的插入位置
            insert_idx = 0
            while insert_idx < len(self.events) and self.events[insert_idx].timestamp <= event.timestamp:
                insert_idx += 1
            
            self.events.insert(insert_idx, event)
            
            # 如果队列大小超过最大限制，移除最旧的事件
            if self.max_size is not None and len(self.events) > self.max_size:
                self.events.pop(0)
    
    def get(self) -> Optional[Event]:
        """
        从队列中获取并移除最早的事件
        
        Returns:
            事件对象，如果队列为空则返回None
        """
        with self.lock:
            if not self.events:
                return None
            return self.events.pop(0)
    
    def peek(self) -> Optional[Event]:
        """
        查看最早的事件但不移除
        
        Returns:
            事件对象，如果队列为空则返回None
        """
        with self.lock:
            if not self.events:
                return None
            return self.events[0]
    
    def empty(self) -> bool:
        """
        检查队列是否为空
        
        Returns:
            如果队列为空则返回True
        """
        with self.lock:
            return len(self.events) == 0
    
    def size(self) -> int:
        """
        获取队列大小
        
        Returns:
            队列中的事件数量
        """
        with self.lock:
            return len(self.events)
    
    def clear(self) -> None:
        """
        清空队列
        """
        with self.lock:
            self.events.clear()


class MarketSimulator:
    """
    市场模拟器
    
    模拟市场行为，处理订单执行
    """
    
    def __init__(self, 
                 initial_data: Optional[Dict[str, pd.DataFrame]] = None,
                 slippage_model: Optional[Callable] = None,
                 commission_model: Optional[Callable] = None,
                 market_impact: Optional[float] = 0.0,
                 min_tick: Optional[float] = None):
        """
        初始化市场模拟器
        
        Args:
            initial_data: 初始市场数据
            slippage_model: 滑点模型函数
            commission_model: 佣金模型函数
            market_impact: 市场冲击成本（百分比）
            min_tick: 最小价格变动单位
        """
        self.data = initial_data or {}
        self.slippage_model = slippage_model or self._default_slippage
        self.commission_model = commission_model or self._default_commission
        self.market_impact = market_impact
        self.min_tick = min_tick
        
        # 订单簿 (简化版)
        self.order_book = {}
        
        # 当前时间
        self.current_timestamp = None
        
        # 历史价格数据
        self.price_history = {}
    
    def _default_slippage(self, order: Order, price: float) -> float:
        """
        默认滑点模型
        
        Args:
            order: 订单
            price: 理论价格
            
        Returns:
            调整后的价格
        """
        # 简单的滑点模型：买入滑点为正，卖出滑点为负
        slippage = price * 0.0001  # 0.01% 的滑点
        
        if order.action in [Order.BUY, Order.COVER]:
            return price + slippage
        elif order.action in [Order.SELL, Order.SHORT]:
            return price - slippage
        return price
    
    def _default_commission(self, order: Order, price: float) -> float:
        """
        默认佣金模型
        
        Args:
            order: 订单
            price: 成交价格
            
        Returns:
            佣金金额
        """
        # 简单的佣金模型：固定比例
        commission_rate = 0.0002  # 0.02% 的佣金
        commission = order.quantity * price * commission_rate
        
        # 最小佣金
        min_commission = 1.0
        return max(commission, min_commission)
    
    def update_data(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> None:
        """
        更新市场数据
        
        Args:
            data: 市场数据
            timestamp: 时间戳
        """
        self.data = data
        self.current_timestamp = timestamp
        
        # 更新价格历史
        for symbol, df in data.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = {}
            
            # 保存OHLCV数据
            if not df.empty:
                latest_row = df.iloc[-1]
                self.price_history[symbol][timestamp] = {
                    'open': latest_row.get('open', 0),
                    'high': latest_row.get('high', 0),
                    'low': latest_row.get('low', 0),
                    'close': latest_row.get('close', 0),
                    'volume': latest_row.get('volume', 0)
                }
    
    def get_current_price(self, symbol: str, price_type: str = 'close') -> Optional[float]:
        """
        获取当前价格
        
        Args:
            symbol: 交易标的
            price_type: 价格类型 ('open', 'high', 'low', 'close')
            
        Returns:
            当前价格，如果不存在则返回None
        """
        if symbol not in self.data or self.data[symbol].empty:
            return None
        
        latest_row = self.data[symbol].iloc[-1]
        return latest_row.get(price_type, None)
    
    def execute_order(self, order: Order, timestamp: pd.Timestamp) -> Tuple[bool, float, float, float]:
        """
        执行订单
        
        Args:
            order: 要执行的订单
            timestamp: 执行时间
            
        Returns:
            (是否成功, 成交价格, 成交数量, 手续费)
        """
        # 检查标的是否存在
        if order.symbol not in self.data or self.data[order.symbol].empty:
            return False, 0.0, 0.0, 0.0
        
        # 获取当前价格
        current_price = self.get_current_price(order.symbol)
        if current_price is None:
            return False, 0.0, 0.0, 0.0
        
        # 应用滑点
        execution_price = self.slippage_model(order, current_price)
        
        # 应用市场冲击
        if self.market_impact > 0:
            impact = current_price * self.market_impact * order.quantity / 1000000  # 简化的市场冲击模型
            if order.action in [Order.BUY, Order.COVER]:
                execution_price += impact
            elif order.action in [Order.SELL, Order.SHORT]:
                execution_price -= impact
        
        # 应用最小价格变动单位
        if self.min_tick is not None:
            execution_price = round(execution_price / self.min_tick) * self.min_tick
        
        # 计算手续费
        commission = self.commission_model(order, execution_price)
        
        # 市价单立即完全成交
        if order.order_type == Order.MARKET:
            return True, execution_price, order.quantity, commission
        
        # 限价单根据条件成交
        elif order.order_type == Order.LIMIT:
            if order.price is None:
                return False, 0.0, 0.0, 0.0
                
            if ((order.action in [Order.BUY, Order.COVER] and execution_price <= order.price) or
                (order.action in [Order.SELL, Order.SHORT] and execution_price >= order.price)):
                # 限价单满足条件，完全成交
                return True, execution_price, order.quantity, commission
            else:
                # 限价单不满足条件，添加到订单簿
                if order.symbol not in self.order_book:
                    self.order_book[order.symbol] = []
                self.order_book[order.symbol].append((timestamp, order))
                return False, 0.0, 0.0, 0.0
        
        # 止损单和止损限价单简化处理
        elif order.order_type in [Order.STOP, Order.STOP_LIMIT]:
            if order.stop_price is None:
                return False, 0.0, 0.0, 0.0
                
            stop_triggered = ((order.action in [Order.BUY, Order.COVER] and execution_price >= order.stop_price) or
                            (order.action in [Order.SELL, Order.SHORT] and execution_price <= order.stop_price))
            
            if stop_triggered:
                if order.order_type == Order.STOP:
                    # 止损单触发后转为市价单
                    return True, execution_price, order.quantity, commission
                else:  # STOP_LIMIT
                    # 止损限价单触发后需要检查限价
                    if order.price is None:
                        return False, 0.0, 0.0, 0.0
                        
                    if ((order.action in [Order.BUY, Order.COVER] and execution_price <= order.price) or
                        (order.action in [Order.SELL, Order.SHORT] and execution_price >= order.price)):
                        return True, execution_price, order.quantity, commission
                    else:
                        # 触发了但价格不满足限价，添加到订单簿
                        if order.symbol not in self.order_book:
                            self.order_book[order.symbol] = []
                        self.order_book[order.symbol].append((timestamp, order))
                        return False, 0.0, 0.0, 0.0
            else:
                # 止损未触发，添加到订单簿
                if order.symbol not in self.order_book:
                    self.order_book[order.symbol] = []
                self.order_book[order.symbol].append((timestamp, order))
                return False, 0.0, 0.0, 0.0
        
        return False, 0.0, 0.0, 0.0
    
    def check_pending_orders(self, timestamp: pd.Timestamp) -> List[Tuple[Order, float, float, float]]:
        """
        检查订单簿中等待的订单
        
        Args:
            timestamp: 当前时间戳
            
        Returns:
            可执行的订单列表 [(订单, 成交价格, 成交数量, 手续费)]
        """
        executable_orders = []
        
        for symbol, orders in list(self.order_book.items()):
            remaining_orders = []
            
            for _, order in orders:
                # 检查订单有效期
                if order.tif == 'day':
                    # 如果是当日有效订单，检查是否过期
                    if timestamp.date() > order.timestamp.date():
                        continue
                
                # 尝试执行订单
                success, price, quantity, commission = self.execute_order(order, timestamp)
                if success:
                    executable_orders.append((order, price, quantity, commission))
                else:
                    remaining_orders.append((timestamp, order))
            
            # 更新订单簿
            if remaining_orders:
                self.order_book[symbol] = remaining_orders
            else:
                del self.order_book[symbol]
        
        return executable_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功取消
        """
        for symbol, orders in list(self.order_book.items()):
            remaining_orders = []
            found = False
            
            for timestamp, order in orders:
                if order.order_id == order_id:
                    found = True
                else:
                    remaining_orders.append((timestamp, order))
            
            if found:
                if remaining_orders:
                    self.order_book[symbol] = remaining_orders
                else:
                    del self.order_book[symbol]
                return True
        
        return False


class ExecutionHandler:
    """
    执行处理器
    
    处理订单的执行逻辑
    """
    
    def __init__(self, 
                 market_simulator: MarketSimulator,
                 event_queue: EventQueue,
                 allow_partial_fills: bool = False):
        """
        初始化执行处理器
        
        Args:
            market_simulator: 市场模拟器
            event_queue: 事件队列
            allow_partial_fills: 是否允许部分成交
        """
        self.market_simulator = market_simulator
        self.event_queue = event_queue
        self.allow_partial_fills = allow_partial_fills
        
        # 跟踪待处理的订单
        self.pending_orders = {}
    
    def handle_order(self, order_event: OrderEvent) -> None:
        """
        处理订单事件
        
        Args:
            order_event: 订单事件
        """
        order = order_event.order
        timestamp = order_event.timestamp
        
        # 执行订单
        success, fill_price, fill_quantity, commission = self.market_simulator.execute_order(order, timestamp)
        
        if success:
            # 订单成功执行
            order.fill(fill_quantity, fill_price, commission, timestamp)
            
            # 创建成交事件
            fill_event = FillEvent(
                timestamp=timestamp,
                order=order,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                commission=commission,
                strategy_id=order_event.strategy_id
            )
            
            self.event_queue.put(fill_event)
            logger.info(f"订单成交: {order.order_id}, 价格: {fill_price}, 数量: {fill_quantity}")
        else:
            # 订单未立即执行，添加到待处理订单
            self.pending_orders[order.order_id] = order
            logger.info(f"订单已添加到待处理队列: {order.order_id}")
    
    def handle_cancel_order(self, order_id: str, timestamp: pd.Timestamp, reason: Optional[str] = None) -> bool:
        """
        处理订单取消
        
        Args:
            order_id: 订单ID
            timestamp: 取消时间
            reason: 取消原因
            
        Returns:
            是否成功取消
        """
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.cancel(timestamp)
            
            # 创建取消事件
            cancel_event = OrderCancelEvent(
                timestamp=timestamp,
                order=order,
                reason=reason
            )
            
            self.event_queue.put(cancel_event)
            del self.pending_orders[order_id]
            logger.info(f"订单已取消: {order_id}, 原因: {reason}")
            return True
        
        # 尝试从市场模拟器的订单簿中取消
        canceled = self.market_simulator.cancel_order(order_id)
        if canceled:
            logger.info(f"订单已从市场订单簿中取消: {order_id}")
        
        return canceled
    
    def process_pending_orders(self, timestamp: pd.Timestamp) -> None:
        """
        处理待执行的订单
        
        Args:
            timestamp: 当前时间戳
        """
        # 检查市场模拟器中的订单簿
        executable_orders = self.market_simulator.check_pending_orders(timestamp)
        
        for order, fill_price, fill_quantity, commission in executable_orders:
            # 执行订单
            order.fill(fill_quantity, fill_price, commission, timestamp)
            
            # 创建成交事件
            fill_event = FillEvent(
                timestamp=timestamp,
                order=order,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                commission=commission
            )
            
            self.event_queue.put(fill_event)
            logger.info(f"待处理订单成交: {order.order_id}, 价格: {fill_price}, 数量: {fill_quantity}")
            
            # 从待处理订单中移除
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]


class EventDrivenBacktest(BacktestEngine):
    """
    事件驱动回测引擎
    
    基于事件驱动架构的回测系统
    """
    
    def __init__(self, 
                 strategies: List[StrategyBase],
                 data_loader: DataLoader,
                 initial_capital: float = 1000000.0,
                 symbols: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 cache_manager: Optional[CacheManager] = None,
                 enable_performance_logging: bool = True,
                 slippage: float = 0.0,
                 commission: float = 0.0,
                 market_impact: float = 0.0):
        """
        初始化事件驱动回测引擎
        
        Args:
            strategies: 策略列表
            data_loader: 数据加载器
            initial_capital: 初始资金
            symbols: 交易标的列表
            start_date: 开始日期
            end_date: 结束日期
            cache_manager: 缓存管理器
            enable_performance_logging: 是否启用性能日志
            slippage: 滑点（百分比）
            commission: 佣金（百分比）
            market_impact: 市场冲击成本（百分比）
        """
        super().__init__(
            strategies=strategies,
            data_loader=data_loader,
            initial_capital=initial_capital,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            cache_manager=cache_manager
        )
        
        # 配置参数
        self.enable_performance_logging = enable_performance_logging
        self.slippage = slippage
        self.commission = commission
        self.market_impact = market_impact
        
        # 初始化组件
        self.event_queue = EventQueue()
        self.portfolio = Portfolio(initial_capital, cache_manager)
        
        # 创建自定义滑点和佣金模型
        def custom_slippage(order, price):
            slippage_amount = price * slippage
            if order.action in [Order.BUY, Order.COVER]:
                return price + slippage_amount
            elif order.action in [Order.SELL, Order.SHORT]:
                return price - slippage_amount
            return price
        
        def custom_commission(order, price):
            return order.quantity * price * commission
        
        # 初始化市场模拟器
        self.market_simulator = MarketSimulator(
            slippage_model=custom_slippage,
            commission_model=custom_commission,
            market_impact=market_impact
        )
        
        # 初始化执行处理器
        self.execution_handler = ExecutionHandler(
            market_simulator=self.market_simulator,
            event_queue=self.event_queue
        )
        
        # 回测状态
        self.is_running = False
        self.current_timestamp = None
        self.data_iterator = None
        self.total_events_processed = 0
        self.start_time = 0
        
        # 结果存储
        self.results = {
            'equity_curve': None,
            'trades': [],
            'orders': [],
            'performance': {},
            'daily_stats': None
        }
    
    def _initialize(self) -> None:
        """
        初始化回测环境
        """
        # 加载数据
        self._load_data()
        
        # 初始化策略
        context = {
            'data_loader': self.data_loader,
            'portfolio': self.portfolio,
            'cache_manager': self.cache_manager
        }
        
        for strategy in self.strategies:
            strategy.initialize(context)
        
        # 初始化数据迭代器
        self._initialize_data_iterator()
        
        logger.info(f"回测初始化完成，策略数量: {len(self.strategies)}")
        logger.info(f"数据时间范围: {self.start_date} 至 {self.end_date}")
        logger.info(f"初始资金: {self.initial_capital}")
    
    def _load_data(self) -> None:
        """
        加载回测数据
        """
        if not self.symbols:
            raise ValueError("必须指定交易标的")
        
        # 加载OHLCV数据
        self.data = self.data_loader.load_ohlcv(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 确保所有标的都有数据
        if not self.data or not all(symbol in self.data for symbol in self.symbols):
            missing_symbols = [symbol for symbol in self.symbols if symbol not in self.data]
            if missing_symbols:
                raise ValueError(f"缺少以下标的的数据: {missing_symbols}")
    
    def _initialize_data_iterator(self) -> None:
        """
        初始化数据迭代器
        
        创建按时间排序的市场数据事件流
        """
        if not self.data:
            raise ValueError("没有可用的数据")
        
        # 获取所有时间戳
        all_timestamps = set()
        for symbol, df in self.data.items():
            if not df.empty:
                all_timestamps.update(df.index)
        
        # 排序时间戳
        self.timestamps = sorted(list(all_timestamps))
        
        # 过滤时间范围
        if self.start_date:
            start_ts = pd.Timestamp(self.start_date)
            self.timestamps = [ts for ts in self.timestamps if ts >= start_ts]
        
        if self.end_date:
            end_ts = pd.Timestamp(self.end_date)
            self.timestamps = [ts for ts in self.timestamps if ts <= end_ts]
        
        if not self.timestamps:
            raise ValueError("指定时间范围内没有数据")
        
        # 创建数据迭代器
        self.data_iterator = iter(enumerate(self.timestamps))
        
        logger.info(f"数据迭代器初始化完成，共有 {len(self.timestamps)} 个时间点")
    
    def _process_market_data_event(self, event: MarketDataEvent) -> None:
        """
        处理市场数据事件
        
        Args:
            event: 市场数据事件
        """
        timestamp = event.timestamp
        self.current_timestamp = timestamp
        
        # 更新市场模拟器数据
        self.market_simulator.update_data(event.data, timestamp)
        
        # 更新投资组合市值
        prices = {}
        for symbol in event.symbols:
            price = self.market_simulator.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        
        if prices:
            self.portfolio.update_market_values(prices, timestamp)
        
        # 处理待执行订单
        self.execution_handler.process_pending_orders(timestamp)
        
        # 发送数据到所有策略
        for strategy in self.strategies:
            orders = strategy.on_data(event.data, timestamp)
            
            # 创建订单事件
            for order in orders:
                order_event = OrderEvent(
                    timestamp=timestamp,
                    order=order,
                    strategy_id=strategy.name
                )
                self.event_queue.put(order_event)
    
    def _process_signal_event(self, event: SignalEvent) -> None:
        """
        处理信号事件
        
        Args:
            event: 信号事件
        """
        # 找到对应的策略
        strategy = next((s for s in self.strategies if s.name == event.strategy_id), None)
        if not strategy:
            logger.warning(f"找不到对应的策略: {event.strategy_id}")
            return
        
        # 将信号转换为订单
        orders = strategy.signals_to_orders(
            signals=event.signals,
            data=self.market_simulator.data,
            portfolio=self.portfolio,
            timestamp=event.timestamp
        )
        
        # 创建订单事件
        for order in orders:
            order_event = OrderEvent(
                timestamp=event.timestamp,
                order=order,
                strategy_id=strategy.name
            )
            self.event_queue.put(order_event)
    
    def _process_order_event(self, event: OrderEvent) -> None:
        """
        处理订单事件
        
        Args:
            event: 订单事件
        """
        # 检查资金是否充足
        order = event.order
        current_price = self.market_simulator.get_current_price(order.symbol)
        
        if current_price is None:
            # 价格不可用，拒绝订单
            reject_event = OrderRejectEvent(
                timestamp=event.timestamp,
                order=order,
                reason="价格数据不可用",
                strategy_id=event.strategy_id
            )
            self.event_queue.put(reject_event)
            return
        
        # 计算所需资金
        required_funds = order.quantity * current_price * 1.1  # 增加10%的缓冲
        
        if order.action in [Order.BUY, Order.COVER] and required_funds > self.portfolio.cash:
            # 资金不足，拒绝订单
            reject_event = OrderRejectEvent(
                timestamp=event.timestamp,
                order=order,
                reason="资金不足",
                strategy_id=event.strategy_id
            )
            self.event_queue.put(reject_event)
            return
        
        # 记录订单
        self.portfolio.order_history.append(order.to_dict())
        
        # 发送订单到执行处理器
        self.execution_handler.handle_order(event)
    
    def _process_fill_event(self, event: FillEvent) -> None:
        """
        处理成交事件
        
        Args:
            event: 成交事件
        """
        order = event.order
        
        # 更新投资组合
        quantity = order.quantity if order.action in [Order.BUY, Order.COVER] else -order.quantity
        
        if order.action == Order.SHORT:
            # 做空操作，数量为负
            quantity = -order.quantity
        elif order.action == Order.COVER:
            # 平空操作，数量为正
            pass  # 已经是正的
        
        self.portfolio.add_position(
            symbol=order.symbol,
            quantity=quantity,
            price=event.fill_price,
            commission=event.commission,
            timestamp=event.timestamp
        )
        
        # 通知策略订单已成交
        strategy = next((s for s in self.strategies if s.name == event.strategy_id), None)
        if strategy:
            strategy.on_order_filled(order, event.timestamp)
        
        # 记录交易
        self.results['trades'].append({
            'timestamp': event.timestamp,
            'symbol': order.symbol,
            'action': order.action,
            'quantity': order.quantity,
            'price': event.fill_price,
            'commission': event.commission,
            'strategy': event.strategy_id,
            'order_id': order.order_id
        })
    
    def _process_order_cancel_event(self, event: OrderCancelEvent) -> None:
        """
        处理订单取消事件
        
        Args:
            event: 订单取消事件
        """
        # 通知策略订单已取消
        strategy = next((s for s in self.strategies if s.name == event.strategy_id), None)
        if strategy:
            strategy.on_order_cancelled(event.order, event.timestamp)
    
    def _process_order_reject_event(self, event: OrderRejectEvent) -> None:
        """
        处理订单拒绝事件
        
        Args:
            event: 订单拒绝事件
        """
        # 通知策略订单已拒绝
        strategy = next((s for s in self.strategies if s.name == event.strategy_id), None)
        if strategy:
            strategy.on_order_rejected(event.order, event.timestamp, event.reason)
    
    def _process_end_of_day_event(self, event: EndOfDayEvent) -> None:
        """
        处理日终事件
        
        Args:
            event: 日终事件
        """
        # 通知所有策略日终处理
        for strategy in self.strategies:
            strategy.on_end_of_day(event.timestamp)
        
        # 记录每日统计
        if self.enable_performance_logging:
            daily_stats = {
                'date': event.timestamp.date(),
                'equity': self.portfolio.get_total_equity(),
                'cash': self.portfolio.cash,
                'market_value': self.portfolio.get_total_market_value(),
                'positions': len(self.portfolio.get_active_positions())
            }
            
            if self.results['daily_stats'] is None:
                self.results['daily_stats'] = []
            
            self.results['daily_stats'].append(daily_stats)
    
    def _process_end_of_backtest_event(self, event: EndOfBacktestEvent) -> None:
        """
        处理回测结束事件
        
        Args:
            event: 回测结束事件
        """
        # 通知所有策略回测结束
        for strategy in self.strategies:
            strategy.on_end_of_backtest()
        
        # 计算最终绩效
        self._calculate_performance()
        
        # 停止回测
        self.is_running = False
    
    def _create_market_data_event(self, timestamp: pd.Timestamp) -> MarketDataEvent:
        """
        创建市场数据事件
        
        Args:
            timestamp: 时间戳
            
        Returns:
            市场数据事件
        """
        # 收集当前时间点的数据
        data_at_timestamp = {}
        
        for symbol, df in self.data.items():
            if timestamp in df.index:
                # 获取该时间点的数据（包括之前的所有历史数据）
                data_at_timestamp[symbol] = df.loc[:timestamp]
        
        return MarketDataEvent(
            timestamp=timestamp,
            data=data_at_timestamp,
            symbols=list(data_at_timestamp.keys())
        )
    
    def _check_end_of_day(self, current_ts: pd.Timestamp, next_ts: Optional[pd.Timestamp]) -> bool:
        """
        检查是否是交易日结束
        
        Args:
            current_ts: 当前时间戳
            next_ts: 下一个时间戳
            
        Returns:
            如果是交易日结束则返回True
        """
        if next_ts is None:
            return True
        
        # 简单判断：如果下一个时间戳是新的一天，则当前是交易日结束
        return next_ts.date() > current_ts.date()
    
    def _calculate_performance(self) -> None:
        """
        计算回测绩效指标
        """
        # 获取权益曲线
        equity_series = self.portfolio.get_equity_series()
        self.results['equity_curve'] = equity_series
        
        # 计算投资组合绩效
        portfolio_metrics = self.portfolio.get_performance_metrics()
        
        # 计算策略绩效
        strategy_metrics = {}
        for strategy in self.strategies:
            strategy_metrics[strategy.name] = strategy.get_performance_summary()
        
        # 汇总绩效
        self.results['performance'] = {
            'portfolio': portfolio_metrics,
            'strategies': strategy_metrics,
            'total_trades': len(self.results['trades']),
            'total_orders': len(self.portfolio.order_history),
            'events_processed': self.total_events_processed,
            'backtest_duration': time.time() - self.start_time
        }
        
        # 转换每日统计为DataFrame
        if self.results['daily_stats']:
            self.results['daily_stats'] = pd.DataFrame(self.results['daily_stats'])
            self.results['daily_stats'].set_index('date', inplace=True)
    
    def run(self) -> Dict:
        """
        运行回测
        
        Returns:
            回测结果
        """
        # 初始化
        self._initialize()
        
        self.is_running = True
        self.start_time = time.time()
        
        logger.info("开始回测...")
        
        try:
            # 主回测循环
            while self.is_running:
                # 尝试获取下一个时间点的数据
                try:
                    idx, timestamp = next(self.data_iterator)
                except StopIteration:
                    # 数据迭代结束
                    end_event = EndOfBacktestEvent(timestamp=self.current_timestamp or pd.Timestamp.now())
                    self.event_queue.put(end_event)
                    break
                
                # 创建并处理市场数据事件
                market_data_event = self._create_market_data_event(timestamp)
                self._process_market_data_event(market_data_event)
                self.total_events_processed += 1
                
                # 处理队列中的其他事件
                while not self.event_queue.empty():
                    event = self.event_queue.get()
                    
                    # 根据事件类型处理
                    if event.event_type == Event.SIGNAL:
                        self._process_signal_event(event)
                    elif event.event_type == Event.ORDER:
                        self._process_order_event(event)
                    elif event.event_type == Event.FILL:
                        self._process_fill_event(event)
                    elif event.event_type == Event.ORDER_CANCEL:
                        self._process_order_cancel_event(event)
                    elif event.event_type == Event.ORDER_REJECT:
                        self._process_order_reject_event(event)
                    elif event.event_type == Event.END_OF_DAY:
                        self._process_end_of_day_event(event)
                    elif event.event_type == Event.END_OF_BACKTEST:
                        self._process_end_of_backtest_event(event)
                    
                    self.total_events_processed += 1
                
                # 检查是否是交易日结束
                next_idx = idx + 1
                next_ts = self.timestamps[next_idx] if next_idx < len(self.timestamps) else None
                
                if self._check_end_of_day(timestamp, next_ts):
                    eod_event = EndOfDayEvent(timestamp=timestamp)
                    self._process_end_of_day_event(eod_event)
                
                # 进度报告
                if idx > 0 and idx % 100 == 0:
                    progress = (idx / len(self.timestamps)) * 100
                    elapsed = time.time() - self.start_time
                    logger.info(f"回测进度: {progress:.2f}%, 已处理 {idx} 个时间点, 耗时 {elapsed:.2f} 秒")
            
        except KeyboardInterrupt:
            logger.info("回测被用户中断")
        except Exception as e:
            logger.error(f"回测过程中出错: {str(e)}", exc_info=True)
        finally:
            if self.is_running:
                self.is_running = False
                
                # 确保计算绩效
                if not self.results['performance']:
                    self._calculate_performance()
        
        # 记录最终结果
        duration = time.time() - self.start_time
        logger.info(f"回测完成! 耗时 {duration:.2f} 秒")
        logger.info(f"总事件处理数: {self.total_events_processed}")
        
        # 输出绩效摘要
        if self.results['performance']:
            portfolio_perf = self.results['performance']['portfolio']
            logger.info(f"\n回测绩效摘要:")
            logger.info(f"总收益率: {portfolio_perf.get('total_return', 0) * 100:.2f}%")
            logger.info(f"年化收益率: {portfolio_perf.get('annual_return', 0) * 100:.2f}%")
            logger.info(f"夏普比率: {portfolio_perf.get('sharpe_ratio', 0):.2f}")
            logger.info(f"最大回撤: {portfolio_perf.get('max_drawdown', 0) * 100:.2f}%")
            logger.info(f"胜率: {portfolio_perf.get('win_rate', 0) * 100:.2f}%")
            logger.info(f"总交易次数: {self.results['performance'].get('total_trades', 0)}")
            logger.info(f"最终权益: {portfolio_perf.get('current_equity', 0):.2f}")
        
        return self.results
    
    def stop(self) -> None:
        """
        停止回测
        """
        self.is_running = False
        logger.info("回测已停止")
    
    def get_results(self) -> Dict:
        """
        获取回测结果
        
        Returns:
            回测结果字典
        """
        return self.results
    
    def plot_results(self, plot_type: str = 'equity') -> None:
        """
        绘制回测结果
        
        Args:
            plot_type: 图表类型 ('equity', 'drawdown', 'returns')
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.error("需要安装matplotlib库来绘制结果")
            return
        
        if not self.results['equity_curve']:
            logger.error("没有可用的权益数据")
            return
        
        equity = self.results['equity_curve']
        
        plt.figure(figsize=(12, 6))
        
        if plot_type == 'equity':
            # 绘制权益曲线
            plt.plot(equity.index, equity.values)
            plt.title('权益曲线')
            plt.ylabel('权益')
            plt.xlabel('日期')
            plt.grid(True)
            
            # 格式化x轴日期
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
        elif plot_type == 'drawdown':
            # 计算回撤
            roll_max = equity.cummax()
            drawdown = (equity - roll_max) / roll_max * 100
            
            plt.plot(drawdown.index, drawdown.values, color='red')
            plt.title('回撤 (%)')
            plt.ylabel('回撤百分比')
            plt.xlabel('日期')
            plt.grid(True)
            
            # 填充回撤区域
            plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            
            # 格式化x轴日期
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
        elif plot_type == 'returns':
            # 计算日收益率
            returns = equity.pct_change() * 100
            
            plt.plot(returns.index, returns.values)
            plt.title('日收益率 (%)')
            plt.ylabel('收益率百分比')
            plt.xlabel('日期')
            plt.grid(True)
            
            # 格式化x轴日期
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        else:
            logger.error(f"不支持的图表类型: {plot_type}")
            return
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        生成回测报告
        
        Args:
            output_file: 输出文件路径（可选）
            
        Returns:
            报告内容
        """
        if not self.results['performance']:
            return "没有可用的回测结果"
        
        # 构建报告
        report = []
        report.append("========== 回测报告 ==========\n")
        
        # 基本信息
        report.append(f"开始日期: {self.start_date}")
        report.append(f"结束日期: {self.end_date}")
        report.append(f"初始资金: {self.initial_capital}")
        report.append(f"交易标的: {', '.join(self.symbols)}")
        report.append(f"策略数量: {len(self.strategies)}")
        report.append("")
        
        # 绩效指标
        portfolio_perf = self.results['performance']['portfolio']
        report.append("------ 投资组合绩效 ------")
        report.append(f"总收益率: {portfolio_perf.get('total_return', 0) * 100:.2f}%")
        report.append(f"年化收益率: {portfolio_perf.get('annual_return', 0) * 100:.2f}%")
        report.append(f"年化波动率: {portfolio_perf.get('annual_volatility', 0) * 100:.2f}%")
        report.append(f"夏普比率: {portfolio_perf.get('sharpe_ratio', 0):.2f}")
        report.append(f"最大回撤: {portfolio_perf.get('max_drawdown', 0) * 100:.2f}%")
        report.append(f"胜率: {portfolio_perf.get('win_rate', 0) * 100:.2f}%")
        report.append(f"总交易次数: {self.results['performance'].get('total_trades', 0)}")
        report.append(f"最终权益: {portfolio_perf.get('current_equity', 0):.2f}")
        report.append("")
        
        # 策略绩效
        if self.results['performance']['strategies']:
            report.append("------ 策略绩效 ------")
            for strategy_name, metrics in self.results['performance']['strategies'].items():
                report.append(f"\n策略: {strategy_name}")
                for key, value in metrics.items():
                    report.append(f"  {key}: {value}")
            report.append("")
        
        # 回测统计
        report.append("------ 回测统计 ------")
        report.append(f"总事件处理数: {self.results['performance'].get('events_processed', 0)}")
        report.append(f"回测耗时: {self.results['performance'].get('backtest_duration', 0):.2f} 秒")
        report.append(f"总订单数: {self.results['performance'].get('total_orders', 0)}")
        report.append("")
        
        report.append("========================\n")
        
        # 组合报告
        report_str = "\n".join(report)
        
        # 输出到文件
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_str)
                logger.info(f"报告已保存到: {output_file}")
            except Exception as e:
                logger.error(f"保存报告失败: {str(e)}")
        
        return report_str


# 便捷函数
def create_event_driven_backtest(strategies: List[StrategyBase],
                                data_loader: DataLoader,
                                config: Optional[Dict] = None) -> EventDrivenBacktest:
    """
    创建事件驱动回测引擎的便捷函数
    
    Args:
        strategies: 策略列表
        data_loader: 数据加载器
        config: 配置字典
        
    Returns:
        回测引擎实例
    """
    config = config or {}
    
    return EventDrivenBacktest(
        strategies=strategies,
        data_loader=data_loader,
        initial_capital=config.get('initial_capital', 1000000.0),
        symbols=config.get('symbols'),
        start_date=config.get('start_date'),
        end_date=config.get('end_date'),
        slippage=config.get('slippage', 0.0),
        commission=config.get('commission', 0.0),
        market_impact=config.get('market_impact', 0.0)
    )


def run_backtest(strategies: List[StrategyBase],
                data_loader: DataLoader,
                config: Optional[Dict] = None) -> Dict:
    """
    运行回测的便捷函数
    
    Args:
        strategies: 策略列表
        data_loader: 数据加载器
        config: 配置字典
        
    Returns:
        回测结果
    """
    backtest = create_event_driven_backtest(strategies, data_loader, config)
    return backtest.run()