import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.backtest.event_driven_backtest import (
    EventDrivenBacktest, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    MarketSimulator, ExecutionHandler, Portfolio
)
from src.backtest.strategy_base import StrategyBase

class TestEventDrivenBacktest(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.dates = pd.date_range(start='2020-01-01', periods=100)
        self.prices = np.cumsum(np.random.randn(100)) + 100
        self.test_data = pd.DataFrame({
            'date': self.dates,
            'open': self.prices + np.random.randn(100) * 2,
            'high': self.prices + np.random.randn(100) * 2 + 3,
            'low': self.prices + np.random.randn(100) * 2 - 3,
            'close': self.prices,
            'volume': np.random.randint(1000, 100000, 100)
        })
        self.test_data.set_index('date', inplace=True)
        
        # 创建一个简单的测试策略
        class SimpleStrategy(StrategyBase):
            def __init__(self, data, symbol='TEST'):
                super().__init__(data)
                self.symbol = symbol
                self.bought = False
            
            def generate_signals(self):
                signals = pd.Series(index=self.data.index, data=0)
                # 在第20天买入，第50天卖出
                signals.iloc[20] = 1  # 买入信号
                signals.iloc[50] = -1  # 卖出信号
                return signals
        
        self.strategy = SimpleStrategy(self.test_data)
        
        # 初始化回测引擎
        self.backtest = EventDrivenBacktest(
            data=self.test_data, 
            initial_capital=100000, 
            commission=0.001,
            slippage=0.0005
        )
        self.backtest.add_strategy(self.strategy)
    
    def test_event_creation(self):
        # 测试事件创建
        market_event = MarketEvent()
        signal_event = SignalEvent('TEST', 'LONG', 1.0)
        order_event = OrderEvent('TEST', 'BUY', 100)
        fill_event = FillEvent('TEST', 'BUY', 100, 99.5, 0.01)
        
        self.assertIsInstance(market_event, MarketEvent)
        self.assertEqual(signal_event.symbol, 'TEST')
        self.assertEqual(signal_event.signal_type, 'LONG')
        self.assertEqual(order_event.quantity, 100)
        self.assertEqual(fill_event.fill_price, 99.5)
    
    def test_market_simulator(self):
        # 测试市场模拟器
        simulator = MarketSimulator(self.test_data)
        
        # 获取初始市场事件
        event = simulator.get_next_event()
        self.assertIsInstance(event, MarketEvent)
        
        # 验证当前时间戳
        self.assertEqual(simulator.current_index, 0)
        self.assertEqual(simulator.get_current_price('TEST'), self.test_data['close'].iloc[0])
    
    def test_portfolio_initialization(self):
        # 测试投资组合初始化
        portfolio = Portfolio(self.test_data, initial_capital=100000)
        
        self.assertEqual(portfolio.initial_capital, 100000)
        self.assertEqual(portfolio.current_cash, 100000)
        self.assertEqual(len(portfolio.positions), 0)
        self.assertEqual(portfolio.equity_history.iloc[0], 100000)
    
    def test_portfolio_update(self):
        # 测试投资组合更新
        portfolio = Portfolio(self.test_data, initial_capital=100000)
        
        # 添加持仓
        portfolio.add_position('TEST', 100, 99.5)
        
        # 更新投资组合
        portfolio.update()
        
        # 验证持仓和现金
        self.assertEqual(portfolio.positions['TEST'], 100)
        expected_cash = 100000 - 100 * 99.5
        self.assertEqual(portfolio.current_cash, expected_cash)
    
    def test_portfolio_on_fill(self):
        # 测试投资组合处理填充事件
        portfolio = Portfolio(self.test_data, initial_capital=100000)
        fill_event = FillEvent('TEST', 'BUY', 100, 99.5, 0.01)
        
        portfolio.on_fill(fill_event)
        
        # 验证持仓已更新
        self.assertEqual(portfolio.positions['TEST'], 100)
        expected_cost = 100 * 99.5 * (1 + 0.01)  # 包含佣金
        expected_cash = 100000 - expected_cost
        self.assertAlmostEqual(portfolio.current_cash, expected_cash)
    
    def test_execution_handler(self):
        # 测试执行处理器
        simulator = MarketSimulator(self.test_data)
        execution_handler = ExecutionHandler(simulator, commission=0.001, slippage=0.0005)
        
        order_event = OrderEvent('TEST', 'BUY', 100)
        fill_events = execution_handler.execute_order(order_event)
        
        # 验证填充事件
        self.assertEqual(len(fill_events), 1)
        fill_event = fill_events[0]
        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.symbol, 'TEST')
        self.assertEqual(fill_event.quantity, 100)
        # 价格应该接近当前市场价格，考虑滑点
        expected_price = self.test_data['close'].iloc[0] * (1 + 0.0005)  # 买入滑点
        self.assertAlmostEqual(fill_event.fill_price, expected_price, places=2)
    
    @patch('src.backtest.event_driven_backtest.EventDrivenBacktest._run_backtest_loop')
    def test_run_backtest(self, mock_run_loop):
        # 测试运行回测
        self.backtest.run()
        mock_run_loop.assert_called_once()
    
    def test_strategy_signal_generation(self):
        # 测试策略信号生成
        signals = self.strategy.generate_signals()
        
        # 验证信号位置
        self.assertEqual(signals.iloc[20], 1)  # 买入信号
        self.assertEqual(signals.iloc[50], -1)  # 卖出信号
        # 其他位置应该是0
        self.assertEqual((signals.iloc[:20] != 0).sum(), 0)
        self.assertEqual((signals.iloc[21:49] != 0).sum(), 0)
    
    def test_calculate_performance_metrics(self):
        # 模拟回测结果以测试绩效计算
        portfolio = Portfolio(self.test_data, initial_capital=100000)
        
        # 创建一些简单的权益历史
        equity_values = np.linspace(100000, 115000, len(self.test_data))
        portfolio.equity_history = pd.Series(equity_values, index=self.test_data.index)
        
        # 计算绩效指标
        metrics = self.backtest._calculate_performance_metrics(portfolio)
        
        # 验证基本指标
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # 总回报应该约为15%
        self.assertAlmostEqual(metrics['total_return'], 0.15, places=3)

if __name__ == '__main__':
    unittest.main()