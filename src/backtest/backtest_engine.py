import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


class BacktestEngine(ABC):
    """
    回测引擎基类，为量化策略回测提供统一框架
    
    设计理念：
    1. 事件驱动架构：基于时间序列的事件处理机制
    2. 模块化设计：清晰分离数据、策略和执行逻辑
    3. 灵活配置：支持多种回测参数和约束条件
    4. 完整记录：详细记录交易历史和绩效指标
    5. 可视化支持：内置绩效分析和可视化功能
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000.0,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.0001,
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金率
            slippage: 滑点
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # 核心数据结构
        self.data = None  # 回测数据
        self.positions = {}  # 持仓状态
        self.trades = []  # 交易记录
        self.equity_curve = None  # 权益曲线
        self.performance = {}  # 绩效指标
        
        # 回测状态
        self.current_date = None
        self.current_capital = initial_capital
        self.is_running = False
        self._trades_executed = 0
    
    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """
        抽象方法：加载回测数据
        
        Returns:
            回测数据DataFrame
        """
        pass
    
    @abstractmethod
    def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        抽象方法：生成交易信号
        
        Args:
            data: 回测数据
            
        Returns:
            包含交易信号的DataFrame
        """
        pass
    
    def _execute_trade(self, 
                      date: datetime,
                      symbol: str,
                      trade_type: str,  # 'buy' 或 'sell'
                      price: float,
                      quantity: float) -> Dict[str, Any]:
        """
        执行交易
        
        Args:
            date: 交易日期
            symbol: 标的代码
            trade_type: 交易类型（'buy'或'sell'）
            price: 交易价格
            quantity: 交易数量
            
        Returns:
            交易记录字典
        """
        # 计算滑点后的实际价格
        if trade_type == 'buy':
            actual_price = price * (1 + self.slippage)
            cost = actual_price * quantity
        else:  # sell
            actual_price = price * (1 - self.slippage)
            cost = -actual_price * quantity
        
        # 计算佣金
        commission = abs(cost) * self.commission_rate
        
        # 更新资金
        self.current_capital -= (cost + commission)
        
        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        
        if trade_type == 'buy':
            self.positions[symbol] += quantity
        else:  # sell
            self.positions[symbol] -= quantity
        
        # 创建交易记录
        trade_record = {
            'date': date,
            'symbol': symbol,
            'trade_type': trade_type,
            'price': actual_price,
            'quantity': quantity,
            'cost': cost,
            'commission': commission,
            'total_cost': cost + commission,
            'capital_after': self.current_capital,
            'position_after': self.positions[symbol]
        }
        
        self.trades.append(trade_record)
        self._trades_executed += 1
        
        return trade_record
    
    def _calculate_equity(self) -> pd.DataFrame:
        """
        计算权益曲线
        
        Returns:
            权益曲线DataFrame
        """
        # 创建权益曲线DataFrame
        equity = pd.DataFrame(index=self.data.index)
        equity['cash'] = self.initial_capital
        equity['holdings_value'] = 0.0
        equity['total_equity'] = self.initial_capital
        equity['daily_return'] = 0.0
        equity['cumulative_return'] = 0.0
        
        # 初始化变量
        current_cash = self.initial_capital
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # 计算每个时间点的权益
        for date in equity.index:
            # 更新现金
            if not trades_df.empty:
                date_trades = trades_df[trades_df['date'] <= date]
                if not date_trades.empty:
                    total_cost = date_trades['total_cost'].sum()
                    current_cash = self.initial_capital + total_cost
            
            # 计算持仓价值
            holdings_value = 0.0
            for symbol, quantity in self.positions.items():
                if quantity != 0 and symbol in self.data.columns:
                    if f'{symbol}_close' in self.data.columns:
                        current_price = self.data.loc[date, f'{symbol}_close']
                    else:
                        current_price = self.data.loc[date, symbol]
                    holdings_value += quantity * current_price
            
            # 更新权益曲线
            equity.loc[date, 'cash'] = current_cash
            equity.loc[date, 'holdings_value'] = holdings_value
            equity.loc[date, 'total_equity'] = current_cash + holdings_value
        
        # 计算收益率
        equity['daily_return'] = equity['total_equity'].pct_change()
        equity['cumulative_return'] = (1 + equity['daily_return']).cumprod() - 1
        
        # 处理NaN值
        equity['daily_return'] = equity['daily_return'].fillna(0)
        equity['cumulative_return'] = equity['cumulative_return'].fillna(0)
        
        return equity
    
    def _calculate_performance(self) -> Dict[str, float]:
        """
        计算绩效指标
        
        Returns:
            绩效指标字典
        """
        if self.equity_curve is None:
            return {}
        
        # 获取权益曲线数据
        equity = self.equity_curve['total_equity']
        returns = self.equity_curve['daily_return']
        
        # 计算基本绩效指标
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        
        # 计算年化收益率
        trading_days = len(equity)
        annual_return = ((1 + total_return/100) ** (252 / trading_days) - 1) * 100
        
        # 计算波动率
        annual_volatility = returns.std() * np.sqrt(252) * 100
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # 计算最大回撤
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # 计算胜率
        winning_days = (returns > 0).sum()
        win_rate = (winning_days / len(returns)) * 100
        
        # 计算平均盈亏比
        avg_win = returns[returns > 0].mean()
        avg_loss = -returns[returns < 0].mean()
        profit_factor = avg_win / avg_loss if avg_loss != 0 else float('inf')
        
        # 构建绩效指标字典
        performance = {
            'initial_capital': self.initial_capital,
            'final_capital': equity.iloc[-1],
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'annual_volatility_pct': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'trading_days': trading_days,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'avg_daily_return_pct': returns.mean() * 100
        }
        
        return performance
    
    def run(self) -> Dict[str, Any]:
        """
        运行回测
        
        Returns:
            回测结果字典
        """
        # 设置回测状态
        self.is_running = True
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        
        try:
            # 1. 加载数据
            print("加载回测数据...")
            self.data = self._load_data()
            
            # 2. 过滤日期范围
            if self.start_date:
                self.data = self.data[self.data.index >= self.start_date]
            if self.end_date:
                self.data = self.data[self.data.index <= self.end_date]
            
            # 3. 生成交易信号
            print("生成交易信号...")
            signals = self._generate_signals(self.data)
            
            # 4. 执行交易
            print("执行回测...")
            for date in self.data.index:
                self.current_date = date
                
                # 处理该日期的所有信号
                if date in signals.index:
                    date_signals = signals.loc[date]
                    
                    # 确保date_signals是DataFrame
                    if isinstance(date_signals, pd.Series):
                        date_signals = date_signals.to_frame().T
                    
                    for _, signal in date_signals.iterrows():
                        symbol = signal.get('symbol')
                        action = signal.get('action')
                        price = signal.get('price')
                        quantity = signal.get('quantity')
                        
                        if symbol and action and price and quantity:
                            self._execute_trade(date, symbol, action, price, quantity)
            
            # 5. 计算权益曲线
            print("计算权益曲线...")
            self.equity_curve = self._calculate_equity()
            
            # 6. 计算绩效指标
            print("计算绩效指标...")
            self.performance = self._calculate_performance()
            
            print(f"回测完成！共执行 {self._trades_executed} 笔交易")
            
        except Exception as e:
            print(f"回测过程中发生错误: {e}")
            self.is_running = False
            raise
        
        self.is_running = False
        
        # 返回回测结果
        return {
            'performance': self.performance,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """
        绘制权益曲线
        
        Args:
            save_path: 保存图片的路径（可选）
        """
        if self.equity_curve is None:
            print("请先运行回测")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['total_equity'], label='总资产')
        plt.plot(self.equity_curve.index, self.equity_curve['cash'], label='现金')
        plt.plot(self.equity_curve.index, self.equity_curve['holdings_value'], label='持仓价值')
        plt.title('回测权益曲线')
        plt.xlabel('日期')
        plt.ylabel('金额')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"权益曲线已保存至: {save_path}")
        
        plt.show()
    
    def plot_returns(self, save_path: Optional[str] = None):
        """
        绘制收益率曲线
        
        Args:
            save_path: 保存图片的路径（可选）
        """
        if self.equity_curve is None:
            print("请先运行回测")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['cumulative_return'] * 100, label='累计收益率(%)')
        plt.title('回测累计收益率')
        plt.xlabel('日期')
        plt.ylabel('累计收益率(%)')
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"收益率曲线已保存至: {save_path}")
        
        plt.show()
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取回测结果
        
        Returns:
            回测结果字典
        """
        return {
            'performance': self.performance,
            'trades': pd.DataFrame(self.trades) if self.trades else None,
            'equity_curve': self.equity_curve
        }
    
    def save_results(self, output_dir: str):
        """
        保存回测结果
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存绩效指标
        performance_path = os.path.join(output_dir, 'performance.json')
        import json
        with open(performance_path, 'w', encoding='utf-8') as f:
            json.dump(self.performance, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存交易记录
        if self.trades:
            trades_path = os.path.join(output_dir, 'trades.csv')
            pd.DataFrame(self.trades).to_csv(trades_path, index=False)
        
        # 保存权益曲线
        if self.equity_curve is not None:
            equity_path = os.path.join(output_dir, 'equity_curve.csv')
            self.equity_curve.to_csv(equity_path)
        
        print(f"回测结果已保存至: {output_dir}")
    
    def __str__(self) -> str:
        """
        返回回测引擎的字符串表示
        """
        status = "运行中" if self.is_running else "已停止"
        return f"回测引擎 ({status}, 初始资金: {self.initial_capital}, 佣金率: {self.commission_rate})
