import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import json
import os
from datetime import datetime, timedelta
import logging
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.event_driven_backtest import EventDrivenBacktest
from src.backtest.strategy_base import StrategyBase, Portfolio
from src.utils.helpers import setup_logger, DEFAULT_LOGGER, timeit, ensure_directory
from src.utils.cache_manager import CacheManager

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class BacktestRunner:
    """
    回测执行器
    
    负责配置、执行和管理回测流程
    """
    
    def __init__(self, 
                 engine_class: type = EventDrivenBacktest,
                 cache_manager: Optional[CacheManager] = None):
        """
        初始化回测执行器
        
        Args:
            engine_class: 回测引擎类
            cache_manager: 缓存管理器
        """
        self.engine_class = engine_class
        self.cache_manager = cache_manager or CacheManager()
        self.results = {}
        self.runs_history = []
    
    @timeit
    def run_backtest(self, 
                    strategy: StrategyBase,
                    data: Dict[str, pd.DataFrame],
                    initial_capital: float = 1000000.0,
                    start_date: Optional[pd.Timestamp] = None,
                    end_date: Optional[pd.Timestamp] = None,
                    name: Optional[str] = None,
                    **kwargs) -> Dict:
        """
        执行单个回测
        
        Args:
            strategy: 交易策略实例
            data: 数据源 {symbol: dataframe}
            initial_capital: 初始资金
            start_date: 开始日期
            end_date: 结束日期
            name: 回测名称
            **kwargs: 传递给回测引擎的额外参数
            
        Returns:
            回测结果
        """
        # 生成回测名称
        if name is None:
            name = f"{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"开始执行回测: {name}")
        
        # 创建回测引擎实例
        engine = self.engine_class(
            strategy=strategy,
            data=data,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        # 执行回测
        result = engine.run()
        
        # 保存结果
        self.results[name] = result
        
        # 记录执行历史
        self.runs_history.append({
            'name': name,
            'strategy': strategy.name,
            'start_time': datetime.now(),
            'initial_capital': initial_capital,
            'start_date': start_date,
            'end_date': end_date
        })
        
        logger.info(f"回测完成: {name}")
        
        # 计算绩效指标
        result['metrics'] = self._calculate_performance_metrics(result['equity_curve'])
        
        return result
    
    @timeit
    def run_parameter_sweep(self,
                           strategy_class: type,
                           data: Dict[str, pd.DataFrame],
                           param_grid: Dict[str, List[Any]],
                           initial_capital: float = 1000000.0,
                           start_date: Optional[pd.Timestamp] = None,
                           end_date: Optional[pd.Timestamp] = None,
                           n_jobs: int = 1,
                           metric: str = 'sharpe_ratio',
                           maximize: bool = True,
                           name_prefix: str = 'param_sweep') -> Dict[str, Dict]:
        """
        参数优化回测
        
        Args:
            strategy_class: 策略类
            data: 数据源
            param_grid: 参数网格 {param_name: [values]}
            initial_capital: 初始资金
            start_date: 开始日期
            end_date: 结束日期
            n_jobs: 并行作业数（当前版本暂不支持并行）
            metric: 优化指标
            maximize: 是否最大化指标
            name_prefix: 回测名称前缀
            
        Returns:
            所有参数组合的回测结果
        """
        logger.info(f"开始参数优化回测: {strategy_class.__name__}")
        
        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_grid)
        
        # 存储所有结果
        all_results = {}
        best_result = None
        best_params = None
        best_score = -np.inf if maximize else np.inf
        
        # 遍历参数组合
        total = len(param_combinations)
        for i, params in enumerate(param_combinations):
            logger.info(f"执行参数组合 {i+1}/{total}: {params}")
            
            # 创建策略实例
            strategy = strategy_class(**params)
            
            # 执行回测
            name = f"{name_prefix}_{i}_{'_'.join([f'{k}={v}' for k, v in params.items()])}"
            result = self.run_backtest(
                strategy=strategy,
                data=data,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
                name=name
            )
            
            # 保存结果
            all_results[name] = {
                'params': params,
                'result': result
            }
            
            # 更新最佳结果
            current_score = result['metrics'].get(metric, 0)
            if (maximize and current_score > best_score) or \
               (not maximize and current_score < best_score):
                best_score = current_score
                best_params = params
                best_result = result
        
        logger.info(f"参数优化完成，最佳参数: {best_params}, 最佳{metric}: {best_score}")
        
        return {
            'all_results': all_results,
            'best_params': best_params,
            'best_result': best_result,
            'best_score': best_score
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        生成参数组合
        
        Args:
            param_grid: 参数网格
            
        Returns:
            参数组合列表
        """
        # 简单实现：递归生成所有组合
        def recursive_generate(params: Dict, keys: List[str], index: int) -> List[Dict]:
            if index == len(keys):
                return [params.copy()]
            
            key = keys[index]
            values = param_grid[key]
            combinations = []
            
            for value in values:
                params[key] = value
                combinations.extend(recursive_generate(params, keys, index + 1))
            
            return combinations
        
        return recursive_generate({}, list(param_grid.keys()), 0)
    
    def _calculate_performance_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        计算绩效指标
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            绩效指标字典
        """
        if equity_curve.empty:
            return {}
        
        # 计算每日收益率
        returns = equity_curve['total_equity'].pct_change().dropna()
        
        # 基本指标
        total_return = (equity_curve['total_equity'].iloc[-1] / equity_curve['total_equity'].iloc[0] - 1) * 100
        annual_return = (1 + total_return / 100) ** (252 / len(equity_curve)) - 1
        annual_return *= 100
        
        # 风险指标
        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annual_return / 100) / (annual_volatility / 100) if annual_volatility != 0 else 0
        
        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        winning_trades = (returns > 0).sum()
        win_rate = (winning_trades / len(returns)) * 100 if len(returns) > 0 else 0
        
        # 平均盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return / 100) / downside_risk if downside_risk != 0 else 0
        
        # Calmar比率
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'annual_volatility_pct': round(annual_volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_trades': len(returns),
            'trading_days': len(equity_curve)
        }
    
    def save_results(self, name: str, output_dir: str = './results') -> str:
        """
        保存回测结果
        
        Args:
            name: 回测名称
            output_dir: 输出目录
            
        Returns:
            保存文件路径
        """
        if name not in self.results:
            raise ValueError(f"回测结果不存在: {name}")
        
        # 确保目录存在
        ensure_directory(output_dir)
        
        # 保存结果
        result = self.results[name]
        
        # 保存为JSON文件
        json_path = os.path.join(output_dir, f"{name}.json")
        
        # 转换不可序列化的对象
        serializable_result = self._make_result_serializable(result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存权益曲线为CSV
        if 'equity_curve' in result:
            csv_path = os.path.join(output_dir, f"{name}_equity.csv")
            result['equity_curve'].to_csv(csv_path)
        
        logger.info(f"回测结果已保存到: {json_path}")
        
        return json_path
    
    def _make_result_serializable(self, result: Dict) -> Dict:
        """
        将回测结果转换为可序列化的格式
        
        Args:
            result: 回测结果
            
        Returns:
            可序列化的结果
        """
        serializable = {}
        
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                # 转换DataFrame为字典
                serializable[key] = {
                    'columns': value.columns.tolist(),
                    'index': value.index.astype(str).tolist(),
                    'data': value.values.tolist()
                }
            elif isinstance(value, (pd.Series, np.ndarray)):
                # 转换Series或数组为列表
                serializable[key] = value.tolist()
            elif isinstance(value, (pd.Timestamp, datetime)):
                # 转换时间戳为字符串
                serializable[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                # 转换NumPy类型为Python原生类型
                serializable[key] = value.item()
            elif isinstance(value, dict):
                # 递归处理字典
                serializable[key] = self._make_result_serializable(value)
            elif isinstance(value, list):
                # 递归处理列表中的字典
                serializable[key] = [
                    self._make_result_serializable(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                serializable[key] = value
        
        return serializable
    
    def load_results(self, json_path: str) -> Dict:
        """
        加载回测结果
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            回测结果
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 恢复DataFrame
        if 'equity_curve' in result and isinstance(result['equity_curve'], dict):
            df_data = result['equity_curve']
            result['equity_curve'] = pd.DataFrame(
                data=df_data['data'],
                columns=df_data['columns'],
                index=pd.to_datetime(df_data['index'])
            )
        
        return result


class BacktestAnalyzer:
    """
    回测结果分析器
    
    用于分析和可视化回测结果
    """
    
    def __init__(self):
        """
        初始化回测结果分析器
        """
        self.results = None
    
    def set_result(self, result: Dict) -> None:
        """
        设置回测结果
        
        Args:
            result: 回测结果
        """
        self.results = result
    
    def analyze(self, result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析回测结果
        
        Args:
            result: 回测结果（可选，如果不提供则使用已设置的结果）
            
        Returns:
            分析结果
        """
        if result is not None:
            self.set_result(result)
        
        if self.results is None:
            raise ValueError("请先设置回测结果")
        
        analysis = {}
        
        # 提取基本信息
        analysis['strategy_name'] = self.results.get('strategy_name', 'Unknown')
        analysis['initial_capital'] = self.results.get('initial_capital', 0)
        analysis['final_equity'] = self.results['equity_curve']['total_equity'].iloc[-1]
        
        # 绩效指标
        analysis['metrics'] = self.results.get('metrics', {})
        
        # 权益曲线分析
        analysis['equity_analysis'] = self._analyze_equity_curve(self.results['equity_curve'])
        
        # 交易分析
        if 'transactions' in self.results:
            analysis['transaction_analysis'] = self._analyze_transactions(self.results['transactions'])
        
        # 风险分析
        analysis['risk_analysis'] = self._analyze_risk(self.results['equity_curve'])
        
        # 季节性分析
        analysis['seasonal_analysis'] = self._analyze_seasonality(self.results['equity_curve'])
        
        return analysis
    
    def _analyze_equity_curve(self, equity_curve: pd.DataFrame) -> Dict:
        """
        分析权益曲线
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            权益曲线分析结果
        """
        # 计算收益率
        returns = equity_curve['total_equity'].pct_change().dropna()
        cum_returns = (1 + returns).cumprod()
        
        # 提取关键时期
        best_month_idx = cum_returns.resample('M').last().pct_change().idxmax()
        worst_month_idx = cum_returns.resample('M').last().pct_change().idxmin()
        
        # 计算月度、季度和年度表现
        monthly_returns = cum_returns.resample('M').last().pct_change().dropna()
        quarterly_returns = cum_returns.resample('Q').last().pct_change().dropna()
        yearly_returns = cum_returns.resample('Y').last().pct_change().dropna()
        
        return {
            'best_month': {
                'date': str(best_month_idx),
                'return': round(monthly_returns.loc[best_month_idx] * 100, 2)
            } if not monthly_returns.empty else None,
            'worst_month': {
                'date': str(worst_month_idx),
                'return': round(monthly_returns.loc[worst_month_idx] * 100, 2)
            } if not monthly_returns.empty else None,
            'monthly_performance': {
                str(idx): round(ret * 100, 2) 
                for idx, ret in monthly_returns.items()
            },
            'quarterly_performance': {
                str(idx): round(ret * 100, 2) 
                for idx, ret in quarterly_returns.items()
            },
            'yearly_performance': {
                str(idx): round(ret * 100, 2) 
                for idx, ret in yearly_returns.items()
            },
            'avg_monthly_return_pct': round(monthly_returns.mean() * 100, 2) if not monthly_returns.empty else 0,
            'positive_months_pct': round((monthly_returns > 0).mean() * 100, 2) if not monthly_returns.empty else 0
        }
    
    def _analyze_transactions(self, transactions: pd.DataFrame) -> Dict:
        """
        分析交易记录
        
        Args:
            transactions: 交易记录
            
        Returns:
            交易分析结果
        """
        if transactions.empty:
            return {}
        
        # 按资产分组
        transactions_by_symbol = transactions.groupby('symbol')
        
        # 计算每个资产的交易统计
        symbol_stats = {}
        for symbol, group in transactions_by_symbol:
            buys = group[group['action'].isin(['BUY', 'COVER'])]
            sells = group[group['action'].isin(['SELL', 'SHORT'])]
            
            symbol_stats[symbol] = {
                'total_trades': len(group),
                'buys': len(buys),
                'sells': len(sells),
                'total_volume': group['quantity'].sum(),
                'avg_trade_size': group['quantity'].mean()
            }
        
        # 交易频率
        daily_trades = transactions.resample('D', on='timestamp').size()
        
        return {
            'total_transactions': len(transactions),
            'transaction_by_symbol': symbol_stats,
            'avg_daily_trades': daily_trades.mean() if not daily_trades.empty else 0,
            'max_daily_trades': daily_trades.max() if not daily_trades.empty else 0,
            'first_transaction': str(transactions['timestamp'].min()),
            'last_transaction': str(transactions['timestamp'].max())
        }
    
    def _analyze_risk(self, equity_curve: pd.DataFrame) -> Dict:
        """
        风险分析
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            风险分析结果
        """
        returns = equity_curve['total_equity'].pct_change().dropna()
        
        # 计算VaR (Value at Risk)
        var_95 = returns.quantile(0.05) * 100
        var_99 = returns.quantile(0.01) * 100
        
        # 计算CVaR (Conditional VaR)
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
        cvar_99 = returns[returns <= returns.quantile(0.01)].mean() * 100
        
        # 计算最大回撤期间
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        
        max_dd_idx = drawdown.idxmin()
        max_dd_start = drawdown[:max_dd_idx][drawdown[:max_dd_idx] == 0].index[-1] if len(drawdown[:max_dd_idx][drawdown[:max_dd_idx] == 0]) > 0 else drawdown.index[0]
        
        # 恢复期间
        recovery_idx = None
        for i, (idx, value) in enumerate(drawdown[max_dd_idx:].items()):
            if value == 0:
                recovery_idx = idx
                break
        
        recovery_days = (recovery_idx - max_dd_idx).days if recovery_idx else None
        
        return {
            'var_95_pct': round(var_95, 2),
            'var_99_pct': round(var_99, 2),
            'cvar_95_pct': round(cvar_95, 2),
            'cvar_99_pct': round(cvar_99, 2),
            'max_drawdown_details': {
                'value_pct': round(drawdown.min() * 100, 2),
                'start_date': str(max_dd_start),
                'end_date': str(max_dd_idx),
                'duration_days': (max_dd_idx - max_dd_start).days,
                'recovery_days': recovery_days
            },
            'skewness': round(returns.skew(), 3),
            'kurtosis': round(returns.kurtosis(), 3),
            'downside_deviation': round(returns[returns < 0].std() * np.sqrt(252) * 100, 2)
        }
    
    def _analyze_seasonality(self, equity_curve: pd.DataFrame) -> Dict:
        """
        季节性分析
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            季节性分析结果
        """
        returns = equity_curve['total_equity'].pct_change().dropna()
        
        # 月度季节性
        monthly_returns = returns.groupby(returns.index.month).mean() * 100
        
        # 周度季节性
        weekday_returns = returns.groupby(returns.index.weekday).mean() * 100
        
        return {
            'monthly_seasonality': {
                f'月份{i}': round(ret, 2) 
                for i, ret in monthly_returns.items()
            },
            'weekday_seasonality': {
                f'星期{d}': round(ret, 2) 
                for d, ret in weekday_returns.items()
            }
        }
    
    def plot_equity_curve(self, 
                         result: Optional[Dict] = None,
                         benchmark: Optional[pd.Series] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制权益曲线
        
        Args:
            result: 回测结果
            benchmark: 基准收益率（可选）
            figsize: 图表大小
            
        Returns:
            matplotlib Figure对象
        """
        if result is not None:
            self.set_result(result)
        
        if self.results is None:
            raise ValueError("请先设置回测结果")
        
        equity_curve = self.results['equity_curve']
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 绘制累计收益率
        returns = equity_curve['total_equity'].pct_change().dropna()
        cum_returns = (1 + returns).cumprod()
        
        ax1.plot(cum_returns.index, cum_returns, 'b-', label='策略累计收益')
        ax1.set_ylabel('累计收益率', fontsize=12)
        ax1.set_title('策略累计收益率曲线', fontsize=14)
        
        # 添加基准
        if benchmark is not None:
            # 对齐基准数据
            aligned_benchmark = benchmark.reindex(cum_returns.index).fillna(method='ffill').dropna()
            if not aligned_benchmark.empty:
                bench_cum_returns = (1 + aligned_benchmark).cumprod()
                ax1.plot(bench_cum_returns.index, bench_cum_returns, 'g--', label='基准累计收益')
        
        # 添加网格和图例
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 格式化y轴为百分比
        def to_percent(x, _):
            return f'{(x-1)*100:.0f}%'
        
        ax1.yaxis.set_major_formatter(FuncFormatter(to_percent))
        
        # 添加第二个y轴显示总资产
        ax2 = ax1.twinx()
        ax2.plot(equity_curve.index, equity_curve['total_equity'], 'r-', alpha=0.3)
        ax2.set_ylabel('总资产', fontsize=12)
        
        # 格式化x轴日期
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, 
                     result: Optional[Dict] = None,
                     figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制回撤曲线
        
        Args:
            result: 回测结果
            figsize: 图表大小
            
        Returns:
            matplotlib Figure对象
        """
        if result is not None:
            self.set_result(result)
        
        if self.results is None:
            raise ValueError("请先设置回测结果")
        
        equity_curve = self.results['equity_curve']
        
        # 计算回撤
        returns = equity_curve['total_equity'].pct_change().dropna()
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制回撤曲线
        ax.fill_between(drawdown.index, drawdown * 100, 0, 
                       where=(drawdown < 0), color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown * 100, 'r-', linewidth=1)
        
        # 添加网格和标签
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('回撤百分比 (%)', fontsize=12)
        ax.set_title('策略回撤曲线', fontsize=14)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        
        # 高亮最大回撤
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.annotate(f'最大回撤: {max_dd*100:.2f}%\n{max_dd_date.date()}',
                   xy=(max_dd_date, max_dd * 100),
                   xytext=(max_dd_date, (max_dd * 100) - 10),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   fontsize=10,
                   ha='center')
        
        plt.tight_layout()
        return fig
    
    def plot_performance_summary(self, 
                                result: Optional[Dict] = None,
                                figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        绘制绩效摘要图表
        
        Args:
            result: 回测结果
            figsize: 图表大小
            
        Returns:
            matplotlib Figure对象
        """
        if result is not None:
            self.set_result(result)
        
        if self.results is None:
            raise ValueError("请先设置回测结果")
        
        metrics = self.results.get('metrics', {})
        equity_curve = self.results['equity_curve']
        
        # 创建子图
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2)
        
        # 1. 权益曲线
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_curve.index, equity_curve['total_equity'], 'b-')
        ax1.set_title('总资产变化', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate(rotation=45, ha='center')
        
        # 2. 月度收益热力图
        ax2 = fig.add_subplot(gs[1, 0])
        returns = equity_curve['total_equity'].pct_change().dropna()
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).mean() * 100
        
        # 重塑为透视表
        monthly_returns_unstacked = monthly_returns.unstack()
        if not monthly_returns_unstacked.empty:
            sns.heatmap(monthly_returns_unstacked, annot=True, fmt='.2f', cmap='RdYlGn', 
                       center=0, ax=ax2, cbar_kws={'label': '收益率 (%)'})
            ax2.set_title('月度平均收益率 (%)', fontsize=12)
            ax2.set_xlabel('月份')
            ax2.set_ylabel('年份')
        else:
            ax2.text(0.5, 0.5, '无足够数据', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. 绩效指标条形图
        ax3 = fig.add_subplot(gs[1, 1])
        metrics_to_plot = {
            '年化收益率': metrics.get('annual_return_pct', 0),
            '年化波动率': metrics.get('annual_volatility_pct', 0),
            '夏普比率': metrics.get('sharpe_ratio', 0),
            '最大回撤': metrics.get('max_drawdown_pct', 0)
        }
        
        colors = ['green' if val > 0 else 'red' for val in metrics_to_plot.values()]
        bars = ax3.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=colors)
        ax3.set_title('关键绩效指标', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # 4. 收益率分布直方图
        ax4 = fig.add_subplot(gs[2, 0])
        if not returns.empty:
            ax4.hist(returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax4.set_title('日收益率分布', fontsize=12)
            ax4.set_xlabel('收益率 (%)')
            ax4.set_ylabel('频次')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '无足够数据', ha='center', va='center', transform=ax4.transAxes)
        
        # 5. 胜率饼图
        ax5 = fig.add_subplot(gs[2, 1])
        if not returns.empty:
            win_count = (returns > 0).sum()
            loss_count = (returns < 0).sum()
            ax5.pie([win_count, loss_count], labels=['盈利', '亏损'], autopct='%1.1f%%',
                   startangle=90, colors=['green', 'red'])
            ax5.set_title('交易胜率', fontsize=12)
        else:
            ax5.text(0.5, 0.5, '无足够数据', ha='center', va='center', transform=ax5.transAxes)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, 
                       result: Optional[Dict] = None,
                       output_dir: str = './reports',
                       filename: Optional[str] = None,
                       include_plots: bool = True) -> str:
        """
        生成回测报告
        
        Args:
            result: 回测结果
            output_dir: 输出目录
            filename: 文件名
            include_plots: 是否包含图表
            
        Returns:
            报告文件路径
        """
        if result is not None:
            self.set_result(result)
        
        if self.results is None:
            raise ValueError("请先设置回测结果")
        
        # 确保目录存在
        ensure_directory(output_dir)
        
        # 生成文件名
        if filename is None:
            strategy_name = self.results.get('strategy_name', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_report_{strategy_name}_{timestamp}.html"
        
        report_path = os.path.join(output_dir, filename)
        
        # 分析结果
        analysis = self.analyze()
        
        # 生成HTML报告
        html_content = self._generate_html_report(analysis, include_plots)
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"回测报告已生成: {report_path}")
        
        return report_path
    
    def _generate_html_report(self, analysis: Dict[str, Any], include_plots: bool = True) -> str:
        """
        生成HTML报告内容
        
        Args:
            analysis: 分析结果
            include_plots: 是否包含图表
            
        Returns:
            HTML内容
        """
        # 这里简单生成HTML，实际项目中可以使用模板引擎
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>回测报告 - {analysis['strategy_name']}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                    margin-top: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metric-card {{
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #0066cc;
                }}
                .metric-label {{
                    color: #666;
                    font-size: 14px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .highlight {{
                    background-color: #fffde7;
                    padding: 15px;
                    border-left: 4px solid #ffd600;
                    margin: 20px 0;
                }}
                .plot-container {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .positive {{
                    color: #4caf50;
                }}
                .negative {{
                    color: #f44336;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>回测报告</h1>
                
                <div class="highlight">
                    <h3>策略概览</h3>
                    <p>策略名称: {analysis['strategy_name']}</p>
                    <p>初始资金: {analysis['initial_capital']:,.2f}</p>
                    <p>最终资产: {analysis['final_equity']:,.2f}</p>
                    <p>总收益率: <span class="{'positive' if analysis['final_equity'] > analysis['initial_capital'] else 'negative'}">
                        {((analysis['final_equity'] / analysis['initial_capital'] - 1) * 100):.2f}%
                    </span></p>
                </div>
                
                <h2>关键绩效指标</h2>
                <div class="grid">
        """
        
        # 添加关键绩效指标卡片
        metrics = analysis['metrics']
        key_metrics = [
            ('年化收益率', 'annual_return_pct', '%'),
            ('年化波动率', 'annual_volatility_pct', '%'),
            ('夏普比率', 'sharpe_ratio', ''),
            ('索提诺比率', 'sortino_ratio', ''),
            ('最大回撤', 'max_drawdown_pct', '%'),
            ('卡玛比率', 'calmar_ratio', ''),
            ('胜率', 'win_rate_pct', '%'),
            ('盈亏比', 'profit_factor', '')
        ]
        
        for label, key, unit in key_metrics:
            value = metrics.get(key, 0)
            color = 'positive' if key not in ['annual_volatility_pct', 'max_drawdown_pct'] or value < 0 else 'negative'
            if key in ['annual_volatility_pct', 'max_drawdown_pct']:
                color = 'negative' if value > 0 else 'positive'
            
            html += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color}">{value:.2f}{unit}</div>
                </div>
            """
        
        # 绩效指标表格
        html += f"""
                </div>
                
                <h2>详细绩效指标</h2>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>数值</th>
                        <th>说明</th>
                    </tr>
        """
        
        metric_descriptions = {
            'total_return_pct': '总收益率',
            'annual_return_pct': '年化收益率',
            'annual_volatility_pct': '年化波动率',
            'sharpe_ratio': '夏普比率 (超额收益/波动率)',
            'sortino_ratio': '索提诺比率 (超额收益/下行风险)',
            'max_drawdown_pct': '最大回撤',
            'calmar_ratio': '卡玛比率 (年化收益/最大回撤)',
            'win_rate_pct': '胜率',
            'profit_factor': '盈亏比',
            'total_trades': '总交易次数',
            'trading_days': '交易日数'
        }
        
        for key, desc in metric_descriptions.items():
            value = metrics.get(key, 0)
            unit = '%' if key.endswith('pct') else ''
            html += f"""
                    <tr>
                        <td>{desc}</td>
                        <td>{value:.2f}{unit}</td>
                        <td></td>
                    </tr>
            """
        
        # 风险分析
        risk_analysis = analysis.get('risk_analysis', {})
        html += f"""
                </table>
                
                <h2>风险分析</h2>
                <table>
                    <tr>
                        <th>风险指标</th>
                        <th>数值</th>
                        <th>说明</th>
                    </tr>
                    <tr>
                        <td>VaR (95%)</td>
                        <td>{risk_analysis.get('var_95_pct', 0):.2f}%</td>
                        <td>95%置信度下的单日最大可能损失</td>
                    </tr>
                    <tr>
                        <td>VaR (99%)</td>
                        <td>{risk_analysis.get('var_99_pct', 0):.2f}%</td>
                        <td>99%置信度下的单日最大可能损失</td>
                    </tr>
                    <tr>
                        <td>CVaR (95%)</td>
                        <td>{risk_analysis.get('cvar_95_pct', 0):.2f}%</td>
                        <td>95%置信度下超过VaR的平均损失</td>
                    </tr>
                    <tr>
                        <td>CVaR (99%)</td>
                        <td>{risk_analysis.get('cvar_99_pct', 0):.2f}%</td>
                        <td>99%置信度下超过VaR的平均损失</td>
                    </tr>
        """
        
        # 最大回撤详情
        max_dd = risk_analysis.get('max_drawdown_details', {})
        html += f"""
                    <tr>
                        <td>最大回撤</td>
                        <td>{max_dd.get('value_pct', 0):.2f}%</td>
                        <td>从{max_dd.get('start_date', 'N/A')}到{max_dd.get('end_date', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>最大回撤持续时间</td>
                        <td>{max_dd.get('duration_days', 0)}天</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>恢复时间</td>
                        <td>{max_dd.get('recovery_days', 'N/A')}天</td>
                        <td>从最大回撤点恢复到新高所需时间</td>
                    </tr>
                </table>
        """
        
        # 季节性分析
        seasonal = analysis.get('seasonal_analysis', {})
        html += f"""
                <h2>季节性分析</h2>
                <h3>月度表现</h3>
                <table>
                    <tr>
                        <th>月份</th>
                        <th>平均收益率 (%)</th>
                    </tr>
        """
        
        monthly = seasonal.get('monthly_seasonality', {})
        for month, ret in sorted(monthly.items()):
            color = 'positive' if ret > 0 else 'negative'
            html += f"""
                    <tr>
                        <td>{month}</td>
                        <td class="{color}">{ret:.2f}%</td>
                    </tr>
            """
        
        html += f"""
                </table>
        """
        
        # 报告结尾
        html += f"""
                <div class="highlight">
                    <h3>结论与建议</h3>
                    <p>本报告展示了{analysis['strategy_name']}策略的回测结果。</p>
                    <p>• 年化收益率: {metrics.get('annual_return_pct', 0):.2f}%</p>
                    <p>• 风险调整后收益 (夏普比率): {metrics.get('sharpe_ratio', 0):.2f}</p>
                    <p>• 最大回撤: {metrics.get('max_drawdown_pct', 0):.2f}%</p>
                    <p>建议根据实际交易环境和风险偏好调整策略参数，并进行充分的实盘测试。</p>
                </div>
                
                <p><small>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
        </body>
        </html>
        """
        
        return html


def compare_strategies(results: List[Dict], 
                       metrics: List[str] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    比较多个策略的绩效
    
    Args:
        results: 回测结果列表
        metrics: 要比较的指标列表
        figsize: 图表大小
        
    Returns:
        matplotlib Figure对象
    """
    if metrics is None:
        metrics = ['annual_return_pct', 'sharpe_ratio', 'max_drawdown_pct']
    
    # 提取数据
    strategy_names = [result.get('strategy_name', f'Strategy {i}') for i, result in enumerate(results)]
    metrics_data = {}
    
    for metric in metrics:
        metrics_data[metric] = [result.get('metrics', {}).get(metric, 0) for result in results]
    
    # 创建子图
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # 指标显示名称映射
    metric_labels = {
        'annual_return_pct': '年化收益率 (%)',
        'sharpe_ratio': '夏普比率',
        'max_drawdown_pct': '最大回撤 (%)',
        'sortino_ratio': '索提诺比率',
        'calmar_ratio': '卡玛比率',
        'win_rate_pct': '胜率 (%)',
        'profit_factor': '盈亏比'
    }
    
    # 绘制每个指标
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax = axes[i]
        
        # 确定颜色
        if metric in ['max_drawdown_pct', 'annual_volatility_pct']:
            # 对于这些指标，值越小越好
            colors = ['green' if v < min(values) * 1.1 else 'red' if v > max(values) * 0.9 else 'blue' for v in values]
        else:
            # 对于其他指标，值越大越好
            colors = ['green' if v > max(values) * 0.9 else 'red' if v < min(values) * 1.1 else 'blue' for v in values]
        
        # 绘制条形图
        bars = ax.bar(strategy_names, values, color=colors)
        ax.set_title(metric_labels.get(metric, metric), fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def run_backtest_from_config(config_path: str) -> Dict:
    """
    从配置文件运行回测
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        回测结果
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 这里需要根据配置创建策略和加载数据
    # 实际项目中需要实现具体的加载逻辑
    logger.warning("此函数需要根据实际项目实现数据加载和策略创建逻辑")
    
    return {}
