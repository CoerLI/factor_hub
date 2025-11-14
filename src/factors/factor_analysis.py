import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import os

from src.utils.helpers import setup_logger, DEFAULT_LOGGER, plot_to_html
from src.utils.cache_manager import CacheManager

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


class FactorAnalyzer:
    """
    因子分析器类
    
    提供因子有效性分析、统计检验、相关性分析等功能
    """
    
    def __init__(self, 
                 cache_manager: Optional[CacheManager] = None,
                 cache_dir: str = './cache/factor_analysis'):
        """
        初始化因子分析器
        
        Args:
            cache_manager: 缓存管理器实例
            cache_dir: 缓存目录
        """
        self.cache_manager = cache_manager or CacheManager(cache_dir=cache_dir)
        self.results = {}
        
        # 设置可视化样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_factor_performance(self, 
                                   factor_values: pd.Series,
                                   returns: pd.Series,
                                   factor_name: str = 'factor',
                                   lag: int = 1,
                                   group_by: Optional[str] = None,
                                   **kwargs) -> Dict:
        """
        分析单个因子的表现
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            lag: 因子滞后阶数
            group_by: 分组字段名称
            
        Returns:
            因子表现分析结果字典
        """
        # 创建缓存键
        cache_key = f"factor_performance_{factor_name}_{lag}"
        if group_by:
            cache_key += f"_groupby_{group_by}"
        
        # 尝试从缓存获取结果
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            logger.info(f"从缓存加载因子表现分析结果: {factor_name}")
            return cached_result
        
        # 对齐因子值和收益率
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'returns': returns
        }).dropna()
        
        # 滞后因子值
        aligned_data['factor_lagged'] = aligned_data['factor'].shift(lag)
        aligned_data.dropna(inplace=True)
        
        # 基本统计信息
        stats_result = {
            'factor_name': factor_name,
            'sample_size': len(aligned_data),
            'factor_mean': aligned_data['factor_lagged'].mean(),
            'factor_std': aligned_data['factor_lagged'].std(),
            'returns_mean': aligned_data['returns'].mean(),
            'returns_std': aligned_data['returns'].std(),
        }
        
        # 计算因子与未来收益率的相关性
        corr = aligned_data[['factor_lagged', 'returns']].corr().iloc[0, 1]
        stats_result['correlation'] = corr
        
        # 计算IC (信息系数) -  Spearman秩相关系数
        ic = stats.spearmanr(aligned_data['factor_lagged'], aligned_data['returns'])[0]
        stats_result['ic'] = ic
        stats_result['ic_t_stat'], stats_result['ic_p_value'] = stats.ttest_1samp(
            [ic], 0, nan_policy='omit')
        
        # 线性回归分析
        lr_model = LinearRegression()
        X = aligned_data[['factor_lagged']].values
        y = aligned_data['returns'].values
        lr_model.fit(X, y)
        
        stats_result['beta'] = lr_model.coef_[0]
        stats_result['alpha'] = lr_model.intercept_
        stats_result['r_squared'] = r2_score(y, lr_model.predict(X))
        
        # 分组分析（如果指定了分组字段）
        if group_by and group_by in aligned_data.columns:
            group_analysis = self._analyze_by_group(
                aligned_data, 'factor_lagged', 'returns', group_by)
            stats_result['group_analysis'] = group_analysis
        else:
            # 按因子分位数分组
            quintile_analysis = self._analyze_by_quantiles(
                aligned_data, 'factor_lagged', 'returns', n_quantiles=5)
            stats_result['quintile_analysis'] = quintile_analysis
        
        # 保存结果
        self.results[factor_name] = stats_result
        
        # 缓存结果
        self.cache_manager.set(cache_key, stats_result)
        
        logger.info(f"完成因子表现分析: {factor_name}, IC={ic:.4f}, Correlation={corr:.4f}")
        
        return stats_result
    
    def _analyze_by_quantiles(self, 
                             data: pd.DataFrame,
                             factor_col: str,
                             returns_col: str,
                             n_quantiles: int = 5) -> Dict:
        """
        按因子分位数分组分析
        
        Args:
            data: 数据DataFrame
            factor_col: 因子列名
            returns_col: 收益率列名
            n_quantiles: 分位数数量
            
        Returns:
            分组分析结果
        """
        # 创建分位数分组
        data['quantile'] = pd.qcut(data[factor_col], n_quantiles, labels=False, duplicates='drop') + 1
        
        # 计算每组的平均收益率
        quantile_stats = data.groupby('quantile').agg({
            factor_col: ['mean', 'std', 'count'],
            returns_col: ['mean', 'std', 'count']
        })
        
        # 计算多空组合收益率（最高分组减去最低分组）
        top_group = n_quantiles
        bottom_group = 1
        
        top_returns = quantile_stats[returns_col]['mean'].iloc[top_group - 1]
        bottom_returns = quantile_stats[returns_col]['mean'].iloc[bottom_group - 1]
        long_short_return = top_returns - bottom_returns
        
        # 计算t统计量
        top_returns_std = quantile_stats[returns_col]['std'].iloc[top_group - 1]
        bottom_returns_std = quantile_stats[returns_col]['std'].iloc[bottom_group - 1]
        top_count = quantile_stats[returns_col]['count'].iloc[top_group - 1]
        bottom_count = quantile_stats[returns_col]['count'].iloc[bottom_group - 1]
        
        # 假设独立样本的t检验
        pooled_std = np.sqrt((top_returns_std**2 / top_count) + (bottom_returns_std**2 / bottom_count))
        t_stat = long_short_return / (pooled_std + 1e-10)
        
        return {
            'quantile_stats': quantile_stats.to_dict(),
            'long_short_return': long_short_return,
            't_statistic': t_stat,
            'top_quantile_return': top_returns,
            'bottom_quantile_return': bottom_returns,
            'n_quantiles': n_quantiles
        }
    
    def _analyze_by_group(self, 
                         data: pd.DataFrame,
                         factor_col: str,
                         returns_col: str,
                         group_col: str) -> Dict:
        """
        按指定字段分组分析因子表现
        
        Args:
            data: 数据DataFrame
            factor_col: 因子列名
            returns_col: 收益率列名
            group_col: 分组列名
            
        Returns:
            分组分析结果
        """
        # 按分组计算统计量
        group_stats = data.groupby(group_col).agg({
            factor_col: ['mean', 'std', 'count'],
            returns_col: ['mean', 'std', 'count']
        })
        
        # 计算每组的因子与收益率相关性
        group_correlations = {}
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            if len(group_data) > 1:
                corr = group_data[[factor_col, returns_col]].corr().iloc[0, 1]
                group_correlations[group] = corr
        
        return {
            'group_stats': group_stats.to_dict(),
            'group_correlations': group_correlations
        }
    
    def analyze_multiple_factors(self, 
                               factor_data: pd.DataFrame,
                               returns_data: pd.Series,
                               factor_columns: List[str],
                               lag: int = 1) -> Dict[str, Dict]:
        """
        批量分析多个因子的表现
        
        Args:
            factor_data: 包含多个因子的DataFrame
            returns_data: 收益率序列
            factor_columns: 要分析的因子列名列表
            lag: 因子滞后阶数
            
        Returns:
            各因子的分析结果字典
        """
        results = {}
        
        for factor_col in factor_columns:
            if factor_col in factor_data.columns:
                try:
                    result = self.analyze_factor_performance(
                        factor_values=factor_data[factor_col],
                        returns=returns_data,
                        factor_name=factor_col,
                        lag=lag
                    )
                    results[factor_col] = result
                except Exception as e:
                    logger.error(f"分析因子 {factor_col} 时出错: {e}")
            else:
                logger.warning(f"因子列 {factor_col} 不存在于数据中")
        
        return results
    
    def calculate_rolling_ic(self, 
                           factor_values: pd.Series,
                           returns: pd.Series,
                           window: int = 20,
                           lag: int = 1) -> pd.Series:
        """
        计算滚动信息系数(IC)
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            window: 滚动窗口大小
            lag: 因子滞后阶数
            
        Returns:
            滚动IC序列
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'returns': returns
        }).dropna()
        
        # 滞后因子值
        aligned_data['factor_lagged'] = aligned_data['factor'].shift(lag)
        aligned_data.dropna(inplace=True)
        
        # 计算滚动IC
        rolling_ic = aligned_data['factor_lagged'].rolling(window=window).corr(
            aligned_data['returns'], method='spearman')
        
        return rolling_ic
    
    def calculate_factor_exposure(self, 
                                returns: pd.DataFrame,
                                factor_data: pd.DataFrame,
                                factor_columns: List[str]) -> pd.DataFrame:
        """
        计算因子暴露度（回归系数）
        
        Args:
            returns: 资产收益率矩阵
            factor_data: 因子数据矩阵
            factor_columns: 因子列名列表
            
        Returns:
            因子暴露度矩阵
        """
        # 对齐数据索引
        aligned_index = returns.index.intersection(factor_data.index)
        returns_aligned = returns.loc[aligned_index]
        factor_data_aligned = factor_data.loc[aligned_index, factor_columns].copy()
        
        # 标准化因子
        scaler = StandardScaler()
        factor_data_scaled = pd.DataFrame(
            scaler.fit_transform(factor_data_aligned),
            index=factor_data_aligned.index,
            columns=factor_data_aligned.columns
        )
        
        # 添加常数项
        factor_data_scaled['intercept'] = 1
        
        # 对每个资产计算因子暴露度
        exposures = pd.DataFrame(index=returns_aligned.columns, columns=factor_data_scaled.columns)
        
        for asset in returns_aligned.columns:
            try:
                lr_model = LinearRegression()
                lr_model.fit(factor_data_scaled, returns_aligned[asset])
                exposures.loc[asset] = lr_model.coef_
            except Exception as e:
                logger.error(f"计算资产 {asset} 的因子暴露度时出错: {e}")
        
        return exposures
    
    def calculate_factor_returns(self, 
                               portfolio_returns: pd.DataFrame,
                               factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子收益率（使用横截面回归）
        
        Args:
            portfolio_returns: 投资组合收益率矩阵
            factor_exposures: 因子暴露度矩阵
            
        Returns:
            因子收益率DataFrame
        """
        # 对齐数据
        aligned_assets = portfolio_returns.columns.intersection(factor_exposures.index)
        aligned_dates = portfolio_returns.index
        
        factor_returns = pd.DataFrame(index=aligned_dates, columns=factor_exposures.columns)
        
        # 对每个时间点进行横截面回归
        for date in aligned_dates:
            try:
                # 获取当前时间点的数据
                current_returns = portfolio_returns.loc[date, aligned_assets]
                current_exposures = factor_exposures.loc[aligned_assets]
                
                # 移除NaN值
                mask = ~current_returns.isna() & ~current_exposures.isna().any(axis=1)
                if mask.sum() < len(factor_exposures.columns) + 1:  # 需要足够的样本
                    continue
                
                # 执行回归
                lr_model = LinearRegression()
                lr_model.fit(current_exposures[mask], current_returns[mask])
                
                # 保存因子收益率
                factor_returns.loc[date] = lr_model.coef_
                
            except Exception as e:
                logger.error(f"计算 {date} 的因子收益率时出错: {e}")
        
        return factor_returns
    
    def factor_attribution(self, 
                          portfolio_returns: pd.Series,
                          factor_returns: pd.DataFrame,
                          factor_exposures: pd.Series) -> pd.DataFrame:
        """
        因子归因分析
        
        Args:
            portfolio_returns: 投资组合收益率
            factor_returns: 因子收益率
            factor_exposures: 投资组合因子暴露度
            
        Returns:
            归因结果DataFrame
        """
        # 对齐时间
        aligned_dates = portfolio_returns.index.intersection(factor_returns.index)
        
        # 计算因子贡献
        factor_contributions = pd.DataFrame(index=aligned_dates, columns=factor_returns.columns)
        
        for date in aligned_dates:
            try:
                # 因子贡献 = 因子暴露度 * 因子收益率
                factor_contributions.loc[date] = factor_exposures * factor_returns.loc[date]
            except Exception as e:
                logger.error(f"计算 {date} 的因子贡献时出错: {e}")
        
        # 计算残差
        total_factor_contribution = factor_contributions.sum(axis=1)
        residual = portfolio_returns.loc[aligned_dates] - total_factor_contribution
        
        # 组合结果
        attribution = factor_contributions.copy()
        attribution['total_factor'] = total_factor_contribution
        attribution['residual'] = residual
        attribution['portfolio_return'] = portfolio_returns.loc[aligned_dates]
        
        return attribution
    
    def plot_factor_distribution(self, 
                               factor_values: pd.Series,
                               factor_name: str = 'factor',
                               bins: int = 50,
                               save_path: Optional[str] = None) -> str:
        """
        绘制因子分布图
        
        Args:
            factor_values: 因子值序列
            factor_name: 因子名称
            bins: 直方图分箱数
            save_path: 保存路径
            
        Returns:
            图表的HTML字符串或保存路径
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图
        plt.hist(factor_values.dropna(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加统计信息
        mean_val = factor_values.mean()
        std_val = factor_values.std()
        
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', label=f'Mean ± Std')
        plt.axvline(mean_val - std_val, color='orange', linestyle='--')
        
        plt.title(f'Factor Distribution: {factor_name}')
        plt.xlabel(f'{factor_name} Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            html_str = plot_to_html(plt)
            plt.close()
            return html_str
    
    def plot_factor_returns_by_quantile(self, 
                                      analysis_result: Dict,
                                      factor_name: str = 'factor',
                                      save_path: Optional[str] = None) -> str:
        """
        绘制不同分位数的因子收益率
        
        Args:
            analysis_result: 因子分析结果字典
            factor_name: 因子名称
            save_path: 保存路径
            
        Returns:
            图表的HTML字符串或保存路径
        """
        if 'quintile_analysis' not in analysis_result:
            raise ValueError("分析结果中缺少分位数分析数据")
        
        quintile_data = analysis_result['quintile_analysis']['quantile_stats']
        n_quantiles = analysis_result['quintile_analysis']['n_quantiles']
        
        # 提取各分位数的平均收益率
        returns_mean = []
        for q in range(1, n_quantiles + 1):
            returns_mean.append(quintile_data['returns']['mean'][q])
        
        plt.figure(figsize=(10, 6))
        
        # 绘制柱状图
        bars = plt.bar(range(1, n_quantiles + 1), returns_mean, color='skyblue', edgecolor='black')
        
        # 在柱状图上标注数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.6f}', ha='center', va='bottom')
        
        # 绘制多空组合收益线
        long_short = analysis_result['quintile_analysis']['long_short_return']
        plt.axhline(long_short, color='red', linestyle='--', label=f'Long-Short: {long_short:.6f}')
        
        plt.title(f'Factor Returns by Quantile: {factor_name}')
        plt.xlabel('Quantile')
        plt.ylabel('Average Return')
        plt.xticks(range(1, n_quantiles + 1))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            html_str = plot_to_html(plt)
            plt.close()
            return html_str
    
    def plot_rolling_ic(self, 
                       rolling_ic: pd.Series,
                       factor_name: str = 'factor',
                       window: int = 20,
                       save_path: Optional[str] = None) -> str:
        """
        绘制滚动信息系数(IC)
        
        Args:
            rolling_ic: 滚动IC序列
            factor_name: 因子名称
            window: 滚动窗口大小
            save_path: 保存路径
            
        Returns:
            图表的HTML字符串或保存路径
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制滚动IC
        plt.plot(rolling_ic.index, rolling_ic, label='Rolling IC', color='blue')
        
        # 添加均值线
        mean_ic = rolling_ic.mean()
        plt.axhline(mean_ic, color='red', linestyle='--', label=f'Mean IC: {mean_ic:.4f}')
        
        # 添加零线
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        plt.title(f'Rolling IC for {factor_name} (Window: {window})')
        plt.xlabel('Date')
        plt.ylabel('Information Coefficient')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 设置合理的y轴范围
        y_min = min(rolling_ic.min() * 1.2, -0.5)
        y_max = max(rolling_ic.max() * 1.2, 0.5)
        plt.ylim(y_min, y_max)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            html_str = plot_to_html(plt)
            plt.close()
            return html_str
    
    def plot_factor_correlation_heatmap(self, 
                                      factor_data: pd.DataFrame,
                                      factor_columns: List[str],
                                      method: str = 'pearson',
                                      save_path: Optional[str] = None) -> str:
        """
        绘制因子相关性热力图
        
        Args:
            factor_data: 因子数据
            factor_columns: 因子列名列表
            method: 相关性计算方法 ('pearson', 'kendall', 'spearman')
            save_path: 保存路径
            
        Returns:
            图表的HTML字符串或保存路径
        """
        # 选择要分析的因子列
        corr_data = factor_data[factor_columns].copy()
        
        # 计算相关系数矩阵
        corr_matrix = corr_data.corr(method=method)
        
        plt.figure(figsize=(12, 10))
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=.5,
                   cbar_kws={"shrink": .8})
        
        plt.title(f'Factor Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            html_str = plot_to_html(plt)
            plt.close()
            return html_str
    
    def generate_factor_report(self, 
                             analysis_results: Dict[str, Dict],
                             factor_data: pd.DataFrame,
                             returns_data: pd.Series,
                             report_dir: str = './reports/factor_analysis') -> Dict[str, str]:
        """
        生成因子分析报告
        
        Args:
            analysis_results: 因子分析结果字典
            factor_data: 因子数据
            returns_data: 收益率数据
            report_dir: 报告保存目录
            
        Returns:
            报告文件路径字典
        """
        # 创建报告目录
        os.makedirs(report_dir, exist_ok=True)
        
        report_files = {}
        
        # 生成综合统计表格
        stats_table = []
        for factor_name, result in analysis_results.items():
            stats_table.append({
                'Factor': factor_name,
                'Correlation': result.get('correlation', 0),
                'IC': result.get('ic', 0),
                'IC t-stat': result.get('ic_t_stat', 0),
                'IC p-value': result.get('ic_p_value', 1),
                'Beta': result.get('beta', 0),
                'R-squared': result.get('r_squared', 0)
            })
        
        stats_df = pd.DataFrame(stats_table)
        stats_csv_path = os.path.join(report_dir, 'factor_stats.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        report_files['stats_csv'] = stats_csv_path
        
        # 生成各因子的可视化图表
        for factor_name, result in analysis_results.items():
            # 因子分布图
            dist_path = os.path.join(report_dir, f'{factor_name}_distribution.png')
            self.plot_factor_distribution(
                factor_values=factor_data[factor_name],
                factor_name=factor_name,
                save_path=dist_path
            )
            report_files[f'{factor_name}_distribution'] = dist_path
            
            # 分位数收益图
            if 'quintile_analysis' in result:
                qreturn_path = os.path.join(report_dir, f'{factor_name}_quantile_returns.png')
                self.plot_factor_returns_by_quantile(
                    analysis_result=result,
                    factor_name=factor_name,
                    save_path=qreturn_path
                )
                report_files[f'{factor_name}_quantile_returns'] = qreturn_path
            
            # 滚动IC图
            rolling_ic = self.calculate_rolling_ic(
                factor_values=factor_data[factor_name],
                returns=returns_data,
                window=20
            )
            ic_path = os.path.join(report_dir, f'{factor_name}_rolling_ic.png')
            self.plot_rolling_ic(
                rolling_ic=rolling_ic,
                factor_name=factor_name,
                save_path=ic_path
            )
            report_files[f'{factor_name}_rolling_ic'] = ic_path
        
        # 生成因子相关性热力图
        factor_columns = list(analysis_results.keys())
        corr_path = os.path.join(report_dir, 'factor_correlation_heatmap.png')
        self.plot_factor_correlation_heatmap(
            factor_data=factor_data,
            factor_columns=factor_columns,
            save_path=corr_path
        )
        report_files['correlation_heatmap'] = corr_path
        
        logger.info(f"因子分析报告已生成，保存在: {report_dir}")
        
        return report_files


class FactorValidator:
    """
    因子验证器类
    
    提供因子有效性验证、稳健性测试和显著性检验等功能
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        初始化因子验证器
        
        Args:
            cache_manager: 缓存管理器实例
        """
        self.cache_manager = cache_manager
        self.analyzer = FactorAnalyzer(cache_manager=cache_manager)
        self.validation_results = {}
    
    def validate_factor_significance(self, 
                                   factor_values: pd.Series,
                                   returns: pd.Series,
                                   factor_name: str = 'factor',
                                   lag: int = 1,
                                   test_type: str = 'ttest') -> Dict:
        """
        验证因子的统计显著性
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            lag: 因子滞后阶数
            test_type: 测试类型，支持'ttest'(t检验), 'wilcoxon'(Wilcoxon符号秩检验)
            
        Returns:
            显著性检验结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'returns': returns
        }).dropna()
        
        # 滞后因子值
        aligned_data['factor_lagged'] = aligned_data['factor'].shift(lag)
        aligned_data.dropna(inplace=True)
        
        # 计算因子分组
        aligned_data['positive_factor'] = aligned_data['factor_lagged'] > aligned_data['factor_lagged'].median()
        
        # 分开正负因子组的收益率
        pos_returns = aligned_data[aligned_data['positive_factor']]['returns']
        neg_returns = aligned_data[~aligned_data['positive_factor']]['returns']
        
        # 执行显著性测试
        if test_type == 'ttest':
            # 双样本t检验
            t_stat, p_value = stats.ttest_ind(pos_returns, neg_returns, equal_var=False)
        elif test_type == 'wilcoxon':
            # Wilcoxon秩和检验
            try:
                t_stat, p_value = stats.ranksums(pos_returns, neg_returns)
            except ValueError:
                # 如果数据有问题，返回NaN
                t_stat, p_value = np.nan, np.nan
        else:
            raise ValueError(f"不支持的测试类型: {test_type}")
        
        # 计算效应量
        mean_diff = pos_returns.mean() - neg_returns.mean()
        pooled_std = np.sqrt((pos_returns.var() + neg_returns.var()) / 2)
        effect_size = mean_diff / (pooled_std + 1e-10) if pooled_std != 0 else np.nan
        
        result = {
            'factor_name': factor_name,
            'test_type': test_type,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'pos_group_mean_return': pos_returns.mean(),
            'neg_group_mean_return': neg_returns.mean(),
            'mean_difference': mean_diff,
            'effect_size': effect_size,
            'pos_group_count': len(pos_returns),
            'neg_group_count': len(neg_returns)
        }
        
        self.validation_results[factor_name] = result
        
        return result
    
    def test_factor_stability(self, 
                            factor_values: pd.Series,
                            returns: pd.Series,
                            factor_name: str = 'factor',
                            time_windows: List[int] = [63, 126, 252],
                            lag: int = 1) -> Dict:
        """
        测试因子在不同时间窗口的稳定性
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            time_windows: 时间窗口大小列表（交易日）
            lag: 因子滞后阶数
            
        Returns:
            因子稳定性测试结果
        """
        stability_results = {}
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'returns': returns
        }).dropna()
        
        aligned_data['factor_lagged'] = aligned_data['factor'].shift(lag)
        aligned_data.dropna(inplace=True)
        
        # 计算全样本IC
        full_sample_ic = stats.spearmanr(aligned_data['factor_lagged'], aligned_data['returns'])[0]
        
        # 对每个时间窗口计算滚动IC
        rolling_ics = {}
        for window in time_windows:
            rolling_ic = self.analyzer.calculate_rolling_ic(
                factor_values=factor_values,
                returns=returns,
                window=window,
                lag=lag
            )
            
            rolling_ics[window] = {
                'mean': rolling_ic.mean(),
                'std': rolling_ic.std(),
                'positive_count': (rolling_ic > 0).sum(),
                'total_count': rolling_ic.count(),
                'positive_ratio': (rolling_ic > 0).sum() / rolling_ic.count() if rolling_ic.count() > 0 else 0,
                'autocorr_1': rolling_ic.autocorr(lag=1),
                'autocorr_5': rolling_ic.autocorr(lag=5)
            }
        
        # 计算IC的偏度和峰度
        all_rolling_ics = pd.Series()
        for window, data in rolling_ics.items():
            if window in locals():
                all_rolling_ics = pd.concat([all_rolling_ics, eval(f'rolling_ic')])
        
        if len(all_rolling_ics) > 0:
            ic_skewness = stats.skew(all_rolling_ics.dropna())
            ic_kurtosis = stats.kurtosis(all_rolling_ics.dropna())
        else:
            ic_skewness = np.nan
            ic_kurtosis = np.nan
        
        result = {
            'factor_name': factor_name,
            'full_sample_ic': full_sample_ic,
            'rolling_ics': rolling_ics,
            'ic_skewness': ic_skewness,
            'ic_kurtosis': ic_kurtosis,
            'time_windows': time_windows
        }
        
        self.validation_results[f"{factor_name}_stability"] = result
        
        return result
    
    def test_factor_robustness(self, 
                             factor_values: pd.Series,
                             returns: pd.Series,
                             factor_name: str = 'factor',
                             variations: List[Dict] = None) -> Dict:
        """
        测试因子在不同参数设置下的稳健性
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            variations: 参数变化列表
            
        Returns:
            因子稳健性测试结果
        """
        if variations is None:
            # 默认的参数变化
            variations = [
                {'lag': 1, 'quantile_method': 'qcut'},
                {'lag': 2, 'quantile_method': 'qcut'},
                {'lag': 5, 'quantile_method': 'qcut'},
                {'lag': 1, 'quantile_method': 'cut'},
            ]
        
        robustness_results = {}
        
        for i, variation in enumerate(variations):
            # 应用参数变化
            lag = variation.get('lag', 1)
            quantile_method = variation.get('quantile_method', 'qcut')
            
            # 对齐数据
            aligned_data = pd.DataFrame({
                'factor': factor_values,
                'returns': returns
            }).dropna()
            
            aligned_data['factor_lagged'] = aligned_data['factor'].shift(lag)
            aligned_data.dropna(inplace=True)
            
            # 计算IC
            try:
                ic = stats.spearmanr(aligned_data['factor_lagged'], aligned_data['returns'])[0]
            except:
                ic = np.nan
            
            # 计算分位数收益（使用不同的分位数方法）
            try:
                if quantile_method == 'qcut':
                    aligned_data['quantile'] = pd.qcut(
                        aligned_data['factor_lagged'], 5, labels=False, duplicates='drop') + 1
                else:  # cut
                    aligned_data['quantile'] = pd.cut(
                        aligned_data['factor_lagged'], 5, labels=False, duplicates='drop') + 1
                
                # 过滤掉无效的分位数
                valid_quantiles = aligned_data['quantile'].dropna().unique()
                if len(valid_quantiles) >= 2:
                    # 取最高和最低分位数
                    top_q = max(valid_quantiles)
                    bottom_q = min(valid_quantiles)
                    
                    top_return = aligned_data[aligned_data['quantile'] == top_q]['returns'].mean()
                    bottom_return = aligned_data[aligned_data['quantile'] == bottom_q]['returns'].mean()
                    long_short = top_return - bottom_return
                else:
                    top_return = bottom_return = long_short = np.nan
                    
            except:
                top_return = bottom_return = long_short = np.nan
            
            robustness_results[i] = {
                'parameters': variation,
                'ic': ic,
                'top_quantile_return': top_return,
                'bottom_quantile_return': bottom_return,
                'long_short_return': long_short
            }
        
        # 计算稳健性统计量
        ics = [r['ic'] for r in robustness_results.values() if not np.isnan(r['ic'])]
        long_shorts = [r['long_short_return'] for r in robustness_results.values() if not np.isnan(r['long_short_return'])]
        
        result = {
            'factor_name': factor_name,
            'variations': robustness_results,
            'avg_ic': np.mean(ics) if ics else np.nan,
            'std_ic': np.std(ics) if ics else np.nan,
            'avg_long_short': np.mean(long_shorts) if long_shorts else np.nan,
            'std_long_short': np.std(long_shorts) if long_shorts else np.nan,
            'consistency': len([ic for ic in ics if ic > 0]) / len(ics) if ics else 0
        }
        
        self.validation_results[f"{factor_name}_robustness"] = result
        
        return result
    
    def validate_factor_predictive_power(self, 
                                       factor_values: pd.Series,
                                       returns: pd.Series,
                                       factor_name: str = 'factor',
                                       lookback: int = 252,
                                       horizon: int = 1,
                                       train_ratio: float = 0.7) -> Dict:
        """
        验证因子的预测能力
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            lookback: 回看窗口大小
            horizon: 预测窗口大小
            train_ratio: 训练集比例
            
        Returns:
            预测能力验证结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'returns': returns
        }).dropna()
        
        # 准备预测目标（未来收益率）
        aligned_data['future_returns'] = aligned_data['returns'].shift(-horizon)
        aligned_data.dropna(inplace=True)
        
        # 划分训练集和测试集
        train_size = int(len(aligned_data) * train_ratio)
        train_data = aligned_data.iloc[:train_size]
        test_data = aligned_data.iloc[train_size:]
        
        # 使用线性回归进行预测
        lr_model = LinearRegression()
        
        # 训练模型
        X_train = train_data[['factor']]
        y_train = train_data['future_returns']
        lr_model.fit(X_train, y_train)
        
        # 测试模型
        X_test = test_data[['factor']]
        y_test = test_data['future_returns']
        
        y_pred_train = lr_model.predict(X_train)
        y_pred_test = lr_model.predict(X_test)
        
        # 计算评估指标
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        train_rmse = np.sqrt(np.mean((y_pred_train - y_train) ** 2))
        test_rmse = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
        
        # 计算预测方向准确率
        train_direction_accuracy = np.mean(np.sign(y_pred_train) == np.sign(y_train))
        test_direction_accuracy = np.mean(np.sign(y_pred_test) == np.sign(y_test))
        
        # 使用逻辑回归进行分类预测（收益率方向）
        y_train_class = (y_train > 0).astype(int)
        y_test_class = (y_test > 0).astype(int)
        
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train_class)
        
        y_pred_class_train = log_model.predict(X_train)
        y_pred_class_test = log_model.predict(X_test)
        
        class_train_accuracy = accuracy_score(y_train_class, y_pred_class_train)
        class_test_accuracy = accuracy_score(y_test_class, y_pred_class_test)
        
        result = {
            'factor_name': factor_name,
            'lookback': lookback,
            'horizon': horizon,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'regression': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_direction_accuracy': train_direction_accuracy,
                'test_direction_accuracy': test_direction_accuracy
            },
            'classification': {
                'train_accuracy': class_train_accuracy,
                'test_accuracy': class_test_accuracy
            },
            'coefficients': lr_model.coef_[0][0],
            'intercept': lr_model.intercept_[0]
        }
        
        self.validation_results[f"{factor_name}_predictive"] = result
        
        return result
    
    def run_comprehensive_validation(self, 
                                   factor_values: pd.Series,
                                   returns: pd.Series,
                                   factor_name: str = 'factor') -> Dict[str, Dict]:
        """
        运行综合验证
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            
        Returns:
            综合验证结果字典
        """
        results = {}
        
        # 基本表现分析
        results['performance'] = self.analyzer.analyze_factor_performance(
            factor_values=factor_values,
            returns=returns,
            factor_name=factor_name
        )
        
        # 统计显著性测试
        results['significance'] = self.validate_factor_significance(
            factor_values=factor_values,
            returns=returns,
            factor_name=factor_name
        )
        
        # 稳定性测试
        results['stability'] = self.test_factor_stability(
            factor_values=factor_values,
            returns=returns,
            factor_name=factor_name
        )
        
        # 稳健性测试
        results['robustness'] = self.test_factor_robustness(
            factor_values=factor_values,
            returns=returns,
            factor_name=factor_name
        )
        
        # 预测能力验证
        results['predictive'] = self.validate_factor_predictive_power(
            factor_values=factor_values,
            returns=returns,
            factor_name=factor_name
        )
        
        self.validation_results[f"{factor_name}_comprehensive"] = results
        
        return results
    
    def generate_validation_report(self, 
                                 validation_results: Dict[str, Dict],
                                 report_dir: str = './reports/factor_validation') -> Dict[str, str]:
        """
        生成验证报告
        
        Args:
            validation_results: 验证结果字典
            report_dir: 报告保存目录
            
        Returns:
            报告文件路径字典
        """
        # 创建报告目录
        os.makedirs(report_dir, exist_ok=True)
        
        report_files = {}
        
        # 生成综合验证结果表格
        validation_table = []
        
        for factor_name, results in validation_results.items():
            if 'comprehensive' in factor_name:
                base_name = factor_name.replace('_comprehensive', '')
                
                row = {
                    'Factor': base_name,
                    'IC': results.get('performance', {}).get('ic', np.nan),
                    'R-squared': results.get('performance', {}).get('r_squared', np.nan),
                    'Significant': results.get('significance', {}).get('significant', False),
                    'P-value': results.get('significance', {}).get('p_value', 1),
                    'Consistency': results.get('robustness', {}).get('consistency', 0),
                    'Test R-squared': results.get('predictive', {}).get('regression', {}).get('test_r2', np.nan),
                    'Test Direction Acc': results.get('predictive', {}).get('regression', {}).get('test_direction_accuracy', 0)
                }
                
                validation_table.append(row)
        
        if validation_table:
            val_df = pd.DataFrame(validation_table)
            val_csv_path = os.path.join(report_dir, 'comprehensive_validation.csv')
            val_df.to_csv(val_csv_path, index=False)
            report_files['comprehensive_validation'] = val_csv_path
        
        # 保存详细验证结果到JSON文件
        import json
        json_path = os.path.join(report_dir, 'validation_results.json')
        
        # 将numpy类型转换为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy(validation_results), f, indent=2, ensure_ascii=False)
        
        report_files['detailed_results'] = json_path
        
        logger.info(f"因子验证报告已生成，保存在: {report_dir}")
        
        return report_files


# 便捷函数
def analyze_factors(factor_data: pd.DataFrame,
                   returns_data: pd.Series,
                   factor_columns: List[str]) -> FactorAnalyzer:
    """
    便捷的因子分析函数
    
    Args:
        factor_data: 因子数据
        returns_data: 收益率数据
        factor_columns: 因子列名列表
        
    Returns:
        因子分析器实例，包含分析结果
    """
    analyzer = FactorAnalyzer()
    analyzer.analyze_multiple_factors(
        factor_data=factor_data,
        returns_data=returns_data,
        factor_columns=factor_columns
    )
    return analyzer


def validate_factors(factor_data: pd.DataFrame,
                     returns_data: pd.Series,
                     factor_columns: List[str]) -> FactorValidator:
    """
    便捷的因子验证函数
    
    Args:
        factor_data: 因子数据
        returns_data: 收益率数据
        factor_columns: 因子列名列表
        
    Returns:
        因子验证器实例，包含验证结果
    """
    validator = FactorValidator()
    
    for factor_col in factor_columns:
        if factor_col in factor_data.columns:
            try:
                validator.run_comprehensive_validation(
                    factor_values=factor_data[factor_col],
                    returns=returns_data,
                    factor_name=factor_col
                )
            except Exception as e:
                logger.error(f"验证因子 {factor_col} 时出错: {e}")
    
    return validator


def generate_factor_analysis_report(factor_data: pd.DataFrame,
                                   returns_data: pd.Series,
                                   factor_columns: List[str],
                                   report_dir: str = './reports') -> Dict[str, str]:
    """
    生成完整的因子分析报告
    
    Args:
        factor_data: 因子数据
        returns_data: 收益率数据
        factor_columns: 因子列名列表
        report_dir: 报告保存目录
        
    Returns:
        报告文件路径字典
    """
    # 确保目录存在
    os.makedirs(report_dir, exist_ok=True)
    
    # 分析因子
    analyzer = analyze_factors(factor_data, returns_data, factor_columns)
    analysis_results = analyzer.results
    
    # 生成分析报告
    analysis_report_dir = os.path.join(report_dir, 'factor_analysis')
    analysis_files = analyzer.generate_factor_report(
        analysis_results=analysis_results,
        factor_data=factor_data,
        returns_data=returns_data,
        report_dir=analysis_report_dir
    )
    
    # 验证因子
    validator = validate_factors(factor_data, returns_data, factor_columns)
    
    # 生成验证报告
    validation_report_dir = os.path.join(report_dir, 'factor_validation')
    validation_files = validator.generate_validation_report(
        validation_results=validator.validation_results,
        report_dir=validation_report_dir
    )
    
    # 合并报告文件
    all_report_files = {
        **{f'analysis_{k}': v for k, v in analysis_files.items()},
        **{f'validation_{k}': v for k, v in validation_files.items()}
    }
    
    logger.info(f"完整的因子分析和验证报告已生成，保存在: {report_dir}")
    
    return all_report_files