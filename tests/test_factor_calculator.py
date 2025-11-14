import unittest
import pandas as pd
import numpy as np
from src.factors.factor_calculator import FactorCalculator

class TestFactorCalculator(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.dates = pd.date_range(start='2020-01-01', periods=100)
        self.prices = np.cumsum(np.random.randn(100)) + 100  # 随机游走价格序列
        self.test_data = pd.DataFrame({
            'date': self.dates,
            'open': self.prices + np.random.randn(100) * 2,
            'high': self.prices + np.random.randn(100) * 2 + 3,
            'low': self.prices + np.random.randn(100) * 2 - 3,
            'close': self.prices,
            'volume': np.random.randint(1000, 100000, 100)
        })
        self.test_data.set_index('date', inplace=True)
        
        # 初始化因子计算器
        self.calculator = FactorCalculator()
    
    def test_calculate_moving_average(self):
        # 测试移动平均线计算
        result = self.calculator.calculate_moving_average(
            self.test_data, 
            window=20, 
            column='close', 
            output_column='ma20'
        )
        
        # 验证结果
        self.assertTrue('ma20' in result.columns)
        # 前19个值应该是NaN
        self.assertTrue(result['ma20'].iloc[:19].isna().all())
        # 第20个值应该等于前20个收盘价的均值
        expected_ma = self.test_data['close'].iloc[:20].mean()
        self.assertAlmostEqual(result['ma20'].iloc[19], expected_ma, places=5)
    
    def test_calculate_rsi(self):
        # 测试RSI计算
        result = self.calculator.calculate_rsi(
            self.test_data, 
            window=14, 
            column='close', 
            output_column='rsi14'
        )
        
        # 验证结果
        self.assertTrue('rsi14' in result.columns)
        # 前13个值应该是NaN
        self.assertTrue(result['rsi14'].iloc[:13].isna().all())
        # RSI值应该在0到100之间
        self.assertTrue((result['rsi14'].dropna() >= 0).all())
        self.assertTrue((result['rsi14'].dropna() <= 100).all())
    
    def test_calculate_macd(self):
        # 测试MACD计算
        result = self.calculator.calculate_macd(
            self.test_data, 
            fast=12, 
            slow=26, 
            signal=9, 
            column='close'
        )
        
        # 验证结果
        self.assertTrue('macd' in result.columns)
        self.assertTrue('macd_signal' in result.columns)
        self.assertTrue('macd_hist' in result.columns)
        
        # 前25个值应该是NaN
        self.assertTrue(result['macd'].iloc[:25].isna().all())
        
        # MACD柱状图应该等于MACD减去信号线
        macd_hist_calculated = result['macd'] - result['macd_signal']
        pd.testing.assert_series_equal(result['macd_hist'], macd_hist_calculated, check_names=False)
    
    def test_calculate_bollinger_bands(self):
        # 测试布林带计算
        result = self.calculator.calculate_bollinger_bands(
            self.test_data, 
            window=20, 
            num_std=2, 
            column='close'
        )
        
        # 验证结果
        self.assertTrue('bb_upper' in result.columns)
        self.assertTrue('bb_middle' in result.columns)
        self.assertTrue('bb_lower' in result.columns)
        
        # 前19个值应该是NaN
        self.assertTrue(result['bb_upper'].iloc[:19].isna().all())
        
        # 上轨应该等于中轨加上2倍标准差
        bb_middle = self.test_data['close'].rolling(20).mean()
        bb_std = self.test_data['close'].rolling(20).std()
        expected_upper = bb_middle + 2 * bb_std
        
        # 比较非NaN部分
        valid_idx = ~result['bb_upper'].isna()
        np.testing.assert_array_almost_equal(
            result['bb_upper'][valid_idx].values, 
            expected_upper[valid_idx].values,
            decimal=5
        )
    
    def test_calculate_volatility(self):
        # 测试波动率计算
        result = self.calculator.calculate_volatility(
            self.test_data, 
            window=20, 
            column='close', 
            output_column='volatility20'
        )
        
        # 验证结果
        self.assertTrue('volatility20' in result.columns)
        # 前19个值应该是NaN
        self.assertTrue(result['volatility20'].iloc[:19].isna().all())
        
        # 计算预期的波动率（收益率的标准差，年化）
        returns = self.test_data['close'].pct_change()
        expected_vol = returns.rolling(20).std() * np.sqrt(252)  # 年化
        
        # 比较非NaN部分
        valid_idx = ~result['volatility20'].isna()
        np.testing.assert_array_almost_equal(
            result['volatility20'][valid_idx].values, 
            expected_vol[valid_idx].values,
            decimal=5
        )
    
    def test_calculate_skewness(self):
        # 测试偏度计算
        result = self.calculator.calculate_skewness(
            self.test_data, 
            window=20, 
            column='close', 
            output_column='skewness20'
        )
        
        # 验证结果
        self.assertTrue('skewness20' in result.columns)
        # 前19个值应该是NaN
        self.assertTrue(result['skewness20'].iloc[:19].isna().all())
    
    def test_calculate_kurtosis(self):
        # 测试峰度计算
        result = self.calculator.calculate_kurtosis(
            self.test_data, 
            window=20, 
            column='close', 
            output_column='kurtosis20'
        )
        
        # 验证结果
        self.assertTrue('kurtosis20' in result.columns)
        # 前19个值应该是NaN
        self.assertTrue(result['kurtosis20'].iloc[:19].isna().all())
    
    def test_calculate_correlation(self):
        # 测试相关性计算
        # 先创建两个相关的因子列
        result = self.calculator.calculate_moving_average(
            self.test_data, 
            window=5, 
            column='close', 
            output_column='ma5'
        )
        result = self.calculator.calculate_moving_average(
            result, 
            window=10, 
            column='close', 
            output_column='ma10'
        )
        
        # 计算相关性
        correlation = self.calculator.calculate_correlation(
            result, 
            columns1=['ma5'], 
            columns2=['ma10'], 
            window=20,
            output_column='corr_ma5_ma10'
        )
        
        # 验证结果
        self.assertTrue('corr_ma5_ma10' in correlation.columns)
        # 前38个值应该是NaN（20+10+5-1-1）
        self.assertTrue(correlation['corr_ma5_ma10'].iloc[:38].isna().all())
        # 相关性值应该在-1到1之间
        self.assertTrue((correlation['corr_ma5_ma10'].dropna() >= -1).all())
        self.assertTrue((correlation['corr_ma5_ma10'].dropna() <= 1).all())
    
    def test_calculate_custom_factor(self):
        # 定义自定义因子函数
        def custom_momentum(series, window=10):
            return series / series.shift(window) - 1
        
        # 测试自定义因子计算
        result = self.calculator.calculate_custom_factor(
            self.test_data, 
            column='close', 
            custom_func=custom_momentum, 
            output_column='momentum10',
            window=10
        )
        
        # 验证结果
        self.assertTrue('momentum10' in result.columns)
        # 前9个值应该是NaN
        self.assertTrue(result['momentum10'].iloc[:9].isna().all())
        
        # 计算预期结果
        expected_momentum = self.test_data['close'] / self.test_data['close'].shift(10) - 1
        
        # 比较非NaN部分
        valid_idx = ~result['momentum10'].isna()
        np.testing.assert_array_almost_equal(
            result['momentum10'][valid_idx].values, 
            expected_momentum[valid_idx].values,
            decimal=5
        )
    
    def test_calculate_multiple_factors(self):
        # 测试同时计算多个因子
        result = self.calculator.calculate_moving_average(
            self.test_data, 
            window=20, 
            column='close', 
            output_column='ma20'
        )
        result = self.calculator.calculate_rsi(
            result, 
            window=14, 
            column='close', 
            output_column='rsi14'
        )
        result = self.calculator.calculate_volatility(
            result, 
            window=30, 
            column='close', 
            output_column='volatility30'
        )
        
        # 验证所有因子都被计算
        for col in ['ma20', 'rsi14', 'volatility30']:
            self.assertTrue(col in result.columns)

if __name__ == '__main__':
    unittest.main()