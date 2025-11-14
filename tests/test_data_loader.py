import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.dates = pd.date_range(start='2020-01-01', periods=10)
        self.test_data = pd.DataFrame({
            'date': self.dates,
            'open': np.random.random(10) * 100,
            'high': np.random.random(10) * 100 + 5,
            'low': np.random.random(10) * 100 - 5,
            'close': np.random.random(10) * 100,
            'volume': np.random.randint(1000, 100000, 10)
        })
        
        # 创建临时CSV文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv_path = os.path.join(self.temp_dir.name, 'test_data.csv')
        self.test_data.to_csv(self.temp_csv_path, index=False)
        
        # 初始化数据加载器
        self.data_loader = DataLoader()
    
    def tearDown(self):
        # 清理临时文件
        self.temp_dir.cleanup()
    
    def test_load_from_csv(self):
        # 测试从CSV加载数据
        data = self.data_loader.load_from_csv(self.temp_csv_path, date_column='date')
        
        # 验证数据加载成功
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), len(self.test_data))
        self.assertTrue('date' in data.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['date']))
    
    def test_load_from_csv_with_index(self):
        # 测试将日期列设为索引
        data = self.data_loader.load_from_csv(self.temp_csv_path, date_column='date', set_index=True)
        
        # 验证索引设置成功
        self.assertTrue(isinstance(data.index, pd.DatetimeIndex))
        self.assertNotIn('date', data.columns)
    
    def test_load_from_csv_missing_file(self):
        # 测试加载不存在的文件
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_from_csv('non_existent_file.csv')
    
    def test_load_from_csv_invalid_date_column(self):
        # 测试无效的日期列
        with self.assertRaises(KeyError):
            self.data_loader.load_from_csv(self.temp_csv_path, date_column='invalid_date')
    
    @patch('src.data.data_loader.sqlalchemy.create_engine')
    def test_load_from_database(self, mock_create_engine):
        # 设置模拟数据库连接
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value.fetchall.return_value = self.test_data.to_dict('records')
        mock_connection.execute.return_value.keys.return_value = self.test_data.columns
        
        # 测试数据库加载
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        data = self.data_loader.load_from_database('test_table', db_config, date_column='date')
        
        # 验证数据库查询被调用
        mock_create_engine.assert_called_once()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), len(self.test_data))
    
    @patch('src.data.data_loader.requests.get')
    def test_load_from_api(self, mock_get):
        # 设置模拟API响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': self.test_data.to_dict('records')
        }
        mock_get.return_value = mock_response
        
        # 测试API加载
        api_config = {
            'api_key': 'test_key',
            'endpoint': 'https://api.example.com/data'
        }
        
        data = self.data_loader.load_from_api(api_config, params={'limit': 10})
        
        # 验证API调用
        mock_get.assert_called_once()
        self.assertIsInstance(data, pd.DataFrame)
    
    def test_save_to_csv(self):
        # 测试保存数据到CSV
        output_path = os.path.join(self.temp_dir.name, 'output.csv')
        self.data_loader.save_to_csv(self.test_data, output_path)
        
        # 验证文件保存成功
        self.assertTrue(os.path.exists(output_path))
        loaded_data = pd.read_csv(output_path)
        self.assertEqual(len(loaded_data), len(self.test_data))
    
    def test_split_data(self):
        # 测试数据分割
        train, test = self.data_loader.split_data(self.test_data, test_size=0.3, shuffle=False)
        
        # 验证分割比例
        self.assertEqual(len(train), 7)  # 70% of 10
        self.assertEqual(len(test), 3)   # 30% of 10
        
        # 验证数据顺序
        pd.testing.assert_series_equal(train['date'], self.test_data['date'].iloc[:7])
        pd.testing.assert_series_equal(test['date'], self.test_data['date'].iloc[7:])
    
    def test_split_data_shuffle(self):
        # 测试随机分割
        train, test = self.data_loader.split_data(self.test_data, test_size=0.3, shuffle=True, random_state=42)
        
        # 验证分割比例
        self.assertEqual(len(train), 7)
        self.assertEqual(len(test), 3)
        
        # 验证数据被打乱（通过比较索引）
        self.assertNotEqual(list(train.index), list(range(7)))
    
    def test_resample_data_daily_to_weekly(self):
        # 创建更密集的测试数据用于重采样测试
        dates = pd.date_range(start='2020-01-01', periods=30)
        dense_data = pd.DataFrame({
            'date': dates,
            'close': np.random.random(30) * 100,
            'volume': np.random.randint(1000, 100000, 30)
        })
        dense_data.set_index('date', inplace=True)
        
        # 测试重采样
        resampled = self.data_loader.resample_data(
            dense_data,
            freq='W',
            agg_dict={'close': 'last', 'volume': 'sum'}
        )
        
        # 验证重采样结果
        self.assertTrue(len(resampled) < len(dense_data))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(resampled.index))

if __name__ == '__main__':
    unittest.main()