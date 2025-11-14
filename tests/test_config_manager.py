import unittest
import tempfile
import os
import yaml
from src.config.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # 创建临时配置文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_path = os.path.join(self.temp_dir.name, 'test_config.yaml')
        
        # 测试配置内容
        self.test_config = {
            'data': {
                'path': 'data/sample.csv',
                'date_column': 'date',
                'features': ['open', 'high', 'low', 'close', 'volume'],
                'target': 'close'
            },
            'model': {
                'type': 'xgboost',
                'params': {
                    'max_depth': 6,
                    'n_estimators': 100,
                    'learning_rate': 0.1
                }
            },
            'backtest': {
                'initial_capital': 100000,
                'commission': 0.001,
                'strategy': 'moving_average_cross',
                'params': {
                    'short_window': 50,
                    'long_window': 200
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/app.log'
            }
        }
        
        # 写入临时配置文件
        with open(self.temp_config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # 创建另一个配置文件用于测试合并
        self.override_config_path = os.path.join(self.temp_dir.name, 'override_config.yaml')
        self.override_config = {
            'model': {
                'params': {
                    'max_depth': 10,
                    'n_estimators': 200
                }
            },
            'backtest': {
                'initial_capital': 200000
            }
        }
        
        with open(self.override_config_path, 'w') as f:
            yaml.dump(self.override_config, f)
        
        # 初始化配置管理器
        self.config_manager = ConfigManager()
    
    def tearDown(self):
        # 清理临时文件
        self.temp_dir.cleanup()
    
    def test_load_config(self):
        # 测试加载配置文件
        config = self.config_manager.load_config(self.temp_config_path)
        
        # 验证配置已加载
        self.assertIsInstance(config, dict)
        self.assertIn('data', config)
        self.assertIn('model', config)
        self.assertIn('backtest', config)
        self.assertIn('logging', config)
        
        # 验证配置内容
        self.assertEqual(config['data']['path'], 'data/sample.csv')
        self.assertEqual(config['model']['type'], 'xgboost')
        self.assertEqual(config['backtest']['initial_capital'], 100000)
    
    def test_get_config(self):
        # 先加载配置
        self.config_manager.load_config(self.temp_config_path)
        
        # 测试获取整个配置
        full_config = self.config_manager.get_config()
        self.assertIsInstance(full_config, dict)
        self.assertIn('data', full_config)
        
        # 测试获取配置子集
        data_config = self.config_manager.get_config('data')
        self.assertEqual(data_config['path'], 'data/sample.csv')
        self.assertEqual(data_config['target'], 'close')
        
        # 测试获取嵌套配置
        model_params = self.config_manager.get_config('model.params')
        self.assertEqual(model_params['max_depth'], 6)
        self.assertEqual(model_params['n_estimators'], 100)
        
        # 测试获取单个配置项
        max_depth = self.config_manager.get_config('model.params.max_depth')
        self.assertEqual(max_depth, 6)
    
    def test_get_config_invalid_path(self):
        # 先加载配置
        self.config_manager.load_config(self.temp_config_path)
        
        # 测试获取不存在的配置路径
        with self.assertRaises(KeyError):
            self.config_manager.get_config('invalid_path')
        
        # 测试获取不存在的嵌套路径
        with self.assertRaises(KeyError):
            self.config_manager.get_config('data.invalid_param')
    
    def test_update_config(self):
        # 先加载配置
        self.config_manager.load_config(self.temp_config_path)
        
        # 测试更新单个配置项
        self.config_manager.update_config('model.type', 'random_forest')
        self.assertEqual(self.config_manager.get_config('model.type'), 'random_forest')
        
        # 测试更新嵌套配置项
        self.config_manager.update_config('model.params.max_depth', 12)
        self.assertEqual(self.config_manager.get_config('model.params.max_depth'), 12)
        
        # 测试添加新的配置项
        self.config_manager.update_config('model.params.subsample', 0.8)
        self.assertEqual(self.config_manager.get_config('model.params.subsample'), 0.8)
        
        # 测试添加新的配置分支
        self.config_manager.update_config('new_section.setting', 'value')
        self.assertEqual(self.config_manager.get_config('new_section.setting'), 'value')
    
    def test_save_config(self):
        # 先加载配置并修改
        self.config_manager.load_config(self.temp_config_path)
        self.config_manager.update_config('model.type', 'random_forest')
        
        # 保存到新文件
        save_path = os.path.join(self.temp_dir.name, 'saved_config.yaml')
        self.config_manager.save_config(save_path)
        
        # 验证文件已保存
        self.assertTrue(os.path.exists(save_path))
        
        # 验证保存的内容
        with open(save_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config['model']['type'], 'random_forest')
        self.assertEqual(saved_config['data']['path'], 'data/sample.csv')
    
    def test_merge_configs(self):
        # 加载主配置
        self.config_manager.load_config(self.temp_config_path)
        
        # 合并另一个配置
        self.config_manager.merge_config(self.override_config_path)
        
        # 验证合并后的配置
        self.assertEqual(self.config_manager.get_config('model.params.max_depth'), 10)  # 被覆盖
        self.assertEqual(self.config_manager.get_config('model.params.n_estimators'), 200)  # 被覆盖
        self.assertEqual(self.config_manager.get_config('model.params.learning_rate'), 0.1)  # 保持不变
        self.assertEqual(self.config_manager.get_config('backtest.initial_capital'), 200000)  # 被覆盖
        self.assertEqual(self.config_manager.get_config('backtest.commission'), 0.001)  # 保持不变
    
    def test_merge_configs_dict(self):
        # 加载主配置
        self.config_manager.load_config(self.temp_config_path)
        
        # 直接合并配置字典
        override_dict = {
            'backtest': {
                'params': {
                    'short_window': 30,
                    'long_window': 150
                }
            }
        }
        
        self.config_manager.merge_config(override_dict)
        
        # 验证合并后的配置
        self.assertEqual(self.config_manager.get_config('backtest.params.short_window'), 30)
        self.assertEqual(self.config_manager.get_config('backtest.params.long_window'), 150)
    
    def test_validate_config(self):
        # 先加载有效的配置
        self.config_manager.load_config(self.temp_config_path)
        
        # 定义验证规则
        validation_schema = {
            'data': {
                'required_keys': ['path', 'date_column', 'target'],
                'types': {'path': str, 'target': str}
            },
            'model': {
                'required_keys': ['type', 'params'],
                'types': {'type': str}
            },
            'backtest': {
                'required_keys': ['initial_capital', 'commission'],
                'types': {'initial_capital': int, 'commission': float},
                'ranges': {'initial_capital': (0, None), 'commission': (0, 1)}
            }
        }
        
        # 验证配置
        is_valid = self.config_manager.validate_config(validation_schema)
        self.assertTrue(is_valid)
        
        # 修改配置使其无效
        self.config_manager.update_config('backtest.initial_capital', -100)  # 无效值
        
        # 再次验证
        is_valid = self.config_manager.validate_config(validation_schema)
        self.assertFalse(is_valid)
    
    def test_load_multiple_configs(self):
        # 测试同时加载多个配置文件
        config = self.config_manager.load_config([self.temp_config_path, self.override_config_path])
        
        # 验证合并结果
        self.assertEqual(config['model']['params']['max_depth'], 10)
        self.assertEqual(config['backtest']['initial_capital'], 200000)
        self.assertEqual(config['data']['path'], 'data/sample.csv')
    
    def test_create_template(self):
        # 测试创建配置模板
        template = self.config_manager.create_template()
        
        # 验证模板结构
        self.assertIsInstance(template, dict)
        self.assertIn('data', template)
        self.assertIn('model', template)
        self.assertIn('backtest', template)
        self.assertIn('logging', template)
        
        # 保存模板
        template_path = os.path.join(self.temp_dir.name, 'template.yaml')
        self.config_manager.save_config(template_path, template)
        
        # 验证模板文件已保存
        self.assertTrue(os.path.exists(template_path))

if __name__ == '__main__':
    unittest.main()