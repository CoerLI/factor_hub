import unittest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
import tempfile

# 导入被测试的模块
from src.factors.factor_persistence import FactorPersistenceManager
from src.factors.factor_metadata import FactorMetadataCollector
from src.factors.factor_base import FactorBase, FactorResult
from src.factors.factor_engine import FactorEngine
from src.factors.moving_averages import SimpleMovingAverage


class DummyFactor(FactorBase):
    """用于测试的虚拟因子"""
    
    def __init__(self, param1=10, param2="test"):
        self.param1 = param1
        self.param2 = param2
        super().__init__(
            name=f"DummyFactor_{param1}_{param2}",
            description="用于测试的虚拟因子",
            dependencies=[],
            params={"param1": param1, "param2": param2}
        )
    
    def compute(self, data: pd.DataFrame, dependency_results: dict, params: dict = None):
        # 生成测试结果
        result_data = pd.Series(np.random.randn(len(data)), index=data.index)
        return FactorResult(
            name=self.name,
            result=result_data,
            params=params or self.params,
            computation_time=0.1
        )


class TestFactorPersistence(unittest.TestCase):
    """测试因子持久化功能"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录作为存储
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建示例数据
        self.data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(100, 1000, size=100)
        }, index=pd.date_range(start='2024-01-01', periods=100))
        
        # 创建测试因子
        self.dummy_factor = DummyFactor(param1=5, param2="test_value")
        self.sma_factor = SimpleMovingAverage(window=10)
        
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_persistence_manager_init(self):
        """测试持久化管理器初始化"""
        # 测试默认初始化
        manager = FactorPersistenceManager(storage_dir=self.temp_dir)
        self.assertEqual(manager.storage_dir, self.temp_dir)
        self.assertEqual(manager.default_format, "sqlite")
        
        # 测试JSON格式初始化
        manager = FactorPersistenceManager(storage_dir=self.temp_dir, default_format="json")
        self.assertEqual(manager.default_format, "json")
        
        # 测试Parquet格式初始化
        manager = FactorPersistenceManager(storage_dir=self.temp_dir, default_format="parquet")
        self.assertEqual(manager.default_format, "parquet")
    
    def test_metadata_collection(self):
        """测试元数据收集"""
        collector = FactorMetadataCollector()
        
        # 收集因子元数据
        metadata = collector.collect_factor_metadata(self.dummy_factor)
        
        # 验证元数据基本信息
        self.assertEqual(metadata["name"], self.dummy_factor.name)
        self.assertEqual(metadata["description"], self.dummy_factor.description)
        self.assertEqual(metadata["class_name"], "DummyFactor")
        
        # 验证参数信息
        self.assertIn("param1", metadata["params"])
        self.assertIn("param2", metadata["params"])
        self.assertEqual(metadata["params"]["param1"]["value"], 5)
        
        # 验证性能信息初始化
        self.assertIn("performance_info", metadata)
        self.assertEqual(metadata["performance_info"]["computation_count"], 0)
    
    def test_save_and_get_metadata(self):
        """测试保存和获取元数据"""
        manager = FactorPersistenceManager(storage_dir=self.temp_dir)
        collector = FactorMetadataCollector()
        
        # 收集和保存元数据
        metadata = collector.collect_factor_metadata(self.dummy_factor)
        manager.save_factor_metadata(self.dummy_factor.name, metadata)
        
        # 获取元数据
        retrieved_metadata = manager.get_factor_metadata(self.dummy_factor.name)
        
        # 验证元数据一致性
        self.assertEqual(retrieved_metadata["name"], metadata["name"])
        self.assertEqual(retrieved_metadata["description"], metadata["description"])
    
    def test_save_and_get_factor(self):
        """测试保存和获取因子数据"""
        manager = FactorPersistenceManager(storage_dir=self.temp_dir)
        collector = FactorMetadataCollector()
        
        # 计算因子
        factor_result = self.dummy_factor.compute(self.data, {})
        
        # 收集元数据
        metadata = collector.collect_factor_metadata(
            self.dummy_factor,
            data=self.data,
            result=factor_result,
            computation_time=0.1
        )
        
        # 保存因子数据
        manager.save_factor(
            factor_id=self.dummy_factor.name,
            metadata=metadata,
            result=factor_result.result
        )
        
        # 获取因子数据
        retrieved_factor = manager.get_factor_by_id(self.dummy_factor.name)
        
        # 验证因子数据
        self.assertEqual(retrieved_factor["metadata"]["name"], self.dummy_factor.name)
        self.assertTrue(isinstance(retrieved_factor["result"], pd.Series))
        self.assertEqual(len(retrieved_factor["result"]), len(self.data))
    
    def test_list_factors(self):
        """测试列出因子"""
        manager = FactorPersistenceManager(storage_dir=self.temp_dir)
        collector = FactorMetadataCollector()
        
        # 保存两个因子
        for factor in [self.dummy_factor, self.sma_factor]:
            metadata = collector.collect_factor_metadata(factor)
            manager.save_factor_metadata(factor.name, metadata)
        
        # 列出因子
        factors = manager.list_factors()
        
        # 验证因子数量
        self.assertEqual(len(factors), 2)
        
        # 验证因子ID存在
        factor_ids = [f["id"] for f in factors]
        self.assertIn(self.dummy_factor.name, factor_ids)
        self.assertIn(self.sma_factor.name, factor_ids)
    
    def test_delete_factor(self):
        """测试删除因子"""
        manager = FactorPersistenceManager(storage_dir=self.temp_dir)
        collector = FactorMetadataCollector()
        
        # 保存因子
        metadata = collector.collect_factor_metadata(self.dummy_factor)
        manager.save_factor_metadata(self.dummy_factor.name, metadata)
        
        # 删除因子
        manager.delete_factor(self.dummy_factor.name)
        
        # 验证因子已删除
        retrieved_metadata = manager.get_factor_metadata(self.dummy_factor.name)
        self.assertIsNone(retrieved_metadata)
    
    def test_export_import_factors(self):
        """测试导出和导入因子"""
        export_dir = os.path.join(self.temp_dir, "exports")
        
        # 保存因子数据
        manager1 = FactorPersistenceManager(storage_dir=self.temp_dir)
        collector = FactorMetadataCollector()
        metadata = collector.collect_factor_metadata(self.dummy_factor)
        manager1.save_factor_metadata(self.dummy_factor.name, metadata)
        
        # 导出因子
        manager1.export_factors(export_dir=export_dir, format="json")
        
        # 创建新的管理器
        import_dir = os.path.join(self.temp_dir, "imports")
        os.makedirs(import_dir, exist_ok=True)
        manager2 = FactorPersistenceManager(storage_dir=import_dir)
        
        # 导入因子
        manager2.import_factors(import_dir=export_dir)
        
        # 验证导入成功
        retrieved_metadata = manager2.get_factor_metadata(self.dummy_factor.name)
        self.assertIsNotNone(retrieved_metadata)
        self.assertEqual(retrieved_metadata["name"], self.dummy_factor.name)


class TestFactorEnginePersistenceIntegration(unittest.TestCase):
    """测试因子引擎与持久化模块的集成"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录作为存储
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建示例数据
        self.data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(100, 1000, size=100)
        }, index=pd.date_range(start='2024-01-01', periods=100))
        
        # 创建测试因子
        self.dummy_factor = DummyFactor(param1=5)
        
        # 创建启用持久化的因子引擎
        self.engine = FactorEngine(
            persistence_enabled=True,
            storage_dir=self.temp_dir,
            use_cache=False  # 禁用缓存以更好地测试持久化
        )
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_factor_registration_persistence(self):
        """测试因子注册时的持久化"""
        # 注册因子
        self.engine.register_factor(self.dummy_factor)
        
        # 验证因子元数据已保存
        retrieved_metadata = self.engine.persistence_manager.get_factor_metadata(
            self.dummy_factor.name
        )
        self.assertIsNotNone(retrieved_metadata)
        self.assertEqual(retrieved_metadata["name"], self.dummy_factor.name)
    
    def test_factor_computation_persistence(self):
        """测试因子计算时的持久化"""
        # 注册并计算因子
        self.engine.register_factor(self.dummy_factor)
        self.engine.compute_factors(
            self.data, 
            factor_names=[self.dummy_factor.name]
        )
        
        # 验证因子数据已保存
        factor_data = self.engine.load_factor_from_storage(self.dummy_factor.name)
        self.assertIsNotNone(factor_data)
        self.assertTrue(isinstance(factor_data["result"], pd.Series))
        
        # 验证性能信息已更新
        self.assertGreater(
            factor_data["metadata"]["performance_info"]["computation_count"],
            0
        )
    
    def test_persist_results_parameter(self):
        """测试persist_results参数"""
        # 注册因子
        self.engine.register_factor(self.dummy_factor)
        
        # 计算但不持久化
        self.engine.compute_factors(
            self.data, 
            factor_names=[self.dummy_factor.name],
            persist_results=False
        )
        
        # 验证因子数据未保存
        factor_data = self.engine.load_factor_from_storage(self.dummy_factor.name)
        # 由于元数据在注册时已保存，所以这里应该只检查结果是否有数据
        self.assertIsNone(factor_data.get("result"))
    
    def test_get_all_persisted_factors(self):
        """测试获取所有已持久化的因子"""
        # 注册两个因子
        self.engine.register_factor(self.dummy_factor)
        self.engine.register_factor(DummyFactor(param1=10))
        
        # 获取所有因子
        all_factors = self.engine.get_all_persisted_factors()
        
        # 验证因子数量
        self.assertEqual(len(all_factors), 2)
    
    def test_container_persistence(self):
        """测试容器持久化"""
        # 注册并计算因子
        self.engine.register_factor(self.dummy_factor)
        container = self.engine.create_factor_container(
            self.data,
            factor_names=[self.dummy_factor.name]
        )
        
        # 验证容器信息已保存
        # 注意：这里需要访问persistence_manager来获取容器信息
        # 由于接口限制，我们只验证因子数据已保存
        factor_data = self.engine.load_factor_from_storage(self.dummy_factor.name)
        self.assertIsNotNone(factor_data)
    
    def test_update_config(self):
        """测试更新配置"""
        # 更新配置禁用持久化
        self.engine.update_factor_config({"persistence_enabled": False})
        self.assertFalse(self.engine.persistence_enabled)
        
        # 更新配置启用持久化
        self.engine.update_factor_config({"persistence_enabled": True})
        self.assertTrue(self.engine.persistence_enabled)


if __name__ == "__main__":
    unittest.main()