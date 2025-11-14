import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch
from src.models.model_training import ModelTrainer, EnsembleModel
from src.models.data_preprocessing import TimeSeriesDataLoader

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.dates = pd.date_range(start='2020-01-01', periods=200)
        
        # 创建一些特征和目标变量（带有一些线性关系）
        np.random.seed(42)
        X = np.random.randn(200, 5)
        # 目标变量与特征有线性关系，加上一些噪声
        y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(200) * 0.5
        
        self.test_data = pd.DataFrame({
            'date': self.dates,
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'feature4': X[:, 3],
            'feature5': X[:, 4],
            'target': y
        })
        
        # 分割训练和测试数据
        self.X_train = X[:150]
        self.y_train = y[:150]
        self.X_test = X[150:]
        self.y_test = y[150:]
        
        # 初始化模型训练器
        self.trainer = ModelTrainer()
    
    def test_train_linear_regression(self):
        # 测试线性回归模型训练
        model = self.trainer.train_model(
            model_type='linear_regression',
            X_train=self.X_train,
            y_train=self.y_train,
            params={}
        )
        
        # 验证模型已训练
        self.assertIsNotNone(model)
        
        # 验证预测
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_train_random_forest(self):
        # 测试随机森林模型训练
        model = self.trainer.train_model(
            model_type='random_forest',
            X_train=self.X_train,
            y_train=self.y_train,
            params={'n_estimators': 50, 'max_depth': 5}
        )
        
        # 验证模型已训练
        self.assertIsNotNone(model)
        
        # 验证预测
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_train_xgboost(self):
        # 测试XGBoost模型训练
        model = self.trainer.train_model(
            model_type='xgboost',
            X_train=self.X_train,
            y_train=self.y_train,
            params={'n_estimators': 50, 'max_depth': 5}
        )
        
        # 验证模型已训练
        self.assertIsNotNone(model)
        
        # 验证预测
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_evaluate_model(self):
        # 先训练一个模型
        model = self.trainer.train_model(
            model_type='linear_regression',
            X_train=self.X_train,
            y_train=self.y_train,
            params={}
        )
        
        # 评估模型
        metrics = self.trainer.evaluate_model(
            model=model,
            X_test=self.X_test,
            y_test=self.y_test,
            metrics=['mse', 'mae', 'r2']
        )
        
        # 验证指标已计算
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # 由于数据有线性关系，R²应该相对较高
        self.assertTrue(metrics['r2'] > 0.8)
    
    def test_hyperparameter_tuning(self):
        # 测试超参数调优
        param_grid = {
            'n_estimators': [30, 50],
            'max_depth': [3, 5]
        }
        
        best_params = self.trainer.hyperparameter_tuning(
            model_type='random_forest',
            X_train=self.X_train,
            y_train=self.y_train,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error'
        )
        
        # 验证最佳参数已返回
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
        self.assertTrue(best_params['n_estimators'] in [30, 50])
        self.assertTrue(best_params['max_depth'] in [3, 5])
    
    def test_create_ensemble(self):
        # 测试创建集成模型
        model_types = ['linear_regression', 'random_forest', 'xgboost']
        
        ensemble = self.trainer.create_ensemble(
            model_types=model_types,
            X_train=self.X_train,
            y_train=self.y_train,
            ensemble_method='mean'
        )
        
        # 验证集成模型已创建
        self.assertIsInstance(ensemble, EnsembleModel)
        
        # 验证预测
        predictions = ensemble.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_save_and_load_model(self):
        # 先训练一个模型
        model = self.trainer.train_model(
            model_type='linear_regression',
            X_train=self.X_train,
            y_train=self.y_train,
            params={}
        )
        
        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.trainer.save_model(model, temp_path)
            
            # 验证文件已保存
            self.assertTrue(os.path.exists(temp_path))
            
            # 加载模型
            loaded_model = self.trainer.load_model(temp_path)
            
            # 验证加载的模型可以进行预测
            predictions = loaded_model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
            # 验证加载的模型预测结果与原模型相同
            original_predictions = model.predict(self.X_test)
            np.testing.assert_array_almost_equal(predictions, original_predictions)
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_feature_importance(self):
        # 测试特征重要性分析（仅适用于树模型）
        model = self.trainer.train_model(
            model_type='random_forest',
            X_train=self.X_train,
            y_train=self.y_train,
            params={'n_estimators': 50, 'max_depth': 5}
        )
        
        feature_importance = self.trainer.get_feature_importance(
            model, 
            feature_names=['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        )
        
        # 验证特征重要性已返回
        self.assertEqual(len(feature_importance), 5)
        self.assertIn('feature1', feature_importance)
        self.assertIn('feature2', feature_importance)
        self.assertIn('feature3', feature_importance)
        
        # 由于特征1和2在生成目标变量时权重较高，它们的重要性应该相对较高
        top_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:2]
        self.assertIn('feature1', top_features)
        self.assertIn('feature2', top_features)
    
    def test_cross_validation(self):
        # 测试交叉验证
        cv_results = self.trainer.cross_validate(
            model_type='linear_regression',
            X=self.X_train,
            y=self.y_train,
            cv=5,
            scoring=['mse', 'r2']
        )
        
        # 验证交叉验证结果
        self.assertIn('test_mse', cv_results)
        self.assertIn('test_r2', cv_results)
        self.assertEqual(len(cv_results['test_mse']), 5)
    
    def test_ensemble_model_weighted(self):
        # 测试加权集成模型
        model_types = ['linear_regression', 'random_forest', 'xgboost']
        weights = [0.2, 0.5, 0.3]  # 随机森林权重最高
        
        ensemble = self.trainer.create_ensemble(
            model_types=model_types,
            X_train=self.X_train,
            y_train=self.y_train,
            ensemble_method='weighted',
            weights=weights
        )
        
        # 验证集成模型已创建
        self.assertIsInstance(ensemble, EnsembleModel)
        
        # 验证预测
        predictions = ensemble.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_compare_models(self):
        # 训练多个模型
        model1 = self.trainer.train_model(
            model_type='linear_regression',
            X_train=self.X_train,
            y_train=self.y_train,
            params={}
        )
        
        model2 = self.trainer.train_model(
            model_type='random_forest',
            X_train=self.X_train,
            y_train=self.y_train,
            params={'n_estimators': 50}
        )
        
        # 比较模型
        comparison = self.trainer.compare_models(
            models={'linear': model1, 'rf': model2},
            X_test=self.X_test,
            y_test=self.y_test,
            metrics=['mse', 'mae', 'r2']
        )
        
        # 验证比较结果
        self.assertIn('linear', comparison)
        self.assertIn('rf', comparison)
        self.assertIn('mse', comparison['linear'])
        self.assertIn('mse', comparison['rf'])

if __name__ == '__main__':
    unittest.main()