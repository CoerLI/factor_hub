import pandas as pd
import numpy as np
import json
import os
import pickle
import sqlite3
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import hashlib
import shutil

from src.factors.factor_base import FactorBase, FactorContainer, FactorResult
from src.utils.helpers import setup_logger, ensure_directory, DEFAULT_LOGGER


class FactorPersistenceManager:
    """
    因子持久化管理器，负责因子信息的详细存储和检索
    
    功能特点：
    1. 支持因子元数据的完整保存
    2. 支持因子计算结果的持久化
    3. 提供多种存储后端（JSON、SQLite、Parquet等）
    4. 支持因子版本管理
    5. 提供高效的查询接口
    6. 支持因子依赖关系的保存和恢复
    """
    
    def __init__(self, 
                 storage_dir: str = ".data/factors",
                 storage_format: str = "sqlite",  # sqlite, json, parquet
                 max_history_versions: int = 5):
        """
        初始化因子持久化管理器
        
        Args:
            storage_dir: 存储目录
            storage_format: 存储格式（sqlite, json, parquet）
            max_history_versions: 最大历史版本数
        """
        self.storage_dir = storage_dir
        self.storage_format = storage_format.lower()
        self.max_history_versions = max_history_versions
        
        # 确保存储目录存在
        ensure_directory(self.storage_dir)
        ensure_directory(os.path.join(self.storage_dir, "metadata"))
        ensure_directory(os.path.join(self.storage_dir, "results"))
        ensure_directory(os.path.join(self.storage_dir, "versions"))
        
        # 设置日志
        self.logger = setup_logger(
            name="factor_persistence",
            log_file=os.path.join("logs", "factor_persistence.log"),
            level=DEFAULT_LOGGER.level
        )
        
        # 初始化存储后端
        self._init_storage_backend()
        
        self.logger.info(f"因子持久化管理器初始化完成，存储格式: {self.storage_format}")
    
    def _init_storage_backend(self):
        """初始化存储后端"""
        if self.storage_format == "sqlite":
            # 初始化SQLite数据库
            db_path = os.path.join(self.storage_dir, "factors.db")
            self.conn = sqlite3.connect(db_path)
            self._create_sqlite_tables()
        elif self.storage_format == "json":
            # 确保JSON存储目录存在
            ensure_directory(os.path.join(self.storage_dir, "metadata"))
            ensure_directory(os.path.join(self.storage_dir, "results"))
        elif self.storage_format == "parquet":
            # 确保Parquet存储目录存在
            ensure_directory(os.path.join(self.storage_dir, "parquet"))
        else:
            raise ValueError(f"不支持的存储格式: {self.storage_format}")
    
    def _create_sqlite_tables(self):
        """创建SQLite表结构"""
        cursor = self.conn.cursor()
        
        # 因子元数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS factor_metadata (
            factor_id TEXT PRIMARY KEY,
            factor_name TEXT NOT NULL,
            factor_type TEXT NOT NULL,
            description TEXT,
            class_name TEXT,
            module_path TEXT,
            created_at TEXT,
            updated_at TEXT,
            version INTEGER,
            is_active INTEGER DEFAULT 1,
            parameters TEXT,
            dependencies TEXT,
            performance_metrics TEXT,
            validation_results TEXT
        )
        ''')
        
        # 因子计算结果表（引用外部文件）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS factor_results (
            result_id TEXT PRIMARY KEY,
            factor_id TEXT,
            timestamp TEXT,
            data_hash TEXT,
            params_hash TEXT,
            result_path TEXT,
            sample_size INTEGER,
            computation_time REAL,
            FOREIGN KEY (factor_id) REFERENCES factor_metadata (factor_id)
        )
        ''')
        
        # 因子版本历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS factor_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id TEXT,
            version INTEGER,
            factor_data TEXT,
            created_at TEXT,
            FOREIGN KEY (factor_id) REFERENCES factor_metadata (factor_id)
        )
        ''')
        
        # 因子关系表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS factor_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id TEXT,
            dependency_id TEXT,
            relationship_type TEXT,
            FOREIGN KEY (factor_id) REFERENCES factor_metadata (factor_id),
            FOREIGN KEY (dependency_id) REFERENCES factor_metadata (factor_id)
        )
        ''')
        
        self.conn.commit()
    
    def save_factor(self, 
                   factor: FactorBase,
                   compute_data: Optional[pd.DataFrame] = None,
                   compute_result: Optional[pd.DataFrame] = None,
                   dependencies: Optional[List[str]] = None,
                   performance_metrics: Optional[Dict] = None,
                   validation_results: Optional[Dict] = None) -> str:
        """
        保存因子及其详细信息
        
        Args:
            factor: 因子实例
            compute_data: 用于计算的原始数据（可选，用于生成哈希值）
            compute_result: 因子计算结果（可选）
            dependencies: 因子依赖列表（可选）
            performance_metrics: 性能指标（可选）
            validation_results: 验证结果（可选）
            
        Returns:
            因子ID
        """
        # 生成因子ID
        factor_id = self._generate_factor_id(factor)
        
        # 收集因子元数据
        metadata = self._collect_factor_metadata(
            factor, factor_id, dependencies, performance_metrics, validation_results
        )
        
        # 保存元数据
        if self.storage_format == "sqlite":
            self._save_to_sqlite(factor_id, metadata, compute_data, compute_result)
        elif self.storage_format == "json":
            self._save_to_json(factor_id, metadata, compute_data, compute_result)
        elif self.storage_format == "parquet":
            self._save_to_parquet(factor_id, metadata, compute_data, compute_result)
        
        self.logger.info(f"因子 {factor.factor_name} (ID: {factor_id}) 保存成功")
        return factor_id
    
    def _generate_factor_id(self, factor: FactorBase) -> str:
        """生成唯一的因子ID"""
        # 结合因子名称和参数生成唯一ID
        params_str = json.dumps(factor.params, sort_keys=True)
        combined = f"{factor.__class__.__name__}:{factor.factor_name}:{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _collect_factor_metadata(self, 
                                factor: FactorBase,
                                factor_id: str,
                                dependencies: Optional[List[str]],
                                performance_metrics: Optional[Dict],
                                validation_results: Optional[Dict]) -> Dict[str, Any]:
        """收集因子的完整元数据"""
        timestamp = datetime.now().isoformat()
        
        # 获取类的模块路径
        module_path = factor.__class__.__module__
        class_name = factor.__class__.__name__
        
        # 收集完整的元数据
        metadata = {
            "factor_id": factor_id,
            "factor_name": factor.factor_name,
            "factor_type": factor.factor_type,
            "description": factor.description,
            "class_name": class_name,
            "module_path": module_path,
            "created_at": timestamp,
            "updated_at": timestamp,
            "version": 1,
            "parameters": factor.params,
            "dependencies": dependencies or [],
            "performance_metrics": performance_metrics or {},
            "validation_results": validation_results or {}
        }
        
        return metadata
    
    def _save_to_sqlite(self, 
                       factor_id: str,
                       metadata: Dict[str, Any],
                       compute_data: Optional[pd.DataFrame],
                       compute_result: Optional[pd.DataFrame]):
        """保存因子到SQLite数据库"""
        cursor = self.conn.cursor()
        
        # 检查因子是否已存在
        cursor.execute("SELECT version FROM factor_metadata WHERE factor_id = ?", (factor_id,))
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有因子
            version = existing[0] + 1
            metadata["version"] = version
            
            # 保存旧版本到历史表
            cursor.execute("SELECT * FROM factor_metadata WHERE factor_id = ?", (factor_id,))
            old_data = cursor.fetchone()
            if old_data:
                # 保存到版本历史
                cursor.execute('''
                INSERT INTO factor_versions (factor_id, version, factor_data, created_at)
                VALUES (?, ?, ?, ?)
                ''', (factor_id, version-1, json.dumps(dict(zip([d[0] for d in cursor.description], old_data))), datetime.now().isoformat()))
            
            # 更新主表
            cursor.execute('''
            UPDATE factor_metadata
            SET factor_name = ?, factor_type = ?, description = ?, 
                class_name = ?, module_path = ?, updated_at = ?, 
                version = ?, parameters = ?, dependencies = ?, 
                performance_metrics = ?, validation_results = ?
            WHERE factor_id = ?
            ''', (
                metadata["factor_name"], metadata["factor_type"], metadata["description"],
                metadata["class_name"], metadata["module_path"], metadata["updated_at"],
                metadata["version"], json.dumps(metadata["parameters"]), 
                json.dumps(metadata["dependencies"]),
                json.dumps(metadata["performance_metrics"]), 
                json.dumps(metadata["validation_results"]),
                factor_id
            ))
        else:
            # 插入新因子
            cursor.execute('''
            INSERT INTO factor_metadata 
            (factor_id, factor_name, factor_type, description, class_name, module_path, 
             created_at, updated_at, version, parameters, dependencies, 
             performance_metrics, validation_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                factor_id, metadata["factor_name"], metadata["factor_type"], 
                metadata["description"], metadata["class_name"], metadata["module_path"],
                metadata["created_at"], metadata["updated_at"], metadata["version"],
                json.dumps(metadata["parameters"]), json.dumps(metadata["dependencies"]),
                json.dumps(metadata["performance_metrics"]), json.dumps(metadata["validation_results"])
            ))
        
        # 保存计算结果（如果提供）
        if compute_result is not None:
            self._save_factor_result(factor_id, compute_data, compute_result)
        
        # 清理旧版本历史
        self._cleanup_old_versions(factor_id)
        
        self.conn.commit()
    
    def _save_to_json(self, 
                     factor_id: str,
                     metadata: Dict[str, Any],
                     compute_data: Optional[pd.DataFrame],
                     compute_result: Optional[pd.DataFrame]):
        """保存因子到JSON文件"""
        # 保存元数据
        metadata_path = os.path.join(self.storage_dir, "metadata", f"{factor_id}.json")
        
        # 检查是否已存在
        if os.path.exists(metadata_path):
            # 读取现有数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            
            # 更新版本
            version = existing.get("version", 0) + 1
            metadata["version"] = version
            
            # 保存历史版本
            version_dir = os.path.join(self.storage_dir, "versions", factor_id)
            ensure_directory(version_dir)
            version_path = os.path.join(version_dir, f"v{version-1}_{existing['updated_at']}.json")
            with open(version_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        
        # 保存当前版本
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 保存计算结果
        if compute_result is not None:
            result_dir = os.path.join(self.storage_dir, "results", factor_id)
            ensure_directory(result_dir)
            
            # 生成结果文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(result_dir, f"result_{timestamp}.parquet")
            
            # 保存为Parquet格式（高效存储DataFrame）
            compute_result.to_parquet(result_path)
            
            # 更新结果引用
            results_index_path = os.path.join(result_dir, "results_index.json")
            results_index = {}
            if os.path.exists(results_index_path):
                with open(results_index_path, 'r', encoding='utf-8') as f:
                    results_index = json.load(f)
            
            data_hash = self._compute_data_hash(compute_data) if compute_data is not None else ""
            params_hash = self._compute_params_hash(metadata["parameters"])
            
            results_index[timestamp] = {
                "result_path": result_path,
                "data_hash": data_hash,
                "params_hash": params_hash,
                "sample_size": len(compute_result),
                "created_at": metadata["updated_at"]
            }
            
            with open(results_index_path, 'w', encoding='utf-8') as f:
                json.dump(results_index, f, ensure_ascii=False, indent=2)
    
    def _save_to_parquet(self, 
                        factor_id: str,
                        metadata: Dict[str, Any],
                        compute_data: Optional[pd.DataFrame],
                        compute_result: Optional[pd.DataFrame]):
        """保存因子到Parquet文件"""
        # 元数据存储为JSON
        metadata_path = os.path.join(self.storage_dir, "parquet", f"{factor_id}_metadata.json")
        
        # 检查是否已存在
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            metadata["version"] = existing.get("version", 0) + 1
        
        # 保存元数据
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 保存计算结果为Parquet
        if compute_result is not None:
            result_path = os.path.join(self.storage_dir, "parquet", f"{factor_id}_result_{metadata['version']}.parquet")
            compute_result.to_parquet(result_path)
    
    def _save_factor_result(self, 
                           factor_id: str,
                           compute_data: Optional[pd.DataFrame],
                           compute_result: pd.DataFrame):
        """保存因子计算结果"""
        # 生成数据哈希
        data_hash = self._compute_data_hash(compute_data) if compute_data is not None else ""
        
        # 生成参数哈希
        params_hash = self._compute_params_hash({})
        
        # 创建结果文件路径
        result_dir = os.path.join(self.storage_dir, "results", factor_id)
        ensure_directory(result_dir)
        
        timestamp = datetime.now().isoformat()
        result_filename = f"{timestamp.replace(':', '-')}_result.parquet"
        result_path = os.path.join(result_dir, result_filename)
        
        # 保存结果为Parquet格式
        compute_result.to_parquet(result_path)
        
        # 在数据库中记录结果
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO factor_results 
        (result_id, factor_id, timestamp, data_hash, params_hash, result_path, sample_size)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            hashlib.md5(f"{factor_id}:{timestamp}".encode()).hexdigest(),
            factor_id,
            timestamp,
            data_hash,
            params_hash,
            result_path,
            len(compute_result)
        ))
    
    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """计算数据的哈希值"""
        if data is None or data.empty:
            return ""
        
        # 使用数据的形状、索引范围和采样值计算哈希
        shape_str = str(data.shape)
        index_str = f"{data.index.min()}:{data.index.max()}"
        
        # 采样部分数据进行哈希
        sample_size = min(1000, len(data))
        sample = data.sample(sample_size).reset_index()
        sample_str = sample.to_json(orient="records")
        
        combined = f"{shape_str}:{index_str}:{sample_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _compute_params_hash(self, params: Dict) -> str:
        """计算参数的哈希值"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _cleanup_old_versions(self, factor_id: str):
        """清理旧版本历史，保持最大版本数限制"""
        if self.storage_format != "sqlite":
            return
        
        cursor = self.conn.cursor()
        
        # 查询并删除超出限制的旧版本
        cursor.execute('''
        DELETE FROM factor_versions
        WHERE version_id IN (
            SELECT version_id FROM factor_versions
            WHERE factor_id = ?
            ORDER BY version DESC
            LIMIT -1 OFFSET ?
        )
        ''', (factor_id, self.max_history_versions))
        
        self.conn.commit()
    
    def get_factor_metadata(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """
        获取因子元数据
        
        Args:
            factor_id: 因子ID
            
        Returns:
            因子元数据字典
        """
        if self.storage_format == "sqlite":
            return self._get_metadata_from_sqlite(factor_id)
        elif self.storage_format == "json":
            return self._get_metadata_from_json(factor_id)
        elif self.storage_format == "parquet":
            return self._get_metadata_from_parquet(factor_id)
        
        return None
    
    def _get_metadata_from_sqlite(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """从SQLite获取因子元数据"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM factor_metadata WHERE factor_id = ?", (factor_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # 转换为字典
        columns = [desc[0] for desc in cursor.description]
        metadata = dict(zip(columns, row))
        
        # 解析JSON字段
        for field in ["parameters", "dependencies", "performance_metrics", "validation_results"]:
            if field in metadata and metadata[field]:
                metadata[field] = json.loads(metadata[field])
        
        return metadata
    
    def _get_metadata_from_json(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """从JSON文件获取因子元数据"""
        metadata_path = os.path.join(self.storage_dir, "metadata", f"{factor_id}.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_metadata_from_parquet(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """从Parquet元数据文件获取因子元数据"""
        metadata_path = os.path.join(self.storage_dir, "parquet", f"{factor_id}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_factor_result(self, 
                         factor_id: str,
                         result_id: Optional[str] = None,
                         latest: bool = True) -> Optional[pd.DataFrame]:
        """
        获取因子计算结果
        
        Args:
            factor_id: 因子ID
            result_id: 结果ID（可选）
            latest: 是否获取最新结果
            
        Returns:
            因子计算结果DataFrame
        """
        if self.storage_format == "sqlite":
            return self._get_result_from_sqlite(factor_id, result_id, latest)
        elif self.storage_format == "json":
            return self._get_result_from_json(factor_id, result_id, latest)
        elif self.storage_format == "parquet":
            return self._get_result_from_parquet(factor_id, result_id, latest)
        
        return None
    
    def _get_result_from_sqlite(self, 
                               factor_id: str,
                               result_id: Optional[str],
                               latest: bool) -> Optional[pd.DataFrame]:
        """从SQLite获取因子结果"""
        cursor = self.conn.cursor()
        
        if result_id:
            # 按结果ID查询
            cursor.execute("SELECT result_path FROM factor_results WHERE result_id = ?", (result_id,))
        elif latest:
            # 获取最新结果
            cursor.execute(
                "SELECT result_path FROM factor_results WHERE factor_id = ? ORDER BY timestamp DESC LIMIT 1", 
                (factor_id,)
            )
        else:
            # 获取第一个结果
            cursor.execute(
                "SELECT result_path FROM factor_results WHERE factor_id = ? ORDER BY timestamp ASC LIMIT 1", 
                (factor_id,)
            )
        
        row = cursor.fetchone()
        if not row or not row[0] or not os.path.exists(row[0]):
            return None
        
        # 读取Parquet文件
        return pd.read_parquet(row[0])
    
    def _get_result_from_json(self, 
                             factor_id: str,
                             result_id: Optional[str],
                             latest: bool) -> Optional[pd.DataFrame]:
        """从JSON索引的结果文件获取因子结果"""
        result_dir = os.path.join(self.storage_dir, "results", factor_id)
        results_index_path = os.path.join(result_dir, "results_index.json")
        
        if not os.path.exists(results_index_path):
            return None
        
        with open(results_index_path, 'r', encoding='utf-8') as f:
            results_index = json.load(f)
        
        if not results_index:
            return None
        
        # 选择时间戳
        if latest:
            timestamps = sorted(results_index.keys(), reverse=True)
        else:
            timestamps = sorted(results_index.keys())
        
        if not timestamps:
            return None
        
        # 读取结果文件
        result_info = results_index[timestamps[0]]
        if os.path.exists(result_info["result_path"]):
            return pd.read_parquet(result_info["result_path"])
        
        return None
    
    def _get_result_from_parquet(self, 
                                factor_id: str,
                                result_id: Optional[str],
                                latest: bool) -> Optional[pd.DataFrame]:
        """从Parquet文件获取因子结果"""
        # 获取元数据以确定版本
        metadata = self._get_metadata_from_parquet(factor_id)
        if not metadata:
            return None
        
        version = metadata["version"] if latest else 1
        result_path = os.path.join(self.storage_dir, "parquet", f"{factor_id}_result_{version}.parquet")
        
        if os.path.exists(result_path):
            return pd.read_parquet(result_path)
        
        return None
    
    def search_factors(self, 
                      factor_name: Optional[str] = None,
                      factor_type: Optional[str] = None,
                      min_version: Optional[int] = None,
                      created_after: Optional[datetime] = None,
                      has_dependency: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索因子
        
        Args:
            factor_name: 因子名称（支持模糊匹配）
            factor_type: 因子类型
            min_version: 最小版本号
            created_after: 创建时间之后
            has_dependency: 包含指定依赖的因子
            
        Returns:
            匹配的因子元数据列表
        """
        if self.storage_format == "sqlite":
            return self._search_factors_in_sqlite(
                factor_name, factor_type, min_version, created_after, has_dependency
            )
        elif self.storage_format == "json":
            return self._search_factors_in_json(
                factor_name, factor_type, min_version, created_after, has_dependency
            )
        elif self.storage_format == "parquet":
            return self._search_factors_in_parquet(
                factor_name, factor_type, min_version, created_after, has_dependency
            )
        
        return []
    
    def _search_factors_in_sqlite(self, 
                                 factor_name: Optional[str],
                                 factor_type: Optional[str],
                                 min_version: Optional[int],
                                 created_after: Optional[datetime],
                                 has_dependency: Optional[str]) -> List[Dict[str, Any]]:
        """在SQLite中搜索因子"""
        cursor = self.conn.cursor()
        
        # 构建查询
        query = "SELECT * FROM factor_metadata WHERE 1=1"
        params = []
        
        if factor_name:
            query += " AND factor_name LIKE ?"
            params.append(f"%{factor_name}%")
        
        if factor_type:
            query += " AND factor_type = ?"
            params.append(factor_type)
        
        if min_version is not None:
            query += " AND version >= ?"
            params.append(min_version)
        
        if created_after:
            query += " AND created_at >= ?"
            params.append(created_after.isoformat())
        
        # 执行查询
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 处理结果
        results = []
        columns = [desc[0] for desc in cursor.description]
        
        for row in rows:
            factor = dict(zip(columns, row))
            
            # 解析JSON字段
            for field in ["parameters", "dependencies", "performance_metrics", "validation_results"]:
                if field in factor and factor[field]:
                    factor[field] = json.loads(factor[field])
            
            # 检查依赖条件
            if has_dependency:
                dependencies = factor.get("dependencies", [])
                if has_dependency not in dependencies:
                    continue
            
            results.append(factor)
        
        return results
    
    def _search_factors_in_json(self, 
                               factor_name: Optional[str],
                               factor_type: Optional[str],
                               min_version: Optional[int],
                               created_after: Optional[datetime],
                               has_dependency: Optional[str]) -> List[Dict[str, Any]]:
        """在JSON文件中搜索因子"""
        results = []
        metadata_dir = os.path.join(self.storage_dir, "metadata")
        
        if not os.path.exists(metadata_dir):
            return results
        
        # 遍历所有元数据文件
        for filename in os.listdir(metadata_dir):
            if not filename.endswith(".json"):
                continue
            
            file_path = os.path.join(metadata_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    factor = json.load(f)
                
                # 应用过滤条件
                if factor_name and factor_name.lower() not in factor["factor_name"].lower():
                    continue
                
                if factor_type and factor_type != factor.get("factor_type"):
                    continue
                
                if min_version is not None and factor.get("version", 1) < min_version:
                    continue
                
                if created_after:
                    created = datetime.fromisoformat(factor.get("created_at", ""))
                    if created <= created_after:
                        continue
                
                if has_dependency:
                    dependencies = factor.get("dependencies", [])
                    if has_dependency not in dependencies:
                        continue
                
                results.append(factor)
            except Exception as e:
                self.logger.error(f"读取因子元数据文件 {filename} 时出错: {e}")
        
        return results
    
    def _search_factors_in_parquet(self, 
                                  factor_name: Optional[str],
                                  factor_type: Optional[str],
                                  min_version: Optional[int],
                                  created_after: Optional[datetime],
                                  has_dependency: Optional[str]) -> List[Dict[str, Any]]:
        """在Parquet元数据中搜索因子"""
        results = []
        parquet_dir = os.path.join(self.storage_dir, "parquet")
        
        if not os.path.exists(parquet_dir):
            return results
        
        # 遍历所有元数据文件
        for filename in os.listdir(parquet_dir):
            if not filename.endswith("_metadata.json"):
                continue
            
            file_path = os.path.join(parquet_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    factor = json.load(f)
                
                # 应用过滤条件
                if factor_name and factor_name.lower() not in factor["factor_name"].lower():
                    continue
                
                if factor_type and factor_type != factor.get("factor_type"):
                    continue
                
                if min_version is not None and factor.get("version", 1) < min_version:
                    continue
                
                if created_after:
                    created = datetime.fromisoformat(factor.get("created_at", ""))
                    if created <= created_after:
                        continue
                
                if has_dependency:
                    dependencies = factor.get("dependencies", [])
                    if has_dependency not in dependencies:
                        continue
                
                results.append(factor)
            except Exception as e:
                self.logger.error(f"读取因子元数据文件 {filename} 时出错: {e}")
        
        return results
    
    def delete_factor(self, factor_id: str, delete_results: bool = True) -> bool:
        """
        删除因子及其所有相关数据
        
        Args:
            factor_id: 因子ID
            delete_results: 是否同时删除计算结果
            
        Returns:
            是否删除成功
        """
        try:
            if self.storage_format == "sqlite":
                return self._delete_factor_from_sqlite(factor_id, delete_results)
            elif self.storage_format == "json":
                return self._delete_factor_from_json(factor_id, delete_results)
            elif self.storage_format == "parquet":
                return self._delete_factor_from_parquet(factor_id, delete_results)
        except Exception as e:
            self.logger.error(f"删除因子 {factor_id} 时出错: {e}")
            return False
        
        return False
    
    def _delete_factor_from_sqlite(self, factor_id: str, delete_results: bool) -> bool:
        """从SQLite中删除因子"""
        cursor = self.conn.cursor()
        
        # 开始事务
        cursor.execute("BEGIN TRANSACTION")
        
        try:
            # 获取所有结果文件路径
            result_paths = []
            if delete_results:
                cursor.execute("SELECT result_path FROM factor_results WHERE factor_id = ?", (factor_id,))
                result_paths = [row[0] for row in cursor.fetchall()]
            
            # 删除关系数据
            cursor.execute("DELETE FROM factor_relationships WHERE factor_id = ? OR dependency_id = ?", 
                          (factor_id, factor_id))
            
            # 删除版本历史
            cursor.execute("DELETE FROM factor_versions WHERE factor_id = ?", (factor_id,))
            
            # 删除结果记录
            if delete_results:
                cursor.execute("DELETE FROM factor_results WHERE factor_id = ?", (factor_id,))
            
            # 删除因子元数据
            cursor.execute("DELETE FROM factor_metadata WHERE factor_id = ?", (factor_id,))
            
            # 提交事务
            self.conn.commit()
            
            # 删除实际文件
            if delete_results:
                for path in result_paths:
                    if os.path.exists(path):
                        os.remove(path)
            
            return True
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"删除因子 {factor_id} 时出错: {e}")
            return False
    
    def _delete_factor_from_json(self, factor_id: str, delete_results: bool) -> bool:
        """从JSON存储中删除因子"""
        try:
            # 删除元数据文件
            metadata_path = os.path.join(self.storage_dir, "metadata", f"{factor_id}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # 删除历史版本
            version_dir = os.path.join(self.storage_dir, "versions", factor_id)
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            
            # 删除结果文件
            if delete_results:
                result_dir = os.path.join(self.storage_dir, "results", factor_id)
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
            
            return True
        except Exception as e:
            self.logger.error(f"删除因子 {factor_id} 时出错: {e}")
            return False
    
    def _delete_factor_from_parquet(self, factor_id: str, delete_results: bool) -> bool:
        """从Parquet存储中删除因子"""
        try:
            # 删除元数据文件
            metadata_path = os.path.join(self.storage_dir, "parquet", f"{factor_id}_metadata.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # 删除结果文件
            if delete_results:
                parquet_dir = os.path.join(self.storage_dir, "parquet")
                for filename in os.listdir(parquet_dir):
                    if filename.startswith(f"{factor_id}_result_"):
                        os.remove(os.path.join(parquet_dir, filename))
            
            return True
        except Exception as e:
            self.logger.error(f"删除因子 {factor_id} 时出错: {e}")
            return False
    
    def get_all_factors(self) -> List[Dict[str, Any]]:
        """
        获取所有因子的元数据
        
        Returns:
            所有因子的元数据列表
        """
        return self.search_factors()
    
    def export_factor(self, 
                     factor_id: str,
                     export_dir: str,
                     include_results: bool = True) -> str:
        """
        导出因子及其数据
        
        Args:
            factor_id: 因子ID
            export_dir: 导出目录
            include_results: 是否包含计算结果
            
        Returns:
            导出文件路径
        """
        # 确保导出目录存在
        ensure_directory(export_dir)
        
        # 获取元数据
        metadata = self.get_factor_metadata(factor_id)
        if not metadata:
            raise ValueError(f"因子 {factor_id} 不存在")
        
        # 导出文件名
        export_filename = f"factor_{metadata['factor_name']}_{factor_id}.json"
        export_path = os.path.join(export_dir, export_filename)
        
        # 准备导出数据
        export_data = {
            "metadata": metadata,
            "results": None
        }
        
        # 包含计算结果
        if include_results:
            result = self.get_factor_result(factor_id, latest=True)
            if result is not None:
                # 将DataFrame转换为可序列化格式
                export_data["results"] = {
                    "data": result.to_dict(orient="records"),
                    "index": result.index.tolist(),
                    "columns": result.columns.tolist()
                }
        
        # 保存导出文件
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        return export_path
    
    def import_factor(self, import_path: str) -> str:
        """
        从导出文件导入因子
        
        Args:
            import_path: 导入文件路径
            
        Returns:
            导入的因子ID
        """
        if not os.path.exists(import_path):
            raise ValueError(f"导入文件 {import_path} 不存在")
        
        # 读取导入文件
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        # 获取元数据
        metadata = import_data.get("metadata")
        if not metadata:
            raise ValueError("导入文件缺少元数据")
        
        # 从元数据中恢复因子ID
        factor_id = metadata.get("factor_id")
        
        # 保存因子（不包含计算结果的情况下）
        if self.storage_format == "sqlite":
            cursor = self.conn.cursor()
            
            # 检查是否已存在
            cursor.execute("SELECT * FROM factor_metadata WHERE factor_id = ?", (factor_id,))
            if cursor.fetchone():
                # 更新版本号
                metadata["version"] += 1
                metadata["updated_at"] = datetime.now().isoformat()
        
        # 保存元数据
        if self.storage_format == "sqlite":
            # 使用SQLite插入或更新逻辑
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO factor_metadata 
            (factor_id, factor_name, factor_type, description, class_name, module_path, 
             created_at, updated_at, version, parameters, dependencies, 
             performance_metrics, validation_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                factor_id, metadata["factor_name"], metadata["factor_type"], 
                metadata["description"], metadata["class_name"], metadata["module_path"],
                metadata["created_at"], metadata["updated_at"], metadata["version"],
                json.dumps(metadata["parameters"]), json.dumps(metadata["dependencies"]),
                json.dumps(metadata["performance_metrics"]), json.dumps(metadata["validation_results"])
            ))
            self.conn.commit()
        elif self.storage_format == "json":
            # 使用JSON保存逻辑
            metadata_path = os.path.join(self.storage_dir, "metadata", f"{factor_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        elif self.storage_format == "parquet":
            # 使用Parquet元数据保存逻辑
            metadata_path = os.path.join(self.storage_dir, "parquet", f"{factor_id}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 导入计算结果
        results_data = import_data.get("results")
        if results_data and results_data.get("data"):
            # 重建DataFrame
            result_df = pd.DataFrame(
                results_data["data"],
                index=results_data["index"],
                columns=results_data["columns"]
            )
            
            # 保存结果
            if self.storage_format == "sqlite":
                self._save_factor_result(factor_id, None, result_df)
            elif self.storage_format == "json":
                result_dir = os.path.join(self.storage_dir, "results", factor_id)
                ensure_directory(result_dir)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_path = os.path.join(result_dir, f"imported_result_{timestamp}.parquet")
                result_df.to_parquet(result_path)
            elif self.storage_format == "parquet":
                result_path = os.path.join(self.storage_dir, "parquet", f"{factor_id}_result_{metadata['version']}.parquet")
                result_df.to_parquet(result_path)
        
        return factor_id
    
    def close(self):
        """关闭持久化管理器，释放资源"""
        if self.storage_format == "sqlite" and hasattr(self, 'conn'):
            self.conn.close()
        
        self.logger.info("因子持久化管理器已关闭")
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.close()
