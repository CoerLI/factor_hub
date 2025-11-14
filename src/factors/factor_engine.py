import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
import os
import threading
import concurrent.futures
import time
from datetime import datetime
import gc

from src.factors.factor_base import FactorBase, FactorContainer, FactorResult
from src.utils.helpers import (
    setup_logger,
    cache_result,
    PerformanceTracker,
    load_config,
    ensure_directory,
    DEFAULT_LOGGER
)


class FactorEngine:
    """
    因子计算引擎，负责因子的注册、计算、缓存和依赖管理
    
    设计理念：
    1. 中心化管理：统一管理所有因子的生命周期
    2. 依赖解析：自动解析和计算因子间的依赖关系
    3. 高效缓存：智能缓存机制减少重复计算
    4. 并行计算：支持多线程/多进程并行计算提高效率
    5. 内存管理：合理管理内存使用，避免内存泄漏
    6. 计算优化：提供多种计算优化策略
    """
    
    def __init__(self, 
                 config_path: str = "config/config.json",
                 use_cache: bool = True,
                 cache_dir: str = ".cache/factors",
                 parallel: bool = True,
                 max_workers: Optional[int] = None):
        """
        初始化因子计算引擎
        
        Args:
            config_path: 配置文件路径
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            parallel: 是否使用并行计算
            max_workers: 最大工作线程数，None表示使用默认值
        """
        # 加载配置
        self.config = load_config(config_path)
        
        # 初始化属性
        self.use_cache = use_cache if use_cache is not None else self.config["factors"]["cache_enabled"]
        self.cache_dir = cache_dir
        self.parallel = parallel if parallel is not None else self.config["factors"]["parallel_computation"]
        self.max_workers = max_workers if max_workers is not None else self.config["factors"]["max_workers"]
        
        # 因子注册字典
        self._factors: Dict[str, FactorBase] = {}
        
        # 因子依赖图
        self._dependencies: Dict[str, Set[str]] = {}
        
        # 计算锁，防止并发问题
        self._lock = threading.RLock()
        
        # 确保缓存目录存在
        if self.use_cache:
            ensure_directory(self.cache_dir)
        
        # 设置日志
        self.logger = setup_logger(
            name="factor_engine",
            log_file=os.path.join("logs", "factor_engine.log"),
            level=DEFAULT_LOGGER.level
        )
        
        # 初始化性能统计
        self._stats = {
            "factors_loaded": 0,
            "factors_computed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "computation_time": 0.0
        }
        
        self.logger.info(f"因子计算引擎初始化完成，并行计算: {self.parallel}, 缓存: {self.use_cache}")
    
    def register_factor(self, factor: FactorBase) -> "FactorEngine":
        """
        注册因子到引擎
        
        Args:
            factor: 因子实例
            
        Returns:
            引擎实例，支持链式调用
        """
        with self._lock:
            factor_name = factor.name
            
            # 检查因子名称是否已存在
            if factor_name in self._factors:
                self.logger.warning(f"因子 '{factor_name}' 已存在，将被覆盖")
            
            # 注册因子
            self._factors[factor_name] = factor
            
            # 解析依赖关系
            self._dependencies[factor_name] = set(factor.dependencies)
            
            # 更新统计信息
            self._stats["factors_loaded"] += 1
            
            self.logger.info(f"已注册因子: {factor_name}, 依赖因子: {factor.dependencies}")
        
        return self
    
    def register_factors(self, factors: List[FactorBase]) -> "FactorEngine":
        """
        批量注册因子
        
        Args:
            factors: 因子实例列表
            
        Returns:
            引擎实例，支持链式调用
        """
        for factor in factors:
            self.register_factor(factor)
        
        return self
    
    def get_factor(self, factor_name: str) -> Optional[FactorBase]:
        """
        获取因子实例
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子实例或None
        """
        return self._factors.get(factor_name)
    
    def get_all_factors(self) -> Dict[str, FactorBase]:
        """
        获取所有已注册的因子
        
        Returns:
            因子字典
        """
        return self._factors.copy()
    
    def _validate_dependencies(self, factor_name: str) -> bool:
        """
        验证因子的依赖关系
        
        Args:
            factor_name: 因子名称
            
        Returns:
            依赖关系是否有效
        """
        visited = set()
        path = set()
        
        def has_cycle(node: str) -> bool:
            """检测是否存在循环依赖"""
            visited.add(node)
            path.add(node)
            
            # 检查所有依赖
            if node in self._dependencies:
                for dependency in self._dependencies[node]:
                    # 检查依赖是否已注册
                    if dependency not in self._factors:
                        self.logger.error(f"因子 '{node}' 依赖未注册的因子: '{dependency}'")
                        return False
                    
                    # 检测循环依赖
                    if dependency in path:
                        self.logger.error(f"检测到循环依赖: {path} -> {dependency}")
                        return False
                    
                    # 递归检查
                    if dependency not in visited and not has_cycle(dependency):
                        return False
            
            # 从路径中移除当前节点
            path.remove(node)
            return True
        
        # 验证依赖存在性和循环依赖
        if factor_name not in self._factors:
            self.logger.error(f"因子 '{factor_name}' 未注册")
            return False
        
        # 检查依赖的依赖
        if factor_name in self._dependencies:
            for dependency in self._dependencies[factor_name]:
                if dependency not in self._factors:
                    self.logger.error(f"因子 '{factor_name}' 依赖未注册的因子: '{dependency}'")
                    return False
        
        # 检查循环依赖
        return has_cycle(factor_name)
    
    def _get_computation_order(self, factor_names: List[str]) -> List[str]:
        """
        获取因子的计算顺序（拓扑排序）
        
        Args:
            factor_names: 要计算的因子名称列表
            
        Returns:
            按计算顺序排序的因子名称列表
        """
        # 确保所有因子都已注册
        for name in factor_names:
            if name not in self._factors:
                raise ValueError(f"因子 '{name}' 未注册")
            
            # 验证依赖关系
            if not self._validate_dependencies(name):
                raise ValueError(f"因子 '{name}' 的依赖关系无效")
        
        # 构建依赖图
        graph = {}
        in_degree = {}
        
        # 初始化图
        for name in self._factors:
            graph[name] = set()
            in_degree[name] = 0
        
        # 添加边
        for name, dependencies in self._dependencies.items():
            for dep in dependencies:
                graph[dep].add(name)  # 依赖 -> 当前因子
                in_degree[name] += 1
        
        # 拓扑排序
        result = []
        queue = [name for name in self._factors if in_degree[name] == 0]
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # 更新相邻节点的入度
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 过滤出需要计算的因子及其依赖
        needed_factors = set()
        
        def collect_dependencies(factor_name: str):
            """递归收集所有依赖因子"""
            if factor_name not in needed_factors:
                needed_factors.add(factor_name)
                if factor_name in self._dependencies:
                    for dep in self._dependencies[factor_name]:
                        collect_dependencies(dep)
        
        # 收集所有需要的因子
        for name in factor_names:
            collect_dependencies(name)
        
        # 过滤出需要的因子，并保持计算顺序
        filtered_result = [name for name in result if name in needed_factors]
        
        return filtered_result
    
    def _compute_factor(self, 
                       factor: FactorBase,
                       data: pd.DataFrame,
                       computed_factors: Dict[str, FactorResult],
                       params: Optional[Dict[str, Any]] = None) -> FactorResult:
        """
        计算单个因子
        
        Args:
            factor: 因子实例
            data: 原始数据
            computed_factors: 已计算的因子结果
            params: 计算参数
            
        Returns:
            因子计算结果
        """
        # 准备依赖因子结果
        dependencies_results = {}
        for dep_name in factor.dependencies:
            if dep_name in computed_factors:
                dependencies_results[dep_name] = computed_factors[dep_name]
            else:
                raise ValueError(f"依赖因子 '{dep_name}' 未计算")
        
        # 生成缓存键
        cache_key = self._generate_cache_key(factor.name, data, params)
        
        # 尝试从缓存获取结果
        if self.use_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self._stats["cache_hits"] += 1
                self.logger.debug(f"从缓存加载因子 '{factor.name}'")
                return cached_result
        
        # 缓存未命中，执行计算
        self._stats["cache_misses"] += 1
        start_time = time.time()
        
        try:
            self.logger.debug(f"开始计算因子 '{factor.name}'")
            result = factor.compute(data, dependencies_results, params)
            
            # 计算耗时
            computation_time = time.time() - start_time
            self._stats["computation_time"] += computation_time
            self._stats["factors_computed"] += 1
            
            self.logger.info(f"因子 '{factor.name}' 计算完成，耗时: {computation_time:.4f}秒")
            
            # 保存到缓存
            if self.use_cache:
                self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算因子 '{factor.name}' 时出错: {str(e)}")
            raise
    
    def _generate_cache_key(self, 
                           factor_name: str,
                           data: pd.DataFrame,
                           params: Optional[Dict[str, Any]] = None) -> str:
        """
        生成缓存键
        
        Args:
            factor_name: 因子名称
            data: 原始数据
            params: 计算参数
            
        Returns:
            缓存键字符串
        """
        # 使用数据的摘要信息作为缓存键的一部分
        data_info = {
            "shape": str(data.shape),
            "index_min": str(data.index.min()),
            "index_max": str(data.index.max())
        }
        
        # 添加参数信息
        key_components = {
            "factor_name": factor_name,
            "data_info": data_info,
            "params": params or {},
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # 生成哈希值
        import hashlib
        import json
        key_str = json.dumps(key_components, sort_keys=True, default=str)
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[FactorResult]:
        """
        从缓存获取因子计算结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            因子结果或None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            
            # 验证缓存是否过期
            ttl = self.config["factors"].get("cache_ttl_seconds", 3600)
            file_mtime = os.path.getmtime(cache_file)
            if time.time() - file_mtime > ttl:
                self.logger.debug(f"缓存 '{cache_key}' 已过期")
                os.remove(cache_file)
                return None
            
            return result
            
        except Exception as e:
            self.logger.warning(f"读取缓存时出错: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: FactorResult) -> bool:
        """
        保存因子计算结果到缓存
        
        Args:
            cache_key: 缓存键
            result: 因子计算结果
            
        Returns:
            是否保存成功
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.debug(f"因子结果已保存到缓存: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.warning(f"保存缓存时出错: {str(e)}")
            return False
    
    def compute_factors(self, 
                       data: pd.DataFrame,
                       factor_names: Optional[List[str]] = None,
                       params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, FactorResult]:
        """
        计算指定的因子
        
        Args:
            data: 原始数据
            factor_names: 要计算的因子名称列表，None表示计算所有因子
            params: 因子计算参数字典，格式为 {"factor_name": {"param1": value1, ...}}
            
        Returns:
            因子计算结果字典
        """
        # 如果没有指定因子，计算所有注册的因子
        if factor_names is None:
            factor_names = list(self._factors.keys())
        
        # 确保参数字典存在
        if params is None:
            params = {}
        
        with PerformanceTracker("因子计算引擎.compute_factors"):
            # 获取计算顺序
            computation_order = self._get_computation_order(factor_names)
            
            self.logger.info(f"因子计算顺序: {computation_order}")
            
            # 存储已计算的因子结果
            computed_results: Dict[str, FactorResult] = {}
            
            # 单线程计算（包含依赖关系）
            if not self.parallel:
                for factor_name in computation_order:
                    factor = self._factors[factor_name]
                    factor_params = params.get(factor_name, {})
                    computed_results[factor_name] = self._compute_factor(
                        factor=factor,
                        data=data,
                        computed_factors=computed_results,
                        params=factor_params
                    )
            else:
                # 并行计算 - 先计算没有依赖的因子
                # 注意：有依赖关系的因子不能并行计算
                # 这里简化处理，按拓扑排序分组并行计算
                groups = self._group_factors_by_level(computation_order)
                
                for i, group in enumerate(groups):
                    self.logger.info(f"并行计算组 {i+1}/{len(groups)}: {group}")
                    
                    group_results = {}
                    
                    # 使用线程池并行计算
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # 提交任务
                        future_to_factor = {}
                        for factor_name in group:
                            factor = self._factors[factor_name]
                            factor_params = params.get(factor_name, {})
                            
                            # 提交计算任务
                            future = executor.submit(
                                self._compute_factor,
                                factor=factor,
                                data=data,
                                computed_factors={**computed_results, **group_results},  # 已计算的因子
                                params=factor_params
                            )
                            future_to_factor[future] = factor_name
                        
                        # 收集结果
                        for future in concurrent.futures.as_completed(future_to_factor):
                            factor_name = future_to_factor[future]
                            try:
                                result = future.result()
                                group_results[factor_name] = result
                            except Exception as e:
                                self.logger.error(f"并行计算因子 '{factor_name}' 时出错: {str(e)}")
                                raise
                    
                    # 将当前组的结果添加到已计算结果中
                    computed_results.update(group_results)
            
            # 清理内存
            gc.collect()
            
            # 记录性能统计
            self.logger.info(f"因子计算完成 - 总耗时: {self._stats['computation_time']:.4f}秒")
            self.logger.info(f"缓存命中: {self._stats['cache_hits']}, 缓存未命中: {self._stats['cache_misses']}")
            
            # 返回指定因子的结果
            return {name: computed_results[name] for name in factor_names if name in computed_results}
    
    def _group_factors_by_level(self, computation_order: List[str]) -> List[List[str]]:
        """
        按依赖级别分组因子，同一级别内的因子可以并行计算
        
        Args:
            computation_order: 拓扑排序后的因子列表
            
        Returns:
            分组后的因子列表
        """
        # 计算每个因子的依赖级别
        levels = {}
        
        for factor_name in computation_order:
            # 找到所有依赖因子的最大级别
            max_level = -1
            for dep in self._dependencies.get(factor_name, []):
                if dep in levels:
                    max_level = max(max_level, levels[dep])
            
            # 当前因子的级别是最大依赖级别 + 1
            levels[factor_name] = max_level + 1
        
        # 按级别分组
        groups = {}
        for factor_name, level in levels.items():
            if level not in groups:
                groups[level] = []
            groups[level].append(factor_name)
        
        # 返回有序的分组
        return [groups[level] for level in sorted(groups.keys())]
    
    def create_factor_container(self, 
                               data: pd.DataFrame,
                               factor_names: Optional[List[str]] = None,
                               params: Optional[Dict[str, Dict[str, Any]]] = None,
                               persist_results: bool = True) -> FactorContainer:
        """
        创建因子容器，包含指定的因子计算结果
        
        Args:
            data: 原始数据
            factor_names: 因子名称列表
            params: 计算参数
            persist_results: 是否持久化计算结果
            
        Returns:
            因子容器实例
        """
        # 计算因子
        factor_results = self.compute_factors(data, factor_names, params, persist_results)
        
        # 创建容器
        container = FactorContainer()
        
        # 添加因子结果
        for name, result in factor_results.items():
            container.add_factor(name, result)
        
        # 持久化容器信息
        if self.persistence_enabled and persist_results:
            try:
                # 为容器生成唯一ID
                import uuid
                container_id = str(uuid.uuid4())
                
                container_metadata = {
                    "container_id": container_id,
                    "factor_names": list(factor_results.keys()),
                    "created_at": datetime.now().isoformat(),
                    "data_shape": data.shape,
                    "result_shapes": {fid: fr.result.shape for fid, fr in factor_results.items() if hasattr(fr, 'result')}
                }
                self.persistence_manager.save_container_info(container_id, container_metadata)
                self.logger.info(f"已持久化容器 '{container_id}' 信息")
            except Exception as e:
                self.logger.warning(f"持久化容器信息失败: {str(e)}")
        
        self.logger.info(f"已创建因子容器，包含 {len(factor_results)} 个因子")
        
        return container
    
    def clear_cache(self) -> None:
        """
        清除所有因子缓存
        """
        if not os.path.exists(self.cache_dir):
            return
        
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            self.logger.info(f"已清除因子缓存: {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"清除缓存时出错: {str(e)}")
    
    def reset(self) -> None:
        """
        重置引擎状态
        """
        with self._lock:
            self._factors.clear()
            self._dependencies.clear()
            
            # 重置统计信息
            self._stats = {
                "factors_loaded": 0,
                "factors_computed": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "computation_time": 0.0
            }
            
            self.logger.info("因子引擎已重置")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        stats = self._stats.copy()
        
        # 计算缓存命中率
        total = stats["cache_hits"] + stats["cache_misses"]
        stats["cache_hit_rate"] = stats["cache_hits"] / total if total > 0 else 0
        
        # 计算平均计算时间
        if stats["factors_computed"] > 0:
            stats["avg_computation_time"] = stats["computation_time"] / stats["factors_computed"]
        else:
            stats["avg_computation_time"] = 0
        
        return stats
    
    def optimize_memory_usage(self) -> None:
        """
        优化内存使用
        """
        # 执行垃圾回收
        gc.collect()
        
        # 检查内存使用
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            self.logger.info(f"当前内存使用: {memory_mb:.2f} MB")
            
            # 如果内存使用超过配置的限制，清理缓存
            max_memory = self.config["factors"].get("max_factor_memory_mb", 512)
            if memory_mb > max_memory:
                self.logger.warning(f"内存使用超过限制 ({max_memory} MB)，清理缓存")
                self.clear_cache()
                
        except ImportError:
            self.logger.warning("psutil 未安装，无法监控内存使用")
    
    def __str__(self) -> str:
        """
        返回引擎的字符串表示
        """
        return (
            f"因子计算引擎 ("  
            f"因子数量: {len(self._factors)}, "
            f"并行计算: {self.parallel}, "
            f"缓存: {self.use_cache}, "
            f"持久化: {self.persistence_enabled}"
            f")"
        )
    
    def load_factor_from_storage(self, factor_name: str) -> Optional[Dict]:
        """
        从存储中加载因子数据
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子数据字典，如果不存在则返回None
        """
        if not self.persistence_enabled:
            self.logger.warning("持久化功能已禁用，无法从存储中加载因子")
            return None
        
        try:
            factor_data = self.persistence_manager.get_factor_by_id(factor_name)
            if factor_data:
                self.logger.info(f"已从存储加载因子 '{factor_name}'")
            return factor_data
        except Exception as e:
            self.logger.error(f"从存储加载因子 '{factor_name}' 失败: {str(e)}")
            return None
    
    def get_all_persisted_factors(self) -> List[Dict]:
        """
        获取所有已持久化的因子列表
        
        Returns:
            因子列表，每个元素包含因子ID和基本信息
        """
        if not self.persistence_enabled:
            self.logger.warning("持久化功能已禁用，无法列出已持久化的因子")
            return []
        
        try:
            factors = self.persistence_manager.list_factors(include_metadata=True)
            self.logger.info(f"已获取 {len(factors)} 个已持久化的因子")
            return factors
        except Exception as e:
            self.logger.error(f"列出已持久化因子失败: {str(e)}")
            return []
    
    def update_factor_config(self, config_updates: Dict) -> None:
        """
        更新因子引擎配置
        
        Args:
            config_updates: 配置更新字典
        """
        # 更新配置
        with self._lock:
            for key, value in config_updates.items():
                if key == "persistence_enabled":
                    self.persistence_enabled = value
                elif key == "storage_dir":
                    self.storage_dir = value
                    ensure_directory(self.storage_dir)
                    # 创建新的持久化管理器
                    self.persistence_manager = FactorPersistenceManager(
                        storage_dir=self.storage_dir,
                        default_format="sqlite"
                    )
        
        self.logger.info(f"已更新因子引擎配置: {list(config_updates.keys())}")
