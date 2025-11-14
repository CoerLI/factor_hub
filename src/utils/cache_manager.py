import os
import pickle
import hashlib
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union, Set, List
import gc
import shutil
from datetime import datetime, timedelta
import logging

from src.utils.helpers import ensure_directory, setup_logger, DEFAULT_LOGGER


class CacheItem:
    """
    缓存项类，用于包装缓存的数据及其元数据
    """
    
    def __init__(self, 
                 data: Any,
                 key: str,
                 created_at: float,
                 ttl: Optional[float] = None):
        """
        初始化缓存项
        
        Args:
            data: 缓存的数据
            key: 缓存键
            created_at: 创建时间戳
            ttl: 生存时间（秒）
        """
        self.data = data
        self.key = key
        self.created_at = created_at
        self.ttl = ttl
        self.last_accessed = created_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """
        检查缓存项是否已过期
        
        Returns:
            是否过期
        """
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """
        访问缓存项，更新访问时间和计数
        
        Returns:
            缓存的数据
        """
        self.last_accessed = time.time()
        self.access_count += 1
        return self.data
    
    def get_size(self) -> int:
        """
        估算缓存项的大小（字节）
        
        Returns:
            估计的大小（字节）
        """
        # 这是一个简化的估计，实际大小可能不同
        try:
            # 使用pickle序列化来估算大小
            serialized = pickle.dumps(self.data)
            return len(serialized)
        except Exception:
            # 如果无法序列化，返回一个默认值
            return 1024  # 假设1KB
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将缓存项转换为字典表示
        
        Returns:
            缓存项的字典表示
        """
        return {
            "key": self.key,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "CacheItem":
        """
        从字典创建缓存项
        
        Args:
            data_dict: 缓存项的字典表示
            
        Returns:
            缓存项实例
        """
        item = cls(
            data=data_dict["data"],
            key=data_dict["key"],
            created_at=data_dict["created_at"],
            ttl=data_dict.get("ttl")
        )
        item.last_accessed = data_dict.get("last_accessed", item.created_at)
        item.access_count = data_dict.get("access_count", 0)
        return item


class MemoryCache:
    """
    内存缓存实现，支持LRU淘汰策略和过期时间
    """
    
    def __init__(self, 
                 max_size_bytes: int = 100 * 1024 * 1024,  # 默认100MB
                 max_items: int = 1000,
                 default_ttl: Optional[float] = None):
        """
        初始化内存缓存
        
        Args:
            max_size_bytes: 最大内存使用量（字节）
            max_items: 最大缓存项数量
            default_ttl: 默认生存时间（秒）
        """
        self.max_size_bytes = max_size_bytes
        self.max_items = max_items
        self.default_ttl = default_ttl
        
        # 存储缓存项的字典
        self._cache: Dict[str, CacheItem] = {}
        
        # 当前内存使用量
        self._current_size_bytes = 0
        
        # 访问锁
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "sets": 0,
            "gets": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据或None
        """
        with self._lock:
            self._stats["gets"] += 1
            
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            item = self._cache[key]
            
            # 检查是否过期
            if item.is_expired():
                self._remove_item(key)
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None
            
            # 更新访问信息
            data = item.access()
            self._stats["hits"] += 1
            
            return data
    
    def set(self, 
            key: str, 
            value: Any,
            ttl: Optional[float] = None) -> bool:
        """
        设置缓存项
        
        Args:
            key: 缓存键
            value: 要缓存的数据
            ttl: 生存时间（秒），None表示使用默认值
            
        Returns:
            是否设置成功
        """
        with self._lock:
            self._stats["sets"] += 1
            
            # 确定TTL
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            # 创建缓存项
            new_item = CacheItem(
                data=value,
                key=key,
                created_at=time.time(),
                ttl=effective_ttl
            )
            
            # 估算新项的大小
            new_size = new_item.get_size()
            
            # 如果键已存在，减去旧项的大小
            if key in self._cache:
                old_item = self._cache[key]
                self._current_size_bytes -= old_item.get_size()
            
            # 检查是否需要清理空间
            self._ensure_space(new_size)
            
            # 添加新项
            self._cache[key] = new_item
            self._current_size_bytes += new_size
            
            return True
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        with self._lock:
            return self._remove_item(key)
    
    def _remove_item(self, key: str) -> bool:
        """
        移除缓存项（内部方法）
        
        Args:
            key: 缓存键
            
        Returns:
            是否移除成功
        """
        if key in self._cache:
            item = self._cache[key]
            self._current_size_bytes -= item.get_size()
            del self._cache[key]
            return True
        return False
    
    def _ensure_space(self, required_space: int) -> None:
        """
        确保有足够的空间存储新项
        
        Args:
            required_space: 需要的空间（字节）
        """
        # 检查项目数量限制
        while len(self._cache) >= self.max_items:
            self._evict_one()
        
        # 检查大小限制
        while self._current_size_bytes + required_space > self.max_size_bytes:
            self._evict_one()
    
    def _evict_one(self) -> None:
        """
        使用LRU策略淘汰一个缓存项
        """
        if not self._cache:
            return
        
        # 找到最久未使用的项
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        self._remove_item(lru_key)
        self._stats["evictions"] += 1
    
    def clear(self) -> None:
        """
        清空缓存
        """
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
    
    def contains(self, key: str) -> bool:
        """
        检查键是否在缓存中
        
        Args:
            key: 缓存键
            
        Returns:
            键是否存在
        """
        with self._lock:
            return key in self._cache
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = self._stats.copy()
            
            # 计算命中率
            total = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total if total > 0 else 0
            
            # 添加其他统计信息
            stats["items_count"] = len(self._cache)
            stats["memory_usage_bytes"] = self._current_size_bytes
            
            return stats
    
    def cleanup(self) -> int:
        """
        清理过期的缓存项
        
        Returns:
            清理的项目数量
        """
        with self._lock:
            expired_keys = [key for key, item in self._cache.items() if item.is_expired()]
            
            for key in expired_keys:
                self._remove_item(key)
                self._stats["expirations"] += 1
            
            return len(expired_keys)
    
    def __len__(self) -> int:
        """
        返回缓存中的项目数量
        """
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """
        检查键是否在缓存中
        """
        return self.contains(key)


class DiskCache:
    """
    磁盘缓存实现
    """
    
    def __init__(self, 
                 cache_dir: str = ".cache",
                 default_ttl: Optional[float] = None):
        """
        初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录
            default_ttl: 默认生存时间（秒）
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        
        # 确保缓存目录存在
        ensure_directory(cache_dir)
        
        # 访问锁
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "gets": 0,
            "expirations": 0
        }
    
    def _get_cache_file_path(self, key: str) -> str:
        """
        获取缓存文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            文件路径
        """
        # 使用哈希值作为文件名，避免特殊字符
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pkl")
    
    def _get_metadata_file_path(self, key: str) -> str:
        """
        获取元数据文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            文件路径
        """
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.meta")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据或None
        """
        with self._lock:
            self._stats["gets"] += 1
            
            cache_file = self._get_cache_file_path(key)
            meta_file = self._get_metadata_file_path(key)
            
            # 检查文件是否存在
            if not os.path.exists(cache_file) or not os.path.exists(meta_file):
                self._stats["misses"] += 1
                return None
            
            # 读取元数据
            try:
                with open(meta_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                # 检查是否过期
                created_at = metadata.get("created_at")
                ttl = metadata.get("ttl", self.default_ttl)
                
                if ttl is not None and time.time() - created_at > ttl:
                    # 删除过期文件
                    os.remove(cache_file)
                    os.remove(meta_file)
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    return None
                
                # 读取数据
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # 更新访问信息
                metadata["last_accessed"] = time.time()
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                
                # 写回元数据
                with open(meta_file, 'wb') as f:
                    pickle.dump(metadata, f)
                
                self._stats["hits"] += 1
                return data
                
            except Exception as e:
                # 如果读取失败，删除文件
                try:
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    if os.path.exists(meta_file):
                        os.remove(meta_file)
                except:
                    pass
                
                self._stats["misses"] += 1
                return None
    
    def set(self, 
            key: str,
            value: Any,
            ttl: Optional[float] = None) -> bool:
        """
        设置缓存项
        
        Args:
            key: 缓存键
            value: 要缓存的数据
            ttl: 生存时间（秒）
            
        Returns:
            是否设置成功
        """
        with self._lock:
            self._stats["sets"] += 1
            
            cache_file = self._get_cache_file_path(key)
            meta_file = self._get_metadata_file_path(key)
            
            # 准备元数据
            metadata = {
                "key": key,
                "created_at": time.time(),
                "ttl": ttl if ttl is not None else self.default_ttl,
                "last_accessed": time.time(),
                "access_count": 0
            }
            
            # 保存数据
            try:
                # 先写入临时文件，再重命名，确保原子性
                temp_cache = f"{cache_file}.tmp"
                temp_meta = f"{meta_file}.tmp"
                
                with open(temp_cache, 'wb') as f:
                    pickle.dump(value, f)
                
                with open(temp_meta, 'wb') as f:
                    pickle.dump(metadata, f)
                
                # 原子性重命名
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                if os.path.exists(meta_file):
                    os.remove(meta_file)
                    
                os.rename(temp_cache, cache_file)
                os.rename(temp_meta, meta_file)
                
                return True
                
            except Exception as e:
                # 清理临时文件
                try:
                    if os.path.exists(temp_cache):
                        os.remove(temp_cache)
                    if os.path.exists(temp_meta):
                        os.remove(temp_meta)
                except:
                    pass
                
                return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        with self._lock:
            cache_file = self._get_cache_file_path(key)
            meta_file = self._get_metadata_file_path(key)
            
            deleted = False
            
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    deleted = True
                
                if os.path.exists(meta_file):
                    os.remove(meta_file)
                    deleted = True
                    
            except Exception:
                deleted = False
            
            return deleted
    
    def clear(self) -> None:
        """
        清空缓存
        """
        with self._lock:
            # 删除所有缓存文件
            for file_name in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file_name)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
    
    def contains(self, key: str) -> bool:
        """
        检查键是否在缓存中
        
        Args:
            key: 缓存键
            
        Returns:
            键是否存在
        """
        with self._lock:
            cache_file = self._get_cache_file_path(key)
            meta_file = self._get_metadata_file_path(key)
            
            return os.path.exists(cache_file) and os.path.exists(meta_file)
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = self._stats.copy()
            
            # 计算命中率
            total = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total if total > 0 else 0
            
            # 计算缓存中的项目数量
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            stats["items_count"] = len(cache_files)
            
            return stats
    
    def cleanup(self) -> int:
        """
        清理过期的缓存项
        
        Returns:
            清理的项目数量
        """
        with self._lock:
            cleaned_count = 0
            
            # 遍历所有元数据文件
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith('.meta'):
                    meta_file = os.path.join(self.cache_dir, file_name)
                    cache_file = meta_file.replace('.meta', '.pkl')
                    
                    try:
                        with open(meta_file, 'rb') as f:
                            metadata = pickle.load(f)
                        
                        # 检查是否过期
                        created_at = metadata.get("created_at")
                        ttl = metadata.get("ttl", self.default_ttl)
                        
                        if ttl is not None and time.time() - created_at > ttl:
                            # 删除过期文件
                            if os.path.exists(cache_file):
                                os.remove(cache_file)
                            os.remove(meta_file)
                            cleaned_count += 1
                            self._stats["expirations"] += 1
                            
                    except Exception:
                        # 如果读取失败，删除文件
                        try:
                            if os.path.exists(cache_file):
                                os.remove(cache_file)
                            os.remove(meta_file)
                        except:
                            pass
            
            return cleaned_count
    
    def get_disk_usage(self) -> int:
        """
        获取磁盘使用量
        
        Returns:
            使用的字节数
        """
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        return total_size


class HybridCache:
    """
    混合缓存实现，结合内存缓存和磁盘缓存
    """
    
    def __init__(self, 
                 memory_cache: Optional[MemoryCache] = None,
                 disk_cache: Optional[DiskCache] = None,
                 default_ttl: Optional[float] = None):
        """
        初始化混合缓存
        
        Args:
            memory_cache: 内存缓存实例
            disk_cache: 磁盘缓存实例
            default_ttl: 默认生存时间（秒）
        """
        # 如果未提供，创建默认的内存缓存
        if memory_cache is None:
            memory_cache = MemoryCache(
                max_size_bytes=100 * 1024 * 1024,  # 100MB
                max_items=1000,
                default_ttl=default_ttl
            )
        
        # 如果未提供，创建默认的磁盘缓存
        if disk_cache is None:
            disk_cache = DiskCache(
                cache_dir=".cache/hybrid",
                default_ttl=default_ttl
            )
        
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache
        self.default_ttl = default_ttl
        
        # 访问锁
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "gets": 0,
            "sets": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        首先尝试从内存缓存获取，如果未命中则从磁盘缓存获取，并加载到内存缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据或None
        """
        with self._lock:
            self._stats["gets"] += 1
            
            # 1. 尝试从内存缓存获取
            value = self.memory_cache.get(key)
            if value is not None:
                self._stats["memory_hits"] += 1
                return value
            
            # 2. 尝试从磁盘缓存获取
            value = self.disk_cache.get(key)
            if value is not None:
                # 将值加载到内存缓存
                self.memory_cache.set(key, value)
                self._stats["disk_hits"] += 1
                return value
            
            # 3. 缓存未命中
            self._stats["misses"] += 1
            return None
    
    def set(self, 
            key: str,
            value: Any,
            ttl: Optional[float] = None,
            memory_only: bool = False) -> bool:
        """
        设置缓存项
        
        Args:
            key: 缓存键
            value: 要缓存的数据
            ttl: 生存时间（秒）
            memory_only: 是否仅存储在内存缓存
            
        Returns:
            是否设置成功
        """
        with self._lock:
            self._stats["sets"] += 1
            
            # 确定TTL
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            # 1. 设置到内存缓存
            memory_success = self.memory_cache.set(key, value, effective_ttl)
            
            # 2. 如果不是仅内存模式，设置到磁盘缓存
            disk_success = True
            if not memory_only:
                disk_success = self.disk_cache.set(key, value, effective_ttl)
            
            return memory_success and disk_success
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        with self._lock:
            memory_deleted = self.memory_cache.delete(key)
            disk_deleted = self.disk_cache.delete(key)
            
            return memory_deleted or disk_deleted
    
    def clear(self) -> None:
        """
        清空缓存
        """
        with self._lock:
            self.memory_cache.clear()
            self.disk_cache.clear()
    
    def contains(self, key: str) -> bool:
        """
        检查键是否在缓存中
        
        Args:
            key: 缓存键
            
        Returns:
            键是否存在
        """
        with self._lock:
            return key in self.memory_cache or self.disk_cache.contains(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = self._stats.copy()
            
            # 计算总命中率
            total_hits = stats["memory_hits"] + stats["disk_hits"]
            total = total_hits + stats["misses"]
            
            stats["total_hit_rate"] = total_hits / total if total > 0 else 0
            stats["memory_hit_rate"] = stats["memory_hits"] / total if total > 0 else 0
            stats["disk_hit_rate"] = stats["disk_hits"] / total if total > 0 else 0
            
            # 添加内存缓存统计
            memory_stats = self.memory_cache.get_stats()
            stats["memory_cache"] = memory_stats
            
            # 添加磁盘缓存统计
            disk_stats = self.disk_cache.get_stats()
            stats["disk_cache"] = disk_stats
            
            return stats
    
    def cleanup(self) -> Dict[str, int]:
        """
        清理过期的缓存项
        
        Returns:
            清理统计信息
        """
        with self._lock:
            memory_cleaned = self.memory_cache.cleanup()
            disk_cleaned = self.disk_cache.cleanup()
            
            return {
                "memory_cleaned": memory_cleaned,
                "disk_cleaned": disk_cleaned,
                "total_cleaned": memory_cleaned + disk_cleaned
            }
    
    def __len__(self) -> int:
        """
        返回内存缓存中的项目数量
        """
        return len(self.memory_cache)
    
    def __contains__(self, key: str) -> bool:
        """
        检查键是否在缓存中
        """
        return self.contains(key)


class CacheManager:
    """
    缓存管理器，提供统一的缓存接口和管理功能
    """
    
    def __init__(self, 
                 config_path: str = "config/config.json"):
        """
        初始化缓存管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        from src.utils.helpers import load_config
        config = load_config(config_path)
        
        # 缓存配置
        cache_config = config.get("data", {})
        
        # 创建缓存实例
        self._caches: Dict[str, HybridCache] = {}
        
        # 默认缓存
        self._default_cache = HybridCache(
            memory_cache=MemoryCache(
                max_size_bytes=cache_config.get("max_memory_cache_mb", 100) * 1024 * 1024,
                max_items=cache_config.get("max_cache_items", 1000),
                default_ttl=cache_config.get("cache_ttl_seconds", 3600)
            ),
            disk_cache=DiskCache(
                cache_dir=cache_config.get("cache_dir", ".cache"),
                default_ttl=cache_config.get("cache_ttl_seconds", 3600)
            )
        )
        
        # 注册默认缓存
        self.register_cache("default", self._default_cache)
        
        # 定时清理任务
        self._cleanup_interval = cache_config.get("cleanup_interval_seconds", 300)  # 默认5分钟
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # 设置日志
        self.logger = setup_logger(
            name="cache_manager",
            log_file=os.path.join("logs", "cache_manager.log"),
            level=DEFAULT_LOGGER.level
        )
        
        # 启动清理线程
        self._start_cleanup_thread()
        
        self.logger.info("缓存管理器初始化完成")
    
    def register_cache(self, name: str, cache: HybridCache) -> "CacheManager":
        """
        注册缓存实例
        
        Args:
            name: 缓存名称
            cache: 缓存实例
            
        Returns:
            管理器实例，支持链式调用
        """
        self._caches[name] = cache
        self.logger.info(f"已注册缓存: {name}")
        return self
    
    def get_cache(self, name: str = "default") -> Optional[HybridCache]:
        """
        获取缓存实例
        
        Args:
            name: 缓存名称
            
        Returns:
            缓存实例或None
        """
        return self._caches.get(name, self._default_cache)
    
    def get(self, 
            key: str,
            cache_name: str = "default",
            default: Any = None) -> Any:
        """
        获取缓存项
        
        Args:
            key: 缓存键
            cache_name: 缓存名称
            default: 默认值
            
        Returns:
            缓存的数据或默认值
        """
        cache = self.get_cache(cache_name)
        value = cache.get(key)
        return value if value is not None else default
    
    def set(self, 
            key: str,
            value: Any,
            cache_name: str = "default",
            ttl: Optional[float] = None,
            memory_only: bool = False) -> bool:
        """
        设置缓存项
        
        Args:
            key: 缓存键
            value: 要缓存的数据
            cache_name: 缓存名称
            ttl: 生存时间（秒）
            memory_only: 是否仅存储在内存缓存
            
        Returns:
            是否设置成功
        """
        cache = self.get_cache(cache_name)
        return cache.set(key, value, ttl, memory_only)
    
    def delete(self, 
               key: str,
               cache_name: str = "default") -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            cache_name: 缓存名称
            
        Returns:
            是否删除成功
        """
        cache = self.get_cache(cache_name)
        return cache.delete(key)
    
    def clear(self, cache_name: str = "default") -> None:
        """
        清空缓存
        
        Args:
            cache_name: 缓存名称
        """
        cache = self.get_cache(cache_name)
        cache.clear()
        self.logger.info(f"已清空缓存: {cache_name}")
    
    def clear_all(self) -> None:
        """
        清空所有缓存
        """
        for name, cache in self._caches.items():
            cache.clear()
        self.logger.info("已清空所有缓存")
    
    def contains(self, 
                 key: str,
                 cache_name: str = "default") -> bool:
        """
        检查键是否在缓存中
        
        Args:
            key: 缓存键
            cache_name: 缓存名称
            
        Returns:
            键是否存在
        """
        cache = self.get_cache(cache_name)
        return cache.contains(key)
    
    def get_stats(self, cache_name: str = "default") -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Args:
            cache_name: 缓存名称
            
        Returns:
            统计信息字典
        """
        cache = self.get_cache(cache_name)
        return cache.get_stats()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有缓存的统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        for name, cache in self._caches.items():
            stats[name] = cache.get_stats()
        return stats
    
    def cleanup(self, cache_name: str = "default") -> Dict[str, int]:
        """
        清理过期的缓存项
        
        Args:
            cache_name: 缓存名称
            
        Returns:
            清理统计信息
        """
        cache = self.get_cache(cache_name)
        result = cache.cleanup()
        self.logger.info(f"清理缓存 {cache_name}: {result}")
        return result
    
    def cleanup_all(self) -> Dict[str, Dict[str, int]]:
        """
        清理所有缓存的过期项
        
        Returns:
            清理统计信息
        """
        results = {}
        for name, cache in self._caches.items():
            results[name] = cache.cleanup()
        
        self.logger.info(f"清理所有缓存: {results}")
        return results
    
    def _cleanup_task(self):
        """
        定时清理任务
        """
        while not self._stop_cleanup.is_set():
            try:
                # 等待指定的时间间隔
                self._stop_cleanup.wait(self._cleanup_interval)
                
                if not self._stop_cleanup.is_set():
                    self.logger.debug("执行定时缓存清理")
                    self.cleanup_all()
                    
                    # 执行垃圾回收
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"定时清理任务出错: {str(e)}")
    
    def _start_cleanup_thread(self):
        """
        启动清理线程
        """
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_task,
                daemon=True
            )
            self._cleanup_thread.start()
            self.logger.info("缓存清理线程已启动")
    
    def stop(self):
        """
        停止缓存管理器
        """
        # 停止清理线程
        if self._cleanup_thread is not None:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
            self.logger.info("缓存管理器已停止")
    
    def __del__(self):
        """
        析构函数，确保线程正确停止
        """
        self.stop()
    
    def __str__(self) -> str:
        """
        返回缓存管理器的字符串表示
        """
        return f"缓存管理器 (缓存数量: {len(self._caches)}, 清理间隔: {self._cleanup_interval}秒)"


# 创建全局缓存管理器实例
_cache_manager_instance: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    获取全局缓存管理器实例
    
    Returns:
        缓存管理器实例
    """
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    
    return _cache_manager_instance


def cache_key_wrapper(prefix: str = "") -> Callable:
    """
    缓存键包装装饰器，生成标准化的缓存键
    
    Args:
        prefix: 键前缀
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 生成缓存键
            key_parts = [prefix, func.__module__, func.__name__]
            
            # 处理参数
            for arg in args:
                key_parts.append(str(arg))
            
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")
            
            # 生成哈希键
            key_str = ":".join(key_parts)
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # 将缓存键添加到kwargs中
            kwargs['_cache_key'] = cache_key
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# 确保导入
import functools
