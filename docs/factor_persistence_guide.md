# 因子持久化与查询使用指南

本指南详细介绍了因子库中新增的因子持久化存储功能和Streamlit因子查询页面的使用方法，帮助用户高效管理和查询因子信息。

## 1. 因子持久化存储功能

### 1.1 核心组件

因子持久化功能主要由以下三个核心模块组成：

- **FactorPersistenceManager**: 负责因子数据的持久化存储和检索
- **FactorMetadataCollector**: 负责收集和序列化因子的详细元数据
- **FactorEngine集成**: 与现有因子引擎无缝集成，实现自动持久化

### 1.2 使用方法

#### 初始化因子引擎（启用持久化）

```python
from src.factors.factor_engine import FactorEngine

# 创建启用持久化功能的因子引擎实例
engine = FactorEngine(
    persistence_enabled=True,  # 启用持久化存储
    storage_dir="data/factor_storage",  # 持久化存储目录
    parallel=True  # 并行计算
)
```

#### 注册和计算因子（自动持久化）

因子引擎会在以下时刻自动保存因子信息：

- 因子注册时：自动保存因子的基本元数据
- 因子计算时：自动保存因子计算结果和详细元数据
- 创建容器时：自动保存容器信息和关联的因子列表

```python
from src.factors.moving_averages import SimpleMovingAverage

# 注册因子（自动保存元数据）
engine.register_factor(SimpleMovingAverage(window=10))

# 计算因子（自动保存结果和更新元数据）
import pandas as pd

# 准备数据
data = pd.DataFrame({
    'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}, index=pd.date_range(start='2024-01-01', periods=11))

# 计算因子（自动持久化）
results = engine.compute_factors(data, factor_names=["SMA_10"])

# 创建容器（自动保存容器信息）
container = engine.create_factor_container(data, factor_names=["SMA_10"])
```

#### 手动控制持久化行为

可以通过参数控制是否持久化计算结果：

```python
# 计算因子但不持久化
results = engine.compute_factors(
    data, 
    factor_names=["SMA_10"],
    persist_results=False  # 禁用此次计算的持久化
)

# 创建容器但不持久化
container = engine.create_factor_container(
    data, 
    factor_names=["SMA_10"],
    persist_results=False  # 禁用此次容器创建的持久化
)
```

#### 从存储中加载因子

```python
# 从存储中加载指定因子
factor_data = engine.load_factor_from_storage("SMA_10")
if factor_data:
    print(f"因子 {factor_data['metadata']['name']} 加载成功")
    print(f"最后计算时间: {factor_data['metadata']['performance_info']['last_computed']}")
    print(f"计算结果形状: {factor_data['result'].shape}")

# 获取所有已持久化的因子
all_factors = engine.get_all_persisted_factors()
print(f"共有 {len(all_factors)} 个已持久化的因子")
for factor_info in all_factors:
    print(f"- {factor_info['id']}: {factor_info['metadata'].get('description', '无描述')}")
```

#### 更新因子引擎配置

```python
# 动态更新因子引擎配置
engine.update_factor_config({
    "persistence_enabled": True,  # 启用或禁用持久化
    "storage_dir": "data/new_factor_storage"  # 更改存储目录
})
```

### 1.3 持久化数据结构

持久化存储包含以下信息：

1. **因子基本信息**：名称、描述、版本、类型等
2. **因子参数**：参数名称、类型、默认值、描述等
3. **计算性能**：计算时间、内存使用、调用次数等
4. **计算结果**：因子值、时间范围、数据格式等
5. **依赖关系**：依赖的其他因子信息
6. **数据特征**：结果的统计摘要、分布特征等
7. **元数据**：创建时间、最后更新时间、版本历史等

## 2. Streamlit因子查询页面

### 2.1 启动查询页面

使用以下命令启动Streamlit因子查询页面：

```bash
streamlit run scripts/factor_explorer.py
```

### 2.2 主要功能模块

#### 因子概览

- 显示所有已持久化因子的列表
- 提供筛选和搜索功能
- 展示因子的基本信息和状态

#### 因子详情

- 显示因子的详细元数据
- 可视化因子的计算性能历史
- 查看因子计算结果的样本数据
- 显示因子参数和配置信息

#### 因子依赖分析

- 可视化因子之间的依赖关系图
- 展示因子计算的上下游关系
- 分析依赖因子的状态和性能

#### 设置

- 配置持久化存储参数
- 设置查询页面显示选项
- 管理存储格式和优化设置

### 2.3 使用示例

#### 搜索特定因子

1. 在搜索框中输入因子名称或关键词
2. 使用筛选器选择因子类型、创建时间等条件
3. 点击搜索按钮查看匹配的因子

#### 查看因子详情

1. 在因子列表中选择一个因子
2. 切换到"因子详情"标签
3. 查看因子的详细信息、性能数据和样本结果

#### 导出因子数据

1. 在因子详情页面
2. 点击"导出数据"按钮
3. 选择导出格式（CSV、JSON、Parquet）
4. 下载因子数据文件

## 3. 高级用法

### 3.1 自定义存储格式

FactorPersistenceManager支持三种存储格式：

```python
from src.factors.factor_persistence import FactorPersistenceManager

# 使用JSON格式
json_manager = FactorPersistenceManager(
    storage_dir="data/factor_storage_json",
    default_format="json"
)

# 使用Parquet格式
parquet_manager = FactorPersistenceManager(
    storage_dir="data/factor_storage_parquet",
    default_format="parquet"
)

# 使用SQLite格式（默认）
sqlite_manager = FactorPersistenceManager(
    storage_dir="data/factor_storage_sqlite",
    default_format="sqlite"
)
```

### 3.2 批量操作

```python
# 批量导出因子
from src.factors.factor_persistence import FactorPersistenceManager

manager = FactorPersistenceManager(storage_dir="data/factor_storage")

# 导出所有因子到指定目录
manager.export_factors(
    export_dir="exports/factors",
    format="json",
    factor_ids=None  # None表示导出所有因子
)

# 导入因子
manager.import_factors(import_dir="exports/factors")
```

### 3.3 因子数据清理

```python
# 删除过期因子
ger.import_factors(import_dir="exports/factors")
```

### 3.3 因子数据清理

```python
# 删除过期因子
manager.delete_factor("old_factor_name")

# 清理旧版本数据
manager.clean_old_versions(days=30)  # 保留30天内的数据
```

## 4. 最佳实践

### 4.1 存储优化

- 对于大规模因子数据，推荐使用SQLite格式
- 定期清理不再需要的因子数据，避免存储空间浪费
- 对于频繁查询的因子，考虑使用缓存机制加速访问

### 4.2 性能优化

- 适当设置持久化频率，避免频繁写入
- 在批量计算场景中，可以考虑关闭单个计算的持久化，在计算完成后统一持久化
- 对于大型因子结果，考虑使用数据压缩或部分持久化策略

### 4.3 数据安全

- 定期备份因子存储目录
- 对于敏感因子数据，考虑使用加密存储
- 实施访问控制，限制对因子数据的修改权限

## 5. 故障排除

### 5.1 常见问题

- **因子数据未保存**：检查persistence_enabled参数是否设置为True
- **存储空间不足**：清理旧数据或扩大存储目录
- **查询性能慢**：考虑切换到SQLite存储格式
- **Streamlit页面无法加载数据**：检查存储目录路径和权限

### 5.2 日志和监控

- 因子引擎会记录持久化相关的日志信息
- 可以通过分析日志定位持久化相关的问题
- 定期检查存储使用情况和性能指标

## 6. 版本历史

- **1.0.0**：初始版本，支持基本的因子持久化和查询功能
- **1.1.0**：添加Streamlit查询页面和高级搜索功能
- **1.2.0**：优化存储格式和性能，添加批量操作功能

---

本指南将随着功能更新而不断完善，如有问题或建议，请联系开发团队。