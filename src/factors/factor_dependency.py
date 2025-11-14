from typing import Dict, List, Set, Optional, Tuple, Any, Callable
import logging
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import os

from src.utils.helpers import setup_logger, DEFAULT_LOGGER
from src.factors.factor_base import FactorBase


class FactorDependencyError(Exception):
    """
    因子依赖相关的异常类
    """
    pass


class CircularDependencyError(FactorDependencyError):
    """
    循环依赖异常
    """
    def __init__(self, cycle: List[str]):
        """
        初始化循环依赖异常
        
        Args:
            cycle: 循环依赖的因子名称列表
        """
        cycle_str = ' -> '.join(cycle)
        message = f"检测到循环依赖: {cycle_str} -> {cycle[0]}"
        super().__init__(message)
        self.cycle = cycle


class MissingDependencyError(FactorDependencyError):
    """
    缺少依赖异常
    """
    def __init__(self, factor_name: str, missing_dep: str):
        """
        初始化缺少依赖异常
        
        Args:
            factor_name: 因子名称
            missing_dep: 缺少的依赖名称
        """
        message = f"因子 '{factor_name}' 依赖于 '{missing_dep}'，但该依赖未注册"
        super().__init__(message)
        self.factor_name = factor_name
        self.missing_dep = missing_dep


class FactorDependencyResolver:
    """
    因子依赖解析器，用于管理和解析因子之间的计算依赖关系
    """
    
    def __init__(self):
        """
        初始化因子依赖解析器
        """
        # 存储因子实例的字典
        self._factors: Dict[str, FactorBase] = {}
        
        # 依赖图: {factor_name: [dependencies]}
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)
        
        # 反向依赖图: {dependency_name: [factors_that_depend_on_it]}
        self._reverse_graph: Dict[str, List[str]] = defaultdict(list)
        
        # 已解析的拓扑顺序缓存
        self._topological_order_cache: Optional[List[str]] = None
        
        # 设置日志
        self.logger = setup_logger(
            name="factor_dependency_resolver",
            log_file=os.path.join("logs", "factor_dependency.log"),
            level=DEFAULT_LOGGER.level
        )
        
        self.logger.info("因子依赖解析器初始化完成")
    
    def register_factor(self, factor: FactorBase) -> "FactorDependencyResolver":
        """
        注册因子及其依赖关系
        
        Args:
            factor: 因子实例
            
        Returns:
            解析器实例，支持链式调用
        """
        factor_name = factor.name
        
        # 存储因子实例
        self._factors[factor_name] = factor
        
        # 获取因子的依赖
        dependencies = factor.get_dependencies()
        
        # 更新依赖图
        self._dependency_graph[factor_name] = dependencies
        
        # 更新反向依赖图
        for dep in dependencies:
            self._reverse_graph[dep].append(factor_name)
        
        # 清除拓扑顺序缓存
        self._topological_order_cache = None
        
        self.logger.info(f"已注册因子: {factor_name}，依赖: {dependencies}")
        
        return self
    
    def register_factors(self, factors: List[FactorBase]) -> "FactorDependencyResolver":
        """
        批量注册因子
        
        Args:
            factors: 因子实例列表
            
        Returns:
            解析器实例，支持链式调用
        """
        for factor in factors:
            self.register_factor(factor)
        return self
    
    def unregister_factor(self, factor_name: str) -> bool:
        """
        注销因子
        
        Args:
            factor_name: 因子名称
            
        Returns:
            是否成功注销
        """
        if factor_name not in self._factors:
            return False
        
        # 从依赖图中移除
        if factor_name in self._dependency_graph:
            del self._dependency_graph[factor_name]
        
        # 从反向依赖图中移除
        for dep_name, dependents in list(self._reverse_graph.items()):
            if factor_name in dependents:
                dependents.remove(factor_name)
                if not dependents:
                    del self._reverse_graph[dep_name]
        
        # 从因子字典中移除
        del self._factors[factor_name]
        
        # 清除拓扑顺序缓存
        self._topological_order_cache = None
        
        self.logger.info(f"已注销因子: {factor_name}")
        
        return True
    
    def get_factor(self, factor_name: str) -> Optional[FactorBase]:
        """
        获取因子实例
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子实例或None
        """
        return self._factors.get(factor_name)
    
    def get_factors(self) -> Dict[str, FactorBase]:
        """
        获取所有注册的因子
        
        Returns:
            因子字典
        """
        return self._factors.copy()
    
    def get_dependencies(self, factor_name: str) -> List[str]:
        """
        获取因子的依赖列表
        
        Args:
            factor_name: 因子名称
            
        Returns:
            依赖因子名称列表
        """
        return self._dependency_graph.get(factor_name, []).copy()
    
    def get_dependents(self, factor_name: str) -> List[str]:
        """
        获取依赖于指定因子的因子列表
        
        Args:
            factor_name: 因子名称
            
        Returns:
            依赖因子名称列表
        """
        return self._reverse_graph.get(factor_name, []).copy()
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        获取依赖图
        
        Returns:
            依赖图字典
        """
        return {k: v.copy() for k, v in self._dependency_graph.items()}
    
    def get_reverse_graph(self) -> Dict[str, List[str]]:
        """
        获取反向依赖图
        
        Returns:
            反向依赖图字典
        """
        return {k: v.copy() for k, v in self._reverse_graph.items()}
    
    def _validate_dependencies(self) -> None:
        """
        验证所有依赖是否存在
        
        Raises:
            MissingDependencyError: 当缺少依赖时
        """
        for factor_name, dependencies in self._dependency_graph.items():
            for dep in dependencies:
                if dep not in self._factors:
                    raise MissingDependencyError(factor_name, dep)
    
    def find_cycles(self) -> List[List[str]]:
        """
        查找所有循环依赖
        
        Returns:
            循环列表
        """
        # 使用深度优先搜索查找循环
        cycles = []
        visited = set()
        recursion_stack = []
        
        def dfs(node: str, path: List[str]):
            if node in recursion_stack:
                # 找到循环
                cycle_start = recursion_stack.index(node)
                cycle = path[cycle_start:]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            recursion_stack.append(node)
            path.append(node)
            
            for neighbor in self._dependency_graph.get(node, []):
                dfs(neighbor, path.copy())
            
            recursion_stack.pop()
        
        # 对所有未访问的节点进行DFS
        for node in self._factors.keys():
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def validate_graph(self) -> None:
        """
        验证依赖图的有效性
        
        Raises:
            MissingDependencyError: 当缺少依赖时
            CircularDependencyError: 当存在循环依赖时
        """
        # 验证依赖存在性
        self._validate_dependencies()
        
        # 检查循环依赖
        cycles = self.find_cycles()
        if cycles:
            # 抛出第一个循环依赖异常
            raise CircularDependencyError(cycles[0])
    
    def topological_sort(self) -> List[str]:
        """
        执行拓扑排序，获取因子计算顺序
        
        Returns:
            拓扑排序后的因子名称列表
        """
        # 如果有缓存，直接返回
        if self._topological_order_cache is not None:
            return self._topological_order_cache.copy()
        
        # 验证图的有效性
        self.validate_graph()
        
        # 使用Kahn算法执行拓扑排序
        # 1. 计算每个节点的入度
        in_degree = {node: 0 for node in self._factors.keys()}
        for node, neighbors in self._dependency_graph.items():
            for neighbor in neighbors:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        # 2. 将所有入度为0的节点加入队列
        queue = [node for node, degree in in_degree.items() if degree == 0]
        
        # 3. 执行拓扑排序
        top_order = []
        while queue:
            # 获取一个入度为0的节点
            node = queue.pop(0)
            top_order.append(node)
            
            # 减少其邻居的入度
            for neighbor in self._dependency_graph.get(node, []):
                in_degree[neighbor] -= 1
                # 如果入度变为0，加入队列
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否所有节点都被访问
        if len(top_order) != len(self._factors):
            # 这应该不会发生，因为我们已经检查了循环依赖
            raise FactorDependencyError("拓扑排序失败，可能存在不可检测的循环依赖")
        
        # 反转顺序，因为我们希望依赖在前面
        top_order.reverse()
        
        # 缓存结果
        self._topological_order_cache = top_order
        
        self.logger.debug(f"拓扑排序结果: {top_order}")
        
        return top_order.copy()
    
    def get_computation_order(self, target_factors: Optional[List[str]] = None) -> List[str]:
        """
        获取指定因子及其依赖的计算顺序
        
        Args:
            target_factors: 目标因子名称列表，如果为None则计算所有因子
            
        Returns:
            计算顺序的因子名称列表
        """
        if target_factors is None:
            # 计算所有因子
            return self.topological_sort()
        
        # 验证目标因子是否存在
        for factor_name in target_factors:
            if factor_name not in self._factors:
                raise ValueError(f"因子 '{factor_name}' 未注册")
        
        # 收集所有需要的因子（目标因子及其所有依赖）
        needed_factors = set()
        
        def collect_dependencies(factor_name: str):
            if factor_name in needed_factors:
                return
            
            needed_factors.add(factor_name)
            for dep in self._dependency_graph.get(factor_name, []):
                collect_dependencies(dep)
        
        # 收集每个目标因子的依赖
        for factor_name in target_factors:
            collect_dependencies(factor_name)
        
        # 创建子图
        sub_graph = {}
        for factor_name in needed_factors:
            dependencies = [dep for dep in self._dependency_graph.get(factor_name, []) 
                           if dep in needed_factors]
            sub_graph[factor_name] = dependencies
        
        # 在子图上执行拓扑排序
        # 1. 计算每个节点的入度
        in_degree = {node: 0 for node in sub_graph.keys()}
        for node, neighbors in sub_graph.items():
            for neighbor in neighbors:
                in_degree[neighbor] += 1
        
        # 2. 将所有入度为0的节点加入队列
        queue = [node for node, degree in in_degree.items() if degree == 0]
        
        # 3. 执行拓扑排序
        top_order = []
        while queue:
            # 获取一个入度为0的节点
            node = queue.pop(0)
            top_order.append(node)
            
            # 减少其邻居的入度
            for neighbor in sub_graph.get(node, []):
                in_degree[neighbor] -= 1
                # 如果入度变为0，加入队列
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 反转顺序
        top_order.reverse()
        
        self.logger.debug(f"目标因子计算顺序 ({target_factors}): {top_order}")
        
        return top_order
    
    def find_leaf_factors(self) -> List[str]:
        """
        查找叶节点因子（没有依赖的因子）
        
        Returns:
            叶节点因子名称列表
        """
        # 叶节点没有依赖
        leaf_factors = [factor_name for factor_name, dependencies in 
                       self._dependency_graph.items() if not dependencies]
        return leaf_factors
    
    def find_root_factors(self) -> List[str]:
        """
        查找根节点因子（不被其他因子依赖的因子）
        
        Returns:
            根节点因子名称列表
        """
        # 根节点不在反向依赖图中
        all_factors = set(self._factors.keys())
        dependent_factors = set(self._reverse_graph.keys())
        root_factors = list(all_factors - dependent_factors)
        return root_factors
    
    def get_factor_depth(self, factor_name: str) -> int:
        """
        获取因子的深度（最长依赖链的长度）
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子深度
        """
        if factor_name not in self._factors:
            raise ValueError(f"因子 '{factor_name}' 未注册")
        
        # 使用DFS计算深度
        max_depth = 0
        
        def dfs(node: str, depth: int):
            nonlocal max_depth
            current_depth = depth + 1
            max_depth = max(max_depth, current_depth)
            
            for dep in self._dependency_graph.get(node, []):
                dfs(dep, current_depth)
        
        dfs(factor_name, 0)
        
        return max_depth
    
    def visualize_dependency_graph(self, 
                                 output_path: Optional[str] = None,
                                 show: bool = True) -> None:
        """
        可视化依赖图
        
        Args:
            output_path: 输出文件路径
            show: 是否显示图形
        """
        try:
            # 创建有向图
            G = nx.DiGraph()
            
            # 添加节点
            for factor_name in self._factors.keys():
                G.add_node(factor_name)
            
            # 添加边
            for factor_name, dependencies in self._dependency_graph.items():
                for dep in dependencies:
                    G.add_edge(dep, factor_name)  # 注意方向：依赖 -> 因子
            
            # 设置布局
            pos = nx.spring_layout(G, seed=42)
            
            # 绘制图形
            plt.figure(figsize=(12, 8))
            
            # 节点颜色
            color_map = ['#ff9999' if factor_name in self._dependency_graph 
                         else '#99ff99' for factor_name in G.nodes()]
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=color_map)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, width=1.5, arrowstyle='->', 
                                  arrowsize=20, edge_color='#333333')
            
            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # 添加标题
            plt.title('因子依赖图', fontsize=16)
            
            # 调整布局
            plt.axis('off')
            plt.tight_layout()
            
            # 保存图片
            if output_path:
                ensure_directory(os.path.dirname(output_path))
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"依赖图已保存至: {output_path}")
            
            # 显示图形
            if show:
                plt.show()
                
            plt.close()
            
        except ImportError:
            self.logger.error("无法可视化依赖图: 请安装 networkx 和 matplotlib")
        except Exception as e:
            self.logger.error(f"可视化依赖图失败: {str(e)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取依赖图的摘要信息
        
        Returns:
            摘要信息字典
        """
        # 基本统计
        total_factors = len(self._factors)
        leaf_factors = self.find_leaf_factors()
        root_factors = self.find_root_factors()
        
        # 依赖统计
        dependency_counts = {factor_name: len(dependencies) 
                           for factor_name, dependencies in self._dependency_graph.items()}
        max_dependencies = max(dependency_counts.values()) if dependency_counts else 0
        avg_dependencies = sum(dependency_counts.values()) / total_factors if total_factors > 0 else 0
        
        # 被依赖统计
        dependent_counts = {factor_name: len(dependents) 
                          for factor_name, dependents in self._reverse_graph.items()}
        max_dependents = max(dependent_counts.values()) if dependent_counts else 0
        avg_dependents = sum(dependent_counts.values()) / total_factors if total_factors > 0 else 0
        
        # 深度统计
        depths = {factor_name: self.get_factor_depth(factor_name) 
                 for factor_name in self._factors.keys()}
        max_depth = max(depths.values()) if depths else 0
        avg_depth = sum(depths.values()) / total_factors if total_factors > 0 else 0
        
        # 检查循环依赖
        has_cycles = len(self.find_cycles()) > 0
        
        # 构建摘要
        summary = {
            "total_factors": total_factors,
            "leaf_factors": {
                "count": len(leaf_factors),
                "factors": leaf_factors
            },
            "root_factors": {
                "count": len(root_factors),
                "factors": root_factors
            },
            "dependencies": {
                "max": max_dependencies,
                "average": avg_dependencies,
                "details": dependency_counts
            },
            "dependents": {
                "max": max_dependents,
                "average": avg_dependents,
                "details": dependent_counts
            },
            "depths": {
                "max": max_depth,
                "average": avg_depth,
                "details": depths
            },
            "has_cycles": has_cycles
        }
        
        return summary
    
    def get_factor_impact(self, factor_name: str) -> List[str]:
        """
        计算因子变更的影响范围
        
        Args:
            factor_name: 因子名称
            
        Returns:
            受影响的因子名称列表
        """
        if factor_name not in self._factors:
            raise ValueError(f"因子 '{factor_name}' 未注册")
        
        # 使用BFS查找所有依赖于该因子的因子
        affected = []
        visited = set()
        queue = [factor_name]
        
        while queue:
            current = queue.pop(0)
            
            # 将当前因子加入受影响列表（除了自己）
            if current != factor_name:
                affected.append(current)
            
            # 获取所有直接依赖于当前因子的因子
            for dependent in self._reverse_graph.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(dependent)
        
        self.logger.debug(f"因子 '{factor_name}' 的影响范围: {affected}")
        
        return affected
    
    def get_factor_path(self, 
                       start_factor: str, 
                       end_factor: str) -> Optional[List[str]]:
        """
        查找两个因子之间的依赖路径
        
        Args:
            start_factor: 起始因子名称
            end_factor: 结束因子名称
            
        Returns:
            依赖路径列表或None（如果不存在路径）
        """
        if start_factor not in self._factors or end_factor not in self._factors:
            raise ValueError("起始因子或结束因子未注册")
        
        # 使用BFS查找路径
        visited = set()
        queue = [(start_factor, [start_factor])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_factor:
                return path
            
            # 查找所有依赖于当前因子的因子
            for dependent in self._reverse_graph.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    new_path = path + [dependent]
                    queue.append((dependent, new_path))
        
        # 未找到路径
        return None
    
    def export_dependency_graph(self, file_path: str) -> bool:
        """
        导出依赖图为JSON格式
        
        Args:
            file_path: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            import json
            
            # 构建导出数据
            export_data = {
                "factors": {}
            }
            
            for factor_name, factor in self._factors.items():
                export_data["factors"][factor_name] = {
                    "class": factor.__class__.__name__,
                    "dependencies": self._dependency_graph.get(factor_name, []),
                    "dependents": self._reverse_graph.get(factor_name, [])
                }
            
            # 确保目录存在
            ensure_directory(os.path.dirname(file_path))
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"依赖图已导出至: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出依赖图失败: {str(e)}")
            return False
    
    def clear(self) -> "FactorDependencyResolver":
        """
        清空所有注册的因子和依赖信息
        
        Returns:
            解析器实例，支持链式调用
        """
        self._factors.clear()
        self._dependency_graph.clear()
        self._reverse_graph.clear()
        self._topological_order_cache = None
        
        self.logger.info("已清空所有因子和依赖信息")
        
        return self
    
    def __str__(self) -> str:
        """
        返回解析器的字符串表示
        """
        return f"因子依赖解析器 (已注册因子数: {len(self._factors)})"
    
    def __len__(self) -> int:
        """
        返回注册的因子数量
        """
        return len(self._factors)


# 确保导入的辅助函数
from src.utils.helpers import ensure_directory


def create_dependency_resolver() -> FactorDependencyResolver:
    """
    创建因子依赖解析器实例
    
    Returns:
        因子依赖解析器实例
    """
    return FactorDependencyResolver()
