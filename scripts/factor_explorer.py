import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入我们的因子相关模块
from src.factors.factor_persistence import FactorPersistenceManager
from src.factors.factor_metadata import FactorMetadataCollector
from src.factors.factor_engine import FactorEngine
from src.factors.factor_base import FactorContainer
from src.config.config_manager import ConfigManager
from src.utils.helpers import setup_logger


class FactorExplorerApp:
    """
    因子探索器Streamlit应用
    提供因子信息的查询、可视化和管理功能
    """
    
    def __init__(self):
        """
        初始化因子探索器应用
        """
        # 设置页面配置
        st.set_page_config(
            page_title="因子探索器 - 时间序列预测与交易因子分析",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 设置应用标题和样式
        st.title("📊 因子探索器")
        st.markdown("### 时间序列预测与交易因子分析框架")
        st.markdown("---")
        
        # 初始化日志
        self.logger = setup_logger("factor_explorer", log_file="logs/factor_explorer.log")
        
        # 加载配置
        self.config_manager = ConfigManager(os.path.join("config", "config.yaml"))
        
        # 初始化因子持久化管理器
        self.persistence_manager = FactorPersistenceManager(
            storage_dir=os.path.join("data", "factor_storage"),
            default_format="sqlite"
        )
        
        # 初始化因子元数据收集器
        self.metadata_collector = FactorMetadataCollector()
        
        # 初始化因子引擎（用于加载和计算因子）
        self.factor_engine = FactorEngine(
            config=self.config_manager.get_section("factors"),
            persistence_manager=self.persistence_manager
        )
        
        # 缓存因子列表
        self.factor_list = None
        
        # 应用状态
        self.app_state = {
            "selected_factor_id": None,
            "current_tab": "overview"
        }
    
    def run(self):
        """
        运行Streamlit应用主循环
        """
        # 侧边栏
        with st.sidebar:
            self._render_sidebar()
        
        # 主内容区
        if self.app_state["selected_factor_id"]:
            # 显示单个因子详情
            self._render_factor_details()
        else:
            # 显示因子列表和概览
            self._render_factor_overview()
        
        # 页脚
        self._render_footer()
    
    def _render_sidebar(self):
        """
        渲染侧边栏
        """
        st.sidebar.header("导航")
        
        # 导航选项
        navigation = st.sidebar.radio(
            "选择视图",
            ["因子概览", "因子详情", "依赖分析", "设置"],
            key="navigation"
        )
        
        # 根据导航选择更新当前标签
        navigation_map = {
            "因子概览": "overview",
            "因子详情": "details",
            "依赖分析": "dependencies",
            "设置": "settings"
        }
        self.app_state["current_tab"] = navigation_map[navigation]
        
        st.sidebar.markdown("---")
        
        # 因子存储统计信息
        stats = self.persistence_manager.get_storage_stats()
        st.sidebar.header("存储统计")
        st.sidebar.info(f"总因子数: {stats.get('total_factors', 0)}")
        st.sidebar.info(f"存储大小: {stats.get('total_size_mb', 0):.2f} MB")
        st.sidebar.info(f"最近更新: {stats.get('last_updated', '未知')}")
        
        # 刷新按钮
        if st.sidebar.button("🔄 刷新因子列表"):
            self.factor_list = None  # 清除缓存
            self.logger.info("刷新因子列表")
            st.experimental_rerun()
        
        # 操作按钮
        with st.sidebar.expander("操作", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 导出选中", use_container_width=True, disabled=not self.app_state["selected_factor_id"]):
                    self._export_factor()
            with col2:
                if st.button("🗑️ 删除选中", use_container_width=True, disabled=not self.app_state["selected_factor_id"]):
                    self._delete_factor()
    
    def _get_factor_list(self):
        """
        获取因子列表
        
        Returns:
            因子列表DataFrame
        """
        if self.factor_list is None:
            try:
                # 获取所有因子的基本信息
                factors = self.persistence_manager.list_factors(include_metadata=True)
                
                # 转换为DataFrame
                if factors:
                    df = pd.DataFrame([
                        {
                            "factor_id": f["factor_id"],
                            "factor_name": f["metadata"].get("basic_info", {}).get("factor_name", "未知"),
                            "factor_type": f["metadata"].get("basic_info", {}).get("factor_type", "未知"),
                            "class_name": f["metadata"].get("class_info", {}).get("class_name", "未知"),
                            "param_count": f["metadata"].get("params_info", {}).get("param_count", 0),
                            "dependency_count": f["metadata"].get("dependency_info", {}).get("dependency_count", 0),
                            "computation_time": f["metadata"].get("performance_info", {}).get("computation_time", "未知"),
                            "collected_at": f["metadata"].get("collected_at", "未知"),
                            "has_results": len(f["metadata"].get("result_stats", {})) > 0
                        }
                        for f in factors
                    ])
                    
                    # 按收集时间排序
                    if "collected_at" in df.columns:
                        df = df.sort_values("collected_at", ascending=False)
                    
                    self.factor_list = df
                else:
                    self.factor_list = pd.DataFrame()
                    st.warning("数据库中没有找到因子信息")
                    
            except Exception as e:
                self.logger.error(f"获取因子列表失败: {e}")
                st.error(f"获取因子列表失败: {str(e)}")
                self.factor_list = pd.DataFrame()
        
        return self.factor_list
    
    def _render_factor_overview(self):
        """
        渲染因子概览页面
        """
        st.header("因子概览")
        
        # 获取因子列表
        df = self._get_factor_list()
        
        if df.empty:
            st.info("暂无因子数据。请先计算并保存一些因子。")
            return
        
        # 筛选条件
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 按因子类型筛选
            factor_types = sorted(df["factor_type"].unique())
            selected_types = st.multiselect(
                "因子类型",
                factor_types,
                default=[],
                key="filter_types"
            )
            
        with col2:
            # 按类名筛选
            class_names = sorted(df["class_name"].unique())
            selected_classes = st.multiselect(
                "因子类",
                class_names,
                default=[],
                key="filter_classes"
            )
            
        with col3:
            # 按参数数量筛选
            min_params, max_params = int(df["param_count"].min()), int(df["param_count"].max())
            param_range = st.slider(
                "参数数量范围",
                min_value=min_params,
                max_value=max_params,
                value=(min_params, max_params),
                key="filter_params"
            )
        
        # 搜索框
        search_term = st.text_input("搜索因子名称或ID", "", key="search_term")
        
        # 应用筛选
        filtered_df = df.copy()
        
        if selected_types:
            filtered_df = filtered_df[filtered_df["factor_type"].isin(selected_types)]
        
        if selected_classes:
            filtered_df = filtered_df[filtered_df["class_name"].isin(selected_classes)]
        
        filtered_df = filtered_df[
            (filtered_df["param_count"] >= param_range[0]) & 
            (filtered_df["param_count"] <= param_range[1])
        ]
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df["factor_name"].str.contains(search_term, case=False) | 
                filtered_df["factor_id"].str.contains(search_term, case=False)
            ]
        
        # 显示统计信息
        st.markdown(f"**显示结果: {len(filtered_df)} / {len(df)} 个因子**")
        
        # 因子列表表格
        st.dataframe(
            filtered_df[ ["factor_name", "factor_type", "class_name", "param_count", "dependency_count", "collected_at"] ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "factor_name": st.column_config.TextColumn("因子名称", width="medium"),
                "factor_type": st.column_config.TextColumn("因子类型", width="small"),
                "class_name": st.column_config.TextColumn("因子类", width="medium"),
                "param_count": st.column_config.NumberColumn("参数数", width="small"),
                "dependency_count": st.column_config.NumberColumn("依赖数", width="small"),
                "collected_at": st.column_config.TextColumn("收集时间", width="medium")
            }
        )
        
        # 选择因子按钮
        if not filtered_df.empty:
            st.markdown("选择一个因子查看详情:")
            
            # 创建因子选择按钮（最多显示50个）
            display_df = filtered_df.head(50)
            for _, row in display_df.iterrows():
                if st.button(
                    f"📋 {row['factor_name']} ({row['factor_type']})",
                    key=f"btn_{row['factor_id']}",
                    use_container_width=True
                ):
                    self.app_state["selected_factor_id"] = row["factor_id"]
                    self.app_state["current_tab"] = "details"
                    st.experimental_rerun()
            
            if len(filtered_df) > 50:
                st.info(f"仅显示前50个因子，共 {len(filtered_df)} 个匹配结果")
        
        # 因子分布可视化
        st.markdown("---")
        st.header("因子分布统计")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 按因子类型分布
            st.subheader("按因子类型分布")
            type_counts = df["factor_type"].value_counts()
            
            if len(type_counts) > 0:
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="因子类型分布",
                    hole=0.3
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暂无数据")
        
        with col2:
            # 按参数数量分布
            st.subheader("按参数数量分布")
            param_counts = df["param_count"].value_counts().sort_index()
            
            if len(param_counts) > 0:
                fig = px.bar(
                    x=param_counts.index,
                    y=param_counts.values,
                    labels={"x": "参数数量", "y": "因子数量"},
                    title="因子参数数量分布"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暂无数据")
    
    def _render_factor_details(self):
        """
        渲染因子详情页面
        """
        if not self.app_state["selected_factor_id"]:
            st.warning("请先选择一个因子")
            return
        
        # 获取因子详细信息
        try:
            factor_data = self.persistence_manager.get_factor_by_id(self.app_state["selected_factor_id"])
            metadata = factor_data["metadata"]
            
            # 显示因子基本信息
            st.header(f"因子详情: {metadata.get('basic_info', {}).get('factor_name', '未知')}")
            
            # 因子ID和摘要
            st.markdown(f"**因子ID:** {self.app_state['selected_factor_id']}")
            st.markdown(f"**收集时间:** {metadata.get('collected_at', '未知')}")
            
            # 显示因子摘要
            st.markdown("---")
            st.subheader("📋 因子摘要")
            st.text(self.metadata_collector.generate_summary(metadata))
            
            # 详细标签页
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "基本信息", "参数配置", "计算结果", 
                "性能分析", "数据特征", "依赖关系"
            ])
            
            with tab1:
                self._render_basic_info(metadata)
            
            with tab2:
                self._render_params_info(metadata)
            
            with tab3:
                self._render_result_info(metadata)
            
            with tab4:
                self._render_performance_info(metadata)
            
            with tab5:
                self._render_data_info(metadata)
            
            with tab6:
                self._render_dependency_info(metadata)
            
        except Exception as e:
            self.logger.error(f"获取因子详情失败: {e}")
            st.error(f"获取因子详情失败: {str(e)}")
            
            # 返回到概览页面
            if st.button("返回概览"):
                self.app_state["selected_factor_id"] = None
                st.experimental_rerun()
    
    def _render_basic_info(self, metadata: dict):
        """
        渲染因子基本信息
        """
        basic_info = metadata.get("basic_info", {})
        class_info = metadata.get("class_info", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("基本信息")
            st.write(f"**因子名称:** {basic_info.get('factor_name', '未知')}")
            st.write(f"**因子类型:** {basic_info.get('factor_type', '未知')}")
            st.write(f"**创建时间:** {basic_info.get('created_at', '未知')}")
            st.write(f"**是否自定义:** {'是' if basic_info.get('is_custom', False) else '否'}")
            st.write(f"**缓存启用:** {'是' if basic_info.get('cache_enabled', False) else '否'}")
        
        with col2:
            st.subheader("类信息")
            st.write(f"**类名:** {class_info.get('class_name', '未知')}")
            st.write(f"**模块名:** {class_info.get('module_name', '未知')}")
            st.write(f"**完整类名:** {class_info.get('full_class_name', '未知')}")
            
            # 继承链
            inheritance = class_info.get('inheritance_chain', [])
            if inheritance:
                st.write("**继承链:**")
                for i, cls_name in enumerate(inheritance):
                    st.write(f"   {i}. {cls_name}")
        
        # 描述
        st.subheader("因子描述")
        description = basic_info.get('description', '无描述')
        st.info(description)
        
        # 类文档
        docstring = class_info.get('docstring', '无文档')
        if docstring:
            st.subheader("类文档")
            st.text_area("", docstring, height=200, disabled=True)
    
    def _render_params_info(self, metadata: dict):
        """
        渲染因子参数信息
        """
        params_info = metadata.get("params_info", {})
        params = params_info.get("parameters", {})
        param_details = params_info.get("param_details", {})
        
        st.subheader("参数配置")
        st.write(f"**参数总数:** {params_info.get('param_count', 0)}")
        st.write(f"**参数哈希:** {params_info.get('params_hash', '未知')}")
        
        # 参数表格
        if params:
            param_data = []
            for param_name, param_value in params.items():
                details = param_details.get(param_name, {})
                param_data.append({
                    "参数名": param_name,
                    "值": param_value,
                    "类型": details.get("type", "未知"),
                    "是否默认": "是" if details.get("is_default", False) else "否"
                })
            
            df = pd.DataFrame(param_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 参数详情展开
            for param_name, details in param_details.items():
                with st.expander(f"📋 参数详情: {param_name}"):
                    st.write(f"**参数值:** {details.get('value')}")
                    st.write(f"**类型:** {details.get('type')}")
                    st.write(f"**是否默认:** {'是' if details.get('is_default') else '否'}")
                    
                    # 参数范围信息
                    param_range = details.get('range')
                    if param_range:
                        st.write("**推荐范围:**")
                        for k, v in param_range.items():
                            st.write(f"   - {k}: {v}")
        else:
            st.info("该因子没有参数")
    
    def _render_result_info(self, metadata: dict):
        """
        渲染因子计算结果信息
        """
        result_stats = metadata.get("result_stats", {})
        
        if not result_stats:
            st.info("没有可用的计算结果信息")
            return
        
        basic_stats = result_stats.get("basic_stats", {})
        numeric_stats = result_stats.get("numeric_stats", {})
        distribution = result_stats.get("distribution", {})
        time_series_features = result_stats.get("time_series_features", {})
        
        st.subheader("计算结果概览")
        
        # 基本统计信息
        shape = basic_stats.get("shape", {})
        st.write(f"**结果形状:** {shape.get('rows', 0)}行 × {shape.get('columns', 0)}列")
        st.write(f"**索引类型:** {basic_stats.get('index_type', '未知')}")
        st.write(f"**是否包含空值:** {'是' if basic_stats.get('has_nulls', False) else '否'}")
        st.write(f"**空值比例:** {basic_stats.get('null_percentage', 0):.2f}%")
        
        # 数值列统计
        if numeric_stats:
            st.subheader("数值列统计")
            
            # 选择要显示的列
            columns = list(numeric_stats.keys())
            selected_column = st.selectbox("选择列查看详细统计", columns, key="result_column_select")
            
            if selected_column and selected_column in numeric_stats:
                col_stats = numeric_stats[selected_column]
                
                # 显示统计数据
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**基本统计量:**")
                    for stat_name in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                        if stat_name in col_stats:
                            st.write(f"   - {stat_name}: {col_stats[stat_name]:.6f}")
                
                with col2:
                    st.write("**高级统计量:**")
                    st.write(f"   - 偏度 (Skew): {col_stats.get('skew', 0):.6f}")
                    st.write(f"   - 峰度 (Kurtosis): {col_stats.get('kurtosis', 0):.6f}")
                    st.write(f"   - 自相关 (lag=1): {col_stats.get('autocorr_1', 0):.6f}")
                    st.write(f"   - 自相关 (lag=5): {col_stats.get('autocorr_5', 0):.6f}")
                
                # 分布可视化
                st.subheader("分布可视化")
                
                # 使用分布统计数据创建直方图
                if selected_column in distribution:
                    dist_data = distribution[selected_column]
                    quantiles = dist_data.get('quantiles', {})
                    
                    if quantiles:
                        # 创建模拟数据用于可视化
                        # 使用分位数和统计信息近似分布
                        mean_val = col_stats.get('mean', 0)
                        std_val = col_stats.get('std', 1)
                        
                        # 生成模拟数据
                        np.random.seed(42)
                        sample_size = 1000
                        if dist_data.get('is_normal', {}).get('is_approximately_normal', False):
                            # 如果是正态分布，使用正态分布生成数据
                            sample_data = np.random.normal(mean_val, std_val, sample_size)
                        else:
                            # 否则使用均匀分布加上一些噪声
                            min_val = col_stats.get('min', mean_val - 3*std_val)
                            max_val = col_stats.get('max', mean_val + 3*std_val)
                            sample_data = np.random.uniform(min_val, max_val, sample_size)
                            sample_data += np.random.normal(0, std_val/5, sample_size)
                        
                        # 创建直方图
                        fig = px.histogram(
                            x=sample_data,
                            nbins=50,
                            title=f"{selected_column} 分布近似图",
                            labels={"x": selected_column, "count": "频数"}
                        )
                        
                        # 添加分位数线
                        for q_name, q_value in quantiles.items():
                            fig.add_vline(x=q_value, line_dash="dash", name=f"{q_name}")
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # 时间序列特征（如果有）
        if time_series_features:
            st.subheader("时间序列特征")
            time_range = time_series_features.get("time_range", {})
            st.write(f"**时间范围:** {time_range.get('start', '未知')} 至 {time_range.get('end', '未知')}")
            st.write(f"**持续时间:** {time_range.get('duration_days', 0)} 天")
            st.write(f"**观测数量:** {time_range.get('observation_count', 0)}")
            
            freq_info = time_series_features.get("frequency", {})
            st.write(f"**推断频率:** {freq_info.get('inferred_freq', '未知')}")
            st.write(f"**主要间隔:** {freq_info.get('main_interval', '未知')}")
            st.write(f"**是否规则间隔:** {'是' if freq_info.get('has_regular_interval', False) else '否'}")
            
            missing_info = time_series_features.get("missing_data", {})
            st.write(f"**缺失时间点:** {missing_info.get('missing_count', 0)} 个 ({missing_info.get('missing_percentage', 0):.2f}%)")
    
    def _render_performance_info(self, metadata: dict):
        """
        渲染因子性能信息
        """
        performance_info = metadata.get("performance_info", {})
        
        st.subheader("性能信息")
        
        # 计算时间
        if performance_info.get("computation_time") is not None:
            st.write(f"**计算时间:** {performance_info['computation_time']:.6f} 秒")
        else:
            st.write("**计算时间:** 未知")
        
        # 上次计算时间
        if performance_info.get("last_computed"):
            st.write(f"**上次计算:** {performance_info['last_computed']}")
        
        # 计算次数
        st.write(f"**计算次数:** {performance_info.get('computation_count', 0)}")
        
        # 缓存信息
        cache_info = performance_info.get("cache_info", {})
        st.write(f"**缓存大小:** {cache_info.get('cache_size', 0)} 条目")
        st.write(f"**缓存内存:** {cache_info.get('cache_memory_usage', 0):.2f} MB")
        
        # 估计复杂度
        complexity = performance_info.get("estimated_complexity", "未知")
        complexity_color = {
            "low": "green",
            "medium": "orange",
            "high": "red"
        }.get(complexity, "gray")
        
        st.markdown(f"**估计复杂度:** <span style='color:{complexity_color};font-weight:bold;'>{complexity.upper()}</span>", 
                    unsafe_allow_html=True)
        
        # 性能评估
        st.subheader("性能评估")
        
        # 基于计算时间的评估
        computation_time = performance_info.get("computation_time", 0)
        if computation_time > 0:
            if computation_time < 0.01:
                perf_level = "极快"
                perf_color = "green"
            elif computation_time < 0.1:
                perf_level = "快速"
                perf_color = "lightgreen"
            elif computation_time < 1.0:
                perf_level = "中等"
                perf_color = "orange"
            elif computation_time < 10.0:
                perf_level = "较慢"
                perf_color = "darkorange"
            else:
                perf_level = "较慢"
                perf_color = "red"
            
            st.markdown(f"**计算性能:** <span style='color:{perf_color};font-weight:bold;'>{perf_level}</span>", 
                        unsafe_allow_html=True)
            
            # 计算时间进度条
            max_time = max(computation_time, 1.0)  # 最小显示1秒
            st.progress(min(computation_time / 10.0, 1.0))
            st.caption(f"相对于10秒参考值")
    
    def _render_data_info(self, metadata: dict):
        """
        渲染输入数据特征信息
        """
        data_features = metadata.get("data_features", {})
        
        if not data_features:
            st.info("没有可用的输入数据特征信息")
            return
        
        data_info = data_features.get("data_info", {})
        data_quality = data_features.get("data_quality", {})
        time_info = data_features.get("time_info", {})
        
        st.subheader("输入数据信息")
        
        # 基本信息
        shape = data_info.get("shape", {})
        st.write(f"**数据形状:** {shape.get('rows', 0)}行 × {shape.get('columns', 0)}列")
        st.write(f"**内存使用:** {data_info.get('memory_usage_mb', 0):.2f} MB")
        st.write(f"**索引类型:** {data_info.get('index_type', '未知')}")
        
        # 数据类型
        data_types = data_info.get("data_types", {})
        if data_types:
            st.subheader("数据类型统计")
            type_counts = {}
            for col_type in data_types.values():
                type_counts[col_type] = type_counts.get(col_type, 0) + 1
            
            # 饼图
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="列数据类型分布"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # 数据质量
        st.subheader("数据质量")
        st.write(f"**空值比例:** {data_quality.get('null_percentage', 0):.2f}%")
        st.write(f"**重复行数:** {data_quality.get('duplicate_rows', 0)}")
        st.write(f"**包含OHLCV列:** {'是' if data_quality.get('has_ohlcv', False) else '否'}")
        
        # 时间信息
        if time_info:
            st.subheader("时间信息")
            time_range = time_info.get("time_range", {})
            st.write(f"**数据时间范围:** {time_range.get('start', '未知')} 至 {time_range.get('end', '未知')}")
            st.write(f"**数据持续时间:** {time_range.get('duration_days', 0)} 天")
    
    def _render_dependency_info(self, metadata: dict):
        """
        渲染因子依赖关系信息
        """
        dependency_info = metadata.get("dependency_info", {})
        dependencies = dependency_info.get("dependencies", [])
        
        st.subheader("依赖关系")
        st.write(f"**依赖因子数量:** {len(dependencies)}")
        
        # 显示依赖列表
        if dependencies:
            st.write("**依赖因子列表:**")
            for dep in dependencies:
                st.write(f"   - {dep}")
            
            # 简单可视化依赖关系
            st.subheader("依赖关系图")
            try:
                # 创建有向图
                G = nx.DiGraph()
                
                # 添加当前因子
                current_factor = metadata.get('basic_info', {}).get('factor_name', 'Current Factor')
                G.add_node(current_factor)
                
                # 添加依赖因子
                for dep in dependencies:
                    G.add_node(dep)
                    G.add_edge(dep, current_factor)
                
                # 绘制图形
                plt.figure(figsize=(10, 6))
                pos = nx.spring_layout(G, seed=42)
                
                nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=['lightblue' if n == current_factor else 'lightgreen' for n in G.nodes])
                nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
                nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
                
                plt.title("因子依赖关系图")
                plt.axis('off')
                
                st.pyplot(plt)
            except Exception as e:
                self.logger.error(f"绘制依赖图失败: {e}")
                st.warning("无法生成依赖关系图")
        else:
            st.info("该因子没有依赖")
    
    def _render_footer(self):
        """
        渲染页脚
        """
        st.markdown("---")
        st.markdown("### 关于因子探索器")
        st.markdown("因子探索器是时间序列预测与交易因子分析框架的可视化组件，" 
                   "用于查询、可视化和管理因子信息。")
        st.markdown(f"**最后更新时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _export_factor(self):
        """
        导出选中的因子
        """
        if not self.app_state["selected_factor_id"]:
            return
        
        try:
            # 获取因子数据
            factor_data = self.persistence_manager.get_factor_by_id(self.app_state["selected_factor_id"])
            
            # 转换为JSON
            json_data = json.dumps(factor_data, ensure_ascii=False, indent=2)
            
            # 下载按钮
            st.download_button(
                label=f"下载因子数据: {factor_data['metadata'].get('basic_info', {}).get('factor_name', 'unknown')}",
                data=json_data,
                file_name=f"factor_{self.app_state['selected_factor_id']}.json",
                mime="application/json"
            )
            
        except Exception as e:
            self.logger.error(f"导出因子失败: {e}")
            st.error(f"导出因子失败: {str(e)}")
    
    def _delete_factor(self):
        """
        删除选中的因子
        """
        if not self.app_state["selected_factor_id"]:
            return
        
        # 二次确认
        if st.warning("确定要删除这个因子吗？此操作不可恢复。"):
            if st.button("确认删除", type="primary", disabled=False):
                try:
                    self.persistence_manager.delete_factor(self.app_state["selected_factor_id"])
                    self.logger.info(f"删除因子: {self.app_state['selected_factor_id']}")
                    st.success("因子已成功删除")
                    
                    # 重置选择并刷新
                    self.app_state["selected_factor_id"] = None
                    self.factor_list = None
                    time.sleep(1)
                    st.experimental_rerun()
                except Exception as e:
                    self.logger.error(f"删除因子失败: {e}")
                    st.error(f"删除因子失败: {str(e)}")


if __name__ == "__main__":
    # 运行应用
    app = FactorExplorerApp()
    app.run()
