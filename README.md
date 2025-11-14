# 时间序列预测与交易因子分析框架

这是一个功能全面的时间序列预测与交易因子分析框架，专为量化交易、金融分析和时间序列建模设计。该框架提供了完整的数据处理、因子计算、回测系统和机器学习模型训练部署流水线。

## 项目架构

```
m_timeseries_model/
├── src/                     # 源代码目录
│   ├── data/                # 数据处理模块
│   ├── factors/             # 因子库模块
│   ├── backtest/            # 回测系统模块
│   ├── models/              # 机器学习模型模块
│   ├── config/              # 配置管理模块
│   └── utils/               # 工具函数模块
├── config/                  # 配置文件目录
├── scripts/                 # 运行脚本目录
├── tests/                   # 测试文件目录
├── Dockerfile               # Docker构建文件
├── requirements.txt         # Python依赖列表
└── README.md                # 项目说明文档
```

## 核心功能模块

### 1. 数据处理模块 (src/data/)
- 多源数据加载器 (CSV, 数据库, API)
- 时间序列数据清洗与预处理
- 异常值检测与处理
- 特征工程流水线
- 数据分割与采样

### 2. 因子库模块 (src/factors/)
- 技术指标因子计算 (MA, RSI, MACD, BB等)
- 统计因子计算 (波动率, 偏度, 峰度等)
- 因子分析与验证工具
- 因子组合与优化
- 因子可视化分析

### 3. 回测系统模块 (src/backtest/)
- 事件驱动回测引擎
- 策略基类与实现框架
- 多种交易策略实现 (均线交叉, RSI, 布林带等)
- 绩效分析与风险评估
- 参数优化功能

### 4. 机器学习模型模块 (src/models/)
- 数据预处理与特征工程
- 多种模型训练与评估
- 模型集成与超参数优化
- 模型部署与服务化
- 模型监控与版本管理

## 安装指南

### 前提条件
- Python 3.10 或更高版本
- pip 包管理器
- Docker (可选，用于容器化部署)

### 安装步骤

1. 克隆项目仓库
```bash
git clone <repository-url>
cd m_timeseries_model
```

2. 创建并激活虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用指南

### 数据生成

使用内置脚本生成测试数据：
```bash
python scripts/generate_data.py --output ./data/sample_data.csv --length 1000 --features 5
```

### 模型训练

训练预测模型：
```bash
python scripts/train.py --data ./data/sample_data.csv --model-type xgboost --target close --save-path ./models/trained_model.pkl
```

### 模型部署

部署模型服务：
```bash
python scripts/deploy.py --model-path ./models/trained_model.pkl --mode local --port 8000
```

### 配置说明

所有配置都集中在 `config/config.yaml` 文件中，可以根据需要修改：
- 数据配置：数据源、预处理参数
- 模型配置：模型类型、超参数
- 因子配置：因子计算参数
- 回测配置：初始资金、手续费等
- 部署配置：服务端口、环境设置

## Docker部署

使用Docker构建和运行：
```bash
docker build -t m_timeseries_model .
docker run -p 8000:8000 m_timeseries_model
```

## API服务

模型部署后，可通过RESTful API进行预测：

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [...]}'
```

## 测试

运行单元测试：
```bash
pytest tests/
```

## 贡献指南

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系项目维护者。