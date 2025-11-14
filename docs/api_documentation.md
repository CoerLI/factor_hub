# API文档

本文档提供了时间序列预测与交易因子分析框架的RESTful API接口说明。

## 目录

- [概述](#概述)
- [认证](#认证)
- [数据相关接口](#数据相关接口)
- [因子相关接口](#因子相关接口)
- [回测相关接口](#回测相关接口)
- [模型相关接口](#模型相关接口)
- [错误码](#错误码)

## 概述

系统提供了一组RESTful API，用于访问数据、计算因子、运行回测、训练模型和获取预测结果。所有API响应均使用JSON格式。

## 认证

API使用API密钥进行认证，需要在请求头中包含`Authorization`字段：

```
Authorization: Bearer {api_key}
```

## 数据相关接口

### 1. 获取数据列表

**请求URL**: `/api/v1/data`
**请求方法**: `GET`
**请求参数**:
- `source` (可选): 数据源
- `limit` (可选): 返回记录数限制，默认100
- `offset` (可选): 偏移量，默认0

**响应示例**:
```json
{
  "data": [
    {
      "id": "data_123",
      "name": "股票数据",
      "source": "csv",
      "created_at": "2023-07-01T12:00:00Z",
      "records_count": 1000
    }
  ],
  "total": 1,
  "page": 1,
  "pages": 1
}
```

### 2. 上传数据

**请求URL**: `/api/v1/data/upload`
**请求方法**: `POST`
**请求体**:
- `file`: CSV文件
- `metadata`: 元数据JSON

**响应示例**:
```json
{
  "id": "data_123",
  "name": "上传的数据",
  "status": "uploaded",
  "created_at": "2023-07-01T12:00:00Z"
}
```

## 因子相关接口

### 1. 计算单个因子

**请求URL**: `/api/v1/factors/compute`
**请求方法**: `POST`
**请求体**:
```json
{
  "data_id": "data_123",
  "factor_type": "moving_average",
  "params": {
    "window": 20,
    "column": "close"
  }
}
```

**响应示例**:
```json
{
  "factor_id": "factor_456",
  "factor_type": "moving_average",
  "status": "computed",
  "computed_at": "2023-07-01T12:05:00Z"
}
```

### 2. 批量计算因子

**请求URL**: `/api/v1/factors/batch_compute`
**请求方法**: `POST`
**请求体**:
```json
{
  "data_id": "data_123",
  "factors": [
    {
      "factor_type": "moving_average",
      "params": {
        "window": 20,
        "column": "close"
      }
    },
    {
      "factor_type": "rsi",
      "params": {
        "window": 14,
        "column": "close"
      }
    }
  ]
}
```

**响应示例**:
```json
{
  "job_id": "job_789",
  "status": "processing",
  "created_at": "2023-07-01T12:10:00Z"
}
```

## 回测相关接口

### 1. 创建回测任务

**请求URL**: `/api/v1/backtest/create`
**请求方法**: `POST`
**请求体**:
```json
{
  "strategy": "moving_average_cross",
  "data_id": "data_123",
  "params": {
    "short_window": 50,
    "long_window": 200,
    "initial_capital": 100000,
    "commission": 0.001
  },
  "start_date": "2020-01-01",
  "end_date": "2022-12-31"
}
```

**响应示例**:
```json
{
  "backtest_id": "backtest_123",
  "status": "created",
  "created_at": "2023-07-01T13:00:00Z"
}
```

### 2. 获取回测结果

**请求URL**: `/api/v1/backtest/{backtest_id}`
**请求方法**: `GET`

**响应示例**:
```json
{
  "backtest_id": "backtest_123",
  "status": "completed",
  "metrics": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": 0.08,
    "annualized_return": 0.07
  },
  "trades_count": 50,
  "completed_at": "2023-07-01T13:30:00Z"
}
```

## 模型相关接口

### 1. 训练模型

**请求URL**: `/api/v1/models/train`
**请求方法**: `POST`
**请求体**:
```json
{
  "data_id": "data_123",
  "model_type": "xgboost",
  "target_column": "close",
  "params": {
    "max_depth": 6,
    "n_estimators": 100
  },
  "features": ["open", "high", "low", "volume"]
}
```

**响应示例**:
```json
{
  "training_job_id": "train_123",
  "status": "started",
  "created_at": "2023-07-01T14:00:00Z"
}
```

### 2. 预测

**请求URL**: `/api/v1/models/{model_id}/predict`
**请求方法**: `POST`
**请求体**:
```json
{
  "features": {
    "open": 100.5,
    "high": 101.2,
    "low": 99.8,
    "volume": 1000000
  }
}
```

**响应示例**:
```json
{
  "prediction": 100.8,
  "confidence": 0.95,
  "timestamp": "2023-07-01T14:30:00Z"
}
```

### 3. 批量预测

**请求URL**: `/api/v1/models/{model_id}/batch_predict`
**请求方法**: `POST`
**请求体**:
```json
{
  "instances": [
    {
      "features": {
        "open": 100.5,
        "high": 101.2,
        "low": 99.8,
        "volume": 1000000
      }
    },
    {
      "features": {
        "open": 101.0,
        "high": 101.5,
        "low": 100.2,
        "volume": 950000
      }
    }
  ]
}
```

**响应示例**:
```json
{
  "predictions": [
    {
      "prediction": 100.8,
      "confidence": 0.95
    },
    {
      "prediction": 101.3,
      "confidence": 0.94
    }
  ]
}
```

## 错误码

| 错误码 | 描述 | HTTP状态码 |
|--------|------|------------|
| 40001 | 请求参数错误 | 400 |
| 40101 | 未授权访问 | 401 |
| 40301 | 禁止访问 | 403 |
| 40401 | 资源不存在 | 404 |
| 50001 | 服务器内部错误 | 500 |
| 50002 | 处理超时 | 504 |

## 示例代码

### Python请求示例

```python
import requests
import json

# 设置API密钥
headers = {
    'Authorization': 'Bearer your_api_key',
    'Content-Type': 'application/json'
}

# 预测请求
payload = {
    'features': {
        'open': 100.5,
        'high': 101.2,
        'low': 99.8,
        'volume': 1000000
    }
}

response = requests.post(
    'http://localhost:8000/api/v1/models/model_123/predict',
    headers=headers,
    data=json.dumps(payload)
)

print(response.json())
```