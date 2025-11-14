# 时间序列模型服务 Dockerfile
# 基于Python官方镜像构建
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 系统依赖安装
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要的目录
RUN mkdir -p data models logs cache deployments

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIMESERIES_CONFIG_PATH="/app/config/config.yaml"

# 暴露API端口和监控端口
EXPOSE 8000 8001

# 健康检查配置
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; response = requests.get('http://localhost:8000/health'); response.raise_for_status()"

# 启动命令
CMD ["python", "-m", "src.models.model_service"]