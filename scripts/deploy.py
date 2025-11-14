#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
部署脚本
用于自动化模型服务的部署流程
"""

import os
import sys
import json
import yaml
import time
import argparse
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 自定义模块
from src.utils.helpers import setup_logger, DEFAULT_LOGGER
from src.config.config_manager import ConfigManager

# 设置日志
logger = setup_logger(DEFAULT_LOGGER)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        参数命名空间
    """
    parser = argparse.ArgumentParser(description='模型服务部署脚本')
    
    # 部署模式
    parser.add_argument('--mode', '-m', 
                      choices=['local', 'docker', 'kubernetes', 'aws', 'azure'],
                      default='local',
                      help='部署模式')
    
    # 配置文件
    parser.add_argument('--config', '-c',
                      type=str,
                      default='./config/config.yaml',
                      help='配置文件路径')
    
    # 环境
    parser.add_argument('--env',
                      choices=['dev', 'test', 'prod'],
                      default='dev',
                      help='部署环境')
    
    # 模型版本
    parser.add_argument('--model-version',
                      type=str,
                      help='要部署的模型版本')
    
    # 服务名称
    parser.add_argument('--service-name',
                      type=str,
                      default='timeseries-model-service',
                      help='服务名称')
    
    # 端口设置
    parser.add_argument('--port',
                      type=int,
                      default=8000,
                      help='API服务端口')
    
    parser.add_argument('--monitoring-port',
                      type=int,
                      default=8001,
                      help='监控服务端口')
    
    # Docker相关参数
    parser.add_argument('--docker-image',
                      type=str,
                      help='Docker镜像名称')
    
    parser.add_argument('--docker-tag',
                      type=str,
                      default='latest',
                      help='Docker镜像标签')
    
    parser.add_argument('--push-image',
                      action='store_true',
                      help='是否推送Docker镜像')
    
    # Kubernetes相关参数
    parser.add_argument('--namespace',
                      type=str,
                      default='timeseries',
                      help='Kubernetes命名空间')
    
    parser.add_argument('--replicas',
                      type=int,
                      default=2,
                      help='Kubernetes副本数')
    
    # 其他选项
    parser.add_argument('--skip-build',
                      action='store_true',
                      help='跳过构建步骤')
    
    parser.add_argument('--skip-tests',
                      action='store_true',
                      help='跳过测试步骤')
    
    parser.add_argument('--verbose',
                      action='store_true',
                      help='详细输出')
    
    return parser.parse_args()


def run_command(command: str,
                cwd: Optional[str] = None,
                env: Optional[Dict] = None,
                check: bool = True) -> subprocess.CompletedProcess:
    """
    执行系统命令
    
    Args:
        command: 要执行的命令
        cwd: 工作目录
        env: 环境变量
        check: 是否检查命令成功执行
        
    Returns:
        命令执行结果
    """
    logger.info(f"执行命令: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env
        )
        
        if result.stdout.strip():
            logger.info(f"命令输出:\n{result.stdout}")
        if result.stderr.strip():
            logger.warning(f"命令错误输出:\n{result.stderr}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")
        logger.error(f"错误输出:\n{e.stderr}")
        raise


def load_config(config_path: str, env: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        env: 环境
        
    Returns:
        配置字典
    """
    logger.info(f"加载配置文件: {config_path}, 环境: {env}")
    
    # 检查是否存在环境特定的配置文件
    base_config_path, ext = os.path.splitext(config_path)
    env_config_path = f"{base_config_path}_{env}{ext}"
    
    config_manager = ConfigManager(config_path)
    
    # 如果存在环境特定配置，加载它
    if os.path.exists(env_config_path):
        logger.info(f"加载环境特定配置: {env_config_path}")
        env_config_manager = ConfigManager(env_config_path)
        # 合并配置
        for section in env_config_manager.get_available_sections():
            config_manager.update_section(section, env_config_manager.get_section(section))
    
    return config_manager.get_full_config()


def validate_environment() -> None:
    """
    验证部署环境
    """
    logger.info("验证部署环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.warning(f"Python版本 {python_version.major}.{python_version.minor} 可能不兼容，推荐使用Python 3.8+")
    
    # 检查项目结构
    required_dirs = ['src', 'config', 'data', 'models']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            logger.error(f"必要的目录不存在: {dir_name}")
            raise FileNotFoundError(f"必要的目录不存在: {dir_name}")
    
    # 检查requirements.txt
    if not os.path.exists('requirements.txt'):
        logger.warning("未找到requirements.txt文件")
    
    logger.info("环境验证通过")


def build_docker_image(config: Dict,
                      image_name: Optional[str] = None,
                      tag: Optional[str] = None,
                      push: bool = False) -> str:
    """
    构建Docker镜像
    
    Args:
        config: 配置字典
        image_name: 镜像名称
        tag: 镜像标签
        push: 是否推送镜像
        
    Returns:
        构建的镜像名称
    """
    logger.info("构建Docker镜像...")
    
    # 获取Docker配置
    docker_config = config.get('deployment', {}).get('docker', {})
    
    # 确定镜像名称和标签
    if not image_name:
        image_name = docker_config.get('image_name', 'timeseries-model-service')
    
    if not tag:
        tag = docker_config.get('tag', 'latest')
    
    # 添加环境和时间戳
    full_tag = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    image_full_name = f"{image_name}:{full_tag}"
    latest_tag = f"{image_name}:latest"
    
    # 检查Docker是否安装
    try:
        run_command('docker --version')
    except subprocess.CalledProcessError:
        logger.error("Docker未安装或不可用")
        raise RuntimeError("Docker未安装或不可用")
    
    # 构建镜像
    dockerfile_path = docker_config.get('dockerfile_path', './Dockerfile')
    
    build_args = []
    for key, value in docker_config.get('build_args', {}).items():
        build_args.append(f"--build-arg {key}={value}")
    
    build_cmd = f"docker build -t {image_full_name} -t {latest_tag} {dockerfile_path}"
    if build_args:
        build_cmd = f"{build_cmd} {' '.join(build_args)}"
    
    run_command(build_cmd)
    
    # 推送镜像（如果需要）
    if push:
        logger.info(f"推送镜像: {image_full_name}")
        run_command(f"docker push {image_full_name}")
        run_command(f"docker push {latest_tag}")
    
    logger.info(f"Docker镜像构建完成: {image_full_name}")
    
    return image_full_name


def deploy_docker(config: Dict,
                  image_name: Optional[str] = None,
                  port: int = 8000,
                  monitoring_port: int = 8001) -> Dict[str, str]:
    """
    使用Docker部署服务
    
    Args:
        config: 配置字典
        image_name: 镜像名称
        port: API端口
        monitoring_port: 监控端口
        
    Returns:
        部署信息
    """
    logger.info("使用Docker部署服务...")
    
    # 获取Docker配置
    docker_config = config.get('deployment', {}).get('docker', {})
    
    # 确定镜像名称
    if not image_name:
        image_name = docker_config.get('image_name', 'timeseries-model-service')
        tag = docker_config.get('tag', 'latest')
        image_name = f"{image_name}:{tag}"
    
    # 服务名称
    service_name = f"timeseries-model-service-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 停止旧服务（如果存在）
    try:
        run_command(f"docker stop {service_name} 2>/dev/null || true")
        run_command(f"docker rm {service_name} 2>/dev/null || true")
    except Exception as e:
        logger.warning(f"停止旧服务失败: {e}")
    
    # 准备卷映射
    volumes = [
        './data:/app/data',
        './models:/app/models',
        './logs:/app/logs',
        './cache:/app/cache'
    ]
    
    # 准备环境变量
    env_vars = []
    # 添加环境变量
    env_vars.append(f"-e TIMESERIES_ENV={config.get('env', 'dev')}")
    
    # 启动容器
    docker_cmd = (
        f"docker run -d "
        f"--name {service_name} "
        f"-p {port}:8000 "
        f"-p {monitoring_port}:8001 "
        f"{' '.join([f'-v {v}' for v in volumes])} "
        f"{' '.join(env_vars)} "
        f"{image_name}"
    )
    
    result = run_command(docker_cmd)
    container_id = result.stdout.strip()
    
    # 等待服务启动
    logger.info("等待服务启动...")
    time.sleep(10)
    
    # 检查服务状态
    try:
        run_command(f"docker logs {container_id} | grep '模型服务已启动'")
        logger.info(f"服务启动成功，容器ID: {container_id}")
    except subprocess.CalledProcessError:
        logger.error("服务启动失败，查看日志以获取详细信息")
        run_command(f"docker logs {container_id}", check=False)
        raise RuntimeError("服务启动失败")
    
    # 部署信息
    deployment_info = {
        'type': 'docker',
        'container_id': container_id,
        'service_name': service_name,
        'image': image_name,
        'api_url': f"http://localhost:{port}",
        'monitoring_url': f"http://localhost:{monitoring_port}",
        'deployed_at': datetime.now().isoformat()
    }
    
    # 保存部署信息
    save_deployment_info(deployment_info)
    
    return deployment_info


def deploy_kubernetes(config: Dict,
                      image_name: Optional[str] = None,
                      namespace: str = 'timeseries',
                      replicas: int = 2) -> Dict[str, str]:
    """
    使用Kubernetes部署服务
    
    Args:
        config: 配置字典
        image_name: 镜像名称
        namespace: 命名空间
        replicas: 副本数
        
    Returns:
        部署信息
    """
    logger.info("使用Kubernetes部署服务...")
    
    # 检查kubectl
    try:
        run_command('kubectl version --client')
    except subprocess.CalledProcessError:
        logger.error("kubectl未安装或不可用")
        raise RuntimeError("kubectl未安装或不可用")
    
    # 获取Kubernetes配置
    k8s_config = config.get('deployment', {}).get('kubernetes', {})
    
    # 确定镜像名称
    if not image_name:
        docker_config = config.get('deployment', {}).get('docker', {})
        image_name = docker_config.get('image_name', 'timeseries-model-service')
        tag = docker_config.get('tag', 'latest')
        image_name = f"{image_name}:{tag}"
    
    # 创建命名空间（如果不存在）
    try:
        run_command(f"kubectl create namespace {namespace} --dry-run=client -o yaml | kubectl apply -f -")
    except Exception as e:
        logger.warning(f"创建命名空间失败: {e}")
    
    # 创建Kubernetes清单
    k8s_manifest = generate_kubernetes_manifest(
        config,
        image_name,
        namespace,
        replicas
    )
    
    # 保存清单文件
    manifest_file = 'kubernetes_manifest.yaml'
    with open(manifest_file, 'w') as f:
        yaml.dump(k8s_manifest, f, default_flow_style=False)
    
    logger.info(f"生成Kubernetes清单: {manifest_file}")
    
    # 应用清单
    run_command(f"kubectl apply -f {manifest_file}")
    
    # 等待部署完成
    logger.info("等待部署完成...")
    run_command(f"kubectl rollout status deployment/timeseries-model-service -n {namespace}")
    
    # 获取服务信息
    service_info = run_command(f"kubectl get service timeseries-model-service -n {namespace} -o json")
    service_data = json.loads(service_info.stdout)
    
    # 确定访问URL
    api_url = "http://localhost:8000"  # 默认
    if service_data['spec']['type'] == 'LoadBalancer':
        # 尝试获取外部IP
        time.sleep(30)  # 等待LoadBalancer分配IP
        service_info = run_command(f"kubectl get service timeseries-model-service -n {namespace} -o json")
        service_data = json.loads(service_info.stdout)
        
        external_ips = service_data['status'].get('loadBalancer', {}).get('ingress', [])
        if external_ips:
            external_ip = external_ips[0].get('ip', external_ips[0].get('hostname', 'localhost'))
            api_url = f"http://{external_ip}:{service_data['spec']['ports'][0]['port']}"
    
    # 部署信息
    deployment_info = {
        'type': 'kubernetes',
        'namespace': namespace,
        'replicas': replicas,
        'image': image_name,
        'api_url': api_url,
        'deployed_at': datetime.now().isoformat()
    }
    
    # 保存部署信息
    save_deployment_info(deployment_info)
    
    return deployment_info


def generate_kubernetes_manifest(config: Dict,
                                 image_name: str,
                                 namespace: str,
                                 replicas: int) -> Dict[str, Any]:
    """
    生成Kubernetes清单
    
    Args:
        config: 配置字典
        image_name: 镜像名称
        namespace: 命名空间
        replicas: 副本数
        
    Returns:
        Kubernetes清单
    """
    k8s_config = config.get('deployment', {}).get('kubernetes', {})
    
    # 创建清单
    manifest = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'timeseries-model-service',
            'namespace': namespace,
            'labels': {
                'app': 'timeseries-model-service'
            }
        },
        'spec': {
            'replicas': replicas,
            'selector': {
                'matchLabels': {
                    'app': 'timeseries-model-service'
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': 'timeseries-model-service'
                    }
                },
                'spec': {
                    'containers': [
                        {
                            'name': 'timeseries-model-service',
                            'image': image_name,
                            'ports': [
                                {'containerPort': 8000},
                                {'containerPort': 8001}
                            ],
                            'resources': k8s_config.get('resources', {
                                'limits': {
                                    'cpu': '1',
                                    'memory': '1Gi'
                                },
                                'requests': {
                                    'cpu': '500m',
                                    'memory': '512Mi'
                                }
                            }),
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            },
                            'env': [
                                {'name': 'TIMESERIES_CONFIG_PATH', 'value': '/app/config/config.yaml'},
                                {'name': 'TIMESERIES_ENV', 'value': config.get('env', 'dev')}
                            ],
                            'volumeMounts': [
                                {
                                    'name': 'config-volume',
                                    'mountPath': '/app/config/'
                                }
                            ]
                        }
                    ],
                    'volumes': [
                        {
                            'name': 'config-volume',
                            'configMap': {
                                'name': 'timeseries-model-config'
                            }
                        }
                    ]
                }
            }
        }
    }
    
    # 添加服务定义
    service = {
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': 'timeseries-model-service',
            'namespace': namespace
        },
        'spec': {
            'selector': {
                'app': 'timeseries-model-service'
            },
            'ports': [
                {
                    'name': 'api',
                    'port': k8s_config.get('service', {}).get('port', 80),
                    'targetPort': 8000
                },
                {
                    'name': 'monitoring',
                    'port': 8001,
                    'targetPort': 8001
                }
            ],
            'type': k8s_config.get('service', {}).get('type', 'LoadBalancer')
        }
    }
    
    # 添加配置映射
    config_map = {
        'apiVersion': 'v1',
        'kind': 'ConfigMap',
        'metadata': {
            'name': 'timeseries-model-config',
            'namespace': namespace
        },
        'data': {
            'config.yaml': yaml.dump(config, default_flow_style=False)
        }
    }
    
    # 如果启用了自动扩缩容
    if k8s_config.get('autoscaling', {}).get('enabled', False):
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'timeseries-model-service',
                'namespace': namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'timeseries-model-service'
                },
                'minReplicas': k8s_config.get('autoscaling', {}).get('min_replicas', 2),
                'maxReplicas': k8s_config.get('autoscaling', {}).get('max_replicas', 10),
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': k8s_config.get('autoscaling', {}).get('cpu_percent', 80)
                            }
                        }
                    }
                ]
            }
        }
        
        return [manifest, service, config_map, hpa]
    
    return [manifest, service, config_map]


def deploy_local(config: Dict,
                 port: int = 8000,
                 monitoring_port: int = 8001) -> Dict[str, str]:
    """
    本地部署服务
    
    Args:
        config: 配置字典
        port: API端口
        monitoring_port: 监控端口
        
    Returns:
        部署信息
    """
    logger.info("本地部署服务...")
    
    # 检查Python环境
    try:
        run_command(f"python -c \"import src.models.model_service; print('Model service module found')\"")
    except subprocess.CalledProcessError:
        logger.error("无法导入模型服务模块")
        raise RuntimeError("无法导入模型服务模块")
    
    # 安装依赖
    if os.path.exists('requirements.txt'):
        logger.info("安装Python依赖...")
        run_command("pip install -r requirements.txt")
    
    # 启动服务（作为后台进程）
    logger.info("启动服务...")
    
    # 检查是否已运行
    try:
        run_command("ps aux | grep 'python -m src.models.model_service' | grep -v grep")
        logger.warning("服务似乎已经在运行")
    except subprocess.CalledProcessError:
        # 服务未运行，可以启动
        pass
    
    # 启动服务
    service_cmd = (
        f"nohup python -m src.models.model_service "
        f"--host 0.0.0.0 "
        f"--port {port} "
        f"--config {config.get('TIMESERIES_CONFIG_PATH', './config/config.yaml')} "
        f"> model_service.log 2>&1 &"
    )
    
    run_command(service_cmd)
    
    # 等待服务启动
    logger.info("等待服务启动...")
    time.sleep(10)
    
    # 检查服务状态
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        response.raise_for_status()
        logger.info("服务启动成功")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        logger.info("查看model_service.log获取详细信息")
        raise RuntimeError("服务启动失败")
    
    # 部署信息
    deployment_info = {
        'type': 'local',
        'api_url': f"http://localhost:{port}",
        'monitoring_url': f"http://localhost:{monitoring_port}",
        'log_file': 'model_service.log',
        'deployed_at': datetime.now().isoformat()
    }
    
    # 保存部署信息
    save_deployment_info(deployment_info)
    
    return deployment_info


def run_tests() -> bool:
    """
    运行测试
    
    Returns:
        测试是否通过
    """
    logger.info("运行测试...")
    
    try:
        # 检查是否有测试目录
        if not os.path.exists('tests'):
            logger.warning("未找到测试目录")
            return True
        
        # 运行单元测试
        result = run_command("python -m pytest tests/ -v")
        return result.returncode == 0
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        return False


def save_deployment_info(deployment_info: Dict[str, Any]) -> None:
    """
    保存部署信息
    
    Args:
        deployment_info: 部署信息
    """
    # 创建部署信息目录
    deploy_info_dir = './deployments/info'
    os.makedirs(deploy_info_dir, exist_ok=True)
    
    # 保存到文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    deploy_info_file = os.path.join(deploy_info_dir, f"deployment_{timestamp}.json")
    
    with open(deploy_info_file, 'w') as f:
        json.dump(deployment_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"部署信息已保存到: {deploy_info_file}")
    
    # 同时更新最新部署信息
    latest_deploy_file = os.path.join(deploy_info_dir, 'latest_deployment.json')
    with open(latest_deploy_file, 'w') as f:
        json.dump(deployment_info, f, indent=2, ensure_ascii=False)


def generate_deployment_report(deployment_info: Dict[str, Any]) -> str:
    """
    生成部署报告
    
    Args:
        deployment_info: 部署信息
        
    Returns:
        部署报告内容
    """
    report = [
        "# 模型服务部署报告",
        f"部署时间: {deployment_info.get('deployed_at', datetime.now().isoformat())}",
        f"部署类型: {deployment_info.get('type', 'unknown')}",
        f"API地址: {deployment_info.get('api_url', 'N/A')}",
    ]
    
    if 'monitoring_url' in deployment_info:
        report.append(f"监控地址: {deployment_info['monitoring_url']}")
    
    if deployment_info.get('type') == 'docker':
        report.extend([
            f"容器ID: {deployment_info.get('container_id', 'N/A')}",
            f"镜像: {deployment_info.get('image', 'N/A')}"
        ])
    
    if deployment_info.get('type') == 'kubernetes':
        report.extend([
            f"命名空间: {deployment_info.get('namespace', 'N/A')}",
            f"副本数: {deployment_info.get('replicas', 'N/A')}"
        ])
    
    report.append("\n## 后续步骤")
    report.append("1. 访问API地址验证服务是否正常运行")
    report.append("2. 检查监控页面查看服务状态")
    report.append("3. 测试API接口功能")
    
    return '\n'.join(report)


def main() -> None:
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        import logging as lg
        lg.getLogger().setLevel(lg.DEBUG)
    
    try:
        # 验证环境
        validate_environment()
        
        # 加载配置
        config = load_config(args.config, args.env)
        # 添加环境信息
        config['env'] = args.env
        
        # 运行测试
        if not args.skip_tests:
            if not run_tests():
                logger.error("测试失败，部署中止")
                sys.exit(1)
        
        # 构建Docker镜像（如果需要）
        image_name = None
        if args.mode in ['docker', 'kubernetes'] and not args.skip_build:
            image_name = build_docker_image(
                config,
                args.docker_image,
                args.docker_tag,
                args.push_image
            )
        elif args.docker_image:
            # 使用指定的镜像
            image_name = f"{args.docker_image}:{args.docker_tag}"
        
        # 部署服务
        deployment_info = {}
        
        if args.mode == 'local':
            deployment_info = deploy_local(
                config,
                args.port,
                args.monitoring_port
            )
        elif args.mode == 'docker':
            deployment_info = deploy_docker(
                config,
                image_name,
                args.port,
                args.monitoring_port
            )
        elif args.mode == 'kubernetes':
            deployment_info = deploy_kubernetes(
                config,
                image_name,
                args.namespace,
                args.replicas
            )
        else:
            logger.error(f"不支持的部署模式: {args.mode}")
            sys.exit(1)
        
        # 生成并显示部署报告
        report = generate_deployment_report(deployment_info)
        print("\n" + "=" * 80)
        print(report)
        print("=" * 80)
        
        logger.info("部署完成")
        
    except Exception as e:
        logger.error(f"部署失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()