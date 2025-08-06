#!/usr/bin/env python3
"""
可配置的ETPNav推理服务
支持通过配置文件或参数指定权重路径、输出路径等
"""
import multiprocessing as mp
import threading
import queue
import time
import numpy as np
import torch
import json
import cv2
from typing import Dict, Any, Optional, Tuple, Callable
import sys
import os
from dataclasses import dataclass
from enum import Enum
from PIL import Image
from datetime import datetime
import yaml


class InferenceStatus(Enum):
    """推理状态枚举"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class InferenceResult:
    """推理结果数据类"""
    request_id: str
    status: InferenceStatus
    content: Optional[Dict] = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """计算推理耗时"""
        return self.end_time - self.start_time


@dataclass
class ETPNavConfig:
    """ETPNav配置类"""
    # 模型配置
    use_real_model: bool = True
    checkpoint_path: str = "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
    config_path: str = "run_r2r/iter_train.yaml"
    device: str = "cuda"

    # 输出配置
    output_dir: str = "etpnav_outputs"
    save_results: bool = True
    result_format: str = "json"  # json, yaml, both

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # 性能配置
    enable_callbacks: bool = True
    max_queue_size: int = 1000
    worker_timeout: float = 30.0

    @classmethod
    def from_file(cls, config_file: str) -> 'ETPNavConfig':
        """从配置文件加载"""
        if config_file.endswith('.json'):
            with open(config_file, 'r') as f:
                data = json.load(f)
        elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_file}")

        return cls(**data)

    def save(self, config_file: str):
        """保存配置到文件"""
        data = self.__dict__

        if config_file.endswith('.json'):
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
            with open(config_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)


class ConfigurableETPNavService:
    """可配置的ETPNav推理服务"""

    def __init__(self, config: Optional[ETPNavConfig] = None, config_file: Optional[str] = None):
        """
        初始化服务

        Args:
            config: ETPNavConfig对象
            config_file: 配置文件路径
        """
        # 加载配置
        if config_file:
            self.config = ETPNavConfig.from_file(config_file)
        elif config:
            self.config = config
        else:
            self.config = ETPNavConfig()

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 通信队列
        self.request_queue = mp.Queue(maxsize=self.config.max_queue_size)
        self.completion_queue = mp.Queue(maxsize=self.config.max_queue_size)
        self.status_queue = mp.Queue(maxsize=self.config.max_queue_size)

        # 进程管理
        self.etpnav_process = None
        self.running = False

        # 回调管理
        self.callbacks = {
            'on_inference_start': [],
            'on_inference_complete': [],
            'on_inference_error': [],
            'on_status_change': []
        }

        # 状态追踪
        self.active_requests = {}

        # 监听线程
        self.listener_thread = None

        # 日志
        self.logger.info(f"初始化ETPNav服务")
        self.logger.info(f"配置: {self.config}")

    def _setup_logging(self):
        """设置日志"""
        import logging

        self.logger = logging.getLogger('ETPNavService')
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

        # 文件处理器
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            self.logger.info(f"已注册回调: {event} -> {callback.__name__}")

    def start_service(self):
        """启动ETPNav服务"""
        self.etpnav_process = mp.Process(
            target=self._etpnav_worker_process,
            args=(self.request_queue, self.completion_queue, self.status_queue, self.config)
        )
        self.etpnav_process.start()
        self.running = True

        # 启动监听线程
        if self.config.enable_callbacks:
            self.listener_thread = threading.Thread(target=self._listen_for_updates)
            self.listener_thread.daemon = True
            self.listener_thread.start()

        self.logger.info("ETPNav服务已启动")
        self.logger.info(f"模型模式: {'真实ETPNav' if self.config.use_real_model else 'Fallback'}")
        self.logger.info(f"权重路径: {self.config.checkpoint_path}")
        self.logger.info(f"输出目录: {self.config.output_dir}")

    def _listen_for_updates(self):
        """监听状态更新并触发回调"""
        while self.running:
            try:
                # 检查状态队列
                try:
                    status_update = self.status_queue.get_nowait()
                    self._handle_status_update(status_update)
                except queue.Empty:
                    pass

                # 检查完成队列
                try:
                    completion = self.completion_queue.get_nowait()
                    self._handle_completion(completion)
                except queue.Empty:
                    pass

                time.sleep(0.001)

            except Exception as e:
                self.logger.error(f"监听线程错误: {e}")

    def _handle_status_update(self, update: Dict):
        """处理状态更新"""
        request_id = update.get('request_id')
        status = update.get('status')

        if request_id in self.active_requests:
            self.active_requests[request_id]['status'] = status

        for callback in self.callbacks['on_status_change']:
            try:
                callback(request_id, status, update)
            except Exception as e:
                self.logger.error(f"状态回调错误: {e}")

    def _handle_completion(self, completion: Dict):
        """处理推理完成"""
        request_id = completion.get('request_id')

        result = InferenceResult(
            request_id=request_id,
            status=InferenceStatus.COMPLETED if completion.get('status') == 'SUCCESS' else InferenceStatus.FAILED,
            content=completion.get('content'),
            error=completion.get('error'),
            start_time=self.active_requests.get(request_id, {}).get('start_time', 0),
            end_time=completion.get('timestamp', time.time())
        )

        # 保存结果
        if self.config.save_results:
            self._save_result(result)

        # 触发回调
        if result.status == InferenceStatus.COMPLETED:
            for callback in self.callbacks['on_inference_complete']:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"完成回调错误: {e}")
        else:
            for callback in self.callbacks['on_inference_error']:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"错误回调错误: {e}")

        if request_id in self.active_requests:
            self.active_requests[request_id]['result'] = result
            self.active_requests[request_id]['completed'] = True

    def _save_result(self, result: InferenceResult):
        """保存推理结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{result.request_id}_{timestamp}"

        result_data = {
            "request_id": result.request_id,
            "status": result.status.value,
            "duration": result.duration,
            "content": result.content,
            "error": result.error,
            "timestamp": time.time()
        }

        # 保存JSON
        if self.config.result_format in ['json', 'both']:
            json_path = os.path.join(self.config.output_dir, f"{base_name}.json")
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            self.logger.debug(f"结果已保存: {json_path}")

        # 保存YAML
        if self.config.result_format in ['yaml', 'both']:
            yaml_path = os.path.join(self.config.output_dir, f"{base_name}.yaml")
            with open(yaml_path, 'w') as f:
                yaml.dump(result_data, f, default_flow_style=False)
            self.logger.debug(f"结果已保存: {yaml_path}")

    def send_inference_request(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                               instruction: str, position: list, rotation: list,
                               priority: int = 0) -> str:
        """发送推理请求"""
        request_id = f"etpnav_{int(time.time() * 1000000)}"
        start_time = time.time()

        self.active_requests[request_id] = {
            'start_time': start_time,
            'status': InferenceStatus.PENDING,
            'completed': False,
            'result': None
        }

        request = {
            "type": "INFERENCE_REQUEST",
            "request_id": request_id,
            "priority": priority,
            "data": {
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "instruction": instruction,
                "position": position,
                "rotation": rotation
            },
            "timestamp": start_time
        }

        self.request_queue.put(request)

        for callback in self.callbacks['on_inference_start']:
            try:
                callback(request_id, request['data'])
            except Exception as e:
                self.logger.error(f"开始回调错误: {e}")

        self.logger.info(f"推理请求已发送: {request_id}")

        return request_id

    def get_inference_result(self, request_id: str, timeout: float = 0.0) -> Optional[InferenceResult]:
        """获取推理结果"""
        if request_id not in self.active_requests:
            return None

        if timeout == 0:
            request_info = self.active_requests.get(request_id)
            return request_info.get('result') if request_info else None
        else:
            start_time = time.time()
            while time.time() - start_time < timeout:
                request_info = self.active_requests.get(request_id)
                if request_info and request_info.get('completed'):
                    return request_info.get('result')
                time.sleep(0.001)
            return None

    def stop_service(self):
        """停止服务"""
        if self.running:
            self.running = False
            self.request_queue.put({"type": "STOP"})

            if self.listener_thread:
                self.listener_thread.join(timeout=1)

            if self.etpnav_process:
                self.etpnav_process.join(timeout=5)
                if self.etpnav_process.is_alive():
                    self.etpnav_process.terminate()

            self.logger.info("ETPNav服务已停止")

    @staticmethod
    def _etpnav_worker_process(request_queue, completion_queue, status_queue, config: ETPNavConfig):
        # 多进程habitat.config修补
        import sys
        import os
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"🔧 工作进程路径修补: {current_dir}")

        """ETPNav工作进程"""
        print(f"🎯 ETPNav工作进程启动")
        print(f"   配置: {config.checkpoint_path}")

        # 初始化工作器
        from real_etpnav_service import RealETPNavWorker

        # 修改RealETPNavWorker以接受配置
        class ConfigurableETPNavWorker(RealETPNavWorker):
            def __init__(self, config: ETPNavConfig):
                self.config = config
                self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
                self.use_real_model = config.use_real_model
                self.trainer = None
                self.model_loaded = False

                print(f"🖥️ 设备: {self.device}")

                if self.use_real_model:
                    if self.load_etpnav_model_with_config():
                        print("✅ 真实ETPNav模型加载成功")
                        self.model_loaded = True
                    else:
                        print("⚠️ ETPNav模型加载失败，使用智能fallback模式")
                        self.model_loaded = False
                else:
                    print("📌 使用Fallback模式")
                    self.model_loaded = False

            def load_etpnav_model_with_config(self):
                """使用配置加载模型"""
                print(f"🔄 正在加载模型: {self.config.checkpoint_path}")
                print(f"📄 使用配置文件: {self.config.config_path}")

                try:
                    # 检查权重文件
                    if not os.path.exists(self.config.checkpoint_path):
                        print(f"❌ 权重文件不存在: {self.config.checkpoint_path}")
                        return False

                    # 检查配置文件
                    if not os.path.exists(self.config.config_path):
                        print(f"❌ 配置文件不存在: {self.config.config_path}")
                        print(f"⚠️  将使用默认配置文件")

                    # 设置要使用的配置路径
                    self.model_config_path = self.config.config_path
                    self.checkpoint_path = self.config.checkpoint_path

                    # 继续原有的加载逻辑
                    return self.load_etpnav_model()

                except Exception as e:
                    print(f"❌ 模型加载失败: {e}")
                    return False

        worker = ConfigurableETPNavWorker(config)

        # 处理请求
        pending_requests = []

        while True:
            try:
                # 收集新请求
                while True:
                    try:
                        request = request_queue.get_nowait()
                        if request.get("type") == "STOP":
                            print("🛑 工作进程收到停止信号")
                            return
                        pending_requests.append(request)
                    except queue.Empty:
                        break

                # 按优先级排序
                pending_requests.sort(key=lambda x: -x.get('priority', 0))

                # 处理请求
                if pending_requests:
                    request = pending_requests.pop(0)

                    if request.get("type") == "INFERENCE_REQUEST":
                        request_id = request["request_id"]
                        data = request["data"]

                        status_queue.put({
                            'request_id': request_id,
                            'status': InferenceStatus.PROCESSING,
                            'timestamp': time.time()
                        })

                        print(f"📥 处理请求: {request_id}")

                        try:
                            result = worker.process_direct_data(
                                rgb_image=data["rgb_image"],
                                depth_image=data["depth_image"],
                                instruction=data["instruction"],
                                position=data["position"],
                                rotation=data["rotation"]
                            )

                            completion = {
                                "request_id": request_id,
                                "timestamp": time.time(),
                                "status": "SUCCESS",
                                "content": result
                            }

                            completion_queue.put(completion)
                            print(f"📤 推理完成: {request_id}")

                        except Exception as e:
                            completion = {
                                "request_id": request_id,
                                "timestamp": time.time(),
                                "status": "FAILED",
                                "error": str(e)
                            }
                            completion_queue.put(completion)
                            print(f"❌ 推理失败: {request_id} - {e}")

                time.sleep(0.001)

            except Exception as e:
                print(f"❌ 工作进程错误: {e}")
                import traceback
                traceback.print_exc()


# 便捷的工厂函数
def create_etpnav_service(config_file: Optional[str] = None, **kwargs) -> ConfigurableETPNavService:
    """
    创建ETPNav服务

    Args:
        config_file: 配置文件路径
        **kwargs: 直接传递的配置参数

    Returns:
        ConfigurableETPNavService实例
    """
    if config_file:
        return ConfigurableETPNavService(config_file=config_file)
    else:
        config = ETPNavConfig(**kwargs)
        return ConfigurableETPNavService(config=config)


# 示例配置文件生成
def generate_example_config(output_file: str = "etpnav_config.yaml"):
    """生成示例配置文件"""
    config = ETPNavConfig()
    config.save(output_file)
    print(f"示例配置文件已生成: {output_file}")


if __name__ == "__main__":
    # 生成示例配置
    generate_example_config()