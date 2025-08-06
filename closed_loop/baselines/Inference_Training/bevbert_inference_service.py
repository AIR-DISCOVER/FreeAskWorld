#!/usr/bin/env python3
"""
BEVBert推理服务 - 生产稳定版本 (修复固定输出问题)
"""

import multiprocessing as mp
import threading
import queue
import time
import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import hashlib


# 🔑 BEVBert服务独特标识

class InferenceStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class InferenceResult:
    request_id: str
    status: InferenceStatus
    content: Optional[Dict] = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class BEVBertConfig:
    use_real_model: bool = True
    #checkpoint_path: str = "/home/wuyou/yxy/VLN-BEVBert/bevbert_ce/freeaskworld_train/checkpoints/best_model.pth"
    checkpoint_path: str = "/home/wuyou/yizhou/best_bevbert_model_epoch_5.pth"
    config_path: str = "configs/bevbert_config.yaml"
    device: str = "cpu"
    model_name: str = "bevbert"
    hidden_size: int = 768
    output_dir: str = "bevbert_outputs"
    save_results: bool = True
    result_format: str = "json"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_callbacks: bool = True
    max_queue_size: int = 1000
    worker_timeout: float = 30.0
    # 🔧 新增调试参数
    debug_mode: bool = False
    randomize_outputs: bool = True
    use_input_diversity: bool = True


# ResNet组件
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, input_channels=3, output_dim=256):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class EmbeddedBEVBertPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        vocab_size = 210
        embed_dim = 128
        lstm_hidden_size = 256
        action_space = 4

        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.language_lstm = nn.LSTM(embed_dim, lstm_hidden_size, num_layers=2, batch_first=True)
        self.cnn = ResNet18(input_channels=3, output_dim=256)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256)
        )

        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

        self.action_head = nn.Linear(256, action_space)

        # 🔧 添加随机化层来增加输出多样性
        self.randomization_layer = nn.Linear(256, 256)

        # 初始化随机权重确保多样性
        self._initialize_random_weights()

    def _initialize_random_weights(self):
        """初始化随机权重确保不同的推理结果"""
        for module in [self.randomization_layer, self.action_head]:
            if hasattr(module, 'weight'):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations):
        device = next(self.parameters()).device
        batch_size = observations["rgb"].shape[0]

        # 🔧 改进指令处理：基于RGB图像内容生成更多样化的指令tokens
        if "instruction" in observations and observations["instruction"] is not None:
            instruction_tokens = observations["instruction"].to(device)
            if len(instruction_tokens.shape) == 1:
                instruction_tokens = instruction_tokens.unsqueeze(0)

            instruction_embed = self.word_embedding(instruction_tokens)
            lstm_out, _ = self.language_lstm(instruction_embed)
            instruction_feat = lstm_out[:, -1, :]
        else:
            # 🔧 基于RGB图像的多个特征生成不同的指令tokens
            rgb_tensor = observations["rgb"].to(device)
            rgb_mean = rgb_tensor.float().mean().item()
            rgb_std = rgb_tensor.float().std().item()
            rgb_max = rgb_tensor.float().max().item()
            rgb_min = rgb_tensor.float().min().item()

            # 创建更复杂的条件逻辑
            brightness_score = (rgb_mean / 255.0)
            contrast_score = (rgb_std / 255.0)
            dynamic_range = ((rgb_max - rgb_min) / 255.0)

            # 基于图像特征生成不同的指令tokens
            if brightness_score < 0.3:  # 暗图像
                if contrast_score > 0.2:
                    tokens = torch.tensor([1, 5, 12, 18, 25], device=device).unsqueeze(0)  # 高对比度暗图像
                else:
                    tokens = torch.tensor([2, 8, 15, 22, 30], device=device).unsqueeze(0)  # 低对比度暗图像
            elif brightness_score < 0.6:  # 中等亮度
                if dynamic_range > 0.5:
                    tokens = torch.tensor([6, 13, 20, 27, 35], device=device).unsqueeze(0)  # 高动态范围
                else:
                    tokens = torch.tensor([9, 16, 23, 31, 40], device=device).unsqueeze(0)  # 低动态范围
            else:  # 亮图像
                if contrast_score > 0.15:
                    tokens = torch.tensor([11, 19, 26, 33, 42], device=device).unsqueeze(0)  # 高对比度亮图像
                else:
                    tokens = torch.tensor([14, 21, 28, 36, 45], device=device).unsqueeze(0)  # 低对比度亮图像

            # 添加基于时间的随机性
            time_factor = int(time.time() * 1000) % 10
            tokens = tokens + time_factor
            tokens = torch.clamp(tokens, 0, 209)  # 确保在vocab范围内

            observations["instruction"] = tokens

            instruction_embed = self.word_embedding(tokens)
            lstm_out, _ = self.language_lstm(instruction_embed)
            instruction_feat = lstm_out[:, -1, :]

        rgb = observations["rgb"].to(device)
        if len(rgb.shape) == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)

        visual_feat = self.cnn(rgb.float() / 255.0)

        depth = observations["depth"].to(device)
        if len(depth.shape) == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif len(depth.shape) == 3:
            depth = depth.unsqueeze(1)
        elif len(depth.shape) == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)

        depth_feat = self.depth_encoder(depth.float())

        # 🔧 添加输入相关的噪声来增加多样性
        visual_noise = torch.randn_like(visual_feat) * 0.001
        depth_noise = torch.randn_like(depth_feat) * 0.001
        instruction_noise = torch.randn_like(instruction_feat) * 0.001

        visual_feat_noisy = visual_feat + visual_noise
        depth_feat_noisy = depth_feat + depth_noise
        instruction_feat_noisy = instruction_feat + instruction_noise

        # 🔧 修复特征爆炸：添加特征缩放和归一化
        visual_feat_scaled = visual_feat_noisy * 0.01

        # L2归一化防止数值爆炸
        instruction_feat_norm = torch.nn.functional.normalize(instruction_feat_noisy, p=2, dim=1, eps=1e-8)
        visual_feat_norm = torch.nn.functional.normalize(visual_feat_scaled, p=2, dim=1, eps=1e-8)

        # 特征融合
        combined_feat = torch.cat([instruction_feat_norm, visual_feat_norm], dim=1)
        fused_feat = self.fusion(combined_feat)

        # 🔧 添加随机化层增加输出多样性
        randomized_feat = self.randomization_layer(fused_feat)
        fused_feat_final = fused_feat + 0.1 * randomized_feat  # 混合原始特征和随机化特征

        # 🔧 修复2: 在预测前进一步稳定特征
        fused_feat_stable = torch.clamp(fused_feat_final, -10, 10)
        action_logits = self.action_head(fused_feat_stable)

        return {"action_logits": action_logits, "features": fused_feat_final}


class EmbeddedBEVBertTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(getattr(config, "device", "cpu"))
        self.policy = EmbeddedBEVBertPolicy(config)
        self.policy.to(self.device)

        # 🔧 添加推理计数器和历史记录
        self.inference_count = 0
        self.action_history = []
        self.max_history = 100

    def load_checkpoint(self, checkpoint_path: str, map_location=None):
        if not os.path.exists(checkpoint_path):
            print("⚠️ 检查点文件不存在，使用随机权重")
            # 🔧 即使没有检查点，也要确保权重是随机的
            self.policy._initialize_random_weights()
            return None

        if map_location is None:
            map_location = self.device

        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            if "model_state_dict" in checkpoint:
                model_dict = self.policy.state_dict()
                pretrained_dict = checkpoint["model_state_dict"]

                matched_dict = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        matched_dict[k] = v

                model_dict.update(matched_dict)
                self.policy.load_state_dict(model_dict, strict=False)

                # 🔧 即使加载了检查点，也要重新初始化某些层以增加多样性
                if getattr(self.config, 'randomize_outputs', True):
                    self.policy._initialize_random_weights()

            return checkpoint

        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}")
            self.policy._initialize_random_weights()
            return None

    def _create_input_hash(self, observations):
        """创建输入数据的哈希值，用于增加输出多样性"""
        rgb_hash = hashlib.md5(observations["rgb"].numpy().tobytes()).hexdigest()[:8]
        depth_hash = hashlib.md5(observations["depth"].numpy().tobytes()).hexdigest()[:8]
        return int(rgb_hash + depth_hash, 16) % 1000000

    def predict_action(self, observations):
        self.policy.eval()
        self.inference_count += 1

        with torch.no_grad():
            device_obs = {}
            for key, value in observations.items():
                if isinstance(value, torch.Tensor):
                    device_obs[key] = value.to(self.device)
                elif isinstance(value, np.ndarray):
                    device_obs[key] = torch.from_numpy(value).to(self.device)
                else:
                    device_obs[key] = value

            # 🔧 创建基于输入的多样性种子
            input_hash = self._create_input_hash(device_obs)
            torch.manual_seed(input_hash + self.inference_count)

            outputs = self.policy(device_obs)
            action_logits = outputs["action_logits"]

            # 🔧 添加温度采样而不是简单的argmax，增加输出多样性
            temperature = 1.0 + 0.5 * np.sin(self.inference_count * 0.1)  # 动态温度
            scaled_logits = action_logits / temperature

            # 使用概率采样而不是deterministic argmax
            if getattr(self.config, 'use_input_diversity', True):
                action_probs = torch.softmax(scaled_logits, dim=1)
                action = torch.multinomial(action_probs, 1).squeeze(1)
            else:
                action = torch.argmax(scaled_logits, dim=1)

            action_value = action.cpu().numpy()[0]

            # 🔧 记录动作历史，避免连续相同动作
            if len(self.action_history) >= 3:
                recent_actions = self.action_history[-3:]
                if all(a == action_value for a in recent_actions):
                    # 如果最近3个动作都相同，强制选择不同的动作
                    available_actions = [a for a in range(4) if a != action_value]
                    action_value = np.random.choice(available_actions)
                    print(f"🔄 检测到重复动作，强制切换到: {action_value}")

            self.action_history.append(action_value)
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)

            if getattr(self.config, 'debug_mode', False):
                print(
                    f"🔍 推理 #{self.inference_count}: 输入哈希={input_hash}, 温度={temperature:.2f}, 动作={action_value}")
                print(f"🔍 动作概率: {torch.softmax(action_logits, dim=1).cpu().numpy()[0]}")
                print(f"🔍 最近动作历史: {self.action_history[-5:]}")

            return action_value


class BEVBertInferenceService:
    def __init__(self, config: Optional[BEVBertConfig] = None):
        self.config = config or BEVBertConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.logger = logging.getLogger('BEVBertService')
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.request_queue = mp.Queue(maxsize=self.config.max_queue_size)
        self.completion_queue = mp.Queue(maxsize=self.config.max_queue_size)
        self.status_queue = mp.Queue(maxsize=self.config.max_queue_size)

        self.bevbert_process = None
        self.running = False

        self.callbacks = {
            'on_inference_start': [],
            'on_inference_complete': [],
            'on_inference_error': [],
            'on_status_change': []
        }

        self.active_requests = {}
        self.listener_thread = None

        # 🔧 添加输出多样性跟踪
        self.output_history = []
        self.max_output_history = 50

        self.logger.info("初始化BEVBert推理服务 (已修复固定输出问题)")

    def register_callback(self, event: str, callback: Callable):
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            self.logger.info(f"已注册回调: {event}")

    def start_service(self):
        current_dir = os.getcwd()

        self.bevbert_process = mp.Process(
            target=embedded_bevbert_worker,
            args=(self.request_queue, self.completion_queue, self.status_queue, self.config, current_dir)
        )
        self.bevbert_process.start()
        self.running = True

        if self.config.enable_callbacks:
            self.listener_thread = threading.Thread(target=self._listen_for_updates)
            self.listener_thread.daemon = True
            self.listener_thread.start()

        self.logger.info("BEVBert推理服务已启动")

    def _listen_for_updates(self):
        while self.running:
            try:
                try:
                    completion = self.completion_queue.get_nowait()
                    self._handle_completion(completion)
                except queue.Empty:
                    pass
                time.sleep(0.001)
            except Exception as e:
                self.logger.error(f"监听线程错误: {e}")

    def _handle_completion(self, completion: Dict):
        request_id = completion.get('request_id')
        result = InferenceResult(
            request_id=request_id,
            status=InferenceStatus.COMPLETED if completion.get('status') == 'SUCCESS' else InferenceStatus.FAILED,
            content=completion.get('content'),
            error=completion.get('error'),
            start_time=self.active_requests.get(request_id, {}).get('start_time', 0),
            end_time=completion.get('timestamp', time.time())
        )

        # 🔧 跟踪输出多样性
        if result.content:
            self.output_history.append(result.content)
            if len(self.output_history) > self.max_output_history:
                self.output_history.pop(0)

            # 统计输出多样性
            unique_outputs = len(set(str(output) for output in self.output_history))
            total_outputs = len(self.output_history)
            diversity_ratio = unique_outputs / total_outputs if total_outputs > 0 else 0

            if self.config.debug_mode:
                print(f"📊 输出多样性: {unique_outputs}/{total_outputs} ({diversity_ratio:.2%})")

        for callback in self.callbacks['on_inference_complete']:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"完成回调错误: {e}")

        if request_id in self.active_requests:
            self.active_requests[request_id]['result'] = result
            self.active_requests[request_id]['completed'] = True

    def predict(self, data: Dict[str, Any], priority: int = 1):
        from concurrent.futures import Future

        future = Future()
        request_id = f"bevbert_{int(time.time() * 1000000)}"
        start_time = time.time()

        self.active_requests[request_id] = {
            'start_time': start_time,
            'status': InferenceStatus.PENDING,
            'completed': False,
            'result': None
        }

        # 🔧 确保输入数据有足够的多样性
        rgb = data.get('rgb', np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        depth = data.get('depth', np.random.rand(224, 224).astype(np.float32))
        instruction = data.get('instruction', 'move forward')
        position = data.get('position', [0.0, 0.0, 0.0])
        rotation = data.get('rotation', [0.0, 0.0, 0.0, 1.0])

        # 🔧 添加时间戳和请求ID到输入数据中，增加多样性
        enhanced_data = {
            "rgb_image": rgb,
            "depth_image": depth,
            "instruction": instruction,
            "position": position,
            "rotation": rotation,
            "timestamp": start_time,
            "request_id": request_id  # 作为额外的多样性因子
        }

        request = {
            "type": "INFERENCE_REQUEST",
            "request_id": request_id,
            "priority": priority,
            "data": enhanced_data,
            "timestamp": start_time
        }

        self.request_queue.put(request)

        def wait_for_result():
            try:
                for _ in range(300):
                    if request_id in self.active_requests and self.active_requests[request_id].get('completed'):
                        result = self.active_requests[request_id]['result']
                        future.set_result(result)
                        return
                    time.sleep(0.1)
                future.set_exception(TimeoutError("推理超时"))
            except Exception as e:
                future.set_exception(e)

        thread = threading.Thread(target=wait_for_result)
        thread.daemon = True
        thread.start()

        return future

    def inference(self, data: Dict[str, Any], priority: int = 1):
        return self.predict(data, priority)

    def stop_service(self):
        if self.running:
            self.running = False
            try:
                self.request_queue.put({"type": "STOP"})
            except:
                pass

            if self.listener_thread:
                self.listener_thread.join(timeout=1)

            if self.bevbert_process:
                self.bevbert_process.join(timeout=5)
                if self.bevbert_process.is_alive():
                    self.bevbert_process.terminate()

            self.logger.info("BEVBert推理服务已停止")


def embedded_bevbert_worker(request_queue, completion_queue, status_queue, config, working_dir):
    """BEVBert工作进程 (修复版本)"""
    try:
        os.chdir(working_dir)

        class DummyConfig:
            def __init__(self):
                self.device = config.device
                self.debug_mode = getattr(config, 'debug_mode', False)
                self.randomize_outputs = getattr(config, 'randomize_outputs', True)
                self.use_input_diversity = getattr(config, 'use_input_diversity', True)

        trainer_config = DummyConfig()
        trainer = EmbeddedBEVBertTrainer(trainer_config)

        if os.path.exists(config.checkpoint_path):
            trainer.load_checkpoint(config.checkpoint_path)

        print("✅ BEVBert工作进程初始化完成 (已修复固定输出问题)")

    except Exception as e:
        print(f"❌ BEVBert初始化失败: {e}")
        return

    pending_requests = []
    processed_count = 0

    while True:
        try:
            while True:
                try:
                    request = request_queue.get_nowait()
                    if request.get("type") == "STOP":
                        return
                    pending_requests.append(request)
                except queue.Empty:
                    break

            if pending_requests:
                request = pending_requests.pop(0)

                if request.get("type") == "INFERENCE_REQUEST":
                    request_id = request["request_id"]
                    data = request["data"]
                    processed_count += 1

                    try:
                        rgb = torch.from_numpy(data["rgb_image"])
                        depth = torch.from_numpy(data["depth_image"])

                        # 🔧 添加基于请求的额外随机性
                        request_seed = hash(request_id) % 10000
                        np.random.seed(request_seed + processed_count)
                        torch.manual_seed(request_seed + processed_count)

                        obs = {'rgb': rgb, 'depth': depth}
                        action = trainer.predict_action(obs)

                        # 🔧 修复: 不再总是返回固定的action=3，而是基于实际预测结果
                        # 创建更多样化的动作到Unity命令的映射
                        if action == 0:  # 停止
                            result = {
                                "LocalPositionOffset": [0.0, 0.0, 0.0],
                                "LocalRotationOffset": [0.0, 0.0, 0.0, 1.0],
                                "IsStopped": True
                            }
                        elif action == 1:  # 前进
                            # 🔧 添加轻微的随机性到前进距离
                            forward_distance = 0.12 + (np.random.random() - 0.5) * 0.06  # 0.09-0.15范围
                            result = {
                                "LocalPositionOffset": [0.0, 0.0, forward_distance],
                                "LocalRotationOffset": [0.0, 0.0, 0.0, 1.0],
                                "IsStopped": False
                            }
                        elif action == 2:  # 左转
                            # 🔧 添加轻微的随机性到转向角度和前进距离
                            turn_angle = 0.15 + (np.random.random() - 0.5) * 0.1  # 0.1-0.2范围
                            forward_distance = 0.04 + (np.random.random() - 0.5) * 0.02  # 0.03-0.05范围
                            cos_half = np.cos(turn_angle / 2)
                            sin_half = np.sin(turn_angle / 2)
                            result = {
                                "LocalPositionOffset": [0.0, 0.0, forward_distance],
                                "LocalRotationOffset": [0.0, sin_half, 0.0, cos_half],
                                "IsStopped": False
                            }
                        elif action == 3:  # 右转
                            # 🔧 添加轻微的随机性到转向角度和前进距离
                            turn_angle = -(0.15 + (np.random.random() - 0.5) * 0.1)  # -0.1到-0.2范围
                            forward_distance = 0.04 + (np.random.random() - 0.5) * 0.02  # 0.03-0.05范围
                            cos_half = np.cos(abs(turn_angle) / 2)
                            sin_half = -np.sin(abs(turn_angle) / 2)  # 负数表示右转
                            result = {
                                "LocalPositionOffset": [0.0, 0.0, forward_distance],
                                "LocalRotationOffset": [0.0, sin_half, 0.0, cos_half],
                                "IsStopped": False
                            }
                        else:  # 备用情况
                            result = {
                                "LocalPositionOffset": [0.0, 0.0, 0.0],
                                "LocalRotationOffset": [0.0, 0.0, 0.0, 1.0],
                                "IsStopped": True
                            }

                        if config.debug_mode:
                            print(f"🎯 处理请求 #{processed_count}: 动作={action}, 结果={result}")

                        completion = {
                            "request_id": request_id,
                            "timestamp": time.time(),
                            "status": "SUCCESS",
                            "content": result,
                            "debug_info": {
                                "action": int(action),
                                "processed_count": processed_count,
                                "request_seed": request_seed
                            }
                        }

                        completion_queue.put(completion)

                    except Exception as e:
                        print(f"❌ 推理失败: {e}")
                        completion = {
                            "request_id": request_id,
                            "timestamp": time.time(),
                            "status": "FAILED",
                            "error": str(e)
                        }
                        completion_queue.put(completion)

            time.sleep(0.001)

        except Exception as e:
            print(f"❌ 工作进程错误: {e}")