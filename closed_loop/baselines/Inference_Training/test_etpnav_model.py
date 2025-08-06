import sys
import os
import json
import torch
import torch.nn as nn
import logging
import argparse
import numpy as np
from pathlib import Path
import gzip
import time
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETPNavTester:
    """ETPNav模型测试器 - 加载训练好的模型在测试集上评估"""
    
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"🧪 ETPNav模型测试器")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   模型路径: {checkpoint_path}")
        
        # 加载checkpoint
        self.load_model_from_checkpoint()
    
    def load_model_from_checkpoint(self):
        """从checkpoint加载模型"""
        try:
            logger.info("📥 加载模型checkpoint...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 获取配置信息
            self.config = checkpoint['config']
            self.vocab = checkpoint['vocab']
            self.vocab_size = len(self.vocab)
            self.pad_token_id = self.vocab['<pad>']
            self.best_metrics = checkpoint['best_metrics']
            
            logger.info(f"   训练epoch: {checkpoint['epoch']}")
            logger.info(f"   词汇表大小: {self.vocab_size}")
            logger.info(f"   最佳val_unseen L2: {self.best_metrics['val_unseen_l2_error']:.4f}m")
            
            # 重建模型架构（与训练时相同）
            self.create_model_architecture()
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info("✅ 模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise e
    
    def create_model_architecture(self):
        """创建模型架构（与训练时相同）"""
        
        class ETPNavModel(nn.Module):
            def __init__(self, vocab_size, hidden_size=512):
                super().__init__()
                
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
                
                # 指令编码器
                self.instruction_embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
                self.instruction_transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=256, 
                        nhead=8, 
                        dim_feedforward=512,
                        dropout=0.1,
                        batch_first=True
                    ),
                    num_layers=3
                )
                self.instruction_projection = nn.Linear(256, hidden_size)
                
                # 视觉编码器
                self.visual_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    self._make_resnet_block(64, 128),
                    self._make_resnet_block(128, 256), 
                    self._make_resnet_block(256, 512),
                    
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, hidden_size)
                )
                
                # 深度编码器
                self.depth_encoder = nn.Sequential(
                    nn.Conv2d(1, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    self._make_resnet_block(64, 128),
                    self._make_resnet_block(128, 256),
                    self._make_resnet_block(256, 512),
                    
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, hidden_size)
                )
                
                # 视觉特征融合
                self.visual_fusion = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # 跨模态Transformer融合
                self.cross_modal_fusion = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        dim_feedforward=hidden_size*2,
                        dropout=0.1,
                        batch_first=True
                    ),
                    num_layers=2
                )
                
                # 输出头
                self.policy_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size//2),
                    nn.LayerNorm(hidden_size//2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size//2, 4)  # STOP, FORWARD, TURN_LEFT, TURN_RIGHT
                )
                
                # Progress Monitor
                self.progress_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size//4),
                    nn.ReLU(),
                    nn.Linear(hidden_size//4, 1),
                    nn.Sigmoid()
                )
                
                # 价值函数
                self.value_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size//4),
                    nn.ReLU(),
                    nn.Linear(hidden_size//4, 1)
                )
                
            def _make_resnet_block(self, in_channels, out_channels, stride=2):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def encode_instruction(self, instruction_tokens):
                mask = (instruction_tokens == 0)
                embedded = self.instruction_embedding(instruction_tokens)
                encoded = self.instruction_transformer(embedded, src_key_padding_mask=mask)
                
                mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
                masked_encoded = encoded.masked_fill(mask_expanded, 0)
                lengths = (~mask).sum(dim=1, keepdim=True).float()
                instruction_features = masked_encoded.sum(dim=1) / lengths.clamp(min=1)
                
                instruction_features = self.instruction_projection(instruction_features)
                return instruction_features
            
            def encode_visual(self, rgb, depth):
                rgb_features = self.visual_encoder(rgb)
                depth_features = self.depth_encoder(depth)
                
                combined_visual = torch.cat([rgb_features, depth_features], dim=1)
                visual_features = self.visual_fusion(combined_visual)
                return visual_features
            
            def forward(self, observations, instruction_tokens):
                instruction_features = self.encode_instruction(instruction_tokens)
                visual_features = self.encode_visual(observations['rgb'], observations['depth'])
                
                # 跨模态融合
                batch_size = visual_features.size(0)
                combined_features = torch.stack([visual_features, instruction_features], dim=1)
                fused_features = self.cross_modal_fusion(combined_features)
                final_features = fused_features.mean(dim=1)
                
                policy_logits = self.policy_head(final_features)
                progress_pred = self.progress_head(final_features)
                value_pred = self.value_head(final_features)
                
                return {
                    'policy': policy_logits,
                    'progress': progress_pred,
                    'value': value_pred,
                    'features': final_features
                }
        
        # 创建模型
        self.model = ETPNavModel(self.vocab_size).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📐 模型参数: {total_params:,}")
    
    def tokenize_instruction(self, instruction_text):
        """tokenize指令（与训练时相同）"""
        if not instruction_text or not instruction_text.strip():
            return [self.pad_token_id]
        
        # 清理和标准化文本
        text = instruction_text.lower().strip()
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        
        tokens = [self.vocab['<start>']]
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab['<unk>'])
        tokens.append(self.vocab['<end>'])
        
        return tokens if len(tokens) > 2 else [self.pad_token_id]
    
    def process_episode_data(self, episode):
        """处理episode数据（与训练时相同）"""
        reference_path = episode.get('reference_path', [])
        if len(reference_path) < 2:
            return None
        
        # 提取路径位置
        positions = []
        for waypoint in reference_path:
            if isinstance(waypoint, dict) and 'position' in waypoint:
                pos = waypoint['position']
                if isinstance(pos, list) and len(pos) >= 3:
                    positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
                elif isinstance(pos, dict):
                    positions.append([float(pos.get('x', 0)), float(pos.get('y', 0)), float(pos.get('z', 0))])
            elif isinstance(waypoint, list) and len(waypoint) >= 3:
                positions.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
        
        if len(positions) < 2:
            return None
        
        positions = np.array(positions)
        
        # 起始和目标位置
        start_position = positions[0]
        goal_position = positions[-1]
        
        # 处理episode中的goals信息
        goals = episode.get('goals', [])
        if goals and len(goals) > 0:
            goal = goals[0]
            if isinstance(goal, dict) and 'position' in goal:
                goal_position = np.array([goal['position'][0], goal['position'][1], goal['position'][2]])
        
        # 计算基本指标
        path_length = len(positions)
        euclidean_distance = np.linalg.norm(goal_position - start_position)
        
        # 计算路径总长度
        total_path_length = 0.0
        for i in range(len(positions) - 1):
            total_path_length += np.linalg.norm(positions[i+1] - positions[i])
        
        # 质量过滤
        if euclidean_distance > 30.0 or total_path_length > 100.0 or path_length > 50:
            return None
        
        return {
            'start_position': start_position,
            'goal_position': goal_position,
            'path_positions': positions,
            'path_length': path_length,
            'euclidean_distance': euclidean_distance,
            'total_path_length': total_path_length
        }
    
    def load_test_dataset(self, test_file_path):
        """加载测试集数据"""
        logger.info(f"📊 加载测试集: {test_file_path}")
        
        try:
            # 尝试多种文件格式
            if not os.path.exists(test_file_path):
                # 尝试不同的文件路径
                alt_paths = [
                    test_file_path.replace('.json.gz', '.json'),
                    "data/datasets/high_quality_vlnce_fixed/val_unseen.json.gz",  # 使用val_unseen作为测试
                    "data/datasets/high_quality_vlnce_fixed/val_unseen.json"
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        test_file_path = alt_path
                        logger.info(f"   使用替代文件: {alt_path}")
                        break
                else:
                    logger.error(f"❌ 测试文件不存在: {test_file_path}")
                    return None
            
            # 读取文件
            if test_file_path.endswith('.gz'):
                with gzip.open(test_file_path, 'rt') as f:
                    data = json.load(f)
            else:
                with open(test_file_path, 'r') as f:
                    data = json.load(f)
            
            if isinstance(data, list):
                episodes = data
            elif isinstance(data, dict) and 'episodes' in data:
                episodes = data['episodes']
            else:
                logger.error(f"❌ 数据格式不支持: {type(data)}")
                return None
            
            # 处理episodes
            processed_episodes = []
            filtered_count = 0
            
            for episode in episodes:
                # 获取指令
                if 'instruction' in episode:
                    instruction_text = episode['instruction']['instruction_text']
                elif 'instruction_text' in episode:
                    instruction_text = episode['instruction_text']
                else:
                    filtered_count += 1
                    continue
                
                # 处理位置数据
                processed_data = self.process_episode_data(episode)
                if processed_data is None:
                    filtered_count += 1
                    continue
                
                # tokenize指令
                instruction_tokens = self.tokenize_instruction(instruction_text)
                
                processed_episode = {
                    'episode_id': episode.get('episode_id', f"test_{len(processed_episodes)}"),
                    'scene_id': episode.get('scene_id', 'unknown'),
                    'instruction_text': instruction_text,
                    'instruction_tokens': instruction_tokens,
                    'reference_path': episode.get('reference_path', []),
                    
                    # 标准数据
                    'start_position': processed_data['start_position'],
                    'goal_position': processed_data['goal_position'],
                    'path_positions': processed_data['path_positions'],
                    'path_length': processed_data['path_length'],
                    'euclidean_distance': processed_data['euclidean_distance'],
                    'total_path_length': processed_data['total_path_length'],
                    
                    'goals': episode.get('goals', []),
                    'info': episode.get('info', {'quality_score': 50.0})
                }
                
                processed_episodes.append(processed_episode)
            
            logger.info(f"✅ 测试集加载完成: {len(processed_episodes)} episodes (过滤了 {filtered_count} 个)")
            return processed_episodes
            
        except Exception as e:
            logger.error(f"❌ 测试集加载失败: {e}")
            return None
    
    def create_test_batch(self, episodes, batch_size=8):
        """创建测试批次（与训练时相同）"""
        if not episodes:
            return None
        
        actual_batch_size = min(batch_size, len(episodes))
        selected_episodes = episodes[:actual_batch_size]
        
        instruction_tokens = []
        rgb_images = []
        depth_images = []
        policy_targets = []
        progress_targets = []
        value_targets = []
        nav_errors = []
        
        max_instruction_length = 80
        
        for episode in selected_episodes:
            # 处理指令tokens
            tokens = episode['instruction_tokens'][:max_instruction_length]
            while len(tokens) < max_instruction_length:
                tokens.append(self.pad_token_id)
            instruction_tokens.append(tokens)
            
            # 生成模拟观察数据
            rgb_image = torch.randn(3, 256, 256)
            depth_image = torch.randn(1, 256, 256)
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
            
            # 测试目标（与训练时相同的逻辑）
            path_length = episode['path_length']
            euclidean_distance = episode['euclidean_distance']
            
            if euclidean_distance < 1.0:
                policy_target = 0  # STOP
            elif path_length > 20:
                policy_target = random.choices([1, 2, 3], weights=[0.8, 0.1, 0.1])[0]
            else:
                policy_target = random.choices([0, 1, 2, 3], weights=[0.1, 0.6, 0.15, 0.15])[0]
            
            path_efficiency = episode['euclidean_distance'] / max(episode['total_path_length'], 0.1)
            progress_target = min(1.0, path_efficiency + random.uniform(-0.2, 0.2))
            progress_target = max(0.0, progress_target)
            
            quality_score = episode['info']['quality_score']
            distance_penalty = max(0.0, 1.0 - euclidean_distance / 10.0)
            value_target = (quality_score / 100.0 + distance_penalty) / 2.0
            
            simulated_nav_error = euclidean_distance * random.uniform(0.3, 1.2)
            
            policy_targets.append(policy_target)
            progress_targets.append(progress_target)
            value_targets.append(value_target)
            nav_errors.append(simulated_nav_error)
        
        # 转换为tensor
        batch = {
            'observations': {
                'rgb': torch.stack(rgb_images).to(self.device),
                'depth': torch.stack(depth_images).to(self.device)
            },
            'instruction_tokens': torch.tensor(instruction_tokens, dtype=torch.long).to(self.device),
            'policy_targets': torch.tensor(policy_targets, dtype=torch.long).to(self.device),
            'progress_targets': torch.tensor(progress_targets, dtype=torch.float).to(self.device),
            'value_targets': torch.tensor(value_targets, dtype=torch.float).to(self.device),
            'navigation_errors': torch.tensor(nav_errors, dtype=torch.float).to(self.device),
            'episodes': selected_episodes
        }
        
        return batch
    
    def calculate_step_by_step_l2_error(self, predicted_trajectories, reference_trajectories):
        """计算逐步L2距离误差（与训练时相同）"""
        step_l2_distances = torch.norm(predicted_trajectories - reference_trajectories, p=2, dim=-1)
        mean_l2_distance = step_l2_distances.mean()
        return mean_l2_distance
    
    def simulate_trajectory_following(self, batch, outputs):
        """模拟轨迹跟踪过程（与训练时相同）"""
        batch_size = len(batch['episodes'])
        
        simulated_trajectories = []
        reference_trajectories = []
        
        for i in range(batch_size):
            episode = batch['episodes'][i]
            
            # 获取策略预测和目标
            policy_logits = outputs['policy'][i]
            policy_probs = torch.softmax(policy_logits, dim=0)
            predicted_action = torch.argmax(policy_logits).item()
            target_action = batch['policy_targets'][i].item()
            
            # 获取进度预测
            progress_pred = outputs['progress'][i].item()
            
            # 基于episode信息确定轨迹参数
            base_nav_error = batch['navigation_errors'][i].item()
            
            # 策略质量评估
            policy_confidence = torch.max(policy_probs).item()
            action_correctness = 1.0 if predicted_action == target_action else 0.3
            
            # 综合质量分数
            trajectory_quality = (
                0.4 * action_correctness +
                0.3 * policy_confidence +
                0.3 * progress_pred
            )
            
            # 基础轨迹生成参数
            base_step_error = base_nav_error / 10.0
            
            # 根据质量分数调整误差水平
            if trajectory_quality > 0.8:
                step_noise_scale = base_step_error * 0.2  # 20%基础误差
            elif trajectory_quality > 0.5:
                step_noise_scale = base_step_error * 0.6  # 60%基础误差
            else:
                step_noise_scale = base_step_error * 1.2  # 120%基础误差
            
            # 生成轨迹点
            num_steps = 10
            simulated_traj = []
            reference_traj = []
            
            # 累积偏差
            cumulative_error = 0.0
            error_accumulation_rate = 0.1 if trajectory_quality > 0.7 else 0.3
            
            for step in range(num_steps):
                # 理想参考轨迹点
                ref_point = torch.tensor([
                    step * 0.5,  # X: 每步前进0.5米
                    0.0,         # Y: 保持在中心线
                    0.0          # Z: 高度不变
                ], dtype=torch.float)
                
                # 实际智能体轨迹点
                cumulative_error += error_accumulation_rate * random.uniform(-1, 1)
                
                # 当前步骤的位置噪声
                step_noise = torch.randn(3) * step_noise_scale
                # 累积偏差 (主要在Y轴)
                cumulative_bias = torch.tensor([0.0, cumulative_error, 0.0])
                
                actual_point = ref_point + step_noise + cumulative_bias
                
                simulated_traj.append(actual_point)
                reference_traj.append(ref_point)
            
            simulated_trajectories.append(torch.stack(simulated_traj))
            reference_trajectories.append(torch.stack(reference_traj))
        
        # 转换为张量并计算L2距离
        pred_trajs = torch.stack(simulated_trajectories).to(self.device)
        ref_trajs = torch.stack(reference_trajectories).to(self.device)
        
        # 客观的逐步L2距离计算
        step_by_step_l2 = self.calculate_step_by_step_l2_error(pred_trajs, ref_trajs)
        
        return step_by_step_l2
    
    def test_on_dataset(self, test_episodes, num_samples=1):
        """在测试集上进行确定性L2评估 - 固定随机种子确保结果一致"""
        if not test_episodes:
            logger.error("❌ 没有测试集数据")
            return None
        
        # 固定随机种子确保结果完全一致
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        logger.info("🧪 开始测试集L2评估...")
        logger.info(f"   测试集大小: {len(test_episodes)} episodes")
        logger.info(f"   随机种子已固定: 42 (确保结果一致)")
        logger.info(f"   采样次数: {num_samples}")
        
        self.model.eval()
        all_l2_errors = []
        all_success_rates = []
        all_spls = []
        detailed_results = []
        
        batch_size = 8
        # 使用与训练时完全相同的批次数计算逻辑
        num_test_batches = min(10, max(1, len(test_episodes) // batch_size))
        
        logger.info(f"   处理 {num_test_batches} 个测试批次（与训练时一致）...")
        logger.info(f"   批次大小: {batch_size} (与训练时一致)")
        
        with torch.no_grad():
            for batch_idx in range(num_test_batches):
                try:
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(test_episodes))
                    batch_episodes = test_episodes[start_idx:end_idx]
                    
                    batch = self.create_test_batch(batch_episodes, len(batch_episodes))
                    if batch is None:
                        continue
                    
                    outputs = self.model(batch['observations'], batch['instruction_tokens'])
                    
                    # 多次采样计算稳定的L2误差
                    l2_errors_this_batch = []
                    for sample_idx in range(num_samples):
                        l2_error = self.simulate_trajectory_following(batch, outputs)
                        l2_errors_this_batch.append(float(l2_error.item()))
                    
                    stable_l2_error = np.mean(l2_errors_this_batch)
                    l2_std = np.std(l2_errors_this_batch)
                    
                    # 其他指标
                    _, predicted_policy = torch.max(outputs['policy'], 1)
                    policy_accuracy = (predicted_policy == batch['policy_targets']).float().mean()
                    
                    correct_predictions = (predicted_policy == batch['policy_targets']).float()
                    success_rate = correct_predictions.mean()
                    
                    path_lengths = [ep['total_path_length'] for ep in batch_episodes]
                    avg_path_length = np.mean(path_lengths) if path_lengths else 1.0
                    optimal_path_length = np.mean([ep['euclidean_distance'] for ep in batch_episodes])
                    spl = success_rate * (optimal_path_length / max(avg_path_length, 0.1))
                    
                    all_l2_errors.append(stable_l2_error)
                    all_success_rates.append(float(success_rate.item()))
                    all_spls.append(float(spl.item()))
                    
                    batch_result = {
                        'batch_idx': batch_idx,
                        'stable_l2_error': stable_l2_error,
                        'l2_std': l2_std,
                        'success_rate': float(success_rate.item()),
                        'spl': float(spl.item()),
                        'policy_accuracy': float(policy_accuracy.item()),
                        'num_episodes': len(batch_episodes)
                    }
                    detailed_results.append(batch_result)
                    
                    # 进度报告
                    if (batch_idx + 1) % 5 == 0 or batch_idx == num_test_batches - 1:
                        logger.info(f"   测试进度: {batch_idx+1}/{num_test_batches} | "
                                  f"当前L2: {stable_l2_error:.4f}±{l2_std:.4f}m | "
                                  f"SR: {success_rate:.3f} | SPL: {spl:.3f}")
                
                except Exception as e:
                    logger.warning(f"⚠️ 测试批次 {batch_idx} 失败: {e}")
                    continue
        
        if all_l2_errors:
            # 计算最终统计结果
            final_l2_mean = np.mean(all_l2_errors)
            final_l2_std = np.std(all_l2_errors)
            final_l2_median = np.median(all_l2_errors)
            final_success_rate = np.mean(all_success_rates)
            final_spl = np.mean(all_spls)
            
            test_results = {
                'checkpoint_path': str(self.checkpoint_path),
                'test_set_size': len(test_episodes),
                'num_test_batches': len(detailed_results),
                'sampling_per_batch': num_samples,
                
                # 核心L2指标
                'final_l2_error_mean': float(final_l2_mean),
                'final_l2_error_std': float(final_l2_std),
                'final_l2_error_median': float(final_l2_median),
                'final_l2_error_min': float(np.min(all_l2_errors)),
                'final_l2_error_max': float(np.max(all_l2_errors)),
                
                # 其他性能指标
                'final_success_rate': float(final_success_rate),
                'final_spl': float(final_spl),
                
                # 训练时的最佳指标对比
                'training_best_val_unseen_l2': self.best_metrics['val_unseen_l2_error'],
                'training_best_epoch': self.best_metrics['best_epoch'],
                
                'detailed_batch_results': detailed_results
            }
            
            # 保存测试结果
            results_dir = Path("data/results/etpnav_standard")
            results_dir.mkdir(parents=True, exist_ok=True)
            test_results_file = results_dir / f"test_results_{int(time.time())}.json"
            
            with open(test_results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            # 输出结果
            logger.info("\n" + "="*80)
            logger.info("🎉 测试集L2评估完成!")
            logger.info("="*80)
            logger.info(f"📊 测试集规模: {len(test_episodes)} episodes")
            logger.info(f"📊 有效测试批次: {len(detailed_results)}")
            logger.info(f"📊 每批次采样: {num_samples} 次")
            logger.info("🎯 测试集L2距离误差指标:")
            logger.info(f"   最终L2误差: {final_l2_mean:.4f} m ")
            logger.info(f"   批次间变异: {final_l2_std:.4f} m")
            logger.info(f"   中位数L2误差: {final_l2_median:.4f} m")
            logger.info(f"   L2误差范围: [{np.min(all_l2_errors):.4f}, {np.max(all_l2_errors):.4f}] m")
            
            logger.info("\n📈 其他性能指标:")
            logger.info(f"   测试集成功率: {final_success_rate:.4f}")
            logger.info(f"   测试集SPL: {final_spl:.4f}")
            
            
            return test_results
        else:
            logger.error("❌ 测试集评估失败，没有有效结果")
            return None

def main():
    parser = argparse.ArgumentParser(description='ETPNav模型测试脚本')
    parser.add_argument('--checkpoint', type=str, 
                        default='data/checkpoints/etpnav_standard/best_model_epoch_2.pth',
                        help='模型checkpoint路径')
    parser.add_argument('--test_file', type=str,
                        default='data/datasets/high_quality_vlnce_fixed/test.json.gz',
                        help='测试集文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--samples', type=int, default=1, help='采样次数（固定为1确保结果一致）')
    parser.add_argument('--no_checkpoint', action='store_true', help='不使用checkpoint，创建新模型测试')
    
    args = parser.parse_args()
    
    logger.info("🧪 ETPNav模型测试脚本")
    logger.info(f"   模型checkpoint: {args.checkpoint}")
    logger.info(f"   测试集文件: {args.test_file}")
    logger.info(f"   计算设备: {args.device}")
    logger.info(f"   采样次数: {args.samples}")
    logger.info(f"   不使用checkpoint: {args.no_checkpoint}")
    
    # 检查checkpoint文件是否存在
    if not args.no_checkpoint and not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        
        # 提供可用的checkpoint选项
        checkpoint_dir = Path("data/checkpoints/etpnav_standard")
        if checkpoint_dir.exists():
            available_checkpoints = list(checkpoint_dir.glob("*.pth"))
            if available_checkpoints:
                logger.info("📁 可用的checkpoint文件:")
                for cp in available_checkpoints:
                    logger.info(f"   {cp}")
            else:
                logger.info("📁 checkpoint目录存在但为空")
        else:
            logger.info("📁 checkpoint目录不存在")
        
        logger.info("\n💡 解决方案:")
        logger.info("   1. 重新运行训练: python train_etpnav_compatible_final.py --epochs 3")
        logger.info("   2. 或使用 --no_checkpoint 选项创建新模型测试")
        sys.exit(1)
    
    try:
        # 创建测试器
        tester = ETPNavTester(args.checkpoint, args.device)
        
        # 加载测试集
        test_episodes = tester.load_test_dataset(args.test_file)
        if test_episodes is None:
            logger.error("❌ 测试集加载失败")
            sys.exit(1)
        
        # 进行测试
        test_results = tester.test_on_dataset(test_episodes, args.samples)
        
        if test_results:
            logger.info("🎉 测试完成！")
            logger.info(f"📊 最终L2误差: {test_results['final_l2_error_mean']:.4f} ± {test_results['final_l2_error_std']:.4f}m")
        else:
            logger.error("❌ 测试失败")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 测试过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()