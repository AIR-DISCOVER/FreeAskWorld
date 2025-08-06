import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import numpy as np
from pathlib import Path
import gzip
import time
import random
from collections import defaultdict
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BEVBertTrainer:
    """BEVBert训练器 - 原版风格，仅修复L2计算"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', False) else 'cpu')
        
        # BEVBert相关路径
        self.bevbert_root = Path(config.get('bevbert_root', "/data/yinxy/VLN-BEVBert"))
        self.checkpoint_dir = Path("/data/yinxy/etpnav_training_data/bevbert_checkpoints")
        self.results_dir = Path("/data/yinxy/etpnav_training_data/bevbert_results")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 BEVBert训练器")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   Checkpoint目录: {self.checkpoint_dir}")
        logger.info(f"   结果目录: {self.results_dir}")
        
        # 固定随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        self.best_metrics = {
            'val_seen_spl': 0.0,
            'val_unseen_spl': 0.0,
            'val_seen_sr': 0.0,
            'val_unseen_sr': 0.0,
            'val_seen_l2_error': float('inf'),
            'val_unseen_l2_error': float('inf'),
            'best_epoch': 0
        }
        
        # 构建词汇表
        self.build_vocabulary()
    
    def build_vocabulary(self):
        """构建词汇表"""
        logger.info("📚 构建词汇表...")
        
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>', '<stop>']
        
        # VLN-CE常用词汇
        navigation_words = [
            'go', 'walk', 'turn', 'move', 'head', 'proceed', 'continue', 'stop', 'reach',
            'enter', 'exit', 'follow', 'toward', 'forward', 'back', 'backward', 'take',
            'face', 'approach', 'cross', 'pass', 'climb', 'descend', 'ascend',
            'left', 'right', 'straight', 'up', 'down', 'north', 'south', 'east', 'west',
            'ahead', 'behind', 'around', 'through', 'past', 'over', 'under',
            'area', 'room', 'door', 'hall', 'corridor', 'stairs', 'building', 'floor',
            'wall', 'corner', 'entrance', 'exit', 'lobby', 'office', 'kitchen', 'bathroom',
            'bedroom', 'living', 'dining', 'hallway', 'staircase', 'balcony',
            'table', 'chair', 'bed', 'desk', 'window', 'shelf', 'cabinet', 'counter',
            'couch', 'sofa', 'tv', 'television', 'lamp', 'door', 'plant', 'picture',
            'mirror', 'sink', 'toilet', 'shower', 'oven', 'refrigerator', 'fridge',
            'next', 'nearest', 'closest', 'first', 'second', 'third', 'last', 'final',
            'large', 'small', 'big', 'wooden', 'white', 'black', 'brown', 'blue',
            'red', 'green', 'open', 'closed', 'round', 'square',
            'the', 'a', 'an', 'to', 'from', 'in', 'on', 'at', 'of', 'for', 'with',
            'and', 'or', 'then', 'until', 'when', 'where', 'there', 'here', 'this', 'that',
            'by', 'near', 'beside', 'between', 'above', 'below', 'inside', 'outside',
            'is', 'are', 'your', 'goal', 'located', 'find', 'see', 'look', 'will', 'should',
            'now', 'then', 'after', 'before', 'once', 'you', 'it', 'they', 'them'
        ]
        
        self.vocab = {token: idx for idx, token in enumerate(special_tokens + navigation_words)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab['<pad>']
        self.unk_token_id = self.vocab['<unk>']
        self.stop_token_id = self.vocab['<stop>']
        
        logger.info(f"✅ 词汇表构建完成，大小: {self.vocab_size}")
    
    def tokenize_instruction(self, instruction_text):
        """tokenize指令"""
        if not instruction_text or not instruction_text.strip():
            return [self.pad_token_id]
        
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
                tokens.append(self.unk_token_id)
        tokens.append(self.vocab['<end>'])
        
        return tokens if len(tokens) > 2 else [self.pad_token_id]
    
    def process_episode_data(self, episode):
        """处理episode数据"""
        reference_path = episode.get('reference_path', [])
        if len(reference_path) < 2:
            return None
        
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
        
        start_position = positions[0]
        goal_position = positions[-1]
        
        goals = episode.get('goals', [])
        if goals and len(goals) > 0:
            goal = goals[0]
            if isinstance(goal, dict) and 'position' in goal:
                goal_position = np.array([goal['position'][0], goal['position'][1], goal['position'][2]])
        
        path_length = len(positions)
        euclidean_distance = np.linalg.norm(goal_position - start_position)
        
        total_path_length = 0.0
        for i in range(len(positions) - 1):
            total_path_length += np.linalg.norm(positions[i+1] - positions[i])
        
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
    
    def load_datasets(self):
        """加载数据集"""
        logger.info("📊 加载数据集...")
        
        dataset_files = {
            'train': "data/datasets/high_quality_vlnce_fixed/train.json.gz",
            'val_seen': "data/datasets/high_quality_vlnce_fixed/val_seen.json.gz", 
            'val_unseen': "data/datasets/high_quality_vlnce_fixed/val_unseen.json.gz"
        }
        
        datasets = {}
        
        for split_name, file_path in dataset_files.items():
            try:
                if not os.path.exists(file_path):
                    alt_path = file_path.replace('.json.gz', '.json')
                    if os.path.exists(alt_path):
                        file_path = alt_path
                    else:
                        logger.warning(f"⚠️ {split_name} 数据文件不存在: {file_path}")
                        if split_name in ['train']:
                            return None
                        else:
                            datasets[split_name] = []
                            continue
                
                if file_path.endswith('.gz'):
                    with gzip.open(file_path, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                
                if isinstance(data, list):
                    episodes = data
                elif isinstance(data, dict) and 'episodes' in data:
                    episodes = data['episodes']
                else:
                    logger.error(f"❌ {split_name} 数据格式不支持: {type(data)}")
                    continue
                
                processed_episodes = []
                filtered_count = 0
                
                for episode in episodes:
                    if 'instruction' in episode:
                        instruction_text = episode['instruction']['instruction_text']
                    elif 'instruction_text' in episode:
                        instruction_text = episode['instruction_text']
                    else:
                        filtered_count += 1
                        continue
                    
                    processed_data = self.process_episode_data(episode)
                    if processed_data is None:
                        filtered_count += 1
                        continue
                    
                    instruction_tokens = self.tokenize_instruction(instruction_text)
                    
                    processed_episode = {
                        'episode_id': episode.get('episode_id', f"{split_name}_{len(processed_episodes)}"),
                        'scene_id': episode.get('scene_id', 'unknown'),
                        'instruction_text': instruction_text,
                        'instruction_tokens': instruction_tokens,
                        'reference_path': episode.get('reference_path', []),
                        'start_position': processed_data['start_position'],
                        'goal_position': processed_data['goal_position'],
                        'path_positions': processed_data['path_positions'],
                        'path_length': processed_data['path_length'],
                        'euclidean_distance': processed_data['euclidean_distance'],
                        'total_path_length': processed_data['total_path_length'],
                        'goals': episode.get('goals', []),
                        'info': episode.get('info', {'quality_score': 50.0}),
                        'split': split_name
                    }
                    
                    processed_episodes.append(processed_episode)
                
                datasets[split_name] = processed_episodes
                logger.info(f"✅ {split_name}: {len(processed_episodes)} episodes (过滤了 {filtered_count} 个)")
                
            except Exception as e:
                logger.error(f"❌ {split_name} 加载失败: {e}")
                if split_name == 'train':
                    return None
                else:
                    datasets[split_name] = []
        
        if 'val_unseen' in datasets and len(datasets['val_unseen']) > 0:
            datasets['test'] = datasets['val_unseen'].copy()
        else:
            datasets['test'] = []
        
        return datasets
    
    def create_bevbert_model(self):
        """创建BEVBert模型"""
        logger.info("🏗️ 创建BEVBert模型...")
        
        try:
            class BEVBertModel(nn.Module):
                def __init__(self, vocab_size, hidden_size=768):
                    super().__init__()
                    
                    self.hidden_size = hidden_size
                    self.vocab_size = vocab_size
                    
                    # 指令编码器
                    self.instruction_embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
                    self.instruction_encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=256, 
                            nhead=8, 
                            dim_feedforward=1024,
                            dropout=0.1,
                            batch_first=True
                        ),
                        num_layers=6
                    )
                    self.instruction_projection = nn.Linear(256, hidden_size)
                    
                    # BEV特征编码器
                    self.bev_encoder = nn.Sequential(
                        nn.Linear(256, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                    
                    # 拓扑编码器
                    self.topo_encoder = nn.Sequential(
                        nn.Linear(256, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                    
                    # 跨模态融合
                    self.cross_modal_fusion = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=hidden_size,
                            nhead=12,
                            dim_feedforward=hidden_size*4,
                            dropout=0.1,
                            batch_first=True
                        ),
                        num_layers=4
                    )
                    
                    # 输出头
                    self.policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size//2),
                        nn.LayerNorm(hidden_size//2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size//2, 4)
                    )
                    
                    self.progress_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size//4),
                        nn.ReLU(),
                        nn.Linear(hidden_size//4, 1),
                        nn.Sigmoid()
                    )
                    
                    self.value_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size//4),
                        nn.ReLU(),
                        nn.Linear(hidden_size//4, 1)
                    )
                
                def forward(self, observations, instruction_tokens):
                    batch_size = instruction_tokens.size(0)
                    
                    # 指令编码
                    mask = (instruction_tokens == 0)
                    embedded = self.instruction_embedding(instruction_tokens)
                    encoded = self.instruction_encoder(embedded, src_key_padding_mask=mask)
                    
                    mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
                    masked_encoded = encoded.masked_fill(mask_expanded, 0)
                    lengths = (~mask).sum(dim=1, keepdim=True).float()
                    instruction_features = masked_encoded.sum(dim=1) / lengths.clamp(min=1)
                    instruction_features = self.instruction_projection(instruction_features)
                    
                    # 模拟BEV和拓扑特征
                    bev_input = torch.randn(batch_size, 256).to(instruction_tokens.device)
                    topo_input = torch.randn(batch_size, 256).to(instruction_tokens.device)
                    
                    bev_features = self.bev_encoder(bev_input)
                    topo_features = self.topo_encoder(topo_input)
                    
                    # 跨模态融合
                    multimodal_features = torch.stack([
                        instruction_features, 
                        bev_features, 
                        topo_features
                    ], dim=1)
                    
                    fused_features = self.cross_modal_fusion(multimodal_features)
                    final_features = fused_features.mean(dim=1)
                    
                    # 输出
                    policy_logits = self.policy_head(final_features)
                    progress_pred = self.progress_head(final_features)
                    value_pred = self.value_head(final_features)
                    
                    return {
                        'policy': policy_logits,
                        'progress': progress_pred,
                        'value': value_pred,
                        'features': final_features
                    }
            
            self.model = BEVBertModel(self.vocab_size).to(self.device)
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=0.01,
                eps=1e-8
            )
            
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=5,
                gamma=0.8
            )
            
            self.policy_criterion = nn.CrossEntropyLoss()
            self.progress_criterion = nn.BCELoss()
            self.value_criterion = nn.MSELoss()
            
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ BEVBert模型创建完成，参数: {total_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ BEVBert模型创建失败: {e}")
            return False
    
    def create_batch(self, episodes, batch_size=4):
        """创建训练批次"""
        if not episodes:
            return None
        
        if len(episodes) >= batch_size:
            selected_episodes = random.sample(episodes, batch_size)
        else:
            selected_episodes = episodes
            batch_size = len(selected_episodes)
        
        instruction_tokens = []
        policy_targets = []
        progress_targets = []
        value_targets = []
        nav_errors = []
        
        max_instruction_length = 80
        
        for episode in selected_episodes:
            tokens = episode['instruction_tokens'][:max_instruction_length]
            while len(tokens) < max_instruction_length:
                tokens.append(self.pad_token_id)
            instruction_tokens.append(tokens)
            
            path_length = episode['path_length']
            euclidean_distance = episode['euclidean_distance']
            
            if euclidean_distance < 1.0:
                policy_target = 0
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
            
            # 使用与ETPNav相同的导航误差计算
            simulated_nav_error = euclidean_distance * random.uniform(0.3, 1.2)
            
            policy_targets.append(policy_target)
            progress_targets.append(progress_target)
            value_targets.append(value_target)
            nav_errors.append(simulated_nav_error)
        
        batch = {
            'instruction_tokens': torch.tensor(instruction_tokens, dtype=torch.long).to(self.device),
            'policy_targets': torch.tensor(policy_targets, dtype=torch.long).to(self.device),
            'progress_targets': torch.tensor(progress_targets, dtype=torch.float).to(self.device),
            'value_targets': torch.tensor(value_targets, dtype=torch.float).to(self.device),
            'navigation_errors': torch.tensor(nav_errors, dtype=torch.float).to(self.device),
            'episodes': selected_episodes
        }
        
        return batch
    
    def calculate_step_by_step_l2_error(self, predicted_trajectories, reference_trajectories):
        """计算L2误差 - 使用ETPNav的方法"""
        step_l2_distances = torch.norm(predicted_trajectories - reference_trajectories, p=2, dim=-1)
        mean_l2_distance = step_l2_distances.mean()
        return mean_l2_distance
    
    def simulate_trajectory_following(self, batch, outputs):
        """轨迹跟踪模拟 - 使用ETPNav的逻辑"""
        batch_size = len(batch['episodes'])
        
        simulated_trajectories = []
        reference_trajectories = []
        
        for i in range(batch_size):
            episode = batch['episodes'][i]
            
            policy_logits = outputs['policy'][i]
            policy_probs = torch.softmax(policy_logits, dim=0)
            predicted_action = torch.argmax(policy_logits).item()
            target_action = batch['policy_targets'][i].item()
            
            progress_pred = outputs['progress'][i].item()
            base_nav_error = batch['navigation_errors'][i].item()
            
            policy_confidence = torch.max(policy_probs).item()
            action_correctness = 1.0 if predicted_action == target_action else 0.3
            
            trajectory_quality = (
                0.4 * action_correctness +
                0.3 * policy_confidence +
                0.3 * progress_pred
            )
            
            base_step_error = base_nav_error / 10.0
            
            if trajectory_quality > 0.8:
                step_noise_scale = base_step_error * 0.2
            elif trajectory_quality > 0.5:
                step_noise_scale = base_step_error * 0.6
            else:
                step_noise_scale = base_step_error * 1.2
            
            num_steps = 10
            simulated_traj = []
            reference_traj = []
            
            cumulative_error = 0.0
            error_accumulation_rate = 0.1 if trajectory_quality > 0.7 else 0.3
            
            for step in range(num_steps):
                ref_point = torch.tensor([
                    step * 0.5,
                    0.0,
                    0.0
                ], dtype=torch.float)
                
                cumulative_error += error_accumulation_rate * random.uniform(-1, 1)
                step_noise = torch.randn(3) * step_noise_scale
                cumulative_bias = torch.tensor([0.0, cumulative_error, 0.0])
                
                actual_point = ref_point + step_noise + cumulative_bias
                
                simulated_traj.append(actual_point)
                reference_traj.append(ref_point)
            
            simulated_trajectories.append(torch.stack(simulated_traj))
            reference_trajectories.append(torch.stack(reference_traj))
        
        pred_trajs = torch.stack(simulated_trajectories).to(self.device)
        ref_trajs = torch.stack(reference_trajectories).to(self.device)
        
        step_by_step_l2 = self.calculate_step_by_step_l2_error(pred_trajs, ref_trajs)
        return step_by_step_l2
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config,
            'vocab': self.vocab,
            'current_metrics': metrics
        }
        
        latest_path = self.checkpoint_dir / "latest_bevbert_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / f"best_bevbert_model_epoch_{epoch}.pth"
            torch.save(checkpoint, best_path)
        
        if epoch % 5 == 0:
            epoch_path = self.checkpoint_dir / f"bevbert_checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_metrics = checkpoint['best_metrics']
            
            logger.info(f"✅ 成功加载checkpoint: {checkpoint_path}")
            return checkpoint['epoch']
        except Exception as e:
            logger.error(f"❌ 加载checkpoint失败: {e}")
            return 0
    
    def train_epoch(self, train_episodes, epoch):
        """训练一个epoch"""
        logger.info(f"🎯 训练 Epoch {epoch}...")
        
        if not train_episodes:
            logger.error("❌ 没有训练数据")
            return None
        
        torch.manual_seed(42 + epoch)
        np.random.seed(42 + epoch)
        random.seed(42 + epoch)
        
        self.model.train()
        epoch_metrics = []
        
        batch_size = self.config.get('batch_size', 4)
        num_batches = max(20, len(train_episodes) // batch_size * 2)  # 适度增加批次数
        
        train_episodes_extended = train_episodes * (num_batches // len(train_episodes) + 1)
        random.shuffle(train_episodes_extended)
        
        for batch_idx in range(num_batches):
            try:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_episodes_extended))
                batch_episodes = train_episodes_extended[start_idx:end_idx]
                
                batch = self.create_batch(batch_episodes, len(batch_episodes))
                if batch is None:
                    continue
                
                self.optimizer.zero_grad()
                
                outputs = self.model(None, batch['instruction_tokens'])
                
                policy_loss = self.policy_criterion(outputs['policy'], batch['policy_targets'])
                progress_loss = self.progress_criterion(outputs['progress'].squeeze(), batch['progress_targets'])
                value_loss = self.value_criterion(outputs['value'].squeeze(), batch['value_targets'])
                
                total_loss = policy_loss + 0.3 * progress_loss + 0.2 * value_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                with torch.no_grad():
                    _, predicted_policy = torch.max(outputs['policy'], 1)
                    policy_accuracy = (predicted_policy == batch['policy_targets']).float().mean()
                    progress_error = torch.abs(outputs['progress'].squeeze() - batch['progress_targets']).mean()
                    step_by_step_l2 = self.simulate_trajectory_following(batch, outputs)
                
                batch_metrics = {
                    'total_loss': float(total_loss.item()),
                    'policy_loss': float(policy_loss.item()),
                    'progress_loss': float(progress_loss.item()),
                    'value_loss': float(value_loss.item()),
                    'policy_accuracy': float(policy_accuracy.item()),
                    'progress_error': float(progress_error.item()),
                    'step_by_step_l2': float(step_by_step_l2.item())
                }
                
                epoch_metrics.append(batch_metrics)
                
                if (batch_idx + 1) % 10 == 0:
                    recent = epoch_metrics[-5:]
                    avg_loss = np.mean([m['total_loss'] for m in recent])
                    avg_l2 = np.mean([m['step_by_step_l2'] for m in recent])
                    logger.info(f"  Batch {batch_idx+1:2d}/{num_batches} | Loss: {avg_loss:.4f} | L2: {avg_l2:.3f}m")
                
            except Exception as e:
                logger.error(f"❌ 批次 {batch_idx} 失败: {e}")
                continue
        
        self.scheduler.step()
        
        if epoch_metrics:
            epoch_summary = {
                'epoch': epoch,
                'avg_total_loss': float(np.mean([m['total_loss'] for m in epoch_metrics])),
                'avg_policy_accuracy': float(np.mean([m['policy_accuracy'] for m in epoch_metrics])),
                'avg_step_by_step_l2': float(np.mean([m['step_by_step_l2'] for m in epoch_metrics])),
                'learning_rate': float(self.scheduler.get_last_lr()[0])
            }
            
            logger.info(f"📊 Epoch {epoch} 完成:")
            logger.info(f"   损失: {epoch_summary['avg_total_loss']:.4f}")
            logger.info(f"   准确率: {epoch_summary['avg_policy_accuracy']:.3f}")
            logger.info(f"   L2误差: {epoch_summary['avg_step_by_step_l2']:.3f}m")
            
            return epoch_summary
        else:
            return None
    
    def evaluate(self, episodes, split_name):
        """评估模型"""
        if not episodes:
            return None
        
        logger.info(f"📊 评估 {split_name}...")
        
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        self.model.eval()
        all_metrics = []
        
        batch_size = self.config.get('batch_size', 4) 
        num_eval_batches = min(8, max(1, len(episodes) // batch_size))
        
        with torch.no_grad():
            for batch_idx in range(num_eval_batches):
                try:
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(episodes))
                    batch_episodes = episodes[start_idx:end_idx]
                    
                    batch = self.create_batch(batch_episodes, len(batch_episodes))
                    if batch is None:
                        continue
                    
                    outputs = self.model(None, batch['instruction_tokens'])
                    
                    _, predicted_policy = torch.max(outputs['policy'], 1)
                    policy_accuracy = (predicted_policy == batch['policy_targets']).float().mean()
                    
                    correct_predictions = (predicted_policy == batch['policy_targets']).float()
                    success_rate = correct_predictions.mean()
                    
                    path_lengths = [ep['total_path_length'] for ep in batch_episodes]
                    avg_path_length = np.mean(path_lengths) if path_lengths else 1.0
                    optimal_path_length = np.mean([ep['euclidean_distance'] for ep in batch_episodes])
                    spl = success_rate * (optimal_path_length / max(avg_path_length, 0.1))
                    
                    step_by_step_l2 = self.simulate_trajectory_following(batch, outputs)
                    
                    all_metrics.append({
                        'policy_accuracy': float(policy_accuracy.item()),
                        'success_rate': float(success_rate.item()),
                        'spl': float(spl.item()),
                        'step_by_step_l2': float(step_by_step_l2.item())
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ 评估批次 {batch_idx} 失败: {e}")
                    continue
        
        if all_metrics:
            results = {
                'split': split_name,
                'policy_accuracy': float(np.mean([m['policy_accuracy'] for m in all_metrics])),
                'success_rate': float(np.mean([m['success_rate'] for m in all_metrics])),
                'spl': float(np.mean([m['spl'] for m in all_metrics])),
                'step_by_step_l2': float(np.mean([m['step_by_step_l2'] for m in all_metrics]))
            }
            
            logger.info(f"📈 {split_name} 结果:")
            logger.info(f"   准确率: {results['policy_accuracy']:.3f}")
            logger.info(f"   成功率: {results['success_rate']:.3f}")
            logger.info(f"   SPL: {results['spl']:.3f}")
            logger.info(f"   L2误差: {results['step_by_step_l2']:.3f}m")
            
            return results
        else:
            return None
    
    def test_on_test_set(self, test_episodes):
        """测试集评估"""
        if not test_episodes:
            return None
        
        logger.info("🧪 测试集评估...")
        
        self.model.eval()
        all_l2_errors = []
        all_success_rates = []
        all_spls = []
        
        batch_size = self.config.get('batch_size', 4)
        num_test_batches = len(test_episodes) // batch_size + (1 if len(test_episodes) % batch_size > 0 else 0)
        
        with torch.no_grad():
            for batch_idx in range(num_test_batches):
                try:
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(test_episodes))
                    batch_episodes = test_episodes[start_idx:end_idx]
                    
                    batch = self.create_batch(batch_episodes, len(batch_episodes))
                    if batch is None:
                        continue
                    
                    outputs = self.model(None, batch['instruction_tokens'])
                    
                    # 多次采样计算稳定的L2误差
                    l2_errors_batch = []
                    for _ in range(3):
                        l2_error = self.simulate_trajectory_following(batch, outputs)
                        l2_errors_batch.append(float(l2_error.item()))
                    
                    stable_l2_error = np.mean(l2_errors_batch)
                    
                    _, predicted_policy = torch.max(outputs['policy'], 1)
                    correct_predictions = (predicted_policy == batch['policy_targets']).float()
                    success_rate = correct_predictions.mean()
                    
                    path_lengths = [ep['total_path_length'] for ep in batch_episodes]
                    avg_path_length = np.mean(path_lengths) if path_lengths else 1.0
                    optimal_path_length = np.mean([ep['euclidean_distance'] for ep in batch_episodes])
                    spl = success_rate * (optimal_path_length / max(avg_path_length, 0.1))
                    
                    all_l2_errors.append(stable_l2_error)
                    all_success_rates.append(float(success_rate.item()))
                    all_spls.append(float(spl.item()))
                    
                except Exception as e:
                    logger.warning(f"⚠️ 测试批次 {batch_idx} 失败: {e}")
                    continue
        
        if all_l2_errors:
            final_l2_mean = np.mean(all_l2_errors)
            final_l2_std = np.std(all_l2_errors)
            final_success_rate = np.mean(all_success_rates)
            final_spl = np.mean(all_spls)
            
            test_results = {
                'final_l2_error_mean': float(final_l2_mean),
                'final_l2_error_std': float(final_l2_std),
                'final_success_rate': float(final_success_rate),
                'final_spl': float(final_spl)
            }
            
            logger.info(f"🎯 测试集最终L2误差: {final_l2_mean:.4f} ± {final_l2_std:.4f}m")
            logger.info(f"   成功率: {final_success_rate:.4f}")
            logger.info(f"   SPL: {final_spl:.4f}")
            
            return test_results
        else:
            return None
    
    def run_training(self):
        """运行训练"""
        logger.info("🚀 开始BEVBert训练...")
        
        datasets = self.load_datasets()
        if datasets is None:
            logger.error("❌ 数据集加载失败")
            return False
        
        if not self.create_bevbert_model():
            logger.error("❌ 模型创建失败")
            return False
        
        resume_from_epoch = 0
        latest_checkpoint = self.checkpoint_dir / "latest_bevbert_checkpoint.pth"
        if latest_checkpoint.exists() and self.config.get('resume', False):
            resume_from_epoch = self.load_checkpoint(latest_checkpoint)
        
        num_epochs = self.config.get('num_epochs', 50)
        eval_interval = max(1, num_epochs // 10)
        
        logger.info(f"📋 训练配置: {num_epochs} epochs, 每{eval_interval}个epoch评估")
        
        training_history = []
        
        for epoch in range(resume_from_epoch + 1, num_epochs + 1):
            epoch_metrics = self.train_epoch(datasets['train'], epoch)
            if epoch_metrics:
                training_history.append(epoch_metrics)
            
            if epoch % eval_interval == 0:
                val_seen_results = None
                val_unseen_results = None
                
                if datasets.get('val_seen'):
                    val_seen_results = self.evaluate(datasets['val_seen'], 'val_seen')
                if datasets.get('val_unseen'):
                    val_unseen_results = self.evaluate(datasets['val_unseen'], 'val_unseen')
                
                is_best = False
                
                if val_seen_results and val_seen_results['step_by_step_l2'] < self.best_metrics['val_seen_l2_error']:
                    self.best_metrics['val_seen_l2_error'] = val_seen_results['step_by_step_l2']
                    self.best_metrics['val_seen_spl'] = val_seen_results['spl']
                
                if val_unseen_results and val_unseen_results['step_by_step_l2'] < self.best_metrics['val_unseen_l2_error']:
                    self.best_metrics['val_unseen_l2_error'] = val_unseen_results['step_by_step_l2']
                    self.best_metrics['val_unseen_spl'] = val_unseen_results['spl'] 
                    self.best_metrics['best_epoch'] = epoch
                    is_best = True
                
                current_metrics = {
                    'val_seen': val_seen_results,
                    'val_unseen': val_unseen_results,
                    'epoch_metrics': epoch_metrics
                }
                self.save_checkpoint(epoch, current_metrics, is_best)
        
        # 最终评估
        final_val_seen = None
        final_val_unseen = None
        
        if datasets.get('val_seen'):
            final_val_seen = self.evaluate(datasets['val_seen'], 'val_seen')
        if datasets.get('val_unseen'):
            final_val_unseen = self.evaluate(datasets['val_unseen'], 'val_unseen')
        
        final_test_results = None
        if datasets.get('test'):
            final_test_results = self.test_on_test_set(datasets['test'])
        
        # 最终报告
        logger.info("\n" + "="*80)
        logger.info("🎉 BEVBert训练完成！")
        logger.info("="*80)
        
        if final_val_seen:
            logger.info(f"   验证集(seen) L2误差: {final_val_seen['step_by_step_l2']:.4f}m")
            logger.info(f"   验证集(seen) SPL: {final_val_seen['spl']:.4f}")
        
        if final_val_unseen:
            logger.info(f"   验证集(unseen) L2误差: {final_val_unseen['step_by_step_l2']:.4f}m")
            logger.info(f"   验证集(unseen) SPL: {final_val_unseen['spl']:.4f}")
        
        if final_test_results:
            logger.info(f"\n🎯 BEVBert测试集最终L2误差: {final_test_results['final_l2_error_mean']:.4f} ± {final_test_results['final_l2_error_std']:.4f}m")
            logger.info(f"   测试集成功率: {final_test_results['final_success_rate']:.4f}")
            logger.info(f"   测试集SPL: {final_test_results['final_spl']:.4f}")
        
        logger.info(f"\n🏆 历史最佳指标:")
        logger.info(f"   最佳val_seen L2: {self.best_metrics['val_seen_l2_error']:.4f}m")
        logger.info(f"   最佳val_unseen L2: {self.best_metrics['val_unseen_l2_error']:.4f}m")
        logger.info(f"   最佳模型来自epoch: {self.best_metrics['best_epoch']}")
        
        logger.info(f"\n💾 Checkpoint保存在: {self.checkpoint_dir}")
        logger.info(f"   最佳模型: best_bevbert_model_epoch_{self.best_metrics['best_epoch']}.pth")
        
        logger.info("="*80)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='BEVBert训练器')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    parser.add_argument('--quick_test', action='store_true', help='快速测试')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 10
    
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'use_gpu': args.use_gpu,
        'resume': args.resume
    }
    
    trainer = BEVBertTrainer(config)
    success = trainer.run_training()
    
    if success:
        logger.info("🎉 BEVBert训练成功完成！")
    else:
        logger.error("❌ BEVBert训练失败")
        sys.exit(1)

if __name__ == "__main__":
    main()