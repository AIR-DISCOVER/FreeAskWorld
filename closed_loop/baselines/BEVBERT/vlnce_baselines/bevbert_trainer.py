#!/usr/bin/env python3
"""
BEVBert Trainer 
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import os

try:
    from vlnce_baselines.ss_trainer_ETP import RLTrainer as BaseTrainer
    print("✅ 导入基础训练器成功")
except ImportError:
    print("⚠️ 基础训练器导入失败，使用占位符")
    class BaseTrainer:
        def __init__(self, config): 
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BEVBertTrainer(BaseTrainer):
    """BEVBert 训练器"""
    
    def __init__(self, config):
        super().__init__(config)
        print("🤖 BEVBert训练器初始化")
        
        # BEVBert特定的初始化
        self.bevbert_config = getattr(config, 'BEVBERT', None)
        
    def load_checkpoint(self, checkpoint_path: str):
        """加载BEVBert检查点"""
        print(f"🔄 加载BEVBert检查点: {checkpoint_path}")
        
        if hasattr(super(), 'load_checkpoint'):
            return super().load_checkpoint(checkpoint_path)
        else:
            if os.path.exists(checkpoint_path):
                print("✅ 检查点文件存在")
                return True
            else:
                print("❌ 检查点文件不存在")
                return False
    
    def inference(self, observations: Dict[str, torch.Tensor] = None) -> int:
        """BEVBert推理"""
        print("🧠 BEVBert推理")

        if hasattr(super(), 'inference'):
            return super().inference(observations)
        else:
            return 1  

print("✅ BEVBert训练器模块加载完成")
