# ETPNav and BEVBert Training & Inference Pipeline

A comprehensive pipeline for training and deploying ETPNav and BEVBert models using FreeAskWorld database.

## Quick Start

### 1. Data Conversion
```bash
python fix_bom_and_reconvert.py
```
Converts FreeAskWorld database to VLN-CE compatible format.

### 2. Training Models

**ETPNav:**
```bash
python train_etpnav_compatible_final.py --epochs 100 --batch_size 8 --use_gpu
```

**BEVBert:**
```bash
python bevbert_trainer.py --epochs 50 --batch_size 4 --use_gpu
```

### 3. Testing Models

**ETPNav:**
```bash
python test_etpnav_model.py --checkpoint checkpoints/best_model_epoch_X.pth
```

**BEVBert:**
```bash
python bevbert_test_script.py --checkpoint bevbert_checkpoints/best_model.pth
```

### 4. Baseline Comparison
```bash
python test_original_etpnav.py
python original_bevbert_tester.py
```

## Inference Services

### ETPNav Inference
```python
from configurable_etpnav_service import ConfigurableETPNavService

service = ConfigurableETPNavService(config_file="etpnav_config.yaml")
service.start_service()

request_id = service.send_inference_request(
    rgb_image=your_rgb_data,
    depth_image=your_depth_data,
    instruction="navigate to target",
    position=[x, y, z],
    rotation=[x, y, z, w]
)

result = service.get_inference_result(request_id, timeout=30.0)
service.stop_service()
```

### BEVBert Inference
```python
from bevbert_inference_service import BEVBertInferenceService, BEVBertConfig
import numpy as np
import time

config = BEVBertConfig(
    use_real_model=True,
    checkpoint_path="path/to/best_model.pth",
    device="cpu"
)

service = BEVBertInferenceService(config=config)
service.start_service()
time.sleep(2)  # Wait for initialization

test_data = {
    'rgb': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    'depth': np.random.rand(224, 224).astype(np.float32),
    'instruction': 'navigate to kitchen'
}

future = service.inference(test_data)
result = future.result()
service.stop_service()
```

## Configuration

**ETPNav Config (etpnav_config.yaml):**
```yaml
use_real_model: true
checkpoint_path: "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
device: "cuda"
output_dir: "etpnav_outputs"
save_results: true
```

**BEVBert Action Mappings:**
- Action 0: Stop
- Action 1: Move forward (0.09-0.15 units)
- Action 2: Turn left (0.1-0.2 radians)
- Action 3: Turn right (0.1-0.2 radians)

## File Structure
```
├── fix_bom_and_reconvert.py           # Data conversion
├── train_etpnav_compatible_final.py   # ETPNav training
├── test_etpnav_model.py               # ETPNav evaluation
├── test_original_etpnav.py            # Original ETPNav baseline
├── bevbert_trainer.py                 # BEVBert training
├── bevbert_test_script.py             # BEVBert evaluation
├── original_bevbert_tester.py         # Original BEVBert baseline
├── configurable_etpnav_service.py     # ETPNav inference service
├── bevbert_inference_service.py       # BEVBert inference service
├── etpnav_config.yaml                 # Configuration file
├── checkpoints/                       # Model checkpoints
├── results/                           # Training logs
└── data/datasets/high_quality_vlnce_fixed/  # Converted dataset
```

## Key Features

- **Multi-modal Architecture**: RGB + Depth + Instruction processing
- **L2 Error Tracking**: Primary evaluation metric for navigation accuracy
- **Configurable Services**: Flexible inference deployment
- **Baseline Comparison**: Direct performance comparison with original models
- **Production Ready**: Multi-process inference services with callbacks

## Requirements
```bash
pip install torch torchvision numpy transformers
pip install habitat-sim habitat-lab opencv-python
pip install pyyaml pathlib gzip argparse logging
```

## Performance Benchmarks
- **ETPNav L2 Error**: ~0.3-0.5m (step-by-step trajectory deviation)
- **BEVBert L2 Error**: ~0.4-0.6m (with cross-modal fusion)
- **Success Rate**: 60-80% (depending on episode difficulty)
- **Training Time**: ETPNav ~2-3 hours, BEVBert ~1-2 hours (GPU)