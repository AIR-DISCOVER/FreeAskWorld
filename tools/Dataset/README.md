# FreeAskWorld Evaluation Toolkit

This repository provides a comprehensive evaluation and data analysis toolkit for the FreeAskWorld benchmark dataset, supporting both training and testing pipelines. It includes scripts for evaluating model performance, analyzing dataset characteristics, and benchmarking navigation tasks.

## Directory Overview

### `TestData/`
Evaluate test set performance:
- Compute **Success Rate (SR)**.
- Compute **Total and Average Trajectory Length (TL)**.

### `Metrics/`
Evaluate training set using metrics defined in `TestMetrics`:
- **Success Rate (SR)** (avg)
- **Trajectory Length (TL)** (avg)
- **Success weighted by Path Length (SPL)**
- **Navigation Error (NE)** (avg)
- **Oracle Navigation Error (ONE)** (avg)
- **Oracle Success Rate (OSR)** (avg)
- **Ask Way Number (AWN)** (avg)

### `Data_analyze/`
Analyze dataset statistics and visualize:
- **Path Length distribution**
- **Start Time of Day** distribution
- **Weather type** distribution

Includes plotting utilities for data visualization.

### `Data_TL_HL/`
Compute average statistics from the dataset:
- **Average Trajectory Length**
- **Average Instruction Length**

Useful for understanding overall dataset difficulty and instruction complexity.

### `Analyze/`
Benchmark closed-loop navigation performance:
- **Success Rate (SR)** (avg)
- **Trajectory Length (TL)** (avg)
- **Success weighted by Path Length (SPL)**
- **Navigation Error (NE)** (avg)
- **Oracle Navigation Error (ONE)** (avg)
- **Oracle Success Rate (OSR)** (avg)
- **Ask Way Number (AWN)** (avg)

This is designed for evaluating looped simulations on the dataset benchmark.

## Metrics Explanation

| Metric | Description |
|--------|-------------|
| **SR** | Success Rate — percentage of episodes where the agent reaches the goal |
| **TL** | Trajectory Length — total/average number of steps taken by the agent |
| **SPL** | Success weighted by Path Length — accounts for both efficiency and success |
| **NE** | Navigation Error — distance between final position and goal |
| **ONE** | Oracle Navigation Error — minimum error across all intermediate positions |
| **OSR** | Oracle Success Rate — percentage where any intermediate position was within goal range |
| **NDI** | Number of Direction Inqury — average number of help requests made by the agent |
