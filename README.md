<p align="center">
  <h1 align="center">FreeAskWorld Simulator (AAAI26 Oral)</h1>
</p>

<p align="center">
  <strong>An Interactive and Closed-Loop Simulator for Human-Centric Embodied AI</strong>
</p>

<p align="center">
  <!-- Badges -->
  <a href="$$LINK_TO_YOUR_PAPER_PDF$$" target="_blank">
    <img src="https://img.shields.io/badge/Paper-AAAI_2026-B31B1B.svg" alt="Paper PDF">
  </a>
  <a href="$$LINK_TO_YOUR_DATASET_PAGE$$" target="_blank">
    <img src="https://img.shields.io/badge/Dataset-FreeAskWorld-blue.svg" alt="Dataset">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://github.com/$$YOUR_USERNAME$$/FreeAskWorld" target="_blank">
    <img src="https://img.shields.io/github/stars/$$YOUR_USERNAME$$/FreeAskWorld?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  FreeAskWorld is an interactive simulation framework that integrates large language models (LLMs) for high-level planning and socially grounded interaction in embodied AI.
</p>

---

<p align="center">
  <img src="$$YOUR_TEASER_IMAGE_OR_GIF_URL$$" alt="FreeAskWorld Overview" width="80%">
</p>

---

## 📌 Introduction

As embodied intelligence progresses, simulation platforms must evolve beyond low-level physics toward **human-centric, socially interactive environments**.  
**FreeAskWorld** introduces:

- A **closed-loop interactive simulator**
- A **scalable human-agent world modeling framework**
- A **modular data generation pipeline**
- A new benchmark: **Direction Inquiry Task**, extending VLN to **active question-asking & guidance following**

This repo contains **simulator code** and **baseline models** from our AAAI 2026 paper.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **LLM-Powered Agents** | Intention modeling, reasoning, natural dialog, instruction generation |
| 🚶 **Realistic Humans** | Personalized profiles, schedules, motion & navigation styles |
| 🌦️ **Dynamic World** | Weather, lighting, traffic, and scene randomization |
| 🔁 **Closed-Loop Sync** | WebSocket-based state exchange for real-time model interaction |
| 🧩 **Direction Inquiry Task** | Agents ask for help, interpret human guidance, adapt plans |
| 📦 **Large-Scale Data** | 6 tasks · 16 object categories · 63,429 frames · 17+ hours |

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/FreeAskWorld.git
cd FreeAskWorld

# Create environment
conda create -n freeaskworld python=3.10 -y
conda activate freeaskworld

# Install dependencies
pip install -r requirements.txt
