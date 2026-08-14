# Real-Time Pedestrian Detection and Monocular Distance Estimation for ADAS

This repository contains the  implementation of the models and experiments developed for my Master's thesis in Artificial Intelligence for Science and Technology at Università degli Studi di Milano-Bicocca.

The project explores multi-task learning architectures for Autonomous Driving Assistance Systems (ADAS). It introduces **SGDA-Net**, an architecture using semantic-guided attention to improve depth estimation boundaries, and adapts the **VidEoMT** (Video Encoder-only Mask Transformer) architecture to a multi-task scenario to solve momentary data loss and occlusion issues via spatio-temporal query propagation.

## Features
* **Multi-Task Learning:** Joint optimization for semantic segmentation and monocular depth estimation.
* **SGDA-Net:** Cross-task attention mechanisms (CSAM) to prevent boundary bleeding and physical distortions in 3D perception.
* **VidEoMT_ADAS:** Temporal object query transfer across consecutive frames to maintain target identity under occlusion.

## Repository Structure

├── configs/             # YAML configuration files for each model architecture
├── datasets/            # Unified dataloaders for standard and temporal (video) data
├── models/              # Model definitions (SGDA-Net, VidEoMT, and baselines)
├── scripts/             # Core scripts for training, evaluation, and inference
├── utils/               # Metrics, logging, losses, and visualization tools
└── weights/             # Directory for downloaded or trained .pth checkpoints

## Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/UmutYildirim15/MultiTask-Depth-Segmentatios.git](https://github.com/UmutYildirim15/MultiTask-Depth-Segmentation.git)
cd MultiTask-Depth-Segmentation
pip install -r requirements.txt
