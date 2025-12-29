# MSMF: Spatial-Guided Multi-level Fusion for YOLOv12

This repository provides the **core implementation** of the proposed
**MSMF module (SGM + FMS + Fusion)** for multi-level feature fusion in YOLOv12,
aimed at robust small-object detection in complex ecological environments.

---

## Overview

MSMF is a lightweight multi-level feature fusion module that integrates
spatial guidance and global feature modulation to enhance small-object
representation under strong background interference.
The overall computational complexity is **O(C·H·W)**.

---

## Repository Structure

papercode/
├── MSMF.py # Proposed MSMF module
├── CoordAtt.py # Coordinate Attention module
├── DySample.py # Dynamic sampling module
├── yolov12-CA-DySample-MSMF.yaml # YOLOv12 model configuration
├── make_isood_small_split.py # Small-object subset construction script
└── heatmap.py # Visualization utility

yaml
复制代码

---

## Dataset

Experiments are conducted on the publicly available **iSOOD dataset**.
The original dataset can be downloaded from:

- https://zenodo.org/records/10903574

The dataset itself is **not included** in this repository.

---

## Small-Object Subset

A small-object subset is constructed by retaining images that contain at least
one instance whose bounding-box area ratio is smaller than **0.5%**.
The subset is split into training, validation, and test sets following an **8:1:1** ratio.

The construction process is fully reproducible using:

```bash
python make_isood_small_split.py
Usage
The MSMF module can be integrated into YOLOv12 using the provided configuration file:

yaml
复制代码
yolov12-CA-DySample-MSMF.yaml
