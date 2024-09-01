# Person-Detector
# Person Detection and Tracking with Unique IDs

## Overview

This project implements a system for detecting and tracking persons in a video, specifically targeting children and therapists, with the goal of understanding behaviors, emotions, and engagement levels. The system assigns unique IDs to detected persons, tracks them across frames, and handles re-entry, occlusion, and post-occlusion scenarios.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- YOLOv8 for object detection
- DeepSORT for tracking

Install the required libraries using:

```bash
pip install -r requirements.txt
