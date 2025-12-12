# Frequency-Residual Defect Saliency: Unsupervised Inspection for Metal SSD Cases

This project implements an unsupervised computer vision pipeline for detecting defects (scratches, dents) on metal SSD casings. It improves upon the classic **Spectral Residual (SR)** saliency model by fusing it with **Oriented Gradient maps** and introducing a **Robust Statistical Thresholding** mechanism.

The system is designed for industrial quality assurance, capable of distinguishing between "clean" panels with heavy texture and "defective" panels with faint scratches, without requiring large labeled training datasets.

## 📌 Overview

Standard saliency detection methods (like Hou & Zhang, CVPR 2007) often fail on metal surfaces because they mistake reflections, texture noise, and screw holes for defects. This project solves those issues by:
1.  **Fusion:** Combining Frequency-domain anomalies (SR) with Spatial-domain directional features (Oriented Gradients).
2.  **Robust Thresholding:** Replacing rank-based percentile thresholding with `Mean + k*StdDev` to handle clean images correctly (0 False Alarms).
3.  **Smart Masking:** Automatically ignoring non-ROI features like corners (screw holes) and borders.
4.  **Confidence Scoring:** Calculating a Contrast-to-Noise Ratio (CNR) to classify parts as *Clean*, *Defect*, or *Not Sure* (for Human-in-the-loop workflows).

## 🛠️ Prerequisites

The code relies on standard Python computer vision libraries.

**Environment:**
* Python 3.8+
* OS: Linux